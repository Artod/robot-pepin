"""Subsystem health probes shared by the CLI check, the menu-bar app and the dashboard.

Two tiers: ``quick`` (a few seconds: board vitals, bridges, servo bus, lidar
rate, ToF stream, camera presence) for periodic polling, and ``full`` which
adds real camera frames and ToF model-ID reads for a launch-readiness check.
Every probe returns a :class:`Probe`; nothing here moves the robot.
"""

from __future__ import annotations

import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pepin.transport import LIDAR_PORT, SERVO_BUS_PORT

EXPECTED_SERVOS = list(range(1, 11))
TOF_SENSORS = {"front": 0x30, "right": 0x31, "left": 0x32}
CAMERAS = {
    "overview": "/dev/v4l/by-id/usb-XIFT_webcam_AC310_20250819-video-index0",
    "wrist": "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0",
}


@dataclass(frozen=True)
class Probe:
    """One subsystem's verdict: name, ok flag, and a short human detail."""

    system: str
    ok: bool
    detail: str


@dataclass
class BoardVitals:
    """Numbers worth graphing over time, parsed from the board."""

    uptime: str = "?"
    cpu_temp_c: float | None = None
    mem_free_mb: int | None = None
    disk_used_pct: str = "?"
    wifi_power_save_off: bool | None = None


@dataclass
class HealthReport:
    probes: list[Probe] = field(default_factory=list)
    vitals: BoardVitals = field(default_factory=BoardVitals)
    started: float = field(default_factory=time.time)
    duration_s: float = 0.0

    @property
    def all_go(self) -> bool:
        return bool(self.probes) and all(p.ok for p in self.probes)

    @property
    def failed(self) -> list[str]:
        return [p.system for p in self.probes if not p.ok]


SSH_TIMED_OUT = 124  # returncode _ssh reports when the command hung, like coreutils timeout(1)
# One handshake per ten minutes instead of one per probe: sshd on the board takes ~3 s to accept a
# session, and a health pass makes eight of them (measured 2026-09-06: 24 s -> ~4 s).
SSH_MULTIPLEX = (
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=/tmp/pepin-ssh-%C",
    "-o", "ControlPersist=10m",
)  # fmt: skip


def _ssh(host: str, cmd: str, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` on the board as root; a hung ssh comes back as returncode 124, never raises."""
    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=6",
        *SSH_MULTIPLEX,
        f"root@{host}",
        cmd,
    ]
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, SSH_TIMED_OUT, "", "ssh timeout")


def port_refused(host: str, port: int, timeout_s: float = 2.0) -> bool:
    """True only when nothing listens on ``host:port``; a timeout counts as "someone may"."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return False
    except ConnectionRefusedError:
        return True
    except OSError:
        return False


def busy_bridge_ports(host: str) -> set[int]:
    """Bridge ports that already have a client — a drive in progress.

    ser2net runs with ``kickolduser``: connecting to a busy port would throw the
    driver off the servo bus, so the probes report those ports as in use instead.
    """
    r = _ssh(host, "ss -Htn state established '( sport = :3333 or sport = :3334 )'")
    return _parse_local_ports(r.stdout)


def _parse_local_ports(ss_output: str) -> set[int]:
    """Local ports with a client that is *not* the board itself, from ``ss -tn state established``.

    Lines are ``Recv-Q Send-Q Local:port Peer:port``. The base server holds the
    servo bridge over loopback permanently; that is ownership, not a driver
    on the laptop, so loopback peers do not count.
    """
    ports: set[int] = set()
    for line in ss_output.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        peer = parts[3]
        if peer.startswith(("127.", "[::1]", "[::ffff:127.")):
            continue
        port = parts[2].rsplit(":", 1)[-1]
        if port.isdigit():
            ports.add(int(port))
    return ports


def probe_board(host: str, report: HealthReport) -> Probe:
    """ssh reachability plus vitals (uptime, CPU temperature, memory, disk, wifi power save)."""
    r = _ssh(
        host,
        "echo ok && uptime && cat /sys/class/thermal/thermal_zone0/temp && "
        "free -m | awk '/Mem:/{print $7}' && df -h / | awk 'NR==2{print $5}' && "
        "iw dev wlan0 get power_save",
    )
    if r.returncode == SSH_TIMED_OUT:
        return Probe("board", False, "ssh timeout")
    if r.returncode != 0 or "ok" not in r.stdout:
        return Probe("board", False, "unreachable")
    lines = r.stdout.strip().splitlines()
    v = report.vitals
    if len(lines) > 1:
        v.uptime = lines[1].split("up")[1].split(",")[0].strip()
    if len(lines) > 2 and lines[2].isdigit():
        v.cpu_temp_c = int(lines[2]) / 1000
    if len(lines) > 3 and lines[3].isdigit():
        v.mem_free_mb = int(lines[3])
    if len(lines) > 4:
        v.disk_used_pct = lines[4]
    # Line 5 is `iw ... get power_save`; if iw printed nothing the flag stays unknown
    # rather than being read off the disk-usage line.
    v.wifi_power_save_off = ("off" in lines[5]) if len(lines) > 5 else None
    temp = f"{v.cpu_temp_c:.0f}C" if v.cpu_temp_c is not None else "?"
    return Probe("board", True, f"up {v.uptime}, cpu {temp}, {v.mem_free_mb} MB free")


def ros_container_status(host: str) -> str | None:
    """``docker ps`` status of the ROS 2 container (``Up 12 minutes``); None when not running."""
    r = _ssh(host, "docker ps --filter name=pepin-ros --format '{{.Status}}' 2>/dev/null")
    status = r.stdout.strip().splitlines()
    return status[0] if status and status[0].startswith("Up") else None


def probe_bridges(host: str) -> Probe:
    """The servo-bus bridge and both udev devices; the lidar bridge only when ROS does not own it.

    On the ros2-nav2 branch the lidar tty belongs to the ROS container (a second reader
    would split the byte stream), so ser2net serves :3333 alone and that is the healthy state.
    """
    r = _ssh(
        host,
        "ss -tln | grep -c ':3333'; ss -tln | grep -c ':3334'; "
        "ls /dev/servo-bus /dev/lidar 2>/dev/null | wc -l",
    )
    parts = r.stdout.split()
    if len(parts) != 3 or parts[0] != "1" or parts[2] != "2":
        return Probe("bridges", False, "servo-bus port or a device missing")
    if parts[1] == "1":
        return Probe("bridges", True, "ser2net 3333+3334, /dev/servo-bus + /dev/lidar")
    ros = ros_container_status(host)
    if ros is None:
        return Probe("bridges", False, "lidar bridge off and no ROS container running")
    return Probe("bridges", True, f"ser2net 3333; lidar owned by ROS ({ros.lower()})")


def probe_servos(host: str) -> Probe:
    """Which servos answer: via the base server when it runs, else pinged directly (read-only)."""
    from pepin.base_link import BASE_PORT, BaseClient
    from pepin.feetech import FeetechTcpClient

    link = BaseClient(host).start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not link.connected:
            time.sleep(0.05)
        if link.connected:
            answers = link.ping(timeout_s=4.0)
            if answers is None:
                return Probe("servo bus", False, "base server did not answer the ping")
            if not answers:
                return Probe("servo bus", True, "driving — servos not pinged")
            missing_names = sorted(k for k, ok in answers.items() if not ok)
            detail = (
                f"all {len(answers)} answer (via base server)"
                if not missing_names
                else f"missing {missing_names}"
            )
            return Probe("servo bus", not missing_names, detail)
    finally:
        link.close()
    if not port_refused(host, BASE_PORT):
        # Listening but quiet (starting up, or its accept loop is stuck): touching :3333
        # now would kick the base server off the bus mid-drive. Report, do not probe.
        return Probe("servo bus", False, "base server listening but silent — bus not probed")
    # No base server (bench mode): talk to ser2net directly.
    motors = {str(i): i for i in EXPECTED_SERVOS}
    try:
        # reconnect=False: if a driver takes the port from us mid-probe we must not take
        # it back (ser2net would kick the driver again); we just report and step aside.
        with FeetechTcpClient(host, SERVO_BUS_PORT, motors, reconnect=False) as bus:
            missing = [i for i in EXPECTED_SERVOS if bus.ping(str(i)) is None]
    except TimeoutError as exc:
        if "link lost" in str(exc):
            return Probe("servo bus", True, "taken over by a driver mid-probe — not probed")
        return Probe("servo bus", False, str(exc)[:60])
    except OSError as exc:
        return Probe("servo bus", False, str(exc)[:60])
    return Probe(
        "servo bus",
        not missing,
        f"all {len(EXPECTED_SERVOS)} answer" if not missing else f"missing {missing}",
    )


def probe_lidar(host: str, seconds: float = 1.0) -> Probe:
    """Frame rate and CRC pass rate of the lidar stream over the bridge."""
    from pepin.lidar import FrameParser, TcpSource

    try:
        source = TcpSource(host, LIDAR_PORT, timeout_s=0.5)
    except OSError as exc:
        ros = ros_container_status(host)
        if ros is not None:
            # The ROS driver holds the tty; check that it really does (one open fd on /dev/lidar).
            fds = _ssh(
                host, "docker exec pepin-ros sh -c 'ls -l /proc/*/fd 2>/dev/null | grep -c lidar'"
            )
            held = fds.stdout.strip().isdigit() and int(fds.stdout.strip()) >= 1
            detail = (
                f"driver in ROS container ({ros.lower()})"
                if held
                else "ROS container up, driver not holding the tty"
            )
            return Probe("lidar", held, detail)
        return Probe("lidar", False, str(exc)[:60])
    parser = FrameParser()
    deadline = time.monotonic() + seconds
    speeds: list[float] = []
    try:
        while time.monotonic() < deadline:
            for frame in parser.feed(source.read(4096)):
                speeds.append(frame.speed_dps)
    except ConnectionError:
        # The bridge closed on us: a driver took the port (kickolduser). Leave it alone.
        return Probe("lidar", True, "taken over by a driver mid-probe — not probed")
    finally:
        source.close()
    if parser.frames < 100 * seconds:
        return Probe("lidar", False, f"{parser.frames} frames in {seconds:.0f}s — motor stopped?")
    rps = sum(speeds) / len(speeds) / 360 if speeds else 0.0
    return Probe(
        "lidar",
        parser.crc_failures == 0,
        f"{parser.frames / seconds:.0f} frames/s, {rps:.1f} rev/s, crc fails {parser.crc_failures}",
    )


def probe_tof(host: str, wait_s: float = 1.5) -> Probe:
    """Range stream alive and which sensors report a distance."""
    from pepin.tof import TofClient

    client = TofClient(host).start()
    try:
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and not client.connected:
            time.sleep(0.05)
        time.sleep(0.3)
        r = client.ranges()
    finally:
        client.close()
    if r.age_s == float("inf"):
        return Probe("tof", False, "no range stream (service down?)")

    def fmt(value: float | None) -> str:
        return "none" if value is None else f"{value:.2f}m"

    # The server keeps streaming when every sensor failed to initialise (all null),
    # so a live stream alone is not health.
    any_range = any(v is not None for v in (r.front, r.left, r.right))
    return Probe(
        "tof", any_range, f"front {fmt(r.front)}, left {fmt(r.left)}, right {fmt(r.right)}"
    )


def probe_imu(host: str) -> Probe:
    """The MPU6050 answers on the board's I2C bus 2 (0x68, next to the three ToF sensors)."""
    r = _ssh(host, "i2cdetect -y 2", timeout=8)
    if r.returncode != 0:
        return Probe(
            "imu",
            False,
            "i2c bus 2 not readable" if r.returncode != SSH_TIMED_OUT else "ssh timeout",
        )
    row = next((line for line in r.stdout.splitlines() if line.startswith("60:")), "")
    present = " 68 " in f"{row} " or row.endswith(" 68")
    return Probe(
        "imu", present, "MPU6050 at 0x68 on i2c-2" if present else "nothing at 0x68 on i2c-2"
    )


def probe_cameras(host: str, grab_frames: bool) -> list[Probe]:
    """Camera device presence, optionally a real MJPEG frame per camera."""
    probes = []
    for name, dev in CAMERAS.items():
        if grab_frames:
            r = _ssh(
                host,
                f"v4l2-ctl -d {dev} --set-fmt-video=pixelformat=MJPG --stream-mmap "
                f"--stream-to=/tmp/health_{name}.jpg --stream-count=1 >/dev/null 2>&1 "
                f"&& stat -c%s /tmp/health_{name}.jpg",
                timeout=30,
            )
            size = int(r.stdout.strip()) if r.stdout.strip().isdigit() else 0
            probes.append(
                Probe(f"camera {name}", size > 5000, f"frame {size} bytes" if size else "no frame")
            )
        else:
            r = _ssh(host, f"test -e {dev} && echo yes")
            probes.append(
                Probe(
                    f"camera {name}",
                    "yes" in r.stdout,
                    "present" if "yes" in r.stdout else "missing",
                )
            )
    return probes


def probe_tof_ids(host: str) -> list[Probe]:
    """Model-ID register of every ToF sensor over I2C (proves the chip, not just the bus)."""
    probes = []
    for name, addr in TOF_SENSORS.items():
        r = _ssh(host, f"i2ctransfer -y 2 w2@0x{addr:02x} 0x01 0x0f r2")
        ok = "0xea 0xcc" in r.stdout
        probes.append(
            Probe(
                f"tof {name} chip",
                ok,
                "VL53L1X" if ok else (r.stdout.strip() or r.stderr.strip())[:40],
            )
        )
    return probes


def run_health(
    host: str, full: bool = False, on_probe: Callable[[Probe], None] | None = None
) -> HealthReport:
    """Run the quick tier (or the full one) and return the report; ``on_probe`` streams results.

    Safe to run during a drive: bridge ports that already have a client are
    reported as in use, never probed (probing would kick the driver off them).
    """
    report = HealthReport()
    t0 = time.monotonic()

    def add(probe: Probe) -> None:
        report.probes.append(probe)
        if on_probe is not None:
            on_probe(probe)

    board = probe_board(host, report)
    add(board)
    if board.ok:
        add(probe_bridges(host))
        # Re-check right before each bridge probe: a drive can start between two probes,
        # and the probes themselves also step aside if they get kicked mid-way.
        in_use = "in use by a driver — not probed"
        busy = busy_bridge_ports(host)
        add(Probe("servo bus", True, in_use) if SERVO_BUS_PORT in busy else probe_servos(host))
        busy = busy_bridge_ports(host)
        add(Probe("lidar", True, in_use) if LIDAR_PORT in busy else probe_lidar(host))
        add(probe_tof(host))
        add(probe_imu(host))
        if full:
            for p in probe_tof_ids(host):
                add(p)
        for p in probe_cameras(host, grab_frames=full):
            add(p)
    report.duration_s = time.monotonic() - t0
    return report


# -- polling cadence for the menu-bar app -------------------------------------------------------

FAST_POLL_S, SLOW_POLL_S, FAST_FOR_S = 30.0, 300.0, 300.0


def next_poll_s(all_go: bool | None, fast_for_left_s: float) -> float:
    """Seconds until the next health poll.

    30 s while things are changing: right after start or a manual refresh, and after ANY result
    that is not ALL GO — a red result must be re-checked soon, not shown for five minutes. The
    slow 5 min cadence is only for a robot that has been all go for a while. (A ten-second Wi-Fi
    flap once painted the tray red until the next slow poll, 2026-09-09.)
    """
    if all_go is not True or fast_for_left_s > 0:
        return FAST_POLL_S
    return SLOW_POLL_S


def is_stale(age_s: float, cadence_s: float) -> bool:
    """A result older than twice its cadence says nothing about now and must be marked so."""
    return age_s > 2.0 * cadence_s
