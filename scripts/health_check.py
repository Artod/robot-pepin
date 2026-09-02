#!/usr/bin/env python
"""Pepin launch-readiness check: poll every subsystem, report GO / NO GO.

Read-only: pings, register reads, one camera frame each. No motion, no
torque, no writes to any device.

Usage:
    uv run python scripts/health_check.py
"""

import re
import subprocess
import sys
import time

HOST = "root@pepin.local"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", HOST]
SERVO_TCP = "tcp:pepin.local:3333"
LIDAR_TCP = "tcp:pepin.local:3334"
EXPECTED_SERVOS = list(range(1, 11))
TOF_SENSORS = {"front": 0x30, "right": 0x31, "left": 0x32}
CAMERAS = {
    "overview (XIFT AC310)": "/dev/v4l/by-id/usb-XIFT_webcam_AC310_20250819-video-index0",
    "wrist (Sonix CAM1)": (
        "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
    ),
}

RESULTS = []


def report(system: str, ok: bool, detail: str) -> None:
    mark = "\033[32m GO \033[0m" if ok else "\033[31mFAIL\033[0m"
    print(f"  [{mark}] {system:<28} {detail}")
    RESULTS.append((system, ok, detail))


def ssh(cmd: str, timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run([*SSH, cmd], capture_output=True, text=True, timeout=timeout)


def bridge(link: str, target: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["socat", f"pty,link={link},raw,echo=0", target],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)
    return proc


def check_board() -> bool:
    try:
        r = ssh(
            "echo ok && uptime && cat /sys/class/thermal/thermal_zone0/temp && "
            "free -m | awk '/Mem:/{print $7}' && df -h / | awk 'NR==2{print $5}' && "
            "iw dev wlan0 get power_save"
        )
    except subprocess.TimeoutExpired:
        report("board link", False, "ssh timeout")
        return False
    if r.returncode != 0 or "ok" not in r.stdout:
        report("board link", False, "unreachable")
        return False
    lines = r.stdout.strip().splitlines()
    up = lines[1].split("up")[1].split(",")[0].strip() if len(lines) > 1 else "?"
    temp = f"{int(lines[2]) / 1000:.0f}C" if len(lines) > 2 and lines[2].isdigit() else "?"
    mem = f"{lines[3]}MB free" if len(lines) > 3 else "?"
    disk = f"disk {lines[4]} used" if len(lines) > 4 else "?"
    report("board link", True, f"up {up}, cpu {temp}, {mem}, {disk}")
    psave_ok = "off" in r.stdout.splitlines()[-1]
    report("wifi power save", psave_ok, "off (low latency)" if psave_ok else "ON — teleop will lag")
    return True


def check_bridge_ports() -> None:
    r = ssh("ss -tln | grep -cE ':3333|:3334' ; ls /dev/servo-bus /dev/lidar 2>/dev/null | wc -l")
    lines = r.stdout.split()
    ports_ok = lines and lines[0] == "2"
    devs_ok = len(lines) > 1 and lines[1] == "2"
    report(
        "ser2net bridges", ports_ok, "tcp 3333 + 3334 listening" if ports_ok else "port(s) missing"
    )
    report(
        "usb serial devices",
        devs_ok,
        "/dev/servo-bus + /dev/lidar" if devs_ok else "device(s) missing",
    )


def check_servos() -> None:
    proc = bridge("/tmp/health-servo", SERVO_TCP)
    try:
        from lerobot.motors.feetech import FeetechMotorsBus

        bus = FeetechMotorsBus(port="/tmp/health-servo", motors={})
        bus.port_handler.baudrate = 115200  # pty ignores it; real baud fixed by ser2net
        bus._connect(handshake=False)
        found = sorted(bus.broadcast_ping() or [])
        bus.port_handler.closePort()
        missing = [i for i in EXPECTED_SERVOS if i not in found]
        ok = not missing
        detail = (
            f"all {len(EXPECTED_SERVOS)} answer" if ok else f"missing IDs {missing}, found {found}"
        )
        report("servo bus (10 motors)", ok, detail)
    except Exception as e:
        report("servo bus (10 motors)", False, str(e)[:70])
    finally:
        proc.terminate()


def check_lidar() -> None:
    proc = bridge("/tmp/health-lidar", LIDAR_TCP)
    try:
        r = subprocess.run(
            [
                sys.executable,
                "scripts/lidar_scan.py",
                "--port",
                "/tmp/health-lidar",
                "--seconds",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        m = re.search(r"Frames: (\d+) \((\d+)/s\), CRC pass: ([\d.]+)%", r.stdout)
        if m and int(m.group(1)) > 100 and float(m.group(3)) > 95:
            report("lidar (LD19)", True, f"{m.group(2)} frames/s, CRC {m.group(3)}%")
        else:
            report(
                "lidar (LD19)", False, (m.group(0) if m else r.stdout.strip()[-70:] or "no data")
            )
    except Exception as e:
        report("lidar (LD19)", False, str(e)[:70])
    finally:
        proc.terminate()


def check_tof() -> None:
    for name, addr in TOF_SENSORS.items():
        r = ssh(f"i2ctransfer -y 2 w2@0x{addr:02x} 0x01 0x0f r2")
        ok = "0xea 0xcc" in r.stdout
        report(
            f"tof {name} (0x{addr:02x})",
            ok,
            "model VL53L1X verified" if ok else r.stdout.strip() or r.stderr.strip()[:60],
        )


def check_cameras() -> None:
    for name, dev in CAMERAS.items():
        r = ssh(
            f"v4l2-ctl -d {dev} --set-fmt-video=pixelformat=MJPG --stream-mmap "
            "--stream-to=/tmp/health_cam.jpg --stream-count=1 >/dev/null 2>&1 "
            "&& stat -c%s /tmp/health_cam.jpg",
            timeout=30,
        )
        size = int(r.stdout.strip()) if r.stdout.strip().isdigit() else 0
        report(f"camera {name}", size > 5000, f"frame {size} bytes" if size else "no frame")


def main() -> None:
    print("PEPIN LAUNCH READINESS CHECK")
    print("=" * 60)
    t0 = time.time()
    if check_board():
        check_bridge_ports()
        check_servos()
        check_lidar()
        check_tof()
        check_cameras()
    failed = [name for name, ok, _ in RESULTS if not ok]
    print("=" * 60)
    if failed:
        print(f"  NO GO — {len(failed)} system(s) down: {', '.join(failed)}")
        sys.exit(1)
    print(f"  ALL SYSTEMS GO ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
