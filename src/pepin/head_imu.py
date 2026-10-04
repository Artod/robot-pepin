"""The head IMU as head_server hands it over (TCP 3340), and what the board's base bridge does
with it: the Python twin of ros/pepin_base_cpp's head_imu.hpp / head_line.hpp.

The ESP32 under the stereo camera samples an MPU6050 glued to the camera body; head_server
(``pepin.head_server``, the face agent's) owns the serial port and serves the samples to
subscribers (vio.md section 2). A client sends ``{"cmd":"subscribe","imu":true}`` and gets first,
and again at every change of the chip's setup::

    {"type":"imu_config","cfg":1,"rate_hz":200,"dlpf":3,"gyro_fs_dps":500,"accel_fs_g":4,
     "filter_delay_s":0.0048}

then one line per serial frame (every ~20 ms)::

    {"type":"imu","cfg":1,"samples":[[t_mono_s, esp_us, gx, gy, gz, ax, ay, az], ...]}

``t_mono_s`` is the sample's data-ready edge on the board's CLOCK_MONOTONIC (the server's
lower-envelope clock map; the filter delay NOT taken off), ``esp_us`` the ESP32's micros
unwrapped, gyro rad/s and accel m/s^2 in the chip's axes, biases not removed. The bridge (the ONE
publisher of the robot's real-time state) dates a sample at ``t_mono_s`` less its config's
``filter_delay_s``, publishes ``/head/imu`` and feeds the mast-sway filter (:mod:`pepin.mast`); a
batch of a config it was not told is refused (a default is a refusal). This module holds the
same parsing, the rate counter and the publish cap, the bridge's parameters from
config/head_imu.json and config/camera.json, and two bench tools::

    python -m pepin.head_imu fake [--port 3340] [--rate 200]     # a head_server stand-in
    python -m pepin.head_imu record --host 10.0.0.187 --seconds 10800 --out parked.jsonl.gz

``fake`` serves the same lines with a synthetic chip (gravity, noise, an optional 5.3 Hz ring),
so the bridge and the VIO run end to end before the hardware exists; ``record`` keeps the raw
lines of a parked session for the Allan variance (``python -m pepin.allan``). Standard library
only, as everything the board imports.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import socket
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HEAD_PORT = 3340
SUBSCRIBE_LINE = '{"cmd":"subscribe","imu":true}\n'
IMU_FRAME = "head_imu"
G = 9.80665
# A batch of samples arrives every ~20 ms (the firmware's frame budget); the fake does the same.
BATCH_S = 0.02


@dataclass(frozen=True)
class HeadConfig:
    """One ``imu_config`` line: what the samples of config ``cfg`` were taken at."""

    cfg: int
    rate_hz: float
    filter_delay_s: float

    @property
    def valid(self) -> bool:
        """A positive finite rate and a finite delay >= 0 (head_imu.hpp config_valid)."""
        return (
            math.isfinite(self.rate_hz)
            and self.rate_hz > 0.0
            and math.isfinite(self.filter_delay_s)
            and self.filter_delay_s >= 0.0
        )


@dataclass(frozen=True)
class HeadSample:
    """One head IMU sample in SI and the chip's axes, with both clocks."""

    t_mono_s: float
    esp_us: float
    gyro: tuple[float, float, float]
    accel: tuple[float, float, float]
    cfg: int


@dataclass(frozen=True)
class HeadBatch:
    """One ``imu`` line: its config id, the samples that parsed, and how many rows did not."""

    cfg: int
    samples: tuple[HeadSample, ...]
    refused: int = 0


@dataclass(frozen=True)
class HeadClock:
    """head_server's clock map from its status line (the bridge's minute line prints it)."""

    ready: bool = False
    spread_ms: float = 0.0
    esp_fast_ppm: float = 0.0
    min_rtt_ms: float = -1.0


def _number(value: Any) -> float | None:
    """A JSON number as a float; ``None`` for anything else (booleans included)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def sample_from_row(row: Sequence[Any], cfg: int) -> HeadSample | None:
    """One wire row ``[t_mono_s, esp_us, gx, gy, gz, ax, ay, az]`` as a sample, or ``None``
    when it is not eight finite numbers (head_imu.hpp sample_from_row)."""
    if len(row) != 8:
        return None
    values = [_number(v) for v in row]
    if any(v is None or not math.isfinite(v) for v in values):
        return None
    t, esp, gx, gy, gz, ax, ay, az = (float(v) for v in values if v is not None)
    return HeadSample(t_mono_s=t, esp_us=esp, gyro=(gx, gy, gz), accel=(ax, ay, az), cfg=cfg)


def parse_config(message: Mapping[str, Any]) -> HeadConfig | None:
    """An ``imu_config`` line; ``None`` for any other line or one that cannot date samples
    (head_line.hpp parse_head_config)."""
    if message.get("type") != "imu_config":
        return None
    cfg, rate, delay = (_number(message.get(k)) for k in ("cfg", "rate_hz", "filter_delay_s"))
    if cfg is None or rate is None or delay is None:
        return None
    config = HeadConfig(int(cfg), rate, delay)
    return config if config.valid else None


def parse_line(message: Mapping[str, Any]) -> HeadBatch | None:
    """An ``imu`` line as a batch; ``None`` for any other line (head_line.hpp parse_head_imu).
    A row that is not eight finite numbers is refused and counted."""
    rows = message.get("samples")
    if message.get("type") != "imu" or not isinstance(rows, list):
        return None
    cfg_value = _number(message.get("cfg"))
    cfg = -1 if cfg_value is None else int(cfg_value)
    samples, refused = [], 0
    for row in rows:
        sample = sample_from_row(row, cfg) if isinstance(row, list) else None
        if sample is None:
            refused += 1
        else:
            samples.append(sample)
    return HeadBatch(cfg, tuple(samples), refused)


def parse_status(message: Mapping[str, Any]) -> HeadClock | None:
    """A ``status`` line's clock map; ``None`` for any other line."""
    if message.get("type") != "status":
        return None
    clock = message.get("clock")
    if not isinstance(clock, dict):
        return HeadClock()
    rtt = _number(clock.get("min_rtt_ms"))
    return HeadClock(
        ready=clock.get("ready") is True,
        spread_ms=_number(clock.get("spread_ms")) or 0.0,
        esp_fast_ppm=_number(clock.get("esp_fast_ppm")) or 0.0,
        min_rtt_ms=-1.0 if rtt is None else rtt,
    )


def sample_age(now_mono_s: float, t_mono_s: float, max_age_s: float) -> float | None:
    """A sample's age on the board's monotonic clock, or ``None`` (date it on arrival) when it
    lies in the future or is older than ``max_age_s`` (head_imu.hpp head_sample_age)."""
    age = now_mono_s - t_mono_s
    if not math.isfinite(age) or age < 0.0 or age > max_age_s:
        return None
    return age


@dataclass
class HeadRate:
    """Samples per second and gaps longer than 1.5 periods, by the samples' own clock."""

    samples: int = 0
    gaps: int = 0
    out_of_order: int = 0
    longest_gap_s: float = 0.0
    _first: float = 0.0
    _last: float | None = None

    def add(self, t_mono_s: float, rate_hz: float) -> None:
        """One sample from a chip at ``rate_hz``."""
        self.samples += 1
        if self._last is None:
            self._first = t_mono_s
        else:
            gap = t_mono_s - self._last
            if gap <= 0.0:
                self.out_of_order += 1
                return
            if rate_hz > 0.0 and gap > 1.5 / rate_hz:
                self.gaps += 1
                self.longest_gap_s = max(self.longest_gap_s, gap)
        self._last = t_mono_s

    @property
    def rate_hz(self) -> float:
        """Samples per second over the span seen (0 with fewer than two)."""
        span = 0.0 if self._last is None else self._last - self._first
        return (self.samples - 1) / span if self.samples > 1 and span > 0.0 else 0.0


@dataclass
class HeadDecimator:
    """The publish cap: every n-th sample, n = ceil(rate / cap); 0 publishes every sample."""

    cap_hz: float = 0.0
    _n: int = 1
    _count: int = 0

    def every(self, rate_hz: float) -> int:
        """The keep-one-in-n of a chip at ``rate_hz`` under this cap."""
        if self.cap_hz <= 0.0 or rate_hz <= self.cap_hz:
            return 1
        return math.ceil(rate_hz / self.cap_hz - 1e-9)

    def due(self, rate_hz: float) -> bool:
        """Whether the next sample goes out (a rate change restarts the count)."""
        n = self.every(rate_hz)
        if n != self._n:
            self._n, self._count = n, 0
        keep = self._count % self._n == 0
        self._count += 1
        return keep


# ---- config ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HeadImuConfig:
    """config/head_imu.json: the bridge's link, publish and stamp numbers, the mast filter's
    tunables, and the noise block the VIO's config is generated from."""

    host: str
    port: int
    frame: str
    publish_hz: float
    extra_delay_s: float
    max_age_s: float
    gyro_var: float
    accel_var: float
    noise: dict[str, float]
    mast: dict[str, float]
    rate_hz: float = 200.0
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> HeadImuConfig:
        """config/head_imu.json (or ``path``); ``KeyError``/``ValueError`` on a broken file."""
        from pepin.deployment import config_file

        source = Path(path) if path is not None else config_file("head_imu.json")
        data = json.loads(source.read_text())
        link, publish, noise, mast = data["link"], data["publish"], data["noise"], data["mast"]
        cfg = cls(
            host=str(link["host"]),
            port=int(link["port"]),
            frame=str(publish["frame"]),
            publish_hz=float(publish["hz"]),
            extra_delay_s=float(data["timing"]["extra_delay_s"]),
            max_age_s=float(data["timing"]["max_age_s"]),
            gyro_var=float(publish["gyro_var"]),
            accel_var=float(publish["accel_var"]),
            noise={k: float(v) for k, v in noise.items() if k != "note"},
            mast={k: float(v) for k, v in mast.items() if k != "note"},
            rate_hz=float(data["timing"]["output_rate_hz"]),
            raw=data,
        )
        if cfg.publish_hz < 0.0 or cfg.max_age_s <= 0.0 or cfg.gyro_var <= 0.0:
            raise ValueError(f"{source}: publish hz >= 0, max age > 0, variances > 0")
        return cfg

    def bridge_parameters(
        self, camera_from_imu: Sequence[float] | None, enable: bool
    ) -> dict[str, Any]:
        """The C++ base bridge's head_imu_* and mast_* parameters. ``camera_from_imu`` is the
        row-major R(camera_link <- head_imu) (:func:`camera_from_imu`), or ``None`` while the
        IMU's extrinsics are unknown: the mast filter then stays off (an empty rotation)."""
        return {
            "head_imu_enable": enable,
            "head_imu_host": self.host,
            "head_imu_port": self.port,
            "head_imu_frame": self.frame,
            "head_imu_publish_hz": self.publish_hz,
            "head_imu_filter_delay_s": self.extra_delay_s,
            "head_imu_max_age_s": self.max_age_s,
            "head_imu_gyro_var": self.gyro_var,
            "head_imu_accel_var": self.accel_var,
            "head_imu_camera_rotation": [float(v) for v in camera_from_imu or ()],
            "mast_crossover_hz": self.mast["crossover_hz"],
            "mast_arm_window_s": self.mast["arm_window_s"],
            "mast_ring_hz": self.mast["ring_hz"],
            "mast_hinge_x_m": self.mast["hinge_x_m"],
            "mast_hinge_z_m": self.mast["hinge_z_m"],
            "mast_publish_hz": self.mast["publish_hz"],
            "mast_bias_s": self.mast["bias_s"],
        }


def camera_from_imu(config_dir: str | Path, camera: str | None = None) -> list[float] | None:
    """R(camera_link <- head_imu), row-major, from config/camera.json: the rig's
    camera_link -> camera_optical (the eye block and the optical axes) times the stored
    ``stereo.head_imu.T_cam_imu``'s rotation (Kalibr's, relative to the rectified left eye's
    optical frame); ``None`` while the rig has no head_imu block."""
    from pepin.mounts import load_camera_mounts

    mounts = load_camera_mounts(Path(config_dir), camera)
    if mounts.imu is None:
        return None
    product = mounts.optical.rotation() @ mounts.imu.rotation()
    return [float(product[i, j]) for i in range(3) for j in range(3)]


# ---- the fake head_server ------------------------------------------------------------------------
@dataclass
class FakeChip:
    """A synthetic MPU6050 at rest under gravity, with white noise and an optional ring: what
    :class:`FakeHeadServer` serves, in SI as head_server's lines carry it."""

    rate_hz: float = 200.0
    gyro_noise_rad_s: float = 1.0e-3
    accel_noise_m_s2: float = 0.015
    ring_deg: float = 0.0  # a 5.3 Hz pitch ring of this amplitude, from the start
    ring_hz: float = 5.3
    gravity_axis: int = 2
    seed: int = 7
    filter_delay_s: float = 0.0048

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def row(self, t: float, esp_us: int) -> list[float]:
        """One wire row at board time ``t``: [t, esp_us, gx, gy, gz, ax, ay, az]."""
        gyro = [self._random.gauss(0.0, self.gyro_noise_rad_s) for _ in range(3)]
        if self.ring_deg:
            omega = 2 * math.pi * self.ring_hz
            gyro[1] += math.radians(self.ring_deg) * omega * math.cos(omega * t)
        accel = [self._random.gauss(0.0, self.accel_noise_m_s2) for _ in range(3)]
        accel[self.gravity_axis] += G
        return [round(t, 6), esp_us, *(round(g, 7) for g in gyro), *(round(a, 5) for a in accel)]

    def config_line(self, cfg: int = 1) -> dict[str, Any]:
        """The ``imu_config`` line head_server sends before the first sample."""
        return {
            "type": "imu_config",
            "cfg": cfg,
            "rate_hz": self.rate_hz,
            "dlpf": 3,
            "gyro_fs_dps": 500,
            "accel_fs_g": 4,
            "filter_delay_s": self.filter_delay_s,
        }


class FakeHeadServer:
    """head_server's IMU half on a TCP port: subscribers get ``imu`` lines every 20 ms with the
    board's monotonic clock in ``t`` and a ``status`` line a second (pepin.head_server's
    schema). For the bridge and the VIO before the hardware, and for the tests."""

    def __init__(self, port: int = HEAD_PORT, chip: FakeChip | None = None, host: str = "") -> None:
        self.chip = chip or FakeChip()
        self._socket = socket.create_server((host, port), reuse_port=False)
        self.port = int(self._socket.getsockname()[1])
        self._stop = threading.Event()
        self._clients: list[socket.socket] = []
        self._subscribed: set[socket.socket] = set()
        self._lock = threading.Lock()
        self.lines_sent = 0
        self._threads = [
            threading.Thread(target=self._accept, daemon=True),
            threading.Thread(target=self._stream, daemon=True),
        ]

    def start(self) -> FakeHeadServer:
        """Start accepting and streaming; returns self."""
        for thread in self._threads:
            thread.start()
        return self

    def stop(self) -> None:
        """Stop and close every socket."""
        self._stop.set()
        self._socket.close()
        with self._lock:
            for client in self._clients:
                client.close()
        for thread in self._threads:
            thread.join(1.0)

    def _accept(self) -> None:
        self._socket.settimeout(0.2)
        while not self._stop.is_set():
            try:
                client, _ = self._socket.accept()
            except (TimeoutError, OSError):
                continue
            with self._lock:
                self._clients.append(client)
            threading.Thread(target=self._read, args=(client,), daemon=True).start()

    def _read(self, client: socket.socket) -> None:
        buffer = b""
        client.settimeout(0.2)
        while not self._stop.is_set():
            try:
                chunk = client.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if message.get("cmd") == "subscribe":
                    self._send(client, self.chip.config_line())  # before the first sample
                    with self._lock:
                        if message.get("imu", True):
                            self._subscribed.add(client)
                        else:
                            self._subscribed.discard(client)
                    self._send(client, {"type": "ack", "cmd": "subscribe", "imu": True})
        with self._lock:
            self._subscribed.discard(client)

    def _send(self, client: socket.socket, message: Mapping[str, Any] | str) -> None:
        text = message if isinstance(message, str) else json.dumps(message)
        try:
            client.sendall(text.encode() + (b"" if text.endswith("\n") else b"\n"))
        except OSError:
            with self._lock:
                self._subscribed.discard(client)

    def _stream(self) -> None:
        chip = self.chip
        period = 1.0 / chip.rate_hz
        start_mono = time.monotonic()
        esp0 = 1_000_000
        n = 0
        next_status = start_mono + 1.0
        while not self._stop.wait(BATCH_S):
            now = time.monotonic()
            rows = []
            while start_mono + n * period <= now:
                rows.append(chip.row(start_mono + n * period, esp0 + round(n * period * 1e6)))
                n += 1
            if rows:
                line = json.dumps({"type": "imu", "cfg": 1, "samples": rows}, separators=(",", ":"))
                with self._lock:
                    targets = list(self._subscribed)
                for client in targets:
                    self._send(client, line)
                self.lines_sent += 1
            if now >= next_status:
                next_status += 1.0
                status = {
                    "type": "status",
                    "t": round(now, 6),
                    "link": "up (fake)",
                    "clock": {"ready": True, "offset_s": 0.0, "esp_fast_ppm": 0.0,
                              "min_rtt_ms": 0.2, "spread_ms": 0.05, "points": 40},
                }  # fmt: skip
                with self._lock:
                    clients = list(self._clients)
                for client in clients:
                    self._send(client, status)


# ---- the recorder ------------------------------------------------------------------------------
def record_lines(host: str, port: int, seconds: float) -> Iterator[str]:
    """head_server's raw ``imu`` and ``status`` lines for ``seconds``, subscribed."""
    with socket.create_connection((host, port), timeout=5.0) as conn:
        conn.sendall(SUBSCRIBE_LINE.encode())
        conn.settimeout(1.0)
        end = time.monotonic() + seconds
        buffer = b""
        while time.monotonic() < end:
            try:
                chunk = conn.recv(65536)
            except TimeoutError:
                continue
            if not chunk:
                return
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                yield line.decode(errors="replace")


def read_recording(path: str | Path) -> Iterator[HeadSample]:
    """The samples of a recording written by ``record`` (gzip or plain JSON lines)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line in stream:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            batch = parse_line(message) if isinstance(message, dict) else None
            if batch is not None:
                yield from batch.samples


def main(argv: list[str] | None = None) -> int:
    """``fake`` or ``record``."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fake = sub.add_parser("fake", help="serve synthetic head IMU lines (head_server's schema)")
    fake.add_argument("--port", type=int, default=HEAD_PORT)
    fake.add_argument("--rate", type=float, default=200.0)
    fake.add_argument("--ring-deg", type=float, default=0.0)
    rec = sub.add_parser("record", help="keep head_server's raw lines (for the Allan variance)")
    rec.add_argument("--host", default="127.0.0.1")
    rec.add_argument("--port", type=int, default=HEAD_PORT)
    rec.add_argument("--seconds", type=float, required=True)
    rec.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fake":
        server = FakeHeadServer(args.port, FakeChip(rate_hz=args.rate, ring_deg=args.ring_deg))
        server.start()
        print(f"fake head_server on :{server.port}, {args.rate:g} Hz", flush=True)
        try:
            while True:
                time.sleep(60.0)
                print(f"fake head_server: {server.lines_sent} lines", flush=True)
        except KeyboardInterrupt:
            server.stop()
        return 0
    opener = gzip.open if str(args.out).endswith(".gz") else open
    count = 0
    with opener(args.out, "wt") as out:
        for line in record_lines(args.host, args.port, args.seconds):
            out.write(line + "\n")
            count += 1
    print(f"{count} lines from {args.host}:{args.port} in {args.seconds:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
