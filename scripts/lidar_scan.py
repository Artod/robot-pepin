#!/usr/bin/env python
"""First-light test for the LDRobot LD19 lidar.

Reads the sensor's UART stream for a few seconds, parses the 47-byte frames,
prints health stats (frame rate, CRC pass rate, rotation speed, nearest
obstacle) and saves a polar scatter of the accumulated points to
data/lidar_scan.png.

The lidar is mounted upside down, so angles are mirrored by default
(true = 360 - reported). Pass --raw to see the sensor's own frame of
reference. The yaw offset (which direction is the robot's "forward") and the
rear sector mask (cart posts) are handled later in the driver, not here.

Usage:
    uv run python scripts/lidar_scan.py                # auto-detect port
    uv run python scripts/lidar_scan.py --port /dev/tty.usbserial-XXXX
"""

import argparse
import glob
import math
import struct
import sys
import time
from pathlib import Path

import serial

BAUDRATE = 230400
FRAME_LEN = 47
HEADER = 0x54
VERLEN = 0x2C  # frame type + 12 points per frame
POINTS_PER_FRAME = 12
PORT_PATTERNS = ["/dev/tty.usbserial*", "/dev/tty.SLAB*", "/dev/tty.wchusbserial*"]
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "lidar_scan.png"

# CRC8, poly 0x4D, init 0 — per the LD19 development manual.
_CRC_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = ((_c << 1) ^ 0x4D if _c & 0x80 else _c << 1) & 0xFF
    _CRC_TABLE.append(_c)


def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc = _CRC_TABLE[crc ^ byte]
    return crc


def autodetect_port() -> str:
    candidates = [p for pattern in PORT_PATTERNS for p in glob.glob(pattern)]
    if len(candidates) != 1:
        sys.exit(
            f"Expected exactly one UART adapter, found: {candidates or 'none'}. "
            "Pass --port explicitly."
        )
    return candidates[0]


def parse_frame(frame: bytes, flip: bool) -> tuple[float, list[tuple[float, int, int]], bool]:
    """Return (speed_dps, [(angle_deg, dist_mm, intensity)], crc_ok)."""
    speed, start_angle = struct.unpack_from("<HH", frame, 2)
    end_angle, _timestamp = struct.unpack_from("<HH", frame, 42)
    crc_ok = crc8(frame[:-1]) == frame[46]

    span = end_angle - start_angle
    if span < 0:
        span += 36000
    step = span / (POINTS_PER_FRAME - 1)

    points = []
    for i in range(POINTS_PER_FRAME):
        dist, intensity = struct.unpack_from("<HB", frame, 6 + 3 * i)
        angle = ((start_angle + step * i) % 36000) / 100.0
        if flip:
            angle = (360.0 - angle) % 360.0
        points.append((angle, dist, intensity))
    return speed, points, crc_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and plot LD19 lidar output.")
    parser.add_argument("--port", help="UART adapter port (default: auto-detect)")
    parser.add_argument("--seconds", type=float, default=5.0, help="capture duration")
    parser.add_argument(
        "--raw", action="store_true", help="do not mirror angles for the upside-down mount"
    )
    args = parser.parse_args()

    port = args.port or autodetect_port()
    print(f"Reading {port} at {BAUDRATE} baud for {args.seconds} s...")

    frames_ok = 0
    crc_pass = 0
    speeds = []
    points = []
    buffer = bytearray()

    with serial.Serial(port, BAUDRATE, timeout=0.5) as ser:
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            buffer += ser.read(4096)
            while True:
                start = buffer.find(bytes([HEADER, VERLEN]))
                if start < 0:
                    del buffer[:-1]  # keep the last byte: it may be a header start
                    break
                if len(buffer) - start < FRAME_LEN:
                    del buffer[:start]
                    break
                frame = bytes(buffer[start : start + FRAME_LEN])
                del buffer[: start + FRAME_LEN]
                speed, frame_points, crc_ok = parse_frame(frame, flip=not args.raw)
                frames_ok += 1
                crc_pass += crc_ok
                speeds.append(speed)
                points.extend(frame_points)

    if not frames_ok:
        sys.exit("No frames received. Is the lidar spinning and on the right port?")

    nonzero = [(a, d, i) for a, d, i in points if d > 0]
    nearest = min(nonzero, key=lambda p: p[1]) if nonzero else None
    print(f"Frames: {frames_ok} ({frames_ok / args.seconds:.0f}/s), "
          f"CRC pass: {100 * crc_pass / frames_ok:.1f}%")
    print(f"Rotation: {sum(speeds) / len(speeds) / 360:.1f} rev/s")
    print(f"Points: {len(points)} total, {100 * len(nonzero) / len(points):.1f}% nonzero")
    if nonzero:
        dists = sorted(d for _, d, _ in nonzero)
        print(f"Range: min {dists[0]} mm, median {dists[len(dists) // 2]} mm, max {dists[-1]} mm")
        print(f"Nearest obstacle: {nearest[1]} mm at {nearest[0]:.1f} deg")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed (dev dependency) — skipping the plot.")
        return

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    ax.scatter(
        [math.radians(a) for a, _, _ in nonzero],
        [d / 1000.0 for _, d, _ in nonzero],
        s=1,
        c=[i for _, _, i in nonzero],
        cmap="viridis",
    )
    ax.set_theta_zero_location("N")
    ax.set_title(f"LD19 first light — {len(nonzero)} points, distance in meters")
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=120, bbox_inches="tight")
    print(f"Plot saved to {OUTPUT}")


if __name__ == "__main__":
    main()
