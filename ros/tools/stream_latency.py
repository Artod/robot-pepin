#!/usr/bin/env python3
"""How late do the key streams reach this laptop, and do the two clocks agree? One line.

Listens for ``seconds`` (default 2) to the streams the drive leans on and takes each one's
stamp -> receipt latency (this container's clock minus the message's header stamp; for the neck
the stamp of the base_link -> camera_link transform on /tf). The board's clock against the
laptop's comes from the board's chrony (``chronyc -c tracking`` over ssh, passed in by
ros/preflight.sh): the board is chrony's client and the laptop's pepin-chrony container its
reference, so ``board - laptop = -(System time)`` while the reference is the laptop.

    python3 /tools/stream_latency.py [seconds=2] [--chrony CSV] [--laptop-ip IP]

Prints "OK board-laptop +0.4 ms (chrony, ref laptop); p50 ms: odom 32, scan 19, ..." or FAIL
first with the reasons (a required stream silent, a p50 over its bound, the offset over
``MAX_OFFSET_MS``); exit 1 on FAIL. Messages are taken raw and only their header stamp is read
(the CDR header sits right after the 4-byte encapsulation), so an image costs no
deserialisation.
"""

from __future__ import annotations

import statistics
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any

MAX_OFFSET_MS = 10.0  # |board - laptop| beyond this skews every latency it is read against
FIRST_WAIT_S = 2.0  # for every stream's first message (discovery over the routers)


@dataclass(frozen=True)
class Stream:
    """One measured stream: its short name, topic, message type, p50 bound and whether its
    silence is a failure (/vo is silent at rest until OpenVINS's first motion)."""

    name: str
    topic: str
    msg_type: str
    bound_ms: float
    required: bool = True


# Normal p50 on 2026-10-05's drives (bags 306/307/325): odom 32, scan 19, odom_laser 66 (rf2o's
# own ~60 ms on the board), head IMU 37, neck 32 ms; /vo ~270 ms (OpenVINS's pipeline); the
# camera 230-238 ms at rest (the grab stamp, 90 ms before the MJPEG send, then decode and the
# laptop's re-publish). After a drive the base bridge's streams reached 0.8-3.8 s.
STREAMS = (
    Stream("odom", "/odom", "nav_msgs/msg/Odometry", 150.0),
    # The scan is stamped at the MIDDLE of its ~100 ms sweep since 2026-10-07 (config/lidar.json
    # scan_stamp), so its age at the laptop reads ~100 ms more than the transport alone, rf2o's
    # output (stamped with the scan) likewise: the limits carry that 100 ms.
    Stream("scan", "/ldlidar_node/scan", "sensor_msgs/msg/LaserScan", 250.0),
    Stream("odom_laser", "/odom_laser", "nav_msgs/msg/Odometry", 300.0),
    Stream("head_imu", "/head/imu", "sensor_msgs/msg/Imu", 150.0),
    Stream("neck", "/tf", "tf2_msgs/msg/TFMessage", 150.0),
    Stream("camera", "/camera/camera_info", "sensor_msgs/msg/CameraInfo", 400.0),
    Stream("vo", "/vo", "nav_msgs/msg/Odometry", 500.0, required=False),
)
NECK_EDGE = ("base_link", "camera_link")  # the base bridge's neck transform (parent, child)


def cdr_header_stamp(data: bytes) -> float:
    """The header stamp (s) of a serialised message whose first field is a std_msgs/Header."""
    little = data[1] & 1 == 1  # encapsulation id 0x0001 (CDR_LE) or 0x0000 (CDR_BE)
    sec, nanosec = struct.unpack_from("<iI" if little else ">iI", data, 4)
    return float(sec) + float(nanosec) * 1e-9


def chrony_offset(csv: str, laptop_ip: str) -> tuple[float, str] | None:
    """(board - reference in ms, "laptop" or the reference's name) from ``chronyc -c tracking``.

    Field 5 is System time, chrony's remaining correction: positive while the board is slow.
    """
    fields = csv.strip().split(",")
    if len(fields) < 5:
        return None
    try:
        correction_s = float(fields[4])
    except ValueError:
        return None
    ref = "laptop" if laptop_ip and fields[1] == laptop_ip else fields[1]
    return -correction_s * 1e3, ref


def verdict(
    latencies: dict[str, list[float]], offset: tuple[float, str] | None
) -> tuple[bool, str]:
    """(passed, the line after OK/FAIL) from each stream's latencies (ms) and the clock offset."""
    fails: list[str] = []
    parts: list[str] = []
    for stream in STREAMS:
        values = latencies.get(stream.name, [])
        if not values:
            parts.append(f"{stream.name} silent")
            if stream.required:
                fails.append(f"{stream.name} silent")
            continue
        p50 = statistics.median(values)
        parts.append(f"{stream.name} {p50:.0f}")
        if p50 > stream.bound_ms:
            fails.append(f"{stream.name} {p50:.0f} ms > {stream.bound_ms:.0f}")
    if offset is None:
        clock = "board-laptop unknown (no chrony answer)"
        fails.append("clock unknown")
    else:
        ms, ref = offset
        clock = f"board-laptop {ms:+.1f} ms (chrony, ref {ref})"
        if abs(ms) > MAX_OFFSET_MS:
            fails.append(f"clock {ms:+.1f} ms > {MAX_OFFSET_MS:.0f}")
        elif ref != "laptop":
            clock = f"board-{ref} {ms:+.1f} ms (chrony: the laptop is not the reference)"
    line = f"{clock}; p50 ms: " + ", ".join(parts)
    return (not fails, ("; ".join(fails) + "; " + line) if fails else line)


def _arg(name: str, default: str = "") -> str:
    """The value after ``--name`` on the command line, else ``default``."""
    argv = sys.argv
    return argv[argv.index(name) + 1] if name in argv[:-1] else default


def main() -> None:
    import importlib

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    flagged = {_arg("--chrony"), _arg("--laptop-ip")}
    positional = [a for a in positional if a not in flagged]
    seconds = float(positional[0]) if positional else 2.0
    offset = chrony_offset(_arg("--chrony"), _arg("--laptop-ip"))

    rclpy.init()
    node = rclpy.create_node("pepin_stream_latency")
    latencies: dict[str, list[float]] = {s.name: [] for s in STREAMS}
    counting = [False]

    def on_raw(name: str, data: bytes) -> None:
        if counting[0]:
            latencies[name].append((time.time() - cdr_header_stamp(data)) * 1e3)
        else:
            latencies[name][:] = [0.0]  # heard once: discovery done for this stream

    def on_tf(msg: Any) -> None:
        now = time.time()
        for tr in msg.transforms:
            if (tr.header.frame_id, tr.child_frame_id) == NECK_EDGE:
                if counting[0]:
                    stamp = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
                    latencies["neck"].append((now - stamp) * 1e3)
                else:
                    latencies["neck"][:] = [0.0]

    try:
        for s in STREAMS:
            package, _, cls = s.msg_type.split("/")
            msg_class = getattr(importlib.import_module(f"{package}.msg"), cls)
            if s.name == "neck":
                tf_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
                node.create_subscription(msg_class, s.topic, on_tf, tf_qos)
            else:
                node.create_subscription(
                    msg_class,
                    s.topic,
                    lambda d, n=s.name: on_raw(n, d),
                    qos_profile_sensor_data,
                    raw=True,
                )
        deadline = time.monotonic() + FIRST_WAIT_S
        while time.monotonic() < deadline and not all(
            latencies[s.name] for s in STREAMS if s.required
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
        for values in latencies.values():  # the window starts now
            values.clear()
        counting[0] = True
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        passed, line = verdict(latencies, offset)
        print(("OK " if passed else "FAIL ") + line)
        if not passed:
            sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
