"""The preflight time line's pure parts: the CDR header stamp, chrony's offset, the verdict."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]


def _tool() -> ModuleType:
    """Import ros/tools/stream_latency.py by path (ros/tools is not a package)."""
    path = REPO / "ros/tools/stream_latency.py"
    spec = importlib.util.spec_from_file_location("stream_latency", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses looks the module up while the class is built
    spec.loader.exec_module(module)
    return module


sl = _tool()
LAPTOP = "10.0.0.167"
HEALTHY = {
    "odom": [20.0, 32.0, 40.0],
    "scan": [19.0],
    "odom_laser": [66.0],
    "head_imu": [37.0],
    "neck": [21.0, 32.0, 900.0],
    "camera": [235.0],
    "vo": [260.0],
}


def test_header_stamp_is_read_after_the_encapsulation_in_either_byte_order() -> None:
    little = b"\x00\x01\x00\x00" + struct.pack("<iI", 1791229015, 485_000_000) + b"frame"
    big = b"\x00\x00\x00\x00" + struct.pack(">iI", 1791229015, 485_000_000)
    assert abs(sl.cdr_header_stamp(little) - 1791229015.485) < 1e-6
    assert abs(sl.cdr_header_stamp(big) - 1791229015.485) < 1e-6


def test_chrony_system_time_is_a_correction_so_a_slow_board_is_negative() -> None:
    slow = "0A0000A7,10.0.0.167,11,1791250559.6,0.000362626,-0.0000757,0,6.7,0,7.7,0,0,16,Normal"
    fast = "0A0000A7,10.0.0.167,11,1791250559.6,-0.000441793,0,0,0,0,0,0,0,16.1,Normal"
    ms, ref = sl.chrony_offset(slow, LAPTOP)
    assert ref == "laptop" and abs(ms + 0.3626) < 1e-3
    assert abs(sl.chrony_offset(fast, LAPTOP)[0] - 0.4418) < 1e-3
    assert sl.chrony_offset(fast, "10.0.0.5")[1] == "10.0.0.167"
    assert sl.chrony_offset("", LAPTOP) is None
    assert sl.chrony_offset("a,b,c,d,not-a-number", LAPTOP) is None


def test_healthy_streams_pass_with_their_medians_in_order() -> None:
    passed, line = sl.verdict(HEALTHY, (0.3, "laptop"))
    assert passed
    assert line == (
        "board-laptop +0.3 ms (chrony, ref laptop); p50 ms: odom 32, scan 19, odom_laser 66, "
        "head_imu 37, neck 32, camera 235, vo 260"
    )


def test_a_late_neck_and_a_silent_camera_fail_first_but_a_silent_vo_does_not() -> None:
    late = {**HEALTHY, "neck": [1600.0, 1700.0, 1800.0], "camera": [], "vo": []}
    passed, line = sl.verdict(late, (0.3, "laptop"))
    assert not passed
    assert line.startswith("neck 1700 ms > 150; camera silent; board-laptop +0.3 ms")
    assert line.endswith("neck 1700, camera silent, vo silent")


def test_the_clock_fails_over_its_bound_and_when_chrony_did_not_answer() -> None:
    passed, line = sl.verdict(HEALTHY, (-25.0, "laptop"))
    assert not passed and line.startswith("clock -25.0 ms > 10; board-laptop -25.0 ms")
    passed, line = sl.verdict(HEALTHY, None)
    assert not passed and line.startswith("clock unknown; board-laptop unknown")


def test_a_board_on_the_pool_says_its_offset_is_not_against_the_laptop() -> None:
    passed, line = sl.verdict(HEALTHY, (1.2, "162.159.200.1"))
    assert passed
    assert line.startswith("board-162.159.200.1 +1.2 ms (chrony: the laptop is not the reference)")
