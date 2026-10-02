"""Wire format of the ToF bridge: reassembling board lines and parsing readings.

The ROS package is not installed on the laptop (no rclpy here), so this test
reaches into ros/pepin_bringup for the one module that has no ROS in it.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ros" / "pepin_bringup"))

from pepin_bringup.protocol import LineReader, parse_tof  # noqa: E402

STATE = {
    "type": "state",
    "t": 1234.5,
    "x": 0.4,
    "y": -0.2,
    "theta": 1.5,
    "dl": 0.01,
    "dr": 0.012,
    "v": 0.22,
    "w": -0.1,
    "moving": True,
    "armed": True,
    "deadman": False,
    "bus_ok": True,
    "bus_p95_ms": 3.5,
}


def test_reader_reassembles_a_message_split_across_chunks() -> None:
    reader = LineReader()
    line = json.dumps(STATE).encode() + b"\n"
    assert reader.feed(line[:20]) == []
    assert reader.feed(line[20:40]) == []
    assert reader.feed(line[40:]) == [STATE]


def test_reader_returns_every_message_a_chunk_finished() -> None:
    reader = LineReader()
    chunk = b'{"a": 1}\n{"a": 2}\n{"a": 3'
    assert reader.feed(chunk) == [{"a": 1}, {"a": 2}]
    assert reader.feed(b"}\n") == [{"a": 3}]


def test_reader_skips_garbage_and_keeps_the_stream() -> None:
    reader = LineReader()
    assert reader.feed(b'not json\n[1, 2]\n"text"\n\n{"a": 1}\n') == [{"a": 1}]


def test_reader_drops_an_endless_line_instead_of_growing() -> None:
    reader = LineReader(max_line_bytes=64)
    assert reader.feed(b"x" * 200) == []
    assert reader.feed(b'junk\n{"a": 1}\n') == [{"a": 1}]


def test_tof_millimetres_become_metres() -> None:
    assert parse_tof({"t": 1.0, "front": 300, "left": 1250, "right": 42}) == {
        "front": 0.3,
        "left": 1.25,
        "right": 0.042,
    }


def test_no_return_stays_none() -> None:
    ranges = parse_tof({"t": 1.0, "front": None, "left": 800, "right": None})
    assert ranges == {"front": None, "left": 0.8, "right": None}


def test_a_missing_sensor_reads_as_no_return() -> None:
    assert parse_tof({"t": 1.0}) == {"front": None, "left": None, "right": None}
