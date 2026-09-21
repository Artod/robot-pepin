"""The saved pose and the frame hold (pepin.lastpose): pure decisions, no ROS."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from pepin.lastpose import FrameHold, known_room, read_saved_pose, saved_start
from pepin.mapcache import MapCache, write_cache

MAP = "144x216@-4.57,-4.92"


def _save(path: Path, **changes: object) -> None:
    record = {"x": -0.4, "y": 2.9, "theta": 1.7, "fit": 0.9, "time": time.time(), "map": MAP}
    path.write_text(json.dumps(record | changes))


def _cache(directory: Path, map_id: str = MAP) -> None:
    write_cache(
        directory,
        MapCache(
            cells=[0] * 4, width=2, height=2, resolution_m=0.05, origin_xy=(0.0, 0.0),
            map_id=map_id, digest="d", stamp=time.time(), source="/map",
        ),
    )  # fmt: skip


def test_a_recent_good_pose_saved_on_this_map_is_where_the_tracker_starts(tmp_path: Path) -> None:
    file = tmp_path / "last_pose.json"
    _save(file)
    answer = saved_start(file, MAP, time.time(), 3600.0, 0.5)
    assert answer.pose is not None and (answer.pose.x, answer.pose.y) == (-0.4, 2.9)
    assert "starting from the last known pose (-0.40, +2.90" in answer.note
    assert math.isclose(answer.pose.theta, 1.7)


def test_every_refusal_of_the_saved_pose_says_why(tmp_path: Path) -> None:
    """A start that silently fell back to the odometry's pose is found an hour late."""
    file = tmp_path / "last_pose.json"
    assert "no saved pose" in saved_start(file, MAP, time.time(), 3600.0, 0.5).note
    _save(file, map="other")
    other = saved_start(file, MAP, time.time(), 3600.0, 0.5)
    assert other.pose is None and "another map (other" in other.note
    _save(file, time=time.time() - 7200.0)
    assert "s old" in saved_start(file, MAP, time.time(), 3600.0, 0.5).note
    _save(file, fit=0.2)
    assert "its fit was 0.20" in saved_start(file, MAP, time.time(), 3600.0, 0.5).note
    file.write_text("{half a rec")
    assert read_saved_pose(file) is None


def test_the_room_is_known_only_when_the_pose_was_saved_on_the_cached_map(tmp_path: Path) -> None:
    file = tmp_path / "last_pose.json"
    assert not known_room(file, tmp_path), "nothing on disk: a map being born"
    _save(file)
    assert not known_room(file, tmp_path), "a pose and no cached map"
    _cache(tmp_path, "another")
    assert not known_room(file, tmp_path), "a pose saved on a map that is not the cached one"
    _cache(tmp_path)
    assert known_room(file, tmp_path)


def test_the_frame_is_held_in_a_known_room_until_a_pose_defines_it() -> None:
    asked: list[int] = []

    def known() -> bool:
        asked.append(1)
        return True

    hold = FrameHold(known)
    assert hold.silent(True) and hold.silent(True)
    assert len(asked) == 1, "the disk is asked once, not twenty times a second"
    assert not hold.silent(False), "the gate off is the start this tracker always made"
    hold.define()
    assert not hold.silent(True)
    assert not FrameHold(lambda: False).silent(True), "a map being born: the identity is the truth"
