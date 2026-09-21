"""Where the cart stood when the stack last ran, and what that is worth at the next start.

The tracker writes its pose every two seconds (``last_pose.json``: x, y, theta, fit, time, and
the id of the map it was tracked on). At the next start that file answers two questions, and both
are decisions, so they live here and not in the ROS node:

* :func:`saved_start` — is the saved pose a place to START from on the map now in hand? Only
  when it is recent, was good, and was saved on THIS map; the answer carries its reason either
  way, because a start that silently fell back to the odometry's own pose is the kind of thing a
  person finds an hour later.
* :class:`FrameHold` — may ``map -> odom`` be SAID yet? A default is a refusal, never (0, 0): in a
  known room (the disk holds a cached map and a pose saved on that very map,
  :func:`known_room`) the identity is a lie for as long as the tracker waits for its map, and the
  jump that ends it moves every consumer's world by metres (2026-09-21: 9.5 s of identity, then
  2.9 m — the jump that sent Nav2's RangeSensorLayer into its endless loop). In a map being born
  under the cart the identity is the truth from the first tick, and stays.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pepin.mapcache import read_cache
from pepin.odometry import Pose2D


@dataclass(frozen=True)
class SavedPose:
    """One record of ``last_pose.json``."""

    pose: Pose2D
    fit: float
    time: float
    map_id: str


@dataclass(frozen=True)
class StartAnswer:
    """The pose to start from, or ``None`` with the reason, and the sentence for the log."""

    pose: Pose2D | None
    note: str


def read_saved_pose(path: Path) -> SavedPose | None:
    """The record at ``path``; ``None`` when there is none or it cannot be read as one."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return SavedPose(
            Pose2D(float(raw["x"]), float(raw["y"]), float(raw["theta"])),
            float(raw.get("fit", 0.0)),
            float(raw["time"]),
            str(raw.get("map", "")),
        )
    except (OSError, KeyError, ValueError, TypeError):
        return None


def saved_start(
    path: Path, map_id: str, now: float, max_age_s: float, min_fit: float
) -> StartAnswer:
    """Whether the pose saved at ``path`` is where a tracker on map ``map_id`` should start:
    saved on this very map, no older than ``max_age_s``, at a fit of ``min_fit`` or better."""
    saved = read_saved_pose(path)
    if saved is None:
        return StartAnswer(None, "no saved pose: starting from the pose the odometry gives")
    age = now - saved.time
    if saved.map_id != map_id:
        why = f"it was saved on another map ({saved.map_id or 'unnamed'}, this is {map_id})"
    elif age > max_age_s:
        why = f"it is {age:.0f} s old"
    elif saved.fit < min_fit:
        why = f"its fit was {saved.fit:.2f}"
    else:
        p = saved.pose
        return StartAnswer(
            p,
            f"starting from the last known pose ({p.x:+.2f}, {p.y:+.2f},"
            f" {math.degrees(p.theta):+.0f} deg), saved {age:.0f} s ago",
        )
    return StartAnswer(None, f"the saved pose is not used: {why}; starting from the odometry's")


def known_room(pose_file: Path, cache_dir: Path) -> bool:
    """Whether the disk says where the cart is: a cached map, and a pose saved on that very map.
    Anything missing or unreadable is "no" — then the identity stands, as in a map being born."""
    saved = read_saved_pose(pose_file)
    try:
        cache = read_cache(cache_dir)
    except (OSError, ValueError, TypeError):
        return False
    return (
        saved is not None
        and cache is not None
        and bool(saved.map_id)
        and (cache.map_id == saved.map_id)
    )


class FrameHold:
    """Whether ``map -> odom`` must stay unsaid: in a known room, until a pose has defined it."""

    def __init__(self, room_is_known: Callable[[], bool]) -> None:
        """``room_is_known`` is asked once, at the first question: the answer is about how this
        process started, and the broadcast asks twenty times a second."""
        self._room_is_known = room_is_known
        self._known: bool | None = None
        self.defined = False

    def define(self) -> None:
        """A pose has set the frame (an update, an accepted seed): it may be said from now on."""
        self.defined = True

    def silent(self, enabled: bool) -> bool:
        """True while the frame must not be broadcast; never with the gate off (``enabled``)."""
        if self.defined or not enabled:
            return False
        if self._known is None:
            self._known = bool(self._room_is_known())
        return self._known
