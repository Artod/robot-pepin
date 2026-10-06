"""The ring: a drive's light topics recorded all the time, and a goal's bag cut out of them.

A goal's bag of its own (``ros2 bag record`` started on the goal's word) hears its first
message 0.3-0.4 s after the word, so it holds nothing of the robot at rest before the command:
a cold OpenVINS replay, which needs a second of rest before the motion
(ros/maps/vio/estimator_config.yaml ``init_window_time``), never initialises on it, and a shove
or a carry before the goal is not in it at all. The ring records the same topics without a
break (:class:`pepin.board_bag.Supervisor`: a file a minute under ``<ring>/<UTC start>/``,
pruned by age, size and free space), and a goal's bag is the window ``[goal - preroll_s,
end + tail_s]`` cut out of it (:mod:`pepin.bag_slice`) into the directory a per-goal recorder
writes, so every reader of ``ros/maps/rec`` finds what it found before.

This module is the arithmetic of that: which files hold a window, when the window is on disk,
and what a fresh subscription would have heard of a latched topic.
"""

from __future__ import annotations

import calendar
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pepin.board_bag import SPLIT_S

# rosbag2's name for the i-th file of a split recording: <directory name>_<i>.mcap
SEGMENT_NAME = re.compile(r"^(?P<bag>.+)_(?P<index>\d+)\.mcap$")
# A recording directory's name is its UTC start (pepin.board_bag.bag_name)
STAMP_FORMAT = "%Y%m%d_%H%M%SZ"
# File times are the host's (the bind mount) and the window the container's: the same clock to a
# few ms under Docker Desktop, and this much is allowed between the two
CLOCK_SLACK_S = 1.0
LATCHED_LIMIT = 64  # distinct messages kept per latched topic: /tf_static has a handful
NS = 1_000_000_000


def segment_index(path: Path) -> int:
    """rosbag2's split number of a file (``<dir>_<i>.mcap`` -> i); -1 for any other name."""
    match = SEGMENT_NAME.match(path.name)
    return int(match["index"]) if match else -1


def bag_start(name: str) -> float | None:
    """A recording directory's start from its name (``20261005_201500Z``); None for another."""
    try:
        return float(calendar.timegm(time.strptime(name, STAMP_FORMAT)))
    except ValueError:
        return None


@dataclass(frozen=True)
class Segment:
    """One file of the ring: where it is, the span it covers (wall seconds) and its size.

    ``end`` is its last write; ``start`` is the previous file's last write, or the recording's
    start for its first file — an estimate that only ever errs early, which costs a file read.
    """

    path: Path
    start: float
    end: float
    size: int


def segments(root: Path) -> list[Segment]:
    """Every file of every recording under ``root``, in recording order then split order."""
    by_bag: dict[Path, list[tuple[int, Path, float, int]]] = {}
    for path in root.glob("*/*.mcap"):
        index = segment_index(path)
        if index < 0:
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:  # pruned between the listing and the stat
            continue
        by_bag.setdefault(path.parent, []).append((index, path, stat.st_mtime, stat.st_size))
    found: list[Segment] = []
    for bag in sorted(by_bag):
        files = sorted(by_bag[bag])
        start = bag_start(bag.name)
        previous = start if start is not None else files[0][2] - SPLIT_S
        for _index, path, mtime, size in files:
            found.append(Segment(path, min(previous, mtime), mtime, size))
            previous = mtime
    return found


def covering(
    found: Sequence[Segment], lo: float, hi: float, slack_s: float = CLOCK_SLACK_S
) -> list[Segment]:
    """The files that may hold a message logged in ``[lo, hi]``, plus the file before the first
    of them in the same recording: the latched topics' last message before ``lo`` is there."""
    chosen = [i for i, s in enumerate(found) if s.end >= lo - slack_s and s.start <= hi + slack_s]
    if not chosen:
        return []
    first = chosen[0]
    if first > 0 and found[first - 1].path.parent == found[first].path.parent:
        chosen.insert(0, first - 1)
    return [found[i] for i in chosen]


def written_past(found: Sequence[Segment], t: float) -> bool:
    """True once the ring's newest file was written after ``t``: rosbag2 writes a chunk at a time,
    so a message logged at ``t`` is on disk once a later write is."""
    return bool(found) and max(s.end for s in found) >= t + CLOCK_SLACK_S


@dataclass(frozen=True)
class Window:
    """The span of a goal's bag in log-time nanoseconds: ``[goal - preroll, end + tail]``."""

    start_ns: int
    end_ns: int

    @classmethod
    def around(cls, goal_s: float, end_s: float, preroll_s: float, tail_s: float) -> Window:
        """The window of a goal sent at ``goal_s`` and ended at ``end_s`` (wall seconds)."""
        if end_s < goal_s:
            raise ValueError(f"a goal cannot end ({end_s}) before it starts ({goal_s})")
        if preroll_s < 0 or tail_s < 0:
            raise ValueError("preroll and tail are durations")
        return cls(round((goal_s - preroll_s) * NS), round((end_s + tail_s) * NS))

    @property
    def start_s(self) -> float:
        """The window's start in wall seconds."""
        return self.start_ns / NS

    @property
    def end_s(self) -> float:
        """The window's end in wall seconds: the cut waits for the ring to be written past it."""
        return self.end_ns / NS

    @property
    def duration_s(self) -> float:
        """The window's length in seconds."""
        return (self.end_ns - self.start_ns) / NS


class LatchedStore:
    """What a fresh subscription to a latched topic hears: every distinct message, newest last.

    The ring subscribes once, when it starts, so a latched topic (``/tf_static``, published once
    by each of its publishers) is in its first file and nowhere after; the file is pruned in a
    few hours. A per-goal recorder heard those messages at every goal, and so must a cut.
    """

    def __init__(self, limit: int = LATCHED_LIMIT) -> None:
        self._limit = limit
        self._messages: OrderedDict[bytes, None] = OrderedDict()
        self._lock = threading.Lock()  # heard on the executor, read by the cutting thread

    def add(self, data: bytes) -> None:
        """A message heard; a repeat moves to the end, the oldest goes past the limit."""
        with self._lock:
            self._messages.pop(data, None)
            self._messages[data] = None
            while len(self._messages) > self._limit:
                self._messages.popitem(last=False)

    def messages(self) -> list[bytes]:
        """The distinct messages, oldest first."""
        with self._lock:
            return list(self._messages)


@dataclass(frozen=True)
class SliceJob:
    """One goal's bag to cut: where it goes, its window, the run's number and the goal's time."""

    out: Path
    window: Window
    run: int
    goal_s: float
