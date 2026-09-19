"""The map the board last adopted, on the board's own card.

Until now the board could only start from a pgm served by map_server: the file was the cold-boot
map, the static layer's map, and the thing a fresh room had to be mapped into before anything could
drive. The owner's rule is one map — the volume — and a board that persists what it ADOPTED needs no
file in the loop: the tracker is already the one holder of the map (it adopts, it rebuilds, it owns
``map -> odom``), so it is the one that can write the map down and read it back.

What is written is what an adoption needs to be repeated: the cells, the geometry, the id the words
of other machines are stamped with, the minted identity when the publisher sends one, the digest the
choice compares, when it was written and which topic it came from. The cells are run-length encoded
because a costmap-shaped grid is runs of one value -- 51385 cells of this flat's map are 195 kB of
raw JSON and 16 kB encoded (scratch/costmap_rle_cost.py).

Nothing here is ROS and nothing here decides: a node offers a grid and asks for one back.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CACHE_NAME",
    "BootAnswer",
    "CacheBoot",
    "MapCache",
    "read_cache",
    "run_length_decode",
    "run_length_encode",
    "save",
    "write_cache",
]

# The one file, beside the maps, named as a literal here and nowhere else. A cache keyed by map id
# would grow a file per room and never be pruned; the board tracks one room at a time and the id it
# holds is inside.
CACHE_NAME = "map_cache.json"
VERSION = 1  # bumped when a field's meaning changes; a cache of another version is refused


def run_length_encode(cells: list[int]) -> list[int]:
    """``[v, n, v, n, ...]``: each value of ``cells`` and how many times it repeats, in order.

    Lossless and flat, so a reader needs no library and the record stays JSON. A map is runs of one
    value, which is why this is worth doing at all: this flat's 51385 cells go from 195 kB of JSON
    to 16 kB (scratch/costmap_rle_cost.py).
    """
    out: list[int] = []
    for value in cells:
        if out and out[-2] == value:
            out[-1] += 1
        else:
            out.extend((int(value), 1))
    return out


def run_length_decode(runs: list[int]) -> list[int]:
    """The cells back, so a reader needs to know nothing about how they were stored."""
    if len(runs) % 2:
        raise ValueError(f"a run-length list is pairs of value and count, not {len(runs)} numbers")
    out: list[int] = []
    for value, count in zip(runs[::2], runs[1::2], strict=True):
        if count < 0:
            raise ValueError(f"a run of {count} cells is not a run")
        out.extend([int(value)] * int(count))
    return out


@dataclass(frozen=True)
class MapCache:
    """One adopted map, as the board keeps it: the cells and their geometry, the id other machines
    stamp their words with, the publisher's minted identity when there is one, the digest the map
    choice compares, the wall clock it was written at and the topic it came from."""

    cells: list[int]
    width: int
    height: int
    resolution_m: float
    origin_xy: tuple[float, float]
    map_id: str
    digest: str
    stamp: float
    source: str
    identity: str = ""  # pepin.worldmap.MapIdentity's minted name, when the publisher sends one

    def age_s(self, now: float | None = None) -> float:
        """Seconds since it was written; ``inf`` when the stamp is not a time at all."""
        if not math.isfinite(self.stamp) or self.stamp <= 0.0:
            return math.inf
        return max(0.0, (time.time() if now is None else now) - self.stamp)

    def phrase(self, now: float | None = None) -> str:
        """``the cache of 239x215@-18.53,-4.38 from map_lidar, written 7.4 h ago`` for a report
        line: an operator must never mistake a cold boot for a live map."""
        age = self.age_s(now)
        when = "of unknown age" if age == math.inf else f"written {age / 3600.0:.1f} h ago"
        return (
            f"the cache of {self.map_id or 'an unnamed map'}"
            f"{f' ({self.identity})' if self.identity else ''} from {self.source}, {when}"
        )

    def to_json(self) -> str:
        """The cache as one JSON object, cells run-length encoded."""
        return json.dumps(
            {
                "version": VERSION,
                "width": int(self.width),
                "height": int(self.height),
                "resolution": float(self.resolution_m),
                "origin": [float(self.origin_xy[0]), float(self.origin_xy[1])],
                "map": self.map_id,
                "identity": self.identity,
                "digest": self.digest,
                "stamp": float(self.stamp),
                "source": self.source,
                "encoding": "rle",
                "cells": run_length_encode(self.cells),
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, text: str) -> MapCache:
        """A cache back from :meth:`to_json`; ``ValueError`` for anything else — a truncated file,
        another version, a cell count that does not fill the grid. Never half a map: a board that
        would rather guess than refuse is how a tracker ends up matching against nothing."""
        raw: Any = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("a map cache is one JSON object")
        if int(raw.get("version", 0)) != VERSION:
            raise ValueError(f"map cache version {raw.get('version')}, not {VERSION}")
        if raw.get("encoding") != "rle":
            raise ValueError(f"map cache encoding {raw.get('encoding')!r}, not 'rle'")
        cells = run_length_decode(list(raw["cells"]))
        width, height = int(raw["width"]), int(raw["height"])
        if width <= 0 or height <= 0 or len(cells) != width * height:
            raise ValueError(f"{len(cells)} cells do not fill a {width}x{height} grid")
        origin = tuple(float(v) for v in raw["origin"])
        if len(origin) != 2:
            raise ValueError(f"an origin is x and y, not {raw['origin']!r}")
        return cls(
            cells=cells,
            width=width,
            height=height,
            resolution_m=float(raw["resolution"]),
            origin_xy=(origin[0], origin[1]),
            map_id=str(raw.get("map", "")),
            digest=str(raw.get("digest", "")),
            stamp=float(raw.get("stamp", 0.0)),
            source=str(raw.get("source", "")),
            identity=str(raw.get("identity", "")),
        )


@dataclass(frozen=True)
class BootAnswer:
    """What a cold-boot attempt came to: the map to adopt as a list of none or one (``taken``) and
    what to say about it as a list of none or one ``(level, text)`` pair (``words``).

    Lists and not optionals because the caller is a ROS node and the node is not allowed to grow
    decisions (tests/unit/test_ros_nodes_importable.py counts its branches): it iterates over what
    this answer holds and does it. Whether there is anything to take, and whether anything needs
    saying, is decided here, where it can be tested without a node."""

    taken: list[MapCache]
    words: list[tuple[str, str]]


@dataclass
class CacheBoot:
    """When a board with no live map may fall back to the map it wrote down itself — one decision,
    kept out of the node: the cache is read at most ONCE per start, only while the node is switched
    on for it, only when nothing has been adopted, and only after the live topics have had their
    patience (``map_fallback_s``). The once-only latch is here because a node that re-read a
    refused cache every check would fill the log with the same refusal."""

    tried: bool = False

    def due(self, *, enabled: bool, have_map: bool, waited_s: float, patience_s: float) -> bool:
        """True exactly once: the moment the cache is the only map left to try."""
        if self.tried or not enabled or have_map or waited_s < patience_s:
            return False
        self.tried = True
        return True

    def attempt(
        self,
        directory: Path,
        *,
        enabled: bool,
        have_map: bool,
        waited_s: float,
        patience_s: float,
    ) -> BootAnswer:
        """Try the cache once, and say what came of it (:class:`BootAnswer`).

        Three outcomes and each of them is loud in its own way: a cache that reads, which the caller
        adopts and announces as a CACHE with its age so nobody mistakes a cold boot for a live map;
        no cache at all, which is a board that has never adopted a map and must be told the two ways
        out; and a cache that does not read, which is refused whole — half a map is worse than none,
        because a tracker would match against it and believe the answer.
        """
        if not self.due(
            enabled=enabled, have_map=have_map, waited_s=waited_s, patience_s=patience_s
        ):
            return BootAnswer([], [])
        try:
            cache = read_cache(directory)
        except (OSError, ValueError) as exc:
            return BootAnswer(
                [],
                [
                    (
                        "error",
                        f"THE MAP CACHE IS NOT READABLE and was refused whole: {exc}. This board"
                        " has no map; start the laptop's volume or launch with map_server:=true",
                    )
                ],
            )
        if cache is None:
            return BootAnswer(
                [],
                [
                    (
                        "error",
                        "NO MAP AND NO CACHE: this board has never adopted a map, so it has"
                        " nothing to track on. Bring up the laptop's volume (map_lidar) once, or"
                        " launch with map_server:=true to seed the room from a file",
                    )
                ],
            )
        return BootAnswer([cache], [("warning", f"no live map: tracking on {cache.phrase()}")])


def save(directory: Path, cache: MapCache, *, enabled: bool) -> str:
    """Write ``cache`` (:func:`write_cache`) and return the one line to log about it.

    Never raises: a board that cannot write its cache still tracks, it only has a colder boot ahead
    of it, and that is worth a line and not an exception. ``enabled`` off writes nothing and says
    so, so an operator reading the log can see which it was.
    """
    if not enabled:
        return "map cache off: this adoption was not written down (flag map_cache)"
    try:
        written = write_cache(directory, cache)
    except OSError as exc:
        return f"the map cache could not be written: {exc}"
    return f"map cache written: {written} ({len(cache.cells)} cells, id {cache.map_id})"


def write_cache(directory: Path, cache: MapCache) -> Path:
    """Write ``cache`` into ``directory`` as :data:`CACHE_NAME`, atomically; returns the path.

    Atomically because the alternative is a board that loses its only map to a power cut in the
    middle of a write: the JSON goes to a temporary file in the SAME directory, is flushed and
    fsynced, and is then moved onto the name with ``os.replace``, which is atomic on one
    filesystem. A reader therefore sees either the previous cache whole or the new one whole, never
    a half-written grid. The directory is fsynced too, so the rename itself survives the cut.
    """
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / CACHE_NAME
    temporary = directory / f".{CACHE_NAME}.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(cache.to_json())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, final)
    handle = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(handle)
    except OSError:  # some filesystems refuse to fsync a directory; the replace is still atomic
        pass
    finally:
        os.close(handle)
    return final


def read_cache(directory: Path) -> MapCache | None:
    """The cache in ``directory``, or ``None`` when there is none; ``ValueError`` when there is one
    and it is not readable — a caller must say that out loud rather than start on a guess."""
    path = directory / CACHE_NAME
    if not path.exists():
        return None
    return MapCache.from_json(path.read_text(encoding="utf-8"))
