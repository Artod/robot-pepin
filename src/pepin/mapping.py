"""Occupancy-grid mapping from recorded poses and lidar scans.

A log-odds grid: every lidar return raises the odds of its cell being
occupied and lowers the odds of every cell the beam crossed. Fed with raw
odometry poses this produces the "before" map — the honest picture of how
far dead reckoning drifts — and later the same grid takes corrected poses.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pepin.odometry import Pose2D

LOG_ODDS_HIT = 0.85
LOG_ODDS_MISS = -0.4
LOG_ODDS_CLAMP = 5.0


@dataclass(frozen=True)
class GridSpec:
    """Grid geometry: cell size and the world extent it covers, in meters."""

    resolution_m: float = 0.05
    x_min_m: float = -10.0
    y_min_m: float = -10.0
    width_m: float = 20.0
    height_m: float = 20.0

    @property
    def shape(self) -> tuple[int, int]:
        """Grid size in cells as (rows, cols) = (height, width), rounded up."""
        return (
            math.ceil(self.height_m / self.resolution_m),
            math.ceil(self.width_m / self.resolution_m),
        )


def transform_to_world(points_xy: NDArray[np.float64], pose: Pose2D) -> NDArray[np.float64]:
    """Robot-frame points (N, 2) into the world frame given the robot pose."""
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    rotation = np.array([[c, -s], [s, c]])
    world: NDArray[np.float64] = points_xy @ rotation.T + np.array([pose.x, pose.y])
    return world


class OccupancyGrid:
    """Log-odds occupancy grid; rows are y, columns are x."""

    def __init__(self, spec: GridSpec) -> None:
        """Allocates the log-odds array for ``spec``; zero everywhere means "unknown"."""
        self.spec = spec
        self.log_odds: NDArray[np.float64] = np.zeros(spec.shape, dtype=np.float64)
        self.version = 0  # bumped by every integrate(); caches built from the grid key on it

    def world_to_cell(self, xy: NDArray[np.float64]) -> NDArray[np.int64]:
        """(N, 2) world points to (N, 2) integer [row, col]; may fall outside the grid."""
        cols = np.floor((xy[:, 0] - self.spec.x_min_m) / self.spec.resolution_m)
        rows = np.floor((xy[:, 1] - self.spec.y_min_m) / self.spec.resolution_m)
        return np.column_stack((rows, cols)).astype(np.int64)

    def _inside(self, cells: NDArray[np.int64]) -> NDArray[np.bool_]:
        """Boolean mask of the (N, 2) cells that actually land on the grid."""
        rows, cols = self.spec.shape
        return (cells[:, 0] >= 0) & (cells[:, 0] < rows) & (cells[:, 1] >= 0) & (cells[:, 1] < cols)

    def integrate(self, pose: Pose2D, points_robot: NDArray[np.float64]) -> None:
        """Add one scan taken at ``pose``: free space along each beam, a hit at its end.

        ``points_robot`` is (N, 2) meters in the robot frame; cells gain
        +0.85 log-odds per hit, -0.4 per crossing, clamped to +-5.
        """
        if len(points_robot) == 0:
            return
        hits = transform_to_world(points_robot, pose)
        origin = np.array([pose.x, pose.y])
        # Sample each beam at one cell spacing, stopping short of the hit cell.
        lengths = np.linalg.norm(hits - origin, axis=1)
        steps = max(1, int(np.ceil(lengths.max() / self.spec.resolution_m)))
        fractions = np.linspace(0.0, 1.0, steps, endpoint=False)[1:]
        samples = origin + (hits - origin)[:, None, :] * fractions[None, :, None]
        keep = fractions[None, :] * lengths[:, None] < lengths[:, None] - self.spec.resolution_m
        free = self.world_to_cell(samples[keep].reshape(-1, 2))
        free = free[self._inside(free)]
        np.add.at(self.log_odds, (free[:, 0], free[:, 1]), LOG_ODDS_MISS)
        hit_cells = self.world_to_cell(hits)
        hit_cells = hit_cells[self._inside(hit_cells)]
        np.add.at(self.log_odds, (hit_cells[:, 0], hit_cells[:, 1]), LOG_ODDS_HIT)
        np.clip(self.log_odds, -LOG_ODDS_CLAMP, LOG_ODDS_CLAMP, out=self.log_odds)
        self.version += 1

    def save(self, path: str | Path) -> None:
        """Write the grid (log-odds and geometry) to a compressed ``.npz`` file."""
        np.savez_compressed(path, log_odds=self.log_odds, spec=np.array(astuple(self.spec)))

    @classmethod
    def load(cls, path: str | Path) -> OccupancyGrid:
        """Read a grid saved with :meth:`save`."""
        data = np.load(path)
        res, x_min, y_min, width, height = (float(v) for v in data["spec"])
        grid = cls(GridSpec(res, x_min, y_min, width, height))
        grid.log_odds = data["log_odds"].astype(np.float64)
        return grid

    def occupied_xy(self, threshold: float = 0.7) -> NDArray[np.float64]:
        """World (x, y) centres of cells whose occupancy probability exceeds ``threshold``."""
        rows, cols = np.nonzero(self.probability() > threshold)
        xs = self.spec.x_min_m + (cols + 0.5) * self.spec.resolution_m
        ys = self.spec.y_min_m + (rows + 0.5) * self.spec.resolution_m
        return np.column_stack((xs, ys))

    def probability(self) -> NDArray[np.float64]:
        """Occupancy probability per cell, 0.5 where nothing was observed."""
        probability: NDArray[np.float64] = 1.0 / (1.0 + np.exp(-self.log_odds))
        return probability


OCCUPIED_LOG_ODDS = 4.0
FREE_LOG_ODDS = -4.0


def grid_from_pgm(yaml_path: str | Path) -> OccupancyGrid:
    """A map_server map (yaml + trinary pgm) as the log-odds grid the tracker searches.

    The same three values the robot gets over /map: a wall pixel (0) becomes occupied, a free
    pixel (254) free, and the unknown grey (205) stays 0 — never free. Calling 205 free once
    moved every relocalisation score by 0.3 in an offline replica.
    """
    import yaml

    path = Path(yaml_path)
    meta = yaml.safe_load(path.read_text())
    with open(path.parent / meta["image"], "rb") as f:
        magic = f.readline()
        if magic.strip() != b"P5":
            raise ValueError(f"{meta['image']}: not a binary pgm")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        width, height = (int(v) for v in line.split())
        f.readline()  # maxval
        pixels = np.frombuffer(f.read(), dtype=np.uint8).reshape(height, width)[::-1]
    res = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    grid = OccupancyGrid(GridSpec(res, ox, oy, width * res, height * res))
    grid.log_odds[:] = np.where(
        pixels < 64, OCCUPIED_LOG_ODDS, np.where(pixels > 250, FREE_LOG_ODDS, 0.0)
    )
    return grid


MAP_TOPIC = "map"  # the map a tracker matches on unless it is told otherwise
MAP_REFRESH_S = 0.0  # ...and how long before a newer one on that same topic may replace it
# How long a tracker waits for the map it was asked to match on before it takes the one that is
# there instead. Only ever while NOTHING has been adopted yet: a map already in use survives its
# publisher going away (it is a grid in memory, not a subscription), so a link lost mid-drive
# costs nothing, while a link that was never up would otherwise leave the board with no map at
# all. Ten seconds because a served file is latched and arrives in the first second, while a grid
# a mapping process builds only starts once that process is up.
MAP_FALLBACK_S = 10.0


class MapChoice:
    """Which of several maps a tracker matches on, and when a newly arrived one replaces it.

    A robot can be handed more than one picture of the same room: the file a map server serves,
    and a grid that is still being built while the cart drives. The node keeps the
    newest message of every topic; this holds the decision, so the node itself branches on
    nothing. A map on a topic nobody asked for is kept and not adopted; the first map on the
    asked-for topic is adopted; a later one on the SAME topic is adopted only once
    ``refresh_s`` seconds have passed AND its cells have actually changed.

    That gate is the point. Adopting a map is expensive and destructive — the caller rebuilds
    its matcher and its tracker and forgets the evidence gathered on the old map — while a
    growing grid is republished every second. ``refresh_s`` 0 is the behaviour a served file
    has always had: the first map, and no other.

    And the wanted map may never come. A grid built by a process on another machine arrives only
    once that machine is up, while a served file is the board's own, so a tracker asking for the
    first with the wifi down would wait for ever with a map sitting on the other topic (CLAUDE.md
    rule 20: nothing on the board may depend on the laptop to start). After ``fallback_after_s``
    with nothing adopted at all, :meth:`lapsed` names ``fallback`` and the caller offers what is
    waiting there; the wanted map still replaces it the moment it arrives. Once something IS
    adopted the fallback is over for good — a grid in memory does not stop working because its
    publisher went away.

    A MAP WITH NOTHING IN IT IS NOT A MAP. A map being built in an unknown room starts as an
    all-unknown grid: adopting that one spends the single adoption a
    ``refresh_s`` of 0 allows, and the tracker then refuses every real map that follows for the
    rest of the session. So :meth:`offer` takes an optional ``empty`` question and turns such a
    grid away — counted like any other refusal, so the wait is visible — until the first sweep
    has put something in it.
    """

    def __init__(
        self,
        wanted: str = MAP_TOPIC,
        refresh_s: float = MAP_REFRESH_S,
        fallback: str = MAP_TOPIC,
        fallback_after_s: float = MAP_FALLBACK_S,
    ) -> None:
        self._wanted = wanted
        self._refresh_s = refresh_s
        self._fallback = fallback
        self._fallback_after_s = fallback_after_s
        self._on_choice: Callable[[], None] = lambda: None
        self.source = ""  # the topic the map in use came from ("" until the first is adopted)
        self.digest = ""  # and what its cells were
        self.taken_s = 0.0  # when it was adopted, on the caller's clock
        self.fell_back = False  # is the map in use the fallback, taken while the wanted one hid?
        self._waiting_since_s: float | None = None  # when the wait for the wanted map began
        self._ignored = 0  # arrivals turned away since the last report

    switches = ("map_topic", "map_refresh_s", "map_fallback_s")

    def on_choice(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` whenever the wanted topic changes: the caller offers it whatever
        each topic last published, so a map published once and latched long ago is adopted now
        rather than never."""
        self._on_choice = callback

    def switch(self, name: str, value: object) -> None:
        """Move one of :attr:`switches` (the node's live flags) — the topic to match on, or the
        least time between two adoptions of it."""
        if name == "map_refresh_s":
            self._refresh_s = float(value)  # type: ignore[arg-type]
            return
        if name == "map_fallback_s":
            self._fallback_after_s = float(value)  # type: ignore[arg-type]
            return
        moved = str(value) != self._wanted
        self._wanted = str(value)
        if moved:
            self._waiting_since_s = None  # the wait for THIS map starts now
            self._on_choice()

    @property
    def wanted(self) -> str:
        """The topic the tracker is asking for, whether or not a map has arrived on it."""
        return self._wanted

    def take_ignored(self) -> int:
        """How many arrivals the gate turned away since this was last asked, and reset."""
        count, self._ignored = self._ignored, 0
        return count

    def offer(
        self,
        source: str,
        digest: Callable[[], str],
        now: float,
        adopt: Callable[[], None],
        empty: Callable[[], bool] | None = None,
    ) -> bool:
        """A map has arrived on ``source``: call ``adopt`` and answer ``True`` when it becomes
        the map in use, else turn it away and answer ``False``.

        ``digest`` and ``empty`` are functions, not values, because reading a whole grid's cells
        costs something and the answer usually does not hang on them. ``empty`` answers "has this
        grid no known cell at all" and is what keeps an unknown room's first, blank publication
        from spending the one adoption a ``refresh_s`` of 0 allows; a caller that does not ask it
        behaves exactly as before.
        """
        if source != self._wanted and not (self.fell_back and source == self._fallback):
            return False
        if empty is not None and empty():
            self._ignored += 1  # a grid with nothing in it: wait for the sweep that fills it
            return False
        fresh = ""
        if self.source == source:
            if self._refresh_s <= 0.0 or now - self.taken_s < self._refresh_s:
                self._ignored += 1
                return False
            fresh = digest()
            if fresh == self.digest:
                self._ignored += 1
                return False
        self.source = source
        self.taken_s = now
        self.digest = fresh or digest()
        self.fell_back = source != self._wanted
        adopt()
        return True

    def lapsed(self, now: float) -> str | None:
        """The topic to take a map from instead, because the wanted one has never spoken: the
        fallback's name once ``fallback_after_s`` has passed with nothing adopted at all, else
        None. The caller then offers whatever is waiting on that topic and :meth:`offer` does
        the rest; calling this every second is the intended use.

        Silent for ever once a map is in use: a tracker that HAS a map has nothing to gain from
        a rebuild on a lesser one, whatever happened to the publisher (CLAUDE.md rule 20 is
        about starting without the laptop, not about surviving it).
        """
        if self.source or self._wanted == self._fallback or self._fallback_after_s <= 0.0:
            return None
        if self._waiting_since_s is None:
            self._waiting_since_s = now  # the clock starts the first time it is asked
            return None
        if now - self._waiting_since_s < self._fallback_after_s:
            return None
        self.fell_back = True  # so the fallback's map is accepted by offer()
        return self._fallback
