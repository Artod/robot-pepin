"""Which sensors feed the tracker, which of them are alive right now, and whose scan drives
the next update.

The tracker matches whatever scan-shaped evidence arrives against the same map: the lidar's
revolution, the camera's virtual scan (:func:`pepin.depth.depth_to_scan`) and the floor-contact
scan (:func:`pepin.contact.contact_scan`). This module is the roster: what each source is (its
frame, its field of view, how far it is trusted, when it counts as stale), the ``sources``
flag that says which of them are enabled, and a health record per source so a report line can
say "lidar fresh 9.9 Hz, depth stale 2.1 s, contact off". :class:`SourceFeed` is the one
trigger path on top of it: every source's newest scan waits at its own gate for the odometry
that covers it, one source — the anchor — drives the updates, and the others ride along, each
carried to the anchor's moment through the odometry. No ROS here: the node feeds stamps and
scans in and reads verdicts out.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pepin.dynamic import to_map
from pepin.scanmatch import relative_motion
from pepin.timeline import GateStats, OdomHistory, ScanGate, TimedScan

LIDAR = "lidar"
DEPTH = "depth"
CONTACT = "contact"
RATE_TAU_S = 2.0  # the rate's time constant: a few seconds of intervals, not the whole run


@dataclass(frozen=True)
class ScanSource:
    """One sensor as the tracker sees it: its name (the flag's word for it), the frame its
    returns arrive in, the fan it sees (degrees), ``trust`` (a weight on its information in a
    fusion, 1.0 for the lidar), ``stale_after_s`` (older than this it is not alive), and
    ``min_points`` / ``vote_min_points`` (fewer returns fix no pose / may not vote alone)."""

    name: str
    frame: str
    fov_deg: float
    trust: float = 1.0
    stale_after_s: float = 0.5
    min_points: int = 50
    vote_min_points: int = 60

    @property
    def partial(self) -> bool:
        """True for a fan narrower than a full turn: the source cannot see behind itself."""
        return self.fov_deg < 359.0


# The cart's three sources. The camera's fans are +-40 degrees (pepin.depth.SCAN_HALF_FOV) and
# already in base_link when they arrive; a fan of 161 half-degree bins holds a few dozen
# returns, so their thin-scan floors are a lidar's third. Their trust is half: a network's
# depth is not a range measurement, and the floor-contact line is a geometry read off one.
DEFAULT_SOURCES: tuple[ScanSource, ...] = (
    ScanSource(LIDAR, "laser", 360.0, trust=1.0, stale_after_s=0.5),
    ScanSource(
        DEPTH, "base_link", 80.0, trust=0.5, stale_after_s=1.0, min_points=20, vote_min_points=20
    ),
    ScanSource(
        CONTACT, "base_link", 80.0, trust=0.5, stale_after_s=1.0, min_points=20, vote_min_points=20
    ),
)


class SourceHealth:
    """One source's liveness from the stamps it delivers: the last stamp, the rate (a running
    mean over the last seconds) and the verdict — ``fresh`` (a scan within ``stale_after_s``),
    ``stale`` (had scans, the last one too old) or ``absent`` (never heard from)."""

    def __init__(self, stale_after_s: float, tau_s: float = RATE_TAU_S) -> None:
        self._stale_after_s = stale_after_s
        self._tau_s = tau_s
        self.last_stamp: float | None = None
        self.rate_hz = 0.0
        self.count = 0

    def observe(self, stamp: float) -> None:
        """A scan arrived with this stamp (seconds, the sensor's clock)."""
        if self.last_stamp is not None and stamp > self.last_stamp:
            dt = stamp - self.last_stamp
            rate = 1.0 / dt
            alpha = 1.0 - math.exp(-dt / self._tau_s) if self.count > 1 else 1.0
            self.rate_hz += alpha * (rate - self.rate_hz)
        self.last_stamp = stamp if self.last_stamp is None else max(self.last_stamp, stamp)
        self.count += 1

    def age_s(self, now: float) -> float:
        """Seconds since the last scan (infinite before the first)."""
        return math.inf if self.last_stamp is None else now - self.last_stamp

    def verdict(self, now: float) -> str:
        """``fresh``, ``stale`` or ``absent`` at time ``now`` (the same clock as the stamps)."""
        if self.last_stamp is None:
            return "absent"
        return "fresh" if self.age_s(now) <= self._stale_after_s else "stale"

    def text(self, now: float) -> str:
        """``fresh 9.9 Hz`` / ``stale 2.1 s`` / ``absent`` for a report line."""
        verdict = self.verdict(now)
        if verdict == "fresh":
            return f"fresh {self.rate_hz:.1f} Hz"
        if verdict == "stale":
            return f"stale {self.age_s(now):.1f} s"
        return verdict


class SourceRegistry:
    """The roster: every known source, which are enabled (the ``sources`` flag) and the health
    of each. ``alive`` is what the tracker may use now: enabled and fresh."""

    def __init__(
        self,
        sources: Iterable[ScanSource] = DEFAULT_SOURCES,
        enabled: Iterable[str] = (LIDAR,),
    ) -> None:
        self._sources = {s.name: s for s in sources}
        self._health = {s.name: SourceHealth(s.stale_after_s) for s in self._sources.values()}
        self._enabled: tuple[str, ...] = ()
        self.enable(enabled)

    @property
    def names(self) -> tuple[str, ...]:
        """Every known source, in roster order."""
        return tuple(self._sources)

    @property
    def enabled(self) -> tuple[str, ...]:
        """The sources the flag has switched on, in roster order."""
        return self._enabled

    def enable(self, names: Iterable[str]) -> None:
        """Set the ``sources`` flag: exactly these sources feed the tracker from now on. An
        unknown name is refused (``ValueError``) so a typo in ``ros2 param set`` cannot
        silently switch a sensor off."""
        wanted = set(names)
        unknown = wanted - set(self._sources)
        if unknown:
            raise ValueError(f"unknown sources {sorted(unknown)}; known: {list(self._sources)}")
        self._enabled = tuple(name for name in self._sources if name in wanted)

    def is_enabled(self, name: str) -> bool:
        """Whether the flag has this source on."""
        return name in self._enabled

    def source(self, name: str) -> ScanSource:
        """The source's description (``KeyError`` for a name not on the roster)."""
        return self._sources[name]

    def health(self, name: str) -> SourceHealth:
        """The source's health record."""
        return self._health[name]

    def observe(self, name: str, stamp: float) -> None:
        """A scan from ``name`` arrived with ``stamp``: its health record takes note."""
        self._health[name].observe(stamp)

    def alive(self, now: float) -> list[ScanSource]:
        """The sources the tracker may use at ``now``: enabled, and fresh."""
        return [
            self._sources[name]
            for name in self._enabled
            if self._health[name].verdict(now) == "fresh"
        ]

    def report(self, now: float) -> str:
        """One phrase per source: ``lidar fresh 9.9 Hz, depth stale 2.1 s, contact off``."""
        return ", ".join(
            f"{name} {self._health[name].text(now) if name in self._enabled else 'off'}"
            for name in self._sources
        )


@dataclass(frozen=True)
class ScanObservation:
    """One source's returns for one update: (N, 2) base-frame metres from ``source`` (a name
    on the tracker's :class:`SourceRegistry`), the scan's stamp, and — for tests — an explicit
    (N,) vote mask instead of the static map's."""

    source: str
    points: NDArray[np.float64]
    stamp: float = 0.0
    vote: NDArray[np.bool_] | None = None


@dataclass
class FeedStats:
    """What the feed did since the last report: every source's gate (scans offered, released
    as the anchor, replaced, expired, the wait for odometry), how many of its scans rode along
    with another source's update (``attached``) and how many were too old to and were dropped
    beside the anchor (``dropped``)."""

    gates: dict[str, GateStats] = field(default_factory=dict)
    attached: Counter[str] = field(default_factory=Counter)
    dropped: Counter[str] = field(default_factory=Counter)

    @property
    def released(self) -> int:
        """Scans handed to the matcher as the anchor, over every source."""
        return sum(g.released for g in self.gates.values())

    def summary(self) -> str:
        """One log line: a single source's gate summary as it always read, or one clause per
        source with its rides and drops."""
        if len(self.gates) == 1:
            return next(iter(self.gates.values())).summary()
        return "; ".join(
            f"{name}: {stats.summary()}, attached {self.attached[name]}, "
            f"dropped {self.dropped[name]}"
            for name, stats in self.gates.items()
        )


class SourceFeed:
    """Every source's newest scan waiting for odometry, and the choice of which one drives
    the tracker's next update.

    One path for lidar-only, camera-only and fused: the anchor (:meth:`anchor`) is the widest
    enabled source while it is fresh — the lidar — and when the registry says it is stale or
    absent, the fresh enabled source heard from last, so a dead lidar hands the tracker to the
    camera without a restart and a returning lidar takes it back. :meth:`take` releases the
    anchor's scan once the history covers its revolution (:class:`~pepin.timeline.ScanGate`);
    :meth:`gather` then carries the other sources' waiting scans to that moment through the
    odometry — the recipe of scratch/camera_only_localization.py — so every source is matched
    around one prediction. With no enabled source fresh there is nothing to release and the
    tracker holds; :meth:`status` says so.
    """

    def __init__(self, registry: SourceRegistry | None = None, max_wait_s: float = 0.5) -> None:
        self.registry = registry if registry is not None else SourceRegistry()
        self._gates = {name: ScanGate(max_wait_s) for name in self.registry.names}
        self._newest: dict[str, TimedScan] = {}
        self.stats = FeedStats()

    def offer(self, name: str, scan: TimedScan) -> None:
        """A scan of ``name`` arrived (enabled or not): its health takes note, it is the
        source's newest picture, and it waits at the source's gate, replacing a waiting one."""
        self.registry.observe(name, scan.stamp)
        self._newest[name] = scan
        self._gates[name].offer(scan)

    def anchor(self, now: float) -> str | None:
        """The source whose scans drive the updates at ``now``: the widest enabled fan while
        it is fresh, else the fresh enabled source heard from last; ``None`` when no enabled
        source is fresh."""
        alive = self.registry.alive(now)
        if not alive:
            return None
        widest = max(
            (self.registry.source(name) for name in self.registry.enabled),
            key=lambda source: source.fov_deg,
        )
        if widest in alive:
            return widest.name
        return min(alive, key=lambda source: self.registry.health(source.name).age_s(now)).name

    def take(self, history: OdomHistory, now: float) -> tuple[str, TimedScan] | None:
        """The anchor's waiting scan, with the anchor's name, once ``history`` covers it;
        ``None`` while nothing is to be released (an expired scan is dropped and counted by
        its gate). The other sources' scans keep waiting for :meth:`gather`."""
        anchor = self.anchor(now)
        if anchor is None:
            return None
        scan = self._gates[anchor].take(history, now)
        return None if scan is None else (anchor, scan)

    def gather(self, anchor: str, stamp: float, history: OdomHistory) -> list[ScanObservation]:
        """The other enabled sources' waiting scans as observations at the anchor's ``stamp``:
        each one's returns moved from the base frame at its own stamp into the base frame at
        the anchor's, through the odometry between the two, and each used once. A scan older
        than its source's ``stale_after_s`` before the stamp is dropped; one the history does
        not cover yet keeps waiting for the next update."""
        out: list[ScanObservation] = []
        at_anchor = history.at(stamp)
        for name in self.registry.enabled:
            scan = self._gates[name].pending
            if name == anchor or scan is None:
                continue
            if stamp - scan.stamp > self.registry.source(name).stale_after_s:
                self._gates[name].drop()
                self.stats.dropped[name] += 1
                continue
            at_scan = history.at(scan.stamp)
            if at_anchor is None or at_scan is None:
                continue
            carry = relative_motion(at_anchor, at_scan)  # the scan's base seen from the anchor's
            out.append(ScanObservation(name, to_map(scan.points, carry), stamp))
            self._gates[name].drop()
            self.stats.attached[name] += 1
        return out

    def picture(self, now: float) -> TimedScan | None:
        """The newest scan, as measured, of the source that drives the tracker at ``now`` —
        what a watch judges the fit on and a whole-map search runs on — or, with nothing
        fresh, the newest scan heard from any enabled source (a standing cart's last picture
        is still true); ``None`` before the first."""
        anchor = self.anchor(now)
        if anchor is not None:
            return self._newest.get(anchor)
        heard = [self._newest[name] for name in self.registry.enabled if name in self._newest]
        return max(heard, key=lambda scan: scan.stamp, default=None)

    def status(self, now: float) -> str:
        """One phrase for the report line: who drives (``anchor lidar``, or ``holding
        map->odom: no fresh source``) and every source's health."""
        anchor = self.anchor(now)
        who = "holding map->odom: no fresh source" if anchor is None else f"anchor {anchor}"
        return f"{who}; {self.registry.report(now)}"

    def report(self) -> FeedStats:
        """The counters since the previous report, which are reset: the enabled sources'
        gates, in roster order."""
        stats, self.stats = self.stats, FeedStats()
        stats.gates = {name: self._gates[name].report() for name in self.registry.enabled}
        for name in set(self._gates) - set(self.registry.enabled):
            self._gates[name].report()  # a source the flag has off: its counters reset too
        return stats
