"""Which sensors feed the tracker, and which of them are alive right now.

The tracker matches whatever scan-shaped evidence arrives against the same map: the lidar's
revolution, the camera's virtual scan (:func:`pepin.depth.depth_to_scan`) and the floor-contact
scan (:func:`pepin.contact.contact_scan`). This module is the roster: what each source is (its
frame, its field of view, how far it is trusted, when it counts as stale), the ``sources``
flag that says which of them are enabled, and a health record per source so a report line can
say "lidar fresh 9.9 Hz, depth stale 2.1 s, contact off". No ROS here: the node feeds stamps
in and reads verdicts out.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

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
