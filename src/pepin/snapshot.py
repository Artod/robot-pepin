"""Which sensors are alive right now, and which of their messages belong in ONE snapshot.

A mapper does not want three synchronised topics; it wants one picture of the room per moment,
made of whatever was looking. This module is that arithmetic, with no ROS in it: every source
offers the stamps (and the payloads) it delivers, and :meth:`SnapshotPacker.plan` answers with
one :class:`Snapshot` — who drove it, at which stamp, which other sources joined and which were
silent. :mod:`pepin_bringup.sensor_pack` is the one caller; the shape of the message it builds
out of the answer is its business, not this module's.

THE CLOCK. There is none here. Every stamp is the SENSOR's own (on this robot: the board's,
carried over the bridge), and the laptop's wall clock is not comparable with it — so nothing in
this module ever asks what time it is. The only "now" is the newest stamp any enabled source has
delivered, which is the one moment the data itself names, and every judgement is made against it.

ALIVE, BY THE SOURCE'S OWN PERIOD. Each source measures its own period from the stamps it
delivers (:class:`Cadence`, the running mean of :class:`pepin.sources.SourceHealth` under another
name) and is alive while its newest stamp is within :data:`LIVE_PERIODS` of that period of the
newest stamp anybody has. Nothing is a chosen timeout: the multiple is the worst arrival lag this
stack was measured to have against the period of the source that showed it — the depth's 497 ms
worst against its 118 ms period, 4.2 periods, so five missed messages is a source that has
stopped rather than one that stumbled (scratch/vo_stamp_pairs.py, 2026-09-14). A source that has
delivered one message has no period yet and is therefore neither alive nor pairable: it takes two
messages to have one interval.

THE PAIRING RULE. The DRIVER is the alive enabled source whose newest stamp is the OLDEST — the
latest moment EVERY alive source has already spoken for. Not the newest stamp, because the
sources do not arrive with the same delay: the camera's frame lands on this machine about 160 ms
after the moment it is stamped with (the network's 0.2-0.3 s, then the depth 77 ms behind the
picture) while a scan lands about 35 ms after its own, so "the newest stamp drives" would always
pick the lidar and hand the camera the whole 130 ms of difference. Driving on the slowest-arriving
source instead costs nobody anything: by the time its frame is here, the scans on both sides of
its moment are here too. A source that has stopped drops out of the choice by itself, because its
stamps stop advancing while everybody else's do not — which is what makes a dead camera hand the
snapshots to the lidar with no restart and no decision.

The snapshot's stamp is the driver's stamp: a board-clock stamp of a real message, never an
average and never a local now. Every other alive source joins with the ONE message of its own
NEAREST that stamp, and only while that message is within :data:`PAIR_PERIODS` of its own period.

THE BOUND, and it is derived, not chosen. A source delivering with period T has a message within
T/2 of any instant, so the nearest message of a live source is at most half its own period from
the driver's stamp; one lost message doubles the far side, which puts "live, having missed one" at
1.5 T. Past that, two messages in a row are missing. Measured on this stack
(scratch/pairing_bound.py over the rates of scratch/vo_stamp_pairs.py): the lidar at 9.9 Hz gives
patience 152 ms and a healthy offset of at most 51 ms, the camera at 8.5 Hz gives 176 ms and
59 ms. At the cart's 0.2 m/s and the 17 deg/s of the 2026-09-11 turn, 51 ms of offset places the
joined member 1.0 cm and 0.86 deg from where the driver stood: a fifth of the 5 cm cell the grid
is built on and a third of a link's linear sigma, but one and a half times a link's ANGULAR sigma
(3.2 cm and 0.57 deg, vslam.launch.py's TF_ODOMETRY_VARIANCE). Small against the map, not small
against the graph while the cart turns — which is why a member is the message NEAREST the
driver's stamp and not simply that source's newest.

NOTHING IS PUBLISHED TWICE. :meth:`SnapshotPacker.plan` answers ``None`` while no alive source has
a stamp at least ``min_gap_s`` newer than the last snapshot it answered with, so a stack where
every sensor has gone quiet produces no snapshots at all instead of republishing the last one for
ever, and a caller that asks on every arriving message still gets one snapshot per ``min_gap_s``
of SENSOR time.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pepin.depth import SCAN_WINDOW_S
from pepin.sources import RATE_TAU_S

# How many of its own measured periods a source's message may be from the driver's stamp and
# still be paired with it. 0.5 is where the nearest message of a source running at period T sits
# in the worst case; +1.0 is one message lost. Past 1.5 T two messages in a row are missing,
# which is a source that has stopped, not one that is jittering.
PAIR_PERIODS = 1.5
NEAREST_PERIODS = 0.5  # the healthy half of the same arithmetic, for the report line's bound
# How many of its own periods a source may be behind the newest stamp anybody has and still count
# as delivering. The depth's worst measured arrival lag is 497 ms against its 118 ms period
# (scratch/vo_stamp_pairs.py, 120 s with the whole stack up, 2026-09-14) — 4.2 periods — so a
# source five of its own messages behind has stopped rather than stumbled.
LIVE_PERIODS = 5.0
MEASURED_STAMPS = 2  # stamps a source must have delivered before its period is a measurement


class Cadence:
    """One source's own period, measured from the stamps it delivers, and the two patiences that
    follow from it: how far behind the newest moment it may be and still be delivering
    (:attr:`liveness_s`), and how far from a moment its nearest message may sit and still be
    paired with it (:attr:`patience_s`).

    The period is a running mean over the last seconds — the recipe and the time constant of
    :class:`pepin.sources.SourceHealth`, which measures the same thing as a rate — so a source
    that slows down is believed within a couple of seconds instead of after a whole window.
    """

    def __init__(
        self,
        tau_s: float = RATE_TAU_S,
        pair_periods: float = PAIR_PERIODS,
        live_periods: float = LIVE_PERIODS,
    ) -> None:
        self._tau_s = tau_s
        self._pair_periods = pair_periods
        self._live_periods = live_periods
        self.last_stamp: float | None = None
        self.count = 0
        self._period_s = 0.0

    def observe(self, stamp: float) -> None:
        """A message of this source arrived carrying ``stamp`` (seconds, the sensor's clock)."""
        if self.last_stamp is not None and stamp > self.last_stamp:
            dt = stamp - self.last_stamp
            alpha = 1.0 - math.exp(-dt / self._tau_s) if self.count > 1 else 1.0
            self._period_s += alpha * (dt - self._period_s)
        self.last_stamp = stamp if self.last_stamp is None else max(self.last_stamp, stamp)
        self.count += 1

    @property
    def measured(self) -> bool:
        """Whether the period is a measurement yet: two stamps make one interval."""
        return self.count >= MEASURED_STAMPS and self._period_s > 0.0

    @property
    def period_s(self) -> float | None:
        """Seconds between messages as measured, or ``None`` before the first interval."""
        return self._period_s if self.measured else None

    @property
    def rate_hz(self) -> float:
        """The measured period as a rate, 0.0 while there is none."""
        return 1.0 / self._period_s if self.measured else 0.0

    @property
    def patience_s(self) -> float | None:
        """How far from a moment this source's nearest message may sit and still be paired with
        it (:data:`PAIR_PERIODS` of the measured period), or ``None`` while unmeasured."""
        return self._period_s * self._pair_periods if self.measured else None

    @property
    def liveness_s(self) -> float | None:
        """How far behind the newest moment this source may be and still count as delivering
        (:data:`LIVE_PERIODS` of the measured period), or ``None`` while unmeasured."""
        return self._period_s * self._live_periods if self.measured else None

    @property
    def pair_periods(self) -> float:
        """The multiple of the period :attr:`patience_s` is built from."""
        return self._pair_periods

    @pair_periods.setter
    def pair_periods(self, value: float) -> None:
        self._pair_periods = float(value)

    def gap_s(self, stamp: float) -> float:
        """How far the newest message is from ``stamp``; ``inf`` before the first message."""
        return math.inf if self.last_stamp is None else abs(stamp - self.last_stamp)

    def alive(self, stamp: float) -> bool:
        """Whether this source is delivering as of the moment ``stamp`` (the newest stamp anybody
        has): it has a measured period and is within :attr:`liveness_s` of it."""
        liveness = self.liveness_s
        return liveness is not None and self.gap_s(stamp) <= liveness

    def verdict(self, stamp: float) -> str:
        """This source at the moment ``stamp``: ``absent`` (never heard from), ``unmeasured``
        (one message, so no period and no patience), ``fresh`` or ``silent``."""
        if self.last_stamp is None:
            return "absent"
        if self.liveness_s is None:
            return "unmeasured"
        return "fresh" if self.alive(stamp) else "silent"

    def text(self, stamp: float) -> str:
        """One phrase for a report line: ``fresh 9.9 Hz (period 101 ms, pairs within 152 ms)``,
        ``silent 2.1 s``, ``unmeasured (1 message)`` or ``absent``."""
        verdict = self.verdict(stamp)
        if verdict == "fresh":
            return (
                f"fresh {self.rate_hz:.1f} Hz (period {self._period_s * 1e3:.0f} ms,"
                f" pairs within {self._period_s * self._pair_periods * 1e3:.0f} ms)"
            )
        if verdict == "silent":
            return f"silent {self.gap_s(stamp):.1f} s"
        if verdict == "unmeasured":
            return f"unmeasured ({self.count} message{'' if self.count == 1 else 's'})"
        return verdict


class StampRing[T]:
    """The recent messages of one source, by stamp: everything within ``window_s`` of the newest.

    A window and not a count, because the window is set by how late a message may ARRIVE while
    its stamp is still worth pairing: the camera's frame reaches this machine about a fifth of a
    second after the moment it is stamped with, so the scans around that moment must still be
    here when it lands. That is the same window :data:`pepin.depth.SCAN_WINDOW_S` was measured
    for, and it is the same number."""

    def __init__(self, window_s: float = SCAN_WINDOW_S) -> None:
        self._window_s = window_s
        self._items: deque[tuple[float, T]] = deque()
        self._newest = -math.inf  # the largest stamp ever offered: the window hangs off it

    def offer(self, stamp: float, item: T) -> None:
        """Keep ``item`` under ``stamp`` and drop whatever fell out of the window. The window
        hangs off the largest stamp ever offered, so a message that arrives out of order cannot
        empty the ring."""
        self._items.append((stamp, item))
        self._newest = max(self._newest, stamp)
        self._items = deque(
            entry for entry in self._items if self._newest - entry[0] <= self._window_s
        )

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> tuple[tuple[float, T], ...]:
        """Everything in the window, oldest first: for a caller that matches on the payload
        itself (an image and its depth share a stamp bit for bit)."""
        return tuple(self._items)

    def newest(self) -> tuple[float, T] | None:
        """The message with the largest stamp, or ``None`` while the ring is empty."""
        return max(self._items, key=lambda entry: entry[0], default=None)

    def nearest(self, stamp: float) -> tuple[float, T] | None:
        """The message whose stamp is closest to ``stamp`` (the older one on a tie, so the
        answer does not depend on the order messages were offered in), or ``None``."""
        return min(self._items, key=lambda entry: (abs(entry[0] - stamp), entry[0]), default=None)


@dataclass(frozen=True)
class Snapshot[T]:
    """One moment of the room: which source's message named it (``driver``), the stamp of that
    message on the sensor's own clock, and every source that is in it with the stamp and payload
    of its own message nearest that moment (``members``, the driver included). ``silent`` names
    the enabled sources that had nothing near enough, in roster order — what the report line
    calls a lidar-only or a camera-only snapshot."""

    driver: str
    stamp: float
    members: dict[str, tuple[float, T]]
    silent: tuple[str, ...]

    @property
    def kind(self) -> str:
        """What went in: ``<name>-only`` for a single member, ``full`` when every enabled source
        is in, else the members joined by ``+`` (a roster of three may lose one of them)."""
        names = tuple(self.members)
        if len(names) == 1:
            return f"{names[0]}-only"
        return "full" if not self.silent else "+".join(names)

    def offset_s(self, name: str) -> float:
        """How far the member ``name`` sits from the snapshot's stamp; ``inf`` when it is not in
        this snapshot at all."""
        entry = self.members.get(name)
        return math.inf if entry is None else abs(entry[0] - self.stamp)


class SnapshotPacker[T]:
    """Every source's recent messages, and one snapshot of whatever is alive when asked.

    The roster's order is the report line's order and the tie-break for equal stamps; which of
    the roster may enter a snapshot is :meth:`enable` (the node's ``sources`` flag). The packer
    keeps no clock and no timer: it answers when asked, and answers ``None`` when nothing has
    happened since the last answer.
    """

    def __init__(
        self,
        names: Sequence[str],
        *,
        window_s: float = SCAN_WINDOW_S,
        pair_periods: float = PAIR_PERIODS,
    ) -> None:
        self._names = tuple(names)
        self._rings = {name: StampRing[T](window_s) for name in self._names}
        self._cadence = {name: Cadence(pair_periods=pair_periods) for name in self._names}
        self._enabled: tuple[str, ...] = self._names
        self._last_stamp: float | None = None  # the stamp of the snapshot last answered with

    @property
    def names(self) -> tuple[str, ...]:
        """The roster, in the order it was declared."""
        return self._names

    @property
    def enabled(self) -> tuple[str, ...]:
        """The sources that may enter a snapshot, in roster order."""
        return self._enabled

    @property
    def last_stamp(self) -> float | None:
        """The stamp of the last snapshot answered with, or ``None`` before the first."""
        return self._last_stamp

    def enable(self, names: Iterable[str]) -> None:
        """Exactly these sources may enter a snapshot from now on; an unknown name is refused
        (``ValueError``) so a typo in ``ros2 param set`` cannot silently blind the packer."""
        wanted = set(names)
        unknown = wanted - set(self._names)
        if unknown:
            raise ValueError(
                f"unknown sources {sorted(unknown)}; the roster is {list(self._names)}"
            )
        self._enabled = tuple(name for name in self._names if name in wanted)

    def cadence(self, name: str) -> Cadence:
        """One source's measured period and its two patiences."""
        return self._cadence[name]

    def set_pair_periods(self, value: float) -> None:
        """Move every source's pairing patience to this multiple of its own measured period."""
        for cadence in self._cadence.values():
            cadence.pair_periods = value

    def offer(self, name: str, stamp: float, item: T) -> None:
        """A message of ``name`` arrived with ``stamp`` (its own clock): it joins the source's
        ring and its cadence takes note. Offered whether the source is enabled or not, so the
        report line can say that a muted sensor is still delivering."""
        self._cadence[name].observe(stamp)
        self._rings[name].offer(stamp, item)

    def now_s(self) -> float | None:
        """The newest stamp any ENABLED source has delivered — the only "now" there is here —
        or ``None`` before the first message."""
        stamps = [
            cadence.last_stamp
            for name, cadence in self._cadence.items()
            if name in self._enabled and cadence.last_stamp is not None
        ]
        return max(stamps) if stamps else None

    def plan(self, min_gap_s: float = 0.0) -> Snapshot[T] | None:
        """One snapshot of what is alive, or ``None`` when no alive source has a stamp at least
        ``min_gap_s`` newer than the last snapshot's (which is also the answer when every sensor
        has gone quiet, and while no source has a measured period yet)."""
        now = self.now_s()
        if now is None:
            return None
        alive = [
            (name, newest)
            for name in self._enabled
            if self._cadence[name].alive(now) and (newest := self._rings[name].newest()) is not None
        ]
        if not alive:
            return None
        # The OLDEST newest-stamp drives: the latest moment every alive source has spoken for.
        # The roster's order breaks a tie, so the answer does not depend on arrival order.
        order = {name: index for index, name in enumerate(self._names)}
        name, (stamp, item) = min(alive, key=lambda entry: (entry[1][0], order[entry[0]]))
        if self._last_stamp is not None and stamp - self._last_stamp < min_gap_s:
            return None
        if self._last_stamp is not None and stamp <= self._last_stamp:
            return None
        members: dict[str, tuple[float, T]] = {name: (stamp, item)}
        silent: list[str] = []
        for other in self._enabled:
            if other == name:
                continue
            entry = self._rings[other].nearest(stamp)
            patience = self._cadence[other].patience_s
            if entry is None or patience is None or abs(entry[0] - stamp) > patience:
                silent.append(other)
            else:
                members[other] = entry
        self._last_stamp = stamp
        return Snapshot(
            driver=name,
            stamp=stamp,
            members={key: members[key] for key in self._names if key in members},
            silent=tuple(silent),
        )

    def report(self, stamp: float | None = None) -> str:
        """Every source's cadence at the newest moment the data names: ``camera fresh 8.5 Hz
        (period 118 ms, pairs within 176 ms), lidar off``."""
        at = self.now_s() if stamp is None else stamp
        if at is None:
            return ", ".join(f"{name} no stamp yet" for name in self._names)
        return ", ".join(
            f"{name} {self._cadence[name].text(at) if name in self._enabled else 'off'}"
            for name in self._names
        )
