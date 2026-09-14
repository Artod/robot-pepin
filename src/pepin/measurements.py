"""A pose measured on one machine, fused on another.

The camera's depth and its floor-contact line are produced on the laptop (the network runs on
its GPU). Until 2026-09-13 the scans themselves crossed the link and the Orange Pi matched them
against the map beside every lidar revolution: three matches a scan took the tracker from 45 ms
to 147, it kept every second revolution (4.7 Hz), and the camera's word — measured on a scan
that was by then a fifth of a second old — pulled the live pose 50 cm p90 and 78 cm max off the
lidar's truth over one drive (scratch/drive_bisect.py on run 0238). Offline, at full rate, the
same fusion cost 0.7 cm: the arithmetic was never the problem, the CPU was.

So the matching happens where the data is. The laptop matches each camera scan in a small
window around the board's own belief and sends the RESULT — a place, a covariance read off the
score surface, the fit, the stamp of the scan it was measured on and the map it was measured
against (:class:`RemoteMeasurement`) — and the board carries it to the moment of its next
update over its own odometry and fuses it by information like any other source
(:class:`MeasurementGate`). The board spends a matrix inverse instead of a scan match, and with
the link down it simply has no camera measurements and tracks on the lidar as before.

Nothing here is ROS: one JSON message in, a :class:`pepin.fusion.PoseMeasurement` out.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from pepin.fusion import Matrix, PoseMeasurement, carried, fuse
from pepin.odometry import Pose2D
from pepin.scanmatch import relative_motion
from pepin.selfcheck import SelfCheck
from pepin.sources import CAMERA, GRAPH, SourceRegistry
from pepin.timeline import OdomTrail

__all__ = [
    "MEASUREMENT_MAX_AGE_S",
    "MeasurementGate",
    "MeasurementUpdate",
    "RemoteMeasurement",
    "compose",
    "graph_anchor",
    "graph_measurement",
    "inverse",
    "remote_update",
]

# How old a measurement may be, in seconds, when the update it rides on happens: past this the
# carry is no longer honest and the measurement is dropped. 0.5 s is the roster's own patience
# for a camera source (pepin.sources.DEFAULT_SOURCES' stale_after_s is 1.0 for the fans, and a
# measurement is half a link older than the scan it was made on): at the cart's 0.3 m/s a half
# second is 15 cm of carry over odometry whose own error there is millimetres, and the failure
# this number exists to stop — a measurement made before a bridge stall arriving after it — is
# seconds, not tenths.
MEASUREMENT_MAX_AGE_S = 0.5
# The least a remote source may claim, whatever its peak says. Measured 2026-09-13 with the
# camera recorded but not fused (scratch/camera_error.py on tapes 221822/221909): its word
# against its own band of the volume was 8.5-10.6 cm and 1.6-4.9 deg off the lidar's truth at
# the median, 12-14 cm and 7-9 deg at p90 — while a fan on a wall claimed 1 deg of heading.
# Fused on that claim the pose spun 20 cm and 40 deg per update (22:08). A floor is what the
# measured error says a camera word is worth; a source that scatters more is still inflated
# by its own self-check on top.
REMOTE_FLOOR_XY_M = 0.08
REMOTE_FLOOR_YAW_DEG = 5.0


@dataclass(frozen=True)
class RemoteMeasurement:
    """One machine's answer about where the cart is, ready to travel: the place (map frame),
    how sure it is per direction (a 3x3 covariance over x, y, yaw), which source measured it,
    the stamp of the scan it was measured on, the fit at that place, the map it was measured
    against, and whether the match sat on its window's edge."""

    x: float
    y: float
    yaw: float
    covariance: Matrix
    source: str
    stamp: float
    fit: float
    map_id: str
    edge: bool = False

    @property
    def pose(self) -> Pose2D:
        """The place that was measured."""
        return Pose2D(self.x, self.y, self.yaw)

    def measurement(self) -> PoseMeasurement:
        """This answer as a measurement any fusion takes (:func:`pepin.fusion.fuse`)."""
        return PoseMeasurement(
            self.x,
            self.y,
            self.yaw,
            np.asarray(self.covariance, dtype=float),
            self.source,
            self.stamp,
            self.fit,
            edge=self.edge,
        )

    @classmethod
    def of(cls, measurement: PoseMeasurement, map_id: str) -> RemoteMeasurement:
        """A measurement made here, ready to send: the same numbers plus the map they mean
        something on."""
        return cls(
            x=measurement.x,
            y=measurement.y,
            yaw=measurement.yaw,
            covariance=np.asarray(measurement.covariance, dtype=float),
            source=measurement.source,
            stamp=measurement.stamp,
            fit=measurement.fit,
            map_id=map_id,
            edge=measurement.edge,
        )

    def to_json(self, **extra: Any) -> str:
        """The measurement as one JSON message — everything the receiver needs to judge and
        fuse it, so it never has to join two topics. ``extra`` adds the sender's own notes (what
        the match cost, how old the belief was) for the operator; a reader ignores what it does
        not know."""
        return json.dumps(
            {
                "x": round(self.x, 4),
                "y": round(self.y, 4),
                "yaw": round(self.yaw, 5),
                "covariance": [[round(float(v), 8) for v in row] for row in self.covariance],
                "source": self.source,
                "stamp": self.stamp,
                "fit": round(self.fit, 4),
                "edge": self.edge,
                "map": self.map_id,
                **extra,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> RemoteMeasurement:
        """A measurement back from :meth:`to_json`; ``ValueError``, ``KeyError`` or
        ``TypeError`` for anything else, which is what a subscriber counts as malformed."""
        raw = json.loads(text)
        covariance: NDArray[np.float64] = np.asarray(raw["covariance"], dtype=float)
        if covariance.shape != (3, 3):
            raise ValueError(f"a measurement's covariance is 3x3, not {covariance.shape}")
        return cls(
            x=float(raw["x"]),
            y=float(raw["y"]),
            yaw=float(raw["yaw"]),
            covariance=covariance,
            source=str(raw["source"]),
            stamp=float(raw["stamp"]),
            fit=float(raw["fit"]),
            map_id=str(raw["map"]),
            edge=bool(raw.get("edge", False)),
        )

    def text(self) -> str:
        """``depth (+1.20, -0.80, +60 deg) fit 0.63`` for a report line."""
        return (
            f"{self.source} ({self.x:+.2f}, {self.y:+.2f}, {math.degrees(self.yaw):+.0f} deg) "
            f"fit {self.fit:.2f}" + (", edge" if self.edge else "")
        )


def compose(outer: Pose2D, inner: Pose2D) -> Pose2D:
    """``inner`` expressed in the frame ``outer`` is expressed in: the planar transform
    ``outer`` applied to the planar pose ``inner``."""
    cos, sin = math.cos(outer.theta), math.sin(outer.theta)
    return Pose2D(
        outer.x + cos * inner.x - sin * inner.y,
        outer.y + sin * inner.x + cos * inner.y,
        math.atan2(math.sin(outer.theta + inner.theta), math.cos(outer.theta + inner.theta)),
    )


def inverse(pose: Pose2D) -> Pose2D:
    """The transform the other way: the frame ``pose`` places, seen from the pose itself."""
    cos, sin = math.cos(pose.theta), math.sin(pose.theta)
    return Pose2D(
        -(cos * pose.x + sin * pose.y),
        -(-sin * pose.x + cos * pose.y),
        math.atan2(math.sin(-pose.theta), math.cos(-pose.theta)),
    )


def graph_anchor(tracker: Pose2D, graph: Pose2D) -> Pose2D:
    """The edge from a pose graph's own frame to the lidar map, learned once at the start of a
    session: where the graph's frame sits on the map, so that the graph's answer about the cart
    reads as a place on the map.

    ``tracker`` is where the board's tracker says the cart is (map frame) and ``graph`` is where
    the same cart is according to the graph at that instant (the graph's frame). Returns the
    transform that carries the second onto the first -- ``map <- graph`` -- and it is a constant
    of the session: RTAB-Map refuses an initial pose in mapping mode ("Initial pose can only be
    set in localization mode (Mem/IncrementalMemory=false), ignoring it", librtabmap_core
    0.22.1), so the graph starts at ITS odometry's origin and the offset is learned here instead
    of imposed there.
    """
    return compose(tracker, inverse(graph))


def graph_measurement(
    pose: Pose2D,
    anchor: Pose2D,
    stamp: float,
    map_id: str,
    source: str = GRAPH,
    fit: float = 1.0,
    floor_xy_m: float = REMOTE_FLOOR_XY_M,
    floor_yaw_deg: float = REMOTE_FLOOR_YAW_DEG,
) -> RemoteMeasurement:
    """A pose graph's verdict about where the cart is, as a measurement on the SAME map the
    tracker drives on: the place the graph has the cart at, moved onto the map by the anchor.

    ``pose`` is where the graph says the cart is IN THE GRAPH'S OWN FRAME (RTAB-Map's
    ``map_to_odom`` composed with the filter's ``odom -> base_link``: the graph's opinion, built
    from its own odometry and every closure it has accepted) and ``anchor`` is where that frame
    sits on the lidar map (:func:`graph_anchor`, learned once at the start of the session).
    Composing the two reads the graph's answer back in map coordinates: no second owner of
    ``map -> odom``, one more word in the fusion.

    The covariance is the floor and nothing else. A graph that has never closed a loop reports
    its own pose variance as the whole session's accumulated odometry (497985 m2, a standard
    deviation of 706 m, measured on this stack 2026-09-14), and one that just closed one reports
    the loop link's alone (6.9e-05 m2, 8 mm) -- neither is what the answer is worth. The floor is
    the camera's measured floor for the same reason it exists there: a remote word about a place
    is worth centimetres, whatever its own arithmetic claims.

    ``fit`` is what the word CLAIMS, and it is the graph's confidence in its own recognition
    rather than anything measured against a scan: 1.0 is "the graph has just tied this place to
    one it knows", and it decays with the distance driven since that tie
    (:class:`pepin.graphtrust.GraphTrust`). It is not a discount on the covariance -- the floor
    above is what the word is worth geometrically either way -- but it is what the receiver
    publishes as its confidence when nothing local measured one, and what the whole-map
    candidate channel reads as a score. 1.0 by default: a caller with no trust clock claims what
    this function claimed before there was one.

    Returns the measurement, ready to travel (:meth:`RemoteMeasurement.to_json`).
    """
    place = compose(anchor, pose)
    x, y, yaw = place.x, place.y, place.theta
    covariance: NDArray[np.float64] = np.diag(
        [floor_xy_m**2, floor_xy_m**2, math.radians(floor_yaw_deg) ** 2]
    )
    return RemoteMeasurement(
        x=x,
        y=y,
        yaw=yaw,
        covariance=covariance,
        source=source,
        stamp=stamp,
        fit=float(fit),
        map_id=map_id,
    )


@dataclass(frozen=True)
class MeasurementUpdate:
    """An update the remote measurements drive by themselves — what a dead lidar leaves: the
    moment it happens at (the newest measurement's own stamp, so nothing has to be carried
    forward), the odometry pose there, the word that drove it (one gate's sources fused into
    one measurement) and ``riders`` — the other gates' words carried to the same moment
    (:func:`remote_update`), so several remote sources make one update and not one each."""

    stamp: float
    odom: Pose2D
    measurement: PoseMeasurement
    riders: tuple[PoseMeasurement, ...] = ()

    @property
    def measurements(self) -> list[PoseMeasurement]:
        """Everything this update fuses: the driving word first, then the riders."""
        return [self.measurement, *self.riders]


class MeasurementGate:
    """The receiving side: the newest measurement per source waits here until an update takes
    it, carried to that update's moment over the receiver's own odometry.

    The same shape the feed gives a riding scan (:meth:`pepin.sources.SourceFeed.gather`): the
    measurements of one update all describe ONE instant, or the fusion is between two different
    nows. A measurement the odometry trail no longer covers, or older than
    :attr:`measurement_max_age_s` at the moment of the update, is dropped and counted rather
    than believed — the failure that made this module necessary was a stale camera measurement
    fused as if it were fresh. Every measurement taken is fused into one, named
    :data:`pepin.sources.CAMERA`, so the tracker sees one word from the camera and its
    ``sources`` flag has one name to switch.

    ``sources`` is the tracker's roster: the gate asks it whether the camera is switched on at
    all (nothing is taken while it is off) and tells it every measurement's stamp, so the
    camera's health reads in the report line exactly as a scan source's does. Without one the
    gate is always on, which is what an offline replay wants.

    Pure: the node offers messages and takes measurements; a report line reads the counters.
    """

    def __init__(
        self,
        sources: SourceRegistry | None = None,
        measurement_max_age_s: float = MEASUREMENT_MAX_AGE_S,
        name: str = CAMERA,
        self_check: bool = True,
        remote_floor_xy_m: float = REMOTE_FLOOR_XY_M,
        remote_floor_yaw_deg: float = REMOTE_FLOOR_YAW_DEG,
    ) -> None:
        self.sources = sources
        self.measurement_max_age_s = measurement_max_age_s
        self.remote_floor_xy_m = remote_floor_xy_m  # 0 switches the floor off
        self.remote_floor_yaw_deg = remote_floor_yaw_deg
        # Every remote source vouches for itself here, BEFORE the sources are fused into one
        # word: each is checked against its own previous measurement carried over this
        # machine's odometry, never against another sensor's pose (pepin.selfcheck).
        self.self_check = SelfCheck(enabled=self_check)
        self.name = name  # what the fused measurement is called on the tracker's roster
        self._pending: dict[str, RemoteMeasurement] = {}
        self._counts: dict[str, int] = {}
        self._taken: dict[str, int] = {}
        self._malformed = 0
        self._reason = ""  # the last malformed message's complaint
        self._last: RemoteMeasurement | None = None
        self._age_s = 0.0  # the newest taken measurement's age
        self._rejected: tuple[str, ...] = ()  # sources the last fusion dropped
        self._used: tuple[str, ...] = ()  # ...and the ones it was made of

    switches: ClassVar[tuple[str, ...]] = (
        "measurement_max_age_s",
        "self_check",
        "remote_floor_xy_m",
        "remote_floor_yaw_deg",
    )

    def switch(self, name: str, value: Any) -> None:
        """A live flag by its name (:attr:`switches`); ``ValueError`` for any other name."""
        if name == "self_check":
            self.self_check.enabled = bool(value)
        elif name == "remote_floor_xy_m":
            self.remote_floor_xy_m = float(value)
        elif name == "remote_floor_yaw_deg":
            self.remote_floor_yaw_deg = float(value)
        elif name == "measurement_max_age_s":
            self.measurement_max_age_s = float(value)
        else:
            raise ValueError(f"{name}: not a switch of the measurement gate")

    def malformed(self, reason: str) -> None:
        """A message that was not a measurement arrived; counted, with the complaint kept."""
        self._malformed += 1
        self._reason = reason

    def offer(self, remote: RemoteMeasurement, map_id: str) -> bool:
        """A measurement arrived: it becomes its source's newest, replacing one not yet taken
        (the newest word is the only one worth having). One measured against another map is
        evidence about nothing here — counted as ``elsewhere`` and refused. Returns whether it
        was kept."""
        if remote.map_id != map_id:
            self._count("elsewhere")
            return False
        if remote.source in self._pending:
            self._count("replaced")
        self._pending[remote.source] = remote
        self._last = remote
        self._count("received")
        if self.sources is not None:
            self.sources.observe(self.name, remote.stamp)  # the camera's health, as a scan's
        return True

    @property
    def pending(self) -> tuple[str, ...]:
        """The sources with a measurement waiting, oldest offer first."""
        return tuple(self._pending)

    def pending_stamp(self) -> float | None:
        """The newest waiting measurement's stamp — the moment an update driven by the camera
        alone happens at — or ``None`` when nothing waits."""
        return max((m.stamp for m in self._pending.values()), default=None)

    def enabled(self) -> bool:
        """Whether the tracker's roster has this source switched on (always, without one)."""
        return self.sources is None or self.sources.is_enabled(self.name)

    def _floored(self, measurement: PoseMeasurement) -> PoseMeasurement:
        """The measurement with its covariance raised to the remote floor: a variance under the
        floor's square gets the difference added on its diagonal (a diagonal addition keeps the
        matrix positive definite where a clamp would not). Zero floors change nothing."""
        cov = np.array(measurement.covariance, dtype=float)
        floors = (
            self.remote_floor_xy_m**2,
            self.remote_floor_xy_m**2,
            math.radians(self.remote_floor_yaw_deg) ** 2,
        )
        raised = False
        for i, floor in enumerate(floors):
            if cov[i, i] < floor:
                cov[i, i] += floor - cov[i, i]
                raised = True
        if not raised:
            return measurement
        return PoseMeasurement(
            measurement.x,
            measurement.y,
            measurement.yaw,
            cov,
            measurement.source,
            measurement.stamp,
            measurement.fit,
            rejected=measurement.rejected,
            edge=measurement.edge,
        )

    def take(self, stamp: float, odometry: OdomTrail) -> list[PoseMeasurement]:
        """Every waiting measurement carried to ``stamp`` and fused into one, as a list of one —
        or an empty list when the source is switched off or nothing survives the carry.

        Each is moved from the moment of its own scan to ``stamp`` over the odometry between
        the two (:func:`pepin.fusion.carried`) and then consumed, whatever became of it: one
        that the trail cannot reach (``uncovered``) or that is older than
        :attr:`measurement_max_age_s` (``stale``) is counted and dropped, because a measurement
        the carry cannot honestly move is not evidence about this instant. The survivors are
        fused by their information with the usual disagreement gate, and the result is renamed
        to :attr:`name`: the tracker fuses one camera word with the lidar's, and which sources
        it was made of is in this gate's own report.
        """
        taken: list[PoseMeasurement] = []
        if not self.enabled():
            return []
        at_stamp = odometry.at(stamp)
        for source, remote in list(self._pending.items()):
            del self._pending[source]
            age = stamp - remote.stamp
            if age > self.measurement_max_age_s:
                self._count("stale")
                continue
            at_scan = odometry.at(remote.stamp)
            if at_stamp is None or at_scan is None:
                self._count("uncovered")
                continue
            self._age_s = max(age, 0.0)
            self._taken[source] = self._taken.get(source, 0) + 1
            self._count("taken")
            # Checked at its own moment, against its own previous word (this source's alone),
            # and only then carried to the update: the widening is the source's repeatability,
            # so what the carry adds is never counted as the sensor scattering.
            checked = self.self_check.checked(remote.measurement(), at_scan)
            moved = carried(checked, relative_motion(at_scan, at_stamp), stamp)
            taken.append(self._floored(moved))
        self._used = tuple(m.source for m in taken)
        fused = fuse(taken)
        if fused is None:
            return []
        self._rejected = fused.rejected
        if fused.rejected:
            self._count("disagreed")
        return [
            PoseMeasurement(
                fused.x,
                fused.y,
                fused.yaw,
                fused.covariance,
                self.name,
                stamp,
                fused.fit,
                rejected=fused.rejected,
                edge=all(m.edge for m in taken),
            )
        ]

    def drive(self, anchor: str | None, odometry: OdomTrail) -> MeasurementUpdate | None:
        """The update the camera's measurements drive by themselves, or ``None``.

        What a dead lidar leaves: no scan waits at the feed and none is fresh, so the feed has
        no ``anchor`` and the only word about where the cart is came over the link. The update
        happens at the newest measurement's own stamp — nothing to carry forward, and the
        odometry there is the pose it is predicted from. ``None`` while a scan source is driving
        (the measurement rides ITS next update instead), while the source is switched off,
        with nothing waiting, or when the odometry trail does not reach that moment.
        """
        stamp = None if anchor is not None or not self.enabled() else self.pending_stamp()
        odom = None if stamp is None else odometry.at(stamp)
        if stamp is None or odom is None:
            return None
        taken = self.take(stamp, odometry)
        return MeasurementUpdate(stamp, odom, taken[0]) if taken else None

    def forget(self) -> None:
        """Drop everything waiting and every source's self-check record: what a new map means
        for measurements made on the old one."""
        self._pending.clear()
        self.self_check.forget()

    def status(self) -> dict[str, Any]:
        """The gate as the operator sees it on ``/localization/sources``: which sources the last
        update's camera word was made of, which of them the fusion rejected for disagreeing, how
        old the newest one was when it was taken, and the age the gate refuses past."""
        return {
            "used": list(self._used),
            "rejected": list(self._rejected),
            "age_ms": round(self._age_s * 1e3, 1),
            "max_age_s": self.measurement_max_age_s,
            "self_check": self.self_check.status(),  # per source: [ratio, inflation]
        }

    def report(self) -> str:
        """One phrase for the tracker's report line; the counters are reset."""
        counts, malformed = self._counts, self._malformed
        seen = counts.get("received", 0)
        body = ", ".join(f"{name} {n}" for name, n in sorted(counts.items())) or "none"
        per_source = ", ".join(f"{name} {n}" for name, n in sorted(self._taken.items())) or "none"
        last = "none yet" if self._last is None else self._last.text()
        self._counts, self._taken, self._malformed = {}, {}, 0
        return (
            f"measurements {seen} ({body}), per source: {per_source}, last {last}, "
            f"{self.self_check.text()}, "
            f"malformed {malformed}{f' ({self._reason})' if self._reason else ''}"
        )

    def _count(self, name: str) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1


def remote_update(
    gates: Sequence[MeasurementGate], anchor: str | None, odometry: OdomTrail
) -> MeasurementUpdate | None:
    """The update the remote gates drive between them when no scan does, or ``None``.

    Each gate is one remote NAME on the tracker's roster — the camera's word, the pose graph's —
    and any of them may drive: the freshest waiting word does, at its own stamp, and every other
    gate's waiting word RIDES it, carried to that same moment (:meth:`MeasurementGate.take`).
    One update per call whatever arrived, because two updates in one breath would fuse the same
    odometry twice and count a second step the cart never took.

    Whether the chosen gate may drive at all stays its own decision
    (:meth:`MeasurementGate.drive`): a scan source driving (``anchor`` not ``None``), a source
    the ``sources`` flag has off, or odometry that does not reach the moment all answer ``None``,
    and then nothing is consumed from any gate.
    """
    driver: MeasurementGate | None = None
    newest = -math.inf
    for gate in gates:
        stamp = gate.pending_stamp() if gate.enabled() else None
        if stamp is not None and stamp > newest:
            newest, driver = stamp, gate
    if driver is None:
        return None
    plan = driver.drive(anchor, odometry)
    if plan is None:
        return None
    riders = [m for gate in gates if gate is not driver for m in gate.take(plan.stamp, odometry)]
    return MeasurementUpdate(plan.stamp, plan.odom, plan.measurement, tuple(riders))
