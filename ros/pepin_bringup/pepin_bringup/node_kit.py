"""What every node here repeated, written once: the pieces a node is made of besides its job.

A node on this robot is a subscription or two, a thread that works on the newest message and
drops the backlog (:class:`Worker`), counters and stage timings that a 30 s timer turns into
one report line (:class:`Tally`), feature flags declared once in the module's ``FLAGS`` table
(:mod:`pepin.flags`), flipped with ``ros2 param set`` and printed in that line
(:class:`Switches`, CLAUDE.md rule 19), transforms looked up at a stamp with the
failure told apart by kind (:class:`TfLookup`, and as the pose history a
:class:`pepin.frame_pose.FramePoser` reads, :class:`TfHistory`), and a ``main`` that leaves
DDS properly on SIGINT (:func:`spin_main`). Each of these was copied between the depth node,
the fusion node and the tracker with small differences — and one of the differences was a
crash: a kicked depth node died with SIGABRT because its worker thread, a daemon, was inside
the network when the interpreter shut down (see :func:`spin_main`). Nothing here knows what a
node does; a node composes these and keeps its own logic.
"""

from __future__ import annotations

import faulthandler
import os
import signal
import sys
import threading
import time
import traceback
import weakref
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rcl_interfaces.msg import (
    FloatingPointRange,
    IntegerRange,
    ParameterDescriptor,
    ParameterType,
    SetParametersResult,
)
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu
from tf2_ros import Buffer, TransformListener

from pepin.deployment import bridged_qos
from pepin.depth import UP_LEVEL, Array
from pepin.flags import Flag, FlagSet
from pepin.lean import Lean, LeanEstimator, LevelPose
from pepin.mounts import Mounts, load_camera_mounts
from pepin.static_facts import StaticWait
from pepin.telemetry import LatencySummary, LatencyTracker
from pepin.tsdf import RigidPose
from pepin_bringup.msgs import (
    imu_arrays,
    pose_from_transform,
    stamp_from_seconds,
    stamp_seconds,
)

BASE_FRAME = "base_link"  # the frame the C++ bridge publishes its IMU readings in

__all__ = [
    "Fatal",
    "LeanFeed",
    "Switches",
    "Tally",
    "TfHistory",
    "TfLookup",
    "Window",
    "Worker",
    "descriptor",
    "spin_main",
    "stamp_seconds",
    "tf_failure_kind",
]

STOP_PATIENCE_S = 5.0  # a worker gets this long to finish its item on the way out


# ---- the newest-item-wins worker -------------------------------------------------------------
class Worker[T]:
    """One thread that works on the newest item offered and drops the rest.

    A camera frame or a depth pair that arrives while the previous one is still being worked
    replaces the one waiting (the model wants the latest view, not a backlog); ``offer`` says
    when that happened so the node can count it. An exception in the work is reported through
    ``on_error`` with its traceback and the thread goes on: a raise must not kill the worker
    silently. ``stop`` ends the thread and waits for its current item, which is what keeps the
    process from aborting at exit (:func:`spin_main`).
    """

    def __init__(
        self, work: Callable[[T], None], *, name: str, on_error: Callable[[str], None]
    ) -> None:
        self._work = work
        self._on_error = on_error
        self._pending: T | None = None
        self._wake = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> Worker[T]:
        """Start the thread; returns self so it chains."""
        self._thread.start()
        return self

    def offer(self, item: T) -> bool:
        """Hand the worker its next item; True when an item still waiting was replaced."""
        with self._wake:
            dropped = self._pending is not None
            self._pending = item
            self._wake.notify()
        return dropped

    @property
    def waiting(self) -> bool:
        """Whether an item is queued that ``offer`` would replace: for a caller with two kinds
        of work on one thread, where dropping the other kind's item is the harm."""
        with self._wake:
            return self._pending is not None

    def clear(self) -> None:
        """Forget the item waiting, if any (a reset)."""
        with self._wake:
            self._pending = None

    def stop(self, timeout_s: float = STOP_PATIENCE_S) -> bool:
        """Ask the thread to end and wait for it: True when it did within ``timeout_s``."""
        with self._wake:
            self._stop.set()
            self._wake.notify()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout_s)
        return not self._thread.is_alive()

    @property
    def alive(self) -> bool:
        """Whether the thread is running."""
        return self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._wake:
                while self._pending is None and not self._stop.is_set():
                    self._wake.wait()
                if self._stop.is_set():
                    return
                item, self._pending = self._pending, None
            if item is None:  # a clear() raced the wake-up
                continue
            try:
                self._work(item)
            except Exception:
                if not self._stop.is_set():  # a failure while leaving is not news
                    self._on_error(traceback.format_exc())


# ---- the report period -----------------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    """One report period as taken from a :class:`Tally`: how long it was, the counts and the
    seconds by name, each stage's latency summary, the samples and the notes collected."""

    elapsed_s: float
    counts: Counter[str]
    seconds: dict[str, float]
    timing: dict[str, LatencySummary]
    samples: dict[str, list[float]]
    notes: dict[str, str]

    def rate(self, name: str) -> float:
        """``counts[name]`` per second of the period."""
        return self.counts[name] / max(self.elapsed_s, 1e-6)

    def ms_per(self, stage: str, per: str) -> float:
        """Milliseconds spent in ``stage`` per event counted under ``per`` (at least one)."""
        return self.seconds.get(stage, 0.0) / max(self.counts[per], 1) * 1e3

    def stages(self) -> str:
        """Every stage as ``name median/max`` milliseconds, in the order they were declared."""
        return " ".join(
            f"{name} {s.median_ms:.0f}/{s.max_ms:.0f}" for name, s in self.timing.items()
        )


class Tally:
    """Counts, seconds, stage timings, samples and notes of the current report period, under
    one lock: the worker fills them on its thread, the report timer empties them on the
    executor's, and a swap racing an increment lost counts (or worse). ``stages`` names the
    timings the report lists even when a period saw none."""

    def __init__(self, stages: Iterable[str] = ()) -> None:
        self._stages = tuple(stages)
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()
        self._seconds: defaultdict[str, float] = defaultdict(float)
        self._timing: dict[str, LatencyTracker] = {}
        self._samples: defaultdict[str, list[float]] = defaultdict(list)
        self._notes: dict[str, str] = {}
        self._since = time.monotonic()
        self._reset()

    def _reset(self) -> None:
        self._counts = Counter()
        self._seconds = defaultdict(float)
        self._timing = {name: LatencyTracker(name) for name in self._stages}
        self._samples = defaultdict(list)
        self._notes = {}

    def count(self, name: str, n: int = 1) -> None:
        """``name`` happened ``n`` more times."""
        with self._lock:
            self._counts[name] += n

    def spent(self, stage: str, seconds: float) -> None:
        """``seconds`` more went into ``stage`` (its sum and its latency window)."""
        with self._lock:
            self._seconds[stage] += seconds
            self._timing.setdefault(stage, LatencyTracker(stage)).add(seconds)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Time the enclosed block into ``stage``, exception or not."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.spent(stage, time.perf_counter() - start)

    def sample(self, name: str, value: float) -> None:
        """One more value of ``name`` for the period's median and max."""
        with self._lock:
            self._samples[name].append(float(value))

    def note(self, kind: str, text: str) -> None:
        """The last text seen for ``kind`` (a failure's message, say)."""
        with self._lock:
            self._notes[kind] = text

    def take(self) -> Window:
        """The period so far, and a fresh one starts."""
        with self._lock:
            now = time.monotonic()
            window = Window(
                elapsed_s=now - self._since,
                counts=self._counts,
                seconds=dict(self._seconds),
                timing={name: tracker.summary() for name, tracker in self._timing.items()},
                samples=dict(self._samples),
                notes=self._notes,
            )
            self._since = now
            self._reset()
        return window


# ---- live switches ---------------------------------------------------------------------------
_WIRE_TYPES = {
    "bool": ParameterType.PARAMETER_BOOL,
    "integer": ParameterType.PARAMETER_INTEGER,
    "double": ParameterType.PARAMETER_DOUBLE,
    "string": ParameterType.PARAMETER_STRING,
}


def descriptor(flag: Flag) -> ParameterDescriptor:
    """The parameter descriptor of ``flag``: its wire type, its help (the description with the
    choices, the range, the overriding variable, whether it is live) and, for a number with a
    range, the integer or floating-point range ``ros2 param describe`` prints and rclpy checks."""
    d = ParameterDescriptor(
        name=flag.name, type=_WIRE_TYPES[flag.wire_type], description=flag.help()
    )
    if flag.kind == "number" and flag.range is not None:
        lo, hi = flag.range
        if flag.integer:
            d.integer_range = [IntegerRange(from_value=int(lo), to_value=int(hi), step=0)]
        else:
            d.floating_point_range = [
                FloatingPointRange(from_value=float(lo), to_value=float(hi), step=0.0)
            ]
    return d


class Switches:
    """A node's feature flags as its live ROS parameters: the table (:class:`pepin.flags.FlagSet`,
    the module's ``FLAGS``) declared with descriptors — type, help, the range of a number — read
    back through a launch override or the flag's ``env`` variable, changed with ``ros2 param
    set`` (ros/flags.sh) while the node runs and printed in its report line by :meth:`state`
    (CLAUDE.md rule 19). A set is refused with the reason when the name is not in the table
    (rclpy runs this callback on declarations too: create the switches after the node's last
    ``declare_parameter``), when the flag is not live, when the value is not one of the flag's,
    or when ``on_change(name, old, new)`` raises ``ValueError``; a refused batch leaves every
    flag as it was. The table is copied: the module's ``FLAGS`` keeps its defaults."""

    def __init__(
        self,
        node: Any,
        flags: FlagSet,
        *,
        on_change: Callable[[str, Any, Any], None] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._node = node
        self._log = node.get_logger()
        self._on_change = on_change
        self.flags = FlagSet(*flags)
        defaults = self.flags.defaults(os.environ if environ is None else environ)
        for flag in self.flags:
            declared = node.declare_parameter(
                flag.name, flag.wire(defaults[flag.name]), descriptor(flag)
            ).value
            self.flags.set(flag.name, declared)  # a launch override, checked like any change
        node.add_on_set_parameters_callback(self._on_set)

    def __getitem__(self, name: str) -> Any:
        return self.flags[name]

    def on(self, name: str) -> bool:
        """Whether switch ``name`` is on."""
        return self.flags.on(name)

    def state(self, live_only: bool = True) -> str:
        """The live flags' values for the report line: ``floor_anchor=on depth_backend=auto``;
        ``live_only=False`` adds the flags that are read at start, for a node whose report is
        not readable without them (which side broadcasts a static transform, say)."""
        return self.flags.state(live_only)

    def set(self, name: str, value: Any) -> Any:
        """Change ``name`` from inside the node (a switch it turns off itself, say): through the
        parameter server, so ``ros2 param get`` agrees and the change is logged like one from
        outside; returns the old value, ``ValueError`` with the reason changes nothing."""
        flag = self.flags.flag(name)
        old = self.flags[name]
        result = self._node.set_parameters([Parameter(name, value=flag.wire(flag.parse(value)))])[0]
        if not result.successful:
            raise ValueError(result.reason)
        return old

    def _apply(self, name: str, value: Any) -> Any:
        """One change into the table and to ``on_change``; the old value back on a refusal."""
        old = self.flags.set(name, value)
        new = self.flags[name]
        if self._on_change is not None:
            try:
                self._on_change(name, old, new)
            except ValueError:
                self.flags.set(name, old)
                raise
        self._log.info(f"{name}={self.flags.flag(name).render(new)}")
        return old

    def _on_set(self, params: list[Any]) -> SetParametersResult:
        unknown = [p.name for p in params if p.name not in self.flags]
        if unknown:
            return SetParametersResult(
                successful=False,
                reason=f"{', '.join(unknown)}: not a flag of this node; the flags are"
                f" {', '.join(self.flags.names)}",
            )
        stale = [p.name for p in params if not self.flags.flag(p.name).live]
        if stale:
            return SetParametersResult(
                successful=False, reason=f"{', '.join(stale)}: not live, set at the next start"
            )
        applied: list[tuple[str, Any]] = []
        for p in params:
            try:
                applied.append((p.name, self._apply(p.name, p.value)))
            except ValueError as exc:
                for name, old in reversed(applied):  # the batch is one change or none
                    self._apply(name, old)
                return SetParametersResult(successful=False, reason=str(exc))
        return SetParametersResult(successful=True)


# ---- transforms ------------------------------------------------------------------------------
def tf_failure_kind(exc: BaseException) -> str:
    """Why a lookup failed, as tf2 names it without the suffix: ``Lookup`` (no such chain),
    ``Extrapolation`` (not for that time), ``Connectivity``, ``Timeout``; ``Unknown`` else."""
    return type(exc).__name__.removesuffix("Exception") or "Unknown"


# WHERE A TfLookup's LISTENER LIVES. tf2_ros's TransformListener(buffer, node, spin_thread=True)
# puts the /tf subscription on OUR node, in a reentrant group, and adds OUR WHOLE NODE to its own
# executor (transform_listener.py:96-114): every /tf message (tens a second: the EKF, RTAB-Map's
# map -> odom, the neck) then wakes both executors, and the main one rebuilds its wait set in
# Python each time — goal_server burnt 43-46 % of an A53 core at rest, 14 of 30 stack samples in
# rclpy's wait-set and take machinery and none in its own code (2026-09-24,
# scratch/link_autopsy/goal_server_profile.sh). It also ran our node's callbacks on the
# listener's thread before its constructor had finished (depth_fusion, 2026-09-18). "own" (the
# default) hands tf2_ros node=None: the listener makes a node of its own and spins only that;
# PEPIN_TF_LISTENER_NODE=shared is the way back.
TF_LISTENER_NODE = os.environ.get("PEPIN_TF_LISTENER_NODE", "own")

# Every TfLookup whose listener thread is still running. tf2_ros starts that thread NON-daemon
# (transform_listener.py:114, ``Thread(target=run_func)``), so a node that never closes its lookup
# leaves the interpreter waiting on it forever after SIGINT: on 2026-09-24 `ros/laptop.sh kick
# rtabmap_frame` ended the spin (the report lines stopped) and the process never exited until a
# SIGTERM — and five nodes carry a TfLookup with no close() of their own. spin_main closes what
# is still open here, after the node's own close().
_OPEN_LOOKUPS: weakref.WeakSet[TfLookup] = weakref.WeakSet()


class TfLookup:
    """A TF buffer, fed by a listener on its own thread unless a buffer is handed in, and
    lookups that answer ``None`` instead of raising: the failure goes to ``on_failure`` as
    ``(kind, text)`` (see :func:`tf_failure_kind`) for a counter and the report line."""

    def __init__(
        self,
        node: Any,
        *,
        buffer: Any | None = None,
        on_failure: Callable[[str, str], None] | None = None,
    ) -> None:
        self.buffer = buffer if buffer is not None else Buffer()
        listener_node = None if TF_LISTENER_NODE == "own" else node
        self._owns_listener_node = listener_node is None
        self._listener = (
            None
            if buffer is not None
            else TransformListener(self.buffer, listener_node, spin_thread=True)
        )
        self._on_failure = on_failure
        self.static_wait: StaticWait | None = None  # set by expect_static
        if self._listener is not None:
            _OPEN_LOOKUPS.add(self)

    def transform(
        self, target: str, source: str, stamp: Any = None, timeout_s: float = 0.0
    ) -> Any | None:
        """``target <- source`` at ``stamp`` (a stamp message; ``None`` is the latest) as the
        ``TransformStamped`` itself, waiting up to ``timeout_s``; ``None`` when it is not
        there. For a caller that wants the message (a mount read by
        :func:`pepin_bringup.msgs.planar_mount`); :meth:`pose` is the usual one."""
        return self._call(
            lambda: self.buffer.lookup_transform(
                target, source, _time(stamp), timeout=Duration(seconds=timeout_s)
            ),
            f"{target}<-{source}",
        )

    def pose(
        self, target: str, source: str, stamp: Any = None, timeout_s: float = 0.0
    ) -> RigidPose | None:
        """``target <- source`` at ``stamp`` (a stamp message; ``None`` is the latest), waiting
        up to ``timeout_s`` for it; ``None`` when it is not there."""
        transform = self.transform(target, source, stamp, timeout_s)
        return None if transform is None else pose_from_transform(transform)

    def motion(
        self, frame: str, from_stamp: Any, to_stamp: Any, fixed: str, timeout_s: float = 0.0
    ) -> RigidPose | None:
        """How ``frame`` moved between two stamps, seen through ``fixed`` (``odom`` for the
        cart's own motion): the transform that carries a point of ``frame`` at ``from_stamp``
        to ``frame`` at ``to_stamp``; ``None`` when the history does not cover it."""
        transform = self._call(
            lambda: self.buffer.lookup_transform_full(
                frame,
                _time(to_stamp),
                frame,
                _time(from_stamp),
                fixed,
                timeout=Duration(seconds=timeout_s),
            ),
            f"{frame} {fixed}",
        )
        return None if transform is None else pose_from_transform(transform)

    def expect_static(
        self, node: Any, edges: Sequence[tuple[str, str]], after_s: Callable[[], float]
    ) -> StaticWait:
        """WARN on ``node``'s logger when these ``/tf_static`` edges are not all in the buffer
        ``after_s()`` seconds (a live knob, ``tf_static_wait_s``) from now, naming the missing
        ones, the usual cause and the debug recipe (:mod:`pepin.static_facts`); looked at once a
        second until the last lands, which is logged once at INFO."""
        wait = StaticWait(tuple(edges), started=time.monotonic(), after_s=after_s())
        timer: Any = None

        def tick() -> None:
            out = wait.update(time.monotonic(), self.has_edge, after_s())
            if out is not None:
                level, text = out
                # Two call sites: rclpy refuses a second severity from one line of code
                # ("Logger severity cannot be changed between calls").
                if level == "warn":
                    node.get_logger().warning(text)
                else:
                    node.get_logger().info(text)
            if wait.done and timer is not None:
                timer.cancel()

        timer = node.create_timer(1.0, tick)
        self.static_wait = wait
        return wait

    def has_edge(self, parent: str, child: str) -> bool:
        """Whether ``parent -> child`` is in the buffer at all (latest; a static edge has no
        time)."""
        try:
            return bool(self.buffer.can_transform(parent, child, Time()))
        except Exception:  # tf2's own errors: not there
            return False

    def close(self) -> None:
        """Stop the listener's thread, before the node is destroyed under it; a second call does
        nothing."""
        _OPEN_LOOKUPS.discard(self)
        listener = self._listener
        if listener is None or getattr(listener, "executor", None) is None:
            return
        listener.executor.shutdown()
        listener.dedicated_listener_thread.join(timeout=STOP_PATIENCE_S)
        own = getattr(listener, "node", None) if self._owns_listener_node else None
        if own is not None:
            own.destroy_node()  # the node tf2_ros made for the listener (TF_LISTENER_NODE own)

    def _call(self, lookup: Callable[[], Any], what: str) -> Any | None:
        try:
            return lookup()
        except Exception as exc:  # tf2's Lookup / Extrapolation / Connectivity / Timeout
            if self._on_failure is not None:
                self._on_failure(tf_failure_kind(exc), f"{what}: {str(exc).strip()[:160]}")
            return None


def camera_static_edges() -> list[tuple[str, str]]:
    """The camera's ``link -> optical`` edge as camera_stream broadcasts it on ``/tf_static``
    (config/camera.json's active camera; the default frame names when the file is unreadable)."""
    try:
        camera = load_camera_mounts()
        return [(camera.link_frame, camera.optical_frame)]
    except (OSError, KeyError, ValueError):
        return [("camera_link", "camera_optical")]


def _time(stamp: Any) -> Any:
    return Time() if stamp is None else Time.from_msg(stamp)


class TfHistory:
    """TF as the :class:`pepin.frame_pose.PoseHistory` a :class:`pepin.frame_pose.FramePoser`
    asks: ``pose_at(stamp, frame, fixed)`` is ``fixed <- frame`` at ``stamp`` seconds through
    a :class:`TfLookup`, waiting up to ``timeout_s`` for the buffer to cover the moment (a
    frame's stamp is newer than the last odometry or neck message by tens of milliseconds);
    ``None`` when it cannot, the failure counted by the lookup's ``on_failure``."""

    def __init__(self, lookup: TfLookup, timeout_s: float = 0.0) -> None:
        self._lookup = lookup
        self.timeout_s = timeout_s

    def pose_at(self, stamp: float, frame: str, fixed: str) -> RigidPose | None:
        """``fixed <- frame`` at ``stamp`` (seconds), or ``None`` when TF does not have it."""
        return self._lookup.pose(fixed, frame, stamp_from_seconds(stamp), timeout_s=self.timeout_s)

    def pose_at_nowait(self, stamp: float, frame: str, fixed: str) -> RigidPose | None:
        """``fixed <- frame`` at ``stamp`` from what the buffer already holds, waiting for
        nothing (:class:`pepin.frame_pose.RecentPoseHistory`) — for the frame path, where a
        lookup that cannot be answered costs its whole timeout and the frame with it."""
        return self._lookup.pose(fixed, frame, stamp_from_seconds(stamp), timeout_s=0.0)

    def latest_pose(self, frame: str, fixed: str) -> tuple[RigidPose, float] | None:
        """The newest ``fixed <- frame`` TF holds and the stamp it is for, or ``None``: TF's own
        "latest" lookup, which never waits because it asks for whatever is there."""
        transform = self._lookup.transform(fixed, frame, None, 0.0)
        if transform is None:
            return None
        stamp = transform.header.stamp
        return pose_from_transform(transform), float(stamp.sec) + float(stamp.nanosec) * 1e-9


def bridged_qos_profile(topic: str) -> Any:
    """The QoS every endpoint of ``topic`` must use when the topic crosses between the machines
    (:data:`pepin.deployment.BRIDGED_QOS`), or the sensor-data default where there is no rule.
    A reader and a writer that disagree on reliability do not match, and the loser hears nothing.
    """
    pinned = bridged_qos(topic)
    if pinned is None:
        return qos_profile_sensor_data
    reliability, depth = pinned
    return QoSProfile(
        depth=depth,
        reliability=(
            ReliabilityPolicy.RELIABLE
            if reliability == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        ),
    )


class LeanFeed:
    """``/imu/data_raw`` as the cart's lean: one subscription feeding one
    :class:`pepin.lean.LeanEstimator`, and that estimator as the
    :class:`pepin.lean.LeanSource` a :class:`pepin.frame_pose.FramePoser` or a floor plane
    takes from — so a node has one owner of "how far the body leans" instead of a copy.

    The mount rule, which each of the depth nodes used to carry its own copy of: a reading
    already published in ``base_link`` (the C++ bridge's — ``base_bridge.cpp`` turns every
    sample into the robot's axes before publishing) is taken as it is, a reading in any other
    frame goes through ``config/imu.json`` (:class:`pepin.mounts.Mounts`), and a reading in
    another frame with no mount is refused: ``on_unmounted(frame_id)`` is called once per
    reading so the node can say what it does about it (turn a stage off, count it, log it).
    The estimator appears on the first usable reading, not before, and starts from
    config/imu.json's measured ``level`` block: the chip's own residual roll and pitch on a
    level floor are subtracted from every lean, and the gyro's offset measured there is the
    bias the filter begins with instead of zero.
    """

    def __init__(
        self,
        node: Any,
        config_dir: Path,
        *,
        topic: str = "/imu/data_raw",
        use_gyro: bool = True,
        on_unmounted: Callable[[str], None] | None = None,
        enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._log = node.get_logger()
        self._mount = self._rotation(config_dir)
        self._level = self._level_pose(config_dir)
        self._on_unmounted = on_unmounted
        self._enabled = enabled
        self._use_gyro = use_gyro
        self.estimator: LeanEstimator | None = None
        node.create_subscription(Imu, topic, self._on_imu, bridged_qos_profile(topic))

    @property
    def use_gyro(self) -> bool:
        """Whether the gyro carries the fast part of the lean (off: the accelerometer alone,
        the filter the floor anchor has always run)."""
        return self._use_gyro

    @use_gyro.setter
    def use_gyro(self, value: bool) -> None:
        self._use_gyro = bool(value)
        if self.estimator is not None:
            self.estimator.use_gyro = self._use_gyro

    @property
    def up(self) -> Array:
        """Which way is up in base_link, level while no reading has been believed yet."""
        return UP_LEVEL if self.estimator is None else self.estimator.up

    def lean_at(self, stamp: float) -> Lean | None:
        """:class:`pepin.lean.LeanSource`: the lean at ``stamp``, or ``None`` when the IMU
        says nothing about that moment (no reading yet, or a stamp outside the history)."""
        return None if self.estimator is None else self.estimator.lean_at(stamp)

    def report(self) -> str:
        """The lean for a report line: ``lean +0.3/-1.8 deg q0.94 bias 0.05 deg/s`` (roll and
        pitch, how much of them gravity voted for, and the zero offset learned for the gyro —
        the chip's drift, which is what a lean nobody can see would be made of), or ``lean
        none`` while no reading has been believed."""
        if self.estimator is None:
            return "lean none"
        roll, pitch = self.estimator.roll_pitch_deg
        # + 0.0 so a level cart reads +0.0 and never the -0.0 that atan2 answers for it
        return (
            f"lean {roll + 0.0:+.1f}/{pitch + 0.0:+.1f} deg q{self.estimator.quality:.2f}"
            f" bias {self.estimator.gyro_bias_deg_s:.2f} deg/s"
        )

    def _level_pose(self, config_dir: Path) -> LevelPose | None:
        """config/imu.json's measured level pose, read once: the chip's residual tilt and the
        gyro's zero offset the estimator starts from. ``None`` (with one warning) when the file
        carries no such block or a broken one — then nothing is subtracted, as before it was
        measured."""
        try:
            return LevelPose.from_config(config_dir / "imu.json")
        except (OSError, KeyError, ValueError, TypeError) as exc:
            self._log.warning(
                f"no level pose in {config_dir}/imu.json ({exc}): the chip's own tilt and the"
                " gyro's offset at rest are not subtracted"
            )
            return None

    def _rotation(self, config_dir: Path) -> Array | None:
        """The rotation from the chip's axes into base_link, read once; ``None`` (with one
        error) when the files are missing or broken."""
        try:
            rotation: Array = Mounts.load(config_dir).imu.rotation()
        except (OSError, KeyError, ValueError, TypeError) as exc:
            self._log.error(
                f"no IMU mount in {config_dir} ({exc}): a reading outside base_link cannot"
                " say how the cart leans"
            )
            return None
        return rotation

    def _on_imu(self, msg: Any) -> None:
        """One reading into the estimator, through the mount when it is not in base_link."""
        if self._enabled is not None and not self._enabled():
            return
        if self.estimator is None:
            rotation = self._start_rotation(msg.header.frame_id)
            if rotation is None:
                return
            self.estimator = LeanEstimator(rotation, use_gyro=self._use_gyro, level=self._level)
        accel, gyro = imu_arrays(msg)
        self.estimator.observe(accel, stamp_seconds(msg.header.stamp), gyro)

    def _start_rotation(self, frame_id: str) -> Array | None:
        """Which rotation this publisher's readings need, or ``None`` when they cannot be used
        (the node is told once per reading, so it can react as it likes)."""
        if frame_id == BASE_FRAME:
            return np.eye(3)  # the bridge rotated it already: a second turn would tilt the floor
        if self._mount is not None:
            return self._mount
        if self._on_unmounted is not None:
            self._on_unmounted(frame_id)
        return None


# ---- main ------------------------------------------------------------------------------------
class Fatal:
    """The node's way out when a worker thread finds it cannot go on (its only backend cannot
    be built, say): the thread leaves the reason here and a timer on the spin thread raises it
    out of ``rclpy.spin`` as ``SystemExit`` — so :func:`spin_main` still joins the workers and
    disposes the participant, and the process exits with code 1 and the reason on stderr, as
    loud as a node that failed in its constructor (the launch respawns it; the respawn is the
    retry). Nothing in rclpy ends a spin from another thread with an exit code: ``shutdown()``
    is the normal end, code 0, "finished cleanly" in the launch's log."""

    def __init__(self, node: Any, period_s: float = 1.0) -> None:
        self._reason: str | None = None
        node.create_timer(period_s, self._raise)

    @property
    def leaving(self) -> bool:
        """Whether a reason was left: the node ends within a period."""
        return self._reason is not None

    def leave(self, reason: str) -> None:
        """Leave the reason (the first one stands); the node ends within a period."""
        if self._reason is None:
            self._reason = reason

    def _raise(self) -> None:
        if self._reason is not None:
            raise SystemExit(self._reason)


def spin_main(
    factory: Callable[[], Any],
    args: list[str] | None = None,
    *,
    executor: Callable[[], Any] | None = None,
) -> None:
    """A node's ``main``: init, build the node, spin, and leave cleanly on SIGINT — the signal
    the launch sends at shutdown and ``ros/laptop.sh kick`` sends by hand. ``executor`` builds
    the executor to spin on (rclpy's single-threaded default when None): a node whose service
    answers only after other callbacks have run needs more than one thread.

    ``kill -USR2`` prints every thread's stack to stderr and changes nothing
    (:func:`listen_for_stack_dumps`).

    rclpy's handler shuts the context down and the spin ends with ``KeyboardInterrupt`` or
    ``ExternalShutdownException`` (whichever lands first); both are the normal end. So is ANY
    exception that finds the context already down: the handler shuts it from a thread of its
    own, and when that lands between the executor's check and its wait set the spin raises
    rcl's "failed to initialize wait set: the given context is not valid" instead — measured
    with the real rclpy (scratch/bag_recorder_sigint.sh: 1 of 16 SIGINTs, exit code 1 before
    this), and a SIGINT during the constructor fails the node's creation the same way. One line
    on stderr names it; with the context still up the exception is real and propagates. The
    loud end is a :class:`Fatal`: ``SystemExit`` with the reason, exit code 1, through the same
    order. Then, in this order: the node's ``close()`` if it has one (its worker threads and
    TF listener are stopped and JOINED), every :class:`TfLookup` still open
    (:func:`close_open_lookups`: tf2_ros's listener thread is not a daemon, and a node with no
    close of its own never exited), ``destroy_node`` (the participant is disposed, so
    the respawn meets no ghost of the name), and the context's
    shutdown (:func:`end_context`, race-proof too). The join is the fix for the depth node's
    SIGABRT: a daemon thread still inside the network's C++ when the interpreter finalised was
    ended with ``pthread_exit``, which unwinds through ``noexcept`` frames into
    ``std::terminate`` ("terminate called without an active exception", CPython 3.12,
    gh-87135). A joined thread has no frames to unwind.
    """
    rclpy.init(args=args)
    listen_for_stack_dumps()
    node: Any = None
    try:
        node = factory()
        if executor is None:
            rclpy.spin(node)
        else:
            rclpy.spin(node, executor=executor())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        if rclpy.ok():
            raise
        print(
            f"{type(exc).__name__} after the context was shut down, taken as the normal end"
            f" (SIGINT's handler got there first): {exc}",
            file=sys.stderr,
        )
    finally:
        if node is not None:
            close = getattr(node, "close", None)
            if close is not None:
                close()
            close_open_lookups()
            node.destroy_node()
        end_context()


# The signal that prints every thread's Python stack to the node's stderr (the launch log): a
# sampling profiler with nothing to install. The board's container has no gdb and no py-spy, and
# goal_server burnt 43-46 % of an A53 core in its main thread with no callback to blame
# (2026-09-23/24); `kill -USR2 <pid>` a few times shows where that thread is.
STACK_DUMP_SIGNAL = signal.SIGUSR2


def listen_for_stack_dumps() -> None:
    """Make :data:`STACK_DUMP_SIGNAL` print every thread's stack to stderr and carry on (the
    process keeps running; without this the signal would end it). Where stderr has no file
    descriptor (a test's captured stream) or the platform has no such signal, nothing is
    registered and the node runs as before."""
    # io.UnsupportedOperation (a stream with no fileno) is both an OSError and a ValueError
    with suppress(OSError, ValueError, RuntimeError):
        faulthandler.register(STACK_DUMP_SIGNAL, all_threads=True, chain=False)


def close_open_lookups() -> int:
    """Close every :class:`TfLookup` whose listener thread still runs; how many there were. The
    thread is not a daemon, so one left open keeps the process alive after its spin has ended."""
    lookups = list(_OPEN_LOOKUPS)
    for lookup in lookups:
        lookup.close()
    return len(lookups)


def end_context() -> None:
    """Shut rclpy's default context down unless something already has.

    rclpy's SIGINT handler shuts the context down TOO, from a thread of its own, so between a
    check that finds the context up and the shutdown that check guards, the handler can get there
    first — rclpy's own ``try_shutdown`` included, whose check and shutdown are two calls the
    handler does not wait for. The bag recorder died on exactly that on 2026-09-23 ("failed to
    shutdown: rcl_shutdown already called on the given context", exit code 1 in the launch's
    log, after ``if rclpy.ok(): rclpy.shutdown()``). A shutdown that fails with the context down
    afterwards IS that race, and the job is done; one that fails with the context still up is a
    real failure and is raised.
    """
    try:
        rclpy.try_shutdown()
    except Exception:
        if rclpy.ok():
            raise
