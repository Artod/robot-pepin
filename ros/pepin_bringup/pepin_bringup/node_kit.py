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

import os
import threading
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

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
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from pepin.flags import Flag, FlagSet
from pepin.telemetry import LatencySummary, LatencyTracker
from pepin.tsdf import RigidPose
from pepin_bringup.msgs import pose_from_transform, stamp_from_seconds, stamp_seconds

__all__ = [
    "Fatal",
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
        self._listener = (
            None if buffer is not None else TransformListener(self.buffer, node, spin_thread=True)
        )
        self._on_failure = on_failure

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

    def close(self) -> None:
        """Stop the listener's thread, before the node is destroyed under it."""
        listener = self._listener
        if listener is None or getattr(listener, "executor", None) is None:
            return
        listener.executor.shutdown()
        listener.dedicated_listener_thread.join(timeout=STOP_PATIENCE_S)

    def _call(self, lookup: Callable[[], Any], what: str) -> Any | None:
        try:
            return lookup()
        except Exception as exc:  # tf2's Lookup / Extrapolation / Connectivity / Timeout
            if self._on_failure is not None:
                self._on_failure(tf_failure_kind(exc), f"{what}: {str(exc).strip()[:160]}")
            return None


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


def spin_main(factory: Callable[[], Any], args: list[str] | None = None) -> None:
    """A node's ``main``: init, build the node, spin, and leave cleanly on SIGINT — the signal
    the launch sends at shutdown and ``ros/laptop.sh kick`` sends by hand.

    rclpy's handler shuts the context down and the spin ends with ``KeyboardInterrupt`` or
    ``ExternalShutdownException`` (whichever lands first); both are the normal end. The loud
    end is a :class:`Fatal`: ``SystemExit`` with the reason, exit code 1, through the same
    order. Then, in this order: the node's ``close()`` if it has one (its worker threads and
    TF listener are stopped and JOINED), ``destroy_node`` (the DDS participant is disposed, so
    the bridge forgets the name at once and the respawn meets no ghost), and the context's
    shutdown. The join is the fix for the depth node's SIGABRT: a daemon thread still inside
    the network's C++ when the interpreter finalised was ended with ``pthread_exit``, which
    unwinds through ``noexcept`` frames into ``std::terminate`` ("terminate called without an
    active exception", CPython 3.12, gh-87135). A joined thread has no frames to unwind.
    """
    rclpy.init(args=args)
    node: Any = None
    try:
        node = factory()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            close = getattr(node, "close", None)
            if close is not None:
                close()
            node.destroy_node()
        rclpy.try_shutdown()
