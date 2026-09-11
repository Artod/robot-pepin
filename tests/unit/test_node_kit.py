"""The node skeleton: the worker, the report tally, the switches, TF lookups, the clean exit."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
import ros_stubs

RCLPY = ros_stubs.install()

from pepin_bringup import node_kit  # noqa: E402
from pepin_bringup.node_kit import Switches, Tally, TfLookup, Worker, spin_main  # noqa: E402


class FakeNode:
    """What node_kit asks of a node: parameters, a set callback, a logger."""

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        self.overrides = overrides or {}
        self.declared: dict[str, Any] = {}
        self.callback: Any = None
        self.lines: list[str] = []
        self.destroyed = False
        self.closed = False

    def declare_parameter(self, name: str, default: Any) -> Any:
        """rclpy's: the default is what the node declares, the override is what it reads back."""
        self.declared[name] = default
        return type("Param", (), {"value": self.overrides.get(name, default)})()

    def add_on_set_parameters_callback(self, callback: Any) -> None:
        self.callback = callback

    def get_logger(self) -> Any:
        node = self

        class Logger:
            def info(self, text: str) -> None:
                node.lines.append(text)

            error = warning = info

        return Logger()

    def destroy_node(self) -> None:
        self.destroyed = True

    def close(self) -> None:
        assert not self.destroyed, "closed after the destroy: the worker would outlive the node"
        self.closed = True


class Param:
    def __init__(self, name: str, value: Any) -> None:
        self.name, self.value = name, value


# ---- Worker --------------------------------------------------------------------------------
def test_the_worker_takes_the_newest_item_and_drops_the_one_waiting() -> None:
    """Three frames offered while the first is being worked: the middle one is never worked
    (dropped, and offer says so), the last one is."""
    worked: list[int] = []
    gate = threading.Event()

    def work(item: int) -> None:
        worked.append(item)
        gate.wait(1.0)  # the first item is slow: the others queue up behind it

    worker = Worker(work, name="test", on_error=lambda text: pytest.fail(text)).start()
    assert worker.offer(1) is False
    time.sleep(0.05)  # the thread took item 1 and is inside work()
    assert worker.offer(2) is False  # nothing waiting: item 1 is being worked, not pending
    assert worker.offer(3) is True  # item 2 was still waiting: replaced
    gate.set()
    time.sleep(0.1)
    assert worked == [1, 3]
    assert worker.stop() and not worker.alive


def test_a_raise_in_the_work_is_reported_and_the_worker_goes_on() -> None:
    errors: list[str] = []
    done = threading.Event()

    def work(item: str) -> None:
        if item == "bad":
            raise RuntimeError("a frame the network choked on")
        done.set()

    worker = Worker(work, name="test", on_error=errors.append).start()
    worker.offer("bad")
    time.sleep(0.05)
    worker.offer("good")
    assert done.wait(1.0)
    assert len(errors) == 1 and "a frame the network choked on" in errors[0]
    assert worker.stop()


def test_stop_waits_for_the_item_in_progress_and_clears_the_rest() -> None:
    """The way out of the SIGABRT: the thread is joined after its current item, so no daemon
    thread is inside C++ frames when the interpreter finalises."""
    started, release = threading.Event(), threading.Event()
    finished: list[str] = []

    def work(item: str) -> None:
        started.set()
        release.wait(1.0)
        finished.append(item)

    worker = Worker(work, name="test", on_error=lambda text: pytest.fail(text)).start()
    worker.offer("in progress")
    assert started.wait(1.0)
    worker.offer("never")
    worker.clear()
    stopper = threading.Thread(target=worker.stop)
    stopper.start()
    time.sleep(0.05)
    assert worker.alive, "stop must wait for the item in progress, not abandon it"
    release.set()
    stopper.join(1.0)
    assert finished == ["in progress"] and not worker.alive
    assert worker.stop(), "a second stop is harmless"


def test_an_error_while_stopping_is_not_reported() -> None:
    errors: list[str] = []
    inside = threading.Event()

    def work(item: str) -> None:
        inside.set()
        time.sleep(0.05)
        raise RuntimeError("publishing on a context that is gone")

    worker = Worker(work, name="test", on_error=errors.append).start()
    worker.offer("last")
    assert inside.wait(1.0)
    assert worker.stop() and errors == []


# ---- Tally ---------------------------------------------------------------------------------
def test_the_tally_hands_over_a_window_and_starts_afresh() -> None:
    tally = Tally(stages=("network", "scan"))
    tally.count("frames")
    tally.count("frames", 2)
    tally.spent("network", 0.2)
    with tally.measure("scan"):
        pass
    tally.spent("extra", 0.01)  # a stage not declared still gets its summary
    tally.sample("age", 0.1)
    tally.sample("age", 0.3)
    tally.note("Lookup", "map<-camera: frame does not exist")
    window = tally.take()
    assert window.counts["frames"] == 3 and window.counts["never"] == 0
    assert window.seconds["network"] == pytest.approx(0.2)
    assert list(window.timing) == ["network", "scan", "extra"]
    assert window.timing["network"].median_ms == pytest.approx(200.0)
    assert window.timing["scan"].count == 1 and window.timing["extra"].count == 1
    assert window.samples["age"] == [0.1, 0.3] and window.notes == {
        "Lookup": "map<-camera: frame does not exist"
    }
    assert window.ms_per("network", "frames") == pytest.approx(200.0 / 3)
    assert window.ms_per("nothing", "frames") == 0.0
    assert window.rate("frames") > 0 and 0 <= window.elapsed_s < 1.0
    assert window.stages().startswith("network 200/200 scan ")
    fresh = tally.take()
    assert fresh.counts["frames"] == 0 and fresh.samples == {} and fresh.notes == {}
    assert list(fresh.timing) == ["network", "scan"] and fresh.timing["network"].count == 0


def test_counts_from_two_threads_are_not_lost() -> None:
    tally = Tally()

    def bump() -> None:
        for _ in range(2000):
            tally.count("hits")

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tally.take().counts["hits"] == 8000


# ---- Switches ------------------------------------------------------------------------------
def test_switches_are_declared_from_their_defaults_and_read_back_typed() -> None:
    node = FakeNode(overrides={"edge_filter": False, "min_weight": 3})
    switches = Switches(node, {"floor_anchor": True, "edge_filter": True, "min_weight": 2.0})
    assert node.declared == {"floor_anchor": True, "edge_filter": True, "min_weight": 2.0}
    assert switches.on("floor_anchor") and not switches.on("edge_filter")
    assert switches["min_weight"] == 3.0 and isinstance(switches["min_weight"], float)
    assert switches.values() == {"floor_anchor": True, "edge_filter": False, "min_weight": 3.0}
    assert switches.state() == "floor_anchor on, edge_filter off, min_weight 3.0"
    assert node.callback is not None, "ros2 param set reaches the switches"


def test_a_set_flips_the_switch_and_is_logged_and_an_unknown_name_is_refused() -> None:
    node = FakeNode()
    switches = Switches(node, {"floor_anchor": True, "min_weight": 2.0})
    result = node.callback([Param("floor_anchor", False), Param("min_weight", 4)])
    assert result.successful and switches.state() == "floor_anchor off, min_weight 4.0"
    assert node.lines == ["floor_anchor off", "min_weight = 4.0"]
    refused = node.callback([Param("threads", 4), Param("floor_anchor", True)])
    assert not refused.successful and refused.reason == "threads: not live, set at the next start"
    assert not switches.on("floor_anchor"), "a refused batch changes nothing"


def test_on_change_sees_every_change_and_can_refuse_one() -> None:
    node = FakeNode()
    seen: list[tuple[str, Any]] = []

    def on_change(name: str, value: Any) -> None:
        seen.append((name, value))
        if name == "surface_hz" and value <= 0:
            raise ValueError("surface_hz must be positive")

    switches = Switches(node, {"enabled": True, "surface_hz": 1.0}, on_change=on_change)
    assert node.callback([Param("surface_hz", 2.0)]).successful
    refused = node.callback([Param("surface_hz", 0.0)])
    assert not refused.successful and refused.reason == "surface_hz must be positive"
    assert switches["surface_hz"] == 2.0, "the old value stays after a refusal"
    switches.set("enabled", False)  # the node flips one itself
    assert seen == [("surface_hz", 2.0), ("surface_hz", 0.0), ("enabled", False)]
    assert node.lines[-1] == "enabled off"


# ---- TfLookup ------------------------------------------------------------------------------
class LookupException(Exception):  # noqa: N818 — tf2's own names are what is classified
    pass


class ExtrapolationException(Exception):  # noqa: N818 — tf2's own names
    pass


def test_a_tf_failure_is_classified_by_its_exception_s_name() -> None:
    assert node_kit.tf_failure_kind(LookupException("x")) == "Lookup"
    assert node_kit.tf_failure_kind(ExtrapolationException("x")) == "Extrapolation"
    assert node_kit.tf_failure_kind(RuntimeError("x")) == "RuntimeError"
    assert node_kit.tf_failure_kind(type("Exception", (Exception,), {})()) == "Unknown"


def test_a_lookup_answers_a_pose_or_none_with_the_failure_told() -> None:
    buffer = ros_stubs.Buffer()
    buffer.transforms[("map", "camera")] = ros_stubs.TransformStamped(
        transform=ros_stubs.Transform(translation=ros_stubs.Vector3(x=1.0, y=2.0, z=0.5))
    )
    failures: list[tuple[str, str]] = []
    tf = TfLookup(
        FakeNode(), buffer=buffer, on_failure=lambda kind, text: failures.append((kind, text))
    )
    stamp = ros_stubs.Time(sec=7, nanosec=5)
    pose = tf.pose("map", "camera", stamp, timeout_s=0.3)
    assert pose is not None and pose.translation.tolist() == [1.0, 2.0, 0.5]
    target, source, at, timeout = buffer.calls[-1]
    assert (target, source, at.nanoseconds, timeout.nanoseconds) == (
        "map",
        "camera",
        7_000_000_005,
        300_000_000,
    )
    assert tf.pose("map", "camera").translation.tolist() == [1.0, 2.0, 0.5]  # type: ignore[union-attr]
    assert buffer.calls[-1][2].nanoseconds == 0, "no stamp asks for the latest"
    buffer.error = ExtrapolationException("  Lookup would require extrapolation into the future  ")
    assert tf.pose("map", "camera", stamp) is None
    assert failures == [
        ("Extrapolation", "map<-camera: Lookup would require extrapolation into the future")
    ]
    motion = tf.motion("base_link", stamp, stamp, "odom", timeout_s=0.2)
    assert (
        motion is None
        and failures[-1][0] == "Extrapolation"
        and failures[-1][1].startswith("base_link odom:")
    )
    buffer.error = None
    buffer.transforms[("base_link", "base_link")] = ros_stubs.TransformStamped()
    carried = tf.motion("base_link", stamp, ros_stubs.Time(sec=8), "odom")
    assert carried is not None and carried.translation.tolist() == [0.0, 0.0, 0.0]
    assert buffer.calls[-1][1].nanoseconds == 8_000_000_000 and buffer.calls[-1][4] == "odom"
    tf.close()  # no listener of its own: nothing to stop


def test_the_listener_s_thread_is_stopped_and_joined_on_close() -> None:
    tf = TfLookup(FakeNode())
    listener: Any = tf._listener
    assert listener is not None and isinstance(tf.buffer, ros_stubs.Buffer)
    tf.close()
    assert listener.executor.shut_down and listener.dedicated_listener_thread.joined


# ---- spin_main -----------------------------------------------------------------------------
@pytest.mark.parametrize("end", [KeyboardInterrupt, ros_stubs.ExternalShutdownException])
def test_the_main_leaves_in_order_on_either_way_the_spin_ends(end: type[BaseException]) -> None:
    """close (workers joined) before destroy_node before the context's shutdown, whichever of
    the two exceptions the signal handler's race delivers; the exit is quiet."""
    RCLPY.log.clear()
    nodes: list[FakeNode] = []

    def factory() -> FakeNode:
        nodes.append(FakeNode())
        return nodes[-1]

    def end_spin() -> None:
        raise end()

    RCLPY.on_spin = end_spin
    try:
        spin_main(factory)
    finally:
        RCLPY.on_spin = None
    assert RCLPY.log == ["init", "spin", "try_shutdown"]
    assert nodes[0].closed and nodes[0].destroyed


def test_the_worker_is_joined_before_the_node_is_destroyed_or_the_context_shut_down() -> None:
    """The SIGABRT, as a test: the depth node's worker was a daemon thread inside torch and the
    network when the interpreter finalised, and CPython ends such a thread with pthread_exit,
    which unwinds into std::terminate. Here the worker is busy when the signal lands; the order
    recorded must be worker-out, destroy, shutdown, and the thread must be gone by then."""
    RCLPY.log.clear()
    order: list[str] = []
    inside, release = threading.Event(), threading.Event()

    class Busy(FakeNode):
        def __init__(self) -> None:
            super().__init__()
            self.worker: Worker[int] = Worker(self._work, name="busy", on_error=print).start()
            self.worker.offer(1)

        def _work(self, item: int) -> None:
            inside.set()  # "inside the network's C++": the interpreter must not finalise now
            release.wait(1.0)
            order.append("worker out")

        def close(self) -> None:
            self.worker.stop()

        def destroy_node(self) -> None:
            order.append("destroy")

    nodes: list[Busy] = []

    def factory() -> Busy:
        nodes.append(Busy())
        return nodes[-1]

    def end_spin() -> None:
        assert inside.wait(1.0)
        threading.Timer(0.05, release.set).start()  # the frame finishes just after the signal
        raise KeyboardInterrupt

    RCLPY.on_spin = end_spin
    try:
        spin_main(factory)
    finally:
        RCLPY.on_spin = None
    order.append("shutdown" if RCLPY.log[-1] == "try_shutdown" else "no shutdown")
    assert order == ["worker out", "destroy", "shutdown"]
    assert not nodes[0].worker.alive, "a daemon thread left running is the abort"


def test_a_node_that_fails_to_build_still_shuts_the_context_down() -> None:
    RCLPY.log.clear()

    def factory() -> Any:
        raise RuntimeError("no such model")

    with pytest.raises(RuntimeError, match="no such model"):
        spin_main(factory)
    assert RCLPY.log == ["init", "try_shutdown"]
