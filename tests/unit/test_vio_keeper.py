"""OpenVINS keeper in pepin-vio: verdicts, the in-process reset, the fallback restart, the seed."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
import ros_stubs

ros_stubs.install()

from diagnostic_msgs.msg import DiagnosticStatus, KeyValue  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from pepin_bringup.vio_keeper import (  # noqa: E402
    EKF_TOPIC,
    HEALTH_TOPIC,
    RESET_SERVICE,
    RESTART_SERVICE,
    SEED_TOPIC,
    VioKeeper,
)
from std_srvs.srv import Trigger  # noqa: E402

from pepin.tsdf import RigidPose  # noqa: E402
from pepin.vio_recover import OPENVINS_FLAGS, OPENVINS_KNOBS  # noqa: E402


class Face:
    def __init__(self) -> None:
        self.events: list[str] = []

    def event(self, name: str, *, end: bool = False) -> None:
        self.events.append(name)

    def clear(self) -> None: ...
    def lease(self, seconds: float) -> None: ...
    def close(self) -> None: ...


class Tf:
    """head_imu <- base_link: the IMU 0.8 m up on the pan axis, panned ``pan`` (90 deg: its x to the
    cart's left)."""

    buffer = None

    def __init__(self) -> None:
        self.pan = math.pi / 2

    def pose(self, target: str, source: str, stamp: Any = None, timeout_s: float = 0.0) -> Any:
        assert (target, source) == ("head_imu", "base_link")
        c, s = math.cos(self.pan), math.sin(self.pan)
        r_i_b = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T
        return RigidPose(r_i_b, -(r_i_b @ np.array([0.0, 0.0, 0.8])))


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make(tf: Tf | None = None) -> tuple[VioKeeper, list[list[str]], Face, Clock]:
    ran: list[list[str]] = []

    def run(command: Sequence[str]) -> int:
        ran.append(list(command))
        return 0

    face, clock = Face(), Clock()
    node = VioKeeper(run=run, face=face, tf=tf or Tf(), clock=clock)
    return node, ran, face, clock


def health_msg(t: float, *, epoch: int = 0, initialized: bool = True, tracks: int = 80) -> Any:
    values = {
        "t": f"{t:.6f}",
        "epoch": str(epoch),
        "initialized": "1" if initialized else "0",
        "tracked": "100",
        "persistent": str(tracks),
        "msckf": "3",
        "slam": "0",
        "zupt": "0",
        "gyro": "0.05",
        "resets": str(epoch),
        "warm_seeds": str(epoch),
        "standard_inits": "0",
        "seed": "warm 0.10 s after the reset's first frame" if epoch else "",
    }
    if initialized:
        values["speed"] = "0.2"
    msg = DiagnosticStatus()
    msg.hardware_id = "global" if epoch == 0 else f"global_{epoch}"
    msg.values = [KeyValue(key=k, value=v) for k, v in values.items()]
    return msg


def test_the_relays_restart_is_answered_by_a_reset_in_the_process_with_the_knobs_pushed() -> None:
    node, ran, face, _ = make()
    reset = node.service_clients[RESET_SERVICE]
    reset.ready = True
    reset.response = Trigger.Response(success=True, message="reset 1 made")
    pushed_at_start = len(node._params.sets)
    _, serve = node.services[RESTART_SERVICE]
    answer = serve(Trigger.Request(), Trigger.Response())
    assert answer.success and answer.message.startswith("reset asked")
    assert len(reset.calls) == 1, "the in-process reset, not a kill"
    assert ran == []
    pushed = node._params.sets[pushed_at_start]
    assert node._params.remote == "/ov_msckf/run_subscribe_msckf"
    assert {p.name for p in pushed} == set(OPENVINS_KNOBS) | set(OPENVINS_FLAGS)
    assert face.events == ["vio_restart"]


def test_an_unserved_reset_falls_back_to_the_process_restart_unless_told_not_to() -> None:
    node, ran, _, _ = make()
    _, serve = node.services[RESTART_SERVICE]
    answer = serve(Trigger.Request(), Trigger.Response())
    assert answer.success and "signalled" in answer.message
    assert ran == [["pkill", "-INT", "-f", "run_subscribe_msckf"]], "SIGINT, never -9"
    with ros_stubs.parameters(vio_restart_fallback=False):
        node, ran, _, _ = make()
    answer = node.services[RESTART_SERVICE][1](Trigger.Request(), Trigger.Response())
    assert not answer.success and ran == []


def test_vio_recover_restart_is_the_old_kill_and_off_does_nothing() -> None:
    with ros_stubs.parameters(vio_recover="restart"):
        node, ran, _, _ = make()
    node.service_clients[RESET_SERVICE].ready = True
    node.services[RESTART_SERVICE][1](Trigger.Request(), Trigger.Response())
    assert ran == [["pkill", "-INT", "-f", "run_subscribe_msckf"]]
    assert node.service_clients[RESET_SERVICE].calls == []
    with ros_stubs.parameters(vio_recover="off"):
        node, ran, face, _ = make()
    answer = node.services[RESTART_SERVICE][1](Trigger.Request(), Trigger.Response())
    assert not answer.success and ran == [] and face.events == []


def test_a_dark_stretch_resets_openvins_and_the_recovery_is_timed_to_its_first_frame() -> None:
    node, ran, _, clock = make()
    reset = node.service_clients[RESET_SERVICE]
    reset.ready = True
    _, on_health = node.subs[HEALTH_TOPIC]
    t = 0.0
    for _i in range(30):  # 3 s of a good picture: past the grace
        on_health(health_msg(t))
        t += 0.1
        clock.now += 0.1
    for _i in range(10):  # 1 s of dark, the head still
        on_health(health_msg(t, tracks=0))
        t += 0.1
        clock.now += 0.1
    assert len(reset.calls) == 1 and ran == []
    asked = clock.now - 0.1
    on_health(health_msg(t, epoch=1, initialized=False))  # the new filter, waiting for a picture
    clock.now += 0.6
    on_health(health_msg(t + 0.6, epoch=1))  # seeded and publishing
    back = [x for x in node.get_logger().texts("info") if x.startswith("OpenVINS back")]
    assert back and back[0].startswith(f"OpenVINS back {clock.now - asked:.2f} s after the reset")
    assert "global_1" in back[0]
    node._report()
    line = node.get_logger().texts("info")[-1]
    assert "lost: disagree 0, dark 1" in line and "1 recovered in p50 0.7 s" in line


def test_the_watch_can_be_switched_off_and_then_only_the_relay_recovers() -> None:
    with ros_stubs.parameters(vio_watch=False):
        node, _, _, _ = make()
    reset = node.service_clients[RESET_SERVICE]
    reset.ready = True
    _, on_health = node.subs[HEALTH_TOPIC]
    for i in range(60):
        on_health(health_msg(i / 10, tracks=0))
    assert reset.calls == []


def ekf(vx: float, wz: float) -> Any:
    msg = Odometry()
    msg.header.stamp.sec = 1791255800
    msg.twist.twist.linear.x = vx
    msg.twist.twist.angular.z = wz
    cov = [0.0] * 36
    cov[0], cov[7], cov[35] = 1e-4, 1e-4, 4e-4
    msg.twist.covariance = cov
    return msg


def test_the_velocity_seed_is_published_in_the_imus_axes_only_while_the_head_is_still() -> None:
    tf = Tf()
    node, _, _, clock = make(tf)
    _, on_ekf = node.subs[EKF_TOPIC]
    seeds = node.pubs[SEED_TOPIC]
    for _i in range(15):  # 0.28 s watched: not yet the whole window
        on_ekf(ekf(0.2, 0.0))
        clock.now += 0.02
    assert seeds.sent == []
    on_ekf(ekf(0.2, 0.0))  # 0.30 s of a still head
    assert len(seeds.sent) == 1
    seed = seeds.sent[0]
    assert seed.header.frame_id == "head_imu" and seed.header.stamp.sec == 1791255800
    v = seed.twist.twist.linear
    assert (v.x, v.y, v.z) == pytest.approx((0.0, -0.2, 0.0))  # forward: the IMU's -y (x left)
    cov = seed.twist.covariance
    assert cov[0] == 1e-4 and cov[7] == 1e-4 and math.isclose(cov[14], 0.02**2)
    tf.pan += math.radians(3.0)  # a saccade
    clock.now += 0.02
    on_ekf(ekf(0.2, 0.0))
    assert len(seeds.sent) == 1
    node._report()
    assert "head moving 1" in node.get_logger().texts("info")[-1]


def test_a_live_knob_change_of_openvinss_own_is_pushed_to_it() -> None:
    node, _, _, _ = make()
    before = len(node._params.sets)
    from rclpy.parameter import Parameter

    node.set_parameters([Parameter("seed_min_features", value=25)])
    assert [p.name for p in node._params.sets[before]] == ["seed_min_features"]
    node.set_parameters([Parameter("vio_dark_s", value=2.5)])
    assert node._watch.dark_s == 2.5 and len(node._params.sets) == before + 1
