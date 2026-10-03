"""The gaze gate: frames whose exposure overlaps a head saccade or a fast body yaw are for nothing.

The core (pepin.gaze_gate) is judged on scripted /gaze/state streams in board seconds, the way the
arbiter's contract (gaze.md 3.4) writes them; the ROS half (pepin_bringup.gaze_feed) and the visual
odometry node under the stubs. depth_stream's and sensor_pack's own gate tests sit beside their
other tests, where their fixtures are.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
import ros_stubs

ros_stubs.install()

from pepin_bringup.gaze_feed import GazeFeed, gate_counts  # noqa: E402
from pepin_bringup.visual_odometry import RAW_TOPIC, VO_TOPIC, VisualOdometry  # noqa: E402

from pepin.flags import load_table  # noqa: E402
from pepin.gaze_gate import (  # noqa: E402
    BLIND,
    GAZE_GATE,
    GAZE_STATE_TOPIC,
    SPINNING,
    STATE_STALE_S,
    FrameGate,
    GazeState,
    YawLog,
    board_seconds,
)

NODES = "ros/pepin_bringup/pepin_bringup"
GATED = ("depth_stream", "sensor_pack", "visual_odometry")
T0 = 1_759_400_000.0  # a board second: the frames' and the states' common clock


def state(phase: str, since: float, blind: bool, **extra: Any) -> GazeState:
    """A state as the arbiter's JSON carries it, read back through the parser."""
    text = json.dumps({"phase": phase, "since": T0 + since, "blind": blind, **extra})
    parsed = GazeState.from_json(text)
    assert parsed is not None
    return parsed


def saccade(gate: FrameGate, now: float = 0.0, settle_s: float = 0.4) -> None:
    """A head that was still, wrote a saccade at +1.0 s and settled at +1.0 + ``settle_s``, every
    state the contract publishes: blind from the write until one frame after settling."""
    gate.observe_state(state("still", 0.0, False), now)
    gate.observe_state(state("saccade", 1.0, True), now)
    gate.observe_state(state("still", 1.0 + settle_s, True), now)
    gate.observe_state(state("still", 1.0 + settle_s, False), now)


# ---- the state -------------------------------------------------------------------------------
def test_a_state_is_the_contract_s_json_and_anything_else_is_refused() -> None:
    full = GazeState.from_json(
        json.dumps(
            {
                "phase": "saccade",
                "pan_rad": 0.5,
                "tilt_rad": 0.4,
                "since": T0,
                "request_id": "r7",
                "source": "nav.stall",
                "blind": True,
                "extra": "ignored",
            }
        )
    )
    assert full == GazeState("saccade", True, T0, 0.5, 0.4, "r7", "nav.stall")
    as_stamp = GazeState.from_json(
        '{"phase": "still", "blind": false, "since": {"sec": 12, "nanosec": 500000000}}'
    )
    assert as_stamp is not None and as_stamp.since == 12.5, "a stamp message reads too"
    for bad in (
        "not json",
        "[]",
        '{"since": 1.0}',
        '{"blind": "yes", "since": 1.0}',
        '{"blind": true}',
        '{"blind": true, "since": "soon"}',
    ):
        assert GazeState.from_json(bad) is None, bad
    assert board_seconds(True) is None and board_seconds(float("nan")) is None


# ---- the blind intervals ---------------------------------------------------------------------
def test_without_any_state_every_frame_passes_as_before() -> None:
    gate = FrameGate(yaw_dps=0.0)
    assert all(gate.verdict(T0 + t / 10, now=0.0) is None for t in range(100))


def test_a_saccade_blinds_from_its_write_to_one_frame_after_settling_by_the_exposure() -> None:
    """Exposure +-35 ms, settle +0.1 s: the saccade written at 1.0 and settled at 1.4 blinds a
    frame whose window reaches 1.0 and one whose window starts before 1.5, nothing else."""
    gate = FrameGate(exposure_s=0.035, settle_s=0.1)
    saccade(gate)
    verdicts = {t: gate.verdict(T0 + t, now=0.5) for t in (0.90, 0.97, 1.2, 1.45, 1.53, 1.6)}
    assert verdicts == {0.90: None, 0.97: BLIND, 1.2: BLIND, 1.45: BLIND, 1.53: BLIND, 1.6: None}


def test_a_lost_state_does_not_shorten_the_interval() -> None:
    """The arbiter's 'settled but still blind' message never arrived: the end is still the
    settle stamp plus one frame period, from the next state's own since."""
    gate = FrameGate(exposure_s=0.0, settle_s=0.1)
    gate.observe_state(state("saccade", 1.0, True), 0.0)
    gate.observe_state(state("still", 1.4, False), 0.0)
    assert gate.verdict(T0 + 1.45, now=0.1) == BLIND
    assert gate.verdict(T0 + 1.55, now=0.1) is None


def test_chained_saccades_are_one_interval_to_the_last_settle() -> None:
    gate = FrameGate(exposure_s=0.0, settle_s=0.1)
    gate.observe_state(state("saccade", 1.0, True), 0.0)
    gate.observe_state(state("saccade", 1.3, True), 0.0)  # re-aimed before settling
    gate.observe_state(state("still", 1.7, True), 0.0)
    gate.observe_state(state("still", 1.7, False), 0.0)
    assert [gate.verdict(T0 + t, now=0.1) for t in (0.95, 1.5, 1.79, 1.81)] == [
        None,
        BLIND,
        BLIND,
        None,
    ]


def test_an_open_saccade_blinds_while_the_arbiter_speaks_and_a_silent_one_blinds_nobody() -> None:
    gate = FrameGate(exposure_s=0.0)
    gate.observe_state(state("saccade", 1.0, True), now=100.0)
    assert gate.verdict(T0 + 5.0, now=100.0 + STATE_STALE_S / 2) == BLIND, "still moving"
    assert gate.verdict(T0 + 0.9, now=100.0) is None, "before the write"
    assert gate.verdict(T0 + 5.0, now=100.0 + STATE_STALE_S + 0.1) is None, (
        "an arbiter that died mid-saccade must not blind the robot"
    )
    assert "STALE" in gate.text(now=100.0 + STATE_STALE_S + 0.1)


def test_the_settle_knob_moves_every_interval_kept_live() -> None:
    gate = FrameGate(exposure_s=0.0, settle_s=0.1)
    saccade(gate)
    assert gate.verdict(T0 + 1.6, now=0.0) is None
    gate.settle_s = 0.3
    assert gate.verdict(T0 + 1.6, now=0.0) == BLIND
    assert gate.settle_s == 0.3


# ---- the body's yaw --------------------------------------------------------------------------
def test_a_fast_body_yaw_inside_the_window_is_spinning_and_zero_turns_it_off() -> None:
    gate = FrameGate(exposure_s=0.02, yaw_dps=60.0)
    for i in range(200):  # 100 Hz, a turn at 90 deg/s between 0.50 and 0.80 s
        t = i / 100
        gate.observe_yaw(T0 + t, math.radians(90.0 if 0.5 <= t <= 0.8 else 10.0))
    assert gate.verdict(T0 + 0.3, now=0.0) is None
    assert gate.verdict(T0 + 0.6, now=0.0) == SPINNING
    assert gate.verdict(T0 + 0.49, now=0.0) == SPINNING, "the window reaches the turn"
    gate.yaw_dps = 0.0
    assert gate.verdict(T0 + 0.6, now=0.0) is None, "0 is off"


def test_a_short_exposure_between_two_imu_samples_is_judged_by_its_neighbours() -> None:
    log = YawLog()
    log.observe(T0 + 0.00, 0.1)
    log.observe(T0 + 0.01, 2.0)
    log.observe(T0 + 0.02, 0.1)
    assert log.peak(T0 + 0.004, T0 + 0.006) == 2.0, "widened by one sample gap"
    assert YawLog().peak(T0, T0 + 1.0) is None, "no IMU: nothing to judge by"
    log.observe(T0 + 0.015, 9.0)  # out of order: ignored
    log.observe(T0 + 0.03, float("nan"))  # not a number: ignored
    assert log.peak(T0 + 0.0, T0 + 0.03) == 2.0


# ---- one flag, one set of knobs, three nodes -------------------------------------------------
def test_every_gated_node_carries_the_one_gaze_gate_flag() -> None:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    for node in GATED:
        flags = load_table(repo / NODES / f"{node}.py")
        assert flags.flag("gaze_gate") == GAZE_GATE, node
    assert GAZE_GATE.default is True, "without /gaze/state nothing is blind: on is today"


# ---- the ROS half ----------------------------------------------------------------------------
class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_feed_hears_the_state_and_subscribes_the_imu_only_for_the_yaw_gate() -> None:
    node = ros_stubs.Node("probe")
    clock = Clock()
    feed = GazeFeed(node, exposure_s=0.0, settle_s=0.1, yaw_dps=0.0, clock=clock)
    assert set(node.subs) == {GAZE_STATE_TOPIC}, "the yaw gate off subscribes nothing more"
    node.subs[GAZE_STATE_TOPIC][1](ros_stubs.String(data="garbage"))
    node.subs[GAZE_STATE_TOPIC][1](
        ros_stubs.String(data=json.dumps({"phase": "saccade", "blind": True, "since": T0}))
    )
    assert feed.verdict(T0 + 0.1) == BLIND
    clock.now = STATE_STALE_S + 1.0
    assert feed.verdict(T0 + 0.1) is None
    assert "1 unreadable states" in feed.text()

    feed.set("gate_yaw_dps", 30.0)
    assert "/imu/data_raw" in node.subs, "subscribed the moment the knob is above zero"
    imu = node.subs["/imu/data_raw"][1]
    for i, frame_id in enumerate(("base_link", "imu_link", "base_link")):
        msg = ros_stubs.Imu(
            header=ros_stubs.Header(
                stamp=ros_stubs.Time(sec=100, nanosec=i * 10_000_000), frame_id=frame_id
            ),
            angular_velocity=ros_stubs.Vector3(z=math.radians(45.0)),
        )
        imu(msg)
    assert feed.verdict(100.01) == SPINNING
    assert feed.foreign_imu == 1, "a reading outside base_link is never guessed at"
    feed.set("gate_exposure_s", 0.05)
    feed.set("gate_settle_s", 0.2)
    assert (feed.gate.exposure_s, feed.gate.settle_s) == (0.05, 0.2)
    assert gate_counts({"gaze_blind": 2}, 10) == "2 blind, 0 spinning of 10 frames"


# ---- the visual odometry ---------------------------------------------------------------------
def _raw(t: float, x: float, yaw: float) -> Any:
    msg = ros_stubs.Odometry(
        header=ros_stubs.Header(
            stamp=ros_stubs.Time(sec=int(T0) + int(t), nanosec=round((t % 1) * 1e9)),
            frame_id="odom_vo",
        )
    )
    msg.pose.pose.position.x = x
    msg.pose.pose.orientation.z = math.sin(yaw / 2)
    msg.pose.pose.orientation.w = math.cos(yaw / 2)
    msg.pose.covariance = [0.001 if i % 7 == 0 else 0.0 for i in range(36)]
    return msg


POSES = (  # (t, x, yaw): a cart creeping 1 cm a frame while the head pans 0.3 rad at 0.18-0.28 s
    (0.0, 0.00, 0.0),
    (0.1, 0.01, 0.0),
    (0.2, 0.06, 0.2),  # inside the saccade: the pan read as the cart's own yaw, and a jump
    (0.3, 0.07, 0.25),
    (0.4, 0.08, 0.3),  # one frame after settling: still blind by the contract
    (0.5, 0.09, 0.3),  # the first pose after: its step starts on a blind one
    (0.6, 0.10, 0.3),
)


def _drive(gate_on: bool) -> tuple[VisualOdometry, list[Any]]:
    with ros_stubs.parameters(gaze_gate=gate_on):
        node = VisualOdometry()
    node.subs[GAZE_STATE_TOPIC][1](
        ros_stubs.String(data=json.dumps({"phase": "saccade", "blind": True, "since": T0 + 0.18}))
    )
    node.subs[GAZE_STATE_TOPIC][1](
        ros_stubs.String(data=json.dumps({"phase": "still", "blind": False, "since": T0 + 0.28}))
    )
    for t, x, yaw in POSES:
        node.subs[RAW_TOPIC][1](_raw(t, x, yaw))
    return node, node.pubs[VO_TOPIC].sent


def test_a_saccade_never_reaches_the_ekf_through_the_visual_odometry() -> None:
    """Without the gate the pan arrives as 0.3 rad of the cart's yaw and 5 cm of jump; with it the
    track stands still across the blind poses and the one after, and walks on with the cart."""
    _node, ungated = _drive(gate_on=False)
    last = ungated[-1].pose.pose
    assert last.position.x == pytest.approx(0.10)
    assert 2 * math.atan2(last.orientation.z, last.orientation.w) == pytest.approx(0.3)

    node, gated = _drive(gate_on=True)
    last = gated[-1].pose.pose
    # The first step and the last, nothing between: 1 cm each. The last is composed in SE(2)
    # (pepin.visual_odometry.VoTrack): the source moved 1 cm along its x while claiming yaw 0.3,
    # which in its own body frame is cos(0.3) forward and sin(0.3) to the right.
    assert last.position.x == pytest.approx(0.01 + 0.01 * math.cos(0.3))
    assert last.position.y == pytest.approx(-0.01 * math.sin(0.3))
    assert 2 * math.atan2(last.orientation.z, last.orientation.w) == pytest.approx(0.0)
    assert len(gated) == 3, "0.0, 0.1 and 0.6 went to the EKF"
    counts = node._tally.take().counts
    assert (counts["gaze_blind"], counts["gaze_after"]) == (3, 1)
    node._report()
    line = node.get_logger().texts("info")[-1]
    assert "gaze gate: 0 blind, 0 spinning of 0 frames withheld" in line  # a fresh window
    assert "head still since" in line
