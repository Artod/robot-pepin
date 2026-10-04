"""The VIO's frame feed under the ROS stubs: camera_stream's two eyes paired by their stamp, judged
by the head's own rate once OpenVINS has initialised, and the admitted pairs republished for it
(the left eye in grey)."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import ros_stubs

ros_stubs.install()

from pepin_bringup.msgs import image_from_array, stamp_from_seconds  # noqa: E402
from pepin_bringup.vio_feed import (  # noqa: E402
    HEAD_IMU_TOPIC,
    INIT_STALE_S,
    OV_POSE_TOPIC,
    VIO_LEFT_TOPIC,
    VIO_RIGHT_TOPIC,
    EyePairs,
    VioFeed,
    grey,
)

T0 = 1_759_500_000.0
RNG = np.random.default_rng(7)
LEFT = RNG.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
RIGHT = RNG.integers(0, 256, size=(6, 8), dtype=np.uint8)


class Clock:
    """The node's monotonic clock, moved by the test."""

    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _eyes(t: float) -> tuple[Any, Any]:
    stamp = stamp_from_seconds(T0 + t)
    return (
        image_from_array(LEFT, "bgr8", stamp, "camera_optical"),
        image_from_array(RIGHT, "mono8", stamp, "camera_optical"),
    )


def _node(initialised: bool = True, clock: Clock | None = None, **params: Any) -> VioFeed:
    """The node; ``initialised`` hands it one OpenVINS pose first (the gate's precondition)."""
    with ros_stubs.parameters(**params):
        node = VioFeed(clock=clock or Clock())
    if initialised:
        node.subs[OV_POSE_TOPIC][1](ros_stubs.PoseWithCovarianceStamped())
    return node


def _feed(node: VioFeed, t: float, right_first: bool = False) -> None:
    left, right = _eyes(t)
    calls = [("/camera/image", left), ("/camera/right/image", right)]
    for topic, msg in reversed(calls) if right_first else calls:
        node.subs[topic][1](msg)


def _head(node: VioFeed, t0: float, t1: float, deg_s: float) -> None:
    """Head IMU samples at 200 Hz over [t0, t1), turning at ``deg_s`` about a skew axis."""
    axis = np.array([0.6, -0.8, 0.0])
    for t in np.arange(t0, t1, 0.005):
        msg = ros_stubs.Imu()
        msg.header.stamp = stamp_from_seconds(T0 + float(t))
        w = msg.angular_velocity
        w.x, w.y, w.z = (float(v) for v in axis * math.radians(deg_s))
        node.subs[HEAD_IMU_TOPIC][1](msg)


def _sent_times(node: VioFeed) -> list[float]:
    stamps = [m.header.stamp for m in node.pubs[VIO_LEFT_TOPIC].sent]
    return [round(s.sec + s.nanosec * 1e-9 - T0, 2) for s in stamps]


def test_a_pair_goes_out_once_both_eyes_are_in_with_the_left_in_grey() -> None:
    node = _node()
    left, right = _eyes(0.0)
    node.subs["/camera/image"][1](left)
    assert not node.pubs[VIO_LEFT_TOPIC].sent, "half a pair waits for its other eye"
    node.subs["/camera/right/image"][1](right)
    (out,) = node.pubs[VIO_LEFT_TOPIC].sent
    assert node.pubs[VIO_RIGHT_TOPIC].sent == [right], "the grey right eye passes as it is"
    assert out.encoding == "mono8" and (out.width, out.height, out.step) == (8, 6, 8)
    assert out.header.stamp == left.header.stamp and out.header.frame_id == "camera_optical"
    expected = cv2.cvtColor(LEFT, cv2.COLOR_BGR2GRAY)
    assert np.array_equal(np.frombuffer(out.data, dtype=np.uint8).reshape(6, 8), expected)
    _feed(node, 0.1, right_first=True)
    assert len(node.pubs[VIO_LEFT_TOPIC].sent) == 2, "either eye may come first"


def test_a_pair_the_head_turned_fast_through_is_held_back_and_a_slow_sweep_passes() -> None:
    """head_rate_dps (60): a 290 deg/s saccade's pairs are held back, a 20 deg/s sweep's pass."""
    node = _node()
    _head(node, 0.0, 1.0, 20.0)  # the arbiter's slow sweep
    _head(node, 1.0, 1.4, 290.0)  # a saccade
    _head(node, 1.4, 2.5, 0.0)  # settled
    for t in (0.5, 0.9, 1.2, 1.35, 1.6, 2.0):
        _feed(node, t)
    # 1.2 and 1.35 are inside the saccade; 1.6 is past it with the window's 10 ms margin
    assert _sent_times(node) == [0.5, 0.9, 1.6, 2.0]
    assert len(node.pubs[VIO_RIGHT_TOPIC].sent) == 4
    node._report()
    line = node.logger.texts("info")[-1]
    assert "OpenVINS initialised, 0 passed ungated before it; 2 of 6 held back" in line
    assert "rate_gate=on" in line and "head_rate_dps=60.0" in line


def test_with_the_gate_off_every_pair_passes() -> None:
    node = _node(rate_gate=False)
    _head(node, 0.0, 2.0, 290.0)
    _feed(node, 1.0)
    assert len(node.pubs[VIO_LEFT_TOPIC].sent) == 1


def test_every_pair_passes_until_openvins_has_initialised_and_again_once_its_poses_stop() -> None:
    """OpenVINS's static init reads the first motion off its frames: before its first pose the
    gate stands aside; INIT_STALE_S without a pose (a restart) and it stands aside again."""
    clock = Clock()
    node = _node(initialised=False, clock=clock)
    _head(node, 0.0, 2.0, 290.0)
    _feed(node, 1.0)
    assert len(node.pubs[VIO_LEFT_TOPIC].sent) == 1, "a fast pair passes before the init"
    node._report()
    assert "OpenVINS NOT initialised, 1 passed ungated before it" in node.logger.texts("info")[-1]
    node.subs[OV_POSE_TOPIC][1](ros_stubs.PoseWithCovarianceStamped())
    _feed(node, 1.1)
    assert len(node.pubs[VIO_LEFT_TOPIC].sent) == 1, "initialised: the fast pair is held"
    clock.t += INIT_STALE_S + 0.1
    _feed(node, 1.2)
    assert len(node.pubs[VIO_LEFT_TOPIC].sent) == 2, "no pose for INIT_STALE_S: ungated again"


def test_the_rate_is_live() -> None:
    node = _node()
    _head(node, 0.0, 2.0, 100.0)
    _feed(node, 0.5)
    assert not node.pubs[VIO_LEFT_TOPIC].sent
    assert node.set_parameters([ros_stubs.Parameter("head_rate_dps", value=150.0)])[0].successful
    _feed(node, 1.0)
    assert len(node.pubs[VIO_LEFT_TOPIC].sent) == 1


def test_a_half_that_never_finds_its_eye_is_dropped_and_counted() -> None:
    pairs = EyePairs(keep=2)
    assert pairs.add("left", 1, "l1") is None
    assert pairs.add("left", 2, "l2") is None
    assert pairs.add("left", 3, "l3") is None
    assert pairs.unpaired == 1
    assert pairs.add("right", 1, "r1") is None, "its left half is gone"
    assert pairs.add("right", 3, "r3") == ("l3", "r3")


def test_grey_refuses_an_encoding_it_does_not_speak() -> None:
    left, right = _eyes(0.0)
    assert grey(right) is right
    left.encoding = "rgb16"
    try:
        grey(left)
    except ValueError as error:
        assert "rgb16" in str(error)
    else:
        raise AssertionError("an unknown encoding must be refused")
