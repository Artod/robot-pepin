"""One owner of the pose of a frame: the carry through the odometry, the camera in the map,
and a history that cannot say."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pepin.depth import rotation_matrix
from pepin.frame_pose import FramePoser, PoseHistory
from pepin.lean import LEAN_QUALITY_FLOOR, Lean, LeanGate
from pepin.tsdf import RigidPose


def _yaw(theta: float, x: float = 0.0, y: float = 0.0) -> RigidPose:
    """A planar pose: a yaw about the vertical axis and a position."""
    return RigidPose(
        rotation_matrix(0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2)),
        np.array([x, y, 0.0]),
    )


class FakeHistory:
    """The cart turning left at 0.1 rad/s in the odometry, offset in the map, its camera on a
    mount; nothing before t = 0."""

    def __init__(self) -> None:
        self.asked: list[tuple[float, str, str]] = []

    def pose_at(self, stamp: float, frame: str, fixed: str) -> RigidPose | None:
        self.asked.append((stamp, frame, fixed))
        if stamp < 0.0:
            return None
        base = _yaw(0.1 * stamp, 0.5 * stamp, 0.0)
        if fixed == "map":
            base = _yaw(0.1 * stamp + 1.0, 0.5 * stamp + 3.0, 2.0)
        if frame == "base_link":
            return base
        mount = RigidPose(np.eye(3), np.array([0.0, 0.0, 1.23]))  # the camera on the neck
        return RigidPose(
            base.rotation @ mount.rotation, base.rotation @ mount.translation + base.translation
        )


def test_a_scan_is_carried_through_the_odometry_to_the_frame_s_moment() -> None:
    """A point 2 m dead ahead at t 0; by t 1 the cart turned 0.1 rad left and moved 0.5 m
    forward, so the point sits 0.1 rad to the right and nearer, and the odometry frame is the
    one asked (the map's corrections must not tear a scan)."""
    history = FakeHistory()
    poser = FramePoser(history)
    ahead = np.array([[2.0, 0.0, 0.2]])
    moved = poser.carry(ahead, 0.0, 1.0)
    assert moved is not None
    world = np.array([2.0, 0.0, 0.2])  # the point in odom: the cart stood at the origin at t 0
    pose = _yaw(0.1, 0.5, 0.0)
    expected = pose.rotation.T @ (world - pose.translation)
    assert moved[0] == pytest.approx(expected, abs=1e-12)
    assert moved[0, 1] < 0.0 and moved[0, 0] < 2.0 and moved[0, 2] == pytest.approx(0.2)
    assert {fixed for _, _, fixed in history.asked} == {"odom"}
    assert poser.carry(ahead, -1.0, 1.0) is None  # before the history begins
    assert poser.motion(1.0, 1.0) is not None
    same = poser.carry(ahead, 1.0, 1.0)
    assert same is not None and same == pytest.approx(ahead, abs=1e-12)


def test_the_motion_between_two_stamps_is_asked_of_the_frame_the_caller_names() -> None:
    """The cart's own motion comes through the odometry, so no tracker correction tears a scan;
    the same question through the map (the baseline a parallax pair rests on, where the
    odometry's drift over a second IS the error) asks the map frame and answers a different
    transform. Neither can speak for a moment the history does not cover."""
    history = FakeHistory()
    poser = FramePoser(history)
    through_odom = poser.motion(0.0, 1.0)
    assert through_odom is not None
    assert {fixed for _, _, fixed in history.asked} == {"odom"}
    history.asked.clear()
    through_map = poser.map_motion(0.0, 1.0)
    assert through_map is not None
    assert {fixed for _, _, fixed in history.asked} == {"map"}
    assert {frame for _, frame, _ in history.asked} == {"base_link"}
    assert through_map.rotation == pytest.approx(through_odom.rotation, abs=1e-12)
    assert through_map.translation != pytest.approx(through_odom.translation, abs=1e-3)
    assert poser.map_motion(-1.0, 1.0) is None and poser.map_motion(0.0, -1.0) is None


def test_the_camera_and_the_cart_are_placed_in_the_map_at_the_stamp() -> None:
    poser = FramePoser(FakeHistory())
    camera = poser.camera_in_map(2.0)
    base = poser.base_in_map(2.0)
    assert camera is not None and base is not None
    assert base.translation == pytest.approx([4.0, 2.0, 0.0])
    assert camera.translation == pytest.approx([4.0, 2.0, 1.23])
    assert camera.rotation == pytest.approx(base.rotation)
    on_map = poser.to_map(np.array([[1.0, 0.0, 0.0]]), 2.0)
    assert on_map is not None
    assert on_map[0] == pytest.approx([4.0 + math.cos(1.2), 2.0 + math.sin(1.2), 0.0])
    assert poser.camera_in_map(-5.0) is None and poser.to_map(np.zeros((1, 3)), -5.0) is None
    on_cart = FramePoser(FakeHistory()).camera_in_base(2.0)  # the fake's mount
    assert on_cart is not None and on_cart.translation[2] == pytest.approx(1.23)
    assert FramePoser(FakeHistory()).camera_in_base(-1.0) is None


def test_the_frames_asked_for_are_the_poser_s_own_names() -> None:
    history = FakeHistory()
    poser = FramePoser(history, base="base", camera="cam", map_frame="world", odom_frame="odo")
    poser.camera_in_map(1.0)
    poser.carry(np.zeros((1, 3)), 0.0, 1.0)
    poser.camera_in_base(2.0)
    assert history.asked == [
        (1.0, "cam", "world"),
        (0.0, "base", "odo"),
        (1.0, "base", "odo"),
        (2.0, "cam", "base"),
    ]
    fake: PoseHistory = history  # the fake satisfies the protocol
    assert fake.pose_at(0.0, "base", "odo") is not None


class FakeLean:
    """A fixed lean at every stamp, and a count of who asked."""

    def __init__(self, roll_deg: float = 0.0, pitch_deg: float = 0.0) -> None:
        self.lean = Lean(math.radians(roll_deg), math.radians(pitch_deg), 0.0, 1.0)
        self.asked: list[float] = []

    def lean_at(self, stamp: float) -> Lean | None:
        self.asked.append(stamp)
        return Lean(self.lean.roll, self.lean.pitch, stamp, self.lean.quality)


class LevelHistory:
    """The cart at the map's origin, facing along x, its camera on the same spot: whatever the
    poser answers is the lean and nothing else."""

    def pose_at(self, stamp: float, frame: str, fixed: str) -> RigidPose | None:
        return RigidPose(np.eye(3), np.zeros(3))


def test_a_five_degree_pitch_moves_a_point_three_metres_out_by_the_geometric_amount() -> None:
    """The whole reason the lean exists: nose down by 5 degrees, a wall 3 m ahead drops
    3 sin 5 = 26 cm and comes 3 (1 - cos 5) = 1.1 cm nearer. The poser puts the lean under the
    planar pose, so the point lands where it really is."""
    lean = FakeLean(pitch_deg=5.0)
    poser = FramePoser(LevelHistory(), lean=lean, apply_lean=True)
    placed = poser.to_map(np.array([[3.0, 0.0, 0.0]]), 1.0)
    assert placed is not None
    assert placed[0, 2] == pytest.approx(-3.0 * math.sin(math.radians(5.0)))
    assert placed[0, 0] == pytest.approx(3.0 * math.cos(math.radians(5.0)))
    assert float(np.linalg.norm(placed[0] - np.array([3.0, 0.0, 0.0]))) == pytest.approx(
        2 * 3.0 * math.sin(math.radians(2.5)), abs=1e-9
    )
    # the camera hangs off base_link, so it swings with the body and its rotation carries the lean
    camera = poser.camera_in_map(1.0)
    assert camera is not None and camera.rotation == pytest.approx(lean.lean.rotation())
    assert lean.asked  # the lean was asked for at the frame's stamp, not read as "now"


def test_the_switch_off_is_the_pose_the_poser_has_always_answered() -> None:
    """``apply_lean`` off, or a source with nothing to say about that moment, and every answer
    is bit for bit the single lookup of before — no multiplication by a rotation of one."""
    history, lean = FakeHistory(), FakeLean(roll_deg=3.0, pitch_deg=-4.0)
    plain = FramePoser(FakeHistory())
    off = FramePoser(history, lean=lean, apply_lean=False)
    for stamp in (0.5, 2.0):
        for name in ("base_in_map", "camera_in_map"):
            want, got = getattr(plain, name)(stamp), getattr(off, name)(stamp)
            assert want is not None and got is not None
            assert np.array_equal(want.rotation, got.rotation)
            assert np.array_equal(want.translation, got.translation)
    assert not lean.asked  # switched off, the source is not even consulted
    assert history.asked == [(0.5, "base_link", "map"), (0.5, "camera_optical", "map")] * 1 + [
        (2.0, "base_link", "map"),
        (2.0, "camera_optical", "map"),
    ]
    carried = FramePoser(FakeHistory()).carry(np.array([[2.0, 0.0, 0.2]]), 0.0, 1.0)
    same = off.carry(np.array([[2.0, 0.0, 0.2]]), 0.0, 1.0)
    assert carried is not None and same is not None and np.array_equal(carried, same)


def test_a_lean_that_changes_between_two_stamps_rides_along_with_the_carry() -> None:
    """A scan taken while the cart was level, carried into a frame taken 5 degrees nose down:
    the point must come out 5 degrees higher in the frame's own axes, because the body under it
    tipped. Two leans, one on each side of the motion."""

    class Changing:
        def lean_at(self, stamp: float) -> Lean | None:
            return Lean(0.0, math.radians(5.0) if stamp > 0.5 else 0.0, stamp, 1.0)

    poser = FramePoser(LevelHistory(), lean=Changing(), apply_lean=True)
    moved = poser.carry(np.array([[3.0, 0.0, 0.0]]), 0.0, 1.0)
    assert moved is not None
    assert moved[0, 2] == pytest.approx(3.0 * math.sin(math.radians(5.0)))
    assert poser.motion(0.0, 1.0) is not None


def test_a_source_with_nothing_to_say_about_the_moment_leaves_the_pose_alone() -> None:
    """The IMU has no reading covering that stamp (the node just started, a gap in the stream):
    the poser answers the planar pose rather than nothing at all."""

    class Silent:
        def lean_at(self, stamp: float) -> Lean | None:
            return None

    poser = FramePoser(FakeHistory(), lean=Silent(), apply_lean=True)
    plain = FramePoser(FakeHistory())
    got, want = poser.camera_in_map(2.0), plain.camera_in_map(2.0)
    assert got is not None and want is not None
    assert np.array_equal(got.rotation, want.rotation)
    assert poser.lean_at(2.0) is None


def test_a_lean_gravity_never_voted_for_is_not_a_lean() -> None:
    """Gyro bias reports a tip nobody made — 3 degrees on a level floor at 0.2 deg/s, 6 at
    0.5 — and the one thing that tells it from a real tip is its quality: gravity has been
    refusing to agree with it for seconds (a real tip the gyro follows keeps quality 1.00,
    scratch/lean_quality_floor_probe.py). Below the floor the lean is unknown, not wrong — the
    pose is the planar one, and the scan gate, which asks the poser and not the estimator,
    admits the revolution instead of throwing a good scan away."""

    class Drifting:
        """The shape of a gyro bias in the estimator's output: a big lean nobody measured."""

        def lean_at(self, stamp: float) -> Lean | None:
            return Lean(0.0, math.radians(4.0), stamp, 0.02)

    poser = FramePoser(LevelHistory(), lean=Drifting(), apply_lean=True)
    assert poser.min_lean_quality == LEAN_QUALITY_FLOOR
    assert poser.lean_at(1.0) is None, "not believed, and so not applied"
    placed = poser.to_map(np.array([[3.0, 0.0, 0.0]]), 1.0)
    assert placed is not None and placed[0, 2] == pytest.approx(0.0), "placed level, as before"
    gate = LeanGate()
    assert gate.admits(poser.lean_at(1.0)) and gate.refused == 0, "the revolution is kept"
    believed = FramePoser(LevelHistory(), lean=Drifting(), apply_lean=True, min_lean_quality=0.0)
    assert believed.lean_at(1.0) is not None, "0: every lean is believed, as it was"
    assert not gate.admits(believed.lean_at(1.0)), "and believed, it would cost the scan too"


def test_a_history_without_the_camera_s_own_edge_answers_the_plain_lookup() -> None:
    """The split chain needs base_link <- camera_optical of its own. A history that has only
    the composed one (an old tape, a TF without the neck's edge) must still place the camera —
    unleaned, which is what it did before — instead of dropping the frame."""

    class NoCameraEdge(FakeHistory):
        def pose_at(self, stamp: float, frame: str, fixed: str) -> RigidPose | None:
            if fixed == "base_link":
                return None
            return super().pose_at(stamp, frame, fixed)

    poser = FramePoser(NoCameraEdge(), lean=FakeLean(pitch_deg=5.0), apply_lean=True)
    placed = poser.camera_in_map(2.0)
    plain = FramePoser(FakeHistory()).camera_in_map(2.0)
    assert placed is not None and plain is not None
    assert np.array_equal(placed.rotation, plain.rotation)
