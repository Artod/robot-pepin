"""The volume read out as the costmap's marks: what a bearing answers and what it refuses to.

The wall of this file is the plane x = 2 m, painted by a camera standing at the origin (a
constant depth image is exactly that plane), and the cart reads it back over the whole turn. What
is checked is what the costmap depends on: the range and the bearing of a real surface under a
moved and turned cart, silence where the model holds nothing, the band, and the one case this
topic exists for — a false observation that later frames look through marks nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pepin.depth import Intrinsics
from pepin.tsdf import GridSpec, RigidPose, Tsdf
from pepin.volume_scan import (
    MARKS_BINS,
    MarksLaw,
    empty_marks,
    marks_box,
    marks_ranges,
    marks_window,
)

INTR = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=45.0, width=160, height=90)
WALL_X = 2.0
CAMERA_Z = 0.6


def spec() -> GridSpec:
    """A 6 x 6 m room 1.7 m tall on the robot's own 5 cm lattice, the cart at its centre."""
    return GridSpec(origin=(-3.0, -3.0, -0.15), shape=(120, 120, 34))


def optical(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> RigidPose:
    """A level camera at (x, y, CAMERA_Z) looking along ``yaw``: map <- optical."""
    c, s = math.cos(yaw), math.sin(yaw)
    forward, left, up = (
        np.array([c, s, 0.0]),
        np.array([-s, c, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    return RigidPose(np.stack([-left, -up, forward], axis=1), np.array([x, y, CAMERA_Z]))


def base(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> RigidPose:
    """The cart standing at (x, y) with heading ``yaw``: map <- base_link, on the floor."""
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return RigidPose(rotation, np.array([x, y, 0.0]))


def wall_depth(blob: float | None = None) -> np.ndarray:
    """The depth image of the plane x = WALL_X seen by a camera at the origin — a constant
    2.0 m, since the depth of a pixel is its distance ALONG the optical axis — with ``blob``
    metres written into a handful of pixels straight ahead when one is asked for: the wrong
    disparity SGBM answers on a herringbone floor, as one frame sees it."""
    depth = np.full((INTR.height, INTR.width), WALL_X)
    if blob is not None:
        # a patch 20 x 30 px wide: at 1 m that is 10 cm by 15 cm, two voxel columns and three
        # rows of them — a blob smaller than a voxel is a blob the volume never hears about
        depth[30:60, 70:90] = blob
    return depth


def painted(frames: int = 3, blob_first: bool = False) -> Tsdf:
    """A volume with the wall integrated ``frames`` times from the origin; with ``blob_first``
    the FIRST frame also carries a blob at 1 m, which every later frame looks through."""
    volume = Tsdf(spec())
    pose = optical()
    for i in range(frames):
        volume.integrate(wall_depth(1.0 if blob_first and i == 0 else None), None, INTR, pose)
    return volume


def at_deg(ranges: np.ndarray, degrees: float, window_deg: float = 1.5) -> float:
    """The nearest range the fan holds within ``window_deg`` of one bearing, degrees CCW from
    the cart's x; NaN when it holds none.

    A window and not a single bin, because the fan's bins are finer than the room's lattice: at
    2 m two neighbouring voxel columns are 1.4 degrees apart, so two bins in three are empty by
    construction. The costmap does not mind — every ray it marks with ends in the very cell the
    surface is in — but a test that asked one bin would be asking where a 5 cm cell happens to
    fall.
    """
    step = MarksLaw().step
    middle = round((math.radians(degrees) + math.pi) / step)
    half = max(1, round(math.radians(window_deg) / step))
    window = ranges[[(middle + k) % MARKS_BINS for k in range(-half, half + 1)]]
    return float(np.nanmin(window)) if np.isfinite(window).any() else math.nan


def test_a_wall_reads_back_at_its_own_range_and_bearing() -> None:
    """The fan is a fan: every bearing carries the distance to the wall ALONG that bearing, so a
    flat wall 2 m ahead reads 2.0 / cos(bearing), within the half voxel the lattice quantises a
    crossing to."""
    ranges = marks_ranges(painted(), base())
    assert at_deg(ranges, 0.0) == pytest.approx(WALL_X, abs=0.05)
    for degrees in (-15.0, -5.0, 5.0, 15.0):
        expected = WALL_X / math.cos(math.radians(degrees))
        assert at_deg(ranges, degrees) == pytest.approx(expected, abs=0.05), degrees
    # ...and the camera saw only its own +-22 degrees: everywhere else the model holds nothing
    # and the fan says NaN, which a costmap neither marks nor clears.
    assert math.isnan(at_deg(ranges, 90.0)) and math.isnan(at_deg(ranges, 180.0))
    assert np.count_nonzero(np.isfinite(ranges)) < MARKS_BINS // 4


def test_the_cart_s_own_pose_carries_the_fan() -> None:
    """The volume is in the map and the scan is in base_link: a cart half a metre nearer and
    turned 30 degrees must read the same wall at the range and bearing IT sees it at, or every
    mark lands somewhere the room is not."""
    volume = painted()
    ranges = marks_ranges(volume, base(0.5, 0.0))
    assert at_deg(ranges, 0.0) == pytest.approx(WALL_X - 0.5, abs=0.05)

    turned = marks_ranges(volume, base(0.0, 0.0, math.radians(30.0)))
    assert math.isnan(at_deg(turned, 0.0)), "the head is turned away from what was painted"
    assert at_deg(turned, -30.0) == pytest.approx(WALL_X, abs=0.05)
    assert at_deg(turned, -20.0) == pytest.approx(WALL_X / math.cos(math.radians(10.0)), abs=0.05)

    # ...and a cart standing off to one side reads the same wall obliquely: 1 m to the left of
    # the wall's foot, the nearest point of it is still 2 m ahead in the map.
    aside = marks_ranges(volume, base(0.0, -0.5))
    assert at_deg(aside, math.degrees(math.atan2(0.5, WALL_X))) == pytest.approx(
        math.hypot(WALL_X, 0.5), abs=0.06
    )


def test_a_surface_under_min_weight_is_not_one() -> None:
    """The criterion is the volume's own (pepin.tsdf.Tsdf.surface, what /fusion/surface draws at
    depth_fusion's min_weight), and it is a criterion about AGREEMENT: one observation of this
    wall from 2 m weighs 1.0 (GridSpec.observation_weight: (2.0 / d)^2), so at min_weight 2 a
    single frame marks nothing at all and two frames mark the wall."""
    once = marks_ranges(painted(frames=1), base())
    assert not np.isfinite(once).any(), "one frame from 2 m weighs 1.0, under min_weight 2"
    twice = marks_ranges(painted(frames=2), base())
    assert at_deg(twice, 0.0) == pytest.approx(WALL_X, abs=0.05)
    # ...and lowering the criterion shows the same wall the one frame did paint
    lowered = marks_ranges(painted(frames=1), base(), MarksLaw(min_weight=0.5))
    assert at_deg(lowered, 0.0) == pytest.approx(WALL_X, abs=0.05)


def test_a_false_observation_the_next_frames_look_through_marks_nothing() -> None:
    """THE WHOLE POINT OF THIS TOPIC. One frame puts a blob at 1 m where the floor is — the
    wrong disparity of a stereo head on a parquet — and in a single frame's fan that is a lethal
    cell. In the volume it is one weak opinion: every later frame measures 2 m through the same
    voxels, the weighted average carries them back to free space, and the blob stops being a
    surface. The wall behind it goes on being one."""
    straight = 0.0
    fresh = marks_ranges(painted(frames=1, blob_first=True), base(), MarksLaw(min_weight=0.5))
    assert at_deg(fresh, straight) == pytest.approx(1.0, abs=0.05), "one frame believes it"
    for frames, marks in ((2, True), (3, True), (5, False), (8, False)):
        ranges = marks_ranges(painted(frames=frames, blob_first=True), base())
        near = at_deg(ranges, straight)
        if marks:
            assert near == pytest.approx(1.0, abs=0.2), frames
        else:
            assert near == pytest.approx(WALL_X, abs=0.05), (
                f"{frames} frames: the blob is carved away and the wall behind it marks"
            )


def test_the_band_is_what_may_mark_at_all() -> None:
    """The fan reads a height band above the cart's own floor plane, the band /depth_scan marks
    in: a model that holds a surface only outside it says nothing."""
    volume = painted()
    inside = marks_ranges(volume, base(), MarksLaw(band_m=(0.15, 1.30)))
    assert np.isfinite(inside).any()
    # the camera of this file paints 0.15-1.05 m of the wall; a band above that is empty
    above = marks_ranges(volume, base(), MarksLaw(band_m=(1.20, 1.30)))
    assert not np.isfinite(above).any()
    # ...and the band travels with the cart: a body standing a metre up reads its own band
    lifted = RigidPose(base().rotation, np.array([0.0, 0.0, 1.0]))
    assert not np.isfinite(marks_ranges(volume, lifted)).any()


def test_nothing_beyond_the_fan_s_reach_and_nothing_outside_the_volume() -> None:
    """A range the fan does not claim, and a cart the volume does not contain: both are silence,
    not a mark at the edge."""
    volume = painted()
    near = marks_ranges(volume, base(), MarksLaw(range_m=1.0))
    assert not np.isfinite(near).any(), "the wall is 2 m away and the fan reaches 1"
    assert marks_box(volume.spec, (40.0, 0.0), (0.15, 1.30), 3.0) is None
    assert marks_window(volume, base(40.0, 0.0)) is None
    assert not np.isfinite(empty_marks()).any() and empty_marks().size == MARKS_BINS


def test_the_window_answers_what_the_whole_volume_answers() -> None:
    """The node copies the cart's neighbourhood under the model's lock and reads the fan out of
    the copy: the copy is the same voxels on their own grid, so it must be the same fan, bin for
    bin — otherwise the costmap and /fusion/surface would be two different rooms."""
    volume = painted()
    pose = base(0.3, -0.2, math.radians(20.0))
    window = marks_window(volume, pose)
    assert window is not None
    assert window.spec.shape[0] < volume.spec.shape[0], "a neighbourhood, not the room"
    np.testing.assert_array_equal(marks_ranges(window, pose), marks_ranges(volume, pose))


def test_the_marks_are_the_surface_fusion_publishes_and_nothing_else() -> None:
    """One criterion, two consumers: every range in the fan is the distance to a point of
    Tsdf.surface at that bearing — the cloud /fusion/surface carries — and no bearing answers
    nearer than the nearest such point."""
    volume = painted()
    pose = base()
    ranges = marks_ranges(volume, pose, MarksLaw())
    points, _colours = volume.surface(MarksLaw().min_weight)
    band = (points[:, 2] >= 0.15) & (points[:, 2] <= 1.30)
    reach = np.hypot(points[:, 0], points[:, 1]) <= MarksLaw().range_m
    kept = points[band & reach]
    assert kept.shape[0] > 0
    assert np.nanmin(ranges) == pytest.approx(
        float(np.hypot(kept[:, 0], kept[:, 1]).min()), abs=1e-9
    )
    assert np.nanmax(ranges) <= float(np.hypot(kept[:, 0], kept[:, 1]).max()) + 1e-9
