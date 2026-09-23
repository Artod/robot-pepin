"""The volume read out as the costmap's marks: what a bearing answers and what it refuses to.

The wall of this file is the plane x = 2 m, painted by a camera standing at the origin (a
constant depth image is exactly that plane), and the cart reads it back over the whole turn. What
is checked is what the costmap depends on: the range and the bearing of a real surface under a
moved and turned cart, silence where the model holds nothing, the band, and the one case this
topic exists for — a false observation that later frames look through marks nothing.

The second half of the file is the fan's CLEARING answer (``free_ranges``, ``/depth_free`` behind
depth_fusion's ``marks_clear``): how far the same walk says the volume is known open, that it
stops at the wall rather than through it, that a bearing nobody looked along is never cleared —
and that the marking fan comes out of the same window bit for bit as it always did, which is what
makes the flag a switch on the publisher and nothing else.
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
    fan_counts,
    free_ranges,
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


# ---- the clearing half: where the volume says it is OPEN -----------------------------------


def free(volume: Tsdf, at: RigidPose, law: MarksLaw | None = None) -> np.ndarray:
    """The fan's free ranges, with this file's default law."""
    return free_ranges(volume, at, law if law is not None else MarksLaw())


def test_a_bearing_with_a_wall_clears_up_to_it_and_not_past_it() -> None:
    """THE POINT OF THE CLEARING HALF. The volume holds the wall at 2 m and the free space the
    camera's rays carved on the way to it; the fan must offer that free space for raytracing and
    stop there, or a costmap would erase the wall it has just been told about.

    One voxel of slack on either side: the walk ends at the last column it saw OPEN, and the
    column in front of a surface is the surface's own halo, which is not open (SliceLaw)."""
    volume = painted()
    marks, open_to = marks_ranges(volume, base()), free(volume, base())
    ahead = at_deg(marks, 0.0)
    assert ahead == pytest.approx(WALL_X, abs=0.05)
    reach = at_deg(open_to, 0.0)
    assert WALL_X - 0.30 <= reach < ahead, "clear up to the wall, never through it"
    # ...and where the camera never looked, the volume vouches for nothing: NaN clears nothing.
    assert math.isnan(at_deg(open_to, 90.0)) and math.isnan(at_deg(open_to, 180.0))


def test_an_open_bearing_answers_how_far_the_volume_looked() -> None:
    """A bearing with free voxels and no surface is the case the marks fan has no word for: it
    answers "free to X", X being the last column the camera carved, and the ray may be cleared
    exactly that far. A wall only 1 m away moves X with it."""
    near = Tsdf(spec())
    for _ in range(4):  # a wall at 1.0 m: the carved run is shorter and so is the answer
        near.integrate(np.full((INTR.height, INTR.width), 1.0), None, INTR, optical())
    far_law = MarksLaw(range_m=3.0)
    assert at_deg(free(near, base(), far_law), 0.0) == pytest.approx(1.0, abs=0.3)
    assert at_deg(free(painted(), base(), far_law), 0.0) == pytest.approx(WALL_X, abs=0.3)
    # the fan's own reach caps it: a volume carved to 2 m read out to 1 m says at most 1 m
    short = free(painted(), base(), MarksLaw(range_m=1.0))
    assert np.nanmax(short) <= 1.0 + 1e-9


def test_a_bearing_nobody_looked_along_is_never_cleared() -> None:
    """Unknown must stay unknown: a costmap that cleared it would forget a cell no camera has
    contradicted. A cart outside its own volume says nothing at all on every bearing."""
    volume = painted()
    outside = free(volume, base(40.0, 0.0))
    assert not np.isfinite(outside).any() and outside.size == MARKS_BINS
    # ...and a band the volume holds nothing in is the same silence
    above = free(volume, base(), MarksLaw(band_m=(1.20, 1.30)))
    assert not np.isfinite(above).any()


def test_the_criterion_is_the_volume_s_own_and_a_weak_voxel_vouches_for_nothing() -> None:
    """The clearing carries the same ``min_weight`` the marks do (one criterion, three answers):
    a single frame's carved space is not agreement, and at the shipped weight it clears nothing.
    Lowering the criterion shows the very space that one frame did carve."""
    once = painted(frames=1)
    assert not np.isfinite(free(once, base())).any(), "one frame from 2 m weighs 1.0"
    lowered = free(once, base(), MarksLaw(min_weight=0.5))
    assert at_deg(lowered, 0.0) == pytest.approx(WALL_X, abs=0.3)


def test_the_marks_fan_is_bit_for_bit_what_it_always_was() -> None:
    """CLAUDE.md rule 19's other half: the clearing is a SECOND answer beside the marks, never a
    change to them. The two are computed from the same window and the marking fan must come out
    of it identical, bin for bin, to the one the costmap has been marking from since
    2026-09-21 — which is what lets ``marks_clear`` be a live switch on the publisher alone."""
    volume = painted(frames=5, blob_first=True)
    pose = base(0.2, -0.1, math.radians(15.0))
    before = marks_ranges(volume, pose)
    free(volume, pose)  # the clearing walk touches nothing
    np.testing.assert_array_equal(marks_ranges(volume, pose), before)
    counted = fan_counts(before, free(volume, pose))
    assert counted[0] == int(np.isfinite(before).sum())
    assert sum(counted) >= MARKS_BINS, "every bearing marks, clears, or says nothing"


def test_the_window_clears_what_the_whole_volume_clears() -> None:
    """The node reads the fan out of a copy of the cart's neighbourhood taken under the model's
    lock; the clearing half must survive that copy exactly as the marks do, or the costmap and
    the model would be two different rooms."""
    volume = painted()
    pose = base(0.3, -0.2, math.radians(20.0))
    window = marks_window(volume, pose)
    assert window is not None
    np.testing.assert_array_equal(free(window, pose), free(volume, pose))
