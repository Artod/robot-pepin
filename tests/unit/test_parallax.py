"""Depth from the cart's own motion: the triangulation is exact on geometry it is handed, the
tracker recovers a known depth from two rendered views to within a percent, the degenerate
motions say so instead of inventing numbers, and the anchor is off until its flag is on."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pepin.depth import CameraPose, Intrinsics, project_all
from pepin.depth_pipeline import Frame, FrameContext, ParallaxAnchor, standard_pipeline
from pepin.parallax import (
    MAX_SAMPSON_PX,
    CameraPlacement,
    Motion,
    camera_motion,
    match,
    parallax_truth,
    perpendicular_baseline,
    sampson,
    to_gray,
    triangulate,
)
from pepin.tsdf import RigidPose

INTR = Intrinsics(fx=400.0, fy=400.0, cx=320.0, cy=180.0, width=640, height=360)
BANDS = ((0, 120, 4.0), (120, 240, 2.0), (240, 360, 4.0 / 3.0))  # rows and the plane's depth
SIDESTEP_M = 0.10  # the camera's own displacement: 10, 20 and 30 px of disparity on the bands
MARGIN = 128  # spare columns either side of the rendered view, so a band may shift into them


def texture(seed: int = 7) -> np.ndarray:
    """A rough indoor-like image: white noise blurred by a box filter, so corners are sharp but
    not single pixels (a single-pixel corner tracks to the wrong place at sub-pixel accuracy)."""
    rng = np.random.default_rng(seed)
    fine = rng.integers(0, 255, size=(INTR.height, INTR.width + 2 * MARGIN), dtype=np.int64)
    k = 3
    blurred = fine.astype(float)
    for axis in (0, 1):
        stack = [np.roll(blurred, s, axis=axis) for s in range(-k, k + 1)]
        blurred = np.mean(stack, axis=0)
    return np.rint(blurred).astype(np.uint8)


def rendered_pair(sidestep_m: float = SIDESTEP_M) -> tuple[np.ndarray, np.ndarray, Motion]:
    """Two views of three fronto-parallel textured planes, the camera stepping ``sidestep_m``
    to its right between them: each band of rows shifts by exactly ``fx * sidestep / z``
    pixels, which the bands are chosen to make whole. Returns (view A, view B, the motion)."""
    wide = texture()
    a = np.ascontiguousarray(wide[:, MARGIN:-MARGIN])
    b = np.empty_like(a)
    for top, bottom, z in BANDS:
        shift = round(INTR.fx * sidestep_m / z)
        b[top:bottom] = wide[top:bottom, MARGIN + shift : MARGIN + shift + INTR.width]
    return a, b, Motion(np.eye(3), np.array([-sidestep_m, 0.0, 0.0]))


def planar_pose(x: float, y: float, yaw: float) -> RigidPose:
    """A cart pose on the floor as a rigid transform (odom <- base_link)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return RigidPose(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.array([x, y, 0.0]))


def base_motion(before: RigidPose, after: RigidPose) -> RigidPose:
    """base_link at ``before`` into base_link at ``after``, the way FramePoser.motion builds it."""
    back = after.inverse()
    return RigidPose(
        back.rotation @ before.rotation, back.rotation @ before.translation + back.translation
    )


class Odometry:
    """A motion source over two known cart poses, the fake a node's FramePoser stands in for."""

    def __init__(self, poses: dict[float, RigidPose]) -> None:
        self._poses = poses

    def motion(self, from_stamp: float, to_stamp: float) -> RigidPose | None:
        """How base_link moved between the two stamps, or ``None`` for a stamp it has no pose at."""
        if from_stamp not in self._poses or to_stamp not in self._poses:
            return None
        return base_motion(self._poses[from_stamp], self._poses[to_stamp])


# ---- the geometry, with no tracker in the way ---------------------------------------------
def test_triangulation_is_exact_on_correspondences_the_projection_itself_produced() -> None:
    """A scene of base_link points seen from two cart poses: the depths triangulated from the
    odometry's transform are the depths the projection used, to a tenth of a millimetre at
    three metres (the float32 pixels' own rounding) — so any error in a real measurement is
    the tracker's or the odometry's, never the geometry here."""
    rng = np.random.default_rng(3)
    points = np.stack(
        [rng.uniform(1.0, 5.0, 200), rng.uniform(-2.0, 2.0, 200), rng.uniform(0.0, 1.8, 200)],
        axis=1,
    )
    cam = CameraPose(0.05, 0.0, 1.23, math.radians(26.0))
    before, after = planar_pose(0.0, 0.0, 0.0), planar_pose(0.22, 0.04, math.radians(5.0))
    in_a = (points - before.translation) @ before.rotation
    in_b = (points - after.translation) @ after.rotation
    u_a, v_a, fwd_a = project_all(in_a, cam, INTR)
    u_b, v_b, fwd_b = project_all(in_b, cam, INTR)
    seen = (fwd_a > 0.3) & (fwd_b > 0.3)
    moved = base_motion(before, after)
    motion = camera_motion(
        moved.rotation, moved.translation, CameraPlacement.of(cam), CameraPlacement.of(cam)
    )
    pts_a = np.stack([u_a[seen], v_a[seen]], axis=1).astype(np.float32)
    pts_b = np.stack([u_b[seen], v_b[seen]], axis=1).astype(np.float32)
    z, sigma = triangulate(pts_a, pts_b, INTR, motion)
    assert seen.sum() > 50
    assert np.allclose(z, fwd_b[seen], rtol=1e-4, atol=0.0)
    assert np.all(sigma > 0) and np.all(np.isfinite(sigma))


def neck_edge(pitch: float, pan: float, height: float = 1.23) -> CameraPlacement:
    """``base_link <- camera_optical`` for a head pitched down by ``pitch`` and panned left by
    ``pan`` — TF's own edge, the pose :class:`CameraPose` cannot hold."""
    c, s = math.cos(pitch), math.sin(pitch)
    optical_from_level = np.array([[0.0, -1.0, 0.0], [-s, 0.0, -c], [c, 0.0, -s]])
    cp, sp = math.cos(pan), math.sin(pan)
    turn = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])  # base_link <- panned base
    return CameraPlacement((optical_from_level @ turn.T).T, np.array([0.05, 0.0, height]))


def seen_by(
    points: np.ndarray, base: RigidPose, place: CameraPlacement
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where odom points land in the picture of a camera placed by ``place`` on a cart standing
    at ``base``: columns, rows and the depth along the optical axis."""
    local = (points - base.translation) @ base.rotation
    optical = (local - place.translation) @ place.rotation
    return (
        INTR.fx * optical[:, 0] / optical[:, 2] + INTR.cx,
        INTR.fy * optical[:, 1] / optical[:, 2] + INTR.cy,
        optical[:, 2],
    )


@pytest.mark.parametrize("pan_a_deg, pan_b_deg", [(0.0, 0.0), (20.0, 20.0), (0.0, 5.0)])
def test_the_motion_carries_the_neck_s_pan_and_a_pitch_only_pose_does_not(
    pan_a_deg: float, pan_b_deg: float
) -> None:
    """A head panned 20 degrees, and a head that pans 5 degrees between the two pictures: the
    depths come back exact, because the motion is built from TF's whole edge at each stamp.
    Built from the pitch-only pose the pipeline projects with, the same correspondences either
    die on the epipolar gate (a pan that changes) or survive it carrying a depth that is not
    the truth (a pan that stands) — the defect this parametrisation exists for."""
    rng = np.random.default_rng(5)
    points = np.stack(
        [rng.uniform(1.0, 5.0, 400), rng.uniform(-2.0, 2.0, 400), rng.uniform(0.0, 1.8, 400)],
        axis=1,
    )
    pitch = math.radians(26.0)
    before, after = planar_pose(0.0, 0.0, 0.0), planar_pose(0.12, 0.0, 0.0)
    place_a = neck_edge(pitch, math.radians(pan_a_deg))
    place_b = neck_edge(pitch, math.radians(pan_b_deg))
    u_a, v_a, z_a = seen_by(points, before, place_a)
    u_b, v_b, z_b = seen_by(points, after, place_b)
    seen = (z_a > 0.3) & (z_b > 0.3)
    for u, v in ((u_a, v_a), (u_b, v_b)):
        seen &= (u >= 0) & (u < INTR.width) & (v >= 0) & (v < INTR.height)
    pts_a = np.stack([u_a[seen], v_a[seen]], axis=1).astype(np.float32)
    pts_b = np.stack([u_b[seen], v_b[seen]], axis=1).astype(np.float32)
    moved = base_motion(before, after)
    motion = camera_motion(moved.rotation, moved.translation, place_a, place_b)
    assert seen.sum() > 100
    assert np.median(sampson(pts_a, pts_b, INTR, motion)) < 1e-3  # float32 pixels' own
    assert np.allclose(triangulate(pts_a, pts_b, INTR, motion)[0], z_b[seen], rtol=1e-4)
    flat = CameraPlacement.of(CameraPose(0.05, 0.0, 1.23, pitch))  # what a CameraPose can say
    blind = camera_motion(moved.rotation, moved.translation, flat, flat)
    survivors = sampson(pts_a, pts_b, INTR, blind) <= MAX_SAMPSON_PX
    if pan_a_deg == pan_b_deg == 0.0:
        assert np.all(survivors), "with no pan the pitch-only pose is the same pose"
        return
    # NaNs among them are the depths that came out behind a lens: gated out, still not truth
    ratio = triangulate(pts_a, pts_b, INTR, blind)[0][survivors] / z_b[seen][survivors]
    assert survivors.sum() == 0 or abs(float(np.nanmedian(ratio)) - 1.0) > 0.1


def test_the_sigma_grows_with_the_square_of_the_range_and_falls_with_the_baseline() -> None:
    """Twice the distance is four times the uncertainty; twice the baseline is half of it."""
    motion = Motion(np.eye(3), np.array([-0.1, 0.0, 0.0]))
    wide = Motion(np.eye(3), np.array([-0.2, 0.0, 0.0]))

    def sigma_at(z: float, m: Motion) -> float:
        disparity = INTR.fx * float(np.linalg.norm(m.translation)) / z
        a = np.array([[300.0, 200.0]], dtype=np.float32)
        b = np.array([[300.0 - disparity, 200.0]], dtype=np.float32)
        return float(triangulate(a, b, INTR, m)[1][0])

    assert sigma_at(4.0, motion) / sigma_at(2.0, motion) == pytest.approx(4.0, rel=0.02)
    assert sigma_at(2.0, wide) / sigma_at(2.0, motion) == pytest.approx(0.5, rel=0.02)


# ---- the tracker on rendered views --------------------------------------------------------
def test_a_ten_centimetre_sidestep_recovers_two_metres_within_one_percent() -> None:
    """Three textured planes at 1.33, 2 and 4 m, the camera stepping 10 cm to its right: every
    band's depth comes back within a percent of what was rendered."""
    a, b, motion = rendered_pair()
    truth = parallax_truth(a, b, INTR, motion)
    assert truth.kept >= 100
    rows = truth.points[:, 1]
    for top, bottom, z in BANDS:
        inside = (rows >= top + 16) & (rows < bottom - 16)
        assert int(inside.sum()) >= 10, f"band {z} m has {int(inside.sum())} pairs"
        ratio = float(np.median(truth.z[inside] / z))
        assert abs(ratio - 1.0) < 0.01, f"band {z} m came back at {ratio:.3f} of its depth"


def test_the_flow_returns_the_same_corners_it_was_given() -> None:
    """The match itself: every kept correspondence sits on the sidestep's own disparity."""
    a, b, _motion = rendered_pair()
    pts_a, pts_b = match(a, b)
    assert len(pts_a) >= 100
    middle = (pts_a[:, 1] >= 136) & (pts_a[:, 1] < 224)  # the 2 m band, away from its seams
    shift = pts_a[middle, 0] - pts_b[middle, 0]
    assert float(np.median(shift)) == pytest.approx(INTR.fx * SIDESTEP_M / 2.0, abs=0.2)


def test_a_pure_rotation_yields_no_pairs_and_says_so() -> None:
    """Turning on the spot moves no camera centre: there is no baseline, and the anchor must
    say ``rotation-only`` rather than triangulate rays that meet nowhere."""
    a, b, _ = rendered_pair()
    yaw = math.radians(8.0)
    spin = Motion(
        np.array(
            [
                [math.cos(yaw), 0.0, math.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-math.sin(yaw), 0.0, math.cos(yaw)],
            ]
        ),
        np.zeros(3),
    )
    truth = parallax_truth(a, b, INTR, spin)
    assert truth.kept == 0 and truth.verdict == "rotation-only"


def test_standing_still_says_still() -> None:
    """No motion at all is not a rotation: the word for it is ``still``."""
    a, b, _ = rendered_pair()
    truth = parallax_truth(a, b, INTR, Motion(np.eye(3), np.zeros(3)))
    assert truth.kept == 0 and truth.verdict == "still"


def test_points_that_do_not_obey_the_given_motion_are_thrown_away() -> None:
    """The same views handed a motion the cart did not make (a step up instead of a step
    right): every correspondence misses that motion's epipolar lines and none survives."""
    a, b, _ = rendered_pair()
    wrong = Motion(np.eye(3), np.array([0.0, -SIDESTEP_M, 0.0]))
    truth = parallax_truth(a, b, INTR, wrong)
    assert truth.kept == 0
    assert truth.rejected["epipolar"] > 100


def test_a_ray_on_the_epipole_sees_none_of_the_baseline() -> None:
    """Driving straight ahead, the pixel on the direction of travel never moves however far the
    cart goes: its perpendicular baseline is zero, which is the gate that drops it — while a
    pixel off to the side sees the whole 30 cm."""
    forward = Motion(np.eye(3), np.array([0.0, 0.0, -0.3]))
    epipole = np.array([[INTR.cx, INTR.cy]], dtype=np.float32)
    aside = np.array([[INTR.cx + INTR.fx, INTR.cy]], dtype=np.float32)  # 45 degrees off the axis
    assert float(perpendicular_baseline(epipole, INTR, forward)[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(perpendicular_baseline(aside, INTR, forward)[0]) == pytest.approx(
        0.3 * math.sin(math.radians(45.0)), rel=1e-6
    )


# ---- the anchor -----------------------------------------------------------------------------
def context(
    stamp: float, gray: np.ndarray, odometry: Odometry, place: CameraPlacement | None = None
) -> FrameContext:
    """A frame's context carrying the picture and the odometry the parallax anchor reads, and
    TF's camera edge when a test has a neck to turn."""
    return FrameContext(
        INTR,
        CameraPose(0.0, 0.0, 1.23, math.radians(26.0)),
        stamp=stamp,
        gray=gray,
        motion=odometry,
        cam_optical=place,
    )


def band_error(pairs: Any) -> float:
    """How far the anchor's depths sit from the rendered planes they came off, as a median
    relative error (the row each pair came from is its lift, inverted)."""
    rows = INTR.cy - np.asarray(pairs.lift) * INTR.fy
    truth = np.select([rows < BANDS[0][1], rows < BANDS[1][1]], [BANDS[0][2], BANDS[1][2]], 4 / 3)
    return float(np.median(np.abs(np.asarray(pairs.z) / truth - 1.0)))


def test_the_anchor_pairs_the_network_s_depth_with_the_triangulated_one() -> None:
    """The first frame only fills the anchor's memory; the second pairs the network's depth at
    the tracked corners with the depth the motion measured, each with its own weight and the
    lift of its row."""
    a, b, _ = rendered_pair()
    poses = {1.0: planar_pose(0.0, 0.0, 0.0), 1.2: planar_pose(0.0, -SIDESTEP_M, 0.0)}
    odometry = Odometry(poses)
    anchor = ParallaxAnchor()
    network = np.full((INTR.height, INTR.width), 3.0)
    assert anchor.pairs(Frame(network, context(1.0, a, odometry))) is None
    pairs = anchor.pairs(Frame(network, context(1.2, b, odometry)))
    assert pairs is not None and pairs.size >= 50
    assert pairs.d.shape == pairs.z.shape == pairs.weight.shape == pairs.lift.shape
    assert np.all(pairs.d == 3.0) and np.all(pairs.z > 0.5) and np.all(pairs.z < 6.0)
    assert np.all(pairs.weight > 0.0) and np.all(pairs.weight <= 1.0)
    assert "baseline" in anchor.describe() and "sigma" in anchor.describe()
    assert band_error(pairs) < 0.05


def test_the_anchor_triangulates_through_a_panned_neck() -> None:
    """The same rendered sidestep with the head turned 25 degrees: the cart steps along the
    camera's own right, so the two pictures are the pair the renderer made, and the anchor
    returns the rendered planes' depths because the context brings TF's whole edge. Fed the
    pitch-only pose instead, the very same step is read as motion into the picture: the pairs
    are gone, or the depths they carry are not the planes'."""
    a, b, _ = rendered_pair()
    place = neck_edge(math.radians(26.0), math.radians(25.0))
    step = SIDESTEP_M * np.asarray(place.rotation)[:, 0]  # the camera's own right, in base_link
    poses = {1.0: planar_pose(0.0, 0.0, 0.0), 1.2: planar_pose(float(step[0]), float(step[1]), 0.0)}
    odometry = Odometry(poses)
    network = np.full((INTR.height, INTR.width), 3.0)
    anchor = ParallaxAnchor()
    assert anchor.pairs(Frame(network, context(1.0, a, odometry, place))) is None
    pairs = anchor.pairs(Frame(network, context(1.2, b, odometry, place)))
    assert pairs is not None and pairs.size >= 50
    assert band_error(pairs) < 0.05
    blind = ParallaxAnchor()
    assert blind.pairs(Frame(network, context(1.0, a, odometry))) is None
    lost = blind.pairs(Frame(network, context(1.2, b, odometry)))
    assert lost is None or band_error(lost) > 0.1


def test_the_anchor_holds_when_the_context_brings_no_picture_or_no_odometry() -> None:
    """A node that does not feed the grey image, or a stamp the odometry cannot cover, costs
    the anchor nothing and contributes nothing."""
    a, b, _ = rendered_pair()
    anchor = ParallaxAnchor()
    network = np.full((INTR.height, INTR.width), 3.0)
    assert anchor.pairs(Frame(network, FrameContext(INTR, CameraPose(0.0, 0.0, 1.23)))) is None
    assert anchor.rejected["no image"] == 1
    blind = Odometry({})
    assert anchor.pairs(Frame(network, context(1.0, a, blind))) is None
    assert anchor.pairs(Frame(network, context(1.2, b, blind))) is None
    assert anchor.rejected["no odometry"] == 1


def test_a_gap_outside_the_window_is_not_triangulated() -> None:
    """Two frames a second and a half apart have moved farther than the flow follows: the pair
    is skipped, not tracked."""
    a, b, _ = rendered_pair()
    poses = {1.0: planar_pose(0.0, 0.0, 0.0), 3.0: planar_pose(0.0, -SIDESTEP_M, 0.0)}
    anchor = ParallaxAnchor()
    network = np.full((INTR.height, INTR.width), 3.0)
    odometry = Odometry(poses)
    assert anchor.pairs(Frame(network, context(1.0, a, odometry))) is None
    assert anchor.pairs(Frame(network, context(3.0, b, odometry))) is None
    assert anchor.rejected["gap"] == 1 and anchor.frames == 0


def test_the_flag_is_off_by_default_and_the_stage_contributes_nothing() -> None:
    """The chain ships with the parallax anchor switched off, in its place among the anchors,
    and switching it on is one call."""
    pipeline = standard_pipeline()
    assert pipeline.names.index("parallax_anchor") == pipeline.names.index("wall_anchor") + 1
    assert pipeline.names.index("parallax_anchor") < pipeline.names.index("affine_law")
    assert pipeline.switches["parallax_anchor"] is False
    a, _b, _ = rendered_pair()
    poses = {1.0: planar_pose(0.0, 0.0, 0.0)}
    ctx = context(1.0, a, Odometry(poses))
    result = pipeline.run(np.full((INTR.height, INTR.width), 3.0), ctx)
    assert result.verdict("parallax_anchor").on is False
    assert result.verdict("parallax_anchor").pairs == 0
    assert standard_pipeline(parallax_anchor=True).switches["parallax_anchor"] is True


def test_a_colour_frame_becomes_the_grey_the_tracker_reads() -> None:
    """The node hands the pipeline an RGB frame; the anchor wants one channel."""
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[:, :, 1] = 200
    grey = to_gray(rgb)
    assert grey.shape == (4, 5) and grey.dtype == np.uint8
    assert int(grey[0, 0]) == round(0.587 * 200)
    assert to_gray(grey) is not None and to_gray(grey).shape == (4, 5)


def test_every_kept_point_s_rounded_pixel_indexes_the_second_image() -> None:
    """The flow follows a corner past the border of B and rounding can push a point at 639.6
    to column 640; neither may reach the pool, because an anchor indexes the depth image with a
    kept point's rounded pixel (13 such points on run 0171, scratch/parallax_vs_lidar.py)."""
    depth = np.full((INTR.height, INTR.width), 3.0)
    for sidestep in (SIDESTEP_M, -0.25):  # corners leaving on the left, then on the right
        a, b, motion = rendered_pair(sidestep)
        truth = parallax_truth(a, b, INTR, motion)
        assert truth.kept > 50
        column = np.rint(truth.points[:, 0]).astype(int)
        row = np.rint(truth.points[:, 1]).astype(int)
        assert column.min() >= 0 and column.max() < INTR.width
        assert row.min() >= 0 and row.max() < INTR.height
        assert depth[row, column].shape == truth.z.shape
        assert truth.rejected["outside"] >= 0
