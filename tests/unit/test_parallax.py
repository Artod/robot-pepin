"""Depth from the cart's own motion: the triangulation is exact on geometry it is handed, the
tracker recovers a known depth from two rendered views to within a percent, the degenerate
motions say so instead of inventing numbers, and the anchor is off until its flag is on."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from pepin.depth import CameraPose, Intrinsics, project_all
from pepin.depth_pipeline import (
    PARALLAX_MAX_GAP_S,
    PARALLAX_ORB_MAX_GAP_S,
    Frame,
    FrameContext,
    Pairs,
    ParallaxAnchor,
    standard_pipeline,
)
from pepin.parallax import (
    DISPARITY_SIGMA_PX,
    MAX_SAMPSON_PX,
    ORB_DISPARITY_SIGMA_PX,
    CameraPlacement,
    Features,
    Motion,
    ParallaxTruth,
    Tracks,
    build_tracks,
    camera_motion,
    match,
    parallax_truth,
    perpendicular_baseline,
    sampson,
    to_gray,
    track_truth,
    triangulate,
    triangulate_tracks,
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


class Tracked(Odometry):
    """A motion source that can also answer through the map, as a node's FramePoser does: the
    tracker's poses are the truth, the odometry's are whatever ``drift`` stretches the step by —
    the wheels' own over-reading. ``blind`` is a tracker that says nothing, the silence the
    anchor must fall back from."""

    def __init__(
        self, poses: dict[float, RigidPose], drift: float = 1.0, blind: bool = False
    ) -> None:
        super().__init__({t: _stretched(pose, drift) for t, pose in poses.items()})
        self._map = poses
        self._blind = blind
        self.map_asks = 0

    def map_motion(self, from_stamp: float, to_stamp: float) -> RigidPose | None:
        """How base_link moved between the two stamps according to the tracker, or ``None``
        when this tracker is silent."""
        self.map_asks += 1
        if self._blind or from_stamp not in self._map or to_stamp not in self._map:
            return None
        return base_motion(self._map[from_stamp], self._map[to_stamp])

    def map_motion_recent(
        self, from_stamp: float, to_stamp: float, max_age_s: float
    ) -> RigidPose | None:
        """The same answer asked the way the frame path asks it (never waiting): this tape's
        poses are all already held, so the two agree — what the anchor must NOT do is call the
        blocking one, and that is what map_asks counts."""
        return self.map_motion(from_stamp, to_stamp)


def _stretched(pose: RigidPose, drift: float) -> RigidPose:
    """The same pose with its position scaled: a metre of travel the wheels call ``drift``
    metres."""
    return RigidPose(pose.rotation, drift * np.asarray(pose.translation))


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


def test_the_describer_recovers_the_rendered_depths_within_two_percent() -> None:
    """The same three planes and the same 10 cm sidestep read by ORB instead of the flow:
    keypoints matched by description recover every band's depth within 2 % — a keypoint is
    placed to about a pixel where a tracked corner is placed to half of one."""
    a, b, motion = rendered_pair()
    truth = parallax_truth(a, b, INTR, motion, matcher="orb")
    assert truth.kept >= 50
    rows = truth.points[:, 1]
    for top, bottom, z in BANDS:
        inside = (rows >= top + 16) & (rows < bottom - 16)
        assert int(inside.sum()) >= 10, f"band {z} m has {int(inside.sum())} pairs"
        ratio = float(np.median(truth.z[inside] / z))
        assert abs(ratio - 1.0) < 0.02, f"band {z} m came back at {ratio:.3f} of its depth"


def test_the_describer_returns_the_same_disparity_the_flow_does() -> None:
    """The match itself, by description: every kept correspondence sits on the sidestep's own
    disparity, to within the pixel a keypoint is placed to."""
    a, b, _motion = rendered_pair()
    pts_a, pts_b = match(a, b, matcher="orb")
    assert len(pts_a) >= 50
    middle = (pts_a[:, 1] >= 136) & (pts_a[:, 1] < 224)  # the 2 m band, away from its seams
    shift = pts_a[middle, 0] - pts_b[middle, 0]
    assert float(np.median(shift)) == pytest.approx(INTR.fx * SIDESTEP_M / 2.0, abs=0.5)


def test_a_keypoint_is_trusted_to_a_pixel_and_a_tracked_corner_to_half_of_one() -> None:
    """The matcher decides the noise it is credited with: the same geometry read by ORB carries
    the wider sigma, because a keypoint's octave places it less precisely than the flow does."""
    a, b, motion = rendered_pair()
    flow = parallax_truth(a, b, INTR, motion)
    orb = parallax_truth(a, b, INTR, motion, matcher="orb")

    def middle_band(truth: ParallaxTruth) -> float:
        """The median sigma of the 2 m band, away from the rendered seams."""
        rows = truth.points[:, 1]
        return float(np.median(truth.sigma[(rows >= 136) & (rows < 224)]))

    assert float(np.median(orb.sigma)) > float(np.median(flow.sigma))
    assert middle_band(orb) == pytest.approx(
        middle_band(flow) * ORB_DISPARITY_SIGMA_PX / DISPARITY_SIGMA_PX, rel=0.5
    )


def test_a_misspelt_matcher_is_refused_rather_than_silently_the_default() -> None:
    """A flag nobody implements must not quietly become the flow: the name is checked."""
    a, b, motion = rendered_pair()
    with pytest.raises(ValueError, match="unknown matcher"):
        parallax_truth(a, b, INTR, motion, matcher="sift")


def test_a_blank_pair_describes_nothing_and_says_so() -> None:
    """Two grey walls have no keypoints: the describer returns empty rather than raising, and
    the anchor's verdict is the matcher's own loss."""
    blank = np.full((INTR.height, INTR.width), 128, dtype=np.uint8)
    pts_a, pts_b = match(blank, blank, matcher="orb")
    assert len(pts_a) == len(pts_b) == 0
    _, _, motion = rendered_pair()
    truth = parallax_truth(blank, blank, INTR, motion, matcher="orb")
    assert truth.kept == 0


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


def band_error(pairs: Pairs) -> float:
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
    anchor = ParallaxAnchor(track_min_obs=2)
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
    anchor = ParallaxAnchor(track_min_obs=2)
    assert anchor.pairs(Frame(network, context(1.0, a, odometry, place))) is None
    pairs = anchor.pairs(Frame(network, context(1.2, b, odometry, place)))
    assert pairs is not None and pairs.size >= 50
    assert band_error(pairs) < 0.05
    blind = ParallaxAnchor(track_min_obs=2)
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


def test_the_flag_ships_off_and_switching_it_on_puts_the_stage_in() -> None:
    """The chain ships with the parallax anchor switched ON — the second ruler of the scale —
    in its place among the anchors, and switching it off is one call. A frame the cart did not
    move between contributes nothing either way."""
    pipeline = standard_pipeline()
    assert pipeline.names.index("parallax_anchor") == pipeline.names.index("wall_anchor") + 1
    assert pipeline.names.index("parallax_anchor") < pipeline.names.index("affine_law")
    # off since 2026-09-15: the blocking motion ask starved the stream
    assert pipeline.switches["parallax_anchor"] is False
    a, _b, _ = rendered_pair()
    poses = {1.0: planar_pose(0.0, 0.0, 0.0)}
    ctx = context(1.0, a, Odometry(poses))
    result = pipeline.run(np.full((INTR.height, INTR.width), 3.0), ctx)
    assert result.verdict("parallax_anchor").on is False
    assert standard_pipeline(parallax_anchor=True).switches["parallax_anchor"] is True


def test_the_window_follows_the_matcher_and_a_given_one_pins_it() -> None:
    """How far back a partner may sit is the matcher's own limit, not a constant: the flow is
    given 0.60 s and the describer 1.5, switching the matcher live moves the window with it,
    and a window asked for by name outranks both."""
    anchor = ParallaxAnchor()
    assert anchor.matcher == "klt"
    assert anchor.max_gap_s == PARALLAX_MAX_GAP_S
    anchor.matcher = "orb"
    assert anchor.max_gap_s == PARALLAX_ORB_MAX_GAP_S
    pinned = ParallaxAnchor(max_gap_s=0.25, matcher="orb")
    assert pinned.max_gap_s == 0.25


def test_the_report_line_names_the_matcher_and_its_window() -> None:
    """The report line says who matched, how far back it looked and whether a corner was a
    track or a pair, so an A/B in the field is readable without asking the node what its
    parameters are."""
    line = ParallaxAnchor(track_min_obs=2).describe()
    assert line.startswith("klt <= 0.60 s")
    assert ParallaxAnchor(matcher="orb", track_min_obs=2).describe().startswith("orb <= 1.50 s")
    tracking = ParallaxAnchor().describe()
    assert tracking.startswith("klt <= 1.50 s >= 3 obs, asks 10 cm total")


def test_the_baseline_is_the_tracker_s_word_and_the_odometry_s_drift_scales_the_depth() -> None:
    """The same two rendered pictures, the same corners, one difference: whose motion the
    baseline is. The tracker knows the 12 cm sidestep; the wheels call it 15 cm (the measured
    over-reading past a second of gap, scratch/parallax_pose_sweep.txt). Every depth is
    proportional to the baseline, so the default reads the rendered planes and the odometry
    reads them a quarter too far — and the report line says which source answered."""
    a, b, _ = rendered_pair()
    poses = {1.0: planar_pose(0.0, 0.0, 0.0), 1.2: planar_pose(0.0, -SIDESTEP_M, 0.0)}
    network = np.full((INTR.height, INTR.width), 3.0)
    source = Tracked(poses, drift=1.25)
    anchor = ParallaxAnchor(track_min_obs=2)
    assert anchor.motion_source == "tracker"
    assert anchor.pairs(Frame(network, context(1.0, a, source))) is None
    pairs = anchor.pairs(Frame(network, context(1.2, b, source)))
    assert pairs is not None and pairs.size >= 50
    assert band_error(pairs) < 0.05
    assert anchor.used == {"tracker": 1, "odom": 0}
    assert "on the tracker's motion (tracker 1)" in anchor.describe()
    wheels = ParallaxAnchor(motion_source="odom", track_min_obs=2)
    assert wheels.pairs(Frame(network, context(1.0, a, source))) is None
    stretched = wheels.pairs(Frame(network, context(1.2, b, source)))
    assert stretched is not None and stretched.size >= 50
    assert float(np.median(np.asarray(stretched.z) / np.asarray(pairs.z))) == pytest.approx(
        1.25, rel=0.02
    )
    assert wheels.used == {"tracker": 0, "odom": 1} and "(odom 1)" in wheels.describe()


def test_a_silent_tracker_falls_back_to_the_odometry_for_that_window() -> None:
    """A tracker that cannot answer these two stamps — none yet, or a map pose too stale for TF
    to interpolate — costs the frame nothing: the window is triangulated on the odometry
    instead, counted as such, and a source with no map at all (a tape, another robot) is that
    same fallback. The tracker is asked once per frame, not once per candidate partner."""
    a, b, _ = rendered_pair()
    poses = {1.0: planar_pose(0.0, 0.0, 0.0), 1.2: planar_pose(0.0, -SIDESTEP_M, 0.0)}
    network = np.full((INTR.height, INTR.width), 3.0)
    silent = Tracked(poses, blind=True)
    anchor = ParallaxAnchor(track_min_obs=2)
    assert anchor.pairs(Frame(network, context(1.0, a, silent))) is None
    pairs = anchor.pairs(Frame(network, context(1.2, b, silent)))
    assert pairs is not None and pairs.size >= 50
    assert anchor.used == {"tracker": 0, "odom": 1}
    assert silent.map_asks == 1  # the second frame's walk; the first has an empty ring
    plain = Odometry(poses)  # no map_motion at all: the protocol simply does not match
    bare = ParallaxAnchor(track_min_obs=2)
    assert bare.pairs(Frame(network, context(1.0, a, plain))) is None
    assert bare.pairs(Frame(network, context(1.2, b, plain))) is not None
    assert bare.used == {"tracker": 0, "odom": 1}


def test_the_anchor_hands_its_matcher_to_the_triangulation() -> None:
    """The stage's switch reaches pepin.parallax: an anchor set to a matcher nobody implements
    fails at the match rather than quietly tracking corners."""
    a, b, _ = rendered_pair()
    poses = {1.0: planar_pose(0.0, 0.0, 0.0), 1.3: planar_pose(0.0, -SIDESTEP_M, 0.0)}
    odometry = Odometry(poses)
    anchor = ParallaxAnchor(matcher="sift")
    network = np.full((INTR.height, INTR.width), 3.0)
    assert anchor.pairs(Frame(network, context(1.0, a, odometry))) is None  # the ring's first
    with pytest.raises(ValueError, match="unknown matcher"):
        anchor.pairs(Frame(network, context(1.3, b, odometry)))


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


def crawling_frames(step_m: float, count: int) -> tuple[list[np.ndarray], dict[float, RigidPose]]:
    """A cart crawling ``step_m`` to its right between frames 0.1 s apart: the views of the
    three planes it sees, and the cart pose at each stamp (0.0, 0.1, ... ). Every band's shift
    is whole pixels, so the tracker has the same corners to follow however far back it pairs."""
    wide = texture()
    frames = []
    poses = {}
    for i in range(count):
        view = np.empty((INTR.height, INTR.width), dtype=np.uint8)
        for top, bottom, z in BANDS:
            shift = round(INTR.fx * step_m * i / z)
            view[top:bottom] = wide[top:bottom, MARGIN + shift : MARGIN + shift + INTR.width]
        frames.append(np.ascontiguousarray(view))
        poses[round(0.1 * i, 3)] = planar_pose(0.0, -step_m * i, 0.0)
    return frames, poses


def test_the_partner_frame_is_chosen_for_its_baseline_not_for_being_the_last_one() -> None:
    """A cart crawling 2 cm a frame: the frame before this one carries 2 cm of baseline, and
    the anchor walks back through its ring until it finds the 10 cm it asks for — five frames,
    half a second. The depths it triangulates across that baseline are the rendered planes'."""
    frames, poses = crawling_frames(0.02, 6)
    odometry = Odometry(poses)
    anchor = ParallaxAnchor(track_min_obs=2)
    network = np.full((INTR.height, INTR.width), 3.0)
    pairs = None
    for i, view in enumerate(frames):
        pairs = anchor.pairs(Frame(network, context(round(0.1 * i, 3), view, odometry)))
    assert anchor.gap_s == pytest.approx(0.5)
    assert pairs is not None and pairs.size >= 50
    assert band_error(pairs) < 0.05
    assert "asks 10 cm" in anchor.describe()


def test_a_zero_ask_pairs_with_the_frame_before_as_the_anchor_always_did() -> None:
    """The flag's off position: asked for no baseline at all, the walk stops at the newest
    partner in the window — the frame before this one, 0.1 s and 2 cm back."""
    frames, poses = crawling_frames(0.02, 6)
    anchor = ParallaxAnchor(min_baseline_m=0.0, track_min_obs=2)
    network = np.full((INTR.height, INTR.width), 3.0)
    for i, view in enumerate(frames):
        anchor.pairs(Frame(network, context(round(0.1 * i, 3), view, Odometry(poses))))
    assert anchor.gap_s == pytest.approx(0.1)
    assert "asks 0 cm" in anchor.describe()


def test_a_ring_that_reaches_no_baseline_pairs_across_the_widest_it_has() -> None:
    """A cart that has not moved: no partner in the ring reaches the asked-for 10 cm, the
    widest one is taken anyway, and the triangulation says ``still`` rather than inventing a
    depth. Nothing is tracked, so the stage costs a standing cart nothing."""
    frames, poses = crawling_frames(0.0, 6)
    anchor = ParallaxAnchor()
    network = np.full((INTR.height, INTR.width), 3.0)
    for i, view in enumerate(frames):
        assert (
            anchor.pairs(Frame(network, context(round(0.1 * i, 3), view, Odometry(poses)))) is None
        )
    assert anchor.gap_s is None and anchor.frames == 5
    assert anchor.rejected["still"] == 5


def test_the_ring_forgets_frames_older_than_the_window() -> None:
    """The ring holds the gap window and no more: a frame a second and a half old is gone by
    the time a partner is looked for, so a long stall pairs nothing rather than tracking across
    a view that has changed completely."""
    frames, poses = crawling_frames(0.02, 2)
    late = round(0.1 * 1 + 1.5, 3)
    poses[late] = planar_pose(0.0, -0.40, 0.0)
    anchor = ParallaxAnchor(track_min_obs=2)
    network = np.full((INTR.height, INTR.width), 3.0)
    odometry = Odometry(poses)
    assert anchor.pairs(Frame(network, context(0.0, frames[0], odometry))) is None
    assert anchor.pairs(Frame(network, context(0.1, frames[1], odometry))) is not None
    assert anchor.pairs(Frame(network, context(late, frames[0], odometry))) is None
    assert anchor.rejected["gap"] == 1


# ---- a corner as a track through many frames ------------------------------------------------
def moving_scene(
    views: int, step_m: float = 0.05, noise_px: float = 0.5, seed: int = 1, count: int = 200
) -> tuple[np.ndarray, Tracks]:
    """A cloud of 3D points in front of the CURRENT camera, seen by a camera that sidestepped
    ``step_m`` between consecutive views, every pixel disturbed by ``noise_px`` of Gaussian
    noise. Returns the points (in the current camera's frame) and the tracks that see them —
    the geometry with no matcher in the way, so a failure here is the triangulation's."""
    rng = np.random.default_rng(seed)
    points = np.stack(
        [
            rng.uniform(-1.5, 1.5, count),
            rng.uniform(-1.0, 1.0, count),
            rng.uniform(1.0, 4.0, count),
        ],
        axis=1,
    )
    motions = [
        Motion(np.eye(3), np.array([-(views - 1 - v) * step_m, 0.0, 0.0])) for v in range(views)
    ]
    pixels = np.full((count, views, 2), np.nan)
    for v, motion in enumerate(motions):
        local = (points - motion.translation) @ np.asarray(motion.rotation)
        pixels[:, v, 0] = INTR.fx * local[:, 0] / local[:, 2] + INTR.cx
        pixels[:, v, 1] = INTR.fy * local[:, 1] / local[:, 2] + INTR.cy
        pixels[:, v] += rng.normal(0.0, noise_px, (count, 2))
    seen = np.ones((count, views), dtype=bool)
    return points, Tracks(pixels, seen, tuple(motions), found=count)


def test_more_observations_shrink_the_sigma_and_the_error_stays_inside_it() -> None:
    """The claim the whole change rests on: a corner triangulated from 5 and 10 views is known
    better than the same corner from 2, the reported sigma says by how much, and the sigma is
    honest — the depth error stays inside about two of them. The baselines add in quadrature,
    so 10 views 5 cm apart are worth one pair at 45 cm and the sigma falls by that factor."""
    told: list[float] = []
    for views in (2, 5, 10):
        points, tracks = moving_scene(views)
        found = triangulate_tracks(tracks, INTR)
        good = np.isfinite(found.z)
        assert good.sum() > 150
        error = np.abs(found.z - points[:, 2])[good]
        sigma = found.sigma[good]
        told.append(float(np.median(sigma)))
        assert np.all(found.observations == views)
        # 2.2 and not 2: the pair's own formula credits ONE pixel of noise where a disparity
        # carries two views' worth, so the sigma runs a little optimistic at every view count.
        assert float(np.percentile(error / sigma, 90)) < 2.2, f"{views} views: sigma is too small"
        assert float(np.median(error / sigma)) > 0.2, f"{views} views: sigma is pessimistic"
    assert told[1] < 0.5 * told[0], "5 views must beat 2 by more than a factor of two"
    assert told[2] < 0.5 * told[1], "10 views must beat 5 by more than a factor of two"


def test_two_observations_are_arithmetically_today_s_pair() -> None:
    """The off position of the knob is not a second code path: a track seen in exactly two
    views carries the depth the pair's own triangulation returns, to the float32 pixels' own
    rounding, and its sigma to a few parts in a thousand — the whole difference being that the
    bundle widens ``sigma_px`` by the residual averaged over BOTH views where the pair reads it
    in the second one only. Which is why 2 is what ``parallax_track_min_obs`` means."""
    for noise, tolerance in ((0.0, 1e-4), (0.5, 5e-3)):
        _points, tracks = moving_scene(2, noise_px=noise)
        found = triangulate_tracks(tracks, INTR)
        pair_a = np.asarray(tracks.pixels[:, 0], dtype=np.float32)
        pair_b = np.asarray(tracks.pixels[:, 1], dtype=np.float32)
        z, sigma = triangulate(pair_a, pair_b, INTR, tracks.motions[0])
        good = np.isfinite(found.z) & np.isfinite(z)
        assert good.sum() > 150
        assert np.allclose(found.z[good], z[good], rtol=1e-4)
        assert np.allclose(found.sigma[good], sigma[good], rtol=tolerance)
        assert np.allclose(found.sigma_two[good], found.sigma[good], rtol=tolerance)


def test_one_corrupted_observation_is_dropped_and_the_track_survives() -> None:
    """A mistracked hop must not cost the whole track: the observation that misses the solved
    point is dropped and the remaining views are solved again, so the depth comes back about as
    good as a clean track's — and measurably better than the same track with nothing dropped."""
    points, tracks = moving_scene(5, noise_px=0.2)
    clean = triangulate_tracks(tracks, INTR)
    spoilt = np.asarray(tracks.pixels).copy()
    spoilt[:, 2, 0] += 25.0  # the middle view's corner slid 25 px: a hop onto the wrong texture
    broken = replace(tracks, pixels=spoilt)
    repaired = triangulate_tracks(broken, INTR)
    good = np.isfinite(repaired.z) & np.isfinite(clean.z)
    assert good.sum() > 150
    assert repaired.repaired == tracks.count, "every track should have lost its bad observation"
    assert np.all(repaired.observations[good] == 4)
    error = np.abs(repaired.z - points[:, 2])[good]
    assert float(np.median(error)) < 3.0 * float(np.median(np.abs(clean.z - points[:, 2])[good]))
    naive = triangulate_tracks(broken, INTR, outlier_px=1e9)  # the same track, nothing dropped
    assert float(np.median(np.abs(naive.z - points[:, 2])[good])) > 2.0 * float(np.median(error)), (
        "the repair must actually be worth something"
    )


def track_window(
    frames: list[np.ndarray], poses: dict[float, RigidPose], place: CameraPlacement
) -> list[Motion]:
    """The camera motion from each of ``frames`` but the last into the last one's optical
    frame — what :func:`build_tracks` and :func:`track_truth` want beside the pictures."""
    stamps = sorted(poses)[: len(frames)]
    now = stamps[-1]
    moved = [base_motion(poses[then], poses[now]) for then in stamps[:-1]]
    return [camera_motion(m.rotation, m.translation, place, place) for m in moved]


@pytest.mark.parametrize("matcher", ["klt", "orb"])
def test_both_matchers_build_tracks_across_a_sequence(matcher: str) -> None:
    """The flow follows a corner hop by hop and the describer recognises it frame by frame:
    either way a corner of the last picture is found in the earlier ones, and the depths the
    whole window triangulates are the rendered planes'."""
    frames, poses = crawling_frames(0.03, 6)
    place = CameraPlacement.of(CameraPose(0.0, 0.0, 1.23, math.radians(26.0)))
    motions = track_window(frames, poses, place)
    tracks = build_tracks(frames, motions, matcher=matcher)
    assert tracks.count >= 50 and tracks.views == 6
    assert float(np.median(tracks.observations)) >= 3
    truth = track_truth(frames, motions, INTR, matcher=matcher)
    assert truth.kept >= 20
    assert truth.observations is not None and float(np.median(truth.observations)) >= 3
    rows = truth.points[:, 1]
    for top, bottom, z in BANDS:
        inside = (rows >= top + 16) & (rows < bottom - 16)
        if int(inside.sum()) < 5:
            continue
        ratio = float(np.median(truth.z[inside] / z))
        assert abs(ratio - 1.0) < 0.05, f"band {z} m came back at {ratio:.3f} of its depth"


def test_the_describer_keeps_what_it_read_of_a_frame() -> None:
    """A frame a dozen tracks reach back through is described once: the cache the caller hands
    in comes back filled, and a second call with it in hand returns the same tracks."""
    frames, poses = crawling_frames(0.03, 4)
    place = CameraPlacement.of(CameraPose(0.0, 0.0, 1.23, math.radians(26.0)))
    motions = track_window(frames, poses, place)
    cache: list[Features | None] = [None] * len(frames)
    first = build_tracks(frames, motions, matcher="orb", features=cache)
    assert all(isinstance(f, Features) for f in cache)
    again = build_tracks(frames, motions, matcher="orb", features=cache)
    assert again.count == first.count
    assert np.array_equal(again.seen, first.seen)


def test_a_window_without_a_motion_for_every_frame_is_refused() -> None:
    """A caller that hands in one motion too few has mislabelled which frame is which, and a
    silent off-by-one there is a depth wrong by a whole frame's step."""
    frames, poses = crawling_frames(0.03, 4)
    place = CameraPlacement.of(CameraPose(0.0, 0.0, 1.23, math.radians(26.0)))
    motions = track_window(frames, poses, place)
    with pytest.raises(ValueError, match="motions"):
        build_tracks(frames, motions[:-1], matcher="klt")


def test_a_standing_cart_tracks_nothing_and_says_which_word() -> None:
    """No view moved: there is no baseline to meet rays across, and the window says ``still``
    rather than triangulating rays that are all the same ray."""
    frames, poses = crawling_frames(0.0, 4)
    place = CameraPlacement.of(CameraPose(0.0, 0.0, 1.23, math.radians(26.0)))
    motions = track_window(frames, poses, place)
    truth = track_truth(frames, motions, INTR)
    assert truth.kept == 0 and truth.verdict == "still"


def test_a_track_too_short_or_too_thin_is_not_a_measurement() -> None:
    """The two gates the knobs name: a corner seen in fewer than ``min_obs`` views is not a
    track, and a window whose views add up to less than ``min_total_baseline_m`` of parallax
    has not measured anything, however many views it has."""
    frames, poses = crawling_frames(0.03, 6)
    place = CameraPlacement.of(CameraPose(0.0, 0.0, 1.23, math.radians(26.0)))
    motions = track_window(frames, poses, place)
    strict = track_truth(frames, motions, INTR, min_obs=6)
    assert strict.rejected["short track"] > 0
    greedy = track_truth(frames, motions, INTR, min_total_baseline_m=5.0)
    assert greedy.kept == 0 and greedy.rejected["total baseline"] > 0


def test_the_anchor_triangulates_its_corners_from_the_whole_window() -> None:
    """The stage's own path: a crawling cart, and the corners of the newest frame are followed
    back through the ring and met from every view at once. The depths are the rendered planes',
    every pair rests on more than two observations, and the sigma the bundle reports is smaller
    than the widest single pair of the same corners would have claimed — which is what the
    report line prints side by side."""
    frames, poses = crawling_frames(0.03, 6)
    odometry = Odometry(poses)
    anchor = ParallaxAnchor()
    network = np.full((INTR.height, INTR.width), 3.0)
    pairs = None
    for i, view in enumerate(frames):
        pairs = anchor.pairs(Frame(network, context(round(0.1 * i, 3), view, odometry)))
    assert pairs is not None and pairs.size >= 20
    assert band_error(pairs) < 0.05
    line = anchor.describe()
    assert "obs a track" in line and "2-view sigma" in line and "tracks a frame" in line
    assert anchor.sigma_m is not None
    two_view = float(np.median(anchor._sigma_two))
    assert anchor.sigma_m < two_view, f"{anchor.sigma_m:.3f} m is not better than {two_view:.3f}"


def test_the_knob_at_two_is_the_pair_the_anchor_always_measured() -> None:
    """The same six frames read both ways: at ``track_min_obs`` 2 the anchor pairs with one
    partner as it always did, at 3 it tracks through the window. Both put the rendered planes
    where they are; the tracks put them there more precisely."""
    frames, poses = crawling_frames(0.03, 6)
    network = np.full((INTR.height, INTR.width), 3.0)
    got: dict[int, tuple[ParallaxAnchor, Pairs]] = {}
    for min_obs in (2, 3):
        anchor = ParallaxAnchor(track_min_obs=min_obs)
        pairs = None
        for i, view in enumerate(frames):
            pairs = anchor.pairs(Frame(network, context(round(0.1 * i, 3), view, Odometry(poses))))
        assert pairs is not None
        got[min_obs] = (anchor, pairs)
    paired, tracked = got[2][0], got[3][0]
    assert paired.tracking is False and tracked.tracking is True
    assert not paired._obs, "the pair path has no observations to report"
    assert float(np.median(tracked._obs)) >= 3
    assert band_error(got[2][1]) < 0.05 and band_error(got[3][1]) < 0.05
    assert tracked.sigma_m is not None and paired.sigma_m is not None
    assert tracked.sigma_m < paired.sigma_m
