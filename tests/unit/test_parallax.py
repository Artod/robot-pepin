"""Depth from the cart's own motion: the triangulation is exact on geometry it is handed, the
tracker recovers a known depth from two rendered views to within a percent, the degenerate
motions say so instead of inventing numbers, and the anchor is off until its flag is on."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest

from pepin.depth import CameraPose, Intrinsics, project_all
from pepin.depth_pipeline import (
    PARALLAX_MAX_GAP_S,
    PARALLAX_ORB_MAX_GAP_S,
    Frame,
    FrameContext,
    Pairs,
    ParallaxAnchor,
    _between,
    _placed,
    standard_pipeline,
)
from pepin.parallax import (
    DISPARITY_SIGMA_PX,
    MAX_REPROJ_PX,
    MAX_SAMPSON_PX,
    ORB_DISPARITY_SIGMA_PX,
    CameraPlacement,
    Features,
    LucasKanade,
    Motion,
    ParallaxTruth,
    PlumbBob,
    Tracks,
    TrackStore,
    build_tracks,
    camera_motion,
    gate_tracks,
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
    is skipped, not tracked. (The gap window is the PAIR's own: a forward track is bounded by
    parallax_track_window_s instead, and reaches as far back as the pose deserves.)"""
    a, b, _ = rendered_pair()
    poses = {1.0: planar_pose(0.0, 0.0, 0.0), 3.0: planar_pose(0.0, -SIDESTEP_M, 0.0)}
    anchor = ParallaxAnchor(tracking="pair")
    network = np.full((INTR.height, INTR.width), 3.0)
    odometry = Odometry(poses)
    assert anchor.pairs(Frame(network, context(1.0, a, odometry))) is None
    assert anchor.pairs(Frame(network, context(3.0, b, odometry))) is None
    assert anchor.rejected["gap"] == 1 and anchor.frames == 0


def test_the_flag_ships_on_and_switching_it_off_takes_the_stage_out() -> None:
    """The chain ships with the parallax anchor switched ON — the second ruler of the scale —
    in its place among the anchors, and switching it off is one call. A frame the cart did not
    move between contributes nothing either way."""
    pipeline = standard_pipeline()
    assert pipeline.names.index("parallax_anchor") == pipeline.names.index("wall_anchor") + 1
    assert pipeline.names.index("parallax_anchor") < pipeline.names.index("affine_law")
    assert pipeline.switches["parallax_anchor"] is True
    a, _b, _ = rendered_pair()
    poses = {1.0: planar_pose(0.0, 0.0, 0.0)}
    ctx = context(1.0, a, Odometry(poses))
    result = pipeline.run(np.full((INTR.height, INTR.width), 3.0), ctx)
    # the first frame of a drive has nothing behind it either way
    assert result.verdict("parallax_anchor").on is True
    assert standard_pipeline(parallax_anchor=False).switches["parallax_anchor"] is False


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


def test_the_per_frame_numbers_are_trimmed_on_every_frame_not_only_on_a_fruitful_one() -> None:
    """The report line's per-frame series (corners live, born, gone, the milliseconds of each
    step) are appended on EVERY frame, while the trim used to sit where the pairs are counted:
    a stage that contributes nothing — a blank wall, a silent odometry — grew them without
    bound, and the medians over 100 000 entries cost 23.8 ms a window (2026-09-16). Six frames
    that give nothing, a window of three: three entries."""
    anchor = ParallaxAnchor(pool_frames=3)
    blank = np.zeros((INTR.height, INTR.width), dtype=np.uint8)
    network = np.full((INTR.height, INTR.width), 3.0)
    blind = Odometry({})  # no pose at any stamp: nothing is ever triangulated
    for k in range(6):
        assert anchor.pairs(Frame(network, context(1.0 + 0.1 * k, blank, blind))) is None
    assert anchor.contributed == 0, "no frame gave a pair"
    for series in (anchor._live, anchor._born, anchor._gone, anchor._hop_ms, anchor._detect_ms):
        assert len(series) == 3, "the window is trimmed at the append"
    assert "live corners" in anchor.describe()


def test_the_report_line_names_the_matcher_and_its_window() -> None:
    """The report line says who matched, how far back it looked and whether a corner was a
    track or a pair, so an A/B in the field is readable without asking the node what its
    parameters are."""
    line = ParallaxAnchor(track_min_obs=2).describe()
    assert line.startswith("klt <= 0.60 s")
    assert ParallaxAnchor(matcher="orb", track_min_obs=2).describe().startswith("orb <= 1.50 s")
    windowed = ParallaxAnchor(tracking="window", track_window_s=1.5).describe()
    assert windowed.startswith("klt <= 1.50 s window >= 3 obs over <= 8 views, asks 10 cm total")
    forward = ParallaxAnchor().describe()
    assert forward.startswith("klt <= 3.00 s forward >= 3 obs over <= 8 views")
    assert "<= 200 corners, detect every 5, lens undone, drift <= 1.0 px every 10" in forward
    assert "raw pixels" in ParallaxAnchor(undistort=False).describe()
    assert "drift unchecked" in ParallaxAnchor(verify_every=0).describe()


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
    anchor = ParallaxAnchor(track_min_obs=2, motion_source="tracker")
    assert anchor.motion_source == "tracker"
    assert anchor.pairs(Frame(network, context(1.0, a, source))) is None
    pairs = anchor.pairs(Frame(network, context(1.2, b, source)))
    assert pairs is not None and pairs.size >= 50
    assert band_error(pairs) < 0.05
    assert anchor.used["tracker"] == 1 and anchor.used["odom"] == 0
    assert "on the tracker's motion (tracker 1)" in anchor.describe()
    wheels = ParallaxAnchor(motion_source="odom", track_min_obs=2)
    assert wheels.pairs(Frame(network, context(1.0, a, source))) is None
    stretched = wheels.pairs(Frame(network, context(1.2, b, source)))
    assert stretched is not None and stretched.size >= 50
    assert float(np.median(np.asarray(stretched.z) / np.asarray(pairs.z))) == pytest.approx(
        1.25, rel=0.02
    )
    assert wheels.used["odom"] == 1 and "(odom 1)" in wheels.describe()


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
    assert anchor.used["tracker"] == 0 and anchor.used["odom"] == 1
    assert silent.map_asks == 1  # the second frame's walk; the first has an empty ring
    plain = Odometry(poses)  # no map_motion at all: the protocol simply does not match
    bare = ParallaxAnchor(track_min_obs=2)
    assert bare.pairs(Frame(network, context(1.0, a, plain))) is None
    assert bare.pairs(Frame(network, context(1.2, b, plain))) is not None
    assert bare.used["tracker"] == 0 and bare.used["odom"] == 1


def test_the_anchor_hands_its_matcher_to_the_triangulation() -> None:
    """The stage's switch reaches pepin.parallax: an anchor set to a matcher nobody implements
    fails at the match rather than quietly tracking corners. The forward store refuses it on
    the first frame, before it has followed anything; the pair path at the first match."""
    a, b, _ = rendered_pair()
    poses = {1.0: planar_pose(0.0, 0.0, 0.0), 1.3: planar_pose(0.0, -SIDESTEP_M, 0.0)}
    odometry = Odometry(poses)
    network = np.full((INTR.height, INTR.width), 3.0)
    paired = ParallaxAnchor(matcher="sift", tracking="pair")
    assert paired.pairs(Frame(network, context(1.0, a, odometry))) is None  # the ring's first
    with pytest.raises(ValueError, match="unknown matcher"):
        paired.pairs(Frame(network, context(1.3, b, odometry)))
    with pytest.raises(ValueError, match="unknown matcher"):
        ParallaxAnchor(matcher="sift").pairs(Frame(network, context(1.0, a, odometry)))


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
    in the second one only. Which is why 2 is what ``parallax_track_min_obs`` means.

    The depth is the pair's whatever sigma is asked for; the closed-form sigma is the pair's to
    a few parts in a thousand, and the solve's own covariance — what the stage reports by
    default since 2026-09-15 — reads within a tenth of it at two views and never below it, the
    departure being the points near the epipole the closed form flatters."""
    for noise, tolerance in ((0.0, 1e-4), (0.5, 5e-3)):
        _points, tracks = moving_scene(2, noise_px=noise)
        found = triangulate_tracks(tracks, INTR, sigma_model="baseline")
        pair_a = np.asarray(tracks.pixels[:, 0], dtype=np.float32)
        pair_b = np.asarray(tracks.pixels[:, 1], dtype=np.float32)
        z, sigma = triangulate(pair_a, pair_b, INTR, tracks.motions[0])
        good = np.isfinite(found.z) & np.isfinite(z)
        assert good.sum() > 150
        assert np.allclose(found.z[good], z[good], rtol=1e-4)
        assert np.allclose(found.sigma[good], sigma[good], rtol=tolerance)
        assert np.allclose(found.sigma_two[good], found.sigma[good], rtol=tolerance)
        shipped = triangulate_tracks(tracks, INTR)
        assert np.allclose(shipped.z[good], z[good], rtol=1e-4)
        ratio = shipped.sigma[good] / sigma[good]
        assert 1.0 <= float(np.median(ratio)) < 1.2
        assert float(np.percentile(ratio, 5)) > 0.9


def driving_scene(
    views: int, step_m: float = 0.0375, noise_px: float = 0.5, seed: int = 3, count: int = 300
) -> tuple[np.ndarray, Tracks]:
    """The same cloud seen by a camera driving FORWARD instead of sidestepping — this cart's
    actual errand, 0.25 m/s at 6-7 frames a second. The views then spread ALONG the rays as well
    as across them, which is the geometry the closed-form sigma gets wrong."""
    rng = np.random.default_rng(seed)
    points = np.stack(
        [
            rng.uniform(-1.5, 1.5, count),
            rng.uniform(-1.0, 1.0, count),
            rng.uniform(1.5, 4.0, count),
        ],
        axis=1,
    )
    motions = [
        Motion(np.eye(3), np.array([0.0, 0.0, -(views - 1 - v) * step_m])) for v in range(views)
    ]
    pixels = np.full((count, views, 2), np.nan)
    for v, motion in enumerate(motions):
        local = (points - motion.translation) @ np.asarray(motion.rotation)
        pixels[:, v, 0] = INTR.fx * local[:, 0] / local[:, 2] + INTR.cx
        pixels[:, v, 1] = INTR.fy * local[:, 1] / local[:, 2] + INTR.cy
        pixels[:, v] += rng.normal(0.0, noise_px, (count, 2))
    return points, Tracks(pixels, np.ones((count, views), dtype=bool), tuple(motions), found=count)


def test_the_sigma_is_honest_for_a_cart_driving_forward_and_the_closed_form_is_not() -> None:
    """The defect the covariance model was added for (scratch/parallax_sigma_mc.txt,
    2026-09-15). Every earlier test sidesteps, where ``sqrt(sum b_v^2)`` is exactly right. Drive
    FORWARD instead and the views spread along the ray as well as across it: a view that sees
    the point from further away reads its pixel into a bigger depth error, which the closed form
    does not know. It then calls a sixteen-view bundle better than it is — and a pair's vote in
    the frame's fit is 1 / sigma^2, so an optimistic sigma is a loud vote."""
    for views in (10, 16):
        points, tracks = driving_scene(views)
        error = {}
        for model in ("baseline", "covariance"):
            found = triangulate_tracks(tracks, INTR, sigma_model=model)
            # the corners the field would keep: TRACK_MIN_TOTAL_BASELINE_M of parallax
            good = np.isfinite(found.z) & np.isfinite(found.sigma) & (found.baseline >= 0.10)
            assert good.sum() > 200
            # an honest sigma covers about two thirds of its own scatter, so this is about 1
            error[model] = float(
                np.percentile(np.abs(found.z - points[:, 2])[good] / found.sigma[good], 68)
            )
        assert error["baseline"] > 1.3, (
            f"{views} views: the closed form under-reads the scatter here, it still should"
        )
        assert error["covariance"] < 1.15, (
            f"{views} views: the solve's own covariance must cover its own scatter"
        )


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


def test_the_view_cap_reaches_the_matcher_and_is_what_the_frame_costs() -> None:
    """85 % of a track's cost is the flow's hops and there is one hop per view, so the cap on
    the views IS the cost per frame (52.5 ms at 16 views against 27.0 at 8 over the four errands
    of 2026-09-14, scratch/parallax_tracks_audit.txt). The knob must therefore really reach the
    matcher: a cap of 3 leaves no track resting on more."""
    frames, poses = crawling_frames(0.03, 8)
    network = np.full((INTR.height, INTR.width), 3.0)
    seen: dict[int, float] = {}
    for cap in (3, 8):
        anchor = ParallaxAnchor(track_max_views=cap)
        for i, view in enumerate(frames):
            anchor.pairs(Frame(network, context(round(0.1 * i, 3), view, Odometry(poses))))
        assert anchor._obs, f"cap {cap} triangulated nothing"
        seen[cap] = max(anchor._obs)
        assert seen[cap] <= cap, f"cap {cap} rested a track on {seen[cap]} views"
    assert seen[8] > seen[3], "a looser cap must actually buy observations"


def test_one_window_rests_on_one_motion_source() -> None:
    """A tracker whose map pose covers the newest frames but not the oldest: a PAIR could not
    mix sources because it chose a single partner, but a window meets a dozen rays in one solve
    and the two sources disagree by about a quarter over a second (the wheels over-read). So the
    window ends where the source changes rather than bundling half of each, and the report line
    counts the windows it cut."""

    class HalfTracked(Odometry):
        """A tracker that can answer through the map only for stamps at or after ``since``."""

        def __init__(self, poses: dict[float, RigidPose], since: float, drift: float) -> None:
            super().__init__({t: _stretched(p, drift) for t, p in poses.items()})
            self._map = {t: p for t, p in poses.items() if t >= since}

        def map_motion(self, from_stamp: float, to_stamp: float) -> RigidPose | None:
            """The tracker's word, or ``None`` for a frame its map pose no longer covers."""
            if from_stamp not in self._map or to_stamp not in self._map:
                return None
            return base_motion(self._map[from_stamp], self._map[to_stamp])

        def map_motion_recent(
            self, from_stamp: float, to_stamp: float, max_age_s: float
        ) -> RigidPose | None:
            """The same answer, asked the non-blocking way the frame path asks it."""
            return self.map_motion(from_stamp, to_stamp)

    frames, poses = crawling_frames(0.03, 10)
    # the wheels over-read the crawl by half; the map pose is the truth, from 0.2 s on
    source = HalfTracked(poses, since=0.2, drift=1.5)
    network = np.full((INTR.height, INTR.width), 3.0)
    got: dict[str, tuple[ParallaxAnchor, Pairs | None]] = {}
    for ruler in ("window", "forward"):
        # on the tracker's own map pose, which is the source that can fall back mid-drive; the
        # tf source cannot, which is the point of it (test_a_map_pose_built_from_tf_s_two_halves)
        anchor = ParallaxAnchor(tracking=ruler, track_window_s=1.5, motion_source="tracker")
        pairs = None
        for i, view in enumerate(frames):
            pairs = anchor.pairs(Frame(network, context(round(0.1 * i, 3), view, source)))
        got[ruler] = (anchor, pairs)
    windowed, backward = got["window"]
    assert windowed.rejected.get("mixed motion", 0) > 0, "a mixed window must be cut and counted"
    assert "mixed motion" in windowed.describe()
    # the forward ruler does not cut a window, it cuts a TRACK's run: the corner lives on and
    # its bundle starts again at the first frame the tracker answered for
    onward, forward = got["forward"]
    assert onward.deaths.get("source", 0) > 0, "the older views must be cut off and counted"
    assert "source cut" in onward.describe()
    # the last frame reaches back to 0.2 s, where the tracker still speaks: had the two oldest
    # views come in on the wheels' stretched baseline, the planes would sit half again too far
    for pairs in (backward, forward):
        assert pairs is not None and pairs.size >= 20
        assert band_error(pairs) < 0.05, "a bundle that mixed the sources would read the wheels'"


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


# ---- the split test: does a track agree with itself? ------------------------------------------
def rolling_track(
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    views: int = 8,
    dt: float = 0.15,
    speed: float = 0.25,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    point: tuple[float, float, float] = (0.6, 0.1, 2.0),
) -> tuple[Tracks, np.ndarray]:
    """scratch/parallax_gates_probe.py's own construction: a cart moving at ``speed`` along
    ``direction`` for ``views`` frames ``dt`` apart, and a corner whose 3D point moves at
    ``velocity`` while it does. Returns the track and the point's true place in the CURRENT
    camera's frame — everything exact, no pixel noise, so what the gates see is the geometry."""
    unit = np.asarray(direction, dtype=float)
    unit /= np.linalg.norm(unit)
    ages = np.arange(views - 1, -1, -1, dtype=float) * dt
    origins = -(ages * speed)[:, None] * unit[None, :]
    truth = np.asarray(point, dtype=float)
    wandered = truth[None, :] - ages[:, None] * np.asarray(velocity, dtype=float)[None, :]
    local = wandered - origins
    pixels = np.stack(
        [
            INTR.fx * local[:, 0] / local[:, 2] + INTR.cx,
            INTR.fy * local[:, 1] / local[:, 2] + INTR.cy,
        ],
        axis=1,
    )
    motions = tuple(Motion(np.eye(3), o) for o in origins)
    return Tracks(pixels[None, :, :], np.ones((1, views), dtype=bool), motions, found=1), truth


def epipolar_slide(track: Tracks, views: int, per_view: float, grow: bool) -> Tracks:
    """The same track with its earlier observations slid along their own epipolar lines — the
    direction the corner would move if its depth changed, which no epipolar gate constrains.
    ``grow`` makes the slide proportional to the view's age (the probe's drift per hop);
    otherwise every one of the oldest ``views`` observations slides by the same amount."""
    pixels = np.asarray(track.pixels, dtype=float).copy()
    total = track.views
    for v in range(total - 1):
        origin = np.asarray(track.motions[v].translation, dtype=float)
        ray = np.array(
            [
                (pixels[0, v, 0] - INTR.cx) / INTR.fx,
                (pixels[0, v, 1] - INTR.cy) / INTR.fy,
                1.0,
            ]
        )
        far = origin + 1.05 * (ray * 2.0)  # the same corner two per cent further down its ray
        towards = np.array(
            [
                INTR.fx * far[0] / far[2] + INTR.cx - pixels[0, v, 0],
                INTR.fy * far[1] / far[2] + INTR.cy - pixels[0, v, 1],
            ]
        )
        length = float(np.linalg.norm(towards))
        if length < 1e-9:
            continue
        how_far = per_view * (total - 1 - v) if grow else per_view * float(v < views)
        pixels[0, v] += how_far * towards / length
    return replace(track, pixels=pixels)


def test_a_static_corner_s_two_halves_agree_about_its_depth() -> None:
    """The test the gate is: a point that does not move has ONE depth, and the older half of the
    views and the newer half must both read it. A clean track's halves sit on top of each other,
    so the gate costs a good corner nothing."""
    for direction in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.1)):
        track, truth = rolling_track(direction=direction)
        found = triangulate_tracks(track, INTR)
        assert float(found.z[0]) == pytest.approx(float(truth[2]), rel=1e-6)
        assert float(found.split[0]) < 0.01, "an exact track's halves must not disagree at all"
    short = replace(track, seen=np.array([[False] * 6 + [True, True]]))
    assert float(triangulate_tracks(short, INTR).split[0]) == 0.0, "two views cannot be cut"


def test_a_half_of_a_window_sliding_off_the_corner_is_what_the_split_test_sees() -> None:
    """What the gate is for: the oldest views slide off the corner together — a flow that jumped
    once and carried the error back through every older hop — and the bundle's own reprojection
    RMS barely notices, because it is divided by the square root of the view count. The halves
    then read different depths, which is the signal.

    The signal is weak, and that is measured, not assumed: at 2 px of slide the depth is 21 %
    wrong and the halves are only 2.0 sigma apart, because cutting the window in two doubles
    each half's own sigma as well. A clean corner at the flow's 0.4 px also reaches 2.0 sigma on
    noise alone (scratch/parallax_split_probe.py), so the shipped 3 sigma does NOT separate a
    slide of this size — it removes only the 1.1 % tail the real errands carry beyond anything
    pixel noise explains (scratch/parallax_tracks_eval.txt)."""
    track, truth = rolling_track()
    gaps = []
    for slide in (0.5, 1.0, 2.0):
        slid = epipolar_slide(track, views=4, per_view=slide, grow=False)
        found = triangulate_tracks(slid, INTR)
        error = abs(float(found.z[0]) / float(truth[2]) - 1.0)
        gaps.append(float(found.split[0]))
        assert float(found.residual[0]) < MAX_REPROJ_PX, "the reprojection gate lets it through"
        if slide == 2.0:
            assert error > 0.15, f"a 2 px slide should cost more than 15 %, cost {error:.1%}"
    assert gaps[0] < gaps[1] < gaps[2], "the halves must disagree more the further they slide"
    assert gaps[2] < 3.0, "and still not reach the shipped tolerance: this is the measured hole"


def test_a_point_moving_along_the_camera_s_own_motion_is_invisible_to_every_gate() -> None:
    """The hole the split test does NOT close, pinned here so nobody assumes it does. A corner
    on something receding at 0.1 m/s from a cart driving at 0.25 puts every one of its rays
    through ONE point — at 3.33 m for a 2.00 m truth, the ratio of the two speeds — so the
    epipolar distance is zero, the reprojection residual is zero, and both halves of the window
    agree exactly on the wrong answer. This is the monocular depth-velocity ambiguity, not a
    missing test: it needs the network's own depth or a second sensor."""
    track, truth = rolling_track(velocity=(0.0, 0.0, 0.1))
    found = triangulate_tracks(track, INTR)
    assert float(found.z[0]) == pytest.approx(0.25 / 0.15 * float(truth[2]), rel=1e-6)
    assert float(found.residual[0]) < 1e-6 and float(found.split[0]) < 1e-6
    towards = triangulate_tracks(rolling_track(velocity=(0.0, 0.0, -0.1))[0], INTR)
    assert float(towards.z[0]) == pytest.approx(0.25 / 0.35 * float(truth[2]), rel=1e-6)
    assert float(towards.split[0]) < 1e-6
    # Motion ACROSS the view is a different matter: it leaves the epipolar lines, and the
    # per-observation epipolar gate of track_truth throws those observations out.
    across, _ = rolling_track(velocity=(0.1, 0.0, 0.0))
    pixels = np.asarray(across.pixels, dtype=np.float32)
    worst = max(
        float(sampson(pixels[:, v], pixels[:, -1], INTR, across.motions[v])[0])
        for v in range(across.views - 1)
    )
    assert worst > MAX_SAMPSON_PX, "motion across the view does leave the epipolar lines"


def test_a_drift_proportional_to_the_baseline_is_invisible_to_the_split_test_too() -> None:
    """The other hole. A corner sliding along its epipolar line by an amount proportional to how
    far back the view sits is a disparity offset proportional to the baseline, which is a pure
    scale error on the depth — every subset of the views reads the same wrong number, so the
    halves agree. Measured here: a 1 px-per-hop drift moves the depth by more than a fifth and
    the halves by well under the shipped tolerance."""
    track, truth = rolling_track(views=10, dt=0.12, direction=(1.0, 0.0, 0.1))
    drifted = triangulate_tracks(epipolar_slide(track, 10, per_view=1.0, grow=True), INTR)
    assert abs(float(drifted.z[0]) / float(truth[2]) - 1.0) > 0.2
    assert float(drifted.residual[0]) < MAX_REPROJ_PX
    assert float(drifted.split[0]) < 1.0, "the halves cannot see a scale error they both share"


def test_the_split_gate_ships_off_and_is_armed_and_counted_by_its_tolerance() -> None:
    """The gate through the whole measurement. Off by default, and off means the halves are not
    solved at all — it is two extra solves, 3.9 ms a frame of the stage's 27.0, and a gate that
    was measured to change no residual should not cost that. Armed at the 3 sigma the real
    errands were read at it still takes nothing from a clean rendered window; armed tight it
    removes tracks and says so in the rejection tally under ``split``."""
    frames, poses = crawling_frames(0.03, 6)
    place = CameraPlacement.of(CameraPose(0.0, 0.0, 1.23, math.radians(26.0)))
    motions = track_window(frames, poses, place)
    off = track_truth(frames, motions, INTR)
    assert off.kept > 10 and off.rejected["split"] == 0
    assert off.split is not None and not np.any(off.split), "off does not solve the halves"
    armed = track_truth(frames, motions, INTR, split_tol_sigma=3.0)
    assert armed.kept == off.kept and armed.rejected["split"] == 0
    assert armed.split is not None and 0.0 < float(np.max(armed.split)) < 3.0
    tight = track_truth(frames, motions, INTR, split_tol_sigma=0.05)
    assert tight.rejected["split"] > 0 and tight.kept < off.kept
    assert tight.split is not None and float(np.max(tight.split)) <= 0.05


# ---- the forward ruler: a corner born once and followed one hop a frame -----------------------
SMALL = Intrinsics(fx=200.0, fy=200.0, cx=160.0, cy=90.0, width=320, height=180)
WALL_M = 2.0  # the one textured plane the small scene is made of
SLIDE_M = 0.02  # the camera's sidestep between frames: exactly 2 px on the wall at SMALL's optics
SLIDE_S = 0.15  # seconds between frames: 40 of them span 5.85 s, which a 5 s window can fill
SMALL_MARGIN = 96  # spare columns either side, so 40 frames of 2 px may shift into them


def wall_frames(
    count: int, step_m: float = SLIDE_M, dt: float = SLIDE_S
) -> tuple[list[np.ndarray], dict[float, RigidPose]]:
    """A camera sidestepping past one fronto-parallel textured wall ``WALL_M`` away: the view
    at each frame (a whole-pixel shift, so there is nothing for the tracker to round) and the
    cart pose at each stamp. Small on purpose — a unit test pays for every flow call."""
    rng = np.random.default_rng(11)
    fine = rng.integers(0, 255, size=(SMALL.height, SMALL.width + 2 * SMALL_MARGIN))
    blurred = fine.astype(float)
    for axis in (0, 1):
        blurred = np.mean([np.roll(blurred, s, axis=axis) for s in range(-3, 4)], axis=0)
    wide = np.rint(blurred).astype(np.uint8)
    frames, poses = [], {}
    for i in range(count):
        shift = SMALL_MARGIN + round(SMALL.fx * step_m * i / WALL_M)
        frames.append(np.ascontiguousarray(wide[:, shift : shift + SMALL.width]))
        poses[round(dt * i, 3)] = planar_pose(0.0, -step_m * i, 0.0)
    return frames, poses


def wall_context(stamp: float, gray: np.ndarray, odometry: Odometry) -> FrameContext:
    """The small scene's frame context: the same fake motion source, SMALL's optics."""
    mount = CameraPose(0.0, 0.0, 1.23, 0.0)
    return FrameContext(
        SMALL,
        mount,
        stamp=stamp,
        gray=gray,
        motion=odometry,
        cam_optical=CameraPlacement.of(mount),
    )


def drive(
    anchor: ParallaxAnchor, frames: list[np.ndarray], poses: dict[float, RigidPose]
) -> Pairs | None:
    """Every frame of the small scene through the anchor, in order; the last frame's pairs."""
    network = np.full((SMALL.height, SMALL.width), 3.0)
    odometry = Odometry(poses)
    pairs = None
    for stamp, view in zip(sorted(poses), frames, strict=True):
        pairs = anchor.pairs(Frame(network, wall_context(stamp, view, odometry)))
    return pairs


def test_a_corner_is_born_once_and_is_still_followed_forty_frames_later() -> None:
    """The whole point of following corners FORWARD: the cost is two flow calls a frame
    whatever the window, so a corner detected in the first frame is still the same corner six
    seconds later, resting on observations spread over the whole window. The depths it gives
    are the wall's, and the report line says how many corners the store is holding."""
    frames, poses = wall_frames(40)
    anchor = ParallaxAnchor(track_window_s=5.0)
    pairs = drive(anchor, frames, poses)
    store = anchor._store
    assert store is not None and store.live > 100
    assert max(track.hops for track in store._tracks) >= 30, "a corner must survive the window"
    assert float(np.median(anchor._obs)) >= 5, "a track must rest on more than a pair's two views"
    assert pairs is not None and pairs.size >= 20
    assert abs(float(np.median(pairs.z)) / WALL_M - 1.0) < 0.05
    assert "live corners" in anchor.describe() and "hop" in anchor.describe()


@pytest.mark.slow
def test_a_longer_window_is_a_wider_baseline_and_a_smaller_sigma() -> None:
    """The reason the window may be long at all: every error term of a parallax depth divides
    by the baseline, and following forward means a longer window costs no more flow. The same
    forty pictures read at 1, 3 and 5 seconds must widen the baseline and shrink the sigma."""
    frames, poses = wall_frames(40)
    sigma: list[float] = []
    baseline: list[float] = []
    for window_s in (1.0, 3.0, 5.0):
        anchor = ParallaxAnchor(track_window_s=window_s)
        drive(anchor, frames, poses)
        assert anchor.sigma_m is not None, f"{window_s} s triangulated nothing"
        sigma.append(anchor.sigma_m)
        baseline.append(float(np.median(anchor._baseline)))
    assert baseline[0] < baseline[1] < baseline[2], f"the baselines did not grow: {baseline}"
    assert sigma[0] > sigma[1] > sigma[2], f"the sigma did not fall: {sigma}"


def test_a_window_shortened_live_uses_fewer_of_the_observations_it_already_holds() -> None:
    """A knob turned in the field takes effect on the next frame and resets nothing: the store
    keeps its corners, their hops and their observations, and simply stops reaching for the
    ones the new window no longer covers."""
    frames, poses = wall_frames(45)
    stamps = sorted(poses)
    network = np.full((SMALL.height, SMALL.width), 3.0)
    odometry = Odometry(poses)
    anchor = ParallaxAnchor(track_window_s=5.0)
    for stamp, view in zip(stamps[:40], frames[:40], strict=True):
        anchor.pairs(Frame(network, wall_context(stamp, view, odometry)))
    store = anchor._store
    assert store is not None
    wide_obs, wide_baseline = anchor._obs[-1], anchor._baseline[-1]
    corners, hops = store.live, max(track.hops for track in store._tracks)
    oldest = min(o.stamp for t in store._tracks for o in t.observations)
    assert stamps[39] - oldest > 4.0, "a 5 s window must be reaching back 5 s"
    anchor.track_window_s = 1.0
    anchor.pairs(Frame(network, wall_context(stamps[40], frames[40], odometry)))
    oldest = min(o.stamp for t in store._tracks for o in t.observations)
    assert stamps[40] - oldest <= 1.0, "the old observations must go on the very next frame"
    assert max(stamps[40] - v.stamp for v in store.views()) <= 1.0
    # the only corners lost are the ones that frame's own flow lost; nothing is reset
    assert store.live >= corners - 5, "shortening the window must not throw the corners away"
    assert max(track.hops for track in store._tracks) == hops + 1
    for stamp, view in zip(stamps[41:], frames[41:], strict=True):
        anchor.pairs(Frame(network, wall_context(stamp, view, odometry)))
    assert anchor._obs[-1] < wide_obs, "a shorter window is fewer observations"
    assert anchor._baseline[-1] < wide_baseline


class SlidingFlow:
    """A flow whose HOP slides ``per_hop`` pixels sideways every frame while a direct re-track
    from an older picture still lands where the corner really is — the drift Lucas-Kanade does
    along an edge, invisible to a per-hop forward-backward check because every single hop is
    self-consistent."""

    def __init__(self, per_hop: float) -> None:
        self.per_hop = per_hop
        self._real = LucasKanade()

    def track(
        self,
        a: npt.NDArray[np.uint8],
        b: npt.NDArray[np.uint8],
        points: np.ndarray,
        *,
        backward: bool = True,
        guess: np.ndarray | None = None,
    ) -> tuple[np.ndarray, npt.NDArray[np.bool_], np.ndarray]:
        """The real flow, nudged sideways on a hop (no guess) and honest on a verification."""
        landed, kept, drift = self._real.track(a, b, points, backward=backward, guess=guess)
        if guess is None:
            landed = landed + np.array([self.per_hop, 0.0], dtype=np.float32)
        return landed, kept, drift


def test_the_drift_bound_closes_a_corner_that_slid_off_its_own_birth_patch() -> None:
    """Thirty hops of two pixels are sixty pixels and a depth that is wrong and consistent, and
    no per-hop check can see it. Re-tracking each corner directly from the picture its oldest
    kept view was taken in does see it: with the bound armed the sliding corners are closed and
    counted, and with it off they live on."""
    frames, poses = wall_frames(30)
    place = CameraPlacement.of(CameraPose(0.0, 0.0, 1.23, 0.0))
    store = TrackStore(window_s=5.0, verify_every=5, drift_tol_px=1.0, flow=SlidingFlow(2.0))
    loose = TrackStore(window_s=5.0, verify_every=0, drift_tol_px=1.0, flow=SlidingFlow(2.0))
    caught = 0
    for stamp, view in zip(sorted(poses), frames, strict=True):
        caught += store.follow(view, stamp, place, "odom").died["drift"]
        loose.follow(view, stamp, place, "odom")
    assert caught > 20, f"the bound closed only {caught} sliding corners"
    assert loose.live > store.live, "with the bound off the sliding corners must survive"


def test_the_drift_bound_leaves_a_corner_that_did_not_slide_alone() -> None:
    """The other half of the bound: an honest flow over the same thirty frames loses nobody to
    drift, so the gate is not simply closing whatever it re-tracks."""
    frames, poses = wall_frames(30)
    place = CameraPlacement.of(CameraPose(0.0, 0.0, 1.23, 0.0))
    store = TrackStore(window_s=5.0, verify_every=5, drift_tol_px=1.0)
    killed = 0
    for stamp, view in zip(sorted(poses), frames, strict=True):
        killed += store.follow(view, stamp, place, "odom").died["drift"]
    # not zero as a matter of principle: a direct re-track five seconds back lands a pixel off
    # now and then on noise alone. Three of two hundred is the false-positive rate, against the
    # sliding flow's twenty-odd in the test above.
    assert killed <= 3, f"an honest flow lost {killed} corners to the drift bound"
    assert store.live > 100


def test_a_track_s_solve_never_spans_two_motion_sources() -> None:
    """The rule with no exception: the wheels and the tracker's map pose disagree by about a
    quarter over a second, so a bundle half measured by each is not a geometry at all. The
    source changes in the middle of the drive; from that frame on, not one observation from
    before it is offered to a solve, and every track's own views agree with one another."""
    frames, poses = wall_frames(20)
    place = CameraPlacement.of(CameraPose(0.0, 0.0, 1.23, 0.0))
    store = TrackStore(window_s=5.0)
    stamps = sorted(poses)
    turn = stamps[10]
    for stamp, view in zip(stamps, frames, strict=True):
        source = "odom" if stamp < turn else "tracker"
        store.follow(view, stamp, place, source)
        assert all(v.source == source for v in store.views()), "a view of the other source"
        for track in store._tracks:
            assert all(obs.source == source for obs in track.used)
    assert min((v.stamp for v in store.views()), default=turn) >= turn
    assert store.live > 100, "a track outlives the change of source: only its bundle restarts"


def necked(pan: float, lens: tuple[float, float, float] = (0.0, 0.0, 1.23)) -> CameraPlacement:
    """``base_link <- camera_optical`` for a head panned ``pan`` radians with the lens at
    ``lens`` on the cart — the TF edge a moving neck publishes on every frame."""
    turn = np.array(
        [[math.cos(pan), -math.sin(pan), 0.0], [math.sin(pan), math.cos(pan), 0.0], [0, 0, 1.0]]
    )
    mount = CameraPlacement.of(CameraPose(lens[0], lens[1], lens[2], 0.0))
    return CameraPlacement(turn @ np.asarray(mount.rotation), np.array(lens))


def wall_frame_context(
    stamp: float, gray: np.ndarray, odometry: Odometry, place: CameraPlacement
) -> FrameContext:
    """The small wall scene's context with a neck edge of the test's own choosing."""
    return replace(wall_context(stamp, gray, odometry), cam_optical=place)


def test_a_camera_pose_and_a_cart_pose_with_its_neck_are_the_same_motion() -> None:
    """The refactor's own invariant, stated so it cannot rot: composing the neck's edge into
    the stored pose (``map <- camera_optical``, one composition when the frame is taken) and
    composing it at solve time (``map <- base_link`` through
    :func:`pepin.parallax.camera_motion`, two) are the same transform to floating point. So
    this is not a repair of a wrong number — the neck was always carried — it is one
    composition instead of two, and a stored pose that means something on its own."""
    rng = np.random.default_rng(5)
    for _ in range(8):
        stood = [planar_pose(*rng.uniform(-2.0, 2.0, 3)) for _ in range(2)]
        necks = [necked(rng.uniform(-0.6, 0.6), tuple(rng.uniform(-0.3, 1.4, 3))) for _ in range(2)]
        through_base = camera_motion(*_between(stood[0], stood[1]), necks[0], necks[1])
        lenses = [_placed(pose, neck) for pose, neck in zip(stood, necks, strict=True)]
        through_lens = Motion(*_between(lenses[0], lenses[1]))
        assert np.allclose(through_base.rotation, through_lens.rotation, atol=1e-12)
        assert np.allclose(through_base.translation, through_lens.translation, atol=1e-12)


def test_a_head_that_turns_over_a_standing_cart_is_a_rotation_and_not_a_baseline() -> None:
    """The head will pan and tilt while the cart drives, and a window whose motions were the
    BASE's would read them as if the camera had stood still. Here the cart never moves and the
    neck pans 10 degrees across the window about the lens itself: the camera's own motion is a
    pure rotation, there is no baseline anywhere in it, and the stage says ``rotation-only``
    instead of triangulating rays that never crossed."""
    frames, poses = wall_frames(12)
    stamps = sorted(poses)
    still = {t: planar_pose(0.0, 0.0, 0.0) for t in stamps}  # the cart does not move at all
    anchor = ParallaxAnchor(track_window_s=3.0)
    network = np.full((SMALL.height, SMALL.width), 3.0)
    odometry = TfPoses(still)
    for i, (stamp, view) in enumerate(zip(stamps, frames, strict=True)):
        odometry.now = stamp
        place = necked(math.radians(10.0) * i / (len(frames) - 1))
        assert (
            anchor.pairs(Frame(network, wall_frame_context(stamp, view, odometry, place))) is None
        )
    # 'rotation-only' on every frame whose window spans more than the gyro's own degree, and
    # 'still' on the one frame whose two views are 0.9 degrees apart: both are refusals
    assert anchor.rejected.get("rotation-only", 0) >= 10, "a pure pan must say so"
    assert not anchor._baseline, "a turning head must not invent a baseline"


def test_the_baseline_is_the_lens_s_own_path_and_not_the_cart_s() -> None:
    """The other way round: the cart stands still and the NECK carries the lens sideways, which
    is exactly the motion the pictures were rendered from. A window built on the base's poses
    would find no baseline at all and give nothing; built on the lens's it reads the wall."""
    frames, poses = wall_frames(16)
    stamps = sorted(poses)
    still = {t: planar_pose(0.0, 0.0, 0.0) for t in stamps}
    anchor = ParallaxAnchor(track_window_s=3.0)
    network = np.full((SMALL.height, SMALL.width), 3.0)
    odometry = TfPoses(still)
    pairs = None
    for i, (stamp, view) in enumerate(zip(stamps, frames, strict=True)):
        odometry.now = stamp
        place = necked(0.0, (0.0, -SLIDE_M * i, 1.23))  # the neck sidesteps the lens itself
        pairs = anchor.pairs(Frame(network, wall_frame_context(stamp, view, odometry, place)))
    assert pairs is not None and pairs.size >= 20, "the lens moved, so the corners triangulate"
    assert abs(float(np.median(pairs.z)) / WALL_M - 1.0) < 0.05
    travelled = float(np.median(anchor._baseline))
    assert travelled > 0.1, f"the baseline must be the lens's path, not the cart's: {travelled}"


DOOR_LENS = (-0.150009, -0.129449, -0.002416, -0.001635, 0.091751)  # config/camera.json's own


def bend(points: np.ndarray, intr: Intrinsics, dist: tuple[float, ...]) -> np.ndarray:
    """Where a plumb-bob lens really puts the pixels a pinhole would have placed at ``points``
    — the forward model, so the test bends a picture the way the camera does and asks the code
    to unbend it."""
    k1, k2, p1, p2, k3 = dist
    x = (points[..., 0] - intr.cx) / intr.fx
    y = (points[..., 1] - intr.cy) / intr.fy
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([intr.fx * xd + intr.cx, intr.fy * yd + intr.cy], axis=-1)


def test_a_bent_pixel_is_straightened_back_to_where_a_pinhole_would_have_put_it() -> None:
    """The lens this cart carries is 83 degrees wide and calibrated, and the published picture
    is the raw one: a pixel at the top edge of a 640x360 frame sits several pixels from where a
    pinhole would have put it, against an epipolar gate 1.5 px wide. Undoing it is a round trip
    to a hundredth of a pixel."""
    lens = PlumbBob(INTR, DOOR_LENS)
    grid = np.array(
        [[x, y] for x in (0.0, 160.0, 320.0, 480.0, 639.0) for y in (0.0, 90.0, 180.0, 359.0)],
        dtype=np.float32,
    )
    bent = bend(grid, INTR, DOOR_LENS).astype(np.float32)
    back = lens.straighten(bent)
    assert np.allclose(back, grid, atol=0.02), "the lens must undo itself"
    moved = np.linalg.norm(bent - grid, axis=1)
    assert moved.max() > 20.0, "a wide lens moves a corner pixel by more than the gate"
    assert float(np.median(moved[grid[:, 1] == 0.0])) > 3.0, "and the top edge by more than it"
    assert float(np.median(moved[grid[:, 1] == 180.0])) < float(np.median(moved))


def test_every_view_is_straightened_with_the_lens_its_own_frame_published() -> None:
    """A corner is straightened when its frame is taken, not when a solve reads it, so an
    observation carries the lens camera_info published for THAT frame. A camera that is
    re-calibrated mid-drive, or whose picture camera_stream starts rectifying, therefore leaves
    the observations already stored exactly as they were measured."""
    frames, poses = wall_frames(8)
    stamps = sorted(poses)
    anchor = ParallaxAnchor(track_window_s=3.0)
    network = np.full((SMALL.height, SMALL.width), 3.0)
    odometry = Odometry(poses)
    seen: list[tuple[float, ...]] = []
    marked: dict[int, np.ndarray] = {}
    for i, (stamp, view) in enumerate(zip(stamps, frames, strict=True)):
        lens = DOOR_LENS if i < 4 else ()  # the picture becomes a rectified one halfway through
        ctx = replace(wall_context(stamp, view, odometry), dist=lens)
        anchor.pairs(Frame(network, ctx))
        store = anchor._store
        assert store is not None
        seen.append(() if store.lens is None else store.lens.dist)
        if i == 3:  # the observations made while the lens was still in the picture
            marked = {
                t.ident: t.observations[0].pixel.copy() for t in store._tracks if t.observations
            }
    assert seen[0] == DOOR_LENS and seen[-1] == (), "the lens follows the frame's own camera_info"
    store = anchor._store
    assert store is not None
    after = {t.ident: t.observations[0].pixel for t in store._tracks if t.observations}
    shared = set(marked) & set(after)
    assert len(shared) > 50, "corners must live across the change for the question to mean anything"
    for ident in shared:
        assert np.array_equal(marked[ident], after[ident]), "a stored view was measured again"


def test_the_geometry_measures_the_straightened_pixels_and_the_depth_the_picture_s_own() -> None:
    """A bundle of exact tracks bent by the lens: read as pinhole pixels the depths are wrong
    and the epipolar gate throws them away, straightened they come back. The pixel the caller
    indexes the depth image with stays the PICTURE's own, because that is where the corner is."""
    points, tracks = moving_scene(6, noise_px=0.0)
    bent = bend(np.asarray(tracks.pixels), INTR, DOOR_LENS)
    lens = PlumbBob(INTR, DOOR_LENS)
    straight = lens.straighten(bent.reshape(-1, 2)).reshape(bent.shape)
    raw = np.asarray(bent[:, -1], dtype=np.float32)
    as_seen = replace(tracks, pixels=bent)
    as_meant = replace(tracks, pixels=straight.astype(float), at=raw)
    wrong = gate_tracks(as_seen, INTR)
    right = gate_tracks(as_meant, INTR)

    def missed(truth: ParallaxTruth) -> float:
        """The median relative depth error of a truth against the scene it was built from."""
        row = np.array([np.argmin(np.abs(points[:, 2] - z)) for z in truth.z])
        return float(np.median(np.abs(truth.z / points[row, 2] - 1.0)))

    # exact correspondences, so every pixel the epipolar gate drops is the lens's doing
    assert wrong.rejected["epipolar"] > 10 * right.rejected["epipolar"]
    assert missed(right) < 0.0005 < missed(wrong), f"{missed(right)} against {missed(wrong)}"
    assert right.kept > wrong.kept
    # and every reported pixel is the picture's own, not the straightened one
    seen_at = {(round(float(c)), round(float(r))) for c, r in right.points}
    picture = {(round(float(c)), round(float(r))) for c, r in raw}
    assert seen_at <= picture


class TfPoses(Odometry):
    """TF's two halves as a motion source, the way a node's FramePoser reads them: a slow
    ``map <- odom`` correction that changes only when the tracker relocalises, and a smooth
    ``odom <- base_link`` at every frame. ``map_motion_recent`` is deliberately blind past
    ``covers`` — this is the shape that broke the window live: the tracker's own map pose is
    published behind the frames and covers their stamps only sometimes."""

    def __init__(
        self,
        poses: dict[float, RigidPose],
        *,
        jump_at: float | None = None,
        jump_m: float = 0.0,
        covers: float = float("inf"),
    ) -> None:
        super().__init__(poses)
        self._jump_at = jump_at
        self._jump_m = jump_m
        self._covers = covers
        self.now = min(poses)  # the frame being processed: which correction is the newest one

    def correction(self) -> RigidPose:
        """``map <- odom`` as it stands at the frame being processed."""
        moved = 0.0 if self._jump_at is None or self.now < self._jump_at else self._jump_m
        return planar_pose(moved, 0.0, 0.0)

    def map_correction_recent(self) -> RigidPose:
        """The newest ``map <- odom`` TF holds."""
        return self.correction()

    def map_pose_recent(self, stamp: float) -> RigidPose | None:
        """The newest correction composed with the odometry at ``stamp`` — the pose that exists
        on every frame."""
        on_odom = self._poses.get(stamp)
        if on_odom is None:
            return None
        fix = self.correction()
        return RigidPose(
            fix.rotation @ on_odom.rotation, fix.rotation @ on_odom.translation + fix.translation
        )

    def map_motion_recent(
        self, from_stamp: float, to_stamp: float, max_age_s: float
    ) -> RigidPose | None:
        """The tracker's own map pose at both stamps, which it has for only ``covers``
        seconds of history."""
        if self.now - from_stamp > self._covers:
            return None
        a, b = self.map_pose_recent(from_stamp), self.map_pose_recent(to_stamp)
        return None if a is None or b is None else base_motion(a, b)


def feed(
    anchor: ParallaxAnchor,
    frames: list[np.ndarray],
    poses: dict[float, RigidPose],
    source: Odometry,
) -> Pairs | None:
    """Every frame of the small wall scene through the anchor on a given motion source."""
    network = np.full((SMALL.height, SMALL.width), 3.0)
    pairs = None
    for stamp, view in zip(sorted(poses), frames, strict=True):
        if isinstance(source, TfPoses):
            source.now = stamp
        pairs = anchor.pairs(Frame(network, wall_context(stamp, view, source)))
    return pairs


def test_a_map_pose_built_from_tf_s_two_halves_never_cuts_the_window() -> None:
    """What broke live on 2026-09-15: the tracker's own map pose covers a frame's stamp only
    sometimes, a bundle may not span two sources, and so every fallback cut the window — a 5 s
    window never reached past 2.1 s. Given a tracker that can only answer for the last half
    second, the tf source still reaches the whole window, because it asks the slow half of TF
    for its newest value and the fast half for this moment and therefore always has an answer."""
    frames, poses = wall_frames(30)
    blinkered = TfPoses(poses, covers=0.5)
    on_tf = ParallaxAnchor(track_window_s=4.0, motion_source="tf")
    on_tracker = ParallaxAnchor(track_window_s=4.0, motion_source="tracker")
    for anchor, source in ((on_tf, blinkered), (on_tracker, TfPoses(poses, covers=0.5))):
        feed(anchor, frames, poses, source)
    assert on_tf.gap_s is not None and on_tf.gap_s > 3.0, "tf must reach the whole window"
    assert on_tf.deaths.get("source", 0) == 0, "one source cannot be cut"
    # the tracker's own pose cannot reach past what it covers, so its window is cut back to
    # half a second — which at this window's view spacing is not even three observations, and
    # the frame yields nothing at all. That is the live failure, reproduced.
    assert on_tracker.gap_s is None or on_tracker.gap_s < 1.0
    assert not on_tracker._obs or float(np.median(on_tf._obs)) > float(np.median(on_tracker._obs))
    assert on_tracker.rejected.get("mixed motion", 0) > 0, "its fallbacks must cut the window"
    assert on_tf.rejected.get("mixed motion", 0) == 0
    assert "TF's continuous map pose" in on_tf.describe()


def test_a_tracker_correction_inside_the_window_is_caught_by_the_correction_gate() -> None:
    """The price of a pose that is the tracker's estimate AS OF ITS OWN FRAME: a relocalisation
    moves every view stored before it relative to every view after it, and the bundle reads a
    displacement the camera never made. The gate watches the correction itself and drops the
    window it has just invalidated — the corners live on."""
    frames, poses = wall_frames(30)
    jump_at = sorted(poses)[15]
    gated = ParallaxAnchor(track_window_s=4.0, motion_source="tf")
    ungated = ParallaxAnchor(track_window_s=4.0, motion_source="tf", correction_tol_m=0.0)
    for anchor in (gated, ungated):
        feed(anchor, frames, poses, TfPoses(poses, jump_at=jump_at, jump_m=0.20))
    assert gated.rejected.get("correction", 0) > 0, "a 20 cm relocalisation must drop the window"
    assert "correction gate 5 cm" in gated.describe()
    assert ungated.rejected.get("correction", 0) == 0
    assert "correction ungated" in ungated.describe()
    store = gated._store
    assert store is not None and store.live > 100, "the corners outlive the correction"
    # the frames after the jump are measured again, and the depths come back to the wall
    assert gated.sigma_m is not None


def test_a_correction_smaller_than_the_gate_rides_along_inside_the_window() -> None:
    """The gate is a gate and not a reset: a correction of a centimetre is the tracker doing
    its job, it lands in the observations as the pose change it is, and the window is kept."""
    frames, poses = wall_frames(30)
    jump_at = sorted(poses)[15]
    anchor = ParallaxAnchor(track_window_s=4.0, motion_source="tf")
    pairs = feed(anchor, frames, poses, TfPoses(poses, jump_at=jump_at, jump_m=0.01))
    assert anchor.rejected.get("correction", 0) == 0
    assert anchor.gap_s is not None and anchor.gap_s > 3.0
    assert pairs is not None and pairs.size >= 20


@pytest.mark.slow
def test_the_describer_follows_corners_forward_too() -> None:
    """The forward store with ``orb``: the frame is described once, its keypoints matched
    against the descriptors the live corners carried out of the previous frame, and whatever
    nobody recognised becomes a new corner. The drift bound does not apply — a match is a
    recognition, not a hop — and the depths are still the wall's."""
    frames, poses = wall_frames(16)
    anchor = ParallaxAnchor(matcher="orb", track_window_s=2.0)
    pairs = drive(anchor, frames, poses)
    store = anchor._store
    assert store is not None and store.live > 50
    assert max(track.hops for track in store._tracks) >= 8, "a keypoint must be re-recognised"
    assert anchor.deaths.get("drift", 0) == 0, "the drift bound is a flow's, not a describer's"
    assert pairs is not None and pairs.size >= 10
    assert abs(float(np.median(pairs.z)) / WALL_M - 1.0) < 0.10


def test_the_forward_ruler_emits_the_pairs_the_window_ruler_does() -> None:
    """The interface downstream is untouched: whoever followed the corners, a frame comes out
    as the same (network depth, triangulated depth, lift, weight, left) pool, with the track's
    own observation count and two-view sigma filled in beside it."""
    frames, poses = wall_frames(12)
    forward = ParallaxAnchor(track_window_s=1.5)
    windowed = ParallaxAnchor(tracking="window", track_window_s=1.5)
    for anchor in (forward, windowed):
        pairs = drive(anchor, frames, poses)
        assert pairs is not None and pairs.size >= 20
        assert pairs.d.shape == pairs.z.shape == pairs.lift.shape == pairs.weight.shape
        assert pairs.left.shape == pairs.d.shape
        assert np.all(np.isfinite(pairs.z)) and np.all(pairs.weight > 0)
        assert abs(float(np.median(pairs.z)) / WALL_M - 1.0) < 0.05
        assert anchor._obs and anchor._sigma_two
