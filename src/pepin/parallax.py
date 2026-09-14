"""Depth from motion: the metre the camera gets from the cart's own movement, with no lidar.

The monocular network gives the shape of the room and the wrong size (:mod:`pepin.depth`), and
every source of truth the pipeline has so far comes from the lidar's plane or from the floor's
geometry. Motion is a third one, and it needs no second sensor: between two frames A and B the
camera moved by a transform the odometry knows to the centimetre (wheels and gyro over a tenth
of a second), so a point seen in both is the intersection of two rays with a known baseline
between their origins — one unknown per point, and the intersection is a depth in metres.

The pieces, in the order they run. :func:`match` tracks corners from A into B (good features +
pyramidal Lucas-Kanade, with a forward-backward check: indoor texture is edges and clutter, and
a tracker that follows a corner and then follows it back to within half a pixel is the cheapest
honest filter there is). :func:`sampson` measures how far each correspondence sits from the
epipolar geometry of the *known* motion — nothing is fitted here, so there is no model to
sample for: the consensus test a RANSAC would end at is available directly, and a point that
misses the epipolar line by more than a pixel and a half moved on its own (a person, a reflection,
a mistracked corner). :func:`triangulate` meets the two rays and returns the depth along B's
optical axis and its own sigma.

What the sigma is, because it decides the weights. A correspondence is a disparity, and a
disparity's noise is the tracker's, about half a pixel. Depth from disparity is
``z = f * b / disparity``, so ``sigma_z = z^2 * sigma_px / (f * b)`` — the quadratic growth with
range every stereo rig has, with the baseline ``b`` here being only the part of the motion
perpendicular to the ray (a camera driving straight at a wall has no parallax on the wall's
centre however far it goes). In inverse depth — the space the laws are fitted in — that same
statement is ``sigma_(1/z) = sigma_px / (f * b_perp)``, which does not depend on range at all:
a far point's inverse depth is as well determined as a near one's. So a pair's weight against a
lidar beam's 1 is the ratio of their inverse-depth variances, capped at 1.

Limits worth knowing before trusting a number: a pure rotation gives no parallax (no pairs,
and the anchor says so); a point on the epipole — straight ahead when driving straight — has no
parallax either whatever the baseline; and the odometry's own error is the floor under every
depth here, so a 2 cm baseline error on a 15 cm baseline is 13 % on every depth it yields.
The module is pure numpy and OpenCV: no ROS, no pipeline, and cv2 is imported inside
:func:`match` so a test that never tracks never pays for it.

Measured on runs 0171 and 0165 against the lidar's own ranges
(scratch/parallax_vs_lidar.py / .txt, 2026-09-12): 3-5 ms a frame at 640x360, 30-190 pairs on a
frame where the cart really stepped, spread over the whole picture (41-56 % in the bottom third,
10-23 % in the top, against the lidar's single row). The depth itself is as good as the step,
and on those two runs the cart crawls: at a 0.5 s gap the median perpendicular baseline is
2.9 cm and only 8 of 29 frame pairs carry 5 cm of odometry. What survives the gates is not
wrong by one factor but by a factor that depends on the range: +25-37 % too far under 1.5 m
(1.365 of the lidar on 0171 over 19 samples, 1.246 on 0165 over 30) and unbiased from 1.5 to
3 m (0.965 over 16, 1.008 over 27). The odometry's own step has a +-25 % band against the
tracker's (0.79-1.26 on 0171, 0.56-1.24 on 0165) and multiplies every parallax depth one for
one, but a scale error is the same factor at every range and cannot leave the far band at 1.00
while the near one reads 1.25-1.37: the cause of that structure is not yet known, and
0.5-1.5 m is the band a close approach and the costmap live in. The weights know the baseline
at least: a 3 cm one weighs 0.02 of a lidar beam, a 15 cm one 0.4. Two things not measured
there would move the number: a calibrated focal length (the depth is proportional to fx and the
optics are still the nominal 78 degree guess) and a run at driving speed.

Both arrived on 2026-09-14 (scratch/parallax_baseline_sweep.py over the errand of 14:12 at
0.2-0.3 m/s, fx 724.1, 4700 points matched to the lidar): the range-dependent bias was the thin
baseline's own skew. Binned by each point's perpendicular baseline, the far field reads 0.75 and
0.49 of the lidar at 1.5-2 and 2-3 m on 2 cm of parallax and 1.02-1.13 from 5 cm on, while the
sigma follows 1 / b as the model says (16.3 cm at 2 cm, 12.2 at 5, 6.9 at 9, 4.1 at 18). A
+9 to +13 % offset at 1.0-1.5 m survives every baseline and is not explained. The gates trade
places with the gap: the epipolar test takes ~45 % of the corners at every short gap, the
parallax test 15.9 % at 0.1 s against 2.0 % at 1.5 s, and the flow 7.8 % against 76.1 % — the
tracker's window follows the longer step up to about 0.6 s, past which the points kept per frame
fall to single figures.

Which is what the describer is for, and what it is not (scratch/parallax_matcher_sweep.py over
all four errands of 2026-09-14, both matchers on the same frame pairs, 2026-09-14): ORB keeps 11
pairs a frame at gaps of 1.0 and 1.5 s where the flow keeps 2 and 0, at 8.4 ms a frame against
the flow's 4.6, and it does reach a baseline the flow cannot hold a pair across. It reaches it at
the wrong depth. At the 0.5 s gap the anchor really pairs across, the flow reads 0.993 of the
lidar at 1.5-2 m against the describer's 1.031, with half the noise (13.5 cm a pair against
29.6): a keypoint is placed to about a pixel where a tracked corner is placed to half of one,
which is why ``orb`` is credited with :data:`ORB_DISPARITY_SIGMA_PX`. And past a second of gap
both matchers read 1.25-1.45 of the lidar at 1.5-2 m, together: the odometry's drift over that
second inflates the baseline every depth is proportional to, so the gap is capped by the pose,
not by the matcher. The flow is the default; the describer is the path for a robot whose pose
over 1.5 s is better than this cart's wheels and gyro.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from pepin.depth import NEAR_M, Array, CameraPose, Intrinsics

Pixels = npt.NDArray[np.float32]  # (n, 2) columns and rows, sub-pixel

# ---- the tracker --------------------------------------------------------------------------
MAX_CORNERS = 400  # corners asked of one frame: 400 on 640x360 costs ~2 ms and fills the room
CORNER_QUALITY = 0.01  # a corner weaker than this share of the best is not a corner
CORNER_MIN_DISTANCE = 8  # pixels between two corners: a cluster on one poster is one measurement
LK_WINDOW = 21  # the flow's window in pixels, at every pyramid level
LK_LEVELS = 3  # pyramid levels: 3 follows ~40 px of motion, a tenth of a second at walking pace
FB_TOL_PX = 0.5  # a corner tracked to B and back must land this close to where it started

# ---- the describer ------------------------------------------------------------------------
# The flow follows a corner; the describer recognises one. Past about half a second of gap the
# view has moved further than the flow's window and the tracker loses three corners in four,
# while a descriptor does not care how far a point travelled as long as it still looks the same.
ORB_FEATURES = 1200  # keypoints asked of one frame: 640x360 has that many corners in a room
ORB_FAST_THRESHOLD = 12  # how much brighter than its ring a pixel must be to start a keypoint
ORB_RATIO = 0.75  # Lowe: a match whose second-best is this close is ambiguous, not a match
ORB_DISPARITY_SIGMA_PX = 1.0  # what a keypoint's octave knows its place to, against the flow's 0.5
MATCHERS = ("klt", "orb")
Matcher = Literal["klt", "orb"]

# ---- the gates ----------------------------------------------------------------------------
MIN_BASELINE_M = 0.02  # a shorter move than this is standing still: 2 cm is the odometry's own
MIN_ROTATION_RAD = math.radians(1.0)  # a turn under this is not a turn (gyro noise)
MIN_PARALLAX_RATIO = 0.01  # perpendicular baseline over depth: under 1 % the depth is a guess
EPIPOLE_MIN_DEG = 3.0  # a ray this close to the direction of travel has no parallax to read
MAX_SAMPSON_PX = 1.5  # distance from the known motion's epipolar line before a point is dropped
MAX_REPROJ_PX = 1.5  # the triangulated point's own reprojection error in B

# ---- the noise ----------------------------------------------------------------------------
DISPARITY_SIGMA_PX = 0.5  # what the flow knows a corner's place to
LIDAR_SIGMA_INV = 0.005  # 1/m: a lidar beam's inverse-depth noise (2 cm at 2 m), the weight's 1
MAX_WEIGHT = 1.0  # no parallax pair outweighs a lidar beam

REASONS = (
    "still",
    "rotation-only",
    "flow",
    "outside",
    "epipolar",
    "behind",
    "parallax",
    "epipole",
    "reproj",
)


@dataclass(frozen=True)
class Motion:
    """A rigid transform between two camera views in the optical frame (x right, y down,
    z forward): a point ``X`` of view A is ``rotation @ X + translation`` in view B."""

    rotation: Array
    translation: Array

    @property
    def baseline(self) -> float:
        """How far the camera moved between the views, in metres."""
        return float(np.linalg.norm(self.translation))

    @property
    def angle(self) -> float:
        """How far the camera turned between the views, in radians."""
        trace = float(np.trace(np.asarray(self.rotation, dtype=float)))
        return math.acos(float(np.clip(0.5 * (trace - 1.0), -1.0, 1.0)))


def optical_from_base(cam: CameraPose) -> Array:
    """The 3x3 rotation taking a base_link direction into the camera's optical axes (x right,
    y down, z forward) for a camera mounted with ``cam``'s pitch — the rotation
    :func:`pepin.depth.project_all` applies inline before dividing by the depth."""
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    return np.array([[0.0, -1.0, 0.0], [-s, 0.0, -c], [c, 0.0, -s]])


class Placement(Protocol):
    """Where the lens sat for one frame: ``base_link <- camera_optical``, a 3x3 rotation and a
    translation — :class:`pepin.tsdf.RigidPose`, what
    :meth:`pepin.frame_pose.FramePoser.camera_in_base` reads off the neck's live TF edge."""

    @property
    def rotation(self) -> Array:
        """The 3x3 rotation of the optical axes in base_link (pitch and pan alike)."""
        ...

    @property
    def translation(self) -> Array:
        """Where the lens sits in base_link, in metres."""
        ...


@dataclass(frozen=True)
class CameraPlacement:
    """A :class:`Placement` built by hand rather than read from TF."""

    rotation: Array
    translation: Array

    @classmethod
    def of(cls, cam: CameraPose) -> CameraPlacement:
        """The placement of a pitch-only pose — config/camera.json's mount, the stand-in while
        TF has no neck edge. A pan it cannot know about is taken to be zero."""
        return cls(optical_from_base(cam).T, np.array([cam.x, cam.y, cam.z]))


def camera_motion(
    base_rotation: Array, base_translation: Array, cam_a: Placement, cam_b: Placement
) -> Motion:
    """The motion between two views in optical coordinates, from the cart's motion between
    their stamps (``base_link`` at A into ``base_link`` at B, what
    :meth:`pepin.frame_pose.FramePoser.motion` returns) and where the lens sat on the cart at
    each — the whole ``base_link <- camera_optical`` of both frames, so a head that is panned,
    or that pans between the two pictures, turns the baseline with it.

    The pan is not a detail: fed a pitch-only pose instead (:class:`pepin.depth.CameraPose`
    carries no pan), exact correspondences under a constant 10 degree pan keep a median 0.89 of
    their true depth and the spread of a hundred survivors runs 0.42-1.85, while a pan of 5
    degrees *between* the frames leaves 0 of 209 alive at the epipolar gate
    (scratch/review_parallax_pan.py, 2026-09-12)."""
    ra = np.asarray(cam_a.rotation, dtype=float)
    rb = np.asarray(cam_b.rotation, dtype=float)
    rba = np.asarray(base_rotation, dtype=float)
    tba = np.asarray(base_translation, dtype=float)
    ta = np.asarray(cam_a.translation, dtype=float)
    tb = np.asarray(cam_b.translation, dtype=float)
    return Motion(rb.T @ rba @ ra, rb.T @ (rba @ ta + tba - tb))


def match(
    gray_a: npt.NDArray[np.uint8],
    gray_b: npt.NDArray[np.uint8],
    *,
    matcher: str = "klt",
    max_corners: int = MAX_CORNERS,
    quality: float = CORNER_QUALITY,
    min_distance: int = CORNER_MIN_DISTANCE,
    fb_tol_px: float = FB_TOL_PX,
) -> tuple[Pixels, Pixels]:
    """Corresponding points of two grey images. With ``matcher`` ``klt`` (the default) corners
    of A are tracked into B by pyramidal Lucas-Kanade and back again, keeping only those that
    return to within ``fb_tol_px`` of where they started; with ``orb`` both frames are described
    and the descriptors matched (:func:`_describe`), which costs more and survives a far longer
    gap. Returns (points in A, points in B), both (n, 2) sub-pixel columns and rows, possibly
    empty."""
    pts_a, pts_b, _found = _correspond(
        gray_a,
        gray_b,
        matcher,
        max_corners=max_corners,
        quality=quality,
        min_distance=min_distance,
        fb_tol_px=fb_tol_px,
    )
    return pts_a, pts_b


def _correspond(
    gray_a: npt.NDArray[np.uint8],
    gray_b: npt.NDArray[np.uint8],
    matcher: str,
    *,
    max_corners: int = MAX_CORNERS,
    quality: float = CORNER_QUALITY,
    min_distance: int = CORNER_MIN_DISTANCE,
    fb_tol_px: float = FB_TOL_PX,
) -> tuple[Pixels, Pixels, int]:
    """The named matcher's correspondences with the number of candidates it started from, so a
    caller can say what share it lost: ``klt`` is :func:`_track`, ``orb`` is :func:`_describe`.
    Raises ``ValueError`` on any other name — a misspelt flag is not silently the default."""
    if matcher == "orb":
        return _describe(gray_a, gray_b)
    if matcher != "klt":
        raise ValueError(f"unknown matcher {matcher!r}: one of {', '.join(MATCHERS)}")
    return _track(
        gray_a,
        gray_b,
        max_corners=max_corners,
        quality=quality,
        min_distance=min_distance,
        fb_tol_px=fb_tol_px,
    )


def _describe(
    gray_a: npt.NDArray[np.uint8],
    gray_b: npt.NDArray[np.uint8],
    *,
    features: int = ORB_FEATURES,
    fast_threshold: int = ORB_FAST_THRESHOLD,
    ratio: float = ORB_RATIO,
) -> tuple[Pixels, Pixels, int]:
    """Correspondences by recognition rather than by tracking: ORB keypoints and their binary
    descriptors in both frames, matched on Hamming distance under Lowe's ratio test (a match
    whose runner-up is within ``ratio`` of it is ambiguous and dropped) and a cross-check (B's
    own best match for the point must be that point again). Returns (points in A, points in B,
    keypoints found in A), the last being what the losses are a share of — the describer's
    equivalent of the flow's corner count."""
    import cv2

    empty: Pixels = np.zeros((0, 2), dtype=np.float32)
    # cv2's stubs type neither the detector's tuple return nor a DMatch's fields
    detector: Any = cv2.ORB.create(nfeatures=features, fastThreshold=fast_threshold)
    kp_a, des_a = detector.detectAndCompute(np.ascontiguousarray(gray_a), None)
    kp_b, des_b = detector.detectAndCompute(np.ascontiguousarray(gray_b), None)
    found = len(kp_a)
    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return empty, empty, found
    brute: Any = cv2.BFMatcher(cv2.NORM_HAMMING)
    mirror = {
        pair[0].queryIdx: pair[0].trainIdx for pair in brute.knnMatch(des_b, des_a, k=2) if pair
    }
    rows_a: list[int] = []
    rows_b: list[int] = []
    for pair in brute.knnMatch(des_a, des_b, k=2):
        if len(pair) < 2 or pair[0].distance > ratio * pair[1].distance:
            continue
        if mirror.get(pair[0].trainIdx) != pair[0].queryIdx:
            continue
        rows_a.append(pair[0].queryIdx)
        rows_b.append(pair[0].trainIdx)
    if not rows_a:
        return empty, empty, found
    pts_a: Pixels = np.array([kp_a[i].pt for i in rows_a], dtype=np.float32)
    pts_b: Pixels = np.array([kp_b[i].pt for i in rows_b], dtype=np.float32)
    return pts_a, pts_b, found


def _track(
    gray_a: npt.NDArray[np.uint8],
    gray_b: npt.NDArray[np.uint8],
    *,
    max_corners: int = MAX_CORNERS,
    quality: float = CORNER_QUALITY,
    min_distance: int = CORNER_MIN_DISTANCE,
    fb_tol_px: float = FB_TOL_PX,
) -> tuple[Pixels, Pixels, int]:
    """:func:`match` with the number of corners it started from, so a caller can say how many
    the flow lost."""
    import cv2

    empty = (np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), 0)
    a = np.ascontiguousarray(gray_a)
    b = np.ascontiguousarray(gray_b)
    corners = cv2.goodFeaturesToTrack(
        a, maxCorners=max_corners, qualityLevel=quality, minDistance=min_distance
    )
    if corners is None or len(corners) == 0:
        return empty
    found = len(corners)
    window = (LK_WINDOW, LK_WINDOW)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    # cv2's stubs admit neither a uint8 image nor the None that asks it to allocate the output
    flow: Any = cv2.calcOpticalFlowPyrLK
    forward, ok_f, _ = flow(
        a, b, corners, None, winSize=window, maxLevel=LK_LEVELS, criteria=criteria
    )
    back, ok_b, _ = flow(b, a, forward, None, winSize=window, maxLevel=LK_LEVELS, criteria=criteria)
    good = (ok_f.ravel() == 1) & (ok_b.ravel() == 1)
    drift = np.linalg.norm((back - corners).reshape(-1, 2), axis=1)
    good &= drift <= fb_tol_px
    if not bool(good.any()):
        return empty[0], empty[1], found
    pts_a: Pixels = corners.reshape(-1, 2)[good].astype(np.float32)
    pts_b: Pixels = forward.reshape(-1, 2)[good].astype(np.float32)
    return pts_a, pts_b, found


def _rays(points: Pixels, intr: Intrinsics) -> Array:
    """The direction of each pixel's ray in the optical frame, scaled so z is 1 (so the ray's
    parameter is the depth along the optical axis, the depth the network and the laws speak)."""
    p = np.asarray(points, dtype=float)
    return np.stack(
        [(p[:, 0] - intr.cx) / intr.fx, (p[:, 1] - intr.cy) / intr.fy, np.ones(p.shape[0])], axis=1
    )


def fundamental(intr: Intrinsics, motion: Motion) -> Array:
    """The fundamental matrix of a known motion: ``x_b^T F x_a`` is zero for a rigid point."""
    k = np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]])
    t = np.asarray(motion.translation, dtype=float)
    skew = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
    k_inv = np.linalg.inv(k)
    f: Array = k_inv.T @ skew @ np.asarray(motion.rotation, dtype=float) @ k_inv
    return f


def sampson(pts_a: Pixels, pts_b: Pixels, intr: Intrinsics, motion: Motion) -> Array:
    """Each correspondence's distance in pixels from the epipolar geometry of ``motion`` (the
    first-order approximation to the distance to the nearest pair that obeys it exactly): the
    consensus test a RANSAC would end at, with nothing left to fit because the motion is known."""
    f = fundamental(intr, motion)
    a = np.concatenate([np.asarray(pts_a, dtype=float), np.ones((len(pts_a), 1))], axis=1)
    b = np.concatenate([np.asarray(pts_b, dtype=float), np.ones((len(pts_b), 1))], axis=1)
    fa = a @ f.T
    ftb = b @ f
    numerator = np.einsum("ij,ij->i", b, fa) ** 2
    denominator = fa[:, 0] ** 2 + fa[:, 1] ** 2 + ftb[:, 0] ** 2 + ftb[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        out: Array = np.sqrt(np.where(denominator > 0, numerator / denominator, np.inf))
    return out


def perpendicular_baseline(points: Pixels, intr: Intrinsics, motion: Motion) -> Array:
    """How much of the camera's movement each pixel's ray actually sees across it (metres): the
    part of the translation perpendicular to that ray. This, and not the distance travelled, is
    the baseline a depth rests on — a pixel on the epipole (dead ahead when driving straight)
    has none of it, however far the cart went."""
    unit = _rays(points, intr)
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    t = np.asarray(motion.translation, dtype=float)
    out: Array = np.linalg.norm(t[None, :] - (unit @ t)[:, None] * unit, axis=1)
    return out


def triangulate(
    pts_a: Pixels,
    pts_b: Pixels,
    intr: Intrinsics,
    motion_ab: Motion,
    *,
    disparity_sigma_px: float = DISPARITY_SIGMA_PX,
) -> tuple[Array, Array]:
    """Where each correspondence sits in front of view B: the depth along B's optical axis in
    metres (the midpoint of the two rays' closest approach) and its one-sigma uncertainty.

    The two rays are ``z_a * d_a`` in A and ``z_b * d_b`` in B, and ``motion_ab`` puts A's ray
    into B's frame, so ``z_b * d_b - z_a * (R d_a) = t`` is three equations in two unknowns —
    solved by least squares, one 2x2 system per point. A depth that comes out behind either
    lens is returned as NaN (with an infinite sigma), never as a negative number."""
    z, sigma, _residual, _baseline = _triangulate_full(
        pts_a, pts_b, intr, motion_ab, disparity_sigma_px=disparity_sigma_px
    )
    return z, sigma


def _triangulate_full(
    pts_a: Pixels,
    pts_b: Pixels,
    intr: Intrinsics,
    motion: Motion,
    *,
    disparity_sigma_px: float = DISPARITY_SIGMA_PX,
) -> tuple[Array, Array, Array, Array]:
    """:func:`triangulate` with the two numbers the gates need as well: each point's
    reprojection residual in B (pixels) and the part of the baseline perpendicular to its ray
    (metres), which is the parallax the depth actually rests on. ``disparity_sigma_px`` is what
    the matcher that produced these points knows a pixel to — the flow's half a pixel, a
    keypoint's whole one."""
    d_a = _rays(pts_a, intr)
    d_b = _rays(pts_b, intr)
    t = np.asarray(motion.translation, dtype=float)
    r = d_a @ np.asarray(motion.rotation, dtype=float).T
    p = np.einsum("ij,ij->i", d_b, d_b)
    q = np.einsum("ij,ij->i", d_b, r)
    s = np.einsum("ij,ij->i", r, r)
    m = d_b @ t
    n = r @ t
    det = p * s - q * q
    with np.errstate(divide="ignore", invalid="ignore"):
        z_b = (s * m - q * n) / det
        z_a = (q * m - p * n) / det
    near = z_b[:, None] * d_b
    far = z_a[:, None] * r + t
    middle = 0.5 * (near + far)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr.fx * middle[:, 0] / middle[:, 2] + intr.cx
        v = intr.fy * middle[:, 1] / middle[:, 2] + intr.cy
    residual = np.hypot(u - np.asarray(pts_b, dtype=float)[:, 0], v - pts_b[:, 1])
    baseline = perpendicular_baseline(pts_b, intr, motion)
    ok = np.isfinite(z_b) & np.isfinite(z_a) & (z_b > NEAR_M) & (z_a > NEAR_M) & (det > 0)
    z = np.where(ok, middle[:, 2], np.nan)
    focal = 0.5 * (intr.fx + intr.fy)
    sigma_px = np.hypot(disparity_sigma_px, np.where(np.isfinite(residual), residual, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(ok & (baseline > 0), z**2 * sigma_px / (focal * baseline), np.inf)
    return z, sigma, residual, baseline


@dataclass(frozen=True)
class ParallaxTruth:
    """What one pair of frames says about the depth in the second of them: the pixels kept
    (columns and rows in B), their triangulated depth and its sigma in metres, each point's
    reprojection residual in pixels, the perpendicular baseline its depth rests on, the weight
    it deserves against a lidar beam's 1 — and the bookkeeping: how many correspondences were
    tracked, how many survived, how many fell to each gate, and the one word for a pair of
    frames that could yield nothing at all (``still`` or ``rotation-only``)."""

    points: Pixels
    z: Array
    sigma: Array
    residual: Array
    baseline: Array
    weight: Array
    tracked: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    verdict: str = ""

    @classmethod
    def nothing(
        cls, verdict: str, *, tracked: int = 0, rejected: dict[str, int] | None = None
    ) -> ParallaxTruth:
        """An empty result with a reason, for a pair of frames that cannot be triangulated."""
        empty = np.zeros(0)
        return cls(
            np.zeros((0, 2), dtype=np.float32),
            empty,
            empty,
            empty,
            empty,
            empty,
            tracked=tracked,
            rejected=dict(rejected or {}),
            verdict=verdict,
        )

    @property
    def kept(self) -> int:
        """How many points came through with a depth."""
        return int(self.z.size)

    def summary(self) -> str:
        """The line for a report: the pairs, the parallax they rest on, their noise, and what
        was thrown away."""
        if self.kept == 0:
            dropped = ", ".join(f"{k} {v}" for k, v in self.rejected.items() if v)
            return f"0 pairs ({self.verdict or 'none'}{'; ' + dropped if dropped else ''})"
        dropped = ", ".join(f"{k} {v}" for k, v in self.rejected.items() if v)
        return (
            f"{self.kept}/{self.tracked} pairs, baseline {np.median(self.baseline) * 100:.1f} cm,"
            f" sigma {np.median(self.sigma) * 100:.1f} cm"
            + (f", rejected: {dropped}" if dropped else "")
        )


def parallax_truth(
    gray_a: npt.NDArray[np.uint8],
    gray_b: npt.NDArray[np.uint8],
    intr: Intrinsics,
    motion: Motion,
    *,
    matcher: str = "klt",
    min_baseline_m: float = MIN_BASELINE_M,
    min_parallax_ratio: float = MIN_PARALLAX_RATIO,
    epipole_min_deg: float = EPIPOLE_MIN_DEG,
    max_sampson_px: float = MAX_SAMPSON_PX,
    max_reproj_px: float = MAX_REPROJ_PX,
    max_weight: float = MAX_WEIGHT,
) -> ParallaxTruth:
    """The whole measurement for one pair of frames: match, gate, triangulate, weigh.

    ``matcher`` picks who finds the correspondences: ``klt`` tracks corners with optical flow
    (cheap, and the one that wins at short gaps), ``orb`` describes and matches keypoints
    (dearer, and the only one left past about half a second of gap). The sigma of every pair
    follows the choice: a tracked corner is placed to half a pixel, a keypoint to one.

    The gates, in the order a point meets them: the motion itself (nothing under
    ``min_baseline_m`` of travel can triangulate — ``still`` when the camera did not turn
    either, ``rotation-only`` when it only turned); the flow's forward-backward check; landing
    inside the second image at all (the flow follows a corner off the edge); the
    epipolar distance to the known motion (``max_sampson_px``); a depth in front of both
    lenses; enough parallax for the depth to mean something (``min_parallax_ratio`` of the
    depth, and ``epipole_min_deg`` away from the direction of travel); and the triangulated
    point's own reprojection error (``max_reproj_px``)."""
    if motion.baseline < min_baseline_m:
        return ParallaxTruth.nothing(
            "rotation-only" if motion.angle > MIN_ROTATION_RAD else "still"
        )
    pts_a, pts_b, found = _correspond(gray_a, gray_b, matcher)
    tracked = len(pts_a)
    rejected = dict.fromkeys(REASONS, 0)
    rejected["flow"] = found - tracked  # 'flow' is whatever the matcher lost, tracker or not
    if tracked == 0:
        return ParallaxTruth.nothing("flow", rejected=rejected)
    # The flow happily follows a corner off the edge of B; a kept point's rounded pixel is a
    # pixel of the image, so a caller may index the depth with it without checking again.
    column = np.rint(np.asarray(pts_b, dtype=float)[:, 0])
    row = np.rint(np.asarray(pts_b, dtype=float)[:, 1])
    keep = (column >= 0) & (column < intr.width) & (row >= 0) & (row < intr.height)
    rejected["outside"] = int((~keep).sum())
    keep &= sampson(pts_a, pts_b, intr, motion) <= max_sampson_px
    rejected["epipolar"] = int((~keep).sum()) - rejected["outside"]
    sigma_px = ORB_DISPARITY_SIGMA_PX if matcher == "orb" else DISPARITY_SIGMA_PX
    z, sigma, residual, baseline = _triangulate_full(
        pts_a, pts_b, intr, motion, disparity_sigma_px=sigma_px
    )
    behind = keep & ~np.isfinite(z)
    rejected["behind"] = int(behind.sum())
    keep &= np.isfinite(z)
    with np.errstate(invalid="ignore"):
        thin = keep & (baseline < min_parallax_ratio * z)
        rejected["parallax"] = int(thin.sum())
        keep &= ~thin
        at_epipole = keep & (baseline < motion.baseline * math.sin(math.radians(epipole_min_deg)))
        rejected["epipole"] = int(at_epipole.sum())
        keep &= ~at_epipole
        bad = keep & ~(residual <= max_reproj_px)
        rejected["reproj"] = int(bad.sum())
        keep &= ~bad
    if not bool(keep.any()):
        worst = max(rejected, key=lambda name: rejected[name])
        return ParallaxTruth.nothing(worst, tracked=tracked, rejected=rejected)
    sigma_inv = sigma[keep] / z[keep] ** 2
    weight = np.minimum(max_weight, (LIDAR_SIGMA_INV / sigma_inv) ** 2)
    return ParallaxTruth(
        np.asarray(pts_b, dtype=np.float32)[keep],
        z[keep],
        sigma[keep],
        residual[keep],
        baseline[keep],
        weight,
        tracked=tracked,
        rejected=rejected,
        verdict="",
    )


def to_gray(rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """An RGB image as the single-channel grey the tracker reads (the luma weights, no cv2)."""
    px = np.asarray(rgb)
    if px.ndim == 2:
        out: npt.NDArray[np.uint8] = px.astype(np.uint8)
        return out
    grey = 0.299 * px[:, :, 0] + 0.587 * px[:, :, 1] + 0.114 * px[:, :, 2]
    return np.ascontiguousarray(np.rint(grey).astype(np.uint8))


__all__ = [
    "DISPARITY_SIGMA_PX",
    "EPIPOLE_MIN_DEG",
    "LIDAR_SIGMA_INV",
    "MATCHERS",
    "MAX_REPROJ_PX",
    "MAX_SAMPSON_PX",
    "MIN_BASELINE_M",
    "MIN_PARALLAX_RATIO",
    "MIN_ROTATION_RAD",
    "ORB_DISPARITY_SIGMA_PX",
    "ORB_FEATURES",
    "ORB_RATIO",
    "CameraPlacement",
    "Matcher",
    "Motion",
    "ParallaxTruth",
    "Pixels",
    "Placement",
    "camera_motion",
    "fundamental",
    "match",
    "optical_from_base",
    "parallax_truth",
    "perpendicular_baseline",
    "sampson",
    "to_gray",
    "triangulate",
]
