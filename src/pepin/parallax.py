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

A corner does not have to be a pair. What mono VO and structure-from-motion do — and what
Consistent Video Depth does before it fits anything — is follow one corner through MANY frames
and meet all of its rays at once: :func:`build_tracks` follows the current frame's corners back
through a window of earlier frames hop by hop (the flow) or recognises them in each (the
describer), :func:`triangulate_tracks` meets every ray of a track in one linear least-squares
solve with one robust pass over the worst observation, and :func:`track_truth` gates and weighs
the result into the same :class:`ParallaxTruth` a pair produces. The win is in the sigma, which
is the pair's own formula with the bundle's parallax in it: ``sigma_z = z^2 * sigma_px /
(f * B_effective)`` with ``B_effective = sqrt(sum_v b_v^2)`` over the views, so four views 5 cm
out are worth one pair at 10 cm, and a two-view track is arithmetically today's pair.

This cart has one, and it is not its wheels (scratch/parallax_pose_sweep.txt, the same four
errands with the motion taken from the lidar tracker's map pose instead of the EKF's odometry,
2026-09-14). Over a 1.0 s gap the odometry claims 15.8 cm of travel where the tracker reads
13.9, and over 1.5 s it claims 25.5 against 18.6; at 1-2 m the flow then reads 1.263 and 1.342
of the lidar on the wheels' baseline against 1.138 and 0.944 on the tracker's, and the describer
1.347 and 1.506 against 0.951 and 0.968. The over-reading past a second was the baseline itself,
and :class:`pepin.depth_pipeline.ParallaxAnchor` triangulates on the tracker's motion by default
(``parallax_motion``). What no pose has cured is the per-pair noise: 7-10 cm for the flow and
15-30 for the describer at every gap measured, which is why a pair is still a measurement to be
weighed and not a truth to be trusted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from pepin.depth import NEAR_M, REF_SIGMA_INV, Array, CameraPose, Intrinsics, pair_weight

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
LIDAR_SIGMA_INV = REF_SIGMA_INV  # 1/m: the noise a weight of 1 stands for (pepin.depth)
MAX_WEIGHT = 1.0  # no parallax pair outweighs a lidar beam at the reference range

# ---- a corner as a track ------------------------------------------------------------------
TRACK_MIN_OBS = 3  # frames a track must be seen in to be a measurement; 2 is today's pair
TRACK_WINDOW_S = 1.5  # how far back a track may reach, seconds: the ring's own span
TRACK_MIN_TOTAL_BASELINE_M = 0.10  # the effective parallax a track's views must add up to
TRACK_MAX_VIEWS = 16  # views one track may rest on: the cap on the matcher's cost per frame
TRACK_OUTLIER_PX = 2.0  # an observation missing the solved point by more is the one dropped
TRACK_MIN_CONDITION = 1e-6  # smallest / largest singular value of a track's 3x3 normal matrix

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
TRACK_REASONS = ("short track", "total baseline", "outlier obs")


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


@dataclass(frozen=True)
class Features:
    """What the describer found in one frame: its keypoints' pixels (n, 2) and their binary
    descriptors (n, 32 uint8, ``None`` for a frame with no keypoints at all). A caller that
    keeps these beside a frame it may match against again — a ring of frames a track reaches
    back through — describes that frame once instead of once per later frame."""

    points: Pixels
    descriptors: npt.NDArray[np.uint8] | None

    @property
    def count(self) -> int:
        """How many keypoints the frame carries."""
        return int(self.points.shape[0])


def describe_frame(
    gray: npt.NDArray[np.uint8],
    *,
    features: int = ORB_FEATURES,
    fast_threshold: int = ORB_FAST_THRESHOLD,
) -> Features:
    """One frame's ORB keypoints and descriptors — the half of the describer that does not
    depend on what the frame is matched against, so it may be computed once and kept."""
    import cv2

    # cv2's stubs type neither the detector's tuple return nor a keypoint's fields
    detector: Any = cv2.ORB.create(nfeatures=features, fastThreshold=fast_threshold)
    keypoints, descriptors = detector.detectAndCompute(np.ascontiguousarray(gray), None)
    if not keypoints:
        return Features(np.zeros((0, 2), dtype=np.float32), None)
    return Features(np.array([k.pt for k in keypoints], dtype=np.float32), descriptors)


def _match_features(
    a: Features, b: Features, ratio: float = ORB_RATIO
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Which keypoint of ``a`` is which keypoint of ``b``: Hamming distance under Lowe's ratio
    test (a match whose runner-up is within ``ratio`` of it is ambiguous and dropped) and a
    cross-check (b's own best match for the point must be that point again). Returns the two
    index arrays, possibly empty."""
    import cv2

    empty = np.zeros(0, dtype=np.int64)
    des_a, des_b = a.descriptors, b.descriptors
    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return empty, empty
    brute: Any = cv2.BFMatcher(cv2.NORM_HAMMING)  # cv2's stubs type no DMatch field
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
    return np.array(rows_a, dtype=np.int64), np.array(rows_b, dtype=np.int64)


def _describe(
    gray_a: npt.NDArray[np.uint8],
    gray_b: npt.NDArray[np.uint8],
    *,
    features: int = ORB_FEATURES,
    fast_threshold: int = ORB_FAST_THRESHOLD,
    ratio: float = ORB_RATIO,
) -> tuple[Pixels, Pixels, int]:
    """Correspondences by recognition rather than by tracking: both frames described
    (:func:`describe_frame`) and their descriptors matched (:func:`_match_features`). Returns
    (points in A, points in B, keypoints found in A), the last being what the losses are a
    share of — the describer's equivalent of the flow's corner count."""
    empty: Pixels = np.zeros((0, 2), dtype=np.float32)
    feat_a = describe_frame(gray_a, features=features, fast_threshold=fast_threshold)
    feat_b = describe_frame(gray_b, features=features, fast_threshold=fast_threshold)
    rows_a, rows_b = _match_features(feat_a, feat_b, ratio)
    if rows_a.size == 0:
        return empty, empty, feat_a.count
    return feat_a.points[rows_a], feat_b.points[rows_b], feat_a.count


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
    observations: Array | None = None  # a track's views (:func:`track_truth`); None for a pair
    sigma_two: Array | None = None  # what the widest single pair of a track alone would claim

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
    weight = pair_weight(sigma_inv, cap=max_weight)
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


# ---- a corner as a track, not a pair ---------------------------------------------------------
@dataclass(frozen=True)
class Tracks:
    """Corners of the current frame followed through the frames before it.

    ``pixels`` is (tracks, views, 2): where each track landed in each view, NaN where it was not
    seen there, and ``seen`` is the mask that says which. ``motions`` carries one transform per
    view, taking a point of THAT view's optical frame into the current camera's
    (``X_current = R X_view + t``); the last view is the current frame itself, so its motion is
    the identity, ``pixels[:, -1]`` is the pixel every depth is reported at and
    ``motions[v].translation`` is where view ``v``'s lens sat in the current camera's frame.
    ``found`` is how many corners the matcher started from, so a caller can say what it lost."""

    pixels: Array
    seen: npt.NDArray[np.bool_]
    motions: tuple[Motion, ...]
    found: int = 0

    @property
    def count(self) -> int:
        """How many tracks."""
        return int(self.pixels.shape[0])

    @property
    def views(self) -> int:
        """How many frames the tracks were looked for in, the current one included."""
        return len(self.motions)

    @property
    def observations(self) -> npt.NDArray[np.int64]:
        """How many views each track was actually seen in (2 is today's pair)."""
        out: npt.NDArray[np.int64] = self.seen.sum(axis=1)
        return out

    @property
    def current(self) -> Pixels:
        """Where each track sits in the current frame — the pixel its depth belongs to."""
        return np.asarray(self.pixels[:, -1], dtype=np.float32)


def _identity() -> Motion:
    """The motion of the current frame into itself."""
    return Motion(np.eye(3), np.zeros(3))


def _view_span(views: int, max_views: int) -> list[int]:
    """Which of ``views`` frames a track is allowed to rest on when the window holds more than
    ``max_views``: evenly spaced over the window, always keeping the oldest frame (the widest
    baseline) and the current one (the frame the depth is reported in)."""
    if views <= max_views:
        return list(range(views))
    picked = np.unique(np.rint(np.linspace(0, views - 1, max_views)).astype(int))
    return [int(i) for i in picked]


def build_tracks(
    grays: Sequence[npt.NDArray[np.uint8]],
    motions: Sequence[Motion],
    *,
    matcher: str = "klt",
    features: list[Features | None] | None = None,
    max_corners: int = MAX_CORNERS,
    quality: float = CORNER_QUALITY,
    min_distance: int = CORNER_MIN_DISTANCE,
    fb_tol_px: float = FB_TOL_PX,
    ratio: float = ORB_RATIO,
) -> Tracks:
    """The corners of the last frame of ``grays`` as tracks through all of them.

    ``grays`` is oldest first and its LAST entry is the current frame; ``motions`` is one per
    EARLIER view, taking that view's optical frame into the current camera's, oldest first.

    ``klt`` (the default) detects corners in the current frame and follows them BACK hop by hop
    — current into the frame before it, that into the one before, and so on — with a
    forward-backward check at every hop, which is why a track survives a window the flow cannot
    jump across in one go: five 0.1 s hops each keep about 92 % of their corners where one
    0.5 s jump keeps 68 %. A track stops where the flow loses it, so its observations are
    contiguous back from the current frame. ``orb`` describes every frame and matches the
    current frame's descriptors against each earlier frame's independently, so a track may skip
    a frame it was occluded in; pass ``features`` — a list as long as ``grays``, entries
    ``None`` until computed — to keep each frame's description between calls, as a ring does.

    Returns a :class:`Tracks` whose last view is the current frame."""
    if len(motions) != len(grays) - 1:
        raise ValueError(
            f"{len(grays)} frames need {len(grays) - 1} motions into the current camera,"
            f" got {len(motions)}"
        )
    if len(grays) < 2:
        raise ValueError("a track needs at least two frames")
    all_motions = (*motions, _identity())
    if matcher == "orb":
        pixels, seen, found = _describe_tracks(grays, features, ratio)
    elif matcher == "klt":
        pixels, seen, found = _flow_tracks(
            grays,
            max_corners=max_corners,
            quality=quality,
            min_distance=min_distance,
            fb_tol_px=fb_tol_px,
        )
    else:
        raise ValueError(f"unknown matcher {matcher!r}: one of {', '.join(MATCHERS)}")
    return Tracks(pixels, seen, all_motions, found)


def _empty_tracks(views: int) -> tuple[Array, npt.NDArray[np.bool_], int]:
    """No corner found at all, in the shape the builders return."""
    return np.zeros((0, views, 2)), np.zeros((0, views), dtype=bool), 0


def _flow_tracks(
    grays: Sequence[npt.NDArray[np.uint8]],
    *,
    max_corners: int,
    quality: float,
    min_distance: int,
    fb_tol_px: float,
) -> tuple[Array, npt.NDArray[np.bool_], int]:
    """The current frame's corners followed back through the earlier frames, one hop at a time,
    each hop forward-backward checked and required to land inside the earlier picture."""
    import cv2

    views = len(grays)
    current = np.ascontiguousarray(grays[-1])
    height, width = current.shape[:2]
    corners = cv2.goodFeaturesToTrack(
        current, maxCorners=max_corners, qualityLevel=quality, minDistance=min_distance
    )
    if corners is None or len(corners) == 0:
        return _empty_tracks(views)
    found = len(corners)
    pixels = np.full((found, views, 2), np.nan)
    seen = np.zeros((found, views), dtype=bool)
    pixels[:, -1] = corners.reshape(-1, 2)
    seen[:, -1] = True
    alive = np.ones(found, dtype=bool)
    here = corners.reshape(-1, 1, 2).astype(np.float32)
    window = (LK_WINDOW, LK_WINDOW)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    flow: Any = cv2.calcOpticalFlowPyrLK  # cv2's stubs admit neither uint8 nor the None output
    for view in range(views - 2, -1, -1):
        rows = np.flatnonzero(alive)
        if rows.size == 0:
            break
        earlier = np.ascontiguousarray(grays[view])
        later = np.ascontiguousarray(grays[view + 1])
        start = here[rows]
        back, ok_back, _ = flow(
            later, earlier, start, None, winSize=window, maxLevel=LK_LEVELS, criteria=criteria
        )
        forth, ok_forth, _ = flow(
            earlier, later, back, None, winSize=window, maxLevel=LK_LEVELS, criteria=criteria
        )
        landed = back.reshape(-1, 2)
        good = (ok_back.ravel() == 1) & (ok_forth.ravel() == 1)
        good &= np.linalg.norm((forth - start).reshape(-1, 2), axis=1) <= fb_tol_px
        good &= (landed[:, 0] >= 0) & (landed[:, 0] < width)
        good &= (landed[:, 1] >= 0) & (landed[:, 1] < height)
        kept = rows[good]
        pixels[kept, view] = landed[good]
        seen[kept, view] = True
        here = np.full((found, 1, 2), np.nan, dtype=np.float32)
        here[kept] = landed[good].reshape(-1, 1, 2).astype(np.float32)
        alive = np.zeros(found, dtype=bool)
        alive[kept] = True
    return pixels, seen, found


def _describe_tracks(
    grays: Sequence[npt.NDArray[np.uint8]],
    features: list[Features | None] | None,
    ratio: float,
) -> tuple[Array, npt.NDArray[np.bool_], int]:
    """The current frame's keypoints recognised in every earlier frame, each frame matched
    against the current one on its own — a track may therefore skip a frame it was hidden in.
    ``features`` is filled in where it was ``None``, so a caller's ring keeps each frame's
    description."""
    views = len(grays)
    cache: list[Features | None] = features if features is not None else [None] * views
    if len(cache) != views:
        raise ValueError(f"{views} frames need {views} feature slots, got {len(cache)}")
    for i, gray in enumerate(grays):
        if cache[i] is None:
            cache[i] = describe_frame(gray)
    described = [f for f in cache if f is not None]
    current = described[-1]
    found = current.count
    if found == 0:
        return _empty_tracks(views)
    pixels = np.full((found, views, 2), np.nan)
    seen = np.zeros((found, views), dtype=bool)
    pixels[:, -1] = current.points
    seen[:, -1] = True
    for view in range(views - 1):
        mine, theirs = _match_features(current, described[view], ratio)
        if mine.size == 0:
            continue
        pixels[mine, view] = described[view].points[theirs]
        seen[mine, view] = True
    return pixels, seen, found


@dataclass(frozen=True)
class TrackDepths:
    """Where every track's point sits in front of the CURRENT camera, and how well it is known.

    ``z`` is the depth along the current optical axis in metres (NaN for a track whose rays do
    not meet in front of every lens that saw it), ``sigma`` its one-sigma noise, ``residual``
    the root-mean-square reprojection error over the observations kept (pixels), ``baseline``
    the effective parallax those observations add up to, ``travel`` how far the camera moved
    over them, ``observations`` how many were kept, ``sigma_two`` what the single widest pair
    alone would have claimed — the "before" a report holds against the "after" — and
    ``repaired`` how many tracks lost their worst observation to the robust pass.

    The sigma is the two-view formula with the whole bundle's parallax in it::

        sigma_z = z^2 * sigma_px / (f * B_effective),  B_effective = sqrt(sum_v b_v^2)

    where ``b_v`` is the part of view ``v``'s baseline perpendicular to the point's ray (the
    only part a depth rests on) and ``sigma_px`` is the matcher's own noise widened by the
    solve's residual. Each view carries independent pixel noise, so their inverse-depth
    informations add and the baselines add in quadrature: two views 5 cm out are worth one at
    7.1 cm, four at 5 cm one at 10. With one earlier view ``B_effective`` is that view's own
    ``b`` and the number is exactly what :func:`triangulate` returns for the same pair."""

    z: Array
    sigma: Array
    residual: Array
    baseline: Array
    travel: Array
    observations: npt.NDArray[np.int64]
    sigma_two: Array
    repaired: int = 0


def _view_rays(pixels: Array, intr: Intrinsics, motions: Sequence[Motion]) -> Array:
    """Every observation's ray as a unit vector in the CURRENT camera's frame: the pixel's ray
    in its own view, turned by that view's rotation. Shape (tracks, views, 3), NaN where the
    track was not seen."""
    tracks, views = int(pixels.shape[0]), int(pixels.shape[1])
    flat = _rays(np.asarray(pixels, dtype=float).reshape(-1, 2), intr).reshape(tracks, views, 3)
    rotation = np.array([np.asarray(m.rotation, dtype=float) for m in motions])
    turned = np.einsum("vij,tvj->tvi", rotation, flat)
    with np.errstate(divide="ignore", invalid="ignore"):
        out: Array = turned / np.linalg.norm(turned, axis=2, keepdims=True)
    return out


def _origins(motions: Sequence[Motion]) -> Array:
    """Where each view's lens sat in the current camera's frame (views, 3): the current frame's
    own origin is the zero at the end."""
    return np.array([np.asarray(m.translation, dtype=float) for m in motions])


def _meet_rays(
    rays: Array, origins: Array, seen: npt.NDArray[np.bool_]
) -> tuple[Array, npt.NDArray[np.bool_]]:
    """Where each track's bundle of rays meets, in least squares: the point minimising the sum
    of its squared perpendicular distances to every ray, ``sum_v ||(I - e e^T)(X - o_v)||^2``,
    which is one symmetric 3x3 system per track and, for two rays, exactly the middle of their
    common perpendicular — today's pair, unchanged. Returns the point per track and whether its
    system was conditioned at all (rays all but parallel leave the depth along them unknown)."""
    unit = np.where(seen[:, :, None], np.nan_to_num(rays), 0.0)
    count = seen.sum(axis=1).astype(float)
    normal = count[:, None, None] * np.eye(3)[None, :, :] - np.einsum("tvi,tvj->tij", unit, unit)
    along = np.einsum("tvi,vi->tv", unit, origins)
    right = np.einsum("tv,vi->ti", seen.astype(float), origins) - np.einsum(
        "tv,tvi->ti", along, unit
    )
    spectrum = np.linalg.svd(normal, compute_uv=False)
    ok = (count >= 2) & (spectrum[:, 2] > TRACK_MIN_CONDITION * np.maximum(spectrum[:, 0], 1e-12))
    point = np.full((rays.shape[0], 3), np.nan)
    if bool(ok.any()):
        # numpy 2 reads a (t, 3) right-hand side as one matrix, not a stack of vectors
        point[ok] = np.linalg.solve(normal[ok], right[ok][:, :, None])[:, :, 0]
    return point, ok


def _reproject(
    point: Array, pixels: Array, intr: Intrinsics, motions: Sequence[Motion]
) -> tuple[Array, Array]:
    """Each observation's reprojection residual in pixels and the depth the solved point has in
    that view: the point carried back into every view (``X_view = R^T (X - t)``) and projected
    with the same optics the tracker read."""
    rotation = np.array([np.asarray(m.rotation, dtype=float) for m in motions])
    origins = _origins(motions)
    local = np.einsum("vji,tvj->tvi", rotation, point[:, None, :] - origins[None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        column = intr.fx * local[:, :, 0] / local[:, :, 2] + intr.cx
        row = intr.fy * local[:, :, 1] / local[:, :, 2] + intr.cy
        residual: Array = np.hypot(column - pixels[:, :, 0], row - pixels[:, :, 1])
    return residual, local[:, :, 2]


def triangulate_tracks(
    tracks: Tracks,
    intr: Intrinsics,
    *,
    disparity_sigma_px: float = DISPARITY_SIGMA_PX,
    outlier_px: float = TRACK_OUTLIER_PX,
) -> TrackDepths:
    """Every track's depth in the current camera, from all of its observations at once.

    The rays of all the views a track was seen in are met in one linear least-squares solve
    (:func:`_meet_rays`); then one robust pass: a track whose worst observation misses the
    solved point by more than ``outlier_px`` drops that single observation and is solved again,
    which repairs a mistracked hop without throwing the other views away (a track left with
    fewer than two observations is simply not a measurement and comes back NaN). The sigma is
    ``z^2 * sigma_px / (f * B_effective)`` with ``B_effective`` the quadrature sum of the views'
    perpendicular baselines — see :class:`TrackDepths`."""
    pixels = np.asarray(tracks.pixels, dtype=float)
    rays = _view_rays(pixels, intr, tracks.motions)
    origins = _origins(tracks.motions)
    seen = tracks.seen.copy()
    point, ok = _meet_rays(rays, origins, seen)
    residual, depth = _reproject(point, pixels, intr, tracks.motions)
    rows = np.arange(seen.shape[0])
    missed = np.where(seen, np.nan_to_num(residual, nan=np.inf), -np.inf)
    worst = np.argmax(missed, axis=1)
    repaired = ok & (seen.sum(axis=1) > 2) & (missed[rows, worst] > outlier_px)
    if bool(repaired.any()):
        seen[rows[repaired], worst[repaired]] = False
        point, ok = _meet_rays(rays, origins, seen)
        residual, depth = _reproject(point, pixels, intr, tracks.motions)
    unit = np.nan_to_num(rays[:, -1, :])  # the current frame's own ray, already a unit vector
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        along = unit @ origins.T
        perpendicular = np.linalg.norm(
            origins[None, :, :] - along[:, :, None] * unit[:, None, :], axis=2
        )
    perpendicular = np.where(seen, perpendicular, 0.0)
    baseline = np.sqrt((perpendicular**2).sum(axis=1))
    stepped = np.where(seen, np.linalg.norm(origins, axis=1)[None, :] ** 2, 0.0)
    travel = np.sqrt(stepped.sum(axis=1))
    counted = seen & np.isfinite(residual)
    rms = np.sqrt(
        (np.where(counted, residual, 0.0) ** 2).sum(axis=1) / np.maximum(counted.sum(axis=1), 1)
    )
    forward = np.all(~seen | (depth > NEAR_M), axis=1)
    z = point[:, 2]
    good = ok & forward & np.isfinite(z) & (z > NEAR_M)
    z = np.where(good, z, np.nan)
    focal = 0.5 * (intr.fx + intr.fy)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_px = np.hypot(disparity_sigma_px, np.where(np.isfinite(rms), rms, 0.0))
        sigma = np.where(good & (baseline > 0), z**2 * sigma_px / (focal * baseline), np.inf)
        widest = np.argmax(np.where(seen, perpendicular, -1.0), axis=1)
        one = perpendicular[rows, widest]
        one_px = np.hypot(disparity_sigma_px, np.nan_to_num(residual[rows, widest]))
        sigma_two = np.where(good & (one > 0), z**2 * one_px / (focal * one), np.inf)
    return TrackDepths(
        z,
        sigma,
        rms,
        baseline,
        travel,
        seen.sum(axis=1),
        sigma_two,
        repaired=int(repaired.sum()),
    )


def track_truth(
    grays: Sequence[npt.NDArray[np.uint8]],
    motions: Sequence[Motion],
    intr: Intrinsics,
    *,
    matcher: str = "klt",
    features: list[Features | None] | None = None,
    min_obs: int = TRACK_MIN_OBS,
    min_total_baseline_m: float = TRACK_MIN_TOTAL_BASELINE_M,
    max_views: int = TRACK_MAX_VIEWS,
    min_baseline_m: float = MIN_BASELINE_M,
    min_parallax_ratio: float = MIN_PARALLAX_RATIO,
    epipole_min_deg: float = EPIPOLE_MIN_DEG,
    max_sampson_px: float = MAX_SAMPSON_PX,
    max_reproj_px: float = MAX_REPROJ_PX,
    max_weight: float = MAX_WEIGHT,
    outlier_px: float = TRACK_OUTLIER_PX,
) -> ParallaxTruth:
    """:func:`parallax_truth` over a window of frames instead of a pair: track, gate,
    triangulate from every view at once, weigh. The result is the same
    :class:`ParallaxTruth` — the pixels of the CURRENT frame, their depth, their sigma, their
    weight against a lidar beam — with ``observations`` and ``sigma_two`` filled in, so an
    anchor emits the same pairs whichever ruler it used.

    ``grays`` is oldest first with the current frame last and ``motions`` is one per earlier
    view into the current camera (:func:`build_tracks`). The gates a track meets: the motion
    itself (no view moved ``min_baseline_m`` and there is nothing to triangulate — ``still``
    when the camera did not turn either, ``rotation-only`` when it only turned); the matcher's
    own check (forward-backward per hop for the flow, the ratio and cross-check for the
    describer); the epipolar distance of each OBSERVATION to the known motion of its view,
    which drops that observation and not the whole track; landing inside the current picture;
    ``min_obs`` observations left; a depth in front of every lens that saw it; the parallax
    those observations add up to (``min_total_baseline_m`` of effective baseline, and
    ``min_parallax_ratio`` of the depth, and ``epipole_min_deg`` off the direction of travel);
    and the bundle's own reprojection error (``max_reproj_px``)."""
    if not motions or max(m.baseline for m in motions) < min_baseline_m:
        turned = bool(motions) and max(m.angle for m in motions) > MIN_ROTATION_RAD
        return ParallaxTruth.nothing("rotation-only" if turned else "still")
    span = _view_span(len(grays), max_views)
    windows = [grays[i] for i in span]
    moved = [motions[i] for i in span if i < len(motions)]
    kept_features = None if features is None else [features[i] for i in span]
    tracks = build_tracks(windows, moved, matcher=matcher, features=kept_features)
    if features is not None and kept_features is not None:
        for slot, i in enumerate(span):
            features[i] = kept_features[slot]
    rejected = dict.fromkeys((*REASONS, *TRACK_REASONS), 0)
    started = tracks.observations >= 2
    tracked = int(started.sum())
    rejected["flow"] = tracks.found - tracked
    if tracked == 0:
        return ParallaxTruth.nothing("flow", rejected=rejected)
    seen = tracks.seen.copy()
    current = tracks.current
    for view in range(tracks.views - 1):
        here = seen[:, view]
        if not bool(here.any()):
            continue
        distance = sampson(
            np.asarray(tracks.pixels[:, view], dtype=np.float32),
            current,
            intr,
            tracks.motions[view],
        )
        lost = here & ~(distance <= max_sampson_px)
        rejected["epipolar"] += int(lost.sum())
        seen[:, view] &= ~lost
    tracks = replace(tracks, seen=seen)
    keep = started.copy()
    column = np.rint(np.asarray(current, dtype=float)[:, 0])
    row = np.rint(np.asarray(current, dtype=float)[:, 1])
    inside = (column >= 0) & (column < intr.width) & (row >= 0) & (row < intr.height)
    rejected["outside"] = int((keep & ~inside).sum())
    keep &= inside
    observations = tracks.observations
    short = keep & (observations < max(min_obs, 2))
    rejected["short track"] = int(short.sum())
    keep &= ~short
    sigma_px = ORB_DISPARITY_SIGMA_PX if matcher == "orb" else DISPARITY_SIGMA_PX
    found = triangulate_tracks(tracks, intr, disparity_sigma_px=sigma_px, outlier_px=outlier_px)
    rejected["outlier obs"] = found.repaired
    rejected["behind"] = int((keep & ~np.isfinite(found.z)).sum())
    keep &= np.isfinite(found.z)
    with np.errstate(invalid="ignore"):
        thin = keep & (found.baseline < min_total_baseline_m)
        rejected["total baseline"] = int(thin.sum())
        keep &= ~thin
        near = keep & (found.baseline < min_parallax_ratio * found.z)
        rejected["parallax"] = int(near.sum())
        keep &= ~near
        ahead = keep & (found.baseline < found.travel * math.sin(math.radians(epipole_min_deg)))
        rejected["epipole"] = int(ahead.sum())
        keep &= ~ahead
        bad = keep & ~(found.residual <= max_reproj_px)
        rejected["reproj"] = int(bad.sum())
        keep &= ~bad
    if not bool(keep.any()):
        worst = max(rejected, key=lambda name: rejected[name])
        return ParallaxTruth.nothing(worst, tracked=tracked, rejected=rejected)
    weight = pair_weight(found.sigma[keep] / found.z[keep] ** 2, cap=max_weight)
    return ParallaxTruth(
        current[keep],
        found.z[keep],
        found.sigma[keep],
        found.residual[keep],
        found.baseline[keep],
        weight,
        tracked=tracked,
        rejected=rejected,
        verdict="",
        observations=found.observations[keep].astype(float),
        sigma_two=found.sigma_two[keep],
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
    "TRACK_MAX_VIEWS",
    "TRACK_MIN_OBS",
    "TRACK_MIN_TOTAL_BASELINE_M",
    "TRACK_OUTLIER_PX",
    "TRACK_REASONS",
    "TRACK_WINDOW_S",
    "CameraPlacement",
    "Features",
    "Matcher",
    "Motion",
    "ParallaxTruth",
    "Pixels",
    "Placement",
    "TrackDepths",
    "Tracks",
    "build_tracks",
    "camera_motion",
    "describe_frame",
    "fundamental",
    "match",
    "optical_from_base",
    "parallax_truth",
    "perpendicular_baseline",
    "sampson",
    "to_gray",
    "track_truth",
    "triangulate",
    "triangulate_tracks",
]
