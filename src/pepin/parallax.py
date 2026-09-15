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

Following a corner BACKWARDS is the wrong way round, and the cost says so: the whole window is
re-tracked on every frame, two flow calls per view, 27 ms a frame at 8 views and 52 at 16
(scratch/parallax_tracks_audit.txt), so a window long enough to matter cannot be afforded. And
the window is what matters. Every error term of a triangulated depth divides by the baseline —
pixel noise as ``z^2 sigma_px / (f B)``, the pose's own centimetre or two as ``1 / B``, its
0.3 degrees of heading as 1.9 px against a disparity of 25 px at 2 m — while the tracker's map
pose is ABSOLUTE, so reaching further back costs the pose nothing at all. Over 1.5 s this cart's
effective baseline is about 14 cm and a parallax-only law leaves 19 % of residual against the
lidar; 28 cm should halve that. :class:`TrackStore` turns the tracking round: a corner is
detected once and followed FORWARD one hop a frame, two flow calls per FRAME whatever the
window, each hop appending an observation to the corner it already has. What a forward track
needs and a backward one never did is a bound on the drift a per-hop check cannot see — the flow
slides along an edge and along the epipolar line by a fraction of a pixel a hop, each hop
passing its own forward-backward test — so every ``verify_every`` frames each corner is
re-tracked DIRECTLY from the grey of its oldest kept view and closed when the two disagree by
more than ``drift_tol_px``.

A track can be asked to agree with itself as well: :func:`_split_gap` solves the older half of a
track's views and the newer half separately and drops the track when the two depths disagree
(``split_tol_sigma``, off by default). It is the only test left that reads a depth changing with
the window, and measured it is worth very little. It does not catch — and nothing on one track
can — a point whose own motion is parallel to the camera's, or a corner sliding along its
epipolar line by an amount proportional to the baseline: both leave every ray meeting exactly,
at the wrong depth. Even a whole half of a window sliding 2 px off the corner, 21 % of depth,
disagrees by only 2.0 sigma, which is what a clean corner reaches on pixel noise alone. On the
four errands of 2026-09-14 a 3 sigma gate removes 1.1 % of the tracks, leaves the law's residual
exactly where it was, and costs 3.9 ms a frame of the stage's 27.0
(scratch/parallax_split_probe.py, scratch/parallax_tracks_eval.txt).

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
import time
from collections.abc import Mapping, Sequence
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
TRACK_MAX_VIEWS = 8  # views one track may rest on: the cap on the matcher's cost per frame.
# 8 and not 16 since 2026-09-15: 85 % of a track's cost is the flow's hops and the hops are
# one per view, so the cap is the cost. Over the four errands of 2026-09-14
# (scratch/parallax_tracks_audit.txt) 16 views cost 52.5 ms a frame and 8 cost 27.0, and on
# the frames both could fit a law the law was no worse at 8 (14.2 % of median residual at the
# lidar's beams against 19.0 % at 16, over the same 12 frames). Raise it on a robot whose
# camera is faster than this one's 6-9 frames/s, where a view is a smaller step.
TRACK_OUTLIER_PX = 2.0  # an observation missing the solved point by more is the one dropped
TRACK_MIN_CONDITION = 1e-6  # smallest / largest singular value of a track's 3x3 normal matrix
TRACK_SIGMA_MODELS = ("covariance", "baseline")
TRACK_SIGMA_MODEL = "covariance"  # the solve's own covariance; "baseline" is the closed form
TRACK_SPLIT_TOL_SIGMA = 0.0  # how far a track's older and newer halves may disagree, in sigmas;
# 0 does not compute the split at all. OFF because it was measured and it buys nothing: at the 3
# sigma a clean synthetic corner never reaches (max 1.97 over 1200 draws,
# scratch/parallax_split_probe.py) it removes 1.1 % of the real errands' tracks
# (scratch/parallax_tracks_eval.txt) and leaves the parallax-only law's residual exactly where it
# was, 19.0 % at 16 views and 14.2 % at 8 on the frames a pair also fitted, while the two extra
# half-solves cost 3.9 ms a frame of the stage's 27.0 (scratch/parallax_tracks_audit.txt).
# Set it above 0 to arm the gate, or to a huge number to measure the distribution without gating.

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
TRACK_REASONS = ("short track", "total baseline", "outlier obs", "split")


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
    frames that could yield nothing at all (``still`` or ``rotation-only``).

    A track (:func:`track_truth`) fills in three more per point: ``observations`` how many views
    its depth rests on, ``sigma_two`` what its widest single pair alone would have claimed, and
    ``split`` how far its older and newer halves disagree in combined sigmas. A pair leaves all
    three ``None``."""

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
    split: Array | None = None  # how far a track's two halves disagree, in combined sigmas

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
    alone would have claimed — the "before" a report holds against the "after" —
    ``repaired`` how many tracks lost their worst observation to the robust pass, and ``split``
    how far each track's older and newer halves disagree about its depth, in combined sigmas
    (:func:`_split_gap`; 0 for a track too short to cut in two).

    The sigma is the solve's own covariance (:func:`_solve_sigma`), a 3x3 inverse per track:
    the midpoint's normal matrix propagated through each view's range, which is honest whatever
    the shape of the bundle. The closed form the stage shipped with is still reachable
    (``sigma_model="baseline"``)::

        sigma_z = z^2 * sigma_px / (f * B_effective),  B_effective = sqrt(sum_v b_v^2)

    where ``b_v`` is the part of view ``v``'s baseline perpendicular to the point's ray (the
    only part a depth rests on) and ``sigma_px`` is the matcher's own noise widened by the
    solve's residual. It is exact for a camera moving ACROSS the ray — a sidestep, which is the
    geometry it was measured on — and it is optimistic wherever the views also spread along the
    ray, because a view that sees the point from further away reads its pixel into a bigger
    depth error while the formula credits it with the same ``z``. Measured on synthetic tracks
    against the solve's own scatter (scratch/parallax_sigma_mc.txt, 2026-09-15): exact for a
    sidestep, 1.5-2.0x optimistic for a cart driving forward at 10-16 views — and a pair's vote
    in the frame's fit is 1 / sigma^2, so that is a vote 2 to 4 times too loud.

    ``baseline`` is still what ``B_effective`` reports and what ``min_total_baseline_m`` gates,
    since it is a length the report line can read."""

    z: Array
    sigma: Array
    residual: Array
    baseline: Array
    travel: Array
    observations: npt.NDArray[np.int64]
    sigma_two: Array
    repaired: int = 0
    split: Array = field(default_factory=lambda: np.zeros(0))


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


def _solve_sigma(
    point: Array,
    rays: Array,
    origins: Array,
    seen: npt.NDArray[np.bool_],
    disparity_sigma_px: float,
    rms: Array,
    focal: float,
) -> Array:
    """The depth noise the midpoint solve ITSELF carries, per track, in metres.

    :func:`_meet_rays` minimises ``sum_v ||P_v (X - o_v)||^2`` with ``P_v = I - e_v e_v^T``, so
    its normal matrix is ``N = sum_v P_v`` and one view's residual carries ``r_v`` metres of
    noise for every radian of ray error (``r_v`` is how far that lens sat from the point).
    Propagating gives ``C = N^-1 (sum_v r_v^2 P_v) N^-1 * sigma_angle^2`` and the depth's own
    variance is ``C[2, 2]``; ``sigma_angle`` is the per-observation pixel noise over the focal
    length. Unlike the baseline formula this knows that a view which sees the point from
    further away reads its pixel into a bigger depth error, which is exactly what a camera
    driving FORWARD does (scratch/parallax_sigma_mc.txt: the formula runs up to 2.0x optimistic
    there, a vote 4 times too loud, and this reads the solve's own scatter to 3 %).

    The bundle's own misfit widens it, but only the part of the misfit that pixel noise does not
    already explain: a bundle of ``n`` views has ``2n - 3`` degrees of freedom left after the
    point is fitted, so noise alone puts ``sigma_1 * sqrt((2n - 3) / n)`` into ``rms``, and only
    the excess over that is evidence the track is worse than a corner should be. (Adding the raw
    ``rms`` in quadrature, which the baseline model does, charges a perfectly good sixteen-view
    bundle 25 % of extra sigma for noise it has already counted.)

    ``inf`` where the bundle is too degenerate to invert."""
    unit = np.where(seen[:, :, None], np.nan_to_num(rays), 0.0)
    count = seen.sum(axis=1).astype(float)
    normal = count[:, None, None] * np.eye(3)[None, :, :] - np.einsum("tvi,tvj->tij", unit, unit)
    reach = np.where(seen, np.linalg.norm(point[:, None, :] - origins[None, :, :], axis=2), 0.0)
    squared = reach**2
    middle = squared.sum(axis=1)[:, None, None] * np.eye(3)[None, :, :] - np.einsum(
        "tv,tvi,tvj->tij", squared, unit, unit
    )
    spectrum = np.linalg.svd(normal, compute_uv=False)
    ok = (count >= 2) & (spectrum[:, 2] > TRACK_MIN_CONDITION * np.maximum(spectrum[:, 0], 1e-12))
    variance = np.full(point.shape[0], np.inf)
    if bool(ok.any()):
        inverse = np.linalg.inv(normal[ok])
        cov = inverse @ middle[ok] @ inverse
        variance[ok] = np.maximum(cov[:, 2, 2], 0.0)
    # disparity_sigma_px is a DISPARITY's noise (two views' pixels); one observation carries
    # 1 / sqrt(2) of it, which is what keeps a two-view track reading the pair's own number.
    per_observation = disparity_sigma_px / math.sqrt(2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = per_observation * np.sqrt(np.maximum(2.0 * count - 3.0, 1.0) / count)
        excess = np.where(np.isfinite(rms) & (expected > 0), np.maximum(rms / expected, 1.0), 1.0)
    out: Array = excess * per_observation * np.sqrt(variance) / focal
    return out


def _perpendicular(rays: Array, origins: Array, seen: npt.NDArray[np.bool_]) -> Array:
    """How much of each view's baseline the point's ray actually sees ACROSS it, per track and
    view (metres, 0 where the track was not seen): the part of that view's offset from the
    current camera perpendicular to the current frame's own ray, which is the parallax a depth
    rests on (:func:`perpendicular_baseline`, one ray against many origins)."""
    unit = np.nan_to_num(rays[:, -1, :])  # the current frame's own ray, already a unit vector
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        along = unit @ origins.T
        out = np.linalg.norm(origins[None, :, :] - along[:, :, None] * unit[:, None, :], axis=2)
    return np.where(seen, out, 0.0)


def _rms_residual(residual: Array, seen: npt.NDArray[np.bool_]) -> Array:
    """The root-mean-square reprojection error over the observations a track actually kept."""
    counted = seen & np.isfinite(residual)
    total = (np.where(counted, residual, 0.0) ** 2).sum(axis=1)
    out: Array = np.sqrt(total / np.maximum(counted.sum(axis=1), 1))
    return out


def _halves(seen: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """One track's observations cut in two by age, each half keeping the CURRENT view as its
    anchor (a half with no anchor would be a depth in a different frame). Of the ``m`` earlier
    views the older half takes the first ``ceil(m / 2)`` and the newer half the rest, so both
    halves have at least two observations whenever ``m >= 2``; a track with fewer earlier views
    cannot be cut at all and comes back with two empty masks."""
    earlier = seen[:, :-1]
    count = earlier.sum(axis=1)
    rank = np.cumsum(earlier, axis=1) - 1
    cut = (count + 1) // 2
    splittable = (count >= 2)[:, None]
    older = earlier & (rank < cut[:, None]) & splittable
    newer = earlier & (rank >= cut[:, None]) & splittable
    anchor = seen[:, -1:] & splittable
    return (
        np.concatenate([older, anchor], axis=1),
        np.concatenate([newer, anchor], axis=1),
    )


def _split_gap(
    rays: Array,
    origins: Array,
    pixels: Array,
    seen: npt.NDArray[np.bool_],
    intr: Intrinsics,
    motions: Sequence[Motion],
    *,
    disparity_sigma_px: float,
    sigma_model: str,
) -> Array:
    """How far a track's older and newer halves disagree about its depth, in combined sigmas.

    A static point has one depth, and every subset of the views that saw it must read that
    depth: solving the older half and the newer half separately (each anchored on the current
    frame, :func:`_halves`) and comparing them is the cheapest test of that there is — three
    3x3 solves instead of one. What it sees is a depth that CHANGES as the window slides, which
    is what a mistracked hop halfway through a window does once the reprojection gate has been
    diluted by the square root of the view count.

    What it does NOT see, and this is a property of monocular geometry rather than of this
    implementation: a point whose own motion is parallel to the camera's, and a corner drifting
    along its epipolar line by an amount proportional to the baseline. Both of those make every
    observation consistent with a STATIC point at a different depth — all the rays meet exactly,
    at the wrong place — so every subset reads the same wrong number and every half agrees.
    Measured (scratch/parallax_split_probe.py, 2026-09-15): an object receding at 0.1 m/s from a
    cart driving at 0.25 reads 3.33 m for a 2.00 m truth with the halves 0.00 sigma apart, and
    it is still only 1.19 sigma apart with the cart turning at 0.5 rad/s; a 2 px-per-hop drift
    along the epipolar line reads 2.49 m for 2.00 m with the halves 0.18 sigma apart. Those two
    holes need a second sensor or the network's own depth, not another geometric test.

    Returns 0 where a track cannot be cut (fewer than two earlier views) or either half fails to
    triangulate, so the gate never fires on a track it could not judge."""
    older, newer = _halves(seen)
    told = []
    for half in (older, newer):
        point, ok = _meet_rays(rays, origins, half)
        residual, depth = _reproject(point, pixels, intr, motions)
        rms = _rms_residual(residual, half)
        forward = np.all(~half | (depth > NEAR_M), axis=1)
        z = point[:, 2]
        good = ok & forward & np.isfinite(z) & (z > NEAR_M)
        focal = 0.5 * (intr.fx + intr.fy)
        baseline = np.sqrt((_perpendicular(rays, origins, half) ** 2).sum(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            noise = (
                _solve_sigma(point, rays, origins, half, disparity_sigma_px, rms, focal)
                if sigma_model == "covariance"
                else np.where(good, z, np.nan) ** 2
                * np.hypot(disparity_sigma_px, np.where(np.isfinite(rms), rms, 0.0))
                / (focal * baseline)
            )
        told.append((np.where(good, z, np.nan), np.where(good & (baseline > 0), noise, np.inf)))
    (z_old, sigma_old), (z_new, sigma_new) = told
    combined = np.hypot(sigma_old, sigma_new)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.abs(z_old - z_new) / combined
    out: Array = np.where(np.isfinite(gap), gap, 0.0)
    return out


def triangulate_tracks(
    tracks: Tracks,
    intr: Intrinsics,
    *,
    disparity_sigma_px: float = DISPARITY_SIGMA_PX,
    outlier_px: float = TRACK_OUTLIER_PX,
    sigma_model: str = TRACK_SIGMA_MODEL,
    measure_split: bool = True,
) -> TrackDepths:
    """Every track's depth in the current camera, from all of its observations at once.

    The rays of all the views a track was seen in are met in one linear least-squares solve
    (:func:`_meet_rays`); then one robust pass: a track whose worst observation misses the
    solved point by more than ``outlier_px`` drops that single observation and is solved again,
    which repairs a mistracked hop without throwing the other views away (a track left with
    fewer than two observations is simply not a measurement and comes back NaN).

    ``sigma_model`` decides what the sigma is. ``covariance`` (the default) takes it from the
    solve's own covariance (:func:`_solve_sigma`), which is right whatever the shape of the
    bundle; ``baseline`` is the closed form ``z^2 * sigma_px / (f * B_effective)`` the stage
    shipped with, which is exact for a camera moving ACROSS the ray and up to twice optimistic
    for one driving along it — see :class:`TrackDepths`.

    ``measure_split`` also solves each track's two halves (:func:`_split_gap`) to fill in
    ``split``. That is two more solves — 3.9 ms a frame at eight views, a seventh of the whole
    stage — so a caller that does not gate on the number should not pay for it."""
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
    perpendicular = _perpendicular(rays, origins, seen)
    baseline = np.sqrt((perpendicular**2).sum(axis=1))
    stepped = np.where(seen, np.linalg.norm(origins, axis=1)[None, :] ** 2, 0.0)
    travel = np.sqrt(stepped.sum(axis=1))
    rms = _rms_residual(residual, seen)
    forward = np.all(~seen | (depth > NEAR_M), axis=1)
    z = point[:, 2]
    good = ok & forward & np.isfinite(z) & (z > NEAR_M)
    z = np.where(good, z, np.nan)
    focal = 0.5 * (intr.fx + intr.fy)
    if sigma_model not in TRACK_SIGMA_MODELS:
        raise ValueError(
            f"unknown sigma_model {sigma_model!r}: one of {', '.join(TRACK_SIGMA_MODELS)}"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_px = np.hypot(disparity_sigma_px, np.where(np.isfinite(rms), rms, 0.0))
        told = (
            _solve_sigma(point, rays, origins, seen, disparity_sigma_px, rms, focal)
            if sigma_model == "covariance"
            else z**2 * sigma_px / (focal * baseline)
        )
        sigma = np.where(good & (baseline > 0), told, np.inf)
        widest = np.argmax(np.where(seen, perpendicular, -1.0), axis=1)
        one = perpendicular[rows, widest]
        one_px = np.hypot(disparity_sigma_px, np.nan_to_num(residual[rows, widest]))
        sigma_two = np.where(good & (one > 0), z**2 * one_px / (focal * one), np.inf)
    split = (
        _split_gap(
            rays,
            origins,
            pixels,
            seen,
            intr,
            tracks.motions,
            disparity_sigma_px=disparity_sigma_px,
            sigma_model=sigma_model,
        )
        if measure_split
        else np.zeros(seen.shape[0])
    )
    return TrackDepths(
        z,
        sigma,
        rms,
        baseline,
        travel,
        seen.sum(axis=1),
        sigma_two,
        repaired=int(repaired.sum()),
        split=split,
    )


def gate_tracks(
    tracks: Tracks,
    intr: Intrinsics,
    *,
    matcher: str = "klt",
    min_obs: int = TRACK_MIN_OBS,
    min_total_baseline_m: float = TRACK_MIN_TOTAL_BASELINE_M,
    min_baseline_m: float = MIN_BASELINE_M,
    min_parallax_ratio: float = MIN_PARALLAX_RATIO,
    epipole_min_deg: float = EPIPOLE_MIN_DEG,
    max_sampson_px: float = MAX_SAMPSON_PX,
    max_reproj_px: float = MAX_REPROJ_PX,
    max_weight: float = MAX_WEIGHT,
    outlier_px: float = TRACK_OUTLIER_PX,
    sigma_model: str = TRACK_SIGMA_MODEL,
    split_tol_sigma: float = TRACK_SPLIT_TOL_SIGMA,
) -> ParallaxTruth:
    """A bundle of tracks gated, triangulated and weighed, whoever followed the corners: the
    half of :func:`track_truth` that does not care how the pixels were found, so the backward
    window (:func:`build_tracks`) and the forward store (:class:`TrackStore`) come out as the
    same :class:`ParallaxTruth` with the same gates in the same order.

    ``tracks`` must carry the current frame as its last view (its motion the identity), which
    is where every depth is reported. The gates a track meets, in order: the motion itself
    (nothing moved ``min_baseline_m`` and there is nothing to triangulate — ``still`` when the
    camera did not turn either, ``rotation-only`` when it only turned); the epipolar distance
    of each OBSERVATION to the known motion of its view, which drops that observation and not
    the whole track; landing inside the current picture; ``min_obs`` observations left; a depth
    in front of every lens that saw it; the parallax those observations add up to
    (``min_total_baseline_m`` of effective baseline, ``min_parallax_ratio`` of the depth, and
    ``epipole_min_deg`` off the direction of travel); the bundle's own reprojection error
    (``max_reproj_px``); and, only when ``split_tol_sigma`` is above 0, the two halves of the
    window agreeing about the depth (:func:`_split_gap`)."""
    earlier = tracks.motions[:-1]
    if not earlier or max(m.baseline for m in earlier) < min_baseline_m:
        turned = bool(earlier) and max(m.angle for m in earlier) > MIN_ROTATION_RAD
        return ParallaxTruth.nothing("rotation-only" if turned else "still")
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
    found = triangulate_tracks(
        tracks,
        intr,
        disparity_sigma_px=sigma_px,
        outlier_px=outlier_px,
        sigma_model=sigma_model,
        measure_split=split_tol_sigma > 0,
    )
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
        if split_tol_sigma > 0:
            disagreed = keep & (found.split > split_tol_sigma)
            rejected["split"] = int(disagreed.sum())
            keep &= ~disagreed
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
        split=found.split[keep],
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
    sigma_model: str = TRACK_SIGMA_MODEL,
    split_tol_sigma: float = TRACK_SPLIT_TOL_SIGMA,
) -> ParallaxTruth:
    """:func:`parallax_truth` over a window of frames instead of a pair: track, gate,
    triangulate from every view at once, weigh. The result is the same
    :class:`ParallaxTruth` — the pixels of the CURRENT frame, their depth, their sigma, their
    weight against a lidar beam — with ``observations`` and ``sigma_two`` filled in, so an
    anchor emits the same pairs whichever ruler it used.

    ``grays`` is oldest first with the current frame last and ``motions`` is one per earlier
    view into the current camera (:func:`build_tracks`). The matcher's own check comes first
    (forward-backward per hop for the flow, the ratio and cross-check for the describer) and
    every gate after it is :func:`gate_tracks`', in its order and under its names — including
    ``split_tol_sigma``, which set huge computes the split and gates on nothing, the way its
    distribution was measured."""
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
    return gate_tracks(
        tracks,
        intr,
        matcher=matcher,
        min_obs=min_obs,
        min_total_baseline_m=min_total_baseline_m,
        min_baseline_m=min_baseline_m,
        min_parallax_ratio=min_parallax_ratio,
        epipole_min_deg=epipole_min_deg,
        max_sampson_px=max_sampson_px,
        max_reproj_px=max_reproj_px,
        max_weight=max_weight,
        outlier_px=outlier_px,
        sigma_model=sigma_model,
        split_tol_sigma=split_tol_sigma,
    )


# ---- the forward ruler: a corner born once and followed one hop a frame -----------------------
TRACKINGS = ("forward", "window", "pair")
Tracking = Literal["forward", "window", "pair"]
TRACK_MAX_TRACKS = 200  # corners the store follows at once: the cost of one LK call, not of many
TRACK_REDETECT_EVERY = 5  # frames between two hunts for new corners
TRACK_DETECT_FLOOR = 0.6  # live corners under this share of the cap start a hunt early
TRACK_VERIFY_EVERY = 10  # frames between two rounds of the long-range drift bound; 0 is off
TRACK_DRIFT_TOL_PX = 1.0  # how far the hopped corner may sit from where its birth patch lands
DETECT_GRID = (4, 3)  # columns x rows the detector spreads new corners over, so the TOP of the
# picture gets corners too — which is exactly where the lidar's one plane never reaches, and the
# only elevation a law over the ray's angle can be fitted at.
TRACK_DEATHS = ("lk", "fb", "edge", "drift", "source")


class Flow(Protocol):
    """Who moves a set of points from one picture to the next — the one thing the forward store
    needs of a tracker, so a test may hand it a fake that drifts on purpose."""

    def track(
        self,
        a: npt.NDArray[np.uint8],
        b: npt.NDArray[np.uint8],
        points: Pixels,
        *,
        backward: bool = True,
        guess: Pixels | None = None,
    ) -> tuple[Pixels, npt.NDArray[np.bool_], Array]:
        """Where each of ``points`` landed in ``b``, whether the tracker kept it at all, and how
        far it came back from where it started when followed back into ``a`` (``inf`` per point
        when ``backward`` is off). ``guess`` starts the search at a known position instead of at
        the point itself, which is what makes a jump across a whole window converge."""
        ...


@dataclass(frozen=True)
class LucasKanade:
    """Pyramidal Lucas-Kanade as the store's :class:`Flow`: one call forward, one back for the
    check, the same window and pyramid the pair path uses."""

    window: int = LK_WINDOW
    levels: int = LK_LEVELS

    def track(
        self,
        a: npt.NDArray[np.uint8],
        b: npt.NDArray[np.uint8],
        points: Pixels,
        *,
        backward: bool = True,
        guess: Pixels | None = None,
    ) -> tuple[Pixels, npt.NDArray[np.bool_], Array]:
        """:meth:`Flow.track` by optical flow: (landed pixels, kept, the forward-backward
        distance in pixels)."""
        import cv2

        empty: Pixels = np.zeros((0, 2), dtype=np.float32)
        start = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if start.shape[0] == 0:
            return empty, np.zeros(0, dtype=bool), np.zeros(0)
        first = np.ascontiguousarray(a)
        second = np.ascontiguousarray(b)
        size = (self.window, self.window)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        flow: Any = cv2.calcOpticalFlowPyrLK  # cv2's stubs admit neither uint8 nor a None output
        seed = None if guess is None else np.asarray(guess, dtype=np.float32).reshape(-1, 1, 2)
        forward, ok, _ = flow(
            first,
            second,
            start,
            None if seed is None else seed.copy(),
            winSize=size,
            maxLevel=self.levels,
            criteria=criteria,
            **({} if seed is None else {"flags": cv2.OPTFLOW_USE_INITIAL_FLOW}),
        )
        kept: npt.NDArray[np.bool_] = ok.ravel() == 1
        drift = np.full(start.shape[0], np.inf)
        if backward and bool(kept.any()):
            back, ok_back, _ = flow(
                second, first, forward, None, winSize=size, maxLevel=self.levels, criteria=criteria
            )
            kept &= ok_back.ravel() == 1
            drift = np.linalg.norm((back - start).reshape(-1, 2), axis=1)
        landed: Pixels = forward.reshape(-1, 2).astype(np.float32)
        return landed, kept, drift


@dataclass(frozen=True)
class FrameView:
    """One frame a track may be triangulated from: when it was taken, the whole ``base_link <-
    camera_optical`` the lens sat at then (the neck's pan included), and whose word the cart's
    motion at that moment was — ``tracker`` (the map pose) or ``odom``."""

    stamp: float
    place: Placement
    source: str


@dataclass(frozen=True)
class Observation:
    """Where one track was seen in one frame: its sub-pixel (column, row) and the frame."""

    pixel: Pixels
    view: FrameView

    @property
    def stamp(self) -> float:
        """When the observation was made, in seconds."""
        return self.view.stamp

    @property
    def source(self) -> str:
        """Whose word the motion at that frame was."""
        return self.view.source


@dataclass(eq=False)
class Track:
    """One corner, born where a detector found it and followed forward one hop a frame: where
    it sits in the CURRENT picture, when it was born, the observations kept for its solve
    (oldest first, all inside the window), the observations this frame actually uses, the ORB
    descriptor its last sighting carried (``None`` for the flow) and how many hops it has
    survived."""

    ident: int
    pixel: Pixels
    born: float
    observations: list[Observation] = field(default_factory=list)
    used: list[Observation] = field(default_factory=list)
    descriptor: npt.NDArray[np.uint8] | None = None
    hops: int = 0

    @property
    def anchor(self) -> float | None:
        """The stamp of the oldest observation still inside the window — the frame the drift
        bound re-tracks this corner from; ``None`` for a track with no observation yet."""
        return self.observations[0].stamp if self.observations else None


@dataclass(frozen=True)
class TrackReport:
    """What one frame cost the store and what it did to the corners: how many are alive, how
    many were born, how many died of each cause (``lk`` the tracker lost it, ``fb`` it failed
    the forward-backward check, ``edge`` it left the picture, ``drift`` it disagreed with its
    own birth patch, ``source`` its older views were cut off by a change of motion source —
    the corner itself lives on, its bundle starts again), the median observations a live track
    carries, and the milliseconds of the hop, the detection and the drift bound."""

    live: int
    born: int
    died: dict[str, int]
    observations: float
    hop_ms: float
    detect_ms: float
    verify_ms: float


class TrackStore:
    """Corners followed FORWARD, one hop a frame, for as long as each of them survives.

    The backward build (:func:`build_tracks`) re-tracks the current frame's corners through
    every frame of the window on every frame, so its cost is two flow calls per view and a
    long window is unaffordable: 27 ms a frame at 8 views, 52 at 16
    (scratch/parallax_tracks_audit.txt). This one pays two flow calls per FRAME whatever the
    window: a corner is detected once, followed from the previous grey to the current one with
    the forward-backward check, and each frame appends an observation to the corner it already
    has. The window is then free to be as long as the pose is good, which is the whole point —
    every error term in a triangulated depth divides by the baseline (pixel noise as
    ``z^2 sigma_px / (f B)``, the pose's own 1-2 cm as ``1 / B``), the effective baseline over
    1.5 s of this cart's errand is about 14 cm, and the tracker's map pose is absolute, so 3 s
    costs the pose nothing and doubles the divisor.

    What the store is careful about, in the order a frame meets it:

    * **the hop** — one flow call for every live corner, one back for the check. A corner the
      tracker loses, that comes back more than ``fb_tol_px`` from where it started, or that
      leaves the picture, is closed.
    * **the drift bound** — a per-hop check cannot see the drift that matters. Lucas-Kanade
      slides along an edge and along the epipolar line by a fraction of a pixel a hop, each hop
      passing its own forward-backward test, and fifty hops later the corner is somewhere else
      at a depth that is wrong and consistent. So every ``verify_every`` frames each corner is
      re-tracked DIRECTLY from the grey of the frame its oldest kept observation was made in,
      started at where the hops say it is, and closed when the two disagree by more than
      ``drift_tol_px``. One flow call per kept grey, one grey a frame, so no frame pays more
      than one.
    * **the window** — observations older than ``window_s`` are dropped and the track lives on.
      A window shortened live is therefore a shorter bundle on the very next frame, with no
      restart and nothing reset.
    * **the views** — a solve rests on at most ``max_views`` observations, evenly spaced, the
      oldest kept and the current frame always. Observations are only STORED every
      ``window_s / max_views`` seconds (and on every frame while the window holds fewer than
      ``max_views`` of them), which is what keeps the number of poses a frame must ask for
      bounded by ``max_views`` however long the window is.
    * **the detector** — new corners every ``redetect_every`` frames, or as soon as the live
      count falls under ``TRACK_DETECT_FLOOR`` of ``max_tracks``, masked away from the corners
      already alive and balanced over a :data:`DETECT_GRID` so the top of the picture is filled
      too. The lidar's plane never reaches there, and a law over the ray's elevation needs it.
    * **the motion source** — every observation carries whose word the motion at its frame was.
      A solve walks back from the current frame while the source stays the same and stops at
      the first observation from the other one: the two disagree by about a quarter over a
      second (scratch/parallax_pose_sweep.txt) and a bundle half measured by each is not a
      geometry at all. A track whose source changed keeps living and starts its bundle again;
      after ``window_s`` the other source's observations have left the window by themselves.

    With ``matcher`` ``orb`` the hop is a descriptor match instead — the current frame is
    described once and matched against the descriptors the live corners carried out of the
    previous frame — and the drift bound does not apply, a match being a recognition rather
    than a hop."""

    def __init__(
        self,
        *,
        window_s: float = TRACK_WINDOW_S,
        max_views: int = TRACK_MAX_VIEWS,
        max_tracks: int = TRACK_MAX_TRACKS,
        redetect_every: int = TRACK_REDETECT_EVERY,
        verify_every: int = TRACK_VERIFY_EVERY,
        drift_tol_px: float = TRACK_DRIFT_TOL_PX,
        matcher: str = "klt",
        quality: float = CORNER_QUALITY,
        min_distance: int = CORNER_MIN_DISTANCE,
        fb_tol_px: float = FB_TOL_PX,
        ratio: float = ORB_RATIO,
        flow: Flow | None = None,
    ) -> None:
        if matcher not in MATCHERS:
            raise ValueError(f"unknown matcher {matcher!r}: one of {', '.join(MATCHERS)}")
        self.window_s = window_s  # live: every knob here is read afresh on each frame
        self.max_views = max_views
        self.max_tracks = max_tracks
        self.redetect_every = redetect_every
        self.verify_every = verify_every
        self.drift_tol_px = drift_tol_px
        self.matcher = matcher
        self.quality = quality
        self.min_distance = min_distance
        self.fb_tol_px = fb_tol_px
        self.ratio = ratio
        self._flow: Flow = flow if flow is not None else LucasKanade()
        self._tracks: list[Track] = []
        self._described: Features | None = None  # the describer's reading of the current frame
        self._taken: npt.NDArray[np.bool_] = np.zeros(0, dtype=bool)  # its keypoints recognised
        self._next = 0
        self._gray: npt.NDArray[np.uint8] | None = None
        self._keys: list[FrameView] = []
        self._greys: dict[float, npt.NDArray[np.uint8]] = {}
        self._pending: list[FrameView] = []
        self._queue: list[float] = []
        self._since_detect = 0
        self._since_verify = 0
        self._frames = 0

    @property
    def live(self) -> int:
        """How many corners the store is following right now."""
        return len(self._tracks)

    def reset(self) -> None:
        """Forget every corner and every kept picture — a new tape, or a jump in time."""
        self._tracks.clear()
        self._gray = None
        self._keys.clear()
        self._greys.clear()
        self._pending.clear()
        self._queue.clear()
        self._since_detect = self._since_verify = self._frames = 0

    def follow(
        self, gray: npt.NDArray[np.uint8], stamp: float, place: Placement, source: str
    ) -> TrackReport:
        """Take one frame: hop every live corner into it, verify one kept grey against it,
        forget what the window no longer covers, detect new corners when they are due, keep the
        observation when the frame is a view, and choose the views every track's next solve
        rests on. Returns what it cost and what it did (:class:`TrackReport`)."""
        view = FrameView(float(stamp), place, str(source))
        died = dict.fromkeys(TRACK_DEATHS, 0)
        hop_ms = detect_ms = verify_ms = 0.0
        if self.matcher == "orb":  # described once a frame, whoever asks for it
            started = time.perf_counter()
            self._described = describe_frame(gray)
            self._taken = np.zeros(self._described.count, dtype=bool)
            hop_ms += 1000.0 * (time.perf_counter() - started)
        if self._gray is not None and self._tracks:
            started = time.perf_counter()
            self._hop(gray, died)
            hop_ms = 1000.0 * (time.perf_counter() - started)
        started = time.perf_counter()
        self._verify(gray, died)
        verify_ms = 1000.0 * (time.perf_counter() - started)
        self._forget(view.stamp)
        born = 0
        self._since_detect += 1
        if self._due():
            started = time.perf_counter()
            born = self._detect(gray, view)
            detect_ms = 1000.0 * (time.perf_counter() - started)
            self._since_detect = 0
        if self._is_view(view, born):
            self._keys.append(view)
            self._greys[view.stamp] = gray
            for track in self._tracks:
                track.observations.append(Observation(track.pixel.copy(), view))
        died["source"] = self._select(view)
        self._gray = gray
        self._frames += 1
        counts = [float(len(t.observations)) for t in self._tracks]
        return TrackReport(
            live=len(self._tracks),
            born=born,
            died=died,
            observations=float(np.median(counts)) if counts else 0.0,
            hop_ms=hop_ms,
            detect_ms=detect_ms,
            verify_ms=verify_ms,
        )

    def views(self) -> list[FrameView]:
        """The earlier frames the next solve needs a motion for, oldest first — at most
        ``max_views`` of them however long the window is, because that is how sparsely the
        store keeps an observation in the first place."""
        return list(self._pending)

    def tracks(self, motions: Mapping[float, Motion]) -> Tracks:
        """The live corners as the bundle :func:`gate_tracks` triangulates: one row per track
        that has at least one usable earlier view, one column per view the caller could answer
        for, and the current frame as the last column (its motion the identity, its pixel the
        one every depth is reported at). ``motions`` maps a view's stamp to the transform
        taking that view's optical frame into the current camera's; a view left out of it is
        simply not used — which is how a caller drops the views its motion source would not
        answer for without mixing two sources into one geometry."""
        stamps = [v.stamp for v in self._pending if v.stamp in motions]
        index = {stamp: i for i, stamp in enumerate(stamps)}
        rows = [t for t in self._tracks if any(o.stamp in index for o in t.used)]
        views = len(stamps) + 1
        pixels = np.full((len(rows), views, 2), np.nan)
        seen = np.zeros((len(rows), views), dtype=bool)
        for r, track in enumerate(rows):
            for obs in track.used:
                slot = index.get(obs.stamp)
                if slot is None:
                    continue
                pixels[r, slot] = obs.pixel
                seen[r, slot] = True
            pixels[r, -1] = track.pixel
            seen[r, -1] = True
        ordered = (*(motions[stamp] for stamp in stamps), _identity())
        return Tracks(pixels, seen, ordered, found=len(self._tracks))

    # ---- one frame, step by step ---------------------------------------------------------
    def _hop(self, gray: npt.NDArray[np.uint8], died: dict[str, int]) -> None:
        """Every live corner moved into this frame, and the ones that did not make it closed."""
        if self.matcher == "orb":
            self._recognise(died)
            return
        previous = self._gray
        if previous is None:
            return
        points = np.array([t.pixel for t in self._tracks], dtype=np.float32)
        landed, kept, drift = self._flow.track(previous, gray, points, backward=True)
        height, width = gray.shape[:2]
        steady = kept & (drift <= self.fb_tol_px)
        inside = (
            (landed[:, 0] >= 0)
            & (landed[:, 0] < width)
            & (landed[:, 1] >= 0)
            & (landed[:, 1] < height)
        )
        died["lk"] += int((~kept).sum())
        died["fb"] += int((kept & ~steady).sum())
        died["edge"] += int((steady & ~inside).sum())
        good = steady & inside
        alive: list[Track] = []
        for track, ok, where in zip(self._tracks, good, landed, strict=True):
            if not bool(ok):
                continue
            track.pixel = np.asarray(where, dtype=np.float32)
            track.hops += 1
            alive.append(track)
        self._tracks = alive

    def _recognise(self, died: dict[str, int]) -> None:
        """The describer's hop: this frame described once, its keypoints matched against the
        descriptors the live corners carried out of the previous frame, and the corners nobody
        recognised closed. What the match does not find is left for :meth:`_detect`."""
        frame = self._described
        if frame is None:
            return
        held = [t for t in self._tracks if t.descriptor is not None]
        if not held or frame.count == 0:
            died["lk"] += len(self._tracks)
            self._tracks = []
            return
        mine = Features(
            np.array([t.pixel for t in held], dtype=np.float32),
            np.array([t.descriptor for t in held], dtype=np.uint8),
        )
        rows, theirs = _match_features(mine, frame, self.ratio)
        taken = np.zeros(frame.count, dtype=bool)
        alive: list[Track] = []
        for row, column in zip(rows, theirs, strict=True):
            track = held[int(row)]
            track.pixel = np.asarray(frame.points[int(column)], dtype=np.float32)
            assert frame.descriptors is not None
            track.descriptor = frame.descriptors[int(column)]
            track.hops += 1
            taken[int(column)] = True
            alive.append(track)
        died["lk"] += len(self._tracks) - len(alive)
        self._tracks = alive
        self._taken = taken

    def _verify(self, gray: npt.NDArray[np.uint8], died: dict[str, int]) -> None:
        """The long-range drift bound: one kept grey a frame, its corners re-tracked directly
        into this picture from where the hops say they are, and every corner whose birth patch
        lands more than ``drift_tol_px`` away closed. A cycle over every kept grey starts every
        ``verify_every`` frames, one grey per frame, so no frame pays for more than one call."""
        if self.matcher != "klt" or self.verify_every <= 0 or self.drift_tol_px <= 0:
            return
        self._since_verify += 1
        if not self._queue and self._since_verify >= self.verify_every:
            self._queue = [key.stamp for key in self._keys if key.stamp in self._greys]
            self._since_verify = 0
        while self._queue:
            stamp = self._queue.pop(0)
            older = self._greys.get(stamp)
            group = [t for t in self._tracks if t.anchor == stamp]
            if older is None or not group:
                continue  # that grey has left the window, or holds nobody's oldest view
            birth = np.array([t.observations[0].pixel for t in group], dtype=np.float32)
            hopped = np.array([t.pixel for t in group], dtype=np.float32)
            landed, kept, _ = self._flow.track(older, gray, birth, backward=False, guess=hopped)
            slid = np.linalg.norm(landed - hopped, axis=1)
            drifted = kept & (slid > self.drift_tol_px)
            if bool(drifted.any()):
                died["drift"] += int(drifted.sum())
                gone = {id(t) for t, bad in zip(group, drifted, strict=True) if bool(bad)}
                self._tracks = [t for t in self._tracks if id(t) not in gone]
            return

    def _forget(self, now: float) -> None:
        """Drop what the window no longer covers: every observation older than ``window_s``,
        the views they were made in and the greys kept for them. The tracks themselves live."""
        cutoff = now - self.window_s
        for track in self._tracks:
            if track.observations and track.observations[0].stamp < cutoff:
                track.observations = [o for o in track.observations if o.stamp >= cutoff]
        self._keys = [key for key in self._keys if key.stamp >= cutoff]
        alive = {key.stamp for key in self._keys}
        self._greys = {stamp: grey for stamp, grey in self._greys.items() if stamp in alive}
        self._queue = [stamp for stamp in self._queue if stamp in alive]

    def _due(self) -> bool:
        """Whether to hunt for new corners: every ``redetect_every`` frames, or as soon as the
        live count falls under :data:`TRACK_DETECT_FLOOR` of the cap (a turn or a doorway can
        take three corners in four in one frame, and waiting for the cadence wastes the window)."""
        if not self._tracks:
            return True
        if self._since_detect >= max(self.redetect_every, 1):
            return True
        return len(self._tracks) < TRACK_DETECT_FLOOR * self.max_tracks

    def _detect(self, gray: npt.NDArray[np.uint8], view: FrameView) -> int:
        """New corners where there are none: masked away from the corners already alive and
        asked for cell by cell over :data:`DETECT_GRID`, so the top of the picture is filled
        even though its corners are weaker than the floor's. Returns how many were born."""
        room = self.max_tracks - len(self._tracks)
        if room <= 0:
            return 0
        if self.matcher == "orb":
            return self._adopt(room, view)
        import cv2

        height, width = gray.shape[:2]
        mask = np.full((height, width), 255, dtype=np.uint8)
        for track in self._tracks:
            centre = (round(float(track.pixel[0])), round(float(track.pixel[1])))
            cv2.circle(mask, centre, int(self.min_distance), 0, -1)
        columns, rows = DETECT_GRID
        share = max(1, -(-room // (columns * rows)))
        picture = np.ascontiguousarray(gray)
        born = 0
        for cx in range(columns):
            for cy in range(rows):
                x0, x1 = width * cx // columns, width * (cx + 1) // columns
                y0, y1 = height * cy // rows, height * (cy + 1) // rows
                found = cv2.goodFeaturesToTrack(
                    picture[y0:y1, x0:x1],
                    maxCorners=min(share, room - born),
                    qualityLevel=self.quality,
                    minDistance=self.min_distance,
                    mask=mask[y0:y1, x0:x1],
                )
                if found is None:
                    continue
                for point in found.reshape(-1, 2):
                    self._born(np.array([point[0] + x0, point[1] + y0], dtype=np.float32), view)
                    born += 1
                    if born >= room:
                        return born
        return born

    def _adopt(self, room: int, view: FrameView) -> int:
        """The describer's detection: the keypoints of this frame nobody recognised become new
        corners, strongest first (ORB returns them in that order)."""
        frame = self._described
        if frame is None or frame.descriptors is None:
            return 0
        free = np.flatnonzero(~self._taken)
        born = 0
        for column in free[:room]:
            track = self._born(np.asarray(frame.points[column], dtype=np.float32), view)
            track.descriptor = frame.descriptors[column]
            born += 1
        return born

    def _born(self, pixel: Pixels, view: FrameView) -> Track:
        """One new corner, alive from this frame on."""
        track = Track(self._next, np.asarray(pixel, dtype=np.float32), view.stamp)
        self._next += 1
        self._tracks.append(track)
        return track

    def _is_view(self, view: FrameView, born: int) -> bool:
        """Whether this frame is kept as a view a solve may rest on: always while the window
        holds fewer than ``max_views`` of them or corners were just born in it (a newborn with
        no observation at its own frame would start its bundle a cadence late), and every
        ``window_s / max_views`` seconds after that — which is what bounds the poses a frame
        must ask for, and the greys the drift bound keeps, however long the window is."""
        if not self._keys or born:
            return True
        if len(self._keys) < max(self.max_views, 2):
            return True
        spacing = self.window_s / max(self.max_views, 1)
        return view.stamp - self._keys[-1].stamp >= spacing

    def _select(self, view: FrameView) -> int:
        """Choose the views every track's next solve rests on and list the frames a motion is
        needed for. A track's run is walked back from the current frame while the motion source
        stays the current frame's and stopped at the first observation from the other one; of
        that run at most ``max_views - 1`` are kept, evenly spaced, the oldest always (the
        current frame is the anchor and is always the last view). Returns how many tracks the
        source cut short."""
        now = view.stamp
        cut = 0
        wanted: dict[float, FrameView] = {}
        for track in self._tracks:
            run: list[Observation] = []
            broke = False
            for obs in reversed(track.observations):
                if obs.source != view.source:
                    broke = True
                    break
                if obs.stamp < now:
                    run.append(obs)
            run.reverse()
            if broke:  # nothing older than the change can ever be used again: drop it once
                cut += 1
                oldest = run[0].stamp if run else now
                track.observations = [o for o in track.observations if o.stamp >= oldest]
            span = _view_span(len(run), max(self.max_views - 1, 1))
            track.used = [run[i] for i in span]
            for obs in track.used:
                wanted.setdefault(obs.stamp, obs.view)
        self._pending = [wanted[stamp] for stamp in sorted(wanted)]
        return cut


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
    "TRACKINGS",
    "TRACK_DEATHS",
    "TRACK_DRIFT_TOL_PX",
    "TRACK_MAX_TRACKS",
    "TRACK_MAX_VIEWS",
    "TRACK_MIN_OBS",
    "TRACK_MIN_TOTAL_BASELINE_M",
    "TRACK_OUTLIER_PX",
    "TRACK_REASONS",
    "TRACK_REDETECT_EVERY",
    "TRACK_SIGMA_MODEL",
    "TRACK_SIGMA_MODELS",
    "TRACK_SPLIT_TOL_SIGMA",
    "TRACK_VERIFY_EVERY",
    "TRACK_WINDOW_S",
    "CameraPlacement",
    "Features",
    "Flow",
    "FrameView",
    "LucasKanade",
    "Matcher",
    "Motion",
    "Observation",
    "ParallaxTruth",
    "Pixels",
    "Placement",
    "Track",
    "TrackDepths",
    "TrackReport",
    "TrackStore",
    "Tracking",
    "Tracks",
    "build_tracks",
    "camera_motion",
    "describe_frame",
    "fundamental",
    "gate_tracks",
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
