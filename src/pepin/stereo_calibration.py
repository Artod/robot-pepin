"""Measuring the stereo head with a printed checkerboard: what to collect, how to solve it, and
what evidence says the answer may be believed.

A stereo pair is two lenses and one rigid bar between them, and the bar is the hard part. Each
eye alone is the mono problem :mod:`pepin.calibration` already solves; what a stereo calibration
adds is the pose of the right eye in the left eye's frame, and it is measured from the SAME
frame seen twice — the board must be found in both eyes of one capture, or the pair says nothing
about the bar.

So this module is the mono module's shape with a second eye bolted on:

* :func:`find_pair` finds the board in both eyes and — the trap of stereo — makes their corner
  ORDERS agree. ``findChessboardCorners`` resolves a rectangular board only up to a turn of 180
  degrees, and a pair whose eyes disagree about which corner is number zero fits a bar that does
  not exist. :func:`match_order` decides it from the rows the two eyes see.
* :class:`PairCollector` keeps a pair when the board is sharp, whole in both eyes, and has MOVED
  since the last kept one — on a 94 degree lens a set shot from one spot leaves the corners of
  the image, where the distortion lives, unmeasured. The coverage bookkeeping is
  :class:`pepin.calibration.Coverage`, unchanged, on the left eye's corners.
* :func:`fit_eye` fits one lens under a named distortion model and :func:`model_evidence` fits
  every model this lens might need and reports what each one buys ON HELD-OUT VIEWS, so the
  choice between plumb_bob and rational is a measurement and not a preference.
* :func:`stereo_fit` fixes the two lenses, solves the bar, then refines everything together, and
  answers a :class:`StereoFit` that knows whether it may be written: the baseline it found, the
  RMS of the stereo solve, and the epipolar error left after rectification — the one number that
  says whether a block matcher will find anything along a row.

OpenCV is imported inside the functions that need it, and no function here touches a camera, a
window or a file: the runner (``ros/tools/stereo_calibrate.py``, wrapped by
``ros/calibrate.sh stereo``) pulls the frames and draws, and hands corners in here.

``cv2.findChessboardCornersSB`` is NOT used: it segfaults on the laptop's OpenCV build, which is
why ``tests/unit/test_calibration.py`` is skipped there. The classic detector plus
``cornerSubPix`` is what runs, and it is what the unit tests may safely import.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.calibration import Board, Coverage, Verdict, View, view_shape
from pepin.stereo import StereoCalibration

Array = npt.NDArray[np.float64]
Corners = npt.NDArray[np.float32]  # (n, 1, 2) image points, the shape OpenCV hands back

# A wide lens needs more views than a narrow one, and needs them further apart: the barrel at the
# corner of an 800 px eye is tens of pixels and only a board held THERE measures it.
TARGET_VIEWS = 30
# How far a board must have travelled since the last kept pair to be worth keeping: any one of a
# move across the frame, a change of distance, or a change of tilt.
MOVE_FRACTION = 0.06  # of the frame's diagonal
SCALE_STEP = 0.18  # |log(area / previous area)|, about a 20 % change of distance
TILT_STEP = 0.07  # of pepin.calibration's tilt, 0 fronto-parallel towards 1 fully squashed
SHARP_MIN = 40.0  # Laplacian variance over the board's box; a blurred board is under this
# A board held far away is a board of small squares, and small squares are badly measured corners.
# Measured 2026-09-21 on the robot's head (800 px an eye): 47 views with squares of 9-16 px fitted
# an eye to 2.8 px, and the views with squares of 16 px and more fitted it to 0.3 px. A far board
# is also nearly the same view every time, which leaves the focal length unmeasured (it came out
# at 543, 603 and 1119 px from three subsets of that session).
MIN_SQUARE_PX = 18.0
# The sub-pixel refinement looks at a window around each corner; a window wider than a square
# sees the neighbouring corners and lands between them. Half-window = this fraction of the
# smallest corner spacing, within the bounds.
SUBPIX_WINDOW_FRACTION = 0.4
SUBPIX_HALF_WINDOW_MIN = 3
SUBPIX_HALF_WINDOW_MAX = 11

# What a fit must reach before it is written. The epipolar error is the operational one: a block
# matcher searches along a row, so a pair whose rows disagree by a pixel is searching the wrong
# row for everything small.
EYE_RMS_MAX_PX = 0.6
STEREO_RMS_MAX_PX = 1.0
EPIPOLAR_MAX_PX = 0.7
EPIPOLAR_GOOD_PX = 0.5
# The module's nominal bar is 63 mm. A baseline far from it means the square size was wrong —
# the board's square is the only length in the whole calibration, and every metre scales with it.
BASELINE_MIN_M = 0.03
BASELINE_MAX_M = 0.12
BASELINE_NOMINAL_M = 0.063

MODELS = ("plumb_bob", "rational")
# How much better a richer model must be ON HELD-OUT VIEWS to be worth its extra coefficients:
# by a real fraction AND by a visible number of pixels. The absolute tie-break matters more than
# it looks — when the simple model already explains the lens to a hundredth of a pixel, a
# relative margin alone hands the win to whichever model is luckier in the fourth decimal.
MODEL_GAIN = 0.05  # relative
MODEL_TIE_PX = 0.05  # a simpler model this close to the best has lost nothing worth having
HOLDOUT_EVERY = 4  # one view in four is kept out of the fit and used to score it
HOLDOUT_SEED = 20260920  # fixed, so two runs over the same views choose the same model


# ---- the pair ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Pair:
    """One accepted capture: the board's corners in both eyes of the SAME frame, and the shape
    facts of the left eye that the coverage counts it by."""

    left: Corners
    right: Corners
    view: View

    @property
    def row_offset_px(self) -> float:
        """Mean |y_left - y_right| of the raw corners: what rectification must remove."""
        return row_offset(self.left, self.right)


def row_offset(left: Corners, right: Corners) -> float:
    """Mean vertical disagreement between two corner lists, pixels — the cheap test of whether
    the two eyes are talking about the same corner in the same order."""
    a = np.asarray(left, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(right, dtype=np.float64).reshape(-1, 2)
    return float(np.abs(a[:, 1] - b[:, 1]).mean())


def match_order(left: Corners, right: Corners) -> tuple[Corners, bool]:
    """The right eye's corners re-ordered to index the same physical corners as the left's, and
    whether they had to be turned.

    ``findChessboardCorners`` pins a rectangular board only up to a turn of 180 degrees in its
    own plane, and the two eyes can disagree — the resulting "correspondence" then matches each
    corner with the one diagonally opposite and fits a bar metres long. Reversing a list is
    exactly that 180 degree relabelling, so the fix is to reverse the one that disagrees; which
    one disagrees is read off the rows, because two eyes 63 mm apart see every corner at very
    nearly the same height.
    """
    direct = row_offset(left, right)
    turned = np.ascontiguousarray(np.asarray(right, dtype=np.float32)[::-1])
    if row_offset(left, turned) < direct:
        return turned, True
    return np.asarray(right, dtype=np.float32), False


def find_pair(left_image: Any, right_image: Any, board: Board) -> tuple[Corners, Corners] | None:
    """The board's sub-pixel corners in both eyes of one frame, ordered alike, or ``None`` when
    it is not wholly visible in both.

    The classic ``findChessboardCorners`` plus ``cornerSubPix`` on purpose: the faster
    ``findChessboardCornersSB`` segfaults on the laptop's OpenCV build.
    """
    found_left = find_corners_classic(left_image, board)
    if found_left is None:
        return None
    found_right = find_corners_classic(right_image, board)
    if found_right is None:
        return None
    ordered, _turned = match_order(found_left, found_right)
    return found_left, ordered


def corner_spacings(corners: Corners, board: Board) -> tuple[float, float]:
    """``(smallest, typical)`` distance between neighbouring corners of a board, pixels. The
    smallest bounds the refinement window; the typical one — the median along the board's less
    foreshortened direction — is how large a square looks, whatever the tilt."""
    grid = np.asarray(corners, dtype=np.float64).reshape(board.rows, board.cols, 2)
    along = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    down = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    smallest = float(min(along.min(), down.min()))
    typical = float(max(np.median(along), np.median(down)))
    return smallest, typical


def subpix_half_window(smallest_spacing_px: float) -> int:
    """The refinement's half-window for a board whose closest corners are this far apart: never
    so wide that it reaches the neighbouring corner."""
    half = int(SUBPIX_WINDOW_FRACTION * smallest_spacing_px)
    return max(SUBPIX_HALF_WINDOW_MIN, min(SUBPIX_HALF_WINDOW_MAX, half))


def find_corners_classic(image: Any, board: Board) -> Corners | None:
    """One eye's sub-pixel corners, or ``None``: ``findChessboardCorners`` refined by
    ``cornerSubPix``, never the SB detector."""
    import cv2

    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(grey, board.size, flags=flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    half = subpix_half_window(corner_spacings(corners, board)[0])
    refined = cv2.cornerSubPix(grey, corners, (half, half), (-1, -1), criteria)
    points = np.asarray(refined, dtype=np.float32)
    height, width = grey.shape[:2]
    flat = points.reshape(-1, 2)
    inside = (
        float(flat[:, 0].min()) >= 0.0
        and float(flat[:, 1].min()) >= 0.0
        and float(flat[:, 0].max()) <= width - 1.0
        and float(flat[:, 1].max()) <= height - 1.0
    )
    return points if inside else None


def board_sharpness(image: Any, corners: Corners) -> float:
    """Laplacian variance inside the board's bounding box: how sharp THE BOARD is, not the room
    behind it. A blurred board is the single biggest cause of a calibration that fits nothing."""
    import cv2

    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flat = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    height, width = grey.shape[:2]
    x0 = max(0, int(flat[:, 0].min()))
    y0 = max(0, int(flat[:, 1].min()))
    x1 = min(width, int(flat[:, 0].max()) + 1)
    y1 = min(height, int(flat[:, 1].max()) + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    return float(cv2.Laplacian(grey[y0:y1, x0:x1], cv2.CV_64F).var())


def _centre(corners: Corners) -> Array:
    """The corner cloud's mean, pixels."""
    mean = np.asarray(corners, dtype=np.float64).reshape(-1, 2).mean(axis=0)
    return np.asarray(mean, dtype=np.float64)


def moved_enough(view: View, last: View | None, size: tuple[int, int]) -> bool:
    """Whether a board has moved far enough since the last kept view to add an equation rather
    than repeat one: across the frame, nearer or further, or turned."""
    if last is None:
        return True
    diagonal = float(np.hypot(*size))
    travel = float(np.linalg.norm(_centre(view.corners) - _centre(last.corners)))
    if travel >= MOVE_FRACTION * diagonal:
        return True
    if abs(np.log(max(view.area, 1e-9) / max(last.area, 1e-9))) >= SCALE_STEP:
        return True
    return abs(view.tilt - last.tilt) >= TILT_STEP


class PairCollector:
    """Gathers well-spread stereo pairs from a stream of frames, without a keypress.

    It takes the corners of one frame's two eyes (or ``None`` when the board was not whole in
    both) and answers a :class:`pepin.calibration.Verdict`. A pair is kept when the board is
    sharp, has moved since the last kept one, and lands somewhere the set is still thin.
    """

    def __init__(
        self,
        board: Board,
        size: tuple[int, int],
        target: int = TARGET_VIEWS,
        sharp_min: float = SHARP_MIN,
        min_square_px: float = MIN_SQUARE_PX,
    ) -> None:
        self.board = board
        self.size = size
        self.target = target
        self.sharp_min = sharp_min
        self.min_square_px = min_square_px
        self.pairs: list[Pair] = []

    @property
    def views(self) -> list[View]:
        """The left eye's view of every kept pair, for the coverage."""
        return [p.view for p in self.pairs]

    @property
    def coverage(self) -> Coverage:
        """What the kept pairs have and have not seen, against this run's target."""
        return Coverage.of(self.views, self.target)

    def done(self) -> bool:
        """Whether enough well-spread pairs are in hand."""
        return self.coverage.good()

    def offer(self, corners: tuple[Corners, Corners] | None, sharpness: float) -> Verdict:
        """One frame's two corner lists and how sharp the board was in them. Answers what
        happened and what the operator should do; the pair is appended to :attr:`pairs` when
        kept."""
        coverage = self.coverage
        if corners is None:
            return Verdict(False, None, "board not found in BOTH eyes — show the whole board")
        left, right = corners
        if len(left) != self.board.corners or len(right) != self.board.corners:
            return Verdict(False, None, "board only partly visible — move it into both eyes")
        square = min(corner_spacings(left, self.board)[1], corner_spacings(right, self.board)[1])
        if square < self.min_square_px:
            return Verdict(
                False,
                None,
                f"too far: a square is {square:.0f} px, under {self.min_square_px:.0f} — "
                "bring the board closer",
            )
        cell, area, tilt = view_shape(left, self.size)
        view = View(np.asarray(left, dtype=np.float32), cell, area, tilt)
        if sharpness < self.sharp_min:
            return Verdict(False, None, f"blurred ({sharpness:.0f}) — hold it still a moment")
        last = self.pairs[-1].view if self.pairs else None
        if not moved_enough(view, last, self.size):
            return Verdict(False, None, f"move the board — {coverage.hint()}")
        if not coverage.wants(view):
            return Verdict(False, None, f"already have this one — {coverage.hint()}")
        self.pairs.append(Pair(view.corners, np.asarray(right, dtype=np.float32), view))
        return Verdict(
            True,
            None,
            f"kept pair {len(self.pairs)}/{self.target} — {self.coverage.hint()}",
        )


# ---- one eye -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class EyeFit:
    """One lens as ``cv2.calibrateCamera`` answered it, with the evidence for its model."""

    model: str
    k: Array
    dist: Array
    rms_px: float
    holdout_px: float
    views: int

    @property
    def fx(self) -> float:
        """The focal length in pixels along x."""
        return float(self.k[0, 0])

    def hfov_deg(self) -> float:
        """The horizontal field of view the pinhole implies, for a glance at plausibility."""
        return float(np.degrees(2.0 * np.arctan(self.k[0, 2] / self.k[0, 0])))

    def line(self, eye: str) -> str:
        """One line of numbers for the operator."""
        return (
            f"{eye:>5} {self.model:<9} fx {self.k[0, 0]:7.2f} fy {self.k[1, 1]:7.2f}"
            f" cx {self.k[0, 2]:7.2f} cy {self.k[1, 2]:7.2f}"
            f"  rms {self.rms_px:.3f} px  held-out {self.holdout_px:.3f} px"
        )


def _model_flags(model: str) -> int:
    """The ``calibrateCamera`` flags that select a distortion model by name."""
    import cv2

    if model == "plumb_bob":
        return 0
    if model == "rational":
        return int(cv2.CALIB_RATIONAL_MODEL)
    raise ValueError(f"{model!r}: the model is one of {', '.join(MODELS)}")


def _points(corners_list: Sequence[Corners], board: Board) -> tuple[list[Corners], list[Corners]]:
    """The board's grid repeated per view, and the image points shaped as OpenCV wants them."""
    object_points: list[Corners] = [
        np.asarray(board.object_points(), dtype=np.float32) for _ in corners_list
    ]
    image_points: list[Corners] = [
        np.asarray(np.asarray(c, dtype=np.float32).reshape(-1, 1, 2), dtype=np.float32)
        for c in corners_list
    ]
    return object_points, image_points


def _blank_optics() -> tuple[Array, Array]:
    """The camera matrix and distortion arrays ``calibrateCamera`` wants as OUTPUTS.

    Without ``CALIB_USE_INTRINSIC_GUESS`` whatever goes in is ignored; the arrays exist only
    because the binding's type wants them, and ``None`` is not something its stubs accept.
    """
    return np.zeros((3, 3), dtype=np.float64), np.zeros((1, 5), dtype=np.float64)


def holdout_split(count: int, every: int = HOLDOUT_EVERY) -> tuple[list[int], list[int]]:
    """Which view indices train and which score, as ``(train, test)``.

    A FIXED STRIDE would be the obvious thing and is a trap: a person collecting views moves the
    board in a rhythm — near, far, tilted, tilted the other way — and a stride that happens to
    share that period holds out one whole kind of view and trains on none of it, which scores
    every model as though it generalised badly. The split is therefore a fixed-seed shuffle: as
    reproducible as a stride, with no period to collide with.
    """
    order = np.random.default_rng(HOLDOUT_SEED).permutation(count)
    take = max(1, count // every)
    test = sorted(int(i) for i in order[:take])
    return [i for i in range(count) if i not in set(test)], test


def _reprojection_rms(
    k: Array, dist: Array, corners_list: Sequence[Corners], board: Board
) -> float:
    """RMS reprojection error of a fixed lens on views it was NOT fitted to: each view's pose is
    solved with ``solvePnP`` and only the pixels then disagree. ``nan`` with no views."""
    import cv2

    if not len(corners_list):
        return float("nan")
    grid = board.object_points().astype(np.float32)
    squares: list[float] = []
    for corners in corners_list:
        image = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(grid, image, k, dist)
        if not ok:
            continue
        projected = np.asarray(
            cv2.projectPoints(grid, rvec, tvec, k, dist)[0], dtype=np.float64
        ).reshape(-1, 2)
        squares.extend(
            np.square(projected - image.reshape(-1, 2).astype(np.float64)).sum(axis=1).tolist()
        )
    if not squares:
        return float("nan")
    return float(np.sqrt(np.mean(squares)))


def fit_eye(
    corners_list: Sequence[Corners],
    board: Board,
    size: tuple[int, int],
    model: str = "plumb_bob",
    holdout: bool = True,
) -> EyeFit:
    """Solve one lens from its views under a named distortion model.

    With ``holdout`` the fit is scored on every fourth view, which was kept out of it: a richer
    model always lowers the RMS of the views it was fitted to, and only a held-out view says
    whether it learned the lens or the noise. ``ValueError`` when there are too few views.
    """
    import cv2

    if len(corners_list) < 6:
        raise ValueError(f"{len(corners_list)} views is not a calibration: collect at least 6")
    flags = _model_flags(model)
    object_points, image_points = _points(corners_list, board)
    guess_k, guess_d = _blank_optics()
    rms, k, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points, image_points, size, guess_k, guess_d, flags=flags
    )
    matrix = np.asarray(k, dtype=np.float64)
    coefficients = np.asarray(dist, dtype=np.float64).reshape(-1)
    held = float("nan")
    if holdout and len(corners_list) >= 3 * HOLDOUT_EVERY:
        train_index, test_index = holdout_split(len(corners_list))
        train = [corners_list[i] for i in train_index]
        test = [corners_list[i] for i in test_index]
        train_objects, train_images = _points(train, board)
        blank_k, blank_d = _blank_optics()
        _rms, k_t, dist_t, _rv, _tv = cv2.calibrateCamera(
            train_objects, train_images, size, blank_k, blank_d, flags=flags
        )
        held = _reprojection_rms(
            np.asarray(k_t, dtype=np.float64),
            np.asarray(dist_t, dtype=np.float64).reshape(-1),
            test,
            board,
        )
    return EyeFit(model, matrix, coefficients, float(rms), held, len(corners_list))


def fisheye_rms(corners_list: Sequence[Corners], board: Board, size: tuple[int, int]) -> float:
    """What an equidistant fisheye model would score on these views, as evidence only.

    ``pepin.stereo.Rectifier`` rectifies with the PINHOLE functions, so a fisheye win is not
    something this tool can write — it is something the caller must say out loud. ``nan`` when
    the solver refuses the views, which it does more readily than the pinhole one.
    """
    import cv2

    try:
        grid = [board.object_points().astype(np.float64).reshape(-1, 1, 3) for _ in corners_list]
        image = [np.asarray(c, dtype=np.float64).reshape(-1, 1, 2) for c in corners_list]
        flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_FIX_SKEW
            | cv2.fisheye.CALIB_CHECK_COND
        )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6)
        rms, _k, _d, _rvecs, _tvecs = cv2.fisheye.calibrate(
            grid,
            image,
            size,
            np.zeros((3, 3), dtype=np.float64),
            np.zeros((4, 1), dtype=np.float64),
            flags=flags,
            criteria=criteria,
        )
        return float(rms)
    except cv2.error:
        return float("nan")


def model_evidence(
    corners_list: Sequence[Corners],
    board: Board,
    size: tuple[int, int],
    models: Sequence[str] = MODELS,
) -> dict[str, EyeFit]:
    """Every candidate distortion model fitted to the same views, by name."""
    return {model: fit_eye(corners_list, board, size, model) for model in models}


def choose_model(evidence: dict[str, EyeFit], order: Sequence[str] = MODELS) -> str:
    """The simplest model that is good enough on held-out views.

    A richer model is taken only when it beats the best held-out score both by a fraction
    (:data:`MODEL_GAIN`) and by a visible number of pixels (:data:`MODEL_TIE_PX`); a model that
    merely ties has bought nothing with its extra coefficients, and coefficients nobody measured
    are what makes a calibration explode at the frame's edge.
    """
    scored = {
        name: (fit.holdout_px if np.isfinite(fit.holdout_px) else fit.rms_px)
        for name, fit in evidence.items()
        if name in order
    }
    if not scored:
        raise ValueError("no evidence to choose a model from")
    best = min(scored.values())
    good_enough = max(best * (1.0 + MODEL_GAIN), best + MODEL_TIE_PX)
    for name in order:
        if name in scored and scored[name] <= good_enough:
            return name
    return min(scored, key=lambda name: scored[name])


# ---- the bar between the eyes ------------------------------------------------------------------
@dataclass(frozen=True)
class StereoFit:
    """What the stereo solve measured, and whether it may be written.

    The three numbers that matter are not the same number: the per-eye RMS says each lens is
    understood, the stereo RMS says the pair is rigid and the correspondences are real, and the
    epipolar error says what a block matcher will actually suffer — it is the one that decides.
    """

    left: EyeFit
    right: EyeFit
    rotation: Array
    translation_m: Array
    rms_px: float
    epipolar_px: float
    size: tuple[int, int]
    views: int
    fisheye_px: float = float("nan")

    @property
    def baseline_m(self) -> float:
        """The distance between the two optical centres, metres."""
        return float(np.linalg.norm(self.translation_m))

    def tilt_deg(self) -> float:
        """How far the two eyes are turned from parallel, degrees — the bar's twist."""
        import cv2

        rvec = np.asarray(cv2.Rodrigues(self.rotation)[0], dtype=np.float64).reshape(3)
        return float(np.degrees(np.linalg.norm(rvec)))

    def refusal(self) -> str | None:
        """Why this fit must not be written, or ``None`` when it may be — with what to reshoot."""
        for eye, fit in (("left", self.left), ("right", self.right)):
            if not np.isfinite(fit.rms_px) or fit.rms_px > EYE_RMS_MAX_PX:
                return (
                    f"the {eye} eye fits to {fit.rms_px:.3f} px, above the"
                    f" {EYE_RMS_MAX_PX:.2f} px bound: some views are blurred or the board moved"
                    " while they were taken — reshoot, holding it still"
                )
        if not np.isfinite(self.rms_px) or self.rms_px > STEREO_RMS_MAX_PX:
            return (
                f"the stereo solve is {self.rms_px:.3f} px, above the"
                f" {STEREO_RMS_MAX_PX:.2f} px bound: the two eyes disagree about where the board"
                " was — reshoot with the whole board well inside BOTH eyes"
            )
        if not (BASELINE_MIN_M <= self.baseline_m <= BASELINE_MAX_M):
            return (
                f"the baseline came out {self.baseline_m * 1000:.1f} mm, nowhere near the"
                f" {BASELINE_NOMINAL_M * 1000:.0f} mm bar: the square size passed to --square is"
                " almost certainly not the square on the paper — measure one square and retry"
            )
        if not np.isfinite(self.epipolar_px) or self.epipolar_px > EPIPOLAR_MAX_PX:
            return (
                f"rectified rows still disagree by {self.epipolar_px:.3f} px, above the"
                f" {EPIPOLAR_MAX_PX:.2f} px bound: a block matcher searches along a row, so this"
                " one would match the wrong row — reshoot with more tilted views and views in"
                " the corners of the frame"
            )
        return None

    @property
    def acceptable(self) -> bool:
        """Whether the fit passes every bound and may be written."""
        return self.refusal() is None

    def report(self) -> str:
        """The numbers a person reads after a run."""
        lines = [
            self.left.line("left"),
            self.right.line("right"),
            "  left  distortion " + " ".join(f"{v:+.4f}" for v in self.left.dist),
            "  right distortion " + " ".join(f"{v:+.4f}" for v in self.right.dist),
            f"  fields of view {self.left.hfov_deg():.1f} / {self.right.hfov_deg():.1f} deg"
            " horizontal",
            f"stereo rms {self.rms_px:.3f} px over {self.views} pairs"
            f" (accepted up to {STEREO_RMS_MAX_PX:.2f})",
            f"baseline {self.baseline_m * 1000:.2f} mm"
            f" (nominal {BASELINE_NOMINAL_M * 1000:.0f}), eyes turned {self.tilt_deg():.3f} deg"
            " from parallel",
            f"translation {np.array2string(self.translation_m * 1000.0, precision=2)} mm"
            " (right eye in the left eye's frame)",
            f"epipolar error after rectification {self.epipolar_px:.3f} px"
            f" (good under {EPIPOLAR_GOOD_PX:.1f}, refused over {EPIPOLAR_MAX_PX:.1f})",
        ]
        if np.isfinite(self.fisheye_px):
            lines.append(
                f"a fisheye model would fit the left eye to {self.fisheye_px:.3f} px"
                f" against the pinhole's {self.left.rms_px:.3f}"
            )
        return "\n".join(lines)

    def to_calibration(
        self, board: Board, method: str = "chessboard", date: str | None = None, note: str = ""
    ) -> StereoCalibration:
        """This fit as the file ``config/stereo_calibration.json``, evidence included."""
        evidence = (
            f"{board} | {self.left.model} | stereo rms {self.rms_px:.3f} px,"
            f" epipolar {self.epipolar_px:.3f} px"
        )
        return StereoCalibration(
            width=self.size[0],
            height=self.size[1],
            k_left=tuple(tuple(float(v) for v in row) for row in self.left.k),
            d_left=tuple(float(v) for v in self.left.dist),
            k_right=tuple(tuple(float(v) for v in row) for row in self.right.k),
            d_right=tuple(float(v) for v in self.right.dist),
            rotation=tuple(tuple(float(v) for v in row) for row in self.rotation),
            translation_m=(
                float(self.translation_m[0]),
                float(self.translation_m[1]),
                float(self.translation_m[2]),
            ),
            rms_px=self.rms_px,
            date=date or datetime.now(UTC).strftime("%Y-%m-%d"),
            method=method,
            views=self.views,
            board=f"{evidence}{f' | {note}' if note else ''}",
        )


def rectified_rows(
    k_left: Array,
    d_left: Array,
    k_right: Array,
    d_right: Array,
    rotation: Array,
    translation: Array,
    size: tuple[int, int],
    pairs: Sequence[tuple[Corners, Corners]],
) -> float:
    """Mean |y_left - y_right| of the board's corners after rectification, pixels.

    The corners are carried through the same ``stereoRectify`` transform
    :class:`pepin.stereo.Rectifier` builds its remap tables from (``CALIB_ZERO_DISPARITY``,
    ``alpha`` 0), as points rather than as images: remapping and re-detecting would add the
    detector's own noise to the number being measured.
    """
    import cv2

    if not pairs:
        return float("nan")
    r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        k_left, d_left, k_right, d_right, size,
        rotation, translation.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0,
    )  # fmt: skip
    errors: list[float] = []
    for left, right in pairs:
        a = cv2.undistortPoints(
            np.asarray(left, dtype=np.float64).reshape(-1, 1, 2), k_left, d_left, R=r1, P=p1
        )
        b = cv2.undistortPoints(
            np.asarray(right, dtype=np.float64).reshape(-1, 1, 2), k_right, d_right, R=r2, P=p2
        )
        rows_a = np.asarray(a, dtype=np.float64).reshape(-1, 2)[:, 1]
        rows_b = np.asarray(b, dtype=np.float64).reshape(-1, 2)[:, 1]
        errors.extend(np.abs(rows_a - rows_b).tolist())
    return float(np.mean(errors))


def stereo_fit(
    pairs: Sequence[tuple[Corners, Corners]],
    board: Board,
    size: tuple[int, int],
    left: EyeFit,
    right: EyeFit,
    refine: bool = True,
) -> StereoFit:
    """Solve the bar between the eyes: ``stereoCalibrate`` with the two lenses fixed, then — when
    ``refine`` — the same solve with everything free, started from that answer.

    Two steps because they fail differently: with the intrinsics fixed, a bad pair shows up
    immediately as a large RMS instead of being absorbed into a bent lens; letting the lenses
    move afterwards then buys the last tenth of a pixel. The refinement is kept only if it
    actually lowered the RMS.
    """
    import cv2

    if len(pairs) < 6:
        raise ValueError(f"{len(pairs)} pairs is not a stereo calibration: collect at least 6")
    grid = [board.object_points().astype(np.float32) for _ in pairs]
    left_points = [np.asarray(p[0], dtype=np.float32).reshape(-1, 1, 2) for p in pairs]
    right_points = [np.asarray(p[1], dtype=np.float32).reshape(-1, 1, 2) for p in pairs]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
    fixed = cv2.stereoCalibrate(
        grid, left_points, right_points,
        left.k.copy(), left.dist.copy(), right.k.copy(), right.dist.copy(), size,
        flags=cv2.CALIB_FIX_INTRINSIC, criteria=criteria,
    )  # fmt: skip
    rms = float(fixed[0])
    k_l, d_l, k_r, d_r = left.k, left.dist, right.k, right.dist
    rotation = np.asarray(fixed[5], dtype=np.float64)
    translation = np.asarray(fixed[6], dtype=np.float64).reshape(3)
    if refine:
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | _model_flags(left.model) | _model_flags(right.model)
        joint = cv2.stereoCalibrate(
            grid, left_points, right_points,
            k_l.copy(), d_l.copy(), k_r.copy(), d_r.copy(), size,
            R=rotation.copy(), T=translation.copy().reshape(3, 1),
            flags=flags, criteria=criteria,
        )  # fmt: skip
        if np.isfinite(float(joint[0])) and float(joint[0]) < rms:
            rms = float(joint[0])
            k_l = np.asarray(joint[1], dtype=np.float64)
            d_l = np.asarray(joint[2], dtype=np.float64).reshape(-1)
            k_r = np.asarray(joint[3], dtype=np.float64)
            d_r = np.asarray(joint[4], dtype=np.float64).reshape(-1)
            rotation = np.asarray(joint[5], dtype=np.float64)
            translation = np.asarray(joint[6], dtype=np.float64).reshape(3)
    epipolar = rectified_rows(k_l, d_l, k_r, d_r, rotation, translation, size, pairs)
    return StereoFit(
        left=EyeFit(left.model, k_l, d_l, left.rms_px, left.holdout_px, left.views),
        right=EyeFit(right.model, k_r, d_r, right.rms_px, right.holdout_px, right.views),
        rotation=rotation,
        translation_m=translation,
        rms_px=rms,
        epipolar_px=epipolar,
        size=size,
        views=len(pairs),
    )


def calibrate_pairs(
    pairs: Sequence[tuple[Corners, Corners]],
    board: Board,
    size: tuple[int, int],
    model: str | None = None,
    fisheye: bool = True,
) -> tuple[StereoFit, dict[str, EyeFit], dict[str, EyeFit]]:
    """The whole solve from accepted pairs: choose the distortion model by evidence on THIS lens
    (unless one is named), fit both eyes with it, solve and refine the bar.

    Answers the fit and the per-eye evidence for every model tried, so the runner can print what
    the choice cost.
    """
    left_corners = [p[0] for p in pairs]
    right_corners = [p[1] for p in pairs]
    left_evidence = model_evidence(left_corners, board, size)
    right_evidence = model_evidence(right_corners, board, size)
    chosen = model or choose_model(
        {
            name: EyeFit(
                name,
                left_evidence[name].k,
                left_evidence[name].dist,
                max(left_evidence[name].rms_px, right_evidence[name].rms_px),
                max(left_evidence[name].holdout_px, right_evidence[name].holdout_px),
                left_evidence[name].views,
            )
            for name in left_evidence
        }
    )
    fit = stereo_fit(pairs, board, size, left_evidence[chosen], right_evidence[chosen])
    if fisheye:
        fit = StereoFit(
            fit.left, fit.right, fit.rotation, fit.translation_m, fit.rms_px, fit.epipolar_px,
            fit.size, fit.views, fisheye_rms(left_corners, board, size),
        )  # fmt: skip
    return fit, left_evidence, right_evidence
