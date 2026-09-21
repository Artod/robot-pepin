"""The stereo calibration on a head whose answer we already know.

A real stereo calibration needs a printed board, two lenses and a person moving it. What can be
tested is everything between: two synthetic pinholes with chosen distortions, a chosen bar
between them, and a board projected through both from thirty poses. The solve must give the
chosen numbers back — the focal length to well under a percent, the baseline to under a
millimetre, and rows that agree after rectification. A bug in the corner order, the object grid
or the translation's sign shows up here as a baseline of metres or an epipolar error of tens of
pixels, which is exactly how those bugs show up on the robot.

Nothing here calls ``findChessboardCornersSB``: it segfaults on this laptop's OpenCV, and a
segfault in one test takes the whole suite with it. The corner detector is not exercised at all
(that needs a rendered board); the maths downstream of it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from pepin.calibration import Board, View, view_shape
from pepin.stereo import Rectifier
from pepin.stereo_calibration import (
    BASELINE_NOMINAL_M,
    EPIPOLAR_MAX_PX,
    TARGET_VIEWS,
    EyeFit,
    PairCollector,
    StereoFit,
    calibrate_pairs,
    choose_model,
    match_order,
    moved_enough,
    row_offset,
    stereo_fit,
)

BOARD = Board(9, 6, 0.0245)
SIZE = (800, 600)
# A 94 deg lens on an 800 px eye: fx = 400 / tan(47 deg) = 373 px, and a barrel that moves the
# frame's corner by tens of pixels.
TRUE_K_LEFT = np.array([[373.0, 0.0, 398.0], [0.0, 374.5, 302.0], [0.0, 0.0, 1.0]])
TRUE_K_RIGHT = np.array([[371.5, 0.0, 403.0], [0.0, 372.0, 299.0], [0.0, 0.0, 1.0]])
TRUE_D_LEFT = np.array([-0.284, 0.082, 0.0004, -0.0006, -0.011])
TRUE_D_RIGHT = np.array([-0.276, 0.076, -0.0003, 0.0005, -0.009])
TRUE_BASELINE_M = 0.0631
TRUE_RVEC = np.array([0.004, -0.010, 0.002])  # the bar's small twist, radians


def _grid_corners(
    centre: tuple[float, float], span: float = 120.0, tilt: float = 0.0
) -> np.ndarray:
    """A fake 9x6 corner cloud around a centre — enough for the pure bookkeeping tests, which
    never look at whether it could be a real board."""
    xs = np.linspace(-span, span, BOARD.cols)
    ys = np.linspace(-span * 0.66 * (1.0 - tilt), span * 0.66, BOARD.rows)
    mesh = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    return (mesh + np.asarray(centre)).astype(np.float32).reshape(-1, 1, 2)


# ---- the corner order ---------------------------------------------------------------------------
def test_two_eyes_that_number_the_board_the_other_way_round_are_turned_back() -> None:
    left = _grid_corners((400.0, 300.0))
    right = left - np.array([[[9.0, 0.0]]], dtype=np.float32)  # a plain disparity
    turned = np.ascontiguousarray(right[::-1])
    fixed, was_turned = match_order(left, turned)
    assert was_turned
    assert row_offset(left, fixed) < 0.01
    kept, was_turned = match_order(left, right)
    assert not was_turned and row_offset(left, kept) < 0.01


def test_row_offset_is_the_mean_vertical_disagreement() -> None:
    left = _grid_corners((400.0, 300.0))
    right = left + np.array([[[-9.0, 2.0]]], dtype=np.float32)
    assert row_offset(left, right) == pytest.approx(2.0, abs=1e-4)


# ---- what gets kept -----------------------------------------------------------------------------
def _view(centre: tuple[float, float], tilt: float = 0.0) -> View:
    corners = _grid_corners(centre, tilt=tilt)
    cell, area, measured_tilt = view_shape(corners, SIZE)
    return View(corners, cell, area, measured_tilt)


def test_a_board_that_has_not_moved_since_the_last_kept_one_is_not_kept_again() -> None:
    first = _view((400.0, 300.0))
    assert moved_enough(first, None, SIZE)
    assert not moved_enough(_view((404.0, 302.0)), first, SIZE)
    assert moved_enough(_view((600.0, 300.0)), first, SIZE)  # across the frame
    assert moved_enough(_grid_view_scaled(first), first, SIZE)  # nearer


def _grid_view_scaled(view: View) -> View:
    """The same board at 70 % of its size: a clear change of distance."""
    corners = _grid_corners((400.0, 300.0), span=84.0)
    cell, area, tilt = view_shape(corners, SIZE)
    return View(corners, cell, area, tilt)


def test_a_blurred_or_half_visible_board_is_refused_with_the_reason() -> None:
    collector = PairCollector(BOARD, SIZE)
    assert "not found" in collector.offer(None, 500.0).message
    left = _grid_corners((400.0, 300.0))
    right = left - np.array([[[9.0, 0.0]]], dtype=np.float32)
    blurred = collector.offer((left, right), 5.0)
    assert not blurred.captured and "blurred" in blurred.message
    half = collector.offer((left[:20], right[:20]), 500.0)
    assert not half.captured and "partly visible" in half.message
    kept = collector.offer((left, right), 500.0)
    assert kept.captured and len(collector.pairs) == 1
    again = collector.offer((left, right), 500.0)
    assert not again.captured and "move the board" in again.message


def test_the_collector_wants_the_corners_of_the_frame_before_it_is_done() -> None:
    collector = PairCollector(BOARD, SIZE, target=TARGET_VIEWS)
    assert not collector.done()
    for x in (150.0, 400.0, 650.0):
        for y in (120.0, 300.0, 480.0):
            left = _grid_corners((x, y), span=100.0)
            collector.offer((left, left - np.array([[[9.0, 0.0]]], dtype=np.float32)), 500.0)
    coverage = collector.coverage
    assert coverage.total == 9  # one per cell of the 3x3 grid, none of them repeats
    assert not coverage.good()  # still short of the second view per cell, and of the quotas
    assert coverage.missing_cells() and "hold the board at" in coverage.hint()


# ---- the refusals -------------------------------------------------------------------------------
def _fit(**changes: float) -> StereoFit:
    """A passing fit with one number pushed out of bounds."""
    eye = EyeFit("plumb_bob", TRUE_K_LEFT, TRUE_D_LEFT, changes.get("eye_rms", 0.25), 0.3, 30)
    baseline = changes.get("baseline_m", BASELINE_NOMINAL_M)
    return StereoFit(
        left=eye,
        right=eye,
        rotation=np.eye(3),
        translation_m=np.array([-baseline, 0.0, 0.0]),
        rms_px=changes.get("rms_px", 0.4),
        epipolar_px=changes.get("epipolar_px", 0.2),
        size=SIZE,
        views=30,
    )


def test_a_good_fit_is_written_and_every_bad_one_says_what_to_reshoot() -> None:
    assert _fit().acceptable and _fit().refusal() is None
    assert "blurred" in str(_fit(eye_rms=1.2).refusal())
    assert "disagree about where the board" in str(_fit(rms_px=2.0).refusal())
    assert "--square" in str(_fit(baseline_m=0.24).refusal())
    assert "along a row" in str(_fit(epipolar_px=EPIPOLAR_MAX_PX + 0.3).refusal())


def test_the_file_it_writes_carries_the_evidence_and_the_baseline() -> None:
    calibration = _fit().to_calibration(BOARD, note="from the unit tier")
    assert calibration.method == "chessboard"
    assert calibration.baseline_m == pytest.approx(BASELINE_NOMINAL_M)
    assert "epipolar" in calibration.board and "24.5 mm" in calibration.board
    assert calibration.width == 800 and calibration.views == 30


def test_the_simplest_model_that_is_good_enough_wins() -> None:
    def fit(name: str, holdout: float) -> EyeFit:
        return EyeFit(name, TRUE_K_LEFT, TRUE_D_LEFT, holdout * 0.9, holdout, 30)

    # rational ties plumb_bob on held-out views: it bought nothing with four more coefficients
    assert (
        choose_model({"plumb_bob": fit("plumb_bob", 0.30), "rational": fit("rational", 0.29)})
        == "plumb_bob"
    )
    # ...and a relative margin alone is not enough when both are already tiny: 0.01 px against
    # 0.02 px is a 50 % "win" that no block matcher could ever feel
    assert (
        choose_model({"plumb_bob": fit("plumb_bob", 0.02), "rational": fit("rational", 0.01)})
        == "plumb_bob"
    )
    # ...and it wins when it really explains the lens better
    assert (
        choose_model({"plumb_bob": fit("plumb_bob", 0.90), "rational": fit("rational", 0.31)})
        == "rational"
    )


# ---- the whole solve, against a head whose answer is known --------------------------------------
def _true_pose() -> tuple[np.ndarray, np.ndarray]:
    """The bar: the right eye's rotation and translation in the left eye's frame."""
    import cv2

    rotation = np.asarray(cv2.Rodrigues(TRUE_RVEC)[0], dtype=np.float64)
    translation = np.array([-TRUE_BASELINE_M, 0.0008, 0.0012])
    return rotation, translation


def _synthetic_pairs(noise_px: float = 0.05) -> list[tuple[np.ndarray, np.ndarray]]:
    """Thirty views of the board through both synthetic eyes: spread over the frame, tilted both
    ways, at three distances — what a careful person collects in five minutes.

    ``cornerSubPix`` finds a corner to a few hundredths of a pixel, never exactly, so the
    corners are dithered by that much from a fixed seed. Without the dither the test is
    degenerate: with no noise to overfit, the richer distortion model always "wins" and the
    model choice never exercises its tie-break.
    """
    import cv2

    rng = np.random.default_rng(20260920)
    rotation, translation = _true_pose()
    grid = BOARD.object_points()
    centre = grid.mean(axis=0)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    shots = (
        (0.42, (0.30, 0.35, 0.05)),
        (0.55, (-0.35, -0.40, -0.10)),
        (0.75, (0.10, -0.15, 0.20)),
        (0.95, (-0.15, 0.20, -0.05)),
    )
    for v_frac in (0.22, 0.5, 0.78):
        for u_frac in (0.22, 0.5, 0.78):
            for z, angles in shots:
                rvec = np.asarray(angles, dtype=np.float64)
                turn = np.asarray(cv2.Rodrigues(rvec)[0], dtype=np.float64)
                ray = np.array(
                    [
                        (u_frac * SIZE[0] - TRUE_K_LEFT[0, 2]) / TRUE_K_LEFT[0, 0],
                        (v_frac * SIZE[1] - TRUE_K_LEFT[1, 2]) / TRUE_K_LEFT[1, 1],
                        1.0,
                    ]
                )
                tvec = (ray * z) - turn @ centre
                left = np.asarray(
                    cv2.projectPoints(grid, rvec, tvec, TRUE_K_LEFT, TRUE_D_LEFT)[0],
                    dtype=np.float32,
                ).reshape(-1, 1, 2)
                right_rvec = np.asarray(cv2.Rodrigues(rotation @ turn)[0], dtype=np.float64)
                right_tvec = rotation @ tvec + translation
                right = np.asarray(
                    cv2.projectPoints(grid, right_rvec, right_tvec, TRUE_K_RIGHT, TRUE_D_RIGHT)[0],
                    dtype=np.float32,
                ).reshape(-1, 1, 2)
                both = np.concatenate([left.reshape(-1, 2), right.reshape(-1, 2)])
                if both.min() < 2.0 or both[:, 0].max() > SIZE[0] - 2:
                    continue
                if both[:, 1].max() > SIZE[1] - 2:
                    continue
                if noise_px > 0.0:
                    left = (left + rng.normal(0.0, noise_px, left.shape)).astype(np.float32)
                    right = (right + rng.normal(0.0, noise_px, right.shape)).astype(np.float32)
                pairs.append((left, right))
    return pairs


def _synthetic_pairs_with_depth() -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    """The same views, with each corner's TRUE distance from the left eye along its optical axis
    — what a perfect stereo depth would report for that pixel."""
    import cv2

    rotation, translation = _true_pose()
    grid = BOARD.object_points()
    pairs = _synthetic_pairs(noise_px=0.0)
    depths: list[np.ndarray] = []
    for left, _right in pairs:
        # recover the pose this view was drawn from: the projection is invertible through PnP,
        # and with no noise it is exact
        ok, rvec, tvec = cv2.solvePnP(
            grid.astype(np.float32),
            left.astype(np.float32).reshape(-1, 1, 2),
            TRUE_K_LEFT,
            TRUE_D_LEFT,
        )
        assert ok
        turn = np.asarray(cv2.Rodrigues(rvec)[0], dtype=np.float64)
        in_camera = (turn @ grid.T).T + np.asarray(tvec, dtype=np.float64).reshape(3)
        depths.append(in_camera[:, 2].astype(np.float32))
    assert rotation.shape == (3, 3) and translation.shape == (3,)
    return pairs, depths


@pytest.mark.slow
def test_the_solve_gives_back_the_head_it_was_shown() -> None:
    pairs = _synthetic_pairs()
    assert len(pairs) >= 20, f"only {len(pairs)} synthetic views landed inside both eyes"
    fit, left_evidence, right_evidence = calibrate_pairs(pairs, BOARD, SIZE, fisheye=False)
    assert fit.left.fx == pytest.approx(TRUE_K_LEFT[0, 0], rel=0.01)
    assert fit.right.fx == pytest.approx(TRUE_K_RIGHT[0, 0], rel=0.01)
    assert fit.left.k[0, 2] == pytest.approx(TRUE_K_LEFT[0, 2], abs=4.0)
    assert fit.baseline_m == pytest.approx(TRUE_BASELINE_M, abs=0.001)
    assert fit.epipolar_px < 0.2
    assert fit.rms_px < 0.5
    assert fit.acceptable, fit.refusal()
    # the noiseless board is explained by plumb_bob, which is the model it was drawn with
    assert "plumb_bob" in left_evidence and "rational" in right_evidence
    assert fit.left.model == "plumb_bob"


@pytest.mark.slow
def test_the_rectifier_built_from_the_result_lines_the_rows_up() -> None:
    """The whole chain this head exists for: a calibration, rectified rows, a disparity, metres.

    Each step can be right on its own and wrong together — a sign slipped in the translation
    rectifies beautifully and then reports every distance as its own reciprocal. So the test
    takes corners whose TRUE distance from the left eye is known, carries them through the
    published Rectifier's own pinhole, and asks for the metres back.
    """
    import cv2

    pairs, depths = _synthetic_pairs_with_depth()
    fit, _left, _right = calibrate_pairs(pairs, BOARD, SIZE, model="plumb_bob", fisheye=False)
    rectifier = Rectifier.from_calibration(fit.to_calibration(BOARD))
    assert rectifier.baseline_m == pytest.approx(TRUE_BASELINE_M, abs=0.001)

    r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        fit.left.k, fit.left.dist, fit.right.k, fit.right.dist, SIZE,
        fit.rotation, fit.translation_m.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0,
    )  # fmt: skip
    seen: list[float] = []
    for (left, right), true_depth in zip(pairs, depths, strict=True):
        a = np.asarray(
            cv2.undistortPoints(
                left.astype(np.float64).reshape(-1, 1, 2), fit.left.k, fit.left.dist, R=r1, P=p1
            )
        ).reshape(-1, 2)
        b = np.asarray(
            cv2.undistortPoints(
                right.astype(np.float64).reshape(-1, 1, 2), fit.right.k, fit.right.dist,
                R=r2, P=p2,
            )
        ).reshape(-1, 2)  # fmt: skip
        disparity = (a[:, 0] - b[:, 0]).astype(np.float32)
        assert (disparity > 0).all(), "the right eye must see every corner further left"
        metres = rectifier.depth_m(disparity)
        # the rectified frame is the left eye turned by R1, so depths shift a little; a percent
        # is far tighter than anything a block matcher will deliver
        seen.extend((metres / true_depth).tolist())
    ratio = float(np.median(seen))
    assert ratio == pytest.approx(1.0, abs=0.01), f"stereo depth reads {ratio:.4f} of the truth"


@pytest.mark.slow
def test_a_pair_whose_right_eye_is_numbered_backwards_is_caught_before_the_solve() -> None:
    """The failure this module exists to prevent: one eye's corner list turned by 180 degrees
    fits a bar that is not there. match_order must put it back, and the solve must then be the
    same solve."""
    pairs = _synthetic_pairs()
    broken = [(left, np.ascontiguousarray(right[::-1])) for left, right in pairs]
    repaired = [(left, match_order(left, right)[0]) for left, right in broken]
    assert all(match_order(left, right)[1] for left, right in broken)
    good = calibrate_pairs(pairs, BOARD, SIZE, model="plumb_bob", fisheye=False)[0]
    fixed = calibrate_pairs(repaired, BOARD, SIZE, model="plumb_bob", fisheye=False)[0]
    assert fixed.baseline_m == pytest.approx(good.baseline_m, abs=1e-6)
    # ...and what it would have cost: the raw turned pairs fit a bar nowhere near 63 mm
    left_fit = EyeFit("plumb_bob", good.left.k, good.left.dist, 0.2, 0.2, len(pairs))
    right_fit = EyeFit("plumb_bob", good.right.k, good.right.dist, 0.2, 0.2, len(pairs))
    wrong = stereo_fit(broken, BOARD, SIZE, left_fit, right_fit, refine=False)
    assert abs(wrong.baseline_m - TRUE_BASELINE_M) > 0.05
    assert wrong.refusal() is not None
