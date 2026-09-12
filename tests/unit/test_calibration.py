"""The checkerboard calibration on a camera we already know the answer for.

A real calibration cannot be tested — it needs a printed board, a lens and a person moving it.
What can be tested is everything between: a synthetic camera with a chosen K and a chosen
distortion projects the board from thirty poses (``cv2.projectPoints``, the same model
``calibrateCamera`` inverts), the collector decides what to keep exactly as it would in front of
the real one, and the solve must give the chosen numbers back. It does, to well under a percent;
a bug in the object-point grid, the corner order or the image size shows up here as a focal
length off by the board's aspect or by a square.

The live stream, the OpenCV window and the countdown a person sees are not exercised here: they
need a camera and a screen. The detector itself (``find_corners``) is exercised on a rendered
board, which is as close as this tier gets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from camera_configs import camera_config

from pepin.calibration import (
    CELL_VIEWS,
    CLOSE_VIEWS,
    GRID,
    HOLD_S,
    TILTED_VIEWS,
    Board,
    Collector,
    Coverage,
    View,
    board_pdf,
    calibrate,
    find_corners,
    undistort_optics,
    view_shape,
)
from pepin.camera import Calibration, CameraConfig, optics, write_calibration

REPO = Path(__file__).resolve().parents[2]
BOARD = Board(9, 6, 0.024)
SIZE = (1280, 720)
TRUE_K = np.array([[905.0, 0.0, 648.0], [0.0, 902.0, 352.0], [0.0, 0.0, 1.0]], dtype=np.float64)
TRUE_DIST = np.array([-0.32, 0.11, 0.0006, -0.0004, 0.0], dtype=np.float64)


def project(rvec: np.ndarray, tvec: np.ndarray, dist: np.ndarray = TRUE_DIST) -> np.ndarray:
    """The board seen from one pose by the synthetic camera, as (n, 1, 2) corners."""
    points, _ = cv2.projectPoints(BOARD.object_points(), rvec, tvec, TRUE_K, dist)
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


def pose_for(u: float, v: float, z: float, yaw: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    """A translation putting the board's centre at pixel (u, v) at distance z, for a board
    turned by (yaw, pitch): the ray through that pixel, less where the rotation carried the
    board's own centre."""
    rvec = np.array([pitch, yaw, 0.0], dtype=np.float64)
    rotation, _ = cv2.Rodrigues(rvec)
    centre = BOARD.object_points().mean(axis=0)
    ray = np.array([(u - TRUE_K[0, 2]) / TRUE_K[0, 0], (v - TRUE_K[1, 2]) / TRUE_K[1, 1], 1.0])
    return (ray * z) - rotation @ centre


# Three shots per cell of the frame, the spread a person is told to collect: one close and
# nearly square to the lens, two turned either way at arm's length. (z metres, rvec radians.)
SHOTS = (
    (0.38, (0.15, -0.15, 0.10)),
    (0.60, (0.25, 0.65, -0.10)),
    (0.60, (-0.30, -0.65, 0.15)),
)
# The board's centre is put a quarter, a half and three quarters across, which lands it in the
# outer thirds without pushing its edges off the frame.
PLACES = (0.25, 0.5, 0.75)


def synthetic_views(dist: np.ndarray = TRUE_DIST) -> list[np.ndarray]:
    """Twenty-seven views of the board spread over the frame — three per grid cell, close and
    tilted — as a careful person collects in a minute. A view whose corners fall outside the
    frame is dropped, the same thing the detector would do."""
    width, height = SIZE
    views: list[np.ndarray] = []
    for v_frac in PLACES:
        for u_frac in PLACES:
            for z, angles in SHOTS:
                rvec = np.array(angles, dtype=np.float64)
                rotation, _ = cv2.Rodrigues(rvec)
                centre = BOARD.object_points().mean(axis=0)
                ray = np.array(
                    [
                        (u_frac * width - TRUE_K[0, 2]) / TRUE_K[0, 0],
                        (v_frac * height - TRUE_K[1, 2]) / TRUE_K[1, 1],
                        1.0,
                    ]
                )
                corners = project(rvec, (ray * z) - rotation @ centre, dist)
                flat = corners.reshape(-1, 2)
                inside = (
                    flat.min() >= 2.0
                    and flat[:, 0].max() <= width - 2
                    and flat[:, 1].max() <= height - 2
                )
                if inside:
                    views.append(corners)
    return views


# ---- the board -------------------------------------------------------------------------------
def test_the_board_spec_is_inner_corners_and_a_square_in_metres() -> None:
    board = Board.parse("9x6", 0.024)
    assert board.size == (9, 6) and board.corners == 54
    grid = board.object_points()
    assert grid.shape == (54, 3) and np.allclose(grid[:, 2], 0.0)
    # OpenCV's order: across first, so the first row spans the 8 gaps of 9 corners
    assert grid[1, 0] == pytest.approx(0.024) and grid[9, 1] == pytest.approx(0.024)
    assert grid[:, 0].max() == pytest.approx(0.024 * 8)


@pytest.mark.parametrize(
    ("spec", "square", "why"),
    [
        ("9", 0.024, "COLSxROWS"),
        ("2x6", 0.024, "3x3"),
        ("7x7", 0.024, "orientation"),
        ("9x6", 0.0, "positive"),
    ],
)
def test_a_board_that_cannot_be_calibrated_is_refused_with_the_reason(
    spec: str, square: float, why: str
) -> None:
    with pytest.raises(ValueError, match=why):
        Board.parse(spec, square)


# ---- the collector ---------------------------------------------------------------------------
def test_a_moving_board_is_never_kept_and_a_still_one_is_counted_down() -> None:
    """The blur of a moving board is the biggest cause of a bad fit, so motion resets the hold;
    a board held still is taken after HOLD_S, with the remaining time to show."""
    collector = Collector(BOARD, SIZE)
    still = project(np.array([0.0, 0.3, 0.0]), pose_for(640, 360, 0.6, 0.3, 0.0))
    moved = still + np.float32(9.0)
    assert not collector.offer(moved, 0.0).captured
    verdict = collector.offer(still, 0.1)
    assert not verdict.captured and verdict.countdown_s is None  # it had just moved 9 px
    holding = collector.offer(still, 0.2)  # now two frames agree: the countdown starts
    assert not holding.captured and holding.countdown_s == pytest.approx(HOLD_S, abs=0.01)
    assert not collector.offer(moved, 0.3).captured  # it moved: the countdown starts over
    collector.offer(still, 0.4)
    collector.offer(still, 0.5)
    assert collector.offer(still, 0.5 + HOLD_S).captured
    assert len(collector.views) == 1


def test_a_frame_without_the_whole_board_says_so() -> None:
    collector = Collector(BOARD, SIZE)
    assert collector.offer(None, 0.0).message.startswith("board not found")
    half = project(np.array([0.0, 0.0, 0.0]), pose_for(640, 360, 0.6, 0.0, 0.0))[:20]
    assert not collector.offer(half, 0.1).captured


def test_a_board_across_the_room_is_refused_as_too_far() -> None:
    collector = Collector(BOARD, SIZE)
    far = project(np.array([0.0, 0.0, 0.0]), pose_for(640, 360, 6.0, 0.0, 0.0))
    assert "too far" in collector.offer(far, 0.0).message


# ---- the coverage ----------------------------------------------------------------------------
def test_coverage_counts_cells_tilt_and_closeness_and_names_what_is_missing() -> None:
    """A board only ever in the middle is the classic bad calibration: the quotas refuse it and
    the hint says where to put it next."""
    middle = [View(np.zeros((1, 1, 2), np.float32), (1, 1), 0.2, 0.4) for _ in range(30)]
    coverage = Coverage.of(middle)
    assert not coverage.good() and len(coverage.missing_cells()) == GRID * GRID - 1
    assert "top-left" in coverage.hint()
    assert "coverage of the frame" in coverage.report() and "next:" in coverage.report()
    full = [
        View(np.zeros((1, 1, 2), np.float32), (r, c), 0.2, 0.4)
        for r in range(GRID)
        for c in range(GRID)
        for _ in range(CELL_VIEWS + 1)
    ]
    good = Coverage.of(full)
    assert good.tilted >= TILTED_VIEWS and good.close >= CLOSE_VIEWS and good.good()


def test_the_run_s_own_view_count_is_the_bar_not_the_module_s_default() -> None:
    """``--views 18`` must lower the bar it is judged against, or a short run collects what it
    was asked for and is then refused for not having collected enough (found end to end,
    scratch/calib_runner_end_to_end.py)."""
    views = [
        View(np.zeros((1, 1, 2), np.float32), (r, c), 0.2, 0.4)
        for r in range(GRID)
        for c in range(GRID)
        for _ in range(CELL_VIEWS)
    ]
    assert len(views) == 18
    assert Coverage.of(views, target=18).good()
    short = Coverage.of(views, target=40)
    assert not short.good() and "18/40" in short.hint() and "18/40" in short.report()
    collector = Collector(BOARD, SIZE, target=18)
    collector.views.extend(views)
    assert collector.done() and collector.coverage.target == 18


def test_a_tilted_board_reads_as_tilted_and_a_square_one_does_not() -> None:
    flat = project(np.array([0.0, 0.0, 0.0]), pose_for(640, 360, 0.6, 0.0, 0.0))
    tilted = project(np.array([0.0, 0.6, 0.0]), pose_for(640, 360, 0.6, 0.6, 0.0))
    _cell, _area, flat_tilt = view_shape(flat, SIZE)
    _cell, _area, tilted_tilt = view_shape(tilted, SIZE)
    assert flat_tilt < 0.05 < tilted_tilt


def test_the_cell_is_where_the_board_sits_in_the_frame() -> None:
    corners = project(np.array([0.0, 0.0, 0.0]), pose_for(200, 120, 0.9, 0.0, 0.0))
    cell, area, _tilt = view_shape(corners, SIZE)
    assert cell == (0, 0) and 0.0 < area < 1.0


# ---- the solve -------------------------------------------------------------------------------
def test_the_solve_gives_the_synthetic_camera_s_own_numbers_back() -> None:
    """The whole point: a known K and a known distortion, thirty poses through cv2.projectPoints,
    and the fit must return them to well under a percent with a near-zero RMS."""
    views = synthetic_views()
    assert len(views) >= 20, "the synthetic spread fell out of frame"
    fit = calibrate(views, BOARD, SIZE)
    c = fit.calibration
    assert c.fx == pytest.approx(TRUE_K[0, 0], rel=0.01)
    assert c.fy == pytest.approx(TRUE_K[1, 1], rel=0.01)
    assert c.cx == pytest.approx(TRUE_K[0, 2], rel=0.01)
    assert c.cy == pytest.approx(TRUE_K[1, 2], rel=0.01)
    assert np.allclose(c.dist[:2], TRUE_DIST[:2], atol=0.02)
    assert c.rms < 0.01 and fit.acceptable  # noiseless corners: the model explains them exactly
    assert (c.width, c.height) == SIZE and c.views == len(views)
    assert c.hfov_deg() == pytest.approx(
        math.degrees(2 * math.atan(SIZE[0] / (2 * TRUE_K[0, 0]))), rel=0.01
    )
    index, worst = fit.worst()
    assert 0 <= index < len(views) and worst < 0.05
    assert "rms" in fit.report() and "field of view" in fit.report()


def test_the_collector_s_own_views_solve_to_the_same_camera() -> None:
    """Not the raw poses but what the collector accepted: the stillness and coverage logic must
    not throw away the spread the solve needs."""
    collector = Collector(BOARD, SIZE, target=18)
    now = 0.0
    for corners in synthetic_views():
        for _ in range(20):  # two seconds of frames at 10 Hz, the board held still
            now += 0.1
            if collector.offer(corners, now).captured:
                break
    assert len(collector.views) >= 18, collector.coverage.report()
    assert collector.coverage.good() and collector.done()
    fit = calibrate(collector.views, BOARD, SIZE)
    assert fit.calibration.fx == pytest.approx(TRUE_K[0, 0], rel=0.01)
    assert fit.acceptable


def test_too_few_views_is_refused_rather_than_fitted() -> None:
    with pytest.raises(ValueError, match="at least 6"):
        calibrate(synthetic_views()[:3], BOARD, SIZE)


def test_a_bad_fit_is_not_acceptable_and_names_its_worst_view() -> None:
    """One view whose corners are scattered by a few pixels — a blurred shot — carries the RMS
    past the bound, and the fit points at it instead of being written."""
    rng = np.random.default_rng(7)
    views = synthetic_views()
    views[4] = views[4] + rng.normal(0.0, 6.0, views[4].shape).astype(np.float32)
    fit = calibrate(views, BOARD, SIZE)
    assert not fit.acceptable and fit.worst()[0] == 4


# ---- the detector on a rendered board --------------------------------------------------------
def render_board(corners: np.ndarray) -> np.ndarray:
    """A picture of the board as those corners describe it: the ideal board warped onto the
    frame by the homography its four outer corners imply. Distortion-free by construction, so
    it is the detector that is under test here, not the lens model."""
    side = 60
    ideal = np.full(((BOARD.rows + 1) * side, (BOARD.cols + 1) * side), 255, dtype=np.uint8)
    for row in range(BOARD.rows + 1):
        for col in range(BOARD.cols + 1):
            if (row + col) % 2 == 0:
                ideal[row * side : (row + 1) * side, col * side : (col + 1) * side] = 0
    flat = corners.reshape(-1, 2)
    source = np.array(
        [
            [side, side],
            [BOARD.cols * side, side],
            [side, BOARD.rows * side],
            [BOARD.cols * side, BOARD.rows * side],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [flat[0], flat[BOARD.cols - 1], flat[-BOARD.cols], flat[-1]], dtype=np.float32
    )
    homography = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(ideal, homography, SIZE, borderValue=255)


def test_the_detector_finds_a_rendered_board_where_it_was_drawn() -> None:
    corners = project(np.array([0.0, 0.0, 0.0]), pose_for(640, 360, 0.5, 0.0, 0.0), np.zeros(5))
    found = find_corners(render_board(corners), BOARD)
    assert found is not None and len(found) == BOARD.corners
    # the detector may run the board the other way round; either order lands on the same corners
    a, b = np.sort(found.reshape(-1, 2), axis=0), np.sort(corners.reshape(-1, 2), axis=0)
    assert np.abs(a - b).max() < 2.0


def test_the_detector_says_nothing_when_there_is_no_board() -> None:
    assert find_corners(np.full((720, 1280), 128, dtype=np.uint8), BOARD) is None


# ---- the config round trip -------------------------------------------------------------------
def test_a_calibration_survives_the_config_and_comes_back_as_the_optics(tmp_path: Path) -> None:
    """Write the fit into a copy of config/camera.json and read it back through the one reader:
    calibrated is on, the intrinsics are the fit's, and the block carries the field of view the
    measured fx implies."""
    config = Path(camera_config(tmp_path))
    nominal_hfov = CameraConfig.load(config).hfov_deg
    fit = calibrate(synthetic_views(), BOARD, SIZE)
    write_calibration(config, fit.calibration)
    data = json.loads(config.read_text())["overview"]
    assert data["calibrated"] is True
    assert data["mount"]["z_m"] == 1.23  # every other key of the file survives the write
    assert data["hfov_deg"] == nominal_hfov, "the nominal field of view is not rewritten"
    assert data["intrinsics"]["hfov_deg"] == pytest.approx(fit.calibration.hfov_deg(), abs=0.01)
    back = Calibration.from_json(data["intrinsics"])
    assert back.fx == pytest.approx(fit.calibration.fx, abs=0.001)
    assert back.dist == pytest.approx(fit.calibration.dist, abs=1e-6)
    assert back.board == str(BOARD) and back.views == fit.calibration.views
    cfg = CameraConfig.load(config)
    assert cfg.calibrated and cfg.calibration is not None
    at_full = optics(cfg, 1280, 720)
    assert at_full.calibrated and at_full.fx == pytest.approx(fit.calibration.fx, abs=0.001)
    assert "calibrated" in at_full.source and "rms" in at_full.source


def test_the_optics_scale_to_the_published_size_and_fall_back_to_the_nominal(
    tmp_path: Path,
) -> None:
    """Half the pixels, half the focal length and half the principal point; the distortion
    coefficients act on normalised coordinates and do not move. With no calibration in the file
    the same function answers the nominal pinhole and says so."""
    calibration = Calibration(
        900.0,
        898.0,
        644.0,
        356.0,
        1280,
        720,
        (-0.3, 0.1, 0.0, 0.0, 0.0),
        0.31,
        "2026-09-12",
        str(BOARD),
        views=25,
    )
    half = calibration.scaled(640, 360)
    assert (half.fx, half.cy) == (450.0, 178.0) and half.dist == calibration.dist
    assert half.hfov_deg() == pytest.approx(calibration.hfov_deg(), abs=1e-9)
    cfg = CameraConfig.load(camera_config(tmp_path))
    nominal = optics(cfg, 640, 360)
    assert not nominal.calibrated and nominal.dist == ()
    assert nominal.hfov_deg == pytest.approx(cfg.hfov_deg, abs=1e-9)
    assert nominal.cx == 320.0 and "uncalibrated" in nominal.source
    k, d, r, p = nominal.camera_info_arrays()
    assert k[0] == nominal.fx and d == [0.0] * 5 and r[0] == 1.0 and p[0] == nominal.fx


def test_an_intrinsics_block_left_in_the_file_but_switched_off_is_not_used(
    tmp_path: Path,
) -> None:
    """``calibrated: false`` beside a full intrinsics block is history, not optics: the reader
    answers the nominal pinhole, so a bad calibration is turned off by one boolean."""
    block = Calibration(
        900.0, 900.0, 640.0, 360.0, 1280, 720, (0.0,) * 5, 0.3, "2026-09-12", str(BOARD)
    ).to_json()
    cfg = CameraConfig.load(camera_config(tmp_path, block, calibrated=False))
    assert cfg.calibration is None and not optics(cfg, 1280, 720).calibrated


def test_switching_a_written_calibration_off_gives_the_nominal_optics_back(
    tmp_path: Path,
) -> None:
    """The boolean is a two-way switch: after a real write, `calibrated: false` answers exactly
    the pinhole the camera had before the checkerboard — the nominal field of view survives the
    write, so turning a bad calibration off is not a different camera again."""
    config = Path(camera_config(tmp_path))
    before = optics(CameraConfig.load(config), 640, 360)
    write_calibration(config, calibrate(synthetic_views(), BOARD, SIZE).calibration)
    data = json.loads(config.read_text())
    data["overview"]["calibrated"] = False
    config.write_text(json.dumps(data))
    after = optics(CameraConfig.load(config), 640, 360)
    assert after == before and not after.calibrated


# ---- rectification and the printable board ----------------------------------------------------
def test_the_rectified_optics_keep_the_picture_and_drop_the_distortion() -> None:
    """What camera_stream needs to publish a straightened picture: the K the raw image is bent
    by, the coefficients, and the K of the result — which has no distortion left in it."""
    fit = calibrate(synthetic_views(), BOARD, SIZE)
    k, d, new_k = undistort_optics(fit.calibration, 640, 360)
    assert k[0, 0] == pytest.approx(fit.calibration.fx / 2, abs=0.01)
    assert d.shape[0] >= 4 and np.allclose(d, fit.calibration.dist)
    # a barrel-distorted image straightened at alpha 0 is cropped, so the new focal length grows
    assert new_k[0, 0] > 0.0 and 0.0 < new_k[0, 2] < 640.0


def test_the_printable_board_is_a_pdf_at_the_size_it_claims() -> None:
    pdf = board_pdf(Board(9, 6, 0.024))
    assert pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF")
    assert b"/MediaBox [0 0 841.89 595.28]" in pdf  # landscape A4: the board is wider than tall
    assert b"24.0 mm squares" in pdf and b"100%" in pdf
    squares = pdf.count(b" re f")
    assert squares == (10 * 7) // 2  # half the squares of the 10x7 sheet are inked
    side = 0.024 * 1000 / 25.4 * 72
    assert f"{side:.3f} {side:.3f} re f".encode() in pdf


def test_a_board_too_big_for_the_paper_is_refused_before_it_is_printed() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        board_pdf(Board(9, 6, 0.06))
    with pytest.raises(ValueError, match="page is one of"):
        board_pdf(Board(9, 6, 0.02), page="a3")
