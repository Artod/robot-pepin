#!/usr/bin/env python
"""Measure the stereo head with a printed checkerboard, in one command.

The head is a stereo module that sends ONE side-by-side MJPEG frame; what a calibration must
measure is both lenses AND the bar between them, from the board seen in both eyes of the same
frame. This script does the whole job with no step to remember: it pulls the board's stream,
cuts the eyes, finds the board in both, tells the operator where to hold it next, keeps a pair
on its own as soon as the board is sharp and has moved, and — only if the evidence is good
enough — writes config/stereo_calibration.json.

    ros/calibrate.sh stereo                  calibrate, with a window showing both eyes
    ros/calibrate.sh stereo --no-window      the same over ssh: a text coverage report
    ros/calibrate.sh stereo --images DIR     re-fit from a session a previous run saved
    ros/calibrate.sh stereo --square 0.0245  the square really on the paper, metres

Every accepted pair is saved under data/stereo_calib/<session>/ before anything is solved, so a
run that ends badly can be re-fitted offline instead of reshot. The maths and the rules live in
pepin.stereo_calibration and are unit-tested against a synthetic head; what is here is the
stream, the window and the command line.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pepin.calibration import Board
from pepin.log import setup_logging
from pepin.stereo import CALIBRATION_FILE, SideBySide
from pepin.stereo_calibration import (
    EPIPOLAR_GOOD_PX,
    SHARP_MIN,
    TARGET_VIEWS,
    Corners,
    EyeFit,
    PairCollector,
    StereoFit,
    board_sharpness,
    calibrate_pairs,
    find_pair,
)

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent.parent
CONFIG = REPO / "config" / CALIBRATION_FILE
SESSIONS = REPO / "data" / "stereo_calib"
STREAM = "http://{board}:8080/stream"
STREAM_TIMEOUT_S = 5.0
STREAM_OPEN_TRIES = 6  # the board's WiFi drops a connect now and then
STREAM_RETRY_S = 2.0
WINDOW = "pepin stereo calibration"
GREEN, AMBER, RED, WHITE = (80, 220, 80), (60, 190, 240), (60, 60, 235), (240, 240, 240)


# ---- the frames --------------------------------------------------------------------------------
def stream_eyes(url: str, upside_down: bool) -> Iterator[tuple[Any, Any]]:
    """The two eyes of every frame of the board's MJPEG stream, upright and in the robot's order.

    The part reader is pepin.mjpeg — the same one the camera node uses, not OpenCV's, whose
    reader throws the capture stamp away.
    """
    from pepin.mjpeg import parts

    cutter = SideBySide(upside_down=upside_down)
    response = None
    for attempt in range(1, STREAM_OPEN_TRIES + 1):
        try:
            response = urllib.request.urlopen(url, timeout=STREAM_TIMEOUT_S)
            break
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("stream open %d/%d failed: %s", attempt, STREAM_OPEN_TRIES, exc)
            if attempt == STREAM_OPEN_TRIES:
                raise
            time.sleep(STREAM_RETRY_S)
    assert response is not None
    with response:
        for _headers, body in parts(response):
            frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield cutter.eyes(frame)


def session_eyes(where: Path) -> Iterator[tuple[Any, Any]]:
    """The saved eye pairs of a previous session, in name order: a re-fit without the camera."""
    lefts = sorted(where.glob("pair_*_left.png"))
    if not lefts:
        raise SystemExit(f"no saved pairs in {where} (expected pair_NN_left.png)")
    for left_path in lefts:
        right_path = left_path.with_name(left_path.name.replace("_left.png", "_right.png"))
        left, right = cv2.imread(str(left_path)), cv2.imread(str(right_path))
        if left is None or right is None:
            logger.warning("could not read the pair of %s", left_path.name)
            continue
        yield left, right


# ---- the window --------------------------------------------------------------------------------
def draw_overlay(
    left: Any, right: Any, corners: tuple[Corners, Corners] | None,
    message: str, captured: bool, collector: PairCollector, board: Board,
) -> Any:  # fmt: skip
    """The operator's picture: both eyes side by side with the corners drawn in each, the 3x3
    coverage grid over the left eye with a count per cell, and the line telling them what next."""
    canvas = np.hstack([left.copy(), right.copy()])
    if corners is not None:
        cv2.drawChessboardCorners(canvas[:, : left.shape[1]], board.size, corners[0], True)
        cv2.drawChessboardCorners(canvas[:, left.shape[1] :], board.size, corners[1], True)
    height, width = left.shape[:2]
    coverage = collector.coverage
    for row in range(3):
        for col in range(3):
            x0, y0 = col * width // 3, row * height // 3
            x1, y1 = (col + 1) * width // 3, (row + 1) * height // 3
            count = coverage.cells.get((row, col), 0)
            colour = GREEN if count >= 2 else (AMBER if count else RED)
            cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), colour, 1)
            cv2.putText(
                canvas, str(count), (x0 + 10, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1
            )
    banner = (
        f"{len(collector.pairs)}/{collector.target} pairs   tilted {coverage.tilted}"
        f"   close {coverage.close}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (20, 20, 20), -1)
    cv2.putText(canvas, banner, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    cv2.putText(canvas, message, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1)
    if captured:
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), GREEN, 6)
    return canvas


def window_works() -> bool:
    """Whether this OpenCV can open a window at all — a headless build raises instead, and the
    run must fall back to text rather than die halfway through a collection."""
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        return True
    except cv2.error as exc:
        logger.warning("no window on this OpenCV (%s) — falling back to the text report", exc)
        return False


# ---- the run -----------------------------------------------------------------------------------
def collect(
    eyes: Iterator[tuple[Any, Any]], board: Board, args: argparse.Namespace, session: Path | None
) -> PairCollector:
    """Run the camera until enough well-spread pairs are in hand (or the operator presses q).

    The size is taken from the first frame rather than from a config: the stereo head's eye is
    whatever half of the transport frame is, and a calibration is always written at that size.
    """
    collector: PairCollector | None = None
    windowed = not args.no_window and window_works()
    last = ""
    for left, right in eyes:
        if collector is None:
            size = (left.shape[1], left.shape[0])
            logger.info("eye %dx%d (the transport frame is twice as wide)", *size)
            collector = PairCollector(board, size, target=args.views, sharp_min=args.sharp_min)
        corners = find_pair(left, right, board)
        sharpness = board_sharpness(left, corners[0]) if corners is not None else 0.0
        verdict = collector.offer(corners, sharpness)
        if verdict.captured and session is not None:
            index = len(collector.pairs)
            cv2.imwrite(str(session / f"pair_{index:02d}_left.png"), left)
            cv2.imwrite(str(session / f"pair_{index:02d}_right.png"), right)
        if windowed:
            cv2.imshow(
                WINDOW,
                draw_overlay(
                    left, right, corners, verdict.message, verdict.captured, collector, board
                ),
            )
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                logger.info("stopped by the operator")
                break
        elif verdict.captured or verdict.message != last:
            logger.info("%s", verdict.message)
            last = verdict.message
        if collector.done():
            logger.info("enough pairs: %d, covered", len(collector.pairs))
            break
    if windowed:
        cv2.destroyAllWindows()
    if collector is None:
        raise SystemExit("no frames arrived from the stream")
    return collector


def report_models(fit: StereoFit, left: dict[str, EyeFit], right: dict[str, EyeFit]) -> None:
    """What each distortion model scored on THIS lens, and why the chosen one won."""
    logger.info("distortion models, scored on views held out of their own fit:")
    for name in left:
        logger.info(
            "  %-9s left rms %.3f held-out %.3f | right rms %.3f held-out %.3f%s",
            name,
            left[name].rms_px,
            left[name].holdout_px,
            right[name].rms_px,
            right[name].holdout_px,
            "   <- chosen" if name == fit.left.model else "",
        )
    if np.isfinite(fit.fisheye_px) and fit.fisheye_px < fit.left.rms_px * 0.6:
        logger.warning(
            "A FISHEYE MODEL FITS THIS LENS MUCH BETTER (%.3f px against the pinhole's %.3f)."
            " pepin.stereo.Rectifier rectifies with the PINHOLE functions, so this file cannot"
            " carry a fisheye: stereo.py needs a model field and a fisheye branch before the"
            " head can be believed at the edges of the frame.",
            fit.fisheye_px,
            fit.left.rms_px,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the stereo head with a checkerboard; writes"
        " config/stereo_calibration.json."
    )
    parser.add_argument("--board", default="9x6", help="inner corners across x down (default 9x6)")
    parser.add_argument(
        "--square", type=float, default=0.0245, help="square side in metres (default 0.0245)"
    )
    parser.add_argument("--host", default="10.0.0.187", help="the board's address")
    parser.add_argument("--stream", default=None, help="an MJPEG URL, instead of the board's")
    parser.add_argument(
        "--images", type=Path, default=None, help="re-fit from a session directory of saved pairs"
    )
    parser.add_argument("--config", type=Path, default=CONFIG, help="the file to write")
    parser.add_argument("--views", type=int, default=TARGET_VIEWS, help="how many pairs to collect")
    parser.add_argument(
        "--model", default=None, choices=("plumb_bob", "rational"),
        help="force a distortion model instead of choosing it by evidence",
    )  # fmt: skip
    parser.add_argument(
        "--sharp-min", type=float, default=SHARP_MIN, help="Laplacian variance a board must reach"
    )
    parser.add_argument(
        "--right-way-up", action="store_true",
        help="the module is NOT taped upside down (it is, tonight: the default turns it back)",
    )  # fmt: skip
    parser.add_argument("--no-window", action="store_true", help="no window: a text report")
    parser.add_argument("--no-save", action="store_true", help="do not keep the accepted pairs")
    parser.add_argument("--dry-run", action="store_true", help="fit and report, write nothing")
    args = parser.parse_args()
    setup_logging("stereo_calibrate")

    board = Board.parse(args.board, args.square)
    logger.info("board: %s", board)
    logger.info(
        "the square size is the ONLY length in this calibration: if the printed square is not"
        " %.1f mm, pass what it really is with --square",
        board.square_m * 1000.0,
    )
    session = None
    if not args.no_save and args.images is None:
        session = SESSIONS / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        session.mkdir(parents=True, exist_ok=True)
        logger.info("keeping every accepted pair in %s", session)
    eyes = (
        session_eyes(args.images)
        if args.images is not None
        else stream_eyes(args.stream or STREAM.format(board=args.host), not args.right_way_up)
    )
    collector = collect(eyes, board, args, session)
    logger.info("\n%s", collector.coverage.report())
    if len(collector.pairs) < 6:
        raise SystemExit(f"{len(collector.pairs)} pairs is not a calibration: nothing written")

    pairs = [(p.left, p.right) for p in collector.pairs]
    size = collector.size
    logger.info("solving %d pairs at %dx%d per eye...", len(pairs), *size)
    fit, left_evidence, right_evidence = calibrate_pairs(pairs, board, size, model=args.model)
    report_models(fit, left_evidence, right_evidence)
    logger.info("\n%s", fit.report())
    rows = float(np.mean([p.row_offset_px for p in collector.pairs]))
    logger.info(
        "the raw eyes' rows disagreed by %.2f px before rectification, %.3f px after", rows,
        fit.epipolar_px,
    )  # fmt: skip
    if session is not None:
        (session / "fit.json").write_text(
            json.dumps(fit.to_calibration(board).to_json(), indent=2) + "\n"
        )

    if not collector.coverage.good():
        logger.warning(
            "the views did not cover the frame — %s; the numbers below describe the middle of"
            " the lens only",
            collector.coverage.hint(),
        )
    refusal = fit.refusal()
    if refusal is not None:
        logger.error("NOT WRITTEN: %s", refusal)
        raise SystemExit(1)
    if args.dry_run:
        logger.info("--dry-run: %s untouched", args.config)
        return
    fit.to_calibration(board).write(args.config)
    logger.info(
        "wrote %s: baseline %.2f mm, epipolar %.3f px (%s), %s at %dx%d per eye",
        args.config,
        fit.baseline_m * 1000.0,
        fit.epipolar_px,
        "good" if fit.epipolar_px <= EPIPOLAR_GOOD_PX else "usable",
        fit.left.model,
        *size,
    )
    logger.info(
        "check it against the room before trusting it: scratch/stereo/depth_vs_lidar.py compares"
        " this head's depth with the lidar's ranges and needs no board"
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
