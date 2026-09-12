#!/usr/bin/env python
"""Calibrate the neck camera against a printed checkerboard, in one command.

The optics in config/camera.json are a guess: one field-of-view number standing in for four
(fx, fy, cx, cy) and a lens that bends straight lines. A checkerboard measures all of it. This
script does the whole job with no step to remember: it prints the board, pulls the board's own
MJPEG stream, finds the corners, tells the operator where to hold the board next, takes a shot
by itself once the board has been still for a moment, and — only if the fit is good enough —
writes the result into config/camera.json with `calibrated: true`.

    ros/calibrate.sh --print                 the board to print, as a PDF (A4, exact scale)
    ros/calibrate.sh                         calibrate, with a window showing the corners
    ros/calibrate.sh --no-window             the same over ssh: a text coverage report
    ros/calibrate.sh --board 9x6 --square 0.024      another board (inner corners, metres)
    ros/calibrate.sh --images DIR            re-fit from frames a previous run saved

The maths, the coverage rules and the config writer live in pepin.calibration and pepin.camera
and are unit-tested against a synthetic camera; what is here is the stream, the window and the
command line.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pepin.calibration import (
    HOLD_S,
    RMS_MAX_PX,
    TARGET_VIEWS,
    Board,
    Collector,
    Fit,
    Verdict,
    board_pdf,
    calibrate,
    find_corners,
)
from pepin.camera import CameraConfig, write_calibration
from pepin.log import setup_logging

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "config" / "camera.json"
BOARD_PDF = REPO / "data" / "checkerboard.pdf"
FRAMES_DIR = REPO / "data" / "camera_calib"
STREAM_TIMEOUT_S = 5.0
WINDOW = "pepin camera calibration"
GREEN, AMBER, RED, WHITE = (80, 220, 80), (60, 190, 240), (60, 60, 235), (240, 240, 240)


# ---- the frames ------------------------------------------------------------------------------
def stream_frames(url: str) -> Iterator[tuple[Any, float]]:
    """Decoded BGR frames from the board's MJPEG stream, each with the moment it arrived, until
    the stream ends or the caller stops reading. The part reader is pepin.mjpeg — the same one
    the camera node uses, not OpenCV's."""
    from pepin.mjpeg import parts

    with urllib.request.urlopen(url, timeout=STREAM_TIMEOUT_S) as response:
        for _headers, body in parts(response):
            frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame, time.monotonic()


def directory_frames(where: Path) -> Iterator[tuple[Any, float]]:
    """The images of a directory, in name order: a previous run's saved frames, re-fitted
    without the camera. Each is offered four times on a clock of its own, HOLD_S apart: the
    first offer reads as motion (the previous picture), the second starts the hold, and the
    fourth is past it — a board held still in front of the lens, as far as the collector can
    tell."""
    files = sorted(p for p in where.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not files:
        raise SystemExit(f"no images in {where}")
    clock = 0.0
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            logger.warning("could not read %s", path)
            continue
        for _ in range(4):
            yield image, clock
            clock += HOLD_S


# ---- the window ------------------------------------------------------------------------------
def draw_overlay(
    frame: Any, corners: Any, verdict: Verdict, collector: Collector, board: Board
) -> Any:
    """The operator's picture: the detected corners, the 3x3 coverage grid with a count in each
    cell, the line telling them what to do next, and the countdown when a shot is coming."""
    canvas = frame.copy()
    if corners is not None:
        cv2.drawChessboardCorners(canvas, board.size, corners, True)
    height, width = canvas.shape[:2]
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
        f"{len(collector.views)}/{collector.target} views   tilted {coverage.tilted}"
        f"   close {coverage.close}"
    )
    cv2.rectangle(canvas, (0, 0), (width, 64), (20, 20, 20), -1)
    cv2.putText(canvas, banner, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    cv2.putText(canvas, verdict.message, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1)
    if verdict.countdown_s is not None:
        filled = int(width * max(0.0, 1.0 - verdict.countdown_s / HOLD_S))
        cv2.rectangle(canvas, (0, 64), (filled, 72), GREEN, -1)
    if verdict.captured:
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), GREEN, 6)
    return canvas


# ---- the run ---------------------------------------------------------------------------------
def collect(
    frames: Iterator[tuple[Any, float]],
    board: Board,
    size: tuple[int, int],
    args: argparse.Namespace,
) -> Collector:
    """Run the camera until enough well-spread views are in hand (or the operator presses q),
    showing the window unless it was turned off. Frames that are not the configured size are
    refused outright: a calibration of one resolution does not describe another."""
    collector = Collector(board, size, target=args.views)
    saved = None if args.no_save else FRAMES_DIR / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    if saved is not None:
        saved.mkdir(parents=True, exist_ok=True)
        logger.info("keeping every accepted frame in %s", saved)
    last = ""
    for frame, now in frames:
        if (frame.shape[1], frame.shape[0]) != size:
            raise SystemExit(
                f"the stream is {frame.shape[1]}x{frame.shape[0]} and config/camera.json says"
                f" {size[0]}x{size[1]}: calibrate at the camera's own resolution"
            )
        corners = find_corners(frame, board)
        verdict = collector.offer(corners, now)
        if verdict.captured and saved is not None:
            cv2.imwrite(str(saved / f"view_{len(collector.views):02d}.png"), frame)
        if args.no_window:
            if verdict.captured or verdict.message != last:
                logger.info("%s", verdict.message)
                last = verdict.message
        else:
            cv2.imshow(WINDOW, draw_overlay(frame, corners, verdict, collector, board))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                logger.info("stopped by the operator")
                break
        if collector.done():
            logger.info("enough views: %d, covered", len(collector.views))
            break
    if not args.no_window:
        cv2.destroyAllWindows()
    return collector


def verdict_on(collector: Collector, fit: Fit) -> str | None:
    """Why the fit must not be written, or ``None`` when it may be: a poor spread of views or an
    RMS above the bound are both refusals, because either makes numbers that look like a
    measurement and are not."""
    if not collector.coverage.good():
        return f"the views are not a calibration yet — {collector.coverage.hint()}"
    if not fit.acceptable:
        index, error = fit.worst()
        return (
            f"rms {fit.calibration.rms:.3f} px is above the {RMS_MAX_PX:.2f} px bound;"
            f" view #{index} alone is {error:.3f} px — drop it and shoot that place again"
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the neck camera with a checkerboard and write config/camera.json."
    )
    parser.add_argument("--board", default="9x6", help="inner corners across x down (default 9x6)")
    parser.add_argument(
        "--square", type=float, default=0.024, help="square side in metres (default 0.024)"
    )
    parser.add_argument("--host", default=None, help="the board's address (default: the config's)")
    parser.add_argument("--stream", default=None, help="an MJPEG URL, instead of the config's")
    parser.add_argument(
        "--images", type=Path, default=None, help="re-fit from a directory of saved frames"
    )
    parser.add_argument("--config", type=Path, default=CONFIG, help="config/camera.json to update")
    parser.add_argument("--camera", default="overview", help="which camera of the config")
    parser.add_argument("--views", type=int, default=TARGET_VIEWS, help="how many views to collect")
    parser.add_argument("--no-window", action="store_true", help="no OpenCV window: a text report")
    parser.add_argument("--no-save", action="store_true", help="do not keep the accepted frames")
    parser.add_argument("--dry-run", action="store_true", help="fit and report, write nothing")
    parser.add_argument(
        "--print",
        dest="print_to",
        nargs="?",
        const=BOARD_PDF,
        type=Path,
        default=None,
        help=f"write the board to print as a PDF (default {BOARD_PDF}) and stop",
    )
    parser.add_argument("--page", default="a4", choices=("a4", "letter"), help="the paper size")
    args = parser.parse_args()
    setup_logging("calibrate_camera")

    board = Board.parse(args.board, args.square)
    if args.print_to is not None:
        args.print_to.parent.mkdir(parents=True, exist_ok=True)
        args.print_to.write_bytes(board_pdf(board, args.page))
        logger.info("printable board: %s (%s)", args.print_to, board)
        logger.info(
            "print it at 100 percent — not 'fit to page' — then measure one square with a ruler"
        )
        logger.info(
            "if a square is not %.1f mm on paper, pass what it really is: --square 0.0235",
            board.square_m * 1000,
        )
        return

    cfg = CameraConfig.load(args.config, name=args.camera, board=args.host or "10.0.0.187")
    size = (cfg.width, cfg.height)
    logger.info("board: %s", board)
    logger.info("camera: %s at %dx%d", args.stream or cfg.stream, *size)
    if cfg.calibrated:
        logger.warning("this camera is already calibrated; a good run replaces those numbers")
    frames = (
        directory_frames(args.images)
        if args.images is not None
        else stream_frames(args.stream or cfg.stream)
    )
    collector = collect(frames, board, size, args)
    logger.info("\n%s", collector.coverage.report())
    if len(collector.views) < 6:
        raise SystemExit(f"{len(collector.views)} views is not a calibration: nothing written")

    fit = calibrate(collector.views, board, size)
    logger.info("\n%s", fit.report())
    refusal = verdict_on(collector, fit)
    if refusal is not None:
        logger.error("not written: %s", refusal)
        raise SystemExit(1)
    if args.dry_run:
        logger.info("--dry-run: config/camera.json untouched")
        return
    write_calibration(args.config, fit.calibration, args.camera)
    logger.info(
        "wrote %s: calibrated true, hfov_deg %.2f (derived from fx)",
        args.config,
        fit.calibration.hfov_deg(),
    )
    logger.info("restart the camera node to publish it: ros/laptop.sh kick camera_stream")


if __name__ == "__main__":
    sys.exit(main())
