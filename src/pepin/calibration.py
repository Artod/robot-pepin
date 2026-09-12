"""Measuring a lens with a checkerboard: what to collect, when to accept it, what it answers.

A pinhole model guessed from a field of view puts the image's edges degrees wrong — on this
camera the AC310's nominal 70 deg was 8 deg off at the border, and the 78 deg the lidar fitted
is one number standing in for four (fx, fy, cx, cy) plus a lens that bends straight lines. A
checkerboard fixes all of it at once: the board's corners are known in metres, so every view is
an equation, and thirty of them from different places and angles pin the focal lengths, the
principal point and the radial-tangential distortion.

The hard part is not the solve, it is the collection. A calibration shot from one distance
straight ahead fits a focal length that explains nothing at the edges, and OpenCV will report a
beautiful RMS for it. So this module is mostly about coverage: :class:`Board` says what is being
looked at, :class:`Collector` accepts a view only when the board stood still and lands somewhere
the set is still thin, :class:`Coverage` says in words what is missing (which third of the frame,
how many tilted views, how many close ones) and refuses to call a set good until it is not. Then
:func:`calibrate` runs ``cv2.calibrateCamera`` and answers a :class:`pepin.camera.Calibration`
with the RMS and the per-view errors; :func:`board_pdf` draws the board to print, at exact scale.

Nothing here touches a camera or a window: the runner (``scripts/calibrate_camera.py``, wrapped
by ``ros/calibrate.sh``) pulls the frames and draws, and hands corners in here. OpenCV is
imported lazily inside the functions that need it, so importing this module stays free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.camera import Calibration

Array = npt.NDArray[np.float64]
Corners = npt.NDArray[np.float32]  # (n, 1, 2) image points, the shape OpenCV hands back

# What a set of views must have before it is called covered. The frame is cut into a 3x3 grid
# by where the board's centre lands; each cell wants CELL_VIEWS views, because the distortion
# at a corner of the image is only seen by a board that was at that corner.
GRID = 3
CELL_VIEWS = 2
TILTED_VIEWS = 6  # views at an angle: fronto-parallel ones alone leave fx and the distance traded
CLOSE_VIEWS = 3  # views filling a good part of the frame: they carry the distortion's far terms
TARGET_VIEWS = 25

TILT_MIN = 0.15  # how squashed a board must look to count as tilted: about 30 deg of turn
CLOSE_AREA = 0.12  # the fraction of the frame a board must cover to count as close
FAR_AREA = 0.01  # smaller than this and the corners are too few pixels apart to be worth it
STILL_PX = 1.5  # mean corner motion between frames, at 1280 px wide, that still counts as still
HOLD_S = 0.8  # how long it must stay still before the shot is taken (the countdown)
RMS_MAX_PX = 0.5  # a fit worse than this is not written to the config

CELL_NAMES = (
    ("top-left", "top-centre", "top-right"),
    ("centre-left", "the centre", "centre-right"),
    ("bottom-left", "bottom-centre", "bottom-right"),
)


@dataclass(frozen=True)
class Board:
    """The checkerboard being looked at: how many inner corners it has across and down, and how
    long one square's side is in metres. ``9x6`` counts corners, not squares — the printed sheet
    has one more square in each direction."""

    cols: int
    rows: int
    square_m: float

    @classmethod
    def parse(cls, spec: str, square_m: float) -> Board:
        """From the command line's ``9x6`` and a square size in metres; ``ValueError`` with the
        reason when the spec is not two numbers or the board is degenerate."""
        parts = spec.lower().split("x")
        if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
            raise ValueError(f"{spec!r} is not a board: write it as COLSxROWS, e.g. 9x6")
        cols, rows = (int(p) for p in parts)
        if cols < 3 or rows < 3:
            raise ValueError(f"{spec}: a board needs at least 3x3 inner corners")
        if cols == rows:
            raise ValueError(f"{spec}: a square board has no unambiguous orientation; use 9x6")
        if square_m <= 0.0:
            raise ValueError("the square size is in metres and must be positive")
        return cls(cols, rows, square_m)

    @property
    def size(self) -> tuple[int, int]:
        """``(cols, rows)`` as OpenCV's ``patternSize`` wants it."""
        return self.cols, self.rows

    @property
    def corners(self) -> int:
        """How many inner corners one view of the board yields."""
        return self.cols * self.rows

    def object_points(self) -> Array:
        """The board's corners in its own frame, metres, ``z = 0``: an ``(n, 3)`` grid in
        OpenCV's order (across first, then down)."""
        grid = np.zeros((self.corners, 3), dtype=np.float64)
        mesh = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        grid[:, :2] = mesh * self.square_m
        return grid

    def __str__(self) -> str:
        return f"{self.cols}x{self.rows} inner corners, {self.square_m * 1000:.1f} mm squares"


def view_shape(corners: Corners, size: tuple[int, int]) -> tuple[tuple[int, int], float, float]:
    """Where and how a detected board sits in a ``(width, height)`` frame: the grid cell its
    centre falls in, the fraction of the frame its bounding box covers, and how tilted it looks
    (0 for a board square to the lens, towards 1 as perspective squashes one side).

    The tilt is read off the quad of the four outer corners: opposite sides of a
    fronto-parallel board are equally long, and perspective makes the near one longer.
    """
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    width, height = size
    centre = points.mean(axis=0)
    cell = (
        min(GRID - 1, max(0, int(centre[1] / height * GRID))),
        min(GRID - 1, max(0, int(centre[0] / width * GRID))),
    )
    span = points.max(axis=0) - points.min(axis=0)
    area = float(span[0] * span[1]) / float(width * height)
    return cell, area, _quad_tilt(points)


def _quad_tilt(points: Array) -> float:
    """How far from fronto-parallel a board looks, from the corner cloud alone: the largest
    relative difference between the two pairs of opposite sides of its bounding quad."""
    hull = points[np.argsort(points[:, 0] + points[:, 1])]
    other = points[np.argsort(points[:, 0] - points[:, 1])]
    a, c = hull[0], hull[-1]  # top-left-most and bottom-right-most
    b, d = other[-1], other[0]  # top-right-most and bottom-left-most
    sides = [
        float(np.linalg.norm(b - a)),
        float(np.linalg.norm(c - b)),
        float(np.linalg.norm(d - c)),
        float(np.linalg.norm(a - d)),
    ]
    pairs = [(sides[0], sides[2]), (sides[1], sides[3])]
    tilts = [abs(p - q) / max(p, q, 1e-9) for p, q in pairs]
    return float(min(1.0, max(tilts)))


@dataclass(frozen=True)
class View:
    """One accepted picture of the board: its corners, and the three facts the coverage counts
    it by — the frame cell it sat in, how much of the frame it covered, how tilted it looked."""

    corners: Corners
    cell: tuple[int, int]
    area: float
    tilt: float


@dataclass(frozen=True)
class Coverage:
    """What a set of views has and has not seen yet, and how to say it to a person.

    Three quotas, because three things go wrong quietly: a board only ever in the middle leaves
    the image's corners unmeasured (the cells), a board always square to the lens trades focal
    length against distance (the tilted views), and a board always far away gives corners too
    few pixels apart to constrain the lens (the close views).
    """

    cells: dict[tuple[int, int], int] = field(default_factory=dict)
    tilted: int = 0
    close: int = 0
    total: int = 0
    target: int = TARGET_VIEWS  # how many views this run asked for (ros/calibrate.sh --views)

    @classmethod
    def of(cls, views: Sequence[View], target: int = TARGET_VIEWS) -> Coverage:
        """The coverage of a list of accepted views, against a run that asked for ``target`` of
        them."""
        cells: dict[tuple[int, int], int] = {}
        for view in views:
            cells[view.cell] = cells.get(view.cell, 0) + 1
        return cls(
            cells=cells,
            tilted=sum(1 for v in views if v.tilt >= TILT_MIN),
            close=sum(1 for v in views if v.area >= CLOSE_AREA),
            total=len(views),
            target=target,
        )

    def missing_cells(self) -> list[tuple[int, int]]:
        """The grid cells still short of their quota, in reading order."""
        return [
            (r, c)
            for r in range(GRID)
            for c in range(GRID)
            if self.cells.get((r, c), 0) < CELL_VIEWS
        ]

    def wants(self, view: View) -> bool:
        """Whether this view adds something the set still needs — a thin cell, a tilt, or a
        close look. A view that adds nothing is not refused outright while the set is small:
        more equations never hurt the solve, they only fail to help it."""
        if self.cells.get(view.cell, 0) < CELL_VIEWS:
            return True
        if view.tilt >= TILT_MIN and self.tilted < TILTED_VIEWS:
            return True
        if view.area >= CLOSE_AREA and self.close < CLOSE_VIEWS:
            return True
        return self.total < self.target

    def good(self) -> bool:
        """Whether the set covers enough to be worth solving: every cell filled, enough tilted
        and close views, and the run's own view count reached."""
        return (
            not self.missing_cells()
            and self.tilted >= TILTED_VIEWS
            and self.close >= CLOSE_VIEWS
            and self.total >= self.target
        )

    def hint(self) -> str:
        """The one thing to do next, in a person's words: where to put the board, or how to
        hold it."""
        missing = self.missing_cells()
        if missing:
            row, col = missing[0]
            return f"hold the board at {CELL_NAMES[row][col]} of the frame"
        if self.tilted < TILTED_VIEWS:
            return f"tilt the board (turn a corner towards the lens): {self.tilted}/{TILTED_VIEWS}"
        if self.close < CLOSE_VIEWS:
            return f"bring the board closer, filling the frame: {self.close}/{CLOSE_VIEWS}"
        if self.total < self.target:
            return f"keep moving it around: {self.total}/{self.target} views"
        return "covered — finishing"

    def report(self) -> str:
        """The whole state as text, for the headless run and the final verdict: the 3x3 grid
        with a count per cell, then the tilted, close and total quotas."""
        lines = [f"coverage of the frame ({CELL_VIEWS} views wanted per cell):"]
        for row in range(GRID):
            counts = " ".join(f"{self.cells.get((row, col), 0):>2}" for col in range(GRID))
            lines.append(f"  {counts}")
        lines.append(
            f"  tilted {self.tilted}/{TILTED_VIEWS}   close {self.close}/{CLOSE_VIEWS}"
            f"   views {self.total}/{self.target}"
        )
        if not self.good():
            lines.append(f"  next: {self.hint()}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Verdict:
    """What the collector made of one frame: whether it kept it, how long the board still has
    to be held still (``None`` when nothing is being counted down), and the line to show."""

    captured: bool
    countdown_s: float | None
    message: str


class Collector:
    """Gathers well-spread views of the board from a stream of frames, without a keypress.

    It takes a frame's corners (or ``None`` when the board was not found) and answers a
    :class:`Verdict`. A view is kept when the board has stood still for :data:`HOLD_S` — the
    countdown the operator sees — and when :class:`Coverage` still wants what that view offers;
    a blurred, moving board is never kept, which is the single biggest cause of a bad fit.
    """

    def __init__(self, board: Board, size: tuple[int, int], target: int = TARGET_VIEWS) -> None:
        self.board = board
        self.size = size
        self.target = target
        self.views: list[View] = []
        self._previous: Array | None = None
        self._still_since: float | None = None
        self._still_px = STILL_PX * size[0] / 1280.0

    @property
    def coverage(self) -> Coverage:
        """The coverage of what has been kept so far, against this run's target."""
        return Coverage.of(self.views, self.target)

    def done(self) -> bool:
        """Whether enough well-spread views are in hand: every one of the coverage's quotas met,
        the target count among them."""
        return self.coverage.good()

    def offer(self, corners: Corners | None, now: float) -> Verdict:
        """One frame's corners at time ``now`` (seconds, monotonic). Answers what happened and
        what the operator should do; the view is appended to :attr:`views` when kept."""
        if corners is None or len(corners) != self.board.corners:
            self._previous, self._still_since = None, None
            return Verdict(False, None, "board not found — show the whole board to the camera")
        points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        motion = self._motion(points)
        self._previous = points
        cell, area, tilt = view_shape(corners, self.size)
        view = View(np.asarray(corners, dtype=np.float32), cell, area, tilt)
        if area < FAR_AREA:
            self._still_since = None
            return Verdict(False, None, "too far: the board is a few pixels — come closer")
        if motion is None or motion > self._still_px:
            self._still_since = None
            return Verdict(False, None, f"hold still ({self.coverage.hint()})")
        if self._still_since is None:
            self._still_since = now
        left = HOLD_S - (now - self._still_since)
        if left > 0.0:
            return Verdict(False, left, f"holding {left:.1f} s — {self.coverage.hint()}")
        self._still_since = None
        if not self.coverage.wants(view):
            return Verdict(False, None, f"already have this one — {self.coverage.hint()}")
        self.views.append(view)
        return Verdict(
            True, None, f"kept view {len(self.views)}/{self.target} — {self.coverage.hint()}"
        )

    def _motion(self, points: Array) -> float | None:
        """Mean corner displacement since the previous frame, pixels; ``None`` when there is no
        previous frame or it had another number of corners."""
        previous = self._previous
        if previous is None or previous.shape != points.shape:
            return None
        return float(np.linalg.norm(points - previous, axis=1).mean())


@dataclass(frozen=True)
class Fit:
    """What ``cv2.calibrateCamera`` answered: the lens as :class:`pepin.camera.Calibration`, and
    the per-view reprojection errors that say whether one bad view carried the whole RMS."""

    calibration: Calibration
    per_view: tuple[float, ...]

    @property
    def acceptable(self) -> bool:
        """Whether the fit is good enough to write into the config (RMS under
        :data:`RMS_MAX_PX`)."""
        return self.calibration.rms <= RMS_MAX_PX

    def worst(self) -> tuple[int, float]:
        """The index and error of the view the fit explains worst — where to look when the RMS
        is too high: one blurred view usually owns most of it."""
        if not self.per_view:
            return -1, 0.0
        index = int(np.argmax(self.per_view))
        return index, self.per_view[index]

    def report(self) -> str:
        """The numbers a person reads after a run: the pinhole, the field of view it implies,
        the distortion, the RMS and the worst view."""
        c = self.calibration
        index, error = self.worst()
        return "\n".join(
            [
                f"fx {c.fx:.1f}  fy {c.fy:.1f}  cx {c.cx:.1f}  cy {c.cy:.1f}"
                f"  ({c.width}x{c.height})",
                f"field of view {c.hfov_deg():.1f} deg horizontal"
                f" (the config's nominal number is derived from fx)",
                "distortion " + " ".join(f"{v:+.4f}" for v in c.dist),
                f"rms {c.rms:.3f} px over {c.views} views"
                f" (accepted up to {RMS_MAX_PX:.2f}); worst view #{index} at {error:.3f} px",
            ]
        )


def find_corners(image: Any, board: Board) -> Corners | None:
    """The board's inner corners in a greyscale or colour image, refined to sub-pixel, or
    ``None`` when the whole board is not visible.

    ``findChessboardCornersSB`` is tried first: it is both faster and steadier on a webcam's
    MJPEG than the classic detector, which it falls back to on an OpenCV without it.
    """
    import cv2

    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    finder = getattr(cv2, "findChessboardCornersSB", None)
    if finder is not None:
        found, corners = finder(
            grey, board.size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
        )
        if found:
            return np.asarray(corners, dtype=np.float32)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(grey, board.size, flags=flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)
    return np.asarray(refined, dtype=np.float32)


def calibrate(
    views: Sequence[View] | Sequence[Corners],
    board: Board,
    size: tuple[int, int],
    date: str | None = None,
) -> Fit:
    """Solve the lens from the collected views: ``cv2.calibrateCamera`` over the board's known
    corner grid, answered as a :class:`Fit` at the frame size the views were shot at.

    ``ValueError`` when there are too few views to constrain the model — a fit from three
    pictures is not a calibration, it is a coincidence.
    """
    import cv2

    corners = [v.corners if isinstance(v, View) else v for v in views]
    if len(corners) < 6:
        raise ValueError(f"{len(corners)} views is not a calibration: collect at least 6")
    object_points = [board.object_points().astype(np.float32) for _ in corners]
    image_points = [np.asarray(c, dtype=np.float32).reshape(-1, 1, 2) for c in corners]
    # cameraMatrix and distCoeffs are outputs here: without CALIB_USE_INTRINSIC_GUESS whatever
    # is passed in is ignored, and the arrays are only there because the binding wants them.
    rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        size,
        np.zeros((3, 3), dtype=np.float64),
        np.zeros((1, 5), dtype=np.float64),
    )
    matrix = np.asarray(k, dtype=np.float64)
    coefficients = tuple(float(v) for v in np.asarray(dist, dtype=np.float64).ravel())
    per_view = tuple(
        float(
            np.linalg.norm(
                np.asarray(
                    cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], k, dist)[0],
                    dtype=np.float64,
                ).reshape(-1, 2)
                - image_points[i].reshape(-1, 2).astype(np.float64),
                axis=1,
            ).mean()
        )
        for i in range(len(corners))
    )
    calibration = Calibration(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        width=size[0],
        height=size[1],
        dist=coefficients,
        rms=float(rms),
        date=date or datetime.now(UTC).strftime("%Y-%m-%d"),
        board=str(board),
        views=len(corners),
    )
    return Fit(calibration, per_view)


# ---- the printable board -----------------------------------------------------------------------
# Page sizes in PostScript points (1 pt = 1/72 inch), portrait.
PAGES = {"a4": (595.28, 841.89), "letter": (612.0, 792.0)}
MARGIN_PT = 28.35  # 10 mm of white around the board: the detector needs a quiet zone
CAPTION_PT = 11.0


def board_pdf(board: Board, page: str = "a4") -> bytes:
    """The board as a one-page PDF to print at 100 % (no "fit to page"), with its size printed
    on the sheet. The page is turned landscape when the board is wider than tall, and
    ``ValueError`` is raised when it does not fit the paper at all — better than silently
    printing a board whose squares are not the size the calibration will be told.

    Written by hand rather than through a drawing library: a PDF of filled rectangles is fifty
    lines, and the squares then land at exactly the millimetre they claim.
    """
    if page not in PAGES:
        raise ValueError(f"{page!r}: the page is one of {', '.join(sorted(PAGES))}")
    squares_x, squares_y = board.cols + 1, board.rows + 1
    side_pt = board.square_m * 1000.0 / 25.4 * 72.0
    width_pt, height_pt = squares_x * side_pt, squares_y * side_pt
    page_w, page_h = PAGES[page]
    if (width_pt > height_pt) != (page_w > page_h):
        page_w, page_h = page_h, page_w
    room_w, room_h = page_w - 2 * MARGIN_PT, page_h - 2 * MARGIN_PT - 3 * CAPTION_PT
    if width_pt > room_w or height_pt > room_h:
        raise ValueError(
            f"a {squares_x}x{squares_y} board of {board.square_m * 1000:.0f} mm squares is"
            f" {width_pt / 72 * 25.4:.0f}x{height_pt / 72 * 25.4:.0f} mm and does not fit"
            f" {page}: print fewer corners or smaller squares"
        )
    origin_x = (page_w - width_pt) / 2.0
    origin_y = (page_h - height_pt) / 2.0 + CAPTION_PT
    draw = ["0 0 0 rg"]
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2:
                continue
            x = origin_x + col * side_pt
            y = origin_y + (squares_y - 1 - row) * side_pt
            draw.append(f"{x:.3f} {y:.3f} {side_pt:.3f} {side_pt:.3f} re f")
    caption = (
        f"{board.cols}x{board.rows} inner corners, {board.square_m * 1000:.1f} mm squares"
        f"  -  print at 100% (not 'fit to page'), then measure one square"
    )
    draw.append(
        f"BT /F1 {CAPTION_PT:.0f} Tf {origin_x:.3f} {origin_y - 2 * CAPTION_PT:.3f} Td"
        f" ({_pdf_text(caption)}) Tj ET"
    )
    return _pdf_page("\n".join(draw), page_w, page_h)


def _pdf_text(text: str) -> str:
    """A string as a PDF literal: the three characters that end one are escaped."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _pdf_page(content: str, width_pt: float, height_pt: float) -> bytes:
    """One page of ``content`` (a PDF content stream) as a complete PDF file, with the cross
    reference table its readers need."""
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width_pt:.2f} {height_pt:.2f}]"
        " /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(content.encode('latin-1'))} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = "%PDF-1.4\n"
    offsets: list[int] = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out.encode("latin-1")))
        out += f"{number} 0 obj\n{obj}\nendobj\n"
    start = len(out.encode("latin-1"))
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n"
    return out.encode("latin-1")


def undistort_optics(
    calibration: Calibration, width: int, height: int
) -> tuple[Array, Array, Array]:
    """What it takes to publish a rectified picture at ``width`` x ``height``: the calibration's
    K scaled to that size, its distortion coefficients, and the new K of the straightened image
    (OpenCV's optimal matrix at alpha 0 — the largest all-valid rectangle, so no black border,
    which is what ``image_proc`` publishes too).

    Three arrays, so the caller builds the remap tables and the ``CameraInfo`` from the same
    numbers and cannot publish a picture rectified by one K and described by another.
    """
    import cv2

    scaled = calibration.scaled(width, height)
    k = np.array(
        [[scaled.fx, 0.0, scaled.cx], [0.0, scaled.fy, scaled.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    d = np.array(scaled.dist, dtype=np.float64)
    new_k, _roi = cv2.getOptimalNewCameraMatrix(k, d, (width, height), 0.0, (width, height))
    return k, d, np.asarray(new_k, dtype=np.float64)
