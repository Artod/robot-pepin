"""The stereo head as a depth source: two rectified eyes in, metric depth out.

The mono network guesses a depth from one picture and needs the lidar to tell it what a metre
is. A calibrated stereo pair MEASURES one: the same speck of wall lands ``disparity`` pixels
further left in the right eye than in the left, and ``z = fx * baseline / disparity`` is metres
by construction. This module is the measuring half and nothing else — rectification and the
calibrated rig live in :mod:`pepin.stereo`.

* :class:`MatcherSettings` is every number OpenCV's semi-global block matcher takes, frozen,
  with the measurement behind each default written beside it.
* :class:`StereoMatcher` runs it: two rectified grey eyes in, a DISPARITY image in pixels out,
  NaN wherever the match is not trusted. Three things make a pixel NaN and they are why this is
  usable for a costmap — the left-right check kills the occlusion band, the uniqueness ratio
  kills ambiguous matches, and the texture gate kills the blank wall. That last one is not
  optional: measured on a rendered wall with a blank rectangle painted on it, SGBM answered
  confidently over 97.3 % of the blank and 5.3 % with the gate on
  (scratch/stereo/sgbm_accuracy.py). A confident plateau on a blank wall is an obstacle nothing
  is standing at.
* :class:`Baseline` is the rectified pinhole's ``fx`` and the eyes' separation, all a disparity
  needs to become metres, built from the two ``camera_info`` messages the stereo camera
  publishes — the node that consumes the eyes has those two numbers and not the remap tables a
  :class:`pepin.stereo.Rectifier` is built from.
* :class:`StereoDepth` is the source: pictures in, float32 metres out, NaN where unknown and
  NaN past :attr:`StereoDepth.reach`.

THE ERROR MODEL is why a reach exists. A disparity measured to ``sigma_d`` pixels gives a depth
good to ``sigma_z = z^2 / (fx * B) * sigma_d``: the error grows with the SQUARE of the range, so
a stereo head is excellent up close and worthless far away — the exact opposite of the mono
network, whose scale law is fitted over the whole room. This rig is 800x600 an eye, 94 degrees
of horizontal field (``fx`` 373 px) and ``B`` 0.063 m, so ``fx * B`` = 23.5 px*m, and with
``sigma_d`` :data:`DISPARITY_SIGMA_PX` the error reaches :data:`DEPTH_SIGMA_M` at 2.17 m. That
is the reach, and it is derived here rather than copied from the network's 3.0 m.

TIMING, 50 real pairs off the board's stream at 800x600 an eye (scratch/stereo/sgbm_sweep.py,
2026-09-20), median/p95 ms per pair:

===================  ================  =================
mode and size        Mac (cv2 4.13)    container (4.6)
===================  ================  =================
3way, full size      14.5 / 14.9       15.9 / 19.8
3way, half size       3.8 /  3.9        4.7 /  7.0
hh, full size        73.8 / 75.0       100.0 / 106.2
hh, half size        11.4 / 11.6       17.8 / 19.8
===================  ================  =================

The node runs in the container (Linux arm64 under Docker Desktop) and the camera delivers
10 fps, so the budget is 100 ms a frame: 3way at full size spends 16 % of it and holds the
camera's rate with the whole correction pipeline still to pay for, while hh at full size IS the
whole budget at 100.0 ms median. 3way costs nothing in accuracy — the two agree to 0.00 % of
median depth error and to a point of valid pixels — so 3way at full size is the default and
``mode`` and ``downscale`` are the knobs.

OpenCV is imported inside the calls that need it: this module is read by unit tests that must
stay millisecond-fast.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from pepin.stereo import MIN_DISPARITY_PX, Array
from pepin.telemetry import LatencyTracker

DISP_SCALE = 16.0  # OpenCV's fixed point: the matcher answers in sixteenths of a pixel
# How well a disparity is measured, in pixels. On rendered pairs the matcher's subpixel fit sits
# a steady 0.12-0.19 px short of the truth at every range (scratch/stereo/sgbm_accuracy.py: 0.80 %
# of 1 m, 1.57 % of 2 m, 2.09 % of 4 m — all of it that one bias), and the pictures it will
# actually see are MJPEG off the board, blocky where the texture is faint. Half a pixel is the
# honest number for those; it is what the reach hangs on, so it is stated, not assumed.
DISPARITY_SIGMA_PX = 0.5
DEPTH_SIGMA_M = 0.10  # the depth error at which this camera stops answering for its own depth
MODES = ("sgbm", "hh", "hh4", "3way")


class StereoUnavailableError(RuntimeError):
    """The stereo source cannot answer this frame: no calibrated geometry yet, or two eyes whose
    shapes do not match."""


def reach_m(
    fx: float,
    baseline_m: float,
    sigma_d_px: float = DISPARITY_SIGMA_PX,
    sigma_z_m: float = DEPTH_SIGMA_M,
) -> float:
    """How far a rig answers for its own depth, metres: where ``sigma_z = z^2 / (fx * B) *
    sigma_d`` crosses ``sigma_z_m``. 0 for a rig with no focal length or no baseline."""
    span = fx * baseline_m
    if span <= 0.0 or sigma_d_px <= 0.0 or sigma_z_m <= 0.0:
        return 0.0
    return math.sqrt(sigma_z_m * span / sigma_d_px)


def near_m(fx: float, baseline_m: float, num_disparities: int) -> float:
    """The nearest depth a search of ``num_disparities`` pixels can still measure, metres:
    ``fx * B / num_disparities``. 0 when the rig or the search is empty."""
    if num_disparities <= 0 or fx * baseline_m <= 0.0:
        return 0.0
    return float(fx * baseline_m) / float(num_disparities)


class DisparityToDepth(Protocol):
    """What a disparity needs to become metres: the rectified pinhole's focal length, the eyes'
    separation, and the conversion itself (:class:`pepin.stereo.Rectifier` is one of these)."""

    @property
    def fx(self) -> float:
        """The rectified pinhole's focal length in pixels."""
        ...

    @property
    def baseline_m(self) -> float:
        """The distance between the two optical centres, metres."""
        ...

    def depth_m(self, disparity_px: Array) -> Array:
        """Metres along the optical axis for a disparity image in pixels; NaN where unknown."""
        ...


@dataclass(frozen=True)
class Baseline:
    """The rectified pinhole's focal length and the eyes' separation, as the two ``camera_info``
    messages of a stereo pair state them — the whole of what turns a disparity into metres,
    without the remap tables a :class:`pepin.stereo.Rectifier` also carries.

    The node that reads the eyes never opens the calibration file: it is handed the rectified
    pictures and their ``camera_info``, and the right eye's ``P[0, 3] = -fx * B`` is where the
    baseline comes from. :meth:`depth_m` is the same arithmetic as
    :meth:`pepin.stereo.Rectifier.depth_m`, and a unit test pins the two together so they cannot
    drift apart in silence."""

    fx: float
    baseline_m: float

    @classmethod
    def from_projection(cls, fx: float, right_tx: float) -> Baseline:
        """From the left eye's ``fx`` and the right eye's ``P[0, 3]`` (``-fx * B``, ROS's stereo
        convention). A zero or positive ``right_tx`` is a right eye that has not been told its
        baseline: the pair would be unusable, so this raises instead of inventing a metre."""
        if fx <= 0.0:
            raise StereoUnavailableError(f"the rectified pinhole has no focal length (fx {fx})")
        if right_tx >= 0.0:
            raise StereoUnavailableError(
                f"the right eye's P[0,3] is {right_tx:+.3f}: it must be -fx * baseline"
            )
        return cls(fx=float(fx), baseline_m=float(-right_tx / fx))

    def depth_m(self, disparity_px: Array) -> Array:
        """Metres along the optical axis for a disparity image in pixels (float32); NaN where the
        disparity is missing, negative or under :data:`pepin.stereo.MIN_DISPARITY_PX`."""
        disparity = np.asarray(disparity_px, dtype=np.float32)
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        seen = np.isfinite(disparity) & (disparity >= MIN_DISPARITY_PX)
        depth[seen] = np.float32(self.fx * self.baseline_m) / disparity[seen]
        return depth


@dataclass(frozen=True)
class MatcherSettings:
    """Every number OpenCV's semi-global block matcher takes, with the measurement behind each
    default: timings from 50 real pairs (scratch/stereo/sgbm_sweep.py) and accuracy from rendered
    pairs of known geometry (scratch/stereo/sgbm_accuracy.py), both 2026-09-20."""

    # The search window, in pixels of the full-size picture. 128 at fx*B = 23.5 px*m measures
    # down to 0.18 m: the cart parks with its bumper against the furniture, so the near end is
    # the end that matters. It is not free — the leftmost num_disparities columns have no right
    # eye to be found in and come back NaN, which is 16 % of an 800 px picture (measured: 83.9 %
    # of a full-frame plane answered at 128 against 91.9 % at 64, exactly the 64 px difference).
    # 96 gives that band back at 12 % and stops at 0.24 m; 192 reaches 0.12 m for 24 % of the
    # picture and 2.7 ms more. Cost of the search itself is mild: 11.0 / 12.8 / 14.5 / 17.2 ms on
    # the Mac for 64 / 96 / 128 / 192.
    num_disparities: int = 128
    min_disparity: int = 0
    # 5 is the peak for thin things, which is the only axis that separates the block sizes here.
    # An 8 px bar at 1 m in front of a wall (a chair leg at range) is recovered over 65.6 / 67.1
    # / 62.4 / 47.1 / 19.8 % of itself at block 3 / 5 / 7 / 9 / 11, while median depth error is
    # 1.57 % for every one of them. On the real frames the wider blocks answer for more pixels
    # (49.5 / 54.2 / 58.4 / 61.2 %) at the same milliseconds — bought by swallowing the chair leg.
    block_size: int = 5
    # The smoothness penalties in OpenCV's own units: P1 for a one-disparity step between
    # neighbouring pixels, P2 for a bigger one. None takes the documented rule, 8 and 32 times
    # channels times block_size squared (200 and 800 at block 5). Halving P2 costs 11 points of
    # valid pixels on the real frames (43.0 % against 54.2 %); doubling it makes the field so
    # stiff that the 8 px bar survives over 24.5 % of itself instead of 67.1 %.
    p1: int | None = None
    p2: int | None = None
    # The margin the winning disparity must beat the runner-up by, in percent. 10 is OpenCV's own
    # middle. 5 keeps 3.6 more points of the real frames and puts them on repeated texture
    # (a radiator, a bookshelf); 15 throws away 4.0 points and gains nothing measurable.
    uniqueness_ratio: int = 10
    # The left-right consistency check, in pixels: what makes the occlusion band NaN instead of
    # the background's disparity smeared across the foreground's shadow. Measured on a rendered
    # 1.0 -> 2.5 m step, only 8.3 % of the true 14 px band comes back with an answer. NOTE:
    # OpenCV's SGBM clamps this to at least 1 — -1, 0 and 1 measured identical to the pixel, 2
    # and 5 progressively looser — so the check cannot be switched off here, only loosened.
    disp12_max_diff: int = 1
    # Blobs of up to this many pixels disagreeing with their surroundings by more than
    # speckle_range disparities are erased. OpenCV's recommended middle; on the real frames it
    # removes 8.2 points of pixels (62.4 % answered without it) and costs 1.0-2.4 ms.
    speckle_window_size: int = 100
    speckle_range: int = 2
    pre_filter_cap: int = 31  # the Birchfield-Tomasi cost clip; OpenCV's own default
    # 3way against hh: the module docstring's table. Identical accuracy, a fifth to a sixth of
    # the time, and hh at full size IS the camera's whole 100 ms frame budget in the container.
    mode: str = "3way"
    # Match at 1/downscale of the picture and scale the disparity back; the search shrinks with
    # the picture, so the nearest measurable depth does not move. It is the speed knob of last
    # resort, not a default: at 1/2 size the 8 px bar is recovered over 1.8 % of itself instead
    # of 67.1 %, the occlusion band leaks twice as much (17.6 % against 8.3 %), and the depth
    # quantisation doubles (2.09 % median error at every range instead of 0.80 % at 1 m).
    downscale: int = 1
    # The texture gate: the mean absolute Sobel response, in grey levels, a pixel's neighbourhood
    # must reach before its disparity is believed. SGBM has no gate of its own — StereoBM's
    # textureThreshold is this idea — and its smoothness term paints a confident plateau across a
    # blank wall: 97.3 % of a rendered blank rectangle came back answered with the gate off
    # against 5.3 % at 6.0. Raising it to 10 or 15 buys another 0.3 and 0.7 points of that
    # rectangle and throws away 3.3 and 6.9 points of the real frames. It costs 0.9 ms on the Mac
    # and 1.4-1.9 ms in the container.
    texture_threshold: float = 6.0
    texture_window: int = 9  # the neighbourhood the response is averaged over, pixels

    def __post_init__(self) -> None:
        if self.num_disparities <= 0 or self.num_disparities % 16:
            raise ValueError(
                f"num_disparities must be a positive multiple of 16, not {self.num_disparities}"
            )
        if self.block_size < 1 or not self.block_size % 2:
            raise ValueError(f"block_size must be odd and positive, not {self.block_size}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {self.mode!r}")
        if self.downscale < 1:
            raise ValueError(f"downscale must be 1 or more, not {self.downscale}")

    @property
    def smoothness(self) -> tuple[int, int]:
        """(P1, P2) as the matcher is given them: the stated pair, or OpenCV's documented rule
        ``8 * block_size^2`` and ``32 * block_size^2`` for a single-channel picture. P2 is always
        held above P1, which OpenCV requires."""
        block = self.block_size * self.block_size
        p1 = 8 * block if self.p1 is None else self.p1
        p2 = 32 * block if self.p2 is None else self.p2
        return int(p1), int(max(p2, p1 + 1))

    @property
    def search_px(self) -> int:
        """How many disparities the matcher searches at its OWN (possibly downscaled) size: the
        full-size search divided by the downscale, rounded up to a multiple of 16 so the nearest
        measurable depth never ends up further away than was asked for."""
        wanted = self.num_disparities / self.downscale
        return max(16, math.ceil(wanted / 16.0) * 16)

    def describe(self) -> str:
        """One phrase for a report line: the search, the block, the mode and the scale."""
        scale = "" if self.downscale == 1 else f", 1/{self.downscale} size"
        return f"{self.num_disparities}px/{self.block_size}px {self.mode}{scale}"


class StereoMatcher:
    """OpenCV's semi-global block matcher behind one call: two rectified grey eyes in, a float32
    disparity image in pixels out, NaN where the match is not trusted.

    The OpenCV object is built on the first pair and reused; one worker thread calls it."""

    def __init__(self, settings: MatcherSettings | None = None) -> None:
        self.settings = settings or MatcherSettings()
        self.timing = {
            "match": LatencyTracker("match"),
            "texture": LatencyTracker("texture"),
            "total": LatencyTracker("total"),
        }
        self._matcher: Any = None

    def _build(self) -> Any:
        """The OpenCV matcher these settings describe, built once and kept."""
        import cv2

        s = self.settings
        p1, p2 = s.smoothness
        mode = {
            "sgbm": cv2.STEREO_SGBM_MODE_SGBM,
            "hh": cv2.STEREO_SGBM_MODE_HH,
            "hh4": cv2.STEREO_SGBM_MODE_HH4,
            "3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        }[s.mode]
        return cv2.StereoSGBM.create(
            minDisparity=s.min_disparity,
            numDisparities=s.search_px,
            blockSize=s.block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=s.disp12_max_diff,
            preFilterCap=s.pre_filter_cap,
            uniquenessRatio=s.uniqueness_ratio,
            speckleWindowSize=s.speckle_window_size,
            speckleRange=s.speckle_range,
            mode=mode,
        )

    def __call__(self, left: Array, right: Array) -> Array:
        """The disparity of one pair, float32 pixels at the pictures' own size — NaN over the
        left band no right eye can see into, and wherever the match failed the left-right check,
        the uniqueness ratio, the speckle filter or the texture gate."""
        import cv2

        s = self.settings
        with self.timing["total"].measure():
            grey_left, grey_right = _grey(left), _grey(right)
            if grey_left.shape != grey_right.shape:
                raise StereoUnavailableError(
                    f"the eyes are {grey_left.shape} and {grey_right.shape}: not one rig"
                )
            small_left, small_right = grey_left, grey_right
            if s.downscale > 1:
                size = (grey_left.shape[1] // s.downscale, grey_left.shape[0] // s.downscale)
                small_left = cv2.resize(grey_left, size, interpolation=cv2.INTER_AREA)
                small_right = cv2.resize(grey_right, size, interpolation=cv2.INTER_AREA)
            if self._matcher is None:
                self._matcher = self._build()
            with self.timing["match"].measure():
                raw = self._matcher.compute(small_left, small_right)
            disparity = raw.astype(np.float32) / np.float32(DISP_SCALE)
            # OpenCV writes (min_disparity - 1) * 16 wherever it refused to answer: the left band
            # the search runs off the picture in, the occlusions the left-right check caught, the
            # ambiguous matches the uniqueness ratio caught, the speckles.
            disparity[raw <= (s.min_disparity - 1) * DISP_SCALE] = np.nan
            if s.downscale > 1:
                disparity *= np.float32(s.downscale)
                disparity = cv2.resize(
                    disparity,
                    (grey_left.shape[1], grey_left.shape[0]),
                    interpolation=cv2.INTER_NEAREST,  # never blend a disparity with a NaN
                )
            with self.timing["texture"].measure():
                blank = _textureless(grey_left, s.texture_threshold, s.texture_window)
            if blank is not None:
                disparity[blank] = np.nan
        out: Array = disparity
        return out

    def describe(self) -> str:
        """The settings and the median/p95 milliseconds of the pairs seen so far."""
        total = self.timing["total"].summary()
        return f"{self.settings.describe()} {total.median_ms:.0f}/{total.p95_ms:.0f} ms"


def _grey(picture: Array) -> Array:
    """One eye as the contiguous uint8 single channel the matcher demands: a colour picture
    through the luma weights, a grey one as it stands."""
    import cv2

    px = np.asarray(picture)
    if px.ndim == 3:
        px = cv2.cvtColor(px, cv2.COLOR_BGR2GRAY)
    if px.dtype != np.uint8:
        px = np.clip(px, 0, 255).astype(np.uint8)
    out: Array = np.ascontiguousarray(px)
    return out


def _textureless(grey: Array, threshold: float, window: int) -> Array | None:
    """Where the left eye carries too little texture for a disparity to mean anything: the mean
    absolute Sobel response over a ``window`` neighbourhood, under ``threshold`` grey levels.
    ``None`` when the gate is off (``threshold`` 0 or below)."""
    import cv2

    if threshold <= 0.0:
        return None
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.boxFilter(np.abs(gx) + np.abs(gy), cv2.CV_32F, (window, window))
    out: Array = energy < np.float32(threshold)
    return out


class StereoDepth:
    """The stereo head as a raw depth source: a left eye and a right eye of the SAME moment, both
    rectified, in — a float32 depth image in metres out, NaN where the match failed and NaN past
    :attr:`reach` (this rig's own error model, not the mono network's 3.0 m).

    The geometry arrives on a topic, not in the constructor: the node that owns this object is
    handed ``camera_info`` after it is built, so :attr:`geometry` is settable and a call before it
    is known raises :class:`StereoUnavailableError` rather than inventing a metre."""

    name = "stereo"

    def __init__(
        self,
        geometry: DisparityToDepth | None = None,
        matcher: StereoMatcher | None = None,
        reach: float = 0.0,
        valid_window: int = 512,
    ) -> None:
        self.matcher = matcher or StereoMatcher()
        self.timing = {"total": LatencyTracker("stereo")}
        self._valid: deque[float] = deque(maxlen=valid_window)
        self._reach = float(reach)
        self._geometry: DisparityToDepth | None = None
        self.frames = 0
        if geometry is not None:
            self.geometry = geometry

    @property
    def geometry(self) -> DisparityToDepth | None:
        """The rig's rectified focal length and baseline, or ``None`` until camera_info says."""
        return self._geometry

    @geometry.setter
    def geometry(self, value: DisparityToDepth) -> None:
        """Take the rig's numbers; a reach nobody stated is derived from them here."""
        self._geometry = value
        if self._reach <= 0.0:
            self._reach = reach_m(value.fx, value.baseline_m)

    @property
    def reach(self) -> float:
        """How far this source answers for its own depth, metres: the constructor's number, or
        the range where the error model crosses :data:`DEPTH_SIGMA_M` once the geometry is known.
        0 while it is not."""
        return self._reach

    @property
    def near(self) -> float:
        """The nearest depth the search reaches, metres; 0 while the geometry is unknown."""
        g = self._geometry
        if g is None:
            return 0.0
        s = self.matcher.settings
        return near_m(g.fx, g.baseline_m, s.search_px * s.downscale)

    @property
    def valid_fraction(self) -> float:
        """The share of pixels that came back with a depth, over the last frames measured."""
        return float(np.mean(self._valid)) if self._valid else 0.0

    def __call__(self, left: Array, right: Array) -> Array:
        """One pair through the matcher and the rig's geometry: float32 metres, NaN elsewhere."""
        geometry = self._geometry
        if geometry is None:
            raise StereoUnavailableError("no rectified camera_info from both eyes yet")
        with self.timing["total"].measure():
            disparity = self.matcher(left, right)
            depth = geometry.depth_m(disparity)
            if self._reach > 0.0:
                depth[depth > np.float32(self._reach)] = np.nan
            self.frames += 1
            self._valid.append(float(np.count_nonzero(np.isfinite(depth))) / max(depth.size, 1))
        return depth

    def describe(self) -> str:
        """One phrase for a report line: the matcher and its milliseconds, the share of pixels
        answered for, and the near and far end of what this rig measures."""
        return (
            f"{self.matcher.describe()}, {self.valid_fraction * 100:.0f}% valid,"
            f" {self.near:.2f}-{self._reach:.2f} m"
        )
