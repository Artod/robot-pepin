"""Rectified stereo pairs rendered from a known depth map, for the matcher's unit tests.

A rectified pair is fully described by one depth image in the LEFT eye: the same speck of the
world sits ``d = fx * B / z`` pixels further left in the right eye. So a scene here is a depth
map, and the two pictures are made from it:

* the left picture IS the texture — band-limited noise, because a lens blurs and because a
  matcher that only works on pure white noise is not being tested honestly;
* the right picture is that texture forward-warped by the disparity with a depth buffer, so a
  near surface covers a far one and the strip of background the near surface uncovers (the
  occlusion band) is filled with texture the left eye never saw. That band is the thing a
  left-right check must answer NaN for, and it cannot be faked by shifting the whole picture.

Nothing here imports ROS or OpenCV's stereo side; only ``cv2.remap`` and ``cv2.GaussianBlur``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]

# The head the tests describe: 800x600 an eye, 94 degrees of horizontal field, 63 mm of baseline.
WIDTH, HEIGHT = 800, 600
FX = 373.0
BASELINE_M = 0.063
FX_B = FX * BASELINE_M  # 23.5 px*m: one pixel of disparity is this many metres of depth


def noise_texture(shape: tuple[int, int], rng: np.random.Generator, blur: float = 0.8) -> Array:
    """Band-limited grey noise of ``shape``: what a textured wall looks like through a lens."""
    import cv2

    raw = rng.integers(0, 256, shape, dtype=np.uint8)
    out: Array = cv2.GaussianBlur(raw, (0, 0), blur)
    return out


def render_pair(
    depth_m: Array, texture: Array, fx_b: float = FX_B, rng: np.random.Generator | None = None
) -> tuple[Array, Array]:
    """The two rectified eyes of a scene: the left picture is ``texture``, the right one is that
    texture moved left by ``fx_b / depth`` pixels with a depth buffer.

    Where the warp leaves a hole — the background a nearer surface uncovers, which only the
    right eye sees — fresh noise is put, as a real scene would show. Pixels whose depth is not
    finite are treated as infinitely far (zero disparity)."""
    import cv2

    rng = rng or np.random.default_rng(0)
    height, width = depth_m.shape
    z = np.asarray(depth_m, dtype=np.float64)
    disparity = np.where(np.isfinite(z) & (z > 0.0), fx_b / np.where(z > 0.0, z, 1.0), 0.0)
    # Forward-warp the DEPTH into the right eye, nearest surface winning, then read the left
    # texture back through it: an inverse map is what cv2.remap wants and what keeps subpixel.
    right_depth = np.full((height, width), np.inf)
    columns = np.arange(width)
    for row in range(height):
        target = np.rint(columns - disparity[row]).astype(int)
        inside = (target >= 0) & (target < width)
        order = np.argsort(-z[row][inside])  # far first, so the near ones overwrite
        right_depth[row, target[inside][order]] = z[row][inside][order]
    filled = np.isfinite(right_depth)
    right_disparity = np.where(filled, fx_b / np.where(filled, right_depth, 1.0), 0.0)
    map_x = (columns[None, :] + right_disparity).astype(np.float32)
    map_y = np.repeat(np.arange(height, dtype=np.float32)[:, None], width, axis=1)
    right = cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    hole = ~filled
    if hole.any():  # the strip of background a nearer surface uncovers: the right eye alone
        right = np.where(hole, noise_texture((height, width), rng), right)
    return np.ascontiguousarray(texture), np.ascontiguousarray(right.astype(np.uint8))


def planes_scene(
    depths_m: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    rng: np.random.Generator | None = None,
) -> tuple[Array, Array, Array]:
    """Vertical bands of fronto-parallel textured planes at ``depths_m``: (left, right, truth)."""
    rng = rng or np.random.default_rng(1)
    truth = np.empty((HEIGHT, WIDTH))
    edges = np.linspace(0, WIDTH, len(depths_m) + 1).astype(int)
    for k, z in enumerate(depths_m):
        truth[:, edges[k] : edges[k + 1]] = z
    left, right = render_pair(truth, noise_texture((HEIGHT, WIDTH), rng), rng=rng)
    return left, right, truth


def step_scene(
    near_m: float = 1.0, far_m: float = 2.5, rng: np.random.Generator | None = None
) -> tuple[Array, Array, Array]:
    """One depth step down the middle of the picture: (left, right, truth). The strip of ``far``
    the near surface hides from the left eye is the occlusion band, and it sits just LEFT of the
    step in the left picture — ``fx_b / near - fx_b / far`` pixels of it."""
    rng = rng or np.random.default_rng(2)
    truth = np.full((HEIGHT, WIDTH), far_m)
    truth[:, WIDTH // 2 :] = near_m
    left, right = render_pair(truth, noise_texture((HEIGHT, WIDTH), rng), rng=rng)
    return left, right, truth


def bars_scene(
    bar_px: int = 8,
    near_m: float = 1.0,
    far_m: float = 2.5,
    rng: np.random.Generator | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Vertical bars ``bar_px`` wide at ``near_m`` in front of a wall at ``far_m``: (left, right,
    truth, bar mask). A chair leg at range is this scene, and it is what a matching block wider
    than the bar loses."""
    rng = rng or np.random.default_rng(4)
    truth = np.full((HEIGHT, WIDTH), far_m)
    bars = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for start in range(WIDTH // 8, WIDTH - WIDTH // 8, WIDTH // 8):
        bars[:, start : start + bar_px] = True
    truth[bars] = near_m
    left, right = render_pair(truth, noise_texture((HEIGHT, WIDTH), rng), rng=rng)
    return left, right, truth, bars


def blank_scene(
    depth_m: float = 1.2, rng: np.random.Generator | None = None
) -> tuple[Array, Array, Array, Array]:
    """A textured wall with a blank rectangle painted on it: (left, right, truth, blank mask).
    The blank is one grey value in BOTH eyes, so every disparity in the search matches it
    equally well and any answer there is the smoothness term talking, not a measurement."""
    rng = rng or np.random.default_rng(3)
    truth = np.full((HEIGHT, WIDTH), depth_m)
    texture = noise_texture((HEIGHT, WIDTH), rng)
    blank = np.zeros((HEIGHT, WIDTH), dtype=bool)
    blank[HEIGHT // 4 : 3 * HEIGHT // 4, WIDTH // 4 : 3 * WIDTH // 4] = True
    texture = np.where(blank, np.uint8(128), texture).astype(np.uint8)
    left, right = render_pair(truth, texture, rng=rng)
    right = np.where(blank, np.uint8(128), right).astype(np.uint8)
    return left, np.ascontiguousarray(right), truth, blank


def occlusion_band(truth: Array, fx_b: float = FX_B) -> Array:
    """Which pixels of the LEFT eye the right eye cannot see: those a nearer surface to their
    right moves in front of. True exactly over the band beside a depth step."""
    height, width = truth.shape
    disparity = fx_b / truth
    hidden = np.zeros((height, width), dtype=bool)
    columns = np.arange(width)
    for row in range(height):
        target = columns - disparity[row]
        # A pixel is hidden when some pixel to its right lands on it or past it in the right eye.
        running = np.minimum.accumulate(target[::-1])[::-1]
        hidden[row] = np.r_[running[1:] <= target[:-1], False]
    return hidden
