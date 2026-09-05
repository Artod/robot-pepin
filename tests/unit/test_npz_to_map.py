"""The .npz -> map_server conversion: thresholds, the vertical flip, and cropping.

The invariant every assertion circles is that a world point keeps its meaning:
map_server reads the pixel at column (x - origin_x) / resolution counting from
the left and row (y - origin_y) / resolution counting from the BOTTOM, so a
cell that was occupied in the grid must be black there — cropped or not.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from numpy.typing import NDArray

from pepin.mapping import GridSpec, OccupancyGrid

REPO = Path(__file__).resolve().parents[2]
RESOLUTION = 0.05
X_MIN, Y_MIN = -1.0, -0.5
OCCUPIED_CELL = (12, 26)  # (row, col) in grid order: row 0 is y_min
FREE_ROWS, FREE_COLS = slice(10, 13), slice(22, 25)
UNKNOWN_CELL = (2, 2)
MARGIN_CELLS = 10  # 0.5 m at 0.05 m/cell


def _load_tool() -> Any:
    """Import ros/tools/npz_to_map.py by path — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location(
        "npz_to_map", REPO / "ros" / "tools" / "npz_to_map.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _grid() -> OccupancyGrid:
    """A 2.0 x 1.0 m grid with one occupied cell, a free block, unknown elsewhere."""
    grid = OccupancyGrid(GridSpec(RESOLUTION, X_MIN, Y_MIN, 2.0, 1.0))
    grid.log_odds[FREE_ROWS, FREE_COLS] = -5.0
    grid.log_odds[OCCUPIED_CELL] = 5.0
    return grid


def _cell_centre(row: int, col: int) -> tuple[float, float]:
    """World (x, y) of the centre of a grid cell, in the map frame."""
    return X_MIN + (col + 0.5) * RESOLUTION, Y_MIN + (row + 0.5) * RESOLUTION


def _read_pgm(path: Path) -> NDArray[np.uint8]:
    """Parse a binary P5 PGM into a (height, width) array; row 0 is the top of the image."""
    magic, dimensions, maxval, raster = path.read_bytes().split(b"\n", 3)
    assert magic == b"P5" and maxval == b"255"
    width, height = (int(value) for value in dimensions.split())
    pixels: NDArray[np.uint8] = np.frombuffer(raster, dtype=np.uint8).reshape(height, width)
    return pixels


def _pixel_at(image: NDArray[np.uint8], origin: list[float], x: float, y: float) -> int:
    """Pixel map_server reads at world (x, y): x picks the column, y the row from below."""
    col = int((x - origin[0]) / RESOLUTION)
    row = image.shape[0] - 1 - int((y - origin[1]) / RESOLUTION)
    return int(image[row, col])


def _convert(tmp_path: Path, crop: bool) -> tuple[NDArray[np.uint8], dict[str, Any]]:
    """Save the fixture grid, convert it, and read back (pixels, parsed yaml)."""
    npz_path = tmp_path / "room.npz"
    _grid().save(npz_path)
    pgm_path, yaml_path = tool.convert(npz_path, tmp_path / "out" / "room", crop=crop)
    meta = yaml.safe_load(yaml_path.read_text())
    assert meta["image"] == pgm_path.name
    return _read_pgm(pgm_path), meta


def test_uncropped_map_keeps_the_grid_extent_and_flips_rows(tmp_path: Path) -> None:
    pixels, meta = _convert(tmp_path, crop=False)

    assert meta["mode"] == "trinary"
    assert meta["negate"] == 0
    assert meta["resolution"] == pytest.approx(RESOLUTION)
    assert meta["origin"] == pytest.approx([X_MIN, Y_MIN, 0.0])
    assert meta["occupied_thresh"] == pytest.approx(0.65)
    assert meta["free_thresh"] == pytest.approx(0.196)

    assert pixels.shape == (20, 40)  # (rows, cols) = (1.0 m, 2.0 m) at 0.05 m
    # Grid row 0 sits at y_min, so it is the last image row.
    row, col = OCCUPIED_CELL
    assert pixels[pixels.shape[0] - 1 - row, col] == 0
    assert pixels[0, col] == 205  # the top of the image is the top of the world, still unknown

    origin = meta["origin"]
    assert _pixel_at(pixels, origin, *_cell_centre(*OCCUPIED_CELL)) == 0
    assert _pixel_at(pixels, origin, *_cell_centre(11, 23)) == 254
    assert _pixel_at(pixels, origin, *_cell_centre(*UNKNOWN_CELL)) == 205
    assert set(np.unique(pixels)) == {0, 205, 254}


def test_cropping_trims_the_unknown_border_and_moves_the_origin(tmp_path: Path) -> None:
    pixels, meta = _convert(tmp_path, crop=True)

    # Known cells span cols 22..26 and rows 10..12; the margin widens that by 10 cells
    # and the grid edge clips what is left, so only x is actually trimmed here.
    first_col = FREE_COLS.start - MARGIN_CELLS
    assert pixels.shape == (20, OCCUPIED_CELL[1] + 1 + MARGIN_CELLS - first_col)
    assert meta["origin"] == pytest.approx([X_MIN + first_col * RESOLUTION, Y_MIN, 0.0])

    # The same world point still resolves to the same cell after the shift.
    origin = meta["origin"]
    assert _pixel_at(pixels, origin, *_cell_centre(*OCCUPIED_CELL)) == 0
    assert _pixel_at(pixels, origin, *_cell_centre(11, 23)) == 254
    assert _pixel_at(pixels, origin, *_cell_centre(19, 26)) == 205


def test_a_map_without_observations_is_not_cropped_away(tmp_path: Path) -> None:
    npz_path = tmp_path / "blank.npz"
    OccupancyGrid(GridSpec(RESOLUTION, X_MIN, Y_MIN, 2.0, 1.0)).save(npz_path)
    pgm_path, yaml_path = tool.convert(npz_path, tmp_path / "blank", crop=True)
    pixels = _read_pgm(pgm_path)
    assert pixels.shape == (20, 40)
    assert (pixels == 205).all()
    assert yaml.safe_load(yaml_path.read_text())["origin"] == pytest.approx([X_MIN, Y_MIN, 0.0])
