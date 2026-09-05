#!/usr/bin/env python
"""Convert a saved occupancy grid into the ROS ``map_server`` pair (.pgm + .yaml).

Our grid stores log-odds with row 0 at ``y_min`` (bottom of the world);
``nav2_map_server`` wants an 8-bit PGM whose row 0 is the top of the image and
whose lower-left pixel sits at the ``origin`` given in the YAML. So the grid is
thresholded into the three trinary values (0 occupied, 254 free, 205 unknown)
and flipped vertically; columns map to x untouched.

Usage:
    uv run python ros/tools/npz_to_map.py data/maps/<name>.npz --out ros/maps/<name>
    uv run python ros/tools/npz_to_map.py data/maps/<name>.npz --no-crop
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pepin.mapping import OccupancyGrid

# Pixel values map_server reads as occupied / free / unknown in trinary mode.
PGM_OCCUPIED = 0
PGM_FREE = 254
PGM_UNKNOWN = 205

# Thresholds written into the YAML; map_server compares them against (255 - p) / 255.
OCCUPIED_THRESH = 0.65
FREE_THRESH = 0.196

# Our own occupancy probabilities for the same three-way split.
OCCUPIED_PROBABILITY = 0.65
FREE_PROBABILITY = 0.35

CROP_MARGIN_M = 0.5


@dataclass(frozen=True)
class MapImage:
    """A trinary map image (row 0 = top) plus the world pose of its lower-left pixel."""

    pixels: NDArray[np.uint8]
    origin_xy: tuple[float, float]
    resolution_m: float

    @property
    def size(self) -> tuple[int, int]:
        """Image size in pixels as (width, height)."""
        return self.pixels.shape[1], self.pixels.shape[0]


def trinary(probability: NDArray[np.float64]) -> NDArray[np.uint8]:
    """Occupancy probabilities to the three map_server pixel values, rows unchanged."""
    pixels = np.full(probability.shape, PGM_UNKNOWN, dtype=np.uint8)
    pixels[probability <= FREE_PROBABILITY] = PGM_FREE
    pixels[probability >= OCCUPIED_PROBABILITY] = PGM_OCCUPIED
    return pixels


def known_bounds(pixels: NDArray[np.uint8], margin_cells: int) -> tuple[slice, slice]:
    """Row and column slices covering every known cell plus ``margin_cells`` of border."""
    known = pixels != PGM_UNKNOWN
    if not known.any():
        return slice(0, pixels.shape[0]), slice(0, pixels.shape[1])
    rows = np.flatnonzero(known.any(axis=1))
    cols = np.flatnonzero(known.any(axis=0))
    row_stop = min(pixels.shape[0], int(rows[-1]) + 1 + margin_cells)
    col_stop = min(pixels.shape[1], int(cols[-1]) + 1 + margin_cells)
    return (
        slice(max(0, int(rows[0]) - margin_cells), row_stop),
        slice(max(0, int(cols[0]) - margin_cells), col_stop),
    )


def to_map_image(grid: OccupancyGrid, crop: bool = True) -> MapImage:
    """Threshold, optionally crop to the known area, and flip the grid into image order."""
    spec = grid.spec
    cells = trinary(grid.probability())
    rows, cols = slice(0, cells.shape[0]), slice(0, cells.shape[1])
    if crop:
        rows, cols = known_bounds(cells, round(CROP_MARGIN_M / spec.resolution_m))
    origin = (
        spec.x_min_m + cols.start * spec.resolution_m,
        spec.y_min_m + rows.start * spec.resolution_m,
    )
    # Grid row 0 is y_min; the PGM's last row is the one that sits at the origin.
    pixels = np.ascontiguousarray(np.flipud(cells[rows, cols]))
    return MapImage(pixels, origin, spec.resolution_m)


def write_pgm(path: Path, pixels: NDArray[np.uint8]) -> None:
    """Write a binary P5 PGM: header, then one byte per pixel, top row first."""
    height, width = pixels.shape
    with path.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        handle.write(pixels.tobytes())


def write_yaml(path: Path, image_name: str, image: MapImage) -> None:
    """Write the map_server YAML describing ``image_name``: geometry and thresholds."""
    x, y = image.origin_xy
    path.write_text(
        f"image: {image_name}\n"
        "mode: trinary\n"
        f"resolution: {image.resolution_m:.6f}\n"
        f"origin: [{x:.6f}, {y:.6f}, 0.0]\n"
        "negate: 0\n"
        f"occupied_thresh: {OCCUPIED_THRESH}\n"
        f"free_thresh: {FREE_THRESH}\n",
        encoding="ascii",
    )


def write_map(image: MapImage, out_stem: Path) -> tuple[Path, Path]:
    """Write ``<out_stem>.pgm`` and ``<out_stem>.yaml``, creating the directory; returns both."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    # Append rather than replace a suffix: map names may carry dots (lap3.loop).
    pgm_path = out_stem.with_name(out_stem.name + ".pgm")
    yaml_path = out_stem.with_name(out_stem.name + ".yaml")
    write_pgm(pgm_path, image.pixels)
    write_yaml(yaml_path, pgm_path.name, image)
    return pgm_path, yaml_path


def convert(npz_path: Path, out_stem: Path, crop: bool = True) -> tuple[Path, Path]:
    """Convert ``npz_path`` into ``<out_stem>.pgm`` and ``<out_stem>.yaml``; returns both."""
    return write_map(to_map_image(OccupancyGrid.load(npz_path), crop=crop), out_stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Occupancy grid .npz to a ROS map_server map.")
    parser.add_argument("npz", type=Path, help="grid saved by OccupancyGrid.save()")
    parser.add_argument(
        "--out", type=Path, help="output stem (default: ros/maps/<npz name>)", default=None
    )
    parser.add_argument(
        "--crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"trim the unknown border, keeping {CROP_MARGIN_M} m of margin (default: on)",
    )
    args = parser.parse_args()

    out_stem = args.out if args.out is not None else Path("ros/maps") / args.npz.stem
    image = to_map_image(OccupancyGrid.load(args.npz), crop=args.crop)
    pgm_path, yaml_path = write_map(image, out_stem)

    width, height = image.size
    x, y = image.origin_xy
    print(f"wrote {pgm_path}")
    print(f"wrote {yaml_path}")
    print(f"image {width} x {height} px at {image.resolution_m:.3f} m/px")
    print(f"origin (lower-left pixel) [{x:.3f}, {y:.3f}, 0.0] m")


if __name__ == "__main__":
    main()
