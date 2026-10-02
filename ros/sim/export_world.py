"""Export the room the stack drives on into a world for the simulator.

The map is RTAB-Map's grid, the one ``/map`` carries: the grid
the database saved at its last shutdown (Admin.opt_map), which is what RTAB-Map publishes the
moment it loads that database to localise. The places are ``rtabmap.places.json`` resolved
against the same saved graph (pepin.places), written as the ``/places`` payload the laptop's
places node publishes. The database is COPIED first and the copy is read: a live RTAB-Map may
hold the file.

    uv run python ros/sim/export_world.py [DB] [PLACES] [OUT_YAML]
    (defaults: ros/maps/rtabmap.db, ros/maps/rtabmap.places.json, ros/sim/worlds/flat.yaml)
"""

from __future__ import annotations

import argparse
import datetime
import shutil
import sys
import tempfile
from pathlib import Path

from pepin.places import load_graph_places, places_json
from pepin.sim import graph_poses_from_rtabmap_db, grid_from_rtabmap_db

ROS = Path(__file__).resolve().parents[1]


def export(db: Path, places: Path, out_yaml: Path) -> str:
    """Write ``out_yaml`` (+ .pgm, + .places.json beside it); returns a one-line summary."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "rtabmap.db"
        shutil.copyfile(db, copy)
        grid = grid_from_rtabmap_db(copy)
        nodes = graph_poses_from_rtabmap_db(copy)
    saved = datetime.datetime.fromtimestamp(db.stat().st_mtime).isoformat(timespec="seconds")
    book = load_graph_places(places)
    payload = places_json(book, nodes)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    grid.save(
        out_yaml,
        note=(
            f"RTAB-Map's saved grid (Admin.opt_map) of {db.name}, last written {saved};"
            f" {len(nodes)} graph nodes.\n"
            "Written by ros/sim/export_world.py; re-export, do not edit."
        ),
    )
    out_yaml.with_suffix(".places.json").write_text(payload + "\n")
    missing = sorted(set(book) - set(places_json_names(payload)))
    return (
        f"{out_yaml}: {grid.width}x{grid.height} cells at {grid.resolution:.3f} m, origin"
        f" ({grid.origin_x:.3f}, {grid.origin_y:.3f}), {len(nodes)} nodes, places"
        f" {places_json_names(payload)}" + (f", NOT in the graph: {missing}" if missing else "")
    )


def places_json_names(payload: str) -> list[str]:
    """The names a ``/places`` payload answers for."""
    import json

    return sorted(json.loads(payload))


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and export."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", nargs="?", type=Path, default=ROS / "maps/rtabmap.db")
    parser.add_argument("places", nargs="?", type=Path, default=ROS / "maps/rtabmap.places.json")
    parser.add_argument("out", nargs="?", type=Path, default=ROS / "sim/worlds/flat.yaml")
    args = parser.parse_args(argv)
    print(export(args.db, args.places, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
