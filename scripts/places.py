#!/usr/bin/env python
"""Name places on a saved map so drives can be asked for by name.

Coordinates are in the map frame (read them off scratch/map_view.png or the
rerun viewer: hover shows x, y). The file lands next to the map as
``<map>.places.json`` and ``navigate.py --goal NAME`` uses it.

Usage:
    uv run python scripts/places.py data/maps/<map>.npz list
    uv run python scripts/places.py data/maps/<map>.npz add kitchen -1.5 2.0 [--face 90]
    uv run python scripts/places.py data/maps/<map>.npz remove kitchen
"""

import argparse
from pathlib import Path

from pepin.places import Place, load_places, places_path, save_places


def main() -> None:
    parser = argparse.ArgumentParser(description="Named places on a saved map.")
    parser.add_argument("map", type=Path, help="occupancy grid .npz the places belong to")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show the places of this map")
    add = sub.add_parser("add", help="add or replace a place")
    add.add_argument("name")
    add.add_argument("x", type=float)
    add.add_argument("y", type=float)
    add.add_argument("--face", type=float, default=None, help="final heading, degrees CCW")
    remove = sub.add_parser("remove", help="forget a place")
    remove.add_argument("name")
    args = parser.parse_args()

    places = load_places(args.map)
    if args.command == "add":
        places[args.name] = Place(args.name, args.x, args.y, args.face)
        print(f"saved to {save_places(args.map, places)}")
    elif args.command == "remove":
        if places.pop(args.name, None) is None:
            raise SystemExit(f"no place named {args.name!r}")
        print(f"saved to {save_places(args.map, places)}")
    if not places:
        print(f"no places yet for {args.map.name} ({places_path(args.map)})")
    for place in sorted(places.values(), key=lambda p: p.name):
        face = f", face {place.theta_deg:.0f} deg" if place.theta_deg is not None else ""
        print(f"  {place.name:<16} x={place.x:+.2f} y={place.y:+.2f}{face}")


if __name__ == "__main__":
    main()
