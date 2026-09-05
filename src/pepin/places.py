"""Named places on a saved map: the words a person or a language model sends the robot to.

A map is a frozen occupancy grid with its own frame; a place is a name for a
pose in that frame ("kitchen" = x, y and, optionally, which way to face on
arrival). Places live next to the map file as ``<map>.places.json`` so a
map and its vocabulary travel together, and the high-level layer never
needs to know coordinates — it asks for a name, :func:`resolve_goal` turns
it into meters.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Place:
    """A named pose in the map frame; ``theta_deg`` is the preferred final heading, or None."""

    name: str
    x: float
    y: float
    theta_deg: float | None = None

    @property
    def xy(self) -> tuple[float, float]:
        """The place as a planner goal."""
        return (self.x, self.y)

    @property
    def theta(self) -> float | None:
        """Preferred heading in radians, or None when any heading will do."""
        return math.radians(self.theta_deg) if self.theta_deg is not None else None


def places_path(map_path: Path) -> Path:
    """Where a map keeps its places: ``data/maps/foo.npz`` -> ``data/maps/foo.places.json``."""
    return map_path.with_suffix(".places.json")


def load_places(map_path: Path) -> dict[str, Place]:
    """Places of a map by name; an empty dict when the map has none yet."""
    path = places_path(map_path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {name: Place(name=name, **entry) for name, entry in data.get("places", {}).items()}


def save_places(map_path: Path, places: dict[str, Place]) -> Path:
    """Write the places file next to the map (sorted by name); returns its path."""
    path = places_path(map_path)
    entries = {
        name: {k: v for k, v in asdict(place).items() if k != "name"}
        for name, place in sorted(places.items())
    }
    payload = {
        "map": map_path.name,
        "frame": "map frame of that grid: meters, x/y as in the .npz, theta_deg counter-clockwise",
        "places": entries,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def resolve_goal(tokens: list[str], map_path: Path) -> tuple[tuple[float, float], Place | None]:
    """``["kitchen"]`` or ``["-2.0", "0.5"]`` into a goal in meters (and the Place, if named).

    Raises ``ValueError`` naming the known places when a name is unknown.
    """
    if len(tokens) == 2:
        return (float(tokens[0]), float(tokens[1])), None
    if len(tokens) != 1:
        raise ValueError("a goal is either a place name or two numbers X Y")
    places = load_places(map_path)
    place = places.get(tokens[0])
    if place is None:
        known = ", ".join(sorted(places)) or "none yet"
        raise ValueError(
            f"unknown place {tokens[0]!r} for {map_path.name}; known: {known} "
            f"(add one: uv run python scripts/places.py {map_path} add NAME X Y)"
        )
    return place.xy, place
