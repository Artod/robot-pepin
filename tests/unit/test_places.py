"""Named places: file round trip and goal resolution by name or by numbers."""

import math
from pathlib import Path

import pytest

from pepin.places import Place, load_places, resolve_goal, save_places


def test_places_round_trip_next_to_the_map(tmp_path: Path) -> None:
    map_path = tmp_path / "flat.npz"
    map_path.write_bytes(b"")
    saved = save_places(
        map_path, {"kitchen": Place("kitchen", -1.5, 2.0, 90.0), "dock": Place("dock", 0.0, 0.0)}
    )
    assert saved == tmp_path / "flat.places.json"
    loaded = load_places(map_path)
    assert loaded["kitchen"].xy == (-1.5, 2.0)
    assert loaded["kitchen"].theta == pytest.approx(math.pi / 2)
    assert loaded["dock"].theta is None


def test_goal_resolves_by_name_or_numbers_and_names_the_known_places(tmp_path: Path) -> None:
    map_path = tmp_path / "flat.npz"
    map_path.write_bytes(b"")
    save_places(map_path, {"kitchen": Place("kitchen", -1.5, 2.0)})
    assert resolve_goal(["-2.0", "0.5"], map_path) == ((-2.0, 0.5), None)
    xy, place = resolve_goal(["kitchen"], map_path)
    assert xy == (-1.5, 2.0) and place is not None and place.name == "kitchen"
    with pytest.raises(ValueError, match="kitchen"):
        resolve_goal(["bedroom"], map_path)
    assert load_places(tmp_path / "other.npz") == {}
