"""The map the board keeps for itself: the encoding, the atomic write, and the refusals.

The owner's rule is one map — the volume — and a board that can only start from a pgm served out
of a file has two. This is the file that makes the pgm unnecessary, so what it must never do is
hand back half a map: a tracker matching a truncated grid believes its own answer.

The expected NAME is a literal here on purpose: a test that re-derives the path with the code's own
expression proves only that the code agrees with itself (an export bug slipped through that way on
2026-09-17).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest

from pepin.mapcache import (
    CACHE_NAME,
    MapCache,
    read_cache,
    run_length_decode,
    run_length_encode,
    write_cache,
)


def cache(cells: list[int] | None = None, **overrides: object) -> MapCache:
    """A cache of a 2x3 grid, or of ``cells``."""
    cells = [-1, -1, 0, 0, 100, 100] if cells is None else cells
    fields: dict[str, object] = {
        "cells": cells,
        "width": 2,
        "height": len(cells) // 2,
        "resolution_m": 0.05,
        "origin_xy": (-18.53, -4.38),
        "map_id": "239x215@-18.53,-4.38",
        "digest": "map_lidar#f5169d80",
        "stamp": 1_789_700_000.0,
        "source": "/map_lidar",
        "identity": "flat3-a1b2",
    }
    fields.update(overrides)
    return MapCache(**fields)  # type: ignore[arg-type]


def test_the_file_is_called_map_cache_json_and_nothing_else() -> None:
    """The literal, so a reader of a board's card knows what to look for."""
    assert CACHE_NAME == "map_cache.json"


def test_the_encoding_is_lossless_and_refuses_nonsense() -> None:
    assert run_length_encode([0, 0, 0, 7, 7, -1]) == [0, 3, 7, 2, -1, 1]
    for cells in ([], [1], [0, 0, 0], [-1, 0, 100, 100, 1, 1, -1]):
        assert run_length_decode(run_length_encode(cells)) == cells
    with pytest.raises(ValueError, match="pairs of value and count"):
        run_length_decode([0, 3, 7])
    with pytest.raises(ValueError, match="is not a run"):
        run_length_decode([0, -3])


def test_a_cache_written_here_is_the_cache_read_back(tmp_path: Path) -> None:
    written = write_cache(tmp_path, cache())
    assert written == tmp_path / "map_cache.json", written
    back = read_cache(tmp_path)
    assert back == cache()
    assert back is not None and back.source == "/map_lidar" and back.identity == "flat3-a1b2"
    # ...and the cells really are run-length encoded on the card, not 51385 numbers.
    on_disk = json.loads(written.read_text())
    assert on_disk["encoding"] == "rle" and on_disk["cells"] == [-1, 2, 0, 2, 100, 2]


def test_no_cache_is_no_answer_and_not_an_error(tmp_path: Path) -> None:
    """A board that has never adopted a map has no cache: the caller says so loudly, and it is the
    caller's business, not an exception here."""
    assert read_cache(tmp_path) is None


def test_a_corrupt_cache_is_refused_whole_and_never_half_loaded(tmp_path: Path) -> None:
    """Half a map is worse than none: the tracker would match against it and believe the answer."""
    write_cache(tmp_path, cache())
    path = tmp_path / CACHE_NAME
    whole = path.read_text()
    path.write_text(whole[: len(whole) // 2])  # a write cut off by a power failure
    with pytest.raises(ValueError):
        read_cache(tmp_path)
    path.write_text(json.dumps({"version": 99, "cells": [], "width": 1, "height": 1}))
    with pytest.raises(ValueError, match="version"):
        read_cache(tmp_path)
    raw = json.loads(whole)
    raw["cells"] = [0, 3]  # three cells for a 2x3 grid
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="do not fill"):
        read_cache(tmp_path)
    raw = json.loads(whole)
    raw["encoding"] = "raw"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="encoding"):
        read_cache(tmp_path)


def test_a_killed_write_leaves_the_previous_cache_whole(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The reason for the temporary file and os.replace: the board must never lose the map it has
    to a power cut in the middle of writing a newer one."""
    write_cache(tmp_path, cache())
    first = read_cache(tmp_path)

    def killed(src: object, dst: object) -> None:
        raise KeyboardInterrupt("the power went")

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        write_cache(tmp_path, cache(cells=[0] * 6, map_id="another"))
    assert read_cache(tmp_path) == first, "the old cache is still whole and still the one"
    assert (tmp_path / f".{CACHE_NAME}.tmp").exists(), (
        "the half-written one is beside it, not on it"
    )


def test_the_phrase_tells_a_cold_boot_from_a_live_map() -> None:
    """The words the report line carries, so nobody reads a cache as a live map."""
    now = 1_789_700_000.0 + 7.4 * 3600.0
    said = cache().phrase(now)
    assert "the cache of 239x215@-18.53,-4.38 (flat3-a1b2) from /map_lidar" in said
    assert "written 7.4 h ago" in said
    assert cache(stamp=0.0).age_s() == math.inf and "unknown age" in cache(stamp=0.0).phrase()
    assert cache(stamp=time.time()).age_s() < 1.0
