"""Run-length coding of a costmap-shaped grid."""

from __future__ import annotations

import pytest

from pepin.mapcache import run_length_decode, run_length_encode


def test_the_encoding_is_lossless_and_refuses_nonsense() -> None:
    assert run_length_encode([0, 0, 0, 7, 7, -1]) == [0, 3, 7, 2, -1, 1]
    for cells in ([], [1], [0, 0, 0], [-1, 0, 100, 100, 1, 1, -1]):
        assert run_length_decode(run_length_encode(cells)) == cells
    with pytest.raises(ValueError, match="pairs of value and count"):
        run_length_decode([0, 3, 7])
    with pytest.raises(ValueError, match="is not a run"):
        run_length_decode([0, -3])
