"""Run-length coding of a costmap-shaped grid: runs of one value, so 51385 cells of this flat's
map are 195 kB of raw JSON and 16 kB encoded (scratch/costmap_rle_cost.py). The tape's global
costmap rows use it (pepin.tape_rows), and the scratch instruments decode them.

The board tracker's map cache that lived here is on the tag alt/tracker-2026-09-22.
"""

from __future__ import annotations

__all__ = ["run_length_decode", "run_length_encode"]


def run_length_encode(cells: list[int]) -> list[int]:
    """``[v, n, v, n, ...]``: each value of ``cells`` and how many times it repeats, in order.

    Lossless and flat, so a reader needs no library and the record stays JSON. A map is runs of one
    value, which is why this is worth doing at all: this flat's 51385 cells go from 195 kB of JSON
    to 16 kB (scratch/costmap_rle_cost.py).
    """
    out: list[int] = []
    for value in cells:
        if out and out[-2] == value:
            out[-1] += 1
        else:
            out.extend((int(value), 1))
    return out


def run_length_decode(runs: list[int]) -> list[int]:
    """The cells back, so a reader needs to know nothing about how they were stored."""
    if len(runs) % 2:
        raise ValueError(f"a run-length list is pairs of value and count, not {len(runs)} numbers")
    out: list[int] = []
    for value, count in zip(runs[::2], runs[1::2], strict=True):
        if count < 0:
            raise ValueError(f"a run of {count} cells is not a run")
        out.extend([int(value)] * int(count))
    return out
