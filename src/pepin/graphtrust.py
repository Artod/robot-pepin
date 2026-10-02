"""RTAB-Map's statistics as a report line reads them: :func:`stat` and the one key it is asked
for, :data:`HIGHEST_HYPOTHESIS`.

The graph-word trust that compared RTAB-Map's words with the board tracker's pose
(``Agreement``) is on the tag alt/tracker-2026-09-22.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["HIGHEST_HYPOTHESIS", "stat"]

# How close RTAB-Map came to recognising a place on its last update, by the name it publishes the
# statistic under (rtabmap/Statistics.h). Read for the report line and nothing else: a graph that
# has said nothing for a quarter of an hour is a different thing from one that is nearly there, and
# without this the two look identical from outside. The key carries a trailing unit segment, which
# :func:`stat` tolerates.
HIGHEST_HYPOTHESIS = "Loop/Highest_hypothesis_value"


def stat(stats: Mapping[str, float], name: str) -> float | None:
    """One RTAB-Map statistic by its name, tolerating the unit segment the key carries
    (``Loop/Id`` finds ``Loop/Id/``); ``None`` when this message does not carry it."""
    if name in stats:
        return stats[name]
    for key, value in stats.items():
        if key.rstrip("/") == name or key.startswith(f"{name}/"):
            return value
    return None
