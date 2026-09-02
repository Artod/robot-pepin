"""Lightweight latency statistics for control loops and bus transactions.

Every link in this robot (wifi bridge, serial bus, sensor streams) has
latency that matters for control quality. Trackers are cheap enough to
wrap every transaction and summarise on demand or at shutdown.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencySummary:
    """Rolling-window statistics in milliseconds."""

    name: str
    count: int
    median_ms: float
    p95_ms: float
    max_ms: float

    def __str__(self) -> str:
        """One-line rendering for logs and shutdown reports."""
        return (
            f"{self.name}: n={self.count} median={self.median_ms:.1f}ms "
            f"p95={self.p95_ms:.1f}ms max={self.max_ms:.1f}ms"
        )


class LatencyTracker:
    """Records durations of an operation over a rolling window."""

    def __init__(self, name: str, window: int = 512) -> None:
        """``name`` labels the link being measured; ``window`` is how many recent
        samples the statistics are computed over (older ones only feed the count)."""
        self.name = name
        self._samples: deque[float] = deque(maxlen=window)
        self._total = 0

    def add(self, seconds: float) -> None:
        """Record one duration, in seconds."""
        self._samples.append(seconds)
        self._total += 1

    @contextmanager
    def measure(self) -> Iterator[None]:
        """Time the enclosed block on the performance counter and record it, exception or not."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(time.perf_counter() - start)

    @property
    def count(self) -> int:
        """Total measurements, including those that fell out of the window."""
        return self._total

    def summary(self) -> LatencySummary:
        """Median, p95 and max over the current window, in milliseconds; zeros if unused."""
        if not self._samples:
            return LatencySummary(self.name, 0, 0.0, 0.0, 0.0)
        ordered = sorted(self._samples)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        return LatencySummary(
            name=self.name,
            count=self._total,
            median_ms=statistics.median(ordered) * 1000,
            p95_ms=p95 * 1000,
            max_ms=ordered[-1] * 1000,
        )
