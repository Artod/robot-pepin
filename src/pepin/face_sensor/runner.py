"""The face sensor's loops: live at its own rate on the newest frame, or over a recording sampled
by its stamps; both publish a line a second and keep the stage times for the summary.

Live, a turn starts every ``1 / rate_hz`` seconds and takes the newest frame the source has
(:class:`pepin.face_sensor.frames.Newest`); a turn that overruns starts the next one at once on
whatever frame is newest then, so a slow model lowers the rate and never builds a queue. A
recording is thinned to the same rate by its own stamps (:func:`pepin.face_sensor.frames.sample`)
and every kept frame is processed, so the outputs do not depend on the laptop's speed.
"""

from __future__ import annotations

import resource
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from pepin.face_sensor.config import FaceSensorConfig
from pepin.face_sensor.frames import FrameSource, Newest, sample
from pepin.face_sensor.pipeline import STAGES, FacePipeline, FaceState, FrameResult, percentiles
from pepin.face_sensor.publish import Publisher, report_line

Report = Callable[[str], None]


def peak_rss_mb() -> float:
    """This process's peak resident memory in MB (``ru_maxrss`` is bytes on macOS, KiB on
    Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak * 1024 / 1e6


class Stats:
    """Every processed frame's stage times, the frames with a face, and the rate by stamps."""

    def __init__(self) -> None:
        """Nothing counted yet."""
        self.ms: dict[str, list[float]] = {stage: [] for stage in STAGES}
        self.frames = 0
        self.with_face = 0
        self.skipped = 0
        self.started = time.monotonic()
        self._stamps: deque[float] = deque(maxlen=256)
        self._since_report: list[dict[str, float]] = []

    def add(self, result: FrameResult) -> None:
        """Count one processed frame."""
        self.frames += 1
        self.with_face += result.best is not None
        for stage, value in result.ms.items():
            self.ms.setdefault(stage, []).append(value)
        self._stamps.append(result.stamp)
        self._since_report.append(result.ms)

    def fps(self, now: float, window_s: float = 2.0) -> float:
        """Processed frames per second over the last ``window_s`` of frame stamps."""
        return sum(1 for t in self._stamps if now - window_s < t <= now) / window_s

    def recent_ms(self) -> dict[str, float]:
        """The mean time of each stage over the frames since the last call."""
        recent, self._since_report = self._since_report, []
        out: dict[str, float] = {}
        for stage in STAGES:
            values = [ms[stage] for ms in recent if stage in ms]
            if values:
                out[stage] = sum(values) / len(values)
        return out

    def summary(self) -> dict[str, Any]:
        """Per-stage p50/p90/mean/max, counts, the achieved rate and the peak RAM."""
        wall = time.monotonic() - self.started
        busy_s = sum(self.ms.get("total", [])) / 1e3
        return {
            "frames": self.frames,
            "frames_with_face": self.with_face,
            "frames_skipped": self.skipped,
            "wall_s": round(wall, 1),
            "fps": round(self.frames / wall, 2) if wall > 0 else 0.0,
            "capacity_fps": round(self.frames / busy_s, 1) if busy_s > 0 else 0.0,
            "ms": {stage: percentiles(values) for stage, values in self.ms.items() if values},
            "rss_peak_mb": round(peak_rss_mb(), 1),
        }


def _publish(
    state: FaceState,
    stats: Stats,
    publisher: Publisher | None,
    report: Report | None,
) -> dict[str, Any] | None:
    """One line out (file and terminal) from the state as it is now."""
    last = state.last
    if last is None:
        return None
    record = state.record(stats.fps(last.stamp))
    if record is None:
        return None
    ms = stats.recent_ms()
    if publisher is not None:
        publisher.publish(record, ms)
    if report is not None:
        report(report_line(record, ms))
    return record


def run_live(
    source: FrameSource,
    pipeline: FacePipeline,
    state: FaceState,
    cfg: FaceSensorConfig,
    *,
    publisher: Publisher | None,
    stop: threading.Event,
    report: Report | None = print,
    clock: Callable[[], float] = time.monotonic,
) -> Stats:
    """Run on a live source until ``stop`` is set or the source ends."""
    stats = Stats()
    newest = Newest(source)
    period = 1.0 / cfg.rate_hz
    publish_period = 1.0 / cfg.publish_hz
    next_turn = clock()
    next_publish = next_turn + publish_period
    try:
        while not stop.is_set():
            wait = next_turn - clock()
            if wait > 0 and stop.wait(wait):
                break
            frame = newest.take(timeout=period)
            if frame is not None:
                result = pipeline.process(frame)
                state.update(result)
                stats.add(result)
            next_turn = max(next_turn + period, clock())
            if clock() >= next_publish:
                if _publish(state, stats, publisher, report) is None and report is not None:
                    report("face: no frame from the source yet")
                next_publish = max(next_publish + publish_period, clock())
            if newest.ended and frame is None:
                break
    finally:
        newest.close()
        stats.skipped = newest.skipped
    return stats


def run_recording(
    source: FrameSource,
    pipeline: FacePipeline,
    state: FaceState,
    cfg: FaceSensorConfig,
    *,
    publisher: Publisher | None,
    report: Report | None = None,
    on_result: Callable[[FrameResult], None] | None = None,
    stop: threading.Event | None = None,
) -> Stats:
    """Run over a recording sampled to ``rate_hz`` by its stamps, publishing a line per second
    of the recording's own time."""
    stats = Stats()
    publish_period = 1.0 / cfg.publish_hz
    next_publish: float | None = None
    try:
        for frame in sample(source.frames(), cfg.rate_hz):
            if stop is not None and stop.is_set():
                break
            result = pipeline.process(frame)
            state.update(result)
            stats.add(result)
            if on_result is not None:
                on_result(result)
            if next_publish is None:
                next_publish = frame.stamp + publish_period
            if frame.stamp >= next_publish:
                _publish(state, stats, publisher, report)
                next_publish = max(next_publish + publish_period, frame.stamp)
    finally:
        source.close()
    return stats
