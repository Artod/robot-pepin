"""One turn's timeline, the point of the cascade: every stage stamped against the moment the
person stopped speaking, printed as one line, and summed over a run as p50/p90 per stage."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# (key, short name in the line, what it is)
STAGES = (
    ("closed", "cut", "the ears closed the utterance (the end-of-turn silence)"),
    ("transcript", "text", "Whisper's words (and the tag line) ready"),
    ("first_token", "token", "the brain's first text"),
    ("first_sentence", "sent", "the brain's first whole sentence"),
    ("tts_first_audio", "tts", "the voice's first audio of the reply"),
    ("speaker_start", "play", "the reply's first sample on the speaker"),
    ("speaker_end", "end", "the reply's last sample on the speaker (or the barge-in)"),
)
STEPS = (  # the cost of each stage on its own: (name, from, to)
    ("silence", "stopped", "closed"),
    ("whisper", "closed", "transcript"),
    ("brain_ttft", "transcript", "first_token"),
    ("sentence", "first_token", "first_sentence"),
    ("tts", "first_sentence", "tts_first_audio"),
    ("pacer", "tts_first_audio", "speaker_start"),
)


@dataclass
class Timeline:
    """Stamps (laptop monotonic seconds) of one turn; ``stopped`` is the zero."""

    stopped: float
    marks: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def mark(self, stage: str, t: float) -> None:
        """Stamp ``stage`` at ``t`` unless it is stamped already (the first time wins)."""
        with self._lock:
            self.marks.setdefault(stage, t)

    def has(self, stage: str) -> bool:
        """Whether ``stage`` is stamped."""
        with self._lock:
            return stage in self.marks

    def ms(self) -> dict[str, int | None]:
        """Each stage in milliseconds after the person stopped (None: never reached)."""
        with self._lock:
            marks = dict(self.marks)
        return {
            key: round((marks[key] - self.stopped) * 1000) if key in marks else None
            for key, _, _ in STAGES
        }

    def steps(self) -> dict[str, int | None]:
        """Each stage's own duration in milliseconds."""
        with self._lock:
            marks = {"stopped": self.stopped, **self.marks}
        return {
            name: round((marks[b] - marks[a]) * 1000) if a in marks and b in marks else None
            for name, a, b in STEPS
        }

    def line(self) -> str:
        """``cut 512 | text 790 | token 1350 | ... ms`` after the person stopped."""
        ms = self.ms()
        cells = [f"{short} {_cell(ms[key])}" for key, short, _ in STAGES]
        return " | ".join(cells) + " ms"


def _cell(value: int | None) -> str:
    return "-" if value is None else str(value)


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q`` quantile (0..1) of ``values``, linearly interpolated; NaN when empty."""
    data = sorted(values)
    if not data:
        return math.nan
    position = (len(data) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return data[low] + (data[high] - data[low]) * (position - low)


def histogram(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], digits: int = 0) -> list[str]:
    """Lines of a table: per key, n, p50, p90 and max of the rows' values (None skipped)."""
    lines = [f"{'stage':<16}{'n':>4}{'p50':>8}{'p90':>8}{'max':>8}"]
    for key in keys:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        if not values:
            lines.append(f"{key:<16}{0:>4}{'-':>8}{'-':>8}{'-':>8}")
            continue
        p50, p90 = percentile(values, 0.5), percentile(values, 0.9)
        top = max(values)
        lines.append(
            f"{key:<16}{len(values):>4}{p50:>8.{digits}f}{p90:>8.{digits}f}{top:>8.{digits}f}"
        )
    return lines


def report(turns: Sequence[Mapping[str, Any]]) -> list[str]:
    """The run's latency report: cumulative stages after the person stopped, each stage's own
    cost, and the brain's tokens and price per turn."""
    ms = [t["ms"] for t in turns if t.get("ms")]
    steps = [t["steps"] for t in turns if t.get("steps")]
    usage = [t["usage"] for t in turns if t.get("usage")]
    lines = ["after the person stopped (ms):"]
    lines += histogram(ms, [key for key, _, _ in STAGES])
    lines += ["", "each stage on its own (ms):"]
    lines += histogram(steps, [name for name, _, _ in STEPS])
    eot = [t["eot"] for t in turns if t.get("eot")]
    if eot:
        by_model = sum(1 for e in eot if e["closed_by"] == "model")
        lines += ["", f"end-of-turn model ({by_model} of {len(eot)} closed by it):"]
        lines += histogram(eot, ["saved_ms", "model_ms"])
        lines += histogram(eot, ["p"], digits=2)[1:]
    if usage:
        cents = [{"usd_cents": u["usd"] * 100} for u in usage]
        lines += ["", "brain per turn (tokens, model calls; US cents):"]
        lines += histogram(usage, ["input", "cache_read", "cache_write", "output", "calls"])
        lines += histogram(cents, ["usd_cents"], digits=4)[1:]
        total = sum(u["usd"] for u in usage)
        lines.append(f"total {total * 100:.3f} US cents over {len(usage)} turns")
    return lines
