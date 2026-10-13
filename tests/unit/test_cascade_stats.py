"""scripts/cascade_stats.py over a session's events: what counts as a turn, and what it cost."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]


def load_stats() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cascade_stats", REPO / "scripts/cascade_stats.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def turn(n: int, outcome: str, usd: float, kind: str = "person", **more: Any) -> dict[str, Any]:
    usage = {"input": 10, "cache_read": 90, "cache_write": 0, "output": 5, "usd": usd}
    return {"event": "turn", "turn": n, "kind": kind, "outcome": outcome, "usage": usage, **more}


def test_a_merged_fragment_is_no_turn_and_every_cost_of_the_session_counts(tmp_path: Path) -> None:
    session = tmp_path / "20261011" / "cascade_120000"
    session.mkdir(parents=True)
    events = [
        {"event": "open", "brain": "opus", "voice": "gemini-lite", "flags": {"barge_in": True}},
        turn(1, "merged", 0.001),  # the person went on: part of turn 2
        turn(2, "committed", 0.002, ms={"speaker_start": 900}),
        {"event": "voice_bill", "turn": 2, "usd": 0.0005},
        turn(3, "committed", 0.002, kind="robot"),  # the drive's report
        {"event": "voice_bill", "turn": 3, "usd": 0.0005},
    ]
    (session / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    row = load_stats().session_row(session)
    assert row is not None and row["turns"] == 1 and row["sound_p50"] == 900
    assert row["cents_total"] == pytest.approx(100 * (0.001 + 0.002 + 0.002 + 0.0005 + 0.0005))
    assert row["cents_per_reply"] == row["cents_total"]  # one reply of the person's
    assert row["cache_share"] == pytest.approx(0.9)


def test_a_reply_asked_early_and_dropped_is_no_turn_but_its_cost_counts(tmp_path: Path) -> None:
    session = tmp_path / "20261012" / "cascade_120000"
    session.mkdir(parents=True)
    events = [
        {"event": "open", "brain": "opus", "voice": "qwen3-1.7b", "flags": {"barge_in": True}},
        {"event": "speculative", "state": "asked"},
        {"event": "speculative", "state": "dropped", "why": "the person went on"},
        turn(1, "speculative", 0.003),  # asked on words the close did not bring
        turn(2, "committed", 0.002, ms={"speaker_start": 700}),
        {"event": "speculative", "state": "asked"},
        {"event": "speculative", "state": "kept"},
        turn(3, "committed", 0.002, ms={"speaker_start": 600}),
    ]
    (session / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    stats = load_stats()
    row = stats.session_row(session)
    assert row is not None and row["turns"] == 2
    assert row["cents_total"] == pytest.approx(100 * 0.007)
    assert row["early"] == (2, 1, 1, pytest.approx(0.3))
    assert stats.format_row(row).endswith(" | early 2/1/1 0.30c")


def test_the_footprint_column_is_the_first_and_the_last_turn_on_the_scales(tmp_path: Path) -> None:
    session = tmp_path / "20261012" / "cascade_130000"
    session.mkdir(parents=True)
    events = [
        {"event": "open", "brain": "opus", "voice": "qwen3-1.7b", "flags": {"barge_in": True}},
        turn(1, "committed", 0.002, ms={"speaker_start": 700}, mb=2810),
        turn(2, "committed", 0.002, ms={"speaker_start": 650}),  # a line from before the cell
        turn(3, "committed", 0.002, ms={"speaker_start": 600}, mb=2905),
    ]
    (session / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    stats = load_stats()
    row = stats.session_row(session)
    assert row is not None and row["mb"] == (2810, 2905)
    assert "MB first  last" in stats.HEADER
    assert stats.format_row(row).endswith(" |    2810  2905")
    assert any(" 2905 MB" in line for line in stats.turn_lines(row))
    assert any("    - MB" in line for line in stats.turn_lines(row))
    bare = tmp_path / "20261012" / "cascade_140000"
    bare.mkdir(parents=True)
    (bare / "events.jsonl").write_text(json.dumps(events[0]) + "\n" + json.dumps(events[2]) + "\n")
    old = stats.session_row(bare)
    assert (
        old is not None and old["mb"] is None and stats.format_row(old).endswith(" |       -     -")
    )
