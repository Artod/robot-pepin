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
