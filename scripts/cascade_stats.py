"""Compare voice-cascade sessions in one fixed table: brain, voice, latency per stage, cost.

Reads the events.jsonl of each session directory (data/voice/<day>/cascade_<HHMMSS>/) and prints one
row per session so combinations can be set side by side: the person's turns, time to the robot's
first sound (p50 / p90, counted from the person's last loud frame, the closing silence included),
the voice process's footprint in MB at the first and the last turn (a climbing pair is a leak),
the stage medians (Whisper, the brain's first token, the first sentence, the TTS's first chunk),
the turns that called a tool and their first token, the barge-ins, and the price per reply (the
brain's calls and a paid voice's sentences). With brain.speculative on, the row ends with the
calls asked before the close: asked/kept/dropped and the cents the dropped ones cost.

    uv run python scripts/cascade_stats.py data/voice/20261011 [data/voice/20261010 ...]
    uv run python scripts/cascade_stats.py data/voice/20261011/cascade_141459 -v  # every turn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _p(values: list[float], q: float) -> float | None:
    """The q-quantile of values (nearest rank), None when there are none."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _fmt(value: float | None, width: int = 5) -> str:
    return f"{value:>{width}.0f}" if value is not None else " " * (width - 1) + "-"


def _events(directory: Path) -> list[dict[str, Any]]:
    path = directory / "events.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


UNANSWERED = ("merged", "speculative")  # dropped for the turn after it: no turn of their own


def session_row(directory: Path) -> dict[str, Any] | None:
    """One session's numbers, or None when it has no person turns. A merged fragment is part
    of the turn after it, a reply asked early on words the close did not bring answered none;
    the cost is everything the session paid: every model call (the robot's own reports, the
    fragments and the dropped early calls included) and every sentence of a paid voice."""
    events = _events(directory)
    rows = [e for e in events if e.get("event") == "turn"]
    turns = [
        e
        for e in rows
        if e.get("kind", "person") == "person" and e.get("outcome") not in UNANSWERED
    ]
    if not turns:
        return None
    opened = next((e for e in events if e.get("event") == "open"), {})
    heard = [e for e in events if e.get("event") == "heard" and not e.get("noise")]
    tools = {e.get("turn") for e in events if e.get("event") == "tool_call"}
    barge = [e for e in events if e.get("event") == "barge_in"]
    engines = {str(e.get("engine", "")).split(":")[0] for e in heard if e.get("engine")}

    def ms(turn: dict[str, Any], key: str) -> float | None:
        value = (turn.get("ms") or {}).get(key)
        return float(value) if value is not None else None

    def step(turn: dict[str, Any], key: str) -> float | None:
        value = (turn.get("steps") or {}).get(key)
        return float(value) if value is not None else None

    plain = [t for t in turns if t.get("turn") not in tools]
    tooled = [t for t in turns if t.get("turn") in tools]
    sound = [v for t in plain if (v := ms(t, "speaker_start")) is not None]
    sound_tools = [v for t in tooled if (v := ms(t, "speaker_start")) is not None]
    usd = sum(float((t.get("usage") or {}).get("usd") or 0.0) for t in rows)
    usd += sum(float(e.get("usd") or 0.0) for e in events if e.get("event") == "voice_bill")
    cache_read = sum(int((t.get("usage") or {}).get("cache_read") or 0) for t in rows)
    inputs = sum(
        int((t.get("usage") or {}).get(k) or 0)
        for t in rows
        for k in ("input", "cache_read", "cache_write")
    )
    whisper = [w for e in heard if (w := e.get("whisper_ms")) is not None and w > 20]
    early = [e for e in events if e.get("event") == "speculative"]
    dropped = [t for t in rows if t.get("outcome") == "speculative"]
    mbs = [int(t["mb"]) for t in rows if t.get("mb") is not None]  # the process on the scales
    return {
        "session": directory.name.replace("cascade_", ""),
        "day": directory.parent.name,
        "brain": opened.get("brain") or opened.get("model", "?"),
        "voice": opened.get("voice", "?"),
        "ears": "+".join(sorted(engines)) or "?",
        "barge_in": (opened.get("flags") or {}).get("barge_in"),
        "turns": len(turns),
        "sound_p50": _p(sound, 0.5),
        "sound_p90": _p(sound, 0.9),
        "whisper": _p(whisper, 0.5),
        "ttft": _p([v for t in plain if (v := step(t, "brain_ttft")) is not None], 0.5),
        "sentence": _p([v for t in plain if (v := step(t, "sentence")) is not None], 0.5),
        "tts": _p([v for t in turns if (v := step(t, "tts")) is not None], 0.5),
        "tool_turns": len(tooled),
        "tool_sound_p50": _p(sound_tools, 0.5),
        "barge_ins": len(barge),
        "interrupted": sum(1 for t in turns if t.get("outcome") == "interrupted"),
        "cents_per_reply": 100 * usd / len(turns),
        "cents_total": 100 * usd,
        "cache_share": cache_read / inputs if inputs else None,
        "early": (
            sum(e.get("state") == "asked" for e in early),
            sum(e.get("state") == "kept" for e in early),
            sum(e.get("state") == "dropped" for e in early),
            100 * sum(float((t.get("usage") or {}).get("usd") or 0.0) for t in dropped),
        ),
        "mb": (mbs[0], mbs[-1]) if mbs else None,
        "turn_events": turns,
    }


HEADER = (
    "session        brain     voice        ears   bi  turns | to sound p50/p90 |"
    " whisper ttft sent  tts | tools  sound | barge int |  c/reply total cache | MB first  last"
)


def format_row(row: dict[str, Any]) -> str:
    bi = {True: "on ", False: "off", None: " ? "}[row["barge_in"]]
    cache = f"{100 * row['cache_share']:3.0f}%" if row["cache_share"] is not None else "   -"
    head = f"{row['day'][4:]}/{row['session']:<8} {row['brain']:<9} {row['voice']:<12}"
    stages = (
        f"{_fmt(row['whisper'], 7)} {_fmt(row['ttft'], 4)} {_fmt(row['sentence'], 4)}"
        f" {_fmt(row['tts'], 4)}"
    )
    return (
        f"{head} {row['ears']:<6} {bi} {row['turns']:>5} |"
        f" {_fmt(row['sound_p50'])} {_fmt(row['sound_p90'])} ms   | {stages}"
        f" | {row['tool_turns']:>5} {_fmt(row['tool_sound_p50'], 6)}"
        f" | {row['barge_ins']:>5} {row['interrupted']:>3}"
        f" | {row['cents_per_reply']:>7.2f} {row['cents_total']:>5.1f} {cache}"
        + _mb_cell(row["mb"])
        + _early_cell(row["early"])
    )


def _mb_cell(mb: tuple[int, int] | None) -> str:
    """`` |  2810  2905``: the voice process's footprint (MB) at the session's first and last
    turn; a steady pair clears the engines of growing, a climbing one says how fast."""
    return f" | {mb[0]:>7} {mb[1]:>5}" if mb else " |       -     -"


def _early_cell(early: tuple[int, int, int, float]) -> str:
    """`` | early 12/9/3 0.84c``: calls asked before the close, kept, dropped, the dropped
    ones' cents ("" when none was asked)."""
    asked, kept, dropped, cents = early
    return f" | early {asked}/{kept}/{dropped} {cents:.2f}c" if asked else ""


def turn_lines(row: dict[str, Any]) -> list[str]:
    """One line per turn: the stages as the loop printed them."""
    out = []
    for t in row["turn_events"]:
        m = t.get("ms") or {}
        s = t.get("steps") or {}
        usd = float((t.get("usage") or {}).get("usd") or 0.0)
        out.append(
            f"  [{t.get('turn')}] {t.get('outcome', '?'):<11}"
            f" sound {m.get('speaker_start', '-'):>5} | whisper {s.get('whisper', '-'):>4}"
            f" ttft {s.get('brain_ttft', '-'):>5} sent {s.get('sentence', '-'):>4}"
            f" tts {s.get('tts', '-'):>4} | {t.get('mb') or '-':>5} MB"
            f" | {100 * usd:.2f} c | {str(t.get('user', ''))[:50]}"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="+", type=Path, help="a day's or a session's directory")
    ap.add_argument("--min-turns", type=int, default=5, help="skip sessions shorter than this")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every turn")
    args = ap.parse_args()
    directories: list[Path] = []
    for path in args.paths:
        if (path / "events.jsonl").exists():
            directories.append(path)
        else:
            directories.extend(sorted(p for p in path.glob("cascade_*") if p.is_dir()))
    print(HEADER)
    for directory in directories:
        row = session_row(directory)
        if row is None or row["turns"] < args.min_turns:
            continue
        print(format_row(row))
        if args.verbose:
            print("\n".join(turn_lines(row)))


if __name__ == "__main__":
    main()
