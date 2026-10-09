"""The face sensor's output: ``data/face/latest.json`` rewritten atomically every second, the day's
``data/face/<YYYY-MM-DD>.jsonl``, a one-line terminal report, and :func:`latest` for readers.

Standard library only, so the voice loop imports :func:`latest` without numpy or the models.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LATEST = "latest.json"
STALE_S = 3.0
REPO = Path(__file__).resolve().parents[3]
DEFAULT_DIR = REPO / "data" / "face"


def latest(
    path: str | Path | None = None, *, max_age_s: float = STALE_S, now: float | None = None
) -> dict[str, Any] | None:
    """The face sensor's newest line, or ``None`` when there is none, it does not parse, or its
    ``stamp`` (the frame's capture time) is more than ``max_age_s`` old."""
    file = Path(path) if path is not None else DEFAULT_DIR / LATEST
    try:
        record = json.loads(file.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    stamp = record.get("stamp")
    if not isinstance(stamp, (int, float)):
        return None
    if (time.time() if now is None else now) - float(stamp) > max_age_s:
        return None
    return record


class Publisher:
    """Writes each line to ``latest.json`` (temporary file, then rename) and appends it with its
    stage times to the day's JSONL; the day is the local date of the line's stamp."""

    def __init__(self, folder: str | Path) -> None:
        """Write into ``folder`` (created on the first line)."""
        self.folder = Path(folder)
        self.lines = 0

    def publish(self, record: Mapping[str, Any], ms: Mapping[str, float] | None = None) -> None:
        """One line out."""
        self.folder.mkdir(parents=True, exist_ok=True)
        text = json.dumps(record, separators=(", ", ": "))
        tmp = self.folder / f".{LATEST}.{os.getpid()}.tmp"
        tmp.write_text(text + "\n")
        os.replace(tmp, self.folder / LATEST)
        day = time.strftime("%Y-%m-%d", time.localtime(float(record.get("stamp", time.time()))))
        logged = dict(record)
        if ms:
            logged["ms"] = {k: round(v, 1) for k, v in ms.items()}
        with (self.folder / f"{day}.jsonl").open("a") as log:
            log.write(json.dumps(logged, separators=(",", ":")) + "\n")
        self.lines += 1


def report_line(record: Mapping[str, Any], ms: Mapping[str, float]) -> str:
    """The 1 Hz terminal line: who, size and bearing, expression, attention, quality, rate and
    the stage milliseconds."""
    clock = time.strftime("%H:%M:%S", time.localtime(float(record["stamp"])))
    stages = " ".join(f"{k} {v:.0f}" for k, v in ms.items())
    head = f"face {clock} {record.get('fps', 0):.1f} fps n={record.get('faces', 0)}"
    if record.get("face_px") is None:
        return f"{head} | no face | ms {stages}"
    who = record.get("who") or {}
    parts = [f"{who.get('id') or '?'} cos {who.get('cos', 0):.2f} {who.get('votes', '-')}"]
    pan_tilt = record.get("bearing_deg") or {}
    pan, tilt = pan_tilt.get("pan", 0.0), pan_tilt.get("tilt", 0.0)
    parts.append(f"{record['face_px']} px pan {pan:+.1f} tilt {tilt:+.1f}")
    expr = record.get("expr")
    parts.append(
        f"{expr['top']} {expr['p_3s']:.2f} V{expr['valence_3s']:+.2f} A{expr['arousal_3s']:+.2f}"
        if expr
        else "expr -"
    )
    att = record.get("attention") or {}
    parts.append(f"look {_yn(att.get('looking'))} talk {_yn(att.get('talking'))}")
    q = record.get("quality") or {}
    parts.append(
        f"{q.get('gate', '-')} sharp {q.get('sharpness', 0)} bright {q.get('brightness', 0)}"
    )
    return f"{head} | " + " | ".join(parts) + f" | ms {stages}"


def _yn(value: object) -> str:
    return "-" if value is None else ("yes" if value else "no")
