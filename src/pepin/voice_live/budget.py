"""What a session costs (an estimate) and whether another may open.

The estimate is the server's own token counts (:class:`Usage`, summed over its usage reports:
one per model pass, each counting the whole context that pass read, which is how the Live API
bills) plus every second of mic audio streamed, priced once as audio in (Google Billing: silence
in an open stream is billed too). The context holds only the person's speech the server's VAD
took, not the stream: on 2026-10-05 a session that streamed 586 s carried 1926 audio tokens
(77 s) in its last pass, and the earlier estimate, which re-billed every streamed second at
every turn, read 3.1x Google's counts (scratch/voice_1005/spend.py). A pass still in flight is
not counted until its report arrives (a few seconds, ~0.003 USD for a fresh session).

:class:`Ledger` keeps every session's latest estimate in one JSON-lines file, written at the
open, every few seconds and at the close, so a crash mid-session still counts what it spent; a
new process reads it back. :meth:`Ledger.refusal` says why a session may not open.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pepin.voice_live.config import LiveConfig, Prices

logger = logging.getLogger(__name__)


@dataclass
class Usage:
    """Tokens billed, by direction and modality, summed over the server's reports."""

    audio_in: int = 0
    text_in: int = 0
    audio_out: int = 0
    text_out: int = 0
    reports: int = 0

    def add(self, other: Usage) -> None:
        """Add one report."""
        self.audio_in += other.audio_in
        self.text_in += other.text_in
        self.audio_out += other.audio_out
        self.text_out += other.text_out
        self.reports += other.reports

    def usd(self, prices: Prices) -> float:
        """The estimated price in USD."""
        return (
            self.audio_in * prices.audio_in
            + self.text_in * prices.text_in
            + self.audio_out * prices.audio_out
            + self.text_out * prices.text_out
        ) / 1e6


@dataclass
class SessionCost:
    """One session's running estimate: the mic seconds streamed, the speech received, the
    server's counts."""

    prices: Prices
    audio_in_s: float = 0.0  # streamed to Live
    audio_out_s: float = 0.0  # spoken by the model
    usage: Usage = field(default_factory=Usage)
    carried_s: float = 0.0  # context brought in by a resumed session (for the resume store)

    @property
    def stream_usd(self) -> float:
        """Every second streamed, billed once as audio in."""
        p = self.prices
        return self.audio_in_s * p.audio_tokens_per_s * p.audio_in / 1e6

    @property
    def usage_usd(self) -> float:
        """The estimate from the server's token counts."""
        return self.usage.usd(self.prices)

    @property
    def usd(self) -> float:
        """The session's estimate: the passes as the server counted them, plus the stream."""
        return self.usage_usd + self.stream_usd


class Ledger:
    """Sessions and their estimates on disk; the caps' memory across restarts."""

    def __init__(self, path: Path, config: LiveConfig, clock: Any = time.time) -> None:
        """``path``: the JSON-lines file (created on the first write)."""
        self.path = path
        self.config = config
        self._clock = clock
        self._sessions: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    self._sessions[str(row["sid"])] = row
                except (ValueError, KeyError, TypeError):
                    logger.warning("ledger: unreadable line skipped: %s", line[:80])

    def today_usd(self, now: float | None = None) -> float:
        """The estimates of the sessions opened today (local day), summed."""
        day = time.strftime("%Y%m%d", time.localtime(self._clock() if now is None else now))
        return sum(
            float(r.get("usd", 0.0))
            for r in self._sessions.values()
            if time.strftime("%Y%m%d", time.localtime(float(r["t_open"]))) == day
        )

    def opened_last_hour(self, now: float | None = None) -> int:
        """Sessions opened in the last 3600 s."""
        t = self._clock() if now is None else now
        return sum(1 for r in self._sessions.values() if t - float(r["t_open"]) < 3600.0)

    def refusal(self, now: float | None = None) -> str | None:
        """Why a new session may not open now, or None when it may."""
        cfg = self.config
        hour = self.opened_last_hour(now)
        if hour >= cfg.max_sessions_per_hour:
            return f"{hour} sessions in the last hour (cap {cfg.max_sessions_per_hour})"
        spent = self.today_usd(now)
        if spent >= cfg.daily_budget_usd:
            return (
                f"today's estimate {spent * cfg.prices.usd_to_cad:.2f} CAD reached the daily"
                f" budget {cfg.daily_budget_cad:.2f} CAD"
            )
        return None

    def over_budget(self, sid: str, usd: float, now: float | None = None) -> bool:
        """Whether today's estimate with this session at ``usd`` reaches the daily budget."""
        others = self.today_usd(now) - float(self._sessions.get(sid, {}).get("usd", 0.0))
        return others + usd >= self.config.daily_budget_usd

    def record(self, sid: str, t_open: float, usd: float, *, closed: bool = False) -> None:
        """The session's latest estimate (appended; the last line of a sid wins)."""
        row = {
            "sid": sid,
            "t_open": round(t_open, 3),
            "t": round(self._clock(), 3),
            "usd": round(usd, 6),
            "closed": closed,
        }
        self._sessions[sid] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
