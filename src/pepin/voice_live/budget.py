"""What a session costs (an estimate) and whether another may open.

Two estimates, the larger one counts. The server's own token counts (:class:`Usage`, summed over
its usage reports) are the bill's basis; before the first report arrives, and in case the
reports undercount, :func:`audio_estimate` prices the audio itself: every turn re-bills the
whole context window (the Live API's rule), which grows with the audio heard and spoken until
compression caps it.

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

TEXT_OVERHEAD_TOKENS = 1500  # system instruction + tool declarations, in the context every turn


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


def audio_estimate(
    turn_contexts_s: list[float],
    audio_out_s: float,
    prices: Prices,
    *,
    context_cap_tokens: int,
    overhead_tokens: int = TEXT_OVERHEAD_TOKENS,
) -> float:
    """USD from the audio alone: each turn bills the context it saw (``turn_contexts_s``: the
    seconds of audio, both ways, in the context at each turn; capped by compression) as input,
    plus the spoken audio as output; the text overhead is billed each turn too."""
    tokens_per_s = prices.audio_tokens_per_s
    audio_in = sum(min(s * tokens_per_s, context_cap_tokens) for s in turn_contexts_s)
    text_in = overhead_tokens * max(1, len(turn_contexts_s))
    return (
        audio_in * prices.audio_in
        + text_in * prices.text_in
        + audio_out_s * tokens_per_s * prices.audio_out
    ) / 1e6


@dataclass
class SessionCost:
    """One session's running estimate: what was streamed, the turns, the server's counts."""

    prices: Prices
    context_cap_tokens: int
    audio_in_s: float = 0.0
    audio_out_s: float = 0.0
    turn_contexts_s: list[float] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    carried_s: float = 0.0  # context brought in by a resumed session (unknown: its last estimate)

    def turn(self) -> None:
        """A model turn happened: it billed the context as it stands."""
        self.turn_contexts_s.append(self.carried_s + self.audio_in_s + self.audio_out_s)

    @property
    def audio_usd(self) -> float:
        """The estimate from the audio."""
        return audio_estimate(
            self.turn_contexts_s,
            self.audio_out_s,
            self.prices,
            context_cap_tokens=self.context_cap_tokens,
        )

    @property
    def usage_usd(self) -> float:
        """The estimate from the server's token counts."""
        return self.usage.usd(self.prices)

    @property
    def usd(self) -> float:
        """The larger of the two: a cap errs on the side of the wallet."""
        return max(self.audio_usd, self.usage_usd)


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
