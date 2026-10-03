"""``config/voice_live.json``: the model, the caps that refuse a session, the prices of the cost
estimate, the session's timings and the wake gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

CONFIG_FILE = "voice_live.json"


@dataclass(frozen=True)
class Prices:
    """USD per million tokens, audio tokens per second, and the CAD the budget is kept in."""

    audio_in: float = 3.0
    audio_out: float = 12.0
    text_in: float = 0.75
    text_out: float = 4.5
    audio_tokens_per_s: float = 25.0
    usd_to_cad: float = 1.38


@dataclass(frozen=True)
class LiveConfig:
    """Everything the voice loop reads at start; see the file's notes for the reasons."""

    model: str = "gemini-3.8-live"
    voice: str = "Puck"
    max_session_s: float = 600.0
    max_sessions_per_hour: int = 20
    daily_budget_cad: float = 1.0
    prices: Prices = Prices()
    idle_close_s: float = 25.0
    resume_valid_s: float = 7000.0
    compression_trigger_tokens: int = 16000
    compression_target_tokens: int = 8000
    barge_in: bool = False
    playback_lead_s: float = 0.5
    whisper_repo: str = "mlx-community/whisper-large-v3-turbo"
    wake_silence_s: float = 0.5
    wake_max_s: float = 8.0
    keep_wake_wavs: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LiveConfig:
        """From the file's object; ``ValueError`` naming a cap out of range."""
        caps = data.get("caps", {})
        prices = data.get("prices_usd_per_mtok", {})
        session = data.get("session", {})
        wake = data.get("wake", {})
        d = cls()
        config = cls(
            model=str(data.get("model", d.model)),
            voice=str(data.get("voice", d.voice)),
            max_session_s=float(caps.get("max_session_s", d.max_session_s)),
            max_sessions_per_hour=int(caps.get("max_sessions_per_hour", d.max_sessions_per_hour)),
            daily_budget_cad=float(caps.get("daily_budget_cad", d.daily_budget_cad)),
            prices=Prices(
                audio_in=float(prices.get("audio_in", d.prices.audio_in)),
                audio_out=float(prices.get("audio_out", d.prices.audio_out)),
                text_in=float(prices.get("text_in", d.prices.text_in)),
                text_out=float(prices.get("text_out", d.prices.text_out)),
                audio_tokens_per_s=float(data.get("audio_tokens_per_s", 25.0)),
                usd_to_cad=float(data.get("usd_to_cad", d.prices.usd_to_cad)),
            ),
            idle_close_s=float(session.get("idle_close_s", d.idle_close_s)),
            resume_valid_s=float(session.get("resume_valid_s", d.resume_valid_s)),
            compression_trigger_tokens=int(
                session.get("compression_trigger_tokens", d.compression_trigger_tokens)
            ),
            compression_target_tokens=int(
                session.get("compression_target_tokens", d.compression_target_tokens)
            ),
            barge_in=bool(session.get("barge_in", d.barge_in)),
            playback_lead_s=float(session.get("playback_lead_s", d.playback_lead_s)),
            whisper_repo=str(wake.get("whisper_repo", d.whisper_repo)),
            wake_silence_s=float(wake.get("silence_s", d.wake_silence_s)),
            wake_max_s=float(wake.get("max_s", d.wake_max_s)),
            keep_wake_wavs=bool(wake.get("keep_wavs", d.keep_wake_wavs)),
        )
        config.check()
        return config

    @classmethod
    def load(cls, path: str | Path | None = None) -> LiveConfig:
        """From ``config/voice_live.json`` wherever this library runs, or ``path``."""
        if path is None:
            from pepin.deployment import config_file

            path = config_file(CONFIG_FILE)
        return cls.from_dict(json.loads(Path(path).read_text()))

    def check(self) -> None:
        """``ValueError`` when a cap is missing its point (zero, negative, absurd)."""
        if not 0 < self.max_session_s <= 900:
            raise ValueError(f"max_session_s {self.max_session_s}: 0 < s <= 900 (Live's 15 min)")
        if self.max_sessions_per_hour < 1 or self.daily_budget_cad <= 0:
            raise ValueError("max_sessions_per_hour >= 1 and daily_budget_cad > 0")
        if not 0 < self.compression_target_tokens < self.compression_trigger_tokens:
            raise ValueError("0 < compression_target_tokens < compression_trigger_tokens")
        if self.idle_close_s <= 0:
            raise ValueError("idle_close_s > 0")

    def with_overrides(self, **changes: Any) -> LiveConfig:
        """A copy with the given fields replaced (command-line overrides), checked again."""
        config = replace(self, **{k: v for k, v in changes.items() if v is not None})
        config.check()
        return config

    @property
    def daily_budget_usd(self) -> float:
        """The daily cap in USD, the currency of the prices."""
        return self.daily_budget_cad / self.prices.usd_to_cad
