"""``config/voice_cascade.json``: every stage of the cascade and its switches, in one file.

The file is read into :class:`CascadeConfig`; ``--set section.key=value`` on the command line
patches the file's object before it is read (:func:`apply_overrides`), so any number or flag is
switched for one run without an edit. ``brain.preset`` picks a named group of brain settings
(model, thinking mode, effort) from ``brain.presets``; keys given beside it win.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

CONFIG_FILE = "voice_cascade.json"
WAKE_MODES = ("name", "always")
THINKING_MODES = ("disabled", "between_tools", "adaptive")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
PROVIDERS = ("anthropic", "gemini")
THINKING_LEVELS = ("minimal", "low", "medium", "high")  # Gemini's
VOICE_ENGINES = ("qwen3", "say")
EMOTION_MODELS = ("audeering", "emotion2vec")
# end-of-turn models: name -> (Hugging Face repo, file); pipecat-ai/smart-turn, BSD-2-Clause
TURN_MODELS = {"smart-turn-v3.2": ("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx")}


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens: fresh input, output, cache reads and 5-minute cache writes."""

    input: float
    output: float
    cache_read: float
    cache_write: float


DEFAULT_PRICES = {
    "claude-haiku-5-5": ModelPrice(0.10, 0.50, 0.01, 0.125),
    "claude-sonnet-5-5": ModelPrice(2.00, 10.00, 0.20, 2.50),
    "claude-opus-5-5": ModelPrice(4.00, 20.00, 0.20, 5.00),
    # Gemini: implicit caching, no write premium (a cached token is billed at the cached rate)
    "gemini-3.5-flash-lite": ModelPrice(0.30, 2.50, 0.03, 0.30),
    "gemini-3.5-flash": ModelPrice(1.50, 9.00, 0.15, 1.50),
    "gemini-3.8-flash": ModelPrice(0.75, 3.75, 0.075, 0.75),
}


@dataclass(frozen=True)
class BrainConfig:
    """The talker: whose model, how it thinks, its persona, and the prompt cache."""

    provider: str = "anthropic"
    model: str = "claude-haiku-5-5"
    thinking: str | None = "disabled"  # Claude; None: the parameter is left out
    effort: str | None = "low"  # Claude; None: left out
    thinking_budget: int | None = None  # Gemini; None: left out
    thinking_level: str | None = None  # Gemini; None: left out
    max_tokens: int = 400
    persona_file: str = "voice_cascade_persona.md"
    cache: bool = True
    prewarm: bool = True
    keep_warm_s: float = 270.0  # 0: never refreshed while idle
    max_tool_rounds: int = 4
    timeout_s: float = 20.0


@dataclass(frozen=True)
class ProsodyConfig:
    """The tag line: who is assumed to speak, the baseline, the z-score that is worth a word."""

    speaker: str = "Артём"
    min_baseline: int = 5
    window: int = 200
    z: float = 2.0
    emotion_model: str | None = None


@dataclass(frozen=True)
class CascadeConfig:
    """Everything the cascade reads at start; see the file's notes for the reasons."""

    # ears
    wake: str = "name"
    silence_s: float = 0.5
    max_utterance_s: float = 15.0
    start_db: float = 10.0
    floor_min_dbfs: float = -75.0
    turn_model: str | None = None
    turn_min_silence_s: float = 0.2
    turn_window_s: float = 8.0
    turn_threshold: float = 0.5
    # transcriber
    whisper_repo: str = "mlx-community/whisper-large-v3-turbo"
    language: str = "ru"
    whisper_prompt: str | None = None
    # brain
    brain: BrainConfig = field(default_factory=BrainConfig)
    # voice
    voice_engine: str = "qwen3"
    tts_model: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
    tts_voice: str = "serena"
    tts_language: str = "russian"
    tts_chunk_s: float = 0.5
    tts_hf_home: str | None = "scratch/tts_voices/hf"
    say_voice: str = "Milena"
    say_rate_wpm: int = 185
    # player
    playback_lead_s: float = 0.5
    playback_block_s: float = 0.04
    # flags
    barge_in: bool = False
    barge_in_over_floor_db: float = 25.0
    barge_in_min_dbfs: float = -45.0
    barge_in_hold_s: float = 0.25
    backchannel: bool = False
    backchannel_delay_s: float = 0.3
    backchannel_min_speech_s: float = 0.8
    backchannel_lines: tuple[str, ...] = ("угу", "мм")  # noqa: RUF001
    prosody_tags: bool = True
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)
    # conversation
    idle_close_s: float = 20.0
    max_session_s: float = 900.0
    report_drive_start: bool = True
    error_line: str = "Извини, у меня сбой связи."  # noqa: RUF001
    tool_filler: str = "Секунду."
    # money
    daily_budget_cad: float = 2.0
    max_sessions_per_hour: int = 60
    usd_to_cad: float = 1.38
    prices: Mapping[str, ModelPrice] = field(default_factory=lambda: dict(DEFAULT_PRICES))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CascadeConfig:
        """From the file's object; ``ValueError`` naming a value out of range."""
        ears = data.get("ears", {})
        tr = data.get("transcriber", {})
        voice = data.get("voice", {})
        player = data.get("player", {})
        flags = data.get("flags", {})
        barge = data.get("barge_in", {})
        back = data.get("backchannel", {})
        pros = data.get("prosody", {})
        turn = data.get("turn", {})
        conv = data.get("conversation", {})
        money = data.get("money", {})
        d = cls()
        config = cls(
            wake=str(ears.get("wake", d.wake)),
            silence_s=float(ears.get("silence_s", d.silence_s)),
            max_utterance_s=float(ears.get("max_utterance_s", d.max_utterance_s)),
            start_db=float(ears.get("start_db", d.start_db)),
            floor_min_dbfs=float(ears.get("floor_min_dbfs", d.floor_min_dbfs)),
            turn_model=ears.get("turn_model") or None,
            turn_min_silence_s=float(turn.get("min_silence_s", d.turn_min_silence_s)),
            turn_window_s=float(turn.get("window_s", d.turn_window_s)),
            turn_threshold=float(turn.get("threshold", d.turn_threshold)),
            whisper_repo=str(tr.get("whisper_repo", d.whisper_repo)),
            language=str(tr.get("language", d.language)),
            whisper_prompt=tr.get("initial_prompt") or None,
            brain=brain_config(data.get("brain", {})),
            voice_engine=str(voice.get("engine", d.voice_engine)),
            tts_model=str(voice.get("model", d.tts_model)),
            tts_voice=str(voice.get("voice", d.tts_voice)),
            tts_language=str(voice.get("language", d.tts_language)),
            tts_chunk_s=float(voice.get("chunk_s", d.tts_chunk_s)),
            tts_hf_home=voice.get("hf_home", d.tts_hf_home) or None,
            say_voice=str(voice.get("say_voice", d.say_voice)),
            say_rate_wpm=int(voice.get("say_rate_wpm", d.say_rate_wpm)),
            playback_lead_s=float(player.get("lead_s", d.playback_lead_s)),
            playback_block_s=float(player.get("block_s", d.playback_block_s)),
            barge_in=bool(flags.get("barge_in", d.barge_in)),
            backchannel=bool(flags.get("backchannel", d.backchannel)),
            prosody_tags=bool(flags.get("prosody_tags", d.prosody_tags)),
            barge_in_over_floor_db=float(barge.get("over_floor_db", d.barge_in_over_floor_db)),
            barge_in_min_dbfs=float(barge.get("min_dbfs", d.barge_in_min_dbfs)),
            barge_in_hold_s=float(barge.get("hold_s", d.barge_in_hold_s)),
            backchannel_delay_s=float(back.get("delay_s", d.backchannel_delay_s)),
            backchannel_min_speech_s=float(back.get("min_speech_s", d.backchannel_min_speech_s)),
            backchannel_lines=tuple(str(x) for x in back.get("lines", d.backchannel_lines)),
            prosody=ProsodyConfig(
                speaker=str(pros.get("speaker", d.prosody.speaker)),
                min_baseline=int(pros.get("min_baseline", d.prosody.min_baseline)),
                window=int(pros.get("window", d.prosody.window)),
                z=float(pros.get("z", d.prosody.z)),
                emotion_model=pros.get("emotion_model") or None,
            ),
            idle_close_s=float(conv.get("idle_close_s", d.idle_close_s)),
            max_session_s=float(conv.get("max_session_s", d.max_session_s)),
            report_drive_start=bool(conv.get("report_drive_start", d.report_drive_start)),
            error_line=str(conv.get("error_line", d.error_line)),
            tool_filler=str(conv.get("tool_filler", d.tool_filler)),
            daily_budget_cad=float(money.get("daily_budget_cad", d.daily_budget_cad)),
            max_sessions_per_hour=int(money.get("max_sessions_per_hour", d.max_sessions_per_hour)),
            usd_to_cad=float(money.get("usd_to_cad", d.usd_to_cad)),
            prices={**DEFAULT_PRICES, **prices_of(money.get("prices_usd_per_mtok", {}))},
        )
        config.check()
        return config

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: Sequence[str] = ()) -> CascadeConfig:
        """From ``config/voice_cascade.json`` wherever this library runs (or ``path``), with
        ``section.key=value`` overrides applied."""
        if path is None:
            from pepin.deployment import config_file

            path = config_file(CONFIG_FILE)
        data = json.loads(Path(path).read_text())
        return cls.from_dict(apply_overrides(data, overrides))

    def check(self) -> None:
        """``ValueError`` when a value cannot work."""
        b = self.brain
        if b.provider not in PROVIDERS:
            raise ValueError(f"brain.provider: one of {', '.join(PROVIDERS)}")
        if b.thinking_level is not None and b.thinking_level not in THINKING_LEVELS:
            raise ValueError(f"brain.thinking_level: one of {', '.join(THINKING_LEVELS)} or null")
        if self.wake not in WAKE_MODES:
            raise ValueError(f"ears.wake: one of {', '.join(WAKE_MODES)}")
        if b.thinking is not None and b.thinking not in THINKING_MODES:
            raise ValueError(f"brain.thinking: one of {', '.join(THINKING_MODES)} or null")
        if b.effort is not None and b.effort not in EFFORTS:
            raise ValueError(f"brain.effort: one of {', '.join(EFFORTS)} or null")
        if b.max_tokens < 16 or b.max_tool_rounds < 0:
            raise ValueError("brain.max_tokens >= 16 and brain.max_tool_rounds >= 0")
        if self.voice_engine not in VOICE_ENGINES:
            raise ValueError(f"voice.engine: one of {', '.join(VOICE_ENGINES)}")
        if self.prosody.emotion_model not in (None, *EMOTION_MODELS):
            raise ValueError(f"prosody.emotion_model: one of {', '.join(EMOTION_MODELS)} or null")
        if self.turn_model is not None and self.turn_model not in TURN_MODELS:
            raise ValueError(f"ears.turn_model: one of {', '.join(TURN_MODELS)} or null")
        if not 0 < self.turn_min_silence_s < self.silence_s:
            raise ValueError("turn.min_silence_s: above 0 and under ears.silence_s")
        if not 1.0 <= self.turn_window_s <= 8.0 or not 0.0 < self.turn_threshold <= 1.0:
            raise ValueError(
                "turn.window_s in 1..8 s (the model's input) and turn.threshold in 0..1"
            )
        if not 0.1 <= self.silence_s <= 3.0 or self.barge_in_hold_s <= 0:
            raise ValueError("ears.silence_s in 0.1..3 s and barge_in.hold_s > 0")
        if self.idle_close_s <= 0 or self.daily_budget_cad <= 0:
            raise ValueError("conversation.idle_close_s > 0 and money.daily_budget_cad > 0")
        if b.model not in self.prices:
            raise ValueError(f"no price for {b.model}: add it to money.prices_usd_per_mtok")

    def with_changes(self, **changes: Any) -> CascadeConfig:
        """A copy with the given fields replaced (None leaves a field alone), checked again."""
        config = replace(self, **{k: v for k, v in changes.items() if v is not None})
        config.check()
        return config

    @property
    def price(self) -> ModelPrice:
        """The brain model's prices."""
        return self.prices[self.brain.model]

    @property
    def daily_budget_usd(self) -> float:
        """The daily cap in USD, the currency of the prices."""
        return self.daily_budget_cad / self.usd_to_cad


def brain_config(section: Mapping[str, Any]) -> BrainConfig:
    """The brain's settings: the chosen preset's, overridden by the keys beside it."""
    presets = section.get("presets", {})
    name = section.get("preset")
    if name is not None and name not in presets:
        raise ValueError(f"brain.preset {name!r}: one of {', '.join(presets)}")
    merged = {**(presets.get(name, {}) if name else {}), **section}
    d = BrainConfig()
    budget = merged.get("thinking_budget", d.thinking_budget)
    return BrainConfig(
        provider=str(merged.get("provider", d.provider)),
        model=str(merged.get("model", d.model)),
        thinking=merged.get("thinking", d.thinking),
        effort=merged.get("effort", d.effort),
        thinking_budget=None if budget is None else int(budget),
        thinking_level=merged.get("thinking_level", d.thinking_level),
        max_tokens=int(merged.get("max_tokens", d.max_tokens)),
        persona_file=str(merged.get("persona_file", d.persona_file)),
        cache=bool(merged.get("cache", d.cache)),
        prewarm=bool(merged.get("prewarm", d.prewarm)),
        keep_warm_s=float(merged.get("keep_warm_s", d.keep_warm_s)),
        max_tool_rounds=int(merged.get("max_tool_rounds", d.max_tool_rounds)),
        timeout_s=float(merged.get("timeout_s", d.timeout_s)),
    )


def prices_of(table: Mapping[str, Any]) -> dict[str, ModelPrice]:
    """``{model: {input, output, cache_read, cache_write}}`` as :class:`ModelPrice` (notes
    skipped)."""
    return {
        model: ModelPrice(
            float(row["input"]),
            float(row["output"]),
            float(row["cache_read"]),
            float(row["cache_write"]),
        )
        for model, row in table.items()
        if isinstance(row, Mapping)
    }


def apply_overrides(data: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """``data`` with each ``section.key=value`` set (the value read as JSON when it parses:
    ``true``, ``0.4``, ``null``, ``["мм"]``; as a plain string otherwise); ``ValueError`` for
    an override without ``=``."""
    out: dict[str, Any] = copy.deepcopy(dict(data))
    for item in overrides:
        path, sep, raw = item.partition("=")
        if not sep or not path.strip():
            raise ValueError(f"override {item!r}: section.key=value")
        try:
            value: Any = json.loads(raw)
        except ValueError:
            value = raw
        node = out
        keys = path.strip().split(".")
        for key in keys[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                child = {}
                node[key] = child
            node = child
        node[keys[-1]] = value
    return out
