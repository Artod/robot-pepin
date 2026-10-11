"""The brain and the voice a config names, built at the run's start and again for a switch.

:func:`build_brain` makes the talker of ``brain`` (Claude, Gemini, an OpenAI-compatible
endpoint, or the echo that calls no API) with the tools its preset offers; :class:`VoiceShop`
makes the voice of ``voice`` and keeps the local models it made, so a cloud voice's fallback
and a switch back to a voice just used load nothing twice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pepin.voice_cascade.config import CascadeConfig, VoiceConfig
from pepin.voice_cascade.stages import Brain, Voice

NOT_FOR_THE_BRAIN = frozenset({"say"})  # the brain speaks through the voice, not a tool
ECHO = "echo"  # the brain that repeats what it heard (no API call)
API_KEYS = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


@dataclass
class BrainSetup:
    """A brain ready to talk, the config it was built from, and the tools it may call."""

    brain: Brain
    config: CascadeConfig
    name: str  # its preset (or ECHO)
    allowed: set[str]
    moving: set[str]
    paid: bool  # it calls a paid API (the ledger records it)


def brain_key(config: CascadeConfig) -> str:
    """The environment variable that holds the brain's API key."""
    b = config.brain
    return API_KEYS.get(b.provider, b.api_key_env or "")


def offered_tools(config: CascadeConfig, wanted: set[str] | None = None) -> list[str]:
    """The registry's tools the brain may call: all but ``say``, only ``wanted`` when given,
    only the preset's when it names some (a compact prompt)."""
    from pepin.tools import TOOLS

    names = [t.name for t in TOOLS if t.name not in NOT_FOR_THE_BRAIN]
    if wanted is not None:
        names = [n for n in names if n in wanted]
    if config.brain.tools is not None:
        names = [n for n in names if n in config.brain.tools]
    return names


def build_brain(
    config: CascadeConfig, *, echo: bool = False, wanted: set[str] | None = None
) -> BrainSetup:
    """The brain ``config.brain`` names (or the echo); ``RuntimeError`` when its key is not in
    the environment."""
    from pepin.tools import TOOLS

    allowed = set(offered_tools(config, wanted))
    moving = {t.name for t in TOOLS if t.moves and t.name in allowed}
    if echo:
        from pepin.voice_cascade.fake import ScriptedBrain

        return BrainSetup(ScriptedBrain(), config, ECHO, allowed, moving, paid=False)
    key = brain_key(config)
    if not os.environ.get(key):
        raise RuntimeError(f"{key} missing: set -a; . ./.env; set +a (the main checkout's .env)")
    from pepin.deployment import config_file
    from pepin.tools.schemas import gemini_function_declarations, openai_tools

    b = config.brain
    persona = config_file(b.persona_file).read_text()
    brain: Brain
    if b.provider == "openai":
        from pepin.voice_cascade.chat_brain import ChatBrain

        functions = [t for t in openai_tools(TOOLS) if t["function"]["name"] in allowed]
        brain = ChatBrain(b, persona=persona, tools=functions, price=config.price)
    elif b.provider == "gemini":
        from pepin.voice_cascade.gemini_brain import GeminiBrain

        declarations = [d for d in gemini_function_declarations(TOOLS) if d["name"] in allowed]
        brain = GeminiBrain(b, persona=persona, declarations=declarations, price=config.price)
    else:
        from pepin.voice_cascade.brain import ClaudeBrain, tool_definitions

        brain = ClaudeBrain(
            b, persona=persona, tools=tool_definitions(TOOLS, allowed), price=config.price
        )
    return BrainSetup(brain, config, b.preset or b.model, allowed, moving, paid=True)


def local_key(v: VoiceConfig) -> tuple[object, ...]:
    """What makes a local voice's sound (two presets alike share one loaded model)."""
    return (v.engine, v.model, v.voice, v.voice_description, v.use_style, v.seed, v.language,
            v.chunk_s, v.hf_home, v.say_voice, v.say_rate_wpm)  # fmt: skip


class VoiceShop:
    """Builds voices and keeps the local ones it built until told which are still in use."""

    def __init__(self, config: CascadeConfig, root: Path, cache_dir: Path | None = None) -> None:
        """``config``: its prices price a cloud voice; ``root``: where a relative ``hf_home``
        lies (the main checkout); ``cache_dir``: where a cloud voice keeps its short lines
        (None: ``~/.cache/pepin/tts``)."""
        from pepin.voice_cascade.gemini_voice import default_cache

        self.prices = config.prices
        self.root = root
        self.cache_dir = cache_dir if cache_dir is not None else default_cache()
        self._local: dict[tuple[object, ...], Voice] = {}

    def voice(self, v: VoiceConfig) -> Voice:
        """The voice ``v`` names (a cloud voice with its fallback from the local ones), at
        ``v.gain_db``: a cloud voice scales its own audio only, its fallback speaks at the
        fallback's gain."""
        if v.engine == "gemini":
            return self._cloud(v)
        from pepin.voice_cascade.voice import gained

        return gained(self.local(v), v.gain_db)

    def _cloud(self, v: VoiceConfig) -> Voice:
        from pepin.voice_cascade.gemini_voice import GeminiVoice

        if not os.environ.get(API_KEYS["gemini"]):
            raise RuntimeError("GEMINI_API_KEY missing: set -a; . ./.env; set +a")
        return GeminiVoice(
            v.model,
            voice=v.voice or "Kore",
            price=self.prices[v.model],
            fallback=self.voice(v.fallback) if v.fallback is not None else None,
            fallback_name=v.fallback.name if v.fallback is not None else "none",
            timeout_s=v.cloud_timeout_s,
            max_calls=v.cloud_max_calls,
            cache_dir=self.cache_dir,
            gain_db=v.gain_db,
        )

    def local(self, v: VoiceConfig) -> Voice:
        """The local voice ``v`` names, the one built before when alike."""
        key = local_key(v)
        voice = self._local.get(key)
        if voice is None:
            voice = self._local[key] = self._build(v)
        return voice

    def keep(self, v: VoiceConfig) -> None:
        """Forget the local voices ``v`` does not use (freed once nothing speaks with them)."""
        used = {local_key(v)} | ({local_key(v.fallback)} if v.fallback is not None else set())
        for key in [k for k in self._local if k not in used]:
            del self._local[key]

    def _build(self, v: VoiceConfig) -> Voice:
        if v.engine == "say":
            from pepin.voice_cascade.voice import SayVoice

            return SayVoice(v.say_voice, v.say_rate_wpm)
        from pepin.voice_cascade.voice import QwenVoice

        hf_home = Path(v.hf_home) if v.hf_home else None
        if hf_home is not None and not hf_home.is_absolute():
            hf_home = self.root / hf_home
        return QwenVoice(
            v.model,
            voice=v.voice,
            language=v.language,
            chunk_s=v.chunk_s,
            hf_home=hf_home,
            use_style=v.use_style,
            description=v.voice_description,
            seed=v.seed,
        )
