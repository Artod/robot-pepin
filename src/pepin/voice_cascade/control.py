"""Switching the brain and the voice of a running cascade: a request in a file, answers in a log.

``scripts/voice_cascade.py switch --brain sonnet --voice gemini`` writes ``control.json`` in the
logs folder (``data/voice``): ``{"id", "t", "brain", "voice"}`` (``voice_preset`` is read as
``voice``). The loop looks at it between ticks and builds what it names (a brain is created and
warmed; a voice loads beside the one speaking), swaps it in at the next turn, and appends each
step to ``control.jsonl`` (``{"id", "t", "what", "name", "state", "ms", "error"}``), which the
switch command reads back. A request left by an earlier run is not applied at start.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pepin.voice_cascade.config import CascadeConfig, VoiceConfig
from pepin.voice_cascade.engines import BrainSetup
from pepin.voice_cascade.stages import Voice

CONTROL = "control.json"
ANSWERS = "control.jsonl"
DONE_STATES = ("applied", "failed")  # the last answer of a switch


@dataclass(frozen=True)
class Request:
    """A switch asked for: a brain preset, a voice preset (or an engine's name), or both."""

    id: str
    brain: str | None = None
    voice: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Request:
        """From the control file's object."""
        voice = data.get("voice_preset") or data.get("voice")
        return cls(str(data.get("id", "")), data.get("brain") or None, voice or None)


def write_request(root: Path, brain: str | None, voice: str | None) -> Request:
    """Ask the running loop for a switch (the file replaced at once, never half written)."""
    request = Request(time.strftime("%H%M%S-") + f"{time.time_ns() % 10**9:09d}", brain, voice)
    root.mkdir(parents=True, exist_ok=True)
    body = {"id": request.id, "t": round(time.time(), 3), "brain": brain, "voice": voice}
    tmp = root / (CONTROL + ".tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False) + "\n")
    tmp.replace(root / CONTROL)
    return request


def answer(root: Path, request: str, what: str, name: str, state: str, **fields: Any) -> None:
    """One step of a switch, appended to the answers log."""
    row = {"id": request, "t": round(time.time(), 3), "what": what, "name": name, "state": state}
    row.update({k: v for k, v in fields.items() if v is not None})
    root.mkdir(parents=True, exist_ok=True)
    with (root / ANSWERS).open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def answers_to(root: Path, request: str) -> list[dict[str, Any]]:
    """The loop's answers to ``request`` so far."""
    path = root / ANSWERS
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("id") == request:
            rows.append(row)
    return rows


class Requests:
    """The control file, read when it changes; each request handed out once."""

    def __init__(self, root: Path) -> None:
        """Watch ``root/control.json``; what it holds now is an earlier run's, never applied."""
        self.root = root
        self.path = root / CONTROL
        self._stamp = self._mtime()
        old = self._read()
        self._seen = old.id if old is not None else ""

    def poll(self) -> Request | None:
        """The request written since the last call, None when there is none."""
        stamp = self._mtime()
        if stamp == self._stamp:
            return None
        self._stamp = stamp
        request = self._read()
        if request is None or request.id == self._seen:
            return None
        self._seen = request.id
        return request

    def _mtime(self) -> int:
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return 0

    def _read(self) -> Request | None:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None
        return Request.from_dict(data) if isinstance(data, dict) else None


@dataclass
class Switchboard:
    """What a loop needs to switch: the requests, the run's config with other presets, and
    the builders. ``brain`` and ``voice`` are the presets in use."""

    requests: Requests
    configure: Callable[[str, str], CascadeConfig]  # (brain preset, voice preset) -> config
    build_brain: Callable[[CascadeConfig, str], BrainSetup]  # (config, brain preset or echo)
    build_voice: Callable[[VoiceConfig], Voice]
    brain: str
    voice: str
    keep_voice: Callable[[VoiceConfig], None] = field(default=lambda v: None)

    def brain_setup(self, name: str) -> BrainSetup:
        """The brain ``name`` with the voice in use (``ValueError`` for an unknown preset)."""
        config = self.configure(name, self.voice)
        return self.build_brain(config, name)

    def voice_config(self, name: str) -> VoiceConfig:
        """The voice ``name`` names, as the run's config resolves it."""
        return self.configure(self.brain, name).voice

    def answer(self, request: str, what: str, name: str, state: str, **fields: Any) -> None:
        """Log one step of a switch beside the control file."""
        answer(self.requests.root, request, what, name, state, **fields)


def with_voice(config: CascadeConfig, voice: VoiceConfig) -> CascadeConfig:
    """``config`` speaking with ``voice``."""
    return replace(config, voice=voice)
