"""Switching the brain, the voice and the transcriber of a running cascade: a request in a file,
answers in a log.

``scripts/voice_cascade.py switch --brain sonnet --voice gemini --transcriber whispercpp``
writes ``control.json`` in the logs folder (``data/voice``): ``{"id", "t", "brain", "voice",
"transcriber"}`` (``voice_preset`` is read as ``voice``; a key left out changes nothing; a file
written by hand needs no ``id``). :class:`ControlFile` is its one reader: the loop polls it
between ticks, builds what a change names (a brain is created and warmed; a voice loads beside
the one speaking; the transcriber, :class:`pepin.voice_cascade.transcriber.SwitchingTranscriber`,
loads its new engine beside the old one) and swaps it in (the brain at the next turn, the voice
at its next sentence, the transcriber between utterances), appending each step to
``control.jsonl`` (``{"id", "t", "what", "name", "state", "ms", "error"}``), which the switch
command reads back. What the file holds at the start is an earlier run's and is never applied.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pepin.voice_cascade.config import CascadeConfig, VoiceConfig
from pepin.voice_cascade.engines import BrainSetup
from pepin.voice_cascade.stages import Voice
from pepin.voice_cascade.transcriber import Choice, SwitchingTranscriber, choice_of

CONTROL = "control.json"
ANSWERS = "control.jsonl"
KEYS = ("brain", "voice", "voice_preset", "transcriber")
DONE_STATES = ("applied", "failed")  # the last answer of a switch


@dataclass(frozen=True)
class Request:
    """A switch asked for: a brain preset, a voice preset (or an engine's name), a transcriber
    (``"whispercpp"``, ``"whispercpp:large-v3"`` or ``{"engine", "model"}``), or several."""

    id: str
    brain: str | None = None
    voice: str | None = None
    transcriber: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_id: str = "") -> Request:
        """From the control file's object (``default_id`` when it carries none)."""
        voice = data.get("voice_preset") or data.get("voice")
        return cls(
            str(data.get("id") or default_id),
            data.get("brain") or None,
            voice or None,
            data.get("transcriber") or None,
        )


def write_request(
    root: Path, brain: str | None, voice: str | None, transcriber: str | None = None
) -> Request:
    """Ask the running loop for a switch (the file replaced at once, never half written)."""
    stamp = time.strftime("%H%M%S-") + f"{time.time_ns() % 10**9:09d}"
    request = Request(stamp, brain, voice, transcriber)
    root.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {"id": request.id, "t": round(time.time(), 3)}
    body.update({k: v for k, v in (("brain", brain), ("voice", voice),
                                   ("transcriber", transcriber)) if v})  # fmt: skip
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


class ControlFile:
    """``control.json``, its one reader: read when it changes (its mtime), each change handed
    out once as a :class:`Request` (:meth:`poll`), the latest value of each key kept for
    whoever asks from any thread (:meth:`latest`). What it holds at the start is an earlier
    run's and is never applied; a half-written file is read again at the next poll."""

    def __init__(self, root: Path) -> None:
        """Watch ``root/control.json`` (it may not exist yet)."""
        self.root = root
        self.path = root / CONTROL
        self._stamp = self._mtime()
        self._latest: dict[str, Any] = {}
        self._lock = threading.Lock()

    def poll(self) -> Request | None:
        """The change made since the last call, None when there is none (or it is half
        written)."""
        stamp = self._mtime()
        if stamp is None or stamp == self._stamp:
            return None
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None  # mid-write: read again at the next poll
        self._stamp = stamp
        if not isinstance(data, dict):
            return None
        with self._lock:
            self._latest.update({k: data[k] for k in KEYS if data.get(k) is not None})
        return Request.from_dict(data, default_id=f"file-{stamp}")

    def latest(self, key: str) -> Any:
        """The value ``key`` had in the latest change that set it (None: none since start)."""
        with self._lock:
            return self._latest.get(key)

    def _mtime(self) -> int | None:
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return None


@dataclass
class Switchboard:
    """What a loop needs to switch: the control file, the run's config with other presets, the
    builders, and the transcriber that switches itself. ``brain`` and ``voice`` are the
    presets in use."""

    control: ControlFile
    configure: Callable[[str, str], CascadeConfig]  # (brain preset, voice preset) -> config
    build_brain: Callable[[CascadeConfig, str], BrainSetup]  # (config, brain preset or echo)
    build_voice: Callable[[VoiceConfig], Voice]
    brain: str
    voice: str
    keep_voice: Callable[[VoiceConfig], None] = field(default=lambda v: None)
    transcriber: SwitchingTranscriber | None = None

    def brain_setup(self, name: str) -> BrainSetup:
        """The brain ``name`` with the voice in use (``ValueError`` for an unknown preset)."""
        config = self.configure(name, self.voice)
        return self.build_brain(config, name)

    def voice_config(self, name: str) -> VoiceConfig:
        """The voice ``name`` names, as the run's config resolves it."""
        return self.configure(self.brain, name).voice

    def answer(self, request: str, what: str, name: str, state: str, **fields: Any) -> None:
        """Log one step of a switch beside the control file."""
        answer(self.control.root, request, what, name, state, **fields)

    def switch_transcriber(self, request: str, value: Any) -> None:
        """Have the transcriber load the engine ``value`` names (it swaps between utterances);
        its outcome answered for ``request``."""
        t = self.transcriber
        label = str(value)
        if t is None:
            self.answer(request, "transcriber", label, "failed", error="not switchable here")
            return
        choice = choice_of(value, self.configure(self.brain, self.voice))
        if choice is None:
            self.answer(request, "transcriber", label, "failed", error="unknown engine")
            return
        if choice == t.choice:
            self.answer(request, "transcriber", choice.label, "applied", note="already in use")
            return
        if choice == t.failed:
            why = "it failed to load; asked again only after another engine"
            self.answer(request, "transcriber", choice.label, "failed", error=why)
            return
        start = time.monotonic()

        def loaded(done: Choice, error: BaseException | None) -> None:  # on its loading thread
            ms = round((time.monotonic() - start) * 1000)
            if error is None:
                self.answer(request, "transcriber", done.label, "applied", ms=ms)
            else:
                why = f"{type(error).__name__}: {error}"[:300]
                self.answer(request, "transcriber", done.label, "failed", ms=ms, error=why)

        t.on_loaded = loaded
        self.answer(request, "transcriber", choice.label, "loading")
        t.poll()


def with_voice(config: CascadeConfig, voice: VoiceConfig) -> CascadeConfig:
    """``config`` speaking with ``voice``."""
    return replace(config, voice=voice)
