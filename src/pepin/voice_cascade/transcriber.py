"""The cascade's speech to text: the engine ``transcriber.engine`` names, switchable while running.

``mlx`` is Whisper on the GPU (:class:`pepin.voice_live.wake.MlxWhisper`); ``whispercpp`` is
whisper.cpp with the encoder on the Neural Engine
(:class:`pepin.voice_live.whisper_cpp.WhisperCppTranscriber`), which leaves the GPU to the voice
and the CPU to the ROS stack. Each engine has its own model (``transcriber.mlx.model``,
``transcriber.whispercpp.model``). :class:`SwitchingTranscriber` holds the one in use and
replaces it between utterances when the control file (``data/voice/control.json``, key
``transcriber``, read by :class:`pepin.voice_cascade.control.ControlFile`) names another after
the start: the new engine loads on its own thread while the old one keeps reading, and the swap
waits for the call in flight.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pepin.voice_cascade.config import TRANSCRIBER_ENGINES, CascadeConfig
from pepin.voice_cascade.stages import Transcriber

CONTROL_KEY = "transcriber"  # the control file's key
POLL_S = 0.5
Loaded = Callable[["Choice", BaseException | None], None]  # (the engine asked for, its error)


@dataclass(frozen=True)
class Choice:
    """An engine and its model: ``mlx`` with a Hugging Face repo, ``whispercpp`` with a ggml
    model name (``large-v3-turbo``, ``large-v3``)."""

    engine: str
    model: str

    @property
    def label(self) -> str:
        """``engine:model``, the model's last path part."""
        return f"{self.engine}:{self.model.rsplit('/', 1)[-1]}"


def configured(config: CascadeConfig) -> Choice:
    """The engine and model the config names."""
    return Choice(config.transcriber_engine, model_of(config, config.transcriber_engine))


def model_of(config: CascadeConfig, engine: str) -> str:
    """The config's model for ``engine``."""
    return config.whisper_repo if engine == "mlx" else config.whispercpp_model


def choice_of(value: Any, config: CascadeConfig) -> Choice | None:
    """A control file's ``transcriber``: an engine (``"whispercpp"``), engine and model
    (``"whispercpp:large-v3"``) or ``{"engine": ..., "model": ...}``; the model defaults to the
    config's for that engine. None for anything else."""
    if isinstance(value, str):
        engine, _, model = value.strip().partition(":")
    elif isinstance(value, dict) and isinstance(value.get("engine"), str):
        engine, model = value["engine"], str(value.get("model") or "")
    else:
        return None
    if engine not in TRANSCRIBER_ENGINES:
        return None
    return Choice(engine, model or model_of(config, engine))


def build_transcriber(
    config: CascadeConfig, choice: Choice, root: Path, log: Path | None = None
) -> Transcriber:
    """The engine of ``choice``, loaded and warmed; ``root`` anchors a relative
    ``transcriber.whispercpp.dir``, ``log`` takes whisper-server's output."""
    if choice.engine == "mlx":
        from pepin.voice_live.wake import MlxWhisper

        return MlxWhisper(choice.model, config.language, config.whisper_prompt)
    from pepin.voice_live.whisper_cpp import WhisperCppTranscriber

    home = Path(config.whispercpp_dir)
    home = home if home.is_absolute() else root / home
    return WhisperCppTranscriber.start(
        home / "bin" / "whisper-server",
        home / "models" / f"ggml-{choice.model}.bin",
        language=config.language,
        initial_prompt=config.whisper_prompt,
        threads=config.whispercpp_threads,
        gpu=config.whispercpp_gpu,
        units=config.whispercpp_units,
        log=log,
    )


class Latest(Protocol):
    """The control file as its one reader keeps it (:class:`pepin.voice_cascade.control.
    ControlFile`): the latest value of a key since the start."""

    def latest(self, key: str) -> Any:
        """The value ``key`` had in the latest change that set it (None: none yet)."""
        ...


def wanted_from(control: Latest, config: CascadeConfig) -> Callable[[], Choice | None]:
    """The ``transcriber`` the control file's latest change named, as a choice (None: none
    since the start, or not a valid one); a choice left from an earlier run never counts."""
    return lambda: choice_of(control.latest(CONTROL_KEY), config)


class SwitchingTranscriber:
    """The transcriber in use, replaced between utterances when ``wanted()`` names another: the
    new one is built on its own thread (``spawn``) while the old one keeps reading, swapped in
    once no call is in flight, and the old one closed. A choice that fails to load is reported
    and not tried again until ``wanted()`` changes; one no longer wanted once it has loaded (the
    person switched back meanwhile) is closed unused."""

    def __init__(
        self,
        current: Transcriber,
        choice: Choice,
        build: Callable[[Choice], Transcriber],
        wanted: Callable[[], Choice | None],
        *,
        report: Callable[[str], None] = print,
        spawn: Callable[[Callable[[], None]], None] | None = None,
        on_loaded: Loaded | None = None,
    ) -> None:
        """``build`` loads an engine; ``wanted`` is polled (:meth:`poll`, or every
        ``POLL_S`` after :meth:`start`); ``on_loaded`` hears each load's outcome (None: the
        engine is in use) on the loading thread."""
        self.choice = choice
        self.on_loaded = on_loaded
        self._current = current
        self._build, self._wanted, self._report = build, wanted, report
        self._spawn = spawn or _thread
        self._lock = threading.Lock()  # held by a call in flight and by the swap
        self._loading: Choice | None = None
        self._failed: Choice | None = None
        self._stop = threading.Event()

    @property
    def name(self) -> str:
        """The engine in use, ``engine:model``."""
        return self.choice.label

    def transcribe(self, pcm: bytes) -> str:
        """The words in ``pcm``, by the engine in use."""
        with self._lock:
            return self._current.transcribe(pcm)

    def poll(self) -> None:
        """Start loading the wanted engine if it differs from the one in use."""
        want = self._wanted()
        if want is None or want in (self.choice, self._loading, self._failed):
            return
        self._loading = want
        on_loaded = self.on_loaded  # this load's outcome goes to whoever asked for it
        self._spawn(lambda: self._load(want, on_loaded))

    def _load(self, want: Choice, on_loaded: Loaded | None = None) -> None:
        t0 = time.monotonic()
        try:
            fresh = self._build(want)
        except Exception as error:  # a missing file, a failed server: keep the one in use
            self._failed = want
            if self._loading == want:
                self._loading = None
            self._report(f"transcriber: {want.label} failed to load ({error}); keeping {self.name}")
            if on_loaded is not None:
                on_loaded(want, error)
            return
        if self._wanted() != want:  # another engine (or the one in use) was asked for meanwhile
            _close(fresh)
            if self._loading == want:
                self._loading = None
            self._report(f"transcriber: {want.label} no longer wanted; keeping {self.name}")
            if on_loaded is not None:
                on_loaded(want, RuntimeError("superseded by a later switch"))
            return
        with self._lock:
            old, was = self._current, self.name
            self._current, self.choice = fresh, want
        if self._loading == want:
            self._loading = None
        self._failed = None
        _close(old)
        self._report(
            f"transcriber: {was} -> {want.label} (loaded in {time.monotonic() - t0:.1f} s)"
        )
        if on_loaded is not None:
            on_loaded(want, None)

    @property
    def failed(self) -> Choice | None:
        """The engine that last failed to load (not tried again until another is wanted)."""
        return self._failed

    def start(self, every_s: float = POLL_S) -> SwitchingTranscriber:
        """Poll ``wanted()`` every ``every_s`` on a daemon thread until :meth:`close`."""

        def run() -> None:
            while not self._stop.wait(every_s):
                self.poll()

        threading.Thread(target=run, name="transcriber-switch", daemon=True).start()
        return self

    def close(self) -> None:
        """Stop polling and close the engine in use."""
        self._stop.set()
        with self._lock:
            _close(self._current)


def _thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name="transcriber-load", daemon=True).start()


def _close(transcriber: Transcriber) -> None:
    close = getattr(transcriber, "close", None)
    if close is not None:
        close()
