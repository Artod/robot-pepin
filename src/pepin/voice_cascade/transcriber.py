"""The cascade's speech to text: the engine ``transcriber.engine`` names, switchable while running.

``mlx`` is Whisper on the GPU (:class:`pepin.voice_live.wake.MlxWhisper`); ``whispercpp`` is
whisper.cpp with the encoder on the Neural Engine
(:class:`pepin.voice_live.whisper_cpp.WhisperCppTranscriber`), which leaves the GPU to the voice
and the CPU to the ROS stack; ``gigaam``, ``parakeet``, ``canary`` and ``voxtral`` are the
engines of :mod:`pepin.voice_cascade.stt`, each from its optional dependency group. Each engine
has its own model (``transcriber.<engine>.model``). :class:`SwitchingTranscriber` holds the one
in use and replaces it between utterances when the control file (``data/voice/control.json``, key
``transcriber``, read by :class:`pepin.voice_cascade.control.ControlFile`) names another after
the start: the new engine loads on its own thread while the old one keeps reading, the swap
waits for the call in flight, and the old one is closed (its memory freed).
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from pepin.voice_cascade.config import STT_DEVICES, TRANSCRIBER_ENGINES, CascadeConfig
from pepin.voice_cascade.stages import TimedTranscriber, Transcriber, Transcript
from pepin.voice_cascade.stt.runtime import footprint_mb

CONTROL_KEY = "transcriber"  # the control file's key
POLL_S = 0.5
# the footprint is read this long after the old engine closed: the GPU driver hands the buffers
# MLX freed back to the system ~0.5 s later (Whisper turbo: 2.1 GB still counted at once, 0.26 GB
# at 0.5 s, 2026-10-12)
SETTLE_S = 1.0
Loaded = Callable[["Choice", BaseException | None], None]  # (the engine asked for, its error)


@dataclass(frozen=True)
class Choice:
    """An engine and its model: ``mlx`` with a Hugging Face repo, ``whispercpp`` with a ggml
    model name (``large-v3-turbo``, ``large-v3``), the others with their block's kind of model
    (:class:`pepin.voice_cascade.config.SttConfig`)."""

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
    if engine in STT_DEVICES:
        return config.stt[engine].model
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


def read_timed(transcriber: Transcriber, pcm: bytes) -> Transcript:
    """The words in ``pcm`` with their timings when ``transcriber`` gives them (none when it
    cannot)."""
    if isinstance(transcriber, TimedTranscriber):
        return transcriber.transcribe_timed(pcm)
    return Transcript(transcriber.transcribe(pcm))


class Reading(concurrent.futures.Future[str]):
    """An early read as the ears hold it, a future of its text, with the timed transcript
    kept beside it for the loop (None until it is read)."""

    transcript: Transcript | None = None


def read_early(
    executor: concurrent.futures.Executor, transcriber: Transcriber, pcm: bytes
) -> Reading:
    """:func:`read_timed` of ``pcm`` on ``executor``; cancelled before it starts, it never
    runs (as an executor's own future)."""
    reading = Reading()

    def run() -> None:
        if not reading.set_running_or_notify_cancel():
            return
        try:
            transcript = read_timed(transcriber, pcm)
        except BaseException as error:  # handed to whoever waits, as an executor would
            reading.set_exception(error)
            return
        reading.transcript = transcript
        reading.set_result(transcript.text)

    executor.submit(run)
    return reading


def transcript_of(
    heard: str | Transcript, early: concurrent.futures.Future[str] | None
) -> Transcript:
    """What a read gave: a late read's transcript, or an early read's text with the transcript
    it kept (the text alone when it kept none)."""
    if isinstance(heard, Transcript):
        return heard
    kept = early.transcript if isinstance(early, Reading) else None
    return kept if kept is not None and kept.text == heard else Transcript(heard)


def build_transcriber(
    config: CascadeConfig, choice: Choice, root: Path, log: Path | None = None
) -> Transcriber:
    """The engine of ``choice``, loaded and warmed; ``root`` anchors a relative
    ``transcriber.whispercpp.dir``, ``log`` takes whisper-server's output."""
    if choice.engine == "mlx":
        from pepin.voice_live.wake import MlxWhisper

        return MlxWhisper(choice.model, config.language, config.whisper_prompt)
    if choice.engine in STT_DEVICES:
        from pepin.voice_cascade import stt

        settings = replace(config.stt[choice.engine], model=choice.model)
        return stt.build(choice.engine, settings, config.language)
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
    once no call is in flight, and the old one closed. The report line gives the load time and
    this process's footprint before the load, with both engines, and after the old one closed.
    A choice that fails to load is reported and not tried again until ``wanted()`` changes; one
    no longer wanted once it has loaded (the person switched back meanwhile) is closed unused."""

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
        footprint: Callable[[], float] = footprint_mb,
        settle_s: float = SETTLE_S,
    ) -> None:
        """``build`` loads an engine; ``wanted`` is polled (:meth:`poll`, or every
        ``POLL_S`` after :meth:`start`); ``on_loaded`` hears each load's outcome (None: the
        engine is in use) on the loading thread; ``footprint`` reads the process's MB,
        ``settle_s`` after the old engine closed."""
        self.choice = choice
        self.on_loaded = on_loaded
        self._footprint, self._settle_s = footprint, settle_s
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

    def transcribe_timed(self, pcm: bytes) -> Transcript:
        """The words in ``pcm`` with their timings, by the engine in use (none from an engine
        that gives none)."""
        with self._lock:
            return read_timed(self._current, pcm)

    def poll(self) -> None:
        """Start loading the wanted engine if it differs from the one in use."""
        want = self._wanted()
        if want is None or want in (self.choice, self._loading, self._failed):
            return
        self._loading = want
        on_loaded = self.on_loaded  # this load's outcome goes to whoever asked for it
        self._spawn(lambda: self._load(want, on_loaded))

    def _load(self, want: Choice, on_loaded: Loaded | None = None) -> None:
        t0, before = time.monotonic(), self._footprint()
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
        loaded_s, both = time.monotonic() - t0, self._footprint()
        with self._lock:
            old, was = self._current, self.name
            self._current, self.choice = fresh, want
        if self._loading == want:
            self._loading = None
        self._failed = None
        _close(old)
        time.sleep(self._settle_s)
        after = self._footprint()
        self._report(
            f"transcriber: {was} -> {want.label} (loaded in {loaded_s:.1f} s; footprint"
            f" {_gb(before)} -> {_gb(both)} with both -> {_gb(after)} GB)"
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


def _gb(mb: float) -> str:
    return f"{mb / 1024:.2f}"


def _thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name="transcriber-load", daemon=True).start()


def _close(transcriber: Transcriber) -> None:
    close = getattr(transcriber, "close", None)
    if close is not None:
        close()
