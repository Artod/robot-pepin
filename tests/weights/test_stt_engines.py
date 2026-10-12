"""The cascade's speech-to-text engines on their real weights: each one, built off the reading
thread as a switch builds it, reads three of Artem's recorded utterances and prints its words,
its time per utterance, its load time and the process's footprint; then a live switch through
them shows the old engine's memory handed back.

    uv run pytest tests/weights --weights -s [-k gigaam]

An engine is skipped when its dependency group is not installed (``uv sync --group
stt-<engine>``) or the recordings (``scratch/prosody_1008/wavs``, local) are absent; weights
missing from the Hugging Face cache are fetched at the first load.
"""

from __future__ import annotations

import importlib.util
import statistics
import threading
import time
import wave
from pathlib import Path

import pytest

from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.stages import Transcriber
from pepin.voice_cascade.stt.runtime import footprint_mb
from pepin.voice_cascade.transcriber import (
    SETTLE_S,
    Choice,
    SwitchingTranscriber,
    build_transcriber,
    choice_of,
)

REPO = Path(__file__).resolve().parents[2]
MAIN = REPO.parents[2] if REPO.parent.parent.name == ".claude" else REPO  # a worktree's main
CONFIG = REPO / "config" / "voice_cascade.json"
WAVS = MAIN / "scratch" / "prosody_1008" / "wavs"
CUTS = (  # 4.1, 2.6 and 3.4 s: a greeting with an English word and the name, two questions
    "live_151845_01_hello_privet_pipin.wav",
    "live_151845_03_skolko_ty_zhresh.wav",
    "live_151845_09_a_funktsiya_perebivaniya.wav",
)
RUNTIMES = {
    "mlx": "mlx_whisper",
    "gigaam": "onnx_asr",
    "parakeet": "mlx_audio",
    "canary": "onnx_asr",
    "voxtral": "mlx_audio",
}
pytestmark = [pytest.mark.weights, pytest.mark.slow]


def pcm_of(path: Path) -> bytes:
    """A 16 kHz mono s16le WAV's samples."""
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000), path
        return w.readframes(w.getnframes())


def ready(engine: str) -> None:
    """Skip unless the engine's runtime and the recordings are here."""
    if importlib.util.find_spec(RUNTIMES[engine]) is None:
        pytest.skip(f"{engine}: its runtime is not installed (uv sync --group stt-{engine})")
    if not all((WAVS / cut).is_file() for cut in CUTS):
        pytest.skip(f"the recordings are not here ({WAVS})")


def loaded_off_thread(config: CascadeConfig, choice: Choice) -> Transcriber:
    """The engine built on a thread of its own, as :class:`SwitchingTranscriber` builds it."""
    built: list[Transcriber] = []
    thread = threading.Thread(target=lambda: built.append(build_transcriber(config, choice, MAIN)))
    thread.start()
    thread.join()
    assert built, f"{choice.label} failed to load (see the thread's error above)"
    return built[0]


@pytest.mark.parametrize("engine", list(RUNTIMES))
def test_an_engine_reads_three_recorded_utterances(engine: str) -> None:
    ready(engine)
    config = CascadeConfig.load(CONFIG)
    choice = choice_of(engine, config)
    assert choice is not None
    before, t0 = footprint_mb(), time.monotonic()
    transcriber = loaded_off_thread(config, choice)
    load_s, loaded = time.monotonic() - t0, footprint_mb()
    lines, ms = [], []
    for cut in CUTS:
        pcm = pcm_of(WAVS / cut)
        text = transcriber.transcribe(pcm)  # the first call of a length may compile
        for _ in range(2):
            start = time.monotonic()
            again = transcriber.transcribe(pcm)
            ms.append((time.monotonic() - start) * 1000)
        assert text and again == text, f"{choice.label} on {cut}: {text!r} then {again!r}"
        lines.append(f"  {len(pcm) / 32000:.1f} s  {ms[-1]:5.0f} ms  {text}")
    close = getattr(transcriber, "close", None)
    if close is not None:
        close()
    time.sleep(SETTLE_S)  # the GPU driver hands freed buffers back a moment later
    after = footprint_mb()
    print(
        f"\n{choice.label}: loaded in {load_s:.1f} s, footprint {before / 1024:.2f} ->"
        f" {loaded / 1024:.2f} GB loaded -> {after / 1024:.2f} GB closed;"
        f" {statistics.median(ms):.0f} ms per utterance (median of {len(ms)})\n" + "\n".join(lines)
    )


def test_a_switch_hands_the_old_engines_memory_back() -> None:
    order = ["gigaam", "parakeet", "mlx"]
    for engine in order:
        ready(engine)
    config = CascadeConfig.load(CONFIG)
    first = choice_of("mlx", config)
    assert first is not None
    lines: list[str] = []
    wanted: list[Choice | None] = [None]
    switch = SwitchingTranscriber(
        loaded_off_thread(config, first),
        first,
        lambda choice: build_transcriber(config, choice, MAIN),
        lambda: wanted[0],
        report=lines.append,
        spawn=lambda work: work(),
    )
    pcm = pcm_of(WAVS / CUTS[1])
    for engine in order:
        wanted[0] = choice_of(engine, config)
        switch.poll()
        assert switch.choice.engine == engine, lines
        assert switch.transcribe(pcm)
    switch.close()
    print("\n" + "\n".join(lines))
    for line, engine in zip(lines, order, strict=True):
        # "... footprint 2.95 -> 4.10 with both -> 1.30 GB)": closing the old engine frees it
        _, both, after = (float(x.split()[0]) for x in line.split("footprint ")[1].split(" -> "))
        assert both - after > 0.3, f"switching to {engine}, the old engine's memory stayed"
