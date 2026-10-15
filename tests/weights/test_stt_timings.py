"""GigaAM's timed read on its real weights: Artem's stress correction of 2026-10-12 ("Не принтер,
а при́нтер", the last и drawn out) comes out with every word timed and that word spelled
stretched, at the cost of a few milliseconds over the plain read.

    uv run pytest tests/weights/test_stt_timings.py --weights -s

Skipped without onnx-asr (``uv sync --group stt-gigaam``) or the recording (the conversation's
``in.wav`` under ``data/voice/20261012``, local); the weights come from the Hugging Face cache.
"""

# ruff: noqa: RUF002, RUF003 -- Russian speech is the data here, not a lookalike of Latin

from __future__ import annotations

import importlib.util
import json
import statistics
import time
import wave
from pathlib import Path

import pytest

from pepin.voice_cascade import timing
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.stt.gigaam import GigaAM

REPO = Path(__file__).resolve().parents[2]
MAIN = REPO.parents[2] if REPO.parent.parent.name == ".claude" else REPO  # a worktree's main
CONFIG = REPO / "config" / "voice_cascade.json"
TALK = MAIN / "data" / "voice" / "20261012" / "cascade_144219"
HEARD = 5  # "К принтер. А почему ты акценты неправильно ставишь? Принтер. Не принтер, а принтер."
USUAL = {"vowel": 0.10, "final": 0.13}  # Artem's medians over 162 utterances (stretch_eval.py)
pytestmark = [pytest.mark.weights, pytest.mark.slow]


def cut() -> bytes:
    """The utterance as the bench cut it: in.wav second = the wake utterance's length + the
    event's second; from 0.7 s before the logged start to the logged close."""
    rows = [json.loads(line) for line in (TALK / "events.jsonl").open()]
    heard = [r for r in rows if r["event"] == "heard"]
    end = float(heard[0]["seconds"]) + heard[HEARD]["t"] - heard[HEARD]["whisper_ms"] / 1000
    start = end - heard[HEARD]["seconds"] - 0.7
    with wave.open(str(TALK / "in.wav")) as w:
        rate = w.getframerate()
        w.setpos(round(start * rate))
        return w.readframes(round((end - start) * rate))


def test_the_drawn_out_word_of_a_stress_correction_is_spelled_stretched() -> None:
    if importlib.util.find_spec("onnx_asr") is None:
        pytest.skip("onnx-asr is not installed (uv sync --group stt-gigaam)")
    if not (TALK / "in.wav").is_file():
        pytest.skip(f"the recording is not here ({TALK})")
    pcm = cut()
    engine = GigaAM(CascadeConfig.load(CONFIG).stt["gigaam"])
    heard = engine.transcribe_timed(pcm)
    assert heard.text == engine.transcribe(pcm)
    printers = [w for w in heard.words if w.text.lower() == "принтер"]
    assert len(printers) >= 3, heard.text
    called = timing.stretches(heard.words, USUAL, ratio=2.0, min_s=0.25)
    assert [s.spelled() for s in called] == ["прииинтер"]
    assert called[0].word is printers[-1] and called[0].vowel.seconds > 0.3
    plain, timed = [], []
    for _ in range(5):
        t0 = time.perf_counter()
        engine.transcribe(pcm)
        t1 = time.perf_counter()
        engine.transcribe_timed(pcm)
        plain.append(t1 - t0)
        timed.append(time.perf_counter() - t1)
    engine.close()
    print(
        f"\n{heard.text}\n{len(pcm) / 32000:.1f} s: plain {statistics.median(plain) * 1000:.0f} ms,"
        f" timed {statistics.median(timed) * 1000:.0f} ms (median of 5)\n"
        + "\n".join(
            f"  {w.text:12} {w.start:5.2f}-{w.end:5.2f} lp {w.logprob:6.2f}  vowels "
            + " ".join(f"{v.seconds:.2f}" for v in w.vowels)
            for w in heard.words
        )
    )
