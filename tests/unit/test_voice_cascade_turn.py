"""The end-of-turn model in the ears: asked once a pause, closing the utterance early when it
says so, the fixed silence otherwise; its worker never blocks and keeps only the latest."""

import time
from pathlib import Path

import numpy as np
import pytest

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.ears import Ears
from pepin.voice_cascade.fake import ScriptedTurn
from pepin.voice_cascade.loop import eot_of
from pepin.voice_cascade.player import Playback
from pepin.voice_cascade.timeline import report
from pepin.voice_cascade.turn import TurnWorker

FRAME_S = FRAME_SAMPLES / RATE
CONFIG = Path(__file__).resolve().parents[2] / "config" / "voice_cascade.json"


def tone(seconds: float) -> list[bytes]:
    t = np.arange(int(seconds * RATE)) / RATE
    pcm = (0.3 * 32767 * np.sin(2 * np.pi * 300 * t)).astype("<i2").tobytes()
    step = FRAME_SAMPLES * 2
    return [pcm[i : i + step] for i in range(0, len(pcm) - step + 1, step)]


def quiet(seconds: float) -> list[bytes]:
    return [b"\x00\x00" * FRAME_SAMPLES] * round(seconds / FRAME_S)


def ears_with(ps: list[float], **changes: object) -> tuple[Ears, ScriptedTurn]:
    detector = ScriptedTurn(ps)
    config = CascadeConfig().with_changes(turn_model="smart-turn-v3.2", **changes)
    playback: list[Playback] = ["silent"]
    return Ears(config, lambda: playback[0], TurnWorker(detector, inline=True)), detector


def segments(ears: Ears, frames: list[bytes]) -> list[tuple[float, object]]:
    out = []
    for i, frame in enumerate(frames):
        for event in ears.feed(frame, i * FRAME_S):
            if event.kind == "segment":
                out.append((i * FRAME_S, event))
    return out


def test_a_confident_model_closes_the_utterance_at_the_short_pause() -> None:
    ears, detector = ears_with([0.9])
    (closed,) = segments(ears, quiet(0.5) + tone(1.0) + quiet(1.0))
    t, event = closed
    assert t == pytest.approx(0.5 + 1.0 + 0.22, abs=0.03)  # the answer is read a frame later
    assert event.closed_by == "model" and event.p == 0.9
    assert event.saved_ms == pytest.approx(300, abs=30)
    assert event.segment.ended_s == pytest.approx(1.5 - FRAME_S, abs=0.03)
    assert len(detector.heard) == 1 and detector.heard[0] >= 1.0 * RATE * 2


def test_an_unsure_model_leaves_the_fixed_silence_to_close_it() -> None:
    ears, _ = ears_with([0.2])
    (closed,) = segments(ears, quiet(0.5) + tone(1.0) + quiet(1.0))
    t, event = closed
    assert t == pytest.approx(0.5 + 1.0 + 0.5, abs=0.03)
    assert event.closed_by == "silence" and event.p == 0.2 and event.saved_ms == 0.0


def test_speech_after_a_pause_cancels_its_question_and_the_next_pause_asks_again() -> None:
    ears, detector = ears_with([0.3, 0.8])
    frames = quiet(0.5) + tone(0.6) + quiet(0.3) + tone(0.6) + quiet(1.0)
    (closed,) = segments(ears, frames)
    t, event = closed
    assert len(detector.heard) == 2 and event.closed_by == "model" and event.p == 0.8
    assert t == pytest.approx(0.5 + 0.6 + 0.3 + 0.6 + 0.22, abs=0.03)


def test_without_a_model_nothing_changes() -> None:
    config = CascadeConfig()
    ears = Ears(config, lambda: "silent")
    (closed,) = segments(ears, quiet(0.5) + tone(1.0) + quiet(1.0))
    assert closed[1].closed_by is None and eot_of(closed[1]) is None


def test_the_worker_answers_on_its_thread_and_keeps_only_the_latest() -> None:
    detector = ScriptedTurn([0.4, 0.7, 0.9], delay_s=0.05)
    worker = TurnWorker(detector)
    started = time.perf_counter()
    worker.ask(1, b"\x00\x00" * 100)
    worker.ask(2, b"\x00\x00" * 100)  # replaces 1 if 1 has not started
    worker.ask(3, b"\x00\x00" * 100)
    assert time.perf_counter() - started < 0.02  # never blocks the mic
    deadline = time.monotonic() + 2.0
    while (worker.latest is None or worker.latest.id != 3) and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.close()
    assert worker.latest is not None and worker.latest.id == 3
    assert len(detector.heard) <= 2  # 2 never ran, or 1 never ran
    assert worker.latest.ms >= 50


def test_a_broken_model_leaves_no_answer() -> None:
    class Broken:
        def probability(self, pcm: bytes) -> float:
            raise RuntimeError("no model")

    worker = TurnWorker(Broken(), inline=True)
    worker.ask(1, b"")
    assert worker.latest is None


def test_the_config_takes_the_model_and_refuses_what_cannot_work() -> None:
    config = CascadeConfig.load(CONFIG, ["ears.turn_model=smart-turn-v3.2"])
    assert config.turn_model == "smart-turn-v3.2"
    assert (config.turn_min_silence_s, config.turn_window_s, config.turn_threshold) == (
        0.2,
        8.0,
        0.5,
    )
    assert CascadeConfig.load(CONFIG).turn_model is None
    for bad in (
        ["ears.turn_model=smart-turn-v9"],
        ["ears.turn_model=smart-turn-v3.2", "turn.min_silence_s=0.6"],
        ["turn.window_s=12"],
    ):
        with pytest.raises(ValueError):
            CascadeConfig.load(CONFIG, bad)


def test_the_report_counts_what_the_model_closed() -> None:
    turn = {
        "ms": {"closed": 230},
        "eot": {"closed_by": "model", "p": 0.95, "model_ms": 9.0, "saved_ms": 280},
    }
    silence = {"ms": {"closed": 500}, "eot": {"closed_by": "silence", "p": 0.3,
                                               "model_ms": 8.0, "saved_ms": 0}}  # fmt: skip
    lines = report([turn, silence])
    assert "end-of-turn model (1 of 2 closed by it):" in lines


@pytest.mark.slow
def test_smart_turn_itself_reads_a_phrase_when_its_weights_are_here() -> None:
    pytest.importorskip("onnxruntime")
    from huggingface_hub import hf_hub_download

    from pepin.voice_cascade.config import TURN_MODELS
    from pepin.voice_cascade.turn import SmartTurn

    repo, filename = TURN_MODELS["smart-turn-v3.2"]
    try:
        path = Path(hf_hub_download(repo, filename, local_files_only=True))
    except Exception:
        pytest.skip("Smart Turn weights not in the Hugging Face cache")
    model = SmartTurn(path=path)
    p = model.probability(b"".join(tone(1.0) + quiet(0.3)))
    assert 0.0 <= p <= 1.0
    worker = TurnWorker(model)
    worker.ask(7, b"".join(tone(0.5)))
    deadline = time.monotonic() + 5.0
    while worker.latest is None and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.close()
    assert worker.latest is not None and worker.latest.ms < 500
