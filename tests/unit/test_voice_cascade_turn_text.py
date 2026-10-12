"""Two gates for the end of a turn: the words judged by the rules or a text model, the decision
table, the ears closing early, at the fixed silence or after a longer wait, and the knobs."""

# ruff: noqa: RUF001, RUF003 -- Russian speech is the data here, not a lookalike of Latin

from __future__ import annotations

import concurrent.futures
from pathlib import Path

import numpy as np
import pytest

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.ears import EarEvent, Ears
from pepin.voice_cascade.fake import ScriptedTurn
from pepin.voice_cascade.loop import eot_of
from pepin.voice_cascade.timeline import report
from pepin.voice_cascade.turn import TurnWorker
from pepin.voice_cascade.turn_text import (
    Judgement,
    NamoJudge,
    RulesJudge,
    build_judge,
    two_gates,
    words_of,
)

FRAME_S = FRAME_SAMPLES / RATE
CONFIG = Path(__file__).resolve().parents[2] / "config" / "voice_cascade.json"


def tone(seconds: float) -> list[bytes]:
    t = np.arange(int(seconds * RATE)) / RATE
    pcm = (0.3 * 32767 * np.sin(2 * np.pi * 300 * t)).astype("<i2").tobytes()
    step = FRAME_SAMPLES * 2
    return [pcm[i : i + step] for i in range(0, len(pcm) - step + 1, step)]


def quiet(seconds: float) -> list[bytes]:
    return [b"\x00\x00" * FRAME_SAMPLES] * round(seconds / FRAME_S)


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("Как у тебя дела?", "complete"),
        ("Заверши сессию.", "complete"),
        ("А что?", "complete"),  # a question word asking
        ("Да.", "complete"),
        ("Привет, Пепин.", "complete"),
        ("Ну, отвечай быстро, и", "incomplete"),  # a trailing comma or a conjunction
        ("Дай мне текст на.", "incomplete"),  # a preposition
        ("Почему у тебя английский лучше, чем.", "incomplete"),
        ("Я сейчас не.", "incomplete"),  # a negation
        ("А ну-ка, э-э.", "incomplete"),  # a hesitation
        ("Это очень.", "incomplete"),  # an intensifier without its word
        ("Я хочу, потому что.", "incomplete"),
        ("Ну.", "incomplete"),  # openers alone
        ("Пипин.", "incomplete"),  # the name alone: the request follows
        ("Скажи, где", "incomplete"),  # a question word asking nothing yet
        ("Скажи, где.", "incomplete"),
        ("Привет,", "incomplete"),
        ("Лангаш.", "unsure"),  # one word
        ("как у тебя дела", "unsure"),  # no final mark (an engine without punctuation)
        ("", "unsure"),
    ],
)
def test_the_rules_read_russian(text: str, verdict: str) -> None:
    assert RulesJudge().judge(text).verdict == verdict


def test_words_keep_hyphens_and_read_yo_as_ie() -> None:
    assert words_of("Ну-ка, ещё э-э раз!") == ["ну-ка", "еще", "э-э", "раз"]


@pytest.mark.parametrize(
    ("verdict", "acoustic", "quiet_s", "closed_by"),
    [
        ("complete", "agrees", 0.25, "both"),
        ("complete", "disagrees", 0.4, "silence"),
        ("complete", "unknown", 0.4, "silence"),
        ("unsure", "agrees", 0.4, "silence"),
        (None, "agrees", 0.4, "silence"),
        ("incomplete", "agrees", 0.8, "wait"),
        ("incomplete", "disagrees", 0.8, "wait"),
    ],
)
def test_the_decision_table(
    verdict: str | None, acoustic: str, quiet_s: float, closed_by: str
) -> None:
    deadline = two_gates(
        verdict,  # type: ignore[arg-type]
        acoustic,  # type: ignore[arg-type]
        early_close_s=0.25,
        silence_s=0.4,
        incomplete_s=0.8,
    )
    assert (deadline.quiet_s, deadline.closed_by) == (quiet_s, closed_by)


class FixedP:
    def __init__(self, p: float | None) -> None:
        self.p = p
        self.texts: list[str] = []

    def p_end(self, text: str) -> float:
        self.texts.append(text)
        if self.p is None:
            raise RuntimeError("no model")
        return self.p


@pytest.mark.parametrize(
    ("text", "p", "verdict"),
    [
        ("Как у тебя дела?", 0.9, "complete"),
        ("Как у тебя дела?", 0.4, "complete"),  # the rules' verdict stands
        ("Как у тебя дела?", 0.01, "incomplete"),  # the model's veto
        ("Лангаш.", 0.9, "complete"),  # a lone word made whole
        ("Лангаш.", 0.4, "unsure"),
    ],
)
def test_the_model_is_a_second_opinion_over_the_rules(text: str, p: float, verdict: str) -> None:
    judge = NamoJudge(FixedP(p), complete_p=0.7, incomplete_p=0.02)
    out = judge.judge(text)
    assert out.verdict == verdict and out.p == p


def test_the_rules_incomplete_wins_over_the_model_and_stand_in_when_it_breaks() -> None:
    model = FixedP(0.99)
    judge = NamoJudge(model, complete_p=0.7, incomplete_p=0.1)
    assert judge.judge("Дай мне текст на").verdict == "incomplete"
    assert model.texts == []  # never asked
    broken = NamoJudge(FixedP(None), complete_p=0.7, incomplete_p=0.1)
    assert broken.judge("Как у тебя дела?").verdict == "complete"  # the rules' verdict


def test_build_judge_falls_back_to_the_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    import pepin.voice_cascade.turn_text as turn_text

    assert isinstance(build_judge(None), RulesJudge)

    def unloadable(name: str) -> None:
        raise OSError("no weights")

    monkeypatch.setattr(turn_text, "NamoClassifier", unloadable)
    assert isinstance(build_judge("namo-ru"), RulesJudge)


class Reader:
    """The transcriber's early read: the texts in order, done at once (or held for
    :meth:`land`; ``running``: already on the transcriber, so a drop cannot cancel it)."""

    def __init__(self, texts: list[str], hold: bool = False, running: bool = False) -> None:
        self.texts, self.hold, self.running = texts, hold, running
        self.asked = 0
        self.held: list[tuple[concurrent.futures.Future[str], str]] = []

    def __call__(self, pcm: bytes) -> concurrent.futures.Future[str]:
        text = self.texts[min(self.asked, len(self.texts) - 1)]
        self.asked += 1
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        if self.running:
            future.set_running_or_notify_cancel()
        if self.hold:
            self.held.append((future, text))
        else:
            future.set_result(text)
        return future

    def land(self) -> None:
        for future, text in self.held:
            future.set_result(text)
        self.held = []


class Recorder:
    """A judge that keeps the texts it was given."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.rules = RulesJudge()

    def judge(self, text: str) -> Judgement:
        self.texts.append(text)
        return self.rules.judge(text)


def gated(
    texts: list[str], *, ps: list[float] | None = None, hold: bool = False, **changes: object
) -> tuple[Ears, Reader]:
    config = CascadeConfig().with_changes(
        turn_model="two-gates",
        silence_s=0.4,
        early_s=0.15,
        turn_early_close_s=0.25,
        turn_incomplete_s=0.8,
        **changes,
    )
    turn = TurnWorker(ScriptedTurn(ps), inline=True) if ps is not None else None
    reader = Reader(texts, hold)
    return Ears(config, lambda: "silent", turn, reader, judge=RulesJudge()), reader


def closes(
    ears: Ears, frames: list[bytes], *, at: dict[int, object] | None = None
) -> list[tuple[float, EarEvent]]:
    out = []
    for i, frame in enumerate(frames):
        if at and i in at:
            at[i]()  # type: ignore[operator]
        for event in ears.feed(frame, i * FRAME_S):
            if event.kind == "segment":
                out.append((i * FRAME_S, event))
    return out


SPEECH = quiet(0.5) + tone(1.0)
END = 0.5 + 1.0 - FRAME_S  # the last loud frame


def test_finished_words_close_the_turn_early() -> None:
    ears, _ = gated(["Как у тебя дела?"])
    ((t, event),) = closes(ears, SPEECH + quiet(1.2))
    assert t == pytest.approx(END + 0.25, abs=0.03)
    assert (event.closed_by, event.verdict) == ("both", "complete")
    assert event.saved_ms == pytest.approx(150, abs=25)
    assert event.segment is not None and event.segment.ended_s == pytest.approx(END, abs=0.03)
    assert event.early is not None and event.early.result() == "Как у тебя дела?"


def test_unfinished_words_wait_longer() -> None:
    ears, _ = gated(["Дай мне текст на"])
    ((t, event),) = closes(ears, SPEECH + quiet(1.2))
    assert t == pytest.approx(END + 0.8, abs=0.03)
    assert (event.closed_by, event.verdict) == ("wait", "incomplete")
    assert event.saved_ms == pytest.approx(-400, abs=25)
    eot = eot_of(event)
    assert eot is not None and eot["verdict"] == "incomplete" and "на" in eot["why"]


def test_unsure_words_keep_the_fixed_silence() -> None:
    ears, _ = gated(["Лангаш."])
    ((t, event),) = closes(ears, SPEECH + quiet(1.2))
    assert t == pytest.approx(END + 0.4, abs=0.03)
    assert (event.closed_by, event.verdict) == ("silence", "unsure")


def test_the_sound_must_agree_before_an_early_close() -> None:
    ears, _ = gated(["Как у тебя дела?"], ps=[0.3], turn_acoustic="smart-turn-v3.2",
                    turn_threshold=0.5)  # fmt: skip
    ((t, event),) = closes(ears, SPEECH + quiet(1.2))
    assert t == pytest.approx(END + 0.4, abs=0.03) and event.closed_by == "silence"
    assert event.p == 0.3
    ears, _ = gated(["Как у тебя дела?"], ps=[0.9], turn_acoustic="smart-turn-v3.2",
                    turn_threshold=0.5)  # fmt: skip
    ((t, event),) = closes(ears, SPEECH + quiet(1.2))
    assert t == pytest.approx(END + 0.25, abs=0.03) and event.closed_by == "both"


def test_words_landing_late_close_on_their_frame() -> None:
    ears, reader = gated(["Как у тебя дела?"], hold=True)
    land_at = round((END + 0.32) / FRAME_S)  # the read lands 0.32 s into the pause
    ((t, event),) = closes(ears, SPEECH + quiet(1.2), at={land_at: reader.land})
    assert t == pytest.approx(END + 0.32, abs=0.03) and event.closed_by == "both"


def test_a_pause_talked_through_drops_its_verdict() -> None:
    # finished-looking words in a 0.2 s pause, then the person goes on and stops unfinished
    ears, reader = gated(["Как у тебя.", "Как у тебя дела, и"])
    frames = SPEECH + quiet(0.2) + tone(0.6) + quiet(1.2)
    ((t, event),) = closes(ears, frames)
    assert reader.asked == 2
    assert event.closed_by == "wait" and t == pytest.approx(END + 0.2 + 0.6 + 0.8, abs=0.03)


def test_a_continuation_is_judged_after_the_words_it_continues() -> None:
    config = CascadeConfig().with_changes(turn_model="two-gates", early_s=0.15)
    recorder = Recorder()
    reader = Reader(["Как у тебя", "дела."])
    ears = Ears(config, lambda: "silent", None, reader, judge=recorder)
    # "Как у тебя" (no mark: unsure, closed at 0.4), the voice back 0.3 s later
    frames = SPEECH + quiet(0.7) + tone(0.6) + quiet(1.2)
    closed = closes(ears, frames)
    assert len(closed) == 2 and closed[1][1].closed_by == "both"
    assert recorder.texts == ["Как у тебя", "Как у тебя дела."]
    # back after the continuation window: judged alone
    recorder.texts.clear()
    reader = Reader(["Как у тебя", "дела."])
    ears = Ears(config, lambda: "silent", None, reader, judge=recorder)
    closes(ears, SPEECH + quiet(1.6) + tone(0.6) + quiet(1.2))
    assert recorder.texts == ["Как у тебя", "дела."]


def told(
    texts: list[str],
    frames: list[bytes],
    *,
    two_gates: bool = True,
    at: dict[int, object] | None = None,
) -> tuple[list[EarEvent], list[tuple[float, EarEvent]]]:
    """The ``words`` events the ears tell (two-gates or the silence rule) and the closes."""
    config = CascadeConfig().with_changes(
        turn_model="two-gates" if two_gates else None, silence_s=0.4, early_s=0.15
    )
    heard: list[EarEvent] = []
    reader = Reader(texts, hold=at is not None, running=True)
    ears = Ears(config, lambda: "silent", None, reader, judge=RulesJudge(), words=heard.append)
    landing = {i: reader.land for i in at} if at else None
    return heard, closes(ears, frames, at=landing)


def test_the_early_words_are_told_with_their_verdict_and_the_audio_so_far() -> None:
    heard, ((_, segment),) = told(["Как у тебя дела?"], SPEECH + quiet(1.2))
    (words,) = heard
    assert words.kind == "words" and words.text == "Как у тебя дела?"
    assert (words.verdict, words.verdict_why) == ("complete", segment.verdict_why)
    assert words.t == pytest.approx(END + 0.15, abs=0.03)
    assert words.segment is not None and words.segment.ended_s == pytest.approx(END, abs=0.03)
    assert words.early is segment.early  # the read that closed the utterance
    heard, _ = told(["Как у тебя дела?"], SPEECH + quiet(1.2), two_gates=False)
    assert [(w.text, w.verdict) for w in heard] == [("Как у тебя дела?", None)]


def test_a_read_landing_after_the_voice_came_back_is_never_told() -> None:
    frames = SPEECH + quiet(0.2) + tone(0.6) + quiet(1.2)
    back = round((0.5 + 1.0 + 0.2 + 0.3) / FRAME_S)  # in the second tone: the first read lands
    end = round((0.5 + 1.0 + 0.2 + 0.6 + 0.3) / FRAME_S)  # 0.3 s into the last pause
    heard, closed = told(["Как у тебя.", "Как у тебя дела, и"], frames, at={back: 0, end: 0})
    assert [(w.text, w.verdict) for w in heard] == [("Как у тебя дела, и", "incomplete")]
    assert len(closed) == 1 and heard[0].early is closed[0][1].early


def test_without_a_judge_two_gates_is_the_fixed_silence() -> None:
    config = CascadeConfig().with_changes(turn_model="two-gates")
    ears = Ears(config, lambda: "silent", None, Reader(["Как у тебя дела?"]))
    ((t, event),) = closes(ears, SPEECH + quiet(1.2))
    assert t == pytest.approx(END + 0.4, abs=0.03) and event.verdict is None


def test_the_config_takes_the_knobs_and_refuses_what_cannot_work() -> None:
    config = CascadeConfig.load(CONFIG, ["ears.turn_model=two-gates"])
    assert config.turn_model == "two-gates" and config.turn_acoustic == "pause"
    assert (config.turn_early_close_s, config.turn_incomplete_s) == (0.25, 0.8)
    assert config.turn_text_model is None
    named = CascadeConfig.load(CONFIG, ["ears.turn_model=two-gates", "turn.text_model=namo-ru",
                                        "turn.acoustic=smart-turn-v3.2"])  # fmt: skip
    assert (named.turn_text_model, named.turn_acoustic) == ("namo-ru", "smart-turn-v3.2")
    # silence alone may go under the early close: the order is two-gates' only
    assert CascadeConfig.load(CONFIG, ["ears.silence_s=0.22"]).silence_s == 0.22
    for bad in (
        ["ears.turn_model=two-gates", "turn.early_close_s=0.5"],  # over ears.silence_s
        ["ears.turn_model=two-gates", "turn.incomplete_s=0.3"],  # under ears.silence_s
        ["ears.turn_model=two-gates", "transcriber.early_s=0.3"],  # read after the early close
        ["ears.turn_model=two-gates", "transcriber.early_s=0"],  # no words at all
        ["ears.turn_model=two-gates", "turn.acoustic=smart-turn-v3.2", "turn.min_silence_s=0.3"],
        ["turn.acoustic=loudness"],
        ["turn.text_model=livekit"],
        ["turn.complete_p=0.1", "turn.incomplete_p=0.2"],
    ):
        with pytest.raises(ValueError):
            CascadeConfig.load(CONFIG, bad)


def test_the_report_counts_early_closes_and_waits() -> None:
    rows = [
        {"ms": {"closed": 250}, "eot": {"closed_by": "both", "p": None, "model_ms": None,
                                         "saved_ms": 150, "verdict": "complete"}},
        {"ms": {"closed": 800}, "eot": {"closed_by": "wait", "p": None, "model_ms": None,
                                         "saved_ms": -400, "verdict": "incomplete"}},
    ]  # fmt: skip
    lines = report(rows)
    assert "end-of-turn model (1 of 2 closed by it):" in lines
    assert "1 of 2 waited for unfinished words" in lines


@pytest.mark.slow
def test_the_namo_model_itself_when_its_weights_are_here() -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from huggingface_hub import hf_hub_download

    from pepin.voice_cascade.config import TEXT_MODELS
    from pepin.voice_cascade.turn_text import NamoClassifier

    repo = TEXT_MODELS["namo-ru"]
    try:
        model = Path(hf_hub_download(repo, "model_quant.onnx", local_files_only=True))
        hf_hub_download(repo, "tokenizer.json", local_files_only=True)
    except Exception:
        pytest.skip("Namo weights not in the Hugging Face cache")
    classifier = NamoClassifier(directory=model.parent)
    finished = classifier.p_end("Как у тебя дела?")
    dangling = classifier.p_end("Дай мне текст на французском языке, чтоб посмотреть, как")
    assert 0.0 <= dangling < finished <= 1.0
