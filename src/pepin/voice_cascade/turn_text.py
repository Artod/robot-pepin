"""The end of a turn by its words: is the transcript so far a finished thought?

The transcriber reads an utterance early, a moment into a pause (``transcriber.early_s``); a
judge reads that text and says ``complete``, ``incomplete`` or ``unsure``. The ears combine the
verdict with the sound (:func:`two_gates`): the words finished and the sound agreeing close the
utterance sooner than the fixed silence, unfinished words make it wait longer (``turn`` block).

:class:`RulesJudge` is a handful of Russian rules: a last word that cannot end a sentence (a
conjunction, a preposition, a negation, a hesitation, an intensifier without its word), an
utterance of openers and hesitations alone (or the robot's name), a question word not read as a
question, a trailing comma. GigaAM's e2e models end every text, a prefix too, with a full stop or a
question mark, so punctuation alone proves nothing; it only marks the text as read.
:class:`NamoJudge` is VideoSDK's Namo Turn Detector v1 for Russian (Apache-2.0; DistilBERT,
136 MB int8 ONNX, ~13 ms a call on the CPU here), a second opinion over the rules: its low p
vetoes an early close, its high p makes a lone word whole; the rules stand in when it fails.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pepin.voice_cascade.config import TEXT_MODELS
from pepin.voice_live.wake import is_name

logger = logging.getLogger(__name__)

TextVerdict = Literal["complete", "incomplete", "unsure"]

_LETTER = r"[a-z\N{CYRILLIC SMALL LETTER A}-\N{CYRILLIC SMALL LETTER YA}]"
_WORD = re.compile(f"{_LETTER}+(?:-{_LETTER}+)*")
_END = re.compile(r"[.?!,;:\N{HORIZONTAL ELLIPSIS}\N{EM DASH}-]+$")


def _set(words: str) -> frozenset[str]:
    return frozenset(words.split())


# words a Russian sentence does not end on; the ones that double as adverbs or replies
# ("против", "после", "да", "пока", "бы", "вот", "надо") are left out on purpose
CONJUNCTIONS = _set(
    "и а но или либо что чтобы чтоб потому поэтому если когда хотя будто словно зато "  # noqa: RUF001
    "причем притом чем нежели ибо однако"
)
PREPOSITIONS = _set(
    "в во на с со к ко у о об обо от ото до по за из изо для без про под подо над "  # noqa: RUF001
    "при через между перед передо среди кроме вместо ради сквозь"
)
PARTICLES = _set("не ни")
HESITATIONS = _set("э ээ эээ э-э э-э-э эм эмм м мм ммм а-а а-а-а ну-у н-ну типа")  # noqa: RUF001
INTENSIFIERS = _set("очень более менее самый самая самое самые наиболее слишком")
OPENERS = _set("а ну ну-ка а-ну-ка э ээ э-э вот слушай смотри знаешь кстати короче значит так")  # noqa: RUF001
QUESTION_WORDS = _set(
    "кто что где куда откуда когда почему зачем как какой какая какое какие сколько чей"
)
DANGLING = CONJUNCTIONS | PREPOSITIONS | PARTICLES | HESITATIONS | INTENSIFIERS
DANGLING_PAIRS = frozenset({("потому", "что"), ("то", "есть"), ("так", "что"), ("как", "бы")})
# one-word turns that are whole on their own
WHOLE_WORDS = _set(
    "да нет ладно хорошо спасибо пожалуйста окей ок пока привет все понятно конечно точно "
    "давай стоп хватит ага угу верно согласен отлично здравствуй"  # noqa: RUF001
)


@dataclass(frozen=True)
class Judgement:
    """A judge's verdict on one text, why (the rule or the model's p) and its milliseconds."""

    verdict: TextVerdict
    why: str
    p: float | None = None
    ms: float = 0.0


class TextJudge(Protocol):
    """Whether a transcript so far is a finished turn."""

    def judge(self, text: str) -> Judgement:
        """The verdict on ``text`` (an engine's transcript, punctuation and capitals kept)."""
        ...


def words_of(text: str) -> list[str]:
    """Lower-case words (hyphenated ones kept whole), the letter yo read as ie."""
    yo, ie = "\N{CYRILLIC SMALL LETTER IO}", "\N{CYRILLIC SMALL LETTER IE}"
    return _WORD.findall(text.lower().replace(yo, ie))


class RulesJudge:
    """:class:`TextJudge` by the rules of this module's notes; microseconds a call."""

    def judge(self, text: str) -> Judgement:
        """``incomplete`` on a rule, ``complete`` for a read sentence of two words or more (or
        one whole word), ``unsure`` otherwise (nothing read, a lone word, no final mark)."""
        said = words_of(text)
        stripped = text.strip()
        end = _END.search(stripped)
        mark = end.group(0) if end else ""
        if not said:
            return Judgement("unsure", "no words")
        last = said[-1]
        asked = "?" in mark
        if mark.endswith((",", ";", ":", "-", "\N{EM DASH}")):
            return Judgement("incomplete", f"ends on {mark!r}")
        if last in QUESTION_WORDS:  # read as a question it asks; otherwise its clause follows
            if not asked:
                return Judgement("incomplete", f"{last!r} asks nothing yet")
        elif last in DANGLING:
            return Judgement("incomplete", f"ends on {last!r}")
        if len(said) >= 2 and (said[-2], last) in DANGLING_PAIRS:
            return Judgement("incomplete", f"ends on {said[-2]} {last}")
        if all(w in OPENERS or w in HESITATIONS or is_name(w) for w in said):
            return Judgement("incomplete", "openers only")
        if not mark or mark.endswith("\N{HORIZONTAL ELLIPSIS}"):
            return Judgement("unsure", "no final mark")
        if len(said) == 1 and last not in WHOLE_WORDS:
            return Judgement("unsure", "one word")
        return Judgement("complete", f"ends on {last!r}{mark}")


class Classifier(Protocol):
    """A text model's probability that a text ends its turn."""

    def p_end(self, text: str) -> float:
        """For ``text``."""
        ...


class NamoJudge:
    """:class:`TextJudge` of the rules with a :class:`Classifier` (the Namo model) as a second
    opinion: the rules' incomplete stands; the model's p at most ``incomplete_p`` vetoes (wait),
    at least ``complete_p`` promotes the rules' unsure (a lone word) to complete; otherwise the
    rules' verdict. The rules alone when the model fails. As the judge on its own the model
    scored short finished turns low (2026-10-12: complete at 42 % of true ends, the rules 83 %)."""

    def __init__(
        self,
        model: Classifier,
        *,
        complete_p: float,
        incomplete_p: float,
        rules: RulesJudge | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """``model`` loaded and warmed."""
        self.model = model
        self.complete_p, self.incomplete_p = complete_p, incomplete_p
        self.rules = rules or RulesJudge()
        self.clock = clock

    def judge(self, text: str) -> Judgement:
        """See the class."""
        t0 = self.clock()
        ruled = self.rules.judge(text)
        if ruled.verdict == "incomplete" or ruled.why == "no words":
            return ruled
        try:
            p = self.model.p_end(text)
        except Exception:  # a broken model leaves the rules to judge
            logger.exception("end-of-turn text model")
            return ruled
        ms = (self.clock() - t0) * 1000
        if p <= self.incomplete_p:
            return Judgement("incomplete", f"p {p:.2f}", p, ms)
        if p >= self.complete_p and ruled.verdict == "unsure":
            return Judgement("complete", f"p {p:.2f}", p, ms)
        return Judgement(ruled.verdict, f"{ruled.why}, p {p:.2f}", p, ms)


class NamoClassifier:
    """:class:`Classifier` over a Namo Turn Detector v1 ONNX build (onnxruntime on the CPU, the
    tokenizer by the ``tokenizers`` library). Loaded and warmed at construction."""

    def __init__(self, name: str = "namo-ru", directory: Path | None = None) -> None:
        """``name``: one of :data:`pepin.voice_cascade.config.TEXT_MODELS` (fetched from
        Hugging Face once, 139 MB);
        ``directory``: a local copy of the repo's ``model_quant.onnx`` and ``tokenizer.json``
        instead."""
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if directory is None:
            from huggingface_hub import hf_hub_download

            repo = TEXT_MODELS[name]
            model = Path(hf_hub_download(repo, "model_quant.onnx"))
            tokenizer = Path(hf_hub_download(repo, "tokenizer.json"))
        else:
            model, tokenizer = directory / "model_quant.onnx", directory / "tokenizer.json"
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 2  # the transcriber's thread runs it, beside the voice
        self._np = np
        self._tokenizer: Any = Tokenizer.from_file(str(tokenizer))
        self._tokenizer.enable_truncation(512)
        self._session: Any = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.p_end("Привет.")

    def p_end(self, text: str) -> float:
        """The model's p for its label 1 (end of turn)."""
        np = self._np
        encoded = self._tokenizer.encode(text)
        feed = {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
        }
        logits = self._session.run(None, feed)[0][0].astype(np.float64)
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())


def build_judge(
    model: str | None, *, complete_p: float = 0.7, incomplete_p: float = 0.02
) -> TextJudge:
    """The judge of ``turn.text_model``: None the rules, a
    :data:`pepin.voice_cascade.config.TEXT_MODELS` name the model over the rules (the rules
    alone when it cannot load)."""
    if model is None:
        return RulesJudge()
    try:
        classifier = NamoClassifier(model)
    except Exception:
        logger.exception("end-of-turn text model %s: the rules judge instead", model)
        return RulesJudge()
    return NamoJudge(classifier, complete_p=complete_p, incomplete_p=incomplete_p)


Acoustic = Literal["agrees", "disagrees", "unknown"]
ClosedBy = Literal["both", "silence", "wait"]


@dataclass(frozen=True)
class Deadline:
    """How much quiet closes the pause in progress, and what will have closed it."""

    quiet_s: float
    closed_by: ClosedBy


def two_gates(
    verdict: TextVerdict | None,
    acoustic: Acoustic,
    *,
    early_close_s: float,
    silence_s: float,
    incomplete_s: float,
) -> Deadline:
    """The decision table: the words complete and the sound agreeing close at
    ``early_close_s``; the words incomplete wait ``incomplete_s``; anything else (unsure, not
    read yet, the sound disagreeing or not heard yet) keeps the fixed ``silence_s``."""
    if verdict == "incomplete":
        return Deadline(incomplete_s, "wait")
    if verdict == "complete" and acoustic == "agrees":
        return Deadline(early_close_s, "both")
    return Deadline(silence_s, "silence")
