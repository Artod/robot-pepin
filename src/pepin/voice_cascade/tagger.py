"""The tag line: how an utterance was said, against how this person usually speaks.

Praat (parselmouth) measures each utterance over its voiced span, the way
scratch/prosody_1008 did on Artem's speech: intensity, the pitch's median and its 5-95 % range
in semitones, the share of pauses of 150 ms or more; the speaking rate is syllables (vowel
letters of Whisper's words) per second of that span. A running baseline of the last ``window``
utterances (one per recognised person, :mod:`pepin.people`; one for all while nobody is)
turns each number into a z-score, and a z beyond ``z`` becomes a plain Russian word:
``[Артём: тише и медленнее обычного, интонация ровнее]``. The line describes and never
interprets (prosody.md section 7): no feeling is named, and nothing is said until the baseline
holds ``min_baseline`` utterances. A neural reading is added only when ``emotion_model`` is set:
audeering's arousal joins the z-scored numbers ("энергичнее"), emotion2vec+'s top class is
quoted as that model's label. When the transcriber times its words (GigaAM), a drawn-out vowel
is one more phrase, the word spelled as people write a stretch: ``растянуто: приииинтер``
(a vowel ``stretch_ratio`` times this person's usual vowel in its place and ``stretch_min_s``
long, :mod:`pepin.voice_cascade.timing`; the vowels join the same person's baseline).
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol

from pepin.audio_link import RATE
from pepin.voice_cascade import timing
from pepin.voice_cascade.config import ProsodyConfig
from pepin.voice_cascade.stages import Prosody, TimedWord
from pepin.voice_cascade.text import syllables

PITCH_FLOOR_HZ, PITCH_CEILING_HZ = 70.0, 400.0
ACTIVE_DB = 25.0  # the voiced span: frames within this of the utterance's 95th percentile
PAUSE_S = 0.15
# The smallest spread a feature's baseline is allowed, so a few alike utterances do not turn a
# small difference into a large z (units: dB, syllables/s, semitones, semitones, 0..1).
MIN_SD = {"loudness": 2.0, "rate": 0.5, "pitch": 1.0, "range": 1.5, "arousal": 0.03}
WORDS = {
    "loudness": ("громче", "тише"),
    "rate": ("быстрее", "медленнее"),
    "pitch": ("выше", "ниже"),
    "arousal": ("энергичнее", "спокойнее"),
}
RANGE_WORDS = ("живее", "ровнее")
PAUSED = 0.25  # pause ratio above which the line says so (absolute: the baseline is ~0)
DEFAULT_KEY = ""  # the one baseline while nobody is recognised (prosody.speaker's)
VOWEL_KINDS = ("vowel", "final")  # a word's vowels and a phrase's last, apart
VOWELS_PER_UTTERANCE = 8  # the vowel baseline keeps window x this many of each kind
MIN_VOWELS = 10  # a kind's median counts once it holds this many vowels
STRETCHED = "растянуто"


class EmotionModel(Protocol):
    """A neural reading of one utterance."""

    def read(self, pcm: bytes) -> tuple[float | None, str | None]:
        """(arousal 0..1 or None, a label with its score or None)."""
        ...


def measure(pcm: bytes) -> Prosody:
    """Praat's numbers for 16 kHz s16le mono ``pcm``."""
    import numpy as np
    import parselmouth

    x = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float64) / 32768
    if len(x) < RATE // 10:
        return Prosody(active_s=len(x) / RATE, loudness_db=-120.0)
    sound = parselmouth.Sound(x, sampling_frequency=RATE)
    intensity = sound.to_intensity(minimum_pitch=75, time_step=0.01)
    values = intensity.values[0]
    times = intensity.xs()
    if len(values) == 0:
        return Prosody(active_s=len(x) / RATE, loudness_db=-120.0)
    top = float(np.percentile(values, 95))
    loud = np.where(values > top - ACTIVE_DB)[0]
    t0, t1 = float(times[loud[0]]) - 0.02, float(times[loud[-1]]) + 0.02
    span = max(0.05, t1 - t0)
    inside = (times >= t0) & (times <= t1)
    quiet = (values < top - ACTIVE_DB) & inside
    paused, run = 0.0, 0
    for q in [*quiet.tolist(), False]:
        if q:
            run += 1
            continue
        if run * 0.01 >= PAUSE_S:
            paused += run * 0.01
        run = 0
    active = sound.extract_part(from_time=max(0.0, t0), to_time=min(sound.duration, t1))
    pitch = active.to_pitch_ac(
        time_step=0.01, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ
    )
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    median = range_st = None
    if len(f0) >= 5:
        st = 12 * np.log2(f0 / 100.0)
        median = float(np.median(st))
        range_st = float(np.percentile(st, 95) - np.percentile(st, 5))
    voiced = values[inside & (values > top - ACTIVE_DB)]
    return Prosody(
        active_s=span,
        loudness_db=float(np.mean(voiced)) if len(voiced) else top,
        f0_median_st=median,
        f0_range_st=range_st,
        pause_ratio=paused / span,
    )


def comparatives(words: list[str]) -> str:
    """``["тише", "медленнее"]`` -> ``"тише и медленнее"``."""
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " и " + words[-1]


class ProsodyTagger:
    """:class:`pepin.voice_cascade.stages.Tagger` on Praat's numbers and a running baseline."""

    def __init__(
        self,
        config: ProsodyConfig,
        *,
        emotion: EmotionModel | None = None,
        measurer: Callable[[bytes], Prosody] = measure,
    ) -> None:
        """``emotion``: the optional neural reading; ``measurer``: Praat (a fake in tests)."""
        self.config = config
        self.emotion = emotion
        self.measurer = measurer
        self.baselines: dict[str, dict[str, deque[float]]] = {}
        self.count = 0

    def baseline(self, key: str = DEFAULT_KEY) -> dict[str, deque[float]]:
        """One person's running baseline (``key``: whose; one for everyone when nobody is
        recognised)."""
        found = self.baselines.get(key)
        if found is None:
            window = self.config.window
            found = {name: deque(maxlen=window) for name in (*WORDS, "range")}
            for kind in VOWEL_KINDS:
                found[kind] = deque(maxlen=window * VOWELS_PER_UTTERANCE)
            self.baselines[key] = found
        return found

    def measure(self, pcm: bytes) -> Prosody:
        """Praat's numbers, and the neural reading when there is a model."""
        prosody = self.measurer(pcm)
        if self.emotion is None:
            return prosody
        arousal, label = self.emotion.read(pcm)
        return Prosody(
            prosody.active_s,
            prosody.loudness_db,
            prosody.f0_median_st,
            prosody.f0_range_st,
            prosody.pause_ratio,
            arousal,
            label,
        )

    def features(self, prosody: Prosody, text: str) -> dict[str, float]:
        """The numbers the baseline keeps (a missing one left out)."""
        out = {"loudness": prosody.loudness_db}
        if text.strip() and prosody.active_s > 0.3:
            out["rate"] = syllables(text) / prosody.active_s
        if prosody.f0_median_st is not None:
            out["pitch"] = prosody.f0_median_st
        if prosody.f0_range_st is not None:
            out["range"] = prosody.f0_range_st
        if prosody.arousal is not None:
            out["arousal"] = prosody.arousal
        return out

    def z(self, name: str, value: float, key: str = DEFAULT_KEY) -> float | None:
        """``value`` in this person's spreads from their mean (None: no baseline yet)."""
        seen = self.baseline(key)[name]
        if len(seen) < self.config.min_baseline:
            return None
        sd = max(statistics.pstdev(seen), MIN_SD[name])
        return (value - statistics.fmean(seen)) / sd

    def line(self, prosody: Prosody, text: str, keep: bool = True) -> str:
        """The tag line, then the utterance joins the baseline (``keep`` False: it does not, a
        line written before the utterance closed)."""
        body = self.describe(prosody, text, keep=keep)
        who = self.config.speaker or "голос"
        return f"[{who}: {body}]" if body else ""

    def describe(
        self, prosody: Prosody, text: str, key: str = DEFAULT_KEY, keep: bool = True
    ) -> str:
        """How the utterance was said against ``key``'s own baseline, in words ("" while the
        baseline is too short or nothing was measured); then it joins that baseline (``keep``
        False: it does not)."""
        if prosody.loudness_db <= -100 or not text.strip():
            return ""
        values = self.features(prosody, text)
        scores = {name: self.z(name, value, key) for name, value in values.items()}
        stretched = self.stretched(prosody.words, key)
        if keep:
            baseline = self.baseline(key)
            for name, value in values.items():
                if math.isfinite(value):
                    baseline[name].append(value)
            for kind, seconds in timing.samples(prosody.words):
                baseline[kind].append(seconds)
            self.count += 1
        if all(score is None for score in scores.values()):
            return f"{STRETCHED}: {stretched}" if stretched else ""
        limit = self.config.z
        said: list[str] = []
        for name, (up, down) in WORDS.items():
            score = scores.get(name)
            if score is not None and abs(score) >= limit:
                said.append(up if score > 0 else down)
        parts = [f"{comparatives(said)} обычного"] if said else []
        score = scores.get("range")
        if score is not None and abs(score) >= limit:
            parts.append("интонация " + (RANGE_WORDS[0] if score > 0 else RANGE_WORDS[1]))
        if prosody.pause_ratio >= PAUSED and prosody.active_s >= 1.5:
            parts.append("с паузами")  # noqa: RUF001
        if stretched:
            parts.append(f"{STRETCHED}: {stretched}")
        if prosody.label:
            parts.append(f"модель эмоций: {prosody.label}")
        return ", ".join(parts) if parts else "голос как обычно"

    def stretched(self, words: tuple[TimedWord, ...], key: str = DEFAULT_KEY) -> str:
        """The utterance's words with a drawn-out vowel against ``key``'s usual vowels, spelled
        as a stretch and joined (``приииинтер, дааа``); "" when none is, when the stretch is
        off (``stretch_ratio`` null) or before the baseline holds ``MIN_VOWELS`` of a kind."""
        ratio = self.config.stretch_ratio
        if ratio is None or not words:
            return ""
        baseline = self.baseline(key)
        usual = timing.medians({kind: baseline[kind] for kind in VOWEL_KINDS}, MIN_VOWELS)
        found = timing.stretches(words, usual, ratio=ratio, min_s=self.config.stretch_min_s)
        return ", ".join(s.spelled() for s in found)


class AudeeringArousal:
    """audeering's wav2vec2 dimensional model (arousal, dominance, valence ~0..1; CC-BY-NC-SA,
    research use) on the Mac's GPU; only arousal is used. Loaded at construction."""

    REPO = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

    def __init__(self, device: str = "mps") -> None:
        """Load the model (a 0.7 GB download the first time)."""
        import torch
        from torch import nn
        from transformers import Wav2Vec2FeatureExtractor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        class Head(nn.Module):  # the model card's regression head
            def __init__(self, config: Any) -> None:
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, x: Any) -> Any:
                return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

        pretrained: Any = Wav2Vec2PreTrainedModel  # untyped upstream: kept out of the checks

        class Model(pretrained):  # type: ignore[misc]
            def __init__(self, config: Any) -> None:
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = Head(config)
                self.post_init()

            def forward(self, input_values: Any) -> Any:
                return self.classifier(self.wav2vec2(input_values)[0].mean(dim=1))

        self._torch = torch
        self.device = device if torch.backends.mps.is_available() else "cpu"
        self._features = Wav2Vec2FeatureExtractor.from_pretrained(self.REPO)
        self._model = Model.from_pretrained(self.REPO).to(self.device).eval()
        self.read(b"\x00\x00" * RATE)

    def read(self, pcm: bytes) -> tuple[float | None, str | None]:
        """Arousal of ``pcm``; no label."""
        import numpy as np

        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        values = self._features(x, sampling_rate=RATE, return_tensors="pt")["input_values"]
        with self._torch.no_grad():
            out = self._model(values.to(self.device))
        return float(out[0][0]), None


class Emotion2Vec:
    """emotion2vec+ large (FunASR; 9 classes) on the Mac's GPU: its top class and score, quoted
    as that model's label. Needs ``funasr`` installed (not a project dependency)."""

    MODEL = "iic/emotion2vec_plus_large"

    def __init__(self, device: str = "mps") -> None:
        """Load the model (a 1.9 GB download the first time)."""
        from funasr import AutoModel

        self._model = AutoModel(model=self.MODEL, device=device, disable_update=True)
        self.read(b"\x00\x00" * RATE)

    def read(self, pcm: bytes) -> tuple[float | None, str | None]:
        """The top class of ``pcm`` with its score."""
        import numpy as np

        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        result = self._model.generate(input=x, granularity="utterance", disable_pbar=True)[0]
        labels, scores = result["labels"], result["scores"]
        best = int(np.argmax(scores))
        return None, f"{str(labels[best]).split('/')[-1]} {float(scores[best]):.2f}"


def emotion_model(name: str | None) -> EmotionModel | None:
    """The configured neural reader (``prosody.emotion_model``), loaded; None when unset."""
    if name is None:
        return None
    if name == "audeering":
        return AudeeringArousal()
    if name == "emotion2vec":
        return Emotion2Vec()
    raise ValueError(f"unknown emotion model {name!r}")
