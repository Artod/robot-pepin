"""Expression and attention over seconds, never from one frame.

A single-frame emotion label is noise (annotators agree on about 60 % of AffectNet's images), so
the published expression is the mean of the last ``window_s`` of probabilities and
valence/arousal. Looking at the robot is head yaw and pitch within a cone, by majority over the
last ``look_window_s``; talking is the jaw moving: the standard deviation of MediaPipe's
``jawOpen`` over the last ``talk_window_s``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from pepin.face_sensor.config import AttentionConfig, ExpressionConfig
from pepin.face_sensor.models import EMOTIONS, Expression, Mesh


@dataclass(frozen=True)
class SmoothExpression:
    """The window's mean: the top label, its mean probability, valence, arousal, sample count."""

    top: str
    p: float
    valence: float
    arousal: float
    samples: int

    def to_json(self) -> dict[str, object]:
        """The published form (REPORT's names: ``p_3s`` and so on, for a 3 s window)."""
        return {
            "top": self.top,
            "p_3s": round(self.p, 2),
            "valence_3s": round(self.valence, 2),
            "arousal_3s": round(self.arousal, 2),
        }


class ExpressionWindow:
    """The expressions of the last ``window_s`` seconds."""

    def __init__(self, cfg: ExpressionConfig) -> None:
        """An empty window."""
        self.cfg = cfg
        self._samples: deque[tuple[float, Expression]] = deque()

    def add(self, stamp: float, expression: Expression) -> None:
        """One frame's expression."""
        self._samples.append((stamp, expression))
        self._trim(stamp)

    def mean(self, now: float) -> SmoothExpression | None:
        """The window's mean at ``now``; ``None`` with fewer than ``min_samples`` in it."""
        self._trim(now)
        if len(self._samples) < self.cfg.min_samples:
            return None
        probs = np.mean([e.probs for _, e in self._samples], axis=0)
        i = int(np.argmax(probs))
        return SmoothExpression(
            top=EMOTIONS[i],
            p=float(probs[i]),
            valence=float(np.mean([e.valence for _, e in self._samples])),
            arousal=float(np.mean([e.arousal for _, e in self._samples])),
            samples=len(self._samples),
        )

    def _trim(self, now: float) -> None:
        while self._samples and self._samples[0][0] < now - self.cfg.window_s:
            self._samples.popleft()


def facing(mesh: Mesh, cfg: AttentionConfig) -> bool:
    """Whether the head points at the camera: yaw and pitch both inside the cone."""
    return abs(mesh.yaw) <= cfg.max_yaw_deg and abs(mesh.pitch) <= cfg.max_pitch_deg


@dataclass(frozen=True)
class AttentionState:
    """Looking at the robot and talking; ``None`` where the window holds too little to say."""

    looking: bool | None
    talking: bool | None
    jaw_std: float | None

    def to_json(self) -> dict[str, object]:
        """The published form: ``{"looking", "talking"}``."""
        return {"looking": self.looking, "talking": self.talking}


class AttentionWindow:
    """Head pose and jaw opening of the last seconds."""

    def __init__(self, cfg: AttentionConfig) -> None:
        """An empty window."""
        self.cfg = cfg
        self._samples: deque[tuple[float, Mesh | None]] = deque()

    def add(self, stamp: float, mesh: Mesh | None) -> None:
        """One frame's landmarker reading; ``None`` when it found no face in the crop (a face
        turned away, or not a face): a not-looking sample with no jaw."""
        self._samples.append((stamp, mesh))
        self._trim(stamp)

    def state(self, now: float) -> AttentionState | None:
        """Looking (majority over ``look_window_s``) and talking (jaw std over
        ``talk_window_s``) at ``now``; ``None`` when no reading is recent."""
        self._trim(now)
        cfg = self.cfg
        look = [
            m is not None and facing(m, cfg)
            for t, m in self._samples
            if t >= now - cfg.look_window_s
        ]
        jaw = [
            m.jaw_open for t, m in self._samples if m is not None and t >= now - cfg.talk_window_s
        ]
        if not look and not jaw:
            return None
        looking = (2 * sum(look) >= len(look)) if look else None
        if len(jaw) < cfg.min_talk_samples:
            return AttentionState(looking, None, None)
        std = float(np.std(jaw))
        return AttentionState(looking, std >= cfg.talk_jaw_std, std)

    def _trim(self, now: float) -> None:
        horizon = max(self.cfg.look_window_s, self.cfg.talk_window_s)
        while self._samples and self._samples[0][0] < now - horizon:
            self._samples.popleft()
