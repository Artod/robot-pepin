"""The breath layer: a slow sinusoid per servo that rides UNDER whatever a part plays while a
unit holds it up (config/animation.json's ``breath`` block,
:class:`pepin.expression.animation.Breath`).

It is a filter, not an atom: :class:`ArmBreath` adds its offsets to the knots of every arm
trajectory planned over a held pose (:meth:`pepin.expression.parts.ArmPart.plan`): a wrist beat,
a mark's stroke from the unit, the posture's renewal and the flat idle clip the listeners send
between them (:mod:`pepin.expression.listeners`) all breathe in every joint, each joint at its
own amplitude, period and phase, as a function of ABSOLUTE time, so one clip's end and the
next's start are the same breath. A trajectory the board holds freezes the breath where its
last knot left it; the next clip blends from that frozen offset to the live breath over
``blend_s`` (no jump: the board's approach finds the arm where the knot says). A clip that ends
at rest blends the breath out the same way, and a clip planned from rest gets none (never on the
base). The voice's accents feed it (:meth:`excite`): each raises the envelope by its ensemble's
gain, decaying over ``accent_decay_s``, so the breath swells with the voice. The offsets are a
few degrees at a few degrees a second: far under the caps; a knot they would push past a limit
plays without them (the part's rule: a gesture never loses to its breath).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence

from pepin.expression.animation import Breath

Knots = list[tuple[float, list[float]]]
EXCITATION_KEPT = 6.0  # an accent's swell is forgotten after this many decays


def smoothstep(share: float) -> float:
    """0 at 0, 1 at 1, flat at both ends (the blend's ease)."""
    u = min(1.0, max(0.0, share))
    return u * u * (3.0 - 2.0 * u)


class ArmBreath:
    """The arm's breath (the module's notes): ``breath`` is the book's block for the arm,
    ``joints`` the trajectory's joint order, ``clock`` the engine's."""

    def __init__(
        self,
        breath: Breath,
        joints: Sequence[str],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Nothing excites it yet."""
        self.breath = breath
        self.joints = tuple(joints)
        self.clock = clock
        self._accents: deque[tuple[float, float]] = deque()  # (when, gain)
        self._lock = threading.Lock()

    def excite(self, gain: float, at: float | None = None) -> None:
        """An accent of ``gain`` at ``at`` (engine clock; None: now) swells the breath."""
        if gain <= 0.0:
            return
        when = self.clock() if at is None else at
        with self._lock:
            self._accents.append((when, gain))
            horizon = when - EXCITATION_KEPT * self.breath.accent_decay_s
            while self._accents and self._accents[0][0] < horizon:
                self._accents.popleft()

    def envelope(self, t: float) -> float:
        """The amplitude's factor at ``t`` (engine clock): 1 plus every accent's swell so far."""
        decay = self.breath.accent_decay_s
        with self._lock:
            accents = list(self._accents)
        swell = 0.0
        for when, gain in accents:
            if t >= when and decay > 0.0:
                swell += gain * math.exp(-(t - when) / decay)
        return 1.0 + swell

    def offset(self, t: float) -> dict[str, float]:
        """Each joint's offset (degrees) at ``t`` (engine clock), the envelope in."""
        factor = self.envelope(t)
        out: dict[str, float] = {}
        for joint, osc in self.breath.joints.items():
            phase = 2.0 * math.pi * (t / osc.period_s + osc.phase)
            out[joint] = factor * osc.amp_deg * math.sin(phase)
        return out

    def breathe(
        self,
        knots: Knots,
        t0: float,
        frozen: Sequence[float] | None,
        *,
        live: bool = True,
    ) -> Knots:
        """``knots`` (seconds from the trajectory's start, an angle per joint) with the breath
        added: from the offsets ``frozen`` at the knots' start (None: none) to the live breath
        (``live``; else to none, for a clip that ends at rest) over the book's ``blend_s``, the
        knot at ``t`` seconds sounding at ``t0 + t`` on the engine's clock."""
        blend = self.breath.blend_s
        start = list(frozen) if frozen is not None else [0.0] * len(self.joints)
        out: Knots = []
        for t, deg in knots:
            share = smoothstep(t / blend) if blend > 0.0 else 1.0
            target = self.offset(t0 + t) if live else {}
            breathed = []
            for j, (joint, value) in enumerate(zip(self.joints, deg, strict=True)):
                aim = target.get(joint, 0.0)
                breathed.append(value + start[j] + (aim - start[j]) * share)
            out.append((t, breathed))
        return out

    def final_offset(self, knots: Knots, breathed: Knots) -> tuple[float, ...]:
        """What the breath added at the last knot (what a hold freezes)."""
        if not knots:
            return tuple(0.0 for _ in self.joints)
        return tuple(b - a for a, b in zip(knots[-1][1], breathed[-1][1], strict=True))
