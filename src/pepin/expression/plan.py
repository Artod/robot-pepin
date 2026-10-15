"""How to face a bearing: the head alone within the forward arc, the body turned toward a person
past it; and where a seek looks when nobody is seen at the bearing (:func:`sweep`).

A person within the forward arc (``arc_deg``, 160: +-80 deg of the cart's nose) is faced by the
head alone. Past it the wheels turn the body toward them in place, at most ``max_turn_deg`` (30)
per request, while the head already looks there (it is the fast one): the person comes back into
the arc, and well inside it, so a person at its edge does not set the body hunting. When the
wheels may not move (a drive, the stub, a refused check) or turn less than asked, the head looks
as far as it reaches. The head never needs the wheels to SEE a person (its reach is about +-156
deg); the wheels only bring a person far round back into the arc.

:class:`BoundedTurn` is the wheels' one recovery made concrete over :class:`TurnSteps` (the
board's measured motion, :class:`pepin.expression.wheels.WheelsExecutor`), one rule for every
way a turn can end short. A turn the wheels could not finish because they stalled or the arc is
blocked steps ``backoff_m`` straight away from the nearest obstacle mark (forward unless that
mark lies ahead of the footprint's middle); when that step cannot go (blocked, or stalled too)
it turns ``backoff_deg`` back the other way instead. Then it tries the rest of the turn once
more, and gives up if that ends short too. The cart never ends more than ``max_turn_deg`` from
the heading the request began at, and never steps more than ``backoff_m``. A turn STOPPED (a
halt, someone else's command, a drive, a precondition) is never retried and never steps away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from pepin.expression.executors import WheelMove, WheelReport
from pepin.expression.speaker import GazePlan
from pepin.gaze import Aim, Reach


@dataclass(frozen=True)
class BodyLimits:
    """config/expression.json's ``wheels`` block: the turn and the rock per request, the turn's
    progress step, the one recovery's step away and turn back."""

    max_turn_deg: float = 180.0
    step_deg: float = 10.0
    max_rock_m: float = 0.02
    backoff_m: float = 0.02
    backoff_deg: float = 10.0


def split(
    bearing_rad: float, tilt_rad: float, *, half_arc_rad: float, max_turn_rad: float
) -> GazePlan:
    """The wheels' share (none within ``half_arc_rad`` of the nose; past it the turn that faces
    the person, at most ``max_turn_rad``) and the head's aim relative to the cart once they have
    turned it."""
    past = abs(bearing_rad) > half_arc_rad
    body = math.copysign(min(abs(bearing_rad), max_turn_rad), bearing_rad) if past else 0.0
    return GazePlan(body, Aim(bearing_rad - body, tilt_rad))


def sweep(pan_rad: float, tilt_rad: float, *, span_rad: float) -> list[Aim]:
    """A seek's stops around a bearing at one tilt: ``span_rad`` outward (away from the cart's
    nose) first, then as far inward. Outward first because the array read a voice nearer the nose
    than the face in all three misses logged on 2026-10-12 (-15 for -43.5, -11 for -18, -4 for
    -15 deg)."""
    if span_rad <= 0.0:
        return []
    side = 1.0 if pan_rad >= 0.0 else -1.0
    return [Aim(pan_rad + side * span_rad, tilt_rad), Aim(pan_rad - side * span_rad, tilt_rad)]


def head_after(bearing_rad: float, turned_rad: float, tilt_rad: float, reach: Reach) -> Aim:
    """The head's aim at a bearing once the wheels turned ``turned_rad`` of it: as far as the
    neck reaches."""
    return reach.clamp(Aim(bearing_rad - turned_rad, tilt_rad))


@dataclass(frozen=True)
class Step:
    """One motion as the wheels made it: how far (degrees of a turn, + left; metres of a step
    away) and why it stopped short: ``blocked`` (a stall, a blocked arc: worth the recovery and
    one more try) or ``stopped`` (a halt, someone else's command, a drive, a precondition:
    final); ``cut`` when a halt cut it (a drive began, a newer motion), so it is worth finishing
    later."""

    amount: float
    blocked: str = ""
    stopped: str = ""
    cut: bool = False


class TurnSteps(Protocol):
    """The wheels one motion at a time (the board's measured motion), blocking its caller (the
    wheels' own thread)."""

    def turn(self, deg: float) -> Step:
        """Turn in place by ``deg`` (+ left) in one go, or as far as the wheels went."""
        ...

    def step_away(self, metres: float) -> Step:
        """Straight by ``metres`` away from the nearest obstacle mark; the distance it went."""
        ...


class BoundedTurn:
    """A turn with its one recovery over :class:`TurnSteps` (runs on the wheels' thread)."""

    def __init__(self, steps: TurnSteps, limits: BodyLimits | None = None) -> None:
        """The steps it drives and the knobs that bound it."""
        self._steps = steps
        self.limits = limits if limits is not None else BodyLimits()

    def move(self, request: WheelMove) -> WheelReport:
        """Turn by ``request.turn_deg``: one go; ended short (stalled, blocked), one recovery
        (the step away, or the turn back when the step cannot go) and one more try of the rest;
        refused past ``max_turn_deg`` or for a rock."""
        lim = self.limits
        if request.rocks or request.rock_m:
            return WheelReport(False, why="a rock is not a turn")
        if abs(request.turn_deg) > lim.max_turn_deg + 1e-9:
            past = f"a turn of {request.turn_deg:+.0f} deg is past {lim.max_turn_deg:.0f}"
            return WheelReport(False, why=f"refused: {past}")
        first = self._steps.turn(request.turn_deg)
        turned = first.amount
        if first.stopped:
            return WheelReport(False, turned, 0.0, 1, first.stopped, first.cut)
        if not first.blocked:
            return WheelReport(True, turned, 0.0, 1)
        away = self._steps.step_away(lim.backoff_m)
        backed = abs(away.amount)
        if away.stopped:
            return WheelReport(False, turned, backed, 1, away.stopped, away.cut)
        if away.blocked:
            back = self._steps.turn(-math.copysign(lim.backoff_deg, request.turn_deg))
            turned += back.amount
            if back.stopped:
                return WheelReport(False, turned, backed, 1, back.stopped, back.cut)
            if back.blocked:
                why = f"{first.blocked}; {away.blocked}; the turn back: {back.blocked}"
                return WheelReport(False, turned, backed, 1, f"gave up: {why}")
        again = self._steps.turn(request.turn_deg - turned)
        turned += again.amount
        if again.stopped:
            return WheelReport(False, turned, backed, 2, again.stopped, again.cut)
        if again.blocked:
            return WheelReport(False, turned, backed, 2, f"gave up: {again.blocked}")
        return WheelReport(True, turned, backed, 2)
