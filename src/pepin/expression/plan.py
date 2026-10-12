"""How to face a bearing: the head alone within its comfortable pan, the wheels for the rest;
and where a seek looks when nobody is seen at the bearing (:func:`sweep`).

A person within ``comfort_pan_deg`` (60) of the cart's nose is faced by the head alone. Past it
the wheels turn the rest in place, at most ``max_turn_deg`` (30) per request, and the head looks
the remaining way relative to the turned cart; when the wheels may not move (a drive, the stub)
or turn less than asked, the head looks as far as it reaches. The head never needs the wheels to
SEE a person (its reach is about +-156 deg); the wheels only bring a person far round back toward
the head's comfortable range.

:class:`BoundedTurn` is the wheels' retry made concrete over :class:`TurnSteps` (the board's
measured motion, :class:`pepin.expression.wheels.WheelsExecutor`): one continuous turn per
attempt; a turn the wheels could not finish because they stalled or the arc is blocked backs the
cart off ``backoff_m`` and tries the rest again, at most ``attempts`` times and ``max_backoff_m``
of backing in all; then it gives up. A turn STOPPED (a halt, someone else's command, a drive, a
precondition) is never retried and never backs off.
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
    progress step, the retries."""

    max_turn_deg: float = 30.0
    step_deg: float = 10.0
    max_rock_m: float = 0.02
    attempts: int = 3
    backoff_m: float = 0.03
    max_backoff_m: float = 0.06


def split(
    bearing_rad: float, tilt_rad: float, *, comfort_rad: float, max_turn_rad: float
) -> GazePlan:
    """The wheels' share (the part past ``comfort_rad``, at most ``max_turn_rad``) and the head's
    aim relative to the cart once they have turned it."""
    excess = abs(bearing_rad) - comfort_rad
    body = math.copysign(min(excess, max_turn_rad), bearing_rad) if excess > 0.0 else 0.0
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
    """One motion as the wheels made it: how far (degrees of a turn, + left; metres of a
    back-off) and why it stopped short: ``blocked`` (a stall, a blocked arc: worth a back-off and
    another try) or ``stopped`` (a halt, someone else's command, a drive, a precondition: final);
    ``cut`` when a halt cut it (the person spoke, a drive began), so it is worth finishing later."""

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

    def back_off(self, metres: float) -> Step:
        """Back straight off by ``metres``; the distance it went."""
        ...


class BoundedTurn:
    """A turn with the bounded retry over :class:`TurnSteps` (runs on the wheels' thread)."""

    def __init__(self, steps: TurnSteps, limits: BodyLimits | None = None) -> None:
        """The steps it drives and the knobs that bound it."""
        self._steps = steps
        self.limits = limits if limits is not None else BodyLimits()

    def move(self, request: WheelMove) -> WheelReport:
        """Turn by ``request.turn_deg``; refused past ``max_turn_deg`` or for a rock."""
        lim = self.limits
        if request.rocks or request.rock_m:
            return WheelReport(False, why="a rock is not a turn")
        if abs(request.turn_deg) > lim.max_turn_deg + 1e-9:
            past = f"a turn of {request.turn_deg:+.0f} deg is past {lim.max_turn_deg:.0f}"
            return WheelReport(False, why=f"refused: {past}")
        left, turned, backed, attempts = request.turn_deg, 0.0, 0.0, 1
        while True:
            step = self._steps.turn(left)
            turned += step.amount
            left -= step.amount
            if step.stopped:
                return WheelReport(False, turned, backed, attempts, step.stopped, step.cut)
            if not step.blocked:
                return WheelReport(True, turned, backed, attempts)
            why = step.blocked
            if attempts >= lim.attempts or backed + lim.backoff_m > lim.max_backoff_m + 1e-9:
                break
            back = self._steps.back_off(lim.backoff_m)
            backed += abs(back.amount)
            if back.stopped:
                return WheelReport(False, turned, backed, attempts, back.stopped, back.cut)
            if back.blocked:
                why = back.blocked
                break
            attempts += 1
        return WheelReport(False, turned, backed, attempts, f"gave up: {why}")
