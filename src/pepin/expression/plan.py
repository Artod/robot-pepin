"""How to face a bearing: the head alone within its comfortable pan, the wheels for the rest.

A person within ``comfort_pan_deg`` (60) of the cart's nose is faced by the head alone. Past it
the wheels turn the rest in place, at most ``max_turn_deg`` (30) per request, and the head looks
the remaining way relative to the turned cart; when the wheels may not move (a drive, the stub
that refuses everything tonight) or turn less than asked, the head looks as far as it reaches.
The head never needs the wheels to SEE a person (its reach is about +-156 deg); the wheels only
bring a person far round back toward the head's comfortable range.

:class:`BoundedTurn` is the wheels' contract with that retry made concrete over one step at a
time (:class:`TurnSteps`, the board's measured-motion path behind it when it is built): a turn in
steps of ``step_deg``; a stalled or blocked step backs the cart off ``backoff_m`` and tries the
turn again, at most ``attempts`` times and ``max_backoff_m`` of backing in all; then it gives up.
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
    """config/expression.json's ``parts.body``: the turn per request, its steps, the retries."""

    max_turn_deg: float = 30.0
    step_deg: float = 10.0
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


def head_after(bearing_rad: float, turned_rad: float, tilt_rad: float, reach: Reach) -> Aim:
    """The head's aim at a bearing once the wheels turned ``turned_rad`` of it: as far as the
    neck reaches."""
    return reach.clamp(Aim(bearing_rad - turned_rad, tilt_rad))


@dataclass(frozen=True)
class Step:
    """One step of a turn as the wheels made it: the heading turned (degrees, + left), and why
    it stopped short (a stall, a blocked arc) or "" when it did all of it."""

    turned_deg: float
    blocked: str = ""


class TurnSteps(Protocol):
    """The wheels one motion at a time (the board's measured-motion path)."""

    def turn(self, deg: float) -> Step:
        """Turn in place by ``deg`` (+ left), or as far as the wheels went."""
        ...

    def back_off(self, metres: float) -> float:
        """Back straight off by ``metres``; the distance it went."""
        ...


class BoundedTurn:
    """:class:`pepin.expression.executors.Wheels` for turns over :class:`TurnSteps`, with the
    bounded retry; a rock is not built (refused)."""

    def __init__(self, steps: TurnSteps, limits: BodyLimits | None = None) -> None:
        """The steps it drives and the knobs that bound it."""
        self._steps = steps
        self.limits = limits if limits is not None else BodyLimits()

    def move(self, request: WheelMove) -> WheelReport:
        """Turn by ``request.turn_deg`` in steps; refused past ``max_turn_deg`` or for a rock."""
        lim = self.limits
        if request.rocks or request.rock_m:
            return WheelReport(False, why="a rock is not built")
        if abs(request.turn_deg) > lim.max_turn_deg + 1e-9:
            return WheelReport(
                False, why=f"a turn of {request.turn_deg:+.0f} deg is past {lim.max_turn_deg:.0f}"
            )
        left, turned, backed, attempts, why = request.turn_deg, 0.0, 0.0, 1, ""
        while abs(left) > 1e-6:
            step = math.copysign(min(abs(left), lim.step_deg), left)
            done = self._steps.turn(step)
            turned += done.turned_deg
            left -= done.turned_deg
            if not done.blocked:
                continue
            why = done.blocked
            if attempts >= lim.attempts or backed + lim.backoff_m > lim.max_backoff_m + 1e-9:
                break
            backed += self._steps.back_off(lim.backoff_m)
            attempts += 1
        complete = abs(left) <= 1e-6
        return WheelReport(
            complete, turned, backed, attempts, "" if complete else f"gave up: {why}"
        )
