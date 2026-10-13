"""The animation data model: motion atoms, the compositions an intent is made of, the idle layer,
and what a rendered performance is.

ATOMS are the vocabulary of primitives, each on one body part with a few parameters (a nod: an
amplitude, a period, repeats). An atom is ONE CYCLE of keyframes at shares of its period
(``at`` 0..1); a value is a number, a parameter (``"$amp_deg"``, ``"-$amp_deg"``) or, on the
face, an expression's name; a key may name a pose of the book (taught by hand) for every joint
its values leave out. ``mode`` says whether the values are positions (``absolute``: an arm pose)
or offsets on the layer below (``additive``: a nod on the head that tracks the person). Reserved
parameters: ``period_s`` (one cycle, required), ``repeats`` (cycles, 1 when absent) and
``hold_s`` (the last key held before the atom ends, 0 when absent). ``stroke`` is the share of
the first cycle where the atom's meaningful moment lands (BML's stroke sync point).

A COMPOSITION is an intent the brain names (``laugh``): weighted variants, each a list of steps;
a step plays one atom with probability ``p``, its parameters drawn from ranges, its stroke placed
``at_s`` from the composition's sync point (the marker's word). Rendering draws the variant, the
steps and every parameter from one seeded generator, so the same intent moves a little
differently every time and a logged seed replays it exactly. A step whose part cannot play now
(the head held by a better band, the wheels turning toward the speaker) is left out and the rest
still plays.

The degrees of freedom, per part: the arm's joints (:data:`pepin.expression.trajectory.JOINTS`,
degrees, config/arm.json's convention), the head's ``pan_deg`` (+ left) and ``tilt_deg`` (+ down),
the face's ``expression``, the wheels' ``yaw_deg`` (+ left) and ``x_m`` (+ forward).

Blending, per part: a gesture's absolute tracks replace the base layer while they play, its
additive tracks add to it; the idle layer adds only while no gesture plays on that part; a new
performance on a part takes it from the one playing (the newest wins), and every change between
layers fades over the ``fade_s`` knob. The person starting to speak fades the gesture layer out
on every part; the base layer (the head on the person) stays.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pepin.expression.markers import Cue

PartName = Literal["arm", "head", "face", "body"]
PARTS: tuple[PartName, ...] = ("arm", "head", "face", "body")
Layer = Literal["base", "gesture", "idle"]
Mode = Literal["absolute", "additive"]
Value = float | str  # a number, "$param" or "-$param"; on the face's ``expression`` a name

PERIOD, REPEATS, HOLD = "period_s", "repeats", "hold_s"  # the reserved parameters


@dataclass(frozen=True)
class Range:
    """A parameter's range, drawn uniformly at render time; a fixed value has ``lo == hi``."""

    lo: float
    hi: float


@dataclass(frozen=True)
class Key:
    """One keyframe of an atom's cycle: its share of the period, the values it sets and the pose
    of the book that supplies every joint the values leave out (None: none)."""

    at: float
    values: Mapping[str, Value] = field(default_factory=dict)
    pose: str | None = None


@dataclass(frozen=True)
class Atom:
    """A parametrised motion primitive of one part (config/animation.json's ``atoms``);
    ``mirror`` names the degrees of freedom whose values change sign when a rendering draws
    the mirror image (a glance left or right), ``callable`` False keeps it out of every
    composition the brain may name (the idle layer's own), ``loop`` marks the idle layer's
    clip of it as repeating on its part's owner until withdrawn (the breath)."""

    name: str
    part: PartName
    mode: Mode
    keys: tuple[Key, ...]
    params: Mapping[str, Range]
    stroke: float = 0.5
    note: str = ""
    mirror: tuple[str, ...] = ()
    callable: bool = True
    loop: bool = False


@dataclass(frozen=True)
class Step:
    """One atom in a composition: its stroke ``at_s`` from the sync point (negative: before),
    the chance it plays, its parameter ranges over the atom's own, and its named parameters
    (``"$name"`` of a face key, ``"$pose"`` of a pose key), one drawn from each list."""

    atom: str
    at_s: Range = Range(0.0, 0.0)
    p: float = 1.0
    params: Mapping[str, Range] = field(default_factory=dict)
    choices: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Variant:
    """One way to perform an intent, and how often it is drawn against the others."""

    steps: tuple[Step, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class Composition:
    """An intent: its variants, and the least time between two of its performances (a second
    one sooner is skipped)."""

    name: str
    variants: tuple[Variant, ...]
    cooldown_s: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class Idle:
    """The idle layer of one part: atoms drawn every ``every_s`` while nothing else plays on it,
    during a conversation only or always."""

    part: PartName
    atoms: tuple[str, ...]
    every_s: Range
    when: Literal["conversation", "always"] = "conversation"


@dataclass(frozen=True)
class Book:
    """config/animation.json: the atoms, the compositions (the intents), the arm's poses (pose
    -> joint -> degrees), the idle layers and the robot's moments mapped to intents."""

    atoms: Mapping[str, Atom]
    compositions: Mapping[str, Composition]
    poses: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    idle: tuple[Idle, ...] = ()
    moments: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Track:
    """One degree of freedom of a rendered clip: values at times (seconds from the clip's
    start), positions or offsets by ``mode``; between keys a part interpolates smoothly."""

    dof: str
    mode: Mode
    times: tuple[float, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class Event:
    """A discrete change of a rendered clip (the face's expression): when, what, how long."""

    t_s: float
    name: str
    hold_s: float | None = None


@dataclass(frozen=True)
class Clip:
    """What one part plays of a performance: its tracks and events, when it starts relative to
    the performance's sync point (negative: before the word), where its stroke and its end are
    (seconds from its start), the layer it plays on, and whether its owner repeats it until
    withdrawn (``loop``: an idle atom's, :attr:`Atom.loop`)."""

    part: PartName
    offset_s: float
    stroke_s: float
    duration_s: float
    tracks: tuple[Track, ...] = ()
    events: tuple[Event, ...] = ()
    layer: Layer = "gesture"
    loop: bool = False


@dataclass(frozen=True)
class Performance:
    """An intent rendered: the seed it was drawn with, the variant drawn, one clip per part
    that plays, and the cue it is placed on (None: now)."""

    intent: str
    seed: int
    variant: int
    clips: tuple[Clip, ...]
    cue: Cue | None = None


@dataclass(frozen=True)
class Scene:
    """What a rendering may use: the parts ready to play now, and whether the robot is
    speaking and whether a conversation is open."""

    ready: frozenset[PartName]
    speaking: bool = False
    conversation: bool = False


class Chooser(Protocol):
    """Picks the variant of a composition (weighted at random; a learned chooser later takes
    the same seat)."""

    def choose(self, composition: Composition, scene: Scene, rng: random.Random) -> int:
        """The index of the variant to perform."""
        ...


class Renderer(Protocol):
    """Turns an intent into a performance (atoms sampled, parameters drawn, clips per part)."""

    def render(self, intent: str, scene: Scene, seed: int, cue: Cue | None = None) -> Performance:
        """The performance of ``intent``; ``KeyError`` for an intent the book does not hold."""
        ...
