"""The animation data model: motion atoms, the compositions an intent is made of, the idle layer,
and what a rendered performance is.

ATOMS are the vocabulary of primitives, each on one body part with a few parameters (a nod: an
amplitude, a period). An atom is ONE CYCLE of keyframes at shares of its period (``at`` 0..1);
a value is a number, a parameter (``"$amp_deg"``, ``"-$amp_deg"``) or, on the face, an
expression's name; a key may name a pose of the book (taught by hand) for every joint its values
leave out. ``mode`` says whether the values are positions (``absolute``: an arm pose) or offsets
on the layer below (``additive``: a nod on the head that tracks the person). ``period_s`` (one
cycle) is the atom's required parameter; ``stroke`` is the share of the first cycle where the
atom's meaningful moment lands (BML's stroke sync point). How often the cycle plays is its
user's, not the atom's: ``repeats`` (cycles, 1 when absent) and ``hold_s`` (the last key held
before the step ends, 0 when absent) are a STEP's reserved parameters, set by the step or by a
mark's arguments, and ``loop`` (repeat until withdrawn: the breath) is a step's flag
(2026-10-14; the atoms carried all three before).

A COMPOSITION is an intent the brain names (``laugh``): weighted variants, each a list of steps;
a step plays one atom, or another composition (NESTING, at most three deep), with probability
``p``, its parameters drawn from ranges, its stroke placed ``at_s`` from the composition's sync
point (the marker's word). Rendering draws the variant, the steps and every parameter from one
seeded generator, so the same intent moves a little differently every time and a logged seed
replays it exactly. A step whose part cannot play now (the head held by a better band, the
wheels turning toward the speaker) is left out and the rest still plays. A nested composition
draws its own variant and parameters from a seed derived from its parent's and the step's
index, its steps placed relative to the parent step's ``at_s``; the caps and the cooldowns count
the parent alone. Every atom that renders bare (no name to choose, no place of the gesture space
to find) and has no composition of its name gets one generated (an AUTO-WRAPPER: one step, the
atom's own ranges, ``wraps`` naming the atom), so the brain may name a single basic move.

The degrees of freedom, per part: the arm's joints (:data:`pepin.expression.trajectory.JOINTS`,
degrees, config/arm.json's convention), the head's ``pan_deg`` (+ left) and ``tilt_deg`` (+ down),
the face's ``expression`` and its drawing ``style`` (one of config/face.json's ``styles``, held
by the head server for the step's hold_s, standing without one; :data:`STYLE_DEFAULT` lets it go;
never the idle layer's), the wheels' ``yaw_deg`` (+ left) and ``x_m`` (+ forward), the eyes'
look (the head server's eye fields, :data:`pepin.head_link.EYE_RANGES`, in config/face.json's
units: ``brow_pos``, ``brow_raise``, ``lid_top``, ``glow``, ...), their moments
(:data:`MOMENT_DOFS`: a ``blink``, a ``wink`` of the robot's left eye for a value above 0 and of
its right below as config/face.json's ``wink`` has it, the look let go: ``release``) and their
named changes (:data:`EVENT_DOFS`: the ``iris`` colour, the inner rings' ``effect``).

Blending, per part: a gesture's absolute tracks replace the base layer while they play, its
additive tracks add to it; the idle layer adds only while no gesture plays on that part; a new
performance on a part takes it from the one playing (the newest wins), and every change between
layers fades over the ``fade_s`` knob. The person starting to speak fades the gesture layer out
on every part; the base layer (the head on the person) stays.

Owners of what this file declares (co-speech gesticulation, DESIGN.md §11): ``Composition.bind,
args, settings, family, long``, ``Variant.needs, floor``, ``Book.aliases`` and :class:`Space`
are authored by C (the book's primitives and the gesture space; the resolver reads them) and
read by B (the ladder); ``Idle.unless`` by A (the breath, the melody); ``Scene``'s
``lead_s``/``arm_at`` are filled by B, ``quiet``/``energy`` by A, ``distance_m`` from the
attention view, ``face``/``attention``/``unseen`` by the engine; ``Clip.then``/``hold_s`` by B
(arm units).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pepin.expression.markers import Cue
from pepin.expression.trajectory import Then

PartName = Literal["arm", "head", "face", "body", "eyes"]
PARTS: tuple[PartName, ...] = ("arm", "head", "face", "body", "eyes")
# The degrees of freedom whose key value is a name (a face's expression or drawing style, the
# eyes' iris colour "#RRGGBB", an effect of pepin.face.EYE_EFFECTS): each key is an event, never
# a track
EVENT_DOFS: tuple[str, ...] = ("expression", "iris", "effect", "style")
# The face's drawing style: its degree of freedom, and the composition (with the atom) whose
# mark draws the mouth in a style of config/face.json ({style:rose}); the style named
# STYLE_DEFAULT lets the face's style go, back to the standing one ({style} alone)
FACE_STYLE = "style"
STYLE_DEFAULT = "default"
# A step's named parameter that is its mark's argument as written (the style of {style:rose})
MARK_ARG = "$arg"
# The eyes' degrees of freedom whose keys are moments (a blink, a wink, the look let go): played
# at their keys' times, never held over a step's hold_s
MOMENT_DOFS: tuple[str, ...] = ("blink", "wink", "release")
Layer = Literal["base", "gesture", "idle"]
Mode = Literal["absolute", "additive"]
Value = float | str  # a number, "$param" or "-$param"; on the face's ``expression`` a name
# How tightly a composition's stroke is bound to its marker's word (DESIGN.md §4.4): on the
# word's stressed vowel, there or on one of the next two words of the phrase, or anywhere in
# the utterance (today's intents)
Bind = Literal["word", "phrase", "utterance"]
BINDS: tuple[Bind, ...] = ("word", "phrase", "utterance")
IdleState = Literal["speaking", "listening"]  # who speaks: the robot, the person
IDLE_STATES: tuple[IdleState, ...] = ("speaking", "listening")

PERIOD, REPEATS, HOLD = "period_s", "repeats", "hold_s"  # the reserved parameters
NUMBER = "number"  # a composition's argument that is a mark's number ({snap:3})
PERSON, SEEN = "person", "seen"  # a variant's need of the person engaged: seen (Scene.unseen)


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
    composition the brain may name (the idle layer's own; its auto-wrapper is not callable
    either)."""

    name: str
    part: PartName
    mode: Mode
    keys: tuple[Key, ...]
    params: Mapping[str, Range]
    stroke: float = 0.5
    note: str = ""
    mirror: tuple[str, ...] = ()
    callable: bool = True


@dataclass(frozen=True)
class Step:
    """One atom in a composition, or a nested composition (``composition``; ``atom`` is then
    ""): its stroke ``at_s`` from the sync point (negative: before; a nested composition's sync
    point), the chance it plays, its parameter ranges over the atom's own (``repeats`` and
    ``hold_s`` among them), its named parameters (``"$name"`` of a face key, ``"$pose"`` of a
    pose key), one drawn from each list, and whether its clip repeats on its part's owner
    until withdrawn (``loop``: the idle layer's breath; never in a callable composition)."""

    atom: str
    at_s: Range = Range(0.0, 0.0)
    p: float = 1.0
    params: Mapping[str, Range] = field(default_factory=dict)
    choices: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    composition: str | None = None
    loop: bool = False


@dataclass(frozen=True)
class Variant:
    """One way to perform an intent, how often it is drawn against the others, what it needs of
    the body (``needs``: a part -> a state, e.g. ``{"arm": "any"}``; the face's state other
    than ``any`` is the expression it shows, ``{"face": "grin"}``; ``attention``: its phase,
    ``{"attention": "track"}``; ``person``: ``seen``, the person engaged seen lately, never
    while :attr:`Scene.unseen` says why not: a wave at nobody; none: anything; :class:`Scene`)
    and of its marker (``"arg"``: the argument, or its kind: ``you``, ``me``, ``place``,
    ``thing``), and whether it is the floor (``floor``: the head and face version that always
    fits)."""

    steps: tuple[Step, ...]
    weight: float = 1.0
    needs: Mapping[str, str] = field(default_factory=dict)
    floor: bool = False


@dataclass(frozen=True)
class Setting:
    """A step parameter a mark's argument sets: ``param`` on the steps of ``atom`` (None: every
    step whose atom takes it, every step for ``repeats`` and ``hold_s``), to ``value`` (None:
    the mark's number)."""

    param: str
    atom: str | None = None
    value: Range | None = None


@dataclass(frozen=True)
class Composition:
    """An intent: its variants, the least time between two of its performances (a second one
    sooner is skipped), how its stroke binds to its word (:data:`Bind`), the arguments a marker
    may give it that choose a variant or a place (``{point:you}``; ``"*"`` any), the family a
    moment of the same kind is absorbed into, whether it is long (hello, bye: the utterance's
    gesture, not a beat, its stroke anywhere in the utterance when it cannot make its word: the
    ladder's in-the-utterance rung), whether a marker may name it (``callable`` False: the
    engine's own ensemble, which its listeners request by name: the pulse's accents, the nods,
    the held arm's sway; never in the brain's vocabulary, never under the markers' caps),
    ``style_args``: an argument -> the style words of the part that choose it for a bare marker
    (``{laugh}`` in a ``{sinister}`` part is ``{laugh:evil}``), ``settings``: the arguments that
    set its steps' parameters instead (a word, or :data:`NUMBER` for a mark's number:
    ``{snap:3,fast}``), ``wraps``: the atom an auto-wrapper plays (None: written by hand), and
    ``bare``: whether a mark may name it without the argument that chooses (its first plays:
    ``{style}`` is ``{style:default}``)."""

    name: str
    variants: tuple[Variant, ...]
    cooldown_s: float = 0.0
    note: str = ""
    bind: Bind = "utterance"
    args: tuple[str, ...] = ()
    family: str | None = None
    long: bool = False
    callable: bool = True
    style_args: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    settings: Mapping[str, tuple[Setting, ...]] = field(default_factory=dict)
    wraps: str | None = None
    bare: bool = False


@dataclass(frozen=True)
class Idle:
    """The idle layer of one part: compositions of that part alone (an atom's auto-wrapper, or
    one written for the idle layer: the breath's, looping) drawn every ``every_s`` while nothing
    else plays on it,
    during a conversation only or always, unless one of ``unless`` holds (the robot speaking,
    the person speaking; by default either: idle only while nobody speaks). A clip of it plays
    on while a state it allows begins (a breath while the person speaks) and fades when one it
    excludes does. ``after_speech_s``: how long after such a state ended it waits (None: the
    engine's ``layers.idle_after_speech_s``; 0: a breath right after a reply)."""

    part: PartName
    compositions: tuple[str, ...]
    every_s: Range
    when: Literal["conversation", "always"] = "conversation"
    unless: tuple[IdleState, ...] = IDLE_STATES
    after_speech_s: float | None = None


@dataclass(frozen=True)
class Space:
    """The gesture space in front of the robot (config/animation.json's ``space``, DESIGN.md
    §4.3): the home pose, the shoulder pan inward of which a way to a target goes through home,
    and the poses always reached through it (``via``); the cells the referents and the counted
    items take, in order (``loci``); the cells each path-drawing mark goes through (``paths``,
    ``"arc:up"`` -> cells); the head's (pan, tilt) offset for a mark whose head turns its
    argument's way (``head``, ``"arc:up"`` -> degrees); the pointing hand (the open and the shut
    hand's poses straight ahead, the clear pan range, the arm's pan axis and the neck's in the
    cart frame, a person's distance when unknown); the marks the brain may repeat in a reply;
    the poses the arm leaves for another after its hold (``home_after``, a pose -> the pose
    the clip ends on: the board's own straight way to rest from the first would pass the
    body; a pose inward of ``via_pan_deg`` goes home); the poses whose ways pass the head's pan
    sweep (``head_pan_min_deg``, a pose -> the least head pan, degrees + left, they keep clear
    at: with the head turned further right they do not resolve)."""

    home: str = "ready"
    via_pan_deg: float | None = None
    via: tuple[str, ...] = ()
    home_after: Mapping[str, str] = field(default_factory=dict)
    head_pan_min_deg: Mapping[str, float] = field(default_factory=dict)
    loci: tuple[str, ...] = ()
    paths: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    head: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    open_hand: str | None = None
    shut_hand: str | None = None
    pan_deg: tuple[float, float] = (0.0, 0.0)
    arm_axis_m: tuple[float, float] = (0.0, 0.0)
    neck_m: tuple[float, float] = (0.0, 0.0)
    person_m: float = 1.2
    again: tuple[str, ...] = ()


@dataclass(frozen=True)
class Oscillation:
    """One servo's share of a breath: its amplitude, its period and its phase (a share of a
    turn: 0.5 is the opposite phase)."""

    amp_deg: float
    period_s: float
    phase: float = 0.0


@dataclass(frozen=True)
class Breath:
    """A part's breath layer (config/animation.json's ``breath``, :mod:`pepin.expression.breath`):
    a slow sinusoid per servo added under whatever the part plays while a unit holds it up
    (``when`` ``"up"``: never at rest), blended in and out over ``blend_s`` from where a motion
    left it; the voice's accents swell it (``accent_gain`` per ensemble, decaying over
    ``accent_decay_s``); the idle clip that carries it alone lasts ``clip_s``."""

    joints: Mapping[str, Oscillation]
    when: Literal["up"] = "up"
    blend_s: float = 0.6
    accent_gain: Mapping[str, float] = field(default_factory=dict)
    accent_decay_s: float = 1.2
    clip_s: float = 3.0


@dataclass(frozen=True)
class Book:
    """config/animation.json: the atoms, the compositions (the intents), the arm's poses (pose
    -> joint -> degrees), the idle layers, the robot's moments mapped to intents, the marks'
    other spellings (``aliases``: ``"beat"`` -> ``"accent"``, ``"point:i"`` -> ``"point:me"``),
    the gesture space and the breath layers by part (``breath``; a part absent breathes by its
    older way: the arm's ``arm_sway`` composition, the head's looping idle atom)."""

    atoms: Mapping[str, Atom]
    compositions: Mapping[str, Composition]
    poses: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    idle: tuple[Idle, ...] = ()
    moments: Mapping[str, str] = field(default_factory=dict)
    aliases: Mapping[str, str] = field(default_factory=dict)
    space: Space = field(default_factory=Space)
    breath: Mapping[PartName, Breath] = field(default_factory=dict)


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
    """A discrete change of a rendered clip (the face's expression, the eyes' iris or effect:
    ``dof``, :data:`EVENT_DOFS`): when, what, how long."""

    t_s: float
    name: str
    hold_s: float | None = None
    dof: str = "expression"


@dataclass(frozen=True)
class Clip:
    """What one part plays of a performance: its tracks and events, when it starts relative to
    the performance's sync point (negative: before the word), where its stroke and its end are
    (seconds from its start), the layer it plays on, and whether its owner repeats it until
    withdrawn (``loop``: a looping step's, :attr:`Step.loop`)."""

    part: PartName
    offset_s: float
    stroke_s: float
    duration_s: float
    tracks: tuple[Track, ...] = ()
    events: tuple[Event, ...] = ()
    layer: Layer = "gesture"
    loop: bool = False
    then: Then = "rest"  # the arm: rest after it, or hold its last pose ``hold_s`` (a unit)
    hold_s: float = 0.0

    @property
    def styles_only(self) -> bool:
        """Whether it only draws the face in a style: the head server arbitrates that beside
        the expression, so such a clip never books nor holds the face."""
        return (
            bool(self.events) and not self.tracks and all(e.dof == FACE_STYLE for e in self.events)
        )


@dataclass(frozen=True)
class Performance:
    """An intent rendered: the seed it was drawn with, the variant drawn, one clip per part
    that plays, the cue it is placed on (None: now), and the nested compositions with the
    variant each drew (depth first, as their steps stand)."""

    intent: str
    seed: int
    variant: int
    clips: tuple[Clip, ...]
    cue: Cue | None = None
    nested: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class Scene:
    """What a rendering may use: the parts ready to play now, whether the robot is speaking and
    whether a conversation is open; the seconds until the anchor word is heard (``lead_s``,
    None: unknown or now), where the arm is (``arm_at``: ``"rest"``, a pose's name, None:
    unknown), the person's rough distance (None: unknown), whether the moment is quiet (head
    and face only), the reply's energy (amplitude x, 1 as authored), the expression the face
    shows now (``face``: the engine's own face clip still holding, by name; None: none of the
    engine's) and attention's phase (``attention``: :data:`pepin.expression.speaker.
    AttentionPhase`, ``"track"`` while it holds a person; None: no attention view), and why the
    person engaged counts as unseen (``unseen``: no face or body of theirs lately, with eyes to
    tell; None: seen, or nobody can tell). A variant may need the last three by name
    (``{"face": "grin"}``, ``{"attention": "track"}``, ``{"person": "seen"}``)."""

    ready: frozenset[PartName]
    speaking: bool = False
    conversation: bool = False
    lead_s: float | None = None
    arm_at: str | None = None
    distance_m: float | None = None
    quiet: bool = False
    energy: float = 1.0
    face: str | None = None
    attention: str | None = None
    unseen: str | None = None


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
