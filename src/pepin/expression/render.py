"""An intent rendered into clips: the variant chosen, each step drawn (whether it plays, when
its stroke lands, its parameters, its names, its mirror image), its references into the gesture
space resolved, its atom's cycle unrolled into tracks, the steps of one part merged into that
part's clip.

Every draw comes from one ``random.Random(seed)`` in a fixed order, whatever ends up playing, so
a logged seed (and variant) replays a performance exactly: ``scripts/gesture.py play laugh
--seed 1234``. The chooser never draws the variant it drew last for that intent (a composition
of one variant repeats it, its parameters still drawn afresh). A step whose part is not ready,
or whose pose has not been taught or cannot be resolved, is left out; when every optional step
is left out by chance, the most likely one that can play still plays.

The chooser draws among the variants a scene allows by their ``needs`` and never a floor; the
lead and the ladder are the scheduler's (:mod:`pepin.expression.schedule`), which renders a
given variant (``variant=``) for each rung. A variant that needs its marker's argument
(``needs: {"arg": ...}``) renders only for that argument; another renders to no clips.

THE RESOLVER (:class:`Resolver`; DESIGN.md §4.3-§4.6, the gesture space of config/animation.json's
``space``). A pose named ``@...`` in an atom's key or a step's ``pose`` is a place in the space:

* ``@from``: where the step starts: the previous arm step of the performance's last pose, else
  rest (a unit's held pose replaces rest when the arm part plans the clip from it);
* ``@via``: home (``space.home``), when the step's target (the key after it) is a place of the
  space inward of ``space.via_pan_deg`` or a pose of ``space.via``: straight from rest the hand
  passes the speaker box at 1.1 cm on its way to the middle, and the ways between ``stop``,
  ``tucked`` and the sweep's top pass the head (scratch/gesticulation_space/cells.py); its time
  divides the way so both legs run at one share of the caps; otherwise no key; and a step that
  ends there, inward of ``space.via_pan_deg``, goes back home after its hold (the board's own
  straight way to rest from the middle would pass the speaker box at 0.2-1.1 cm);
* ``@pose:NAME``: the book's pose NAME as a place of the space (the via applies);
* ``@toward``: the open hand (``space.open_hand``) turned at the person engaged (the attention
  view's bearing and distance), ``@bearing``: the shut hand at the marker's place (the room's);
  both only inside the clear pan range (``space.pan_deg``), else unresolved;
* ``@locus``: the cell of the marker's referent (a catchment), the gripper shut;
* ``@count``: the n-th of ``space.loci`` for ``{count:n}``;
* ``@path:k``: the k-th cell of ``space.paths["name:arg"]`` (its last beyond its end).

On a head step a reference is a direction: its ``pan_deg`` and ``tilt_deg`` become the
reference's, clamped into the step's ranges: ``@bearing`` the place seen from the head against
the person it looks at (at most the step's range, 25 deg: an OVERLAY on attention's aim, never a
change of where attention looks; a place more than 90 deg from the person gets none), ``@arg``
``space.head["name:arg"]``.

CATCHMENTS v1 (DESIGN.md §4.6, critic_architecture #9). A ``{point:NAME}`` whose NAME is no place
is a referent: it takes the first free cell of ``space.loci`` when the brain's marker is first
rendered, and every later mention points there again; with every cell taken, the least recently
pointed one is evicted for it; ``{count:n}`` takes the n-th cell, overwriting what sat there.
Forgotten at the conversation's close (:meth:`Resolver.forget`). Logged as ``catchment`` rows
(referent, cell, new / again / evicted / overwritten). A mark the brain repeats in one reply with
the same argument plays once (``repeat`` rows), but those of ``space.again`` (pointing back at a
referent, the accent).

Owners after slice 0: B (the chooser, the scenes, the ladder's renderings), C (the resolver).
"""

from __future__ import annotations

import itertools
import logging
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pepin.expression.animation import (
    HOLD,
    PERIOD,
    REPEATS,
    Atom,
    Book,
    Clip,
    Composition,
    Event,
    Key,
    Layer,
    PartName,
    Performance,
    Scene,
    Step,
    Track,
    Variant,
)
from pepin.expression.markers import Cue
from pepin.expression.places import Room
from pepin.expression.speaker import AttentionView

logger = logging.getLogger(__name__)

Record = Callable[[str, dict[str, Any]], None]
Pose = Mapping[str, float]
REF = "@"  # a pose that names a place of the gesture space
FROM, VIA, THERE, PATH = "@from", "@via", "@pose:", "@path:"
ARG_NEED = "arg"  # a variant's need of its marker's argument
# The kinds of a pointing argument: the person, the robot, a place of the room, a referent, and
# a name no room was there to tell (its floor only: never a referent's cell for a place)
YOU, ME, PLACE, THING, NAME = "you", "me", "place", "thing", "name"
SHUT = {"gripper": 0.0}  # a referent's spot is pointed at with the gripper shut (the finger)
AWAY_DEG = 90.0  # a place further than this from the person gets no head glance
GLANCE_DEG = 1.0  # ... and one nearer than this none either: the head is looking there
BACK_SHARE = 0.8  # the way home after a hold in the middle runs at this share of the caps
# The caps the book's timings are sized for (config/arm.json): their ratio places a via
CAPS_DEG_S, CAPS_DEG_S2 = 240.0, 2232.0


@dataclass(frozen=True)
class Meaning:
    """What a marker's argument means here: the argument (the composition's first when the
    marker gave none: a tool's ask), its kind for a pointing one (``you``, ``me``, ``place``,
    ``thing``; else the argument itself), the places book's name of a place, and the reply it
    was written in (None: not the brain's)."""

    arg: str | None
    kind: str | None
    place: str | None = None
    turn: int | None = None


@dataclass
class Locus:
    """A referent's cell for the conversation, and when it was last pointed at (an order)."""

    referent: str
    cell: str
    used: int


@dataclass
class Context:
    """One rendering's: the book, the composition, the marker's meaning, whether the rendering
    is the brain's first of its marker (it places catchments), and what it resolved."""

    book: Book
    composition: Composition
    meaning: Meaning
    commit: bool
    resolved: dict[str, Pose | None] = field(default_factory=dict)


class Resolver:
    """The gesture space's references, the catchments and the repeats (the module's notes)."""

    def __init__(
        self,
        *,
        attention: AttentionView | None = None,
        room: Room | None = None,
        record: Record | None = None,
    ) -> None:
        """``attention``: the person's bearing and distance (None: ahead, ``space.person_m``
        away); ``room``: the places (None: no place is known); ``record(kind, fields)``: the
        ``catchment`` and ``repeat`` rows (the log by default)."""
        self.attention = attention
        self.room = room
        self.record = record or (lambda kind, fields: logger.info("%s %s", kind, fields))
        self.loci: dict[str, Locus] = {}  # cell -> its referent
        self._order = itertools.count(1)
        self._said: dict[int, set[tuple[str, str | None]]] = {}

    def forget(self) -> None:
        """The conversation closed: every catchment and the replies' marks forgotten."""
        self.loci.clear()
        self._said.clear()

    # -- the marker ------------------------------------------------------------------------------

    def meaning(self, composition: Composition, cue: Cue | None) -> Meaning:
        """The meaning of ``cue``'s argument for ``composition``."""
        arg = cue.marker.arg if cue is not None and cue.marker.name == composition.name else None
        if arg is None:
            arg = next((a for a in composition.args if a != "*"), None)
        turn = cue.turn if cue is not None else None
        if arg is None or "*" not in composition.args or arg in composition.args:
            return Meaning(arg, arg, None, turn)
        if self.room is None:  # no places to ask: neither a place nor a referent for sure
            return Meaning(arg, NAME, None, turn)
        place = self.room.find(arg)
        return Meaning(arg, THING if place is None else PLACE, place, turn)

    def fits(self, variant: Variant, meaning: Meaning) -> bool:
        """Whether ``variant`` is written for the marker's argument (or for any)."""
        need = variant.needs.get(ARG_NEED)
        return need is None or need in (meaning.kind, meaning.arg)

    def repeated(self, book: Book, composition: Composition, meaning: Meaning) -> bool:
        """Whether the brain already wrote this mark with this argument in this reply (then it
        plays once; logged); the reply's marks remembered."""
        if meaning.turn is None or composition.name in book.space.again:
            return False
        said = self._said.setdefault(meaning.turn, set())
        mark = (composition.name, meaning.arg)
        if mark in said:
            self.record("repeat", {"intent": composition.name, "arg": meaning.arg,
                                   "turn": meaning.turn})  # fmt: skip
            return True
        said.add(mark)
        for old in [t for t in self._said if t < meaning.turn - 2]:
            del self._said[old]
        return False

    # -- the arm's places ------------------------------------------------------------------------

    def can(self, ref: str, ctx: Context) -> bool:
        """Whether ``ref`` would resolve (asking places no catchment)."""
        if ref in (FROM, VIA):
            return True
        if ref == "@locus":  # placed by the brain's first rendering, found by the others
            if ctx.meaning.kind != THING or not ctx.book.space.loci:
                return False
            return ctx.commit or any(x.referent == ctx.meaning.arg for x in self.loci.values())
        if ref == "@count":
            return self._count_cell(ctx) is not None
        return self.pose(ref, ctx, {}) is not None

    def pose(self, ref: str, ctx: Context, start: Pose) -> Pose | None:
        """The pose ``ref`` names (``@via`` aside: :meth:`via`); None: unresolved."""
        if ref == FROM:
            return start or ctx.book.poses.get("rest")
        if ref in ctx.resolved:
            return ctx.resolved[ref]
        found = self._pose(ref, ctx)
        if ref not in ("@locus", "@count") or ctx.commit:
            ctx.resolved[ref] = found
        return found

    def _pose(self, ref: str, ctx: Context) -> Pose | None:
        book, space, meaning = ctx.book, ctx.book.space, ctx.meaning
        if not ref.startswith(REF):
            return book.poses.get(ref)
        if ref.startswith(THERE):
            return book.poses.get(ref[len(THERE) :])
        if ref.startswith(PATH):
            path = space.paths.get(f"{ctx.composition.name}:{meaning.arg}", ())
            k = ref[len(PATH) :]
            if not path or not k.isdigit():
                return None
            return book.poses.get(path[min(int(k), len(path) - 1)])
        if ref == "@toward":
            return self._pointing(book, space.open_hand, self._person(book))
        if ref == "@bearing":
            return self._pointing(book, space.shut_hand, self._place(meaning))
        if ref == "@locus":
            cell = self._locus(ctx)
            return None if cell is None else {**book.poses[cell], **SHUT}
        if ref == "@count":
            cell = self._count_cell(ctx)
            if cell is not None and ctx.commit:
                self._take(cell, f"#{meaning.arg}", "overwritten")
            return None if cell is None else book.poses.get(cell)
        return None

    def back(self, ctx: Context, final: Pose, final_ref: str) -> Pose | None:
        """Home, where an arm step that ends on a place of the space inward of
        ``space.via_pan_deg`` goes after its hold: the board's own way to rest from there (a
        straight line, when the hold lapses or the clip ends) would pass the speaker box at
        0.2-1.1 cm; from home it keeps 1.7 cm. None: it stays."""
        space, book = ctx.book.space, ctx.book
        home = book.poses.get(space.home)
        if home is None or space.via_pan_deg is None or not final_ref.startswith(REF):
            return None
        if final_ref in (FROM, VIA) or _same(final, home):
            return None
        return home if final["shoulder_pan"] < space.via_pan_deg else None

    def via(self, ctx: Context, start: Pose, target: Pose, target_ref: str) -> Pose | None:
        """Home, when the way from ``start`` to ``target`` (a place of the space) must pass it;
        None: straight."""
        space, book = ctx.book.space, ctx.book
        home = book.poses.get(space.home)
        if home is None or not target_ref.startswith(REF) or target_ref in (FROM, VIA):
            return None
        if _same(start, home) or _same(target, home):
            return None
        named = target_ref[len(THERE) :] if target_ref.startswith(THERE) else ""
        inward = space.via_pan_deg is not None and target["shoulder_pan"] < space.via_pan_deg
        return home if inward or named in space.via else None

    # -- the head's directions -------------------------------------------------------------------

    def head(self, ref: str, ctx: Context) -> tuple[float, float] | None:
        """The (pan, tilt) offset in degrees a head step's ``ref`` turns toward; None: none."""
        space, meaning = ctx.book.space, ctx.meaning
        if ref == "@arg":
            return space.head.get(f"{ctx.composition.name}:{meaning.arg}")
        if ref == "@bearing":
            there = self._place(meaning)
            if there is None:
                return None
            person = self._person(ctx.book)
            look = _seen_from(space.neck_m, person)
            target = _seen_from(space.neck_m, there)
            turn = (target - look + 180.0) % 360.0 - 180.0
            return None if not GLANCE_DEG <= abs(turn) <= AWAY_DEG else (turn, 0.0)
        return None

    # -- what is where ---------------------------------------------------------------------------

    def _person(self, book: Book) -> tuple[float, float]:
        """The person engaged as a point of the cart's frame (metres): the attention view's
        bearing and distance, else straight ahead at ``space.person_m``."""
        space = book.space
        bearing, distance = 0.0, space.person_m
        if self.attention is not None:
            try:
                view = self.attention.attention()
            except Exception:
                logger.exception("resolver: the attention view broke")
            else:
                bearing = view.bearing_deg if view.bearing_deg is not None else 0.0
                distance = view.distance_m if view.distance_m is not None else space.person_m
        x, y = space.neck_m
        return x + distance * math.cos(math.radians(bearing)), y + distance * math.sin(
            math.radians(bearing)
        )

    def _place(self, meaning: Meaning) -> tuple[float, float] | None:
        """The marker's place as a point of the cart's frame, None without one or a pose."""
        if meaning.kind != PLACE or meaning.place is None or self.room is None:
            return None
        where = self.room.where(meaning.place)
        if where is None:
            return None
        bearing, distance = where
        return distance * math.cos(math.radians(bearing)), distance * math.sin(
            math.radians(bearing)
        )

    def _pointing(
        self, book: Book, hand: str | None, there: tuple[float, float] | None
    ) -> Pose | None:
        """``hand`` (a pose pointing straight ahead) turned on its pan at ``there``, None
        outside the clear pan range."""
        if hand is None or there is None or hand not in book.poses:
            return None
        pose = book.poses[hand]
        pan = pose["shoulder_pan"] - _seen_from(book.space.arm_axis_m, there)
        lo, hi = book.space.pan_deg
        return {**pose, "shoulder_pan": round(pan, 1)} if lo <= pan <= hi else None

    # -- catchments ------------------------------------------------------------------------------

    def _locus(self, ctx: Context) -> str | None:
        """The referent's cell: its own, or (the brain's first rendering of the marker) the
        first free one, the least recently pointed one evicted when none is free."""
        referent, loci = ctx.meaning.arg, ctx.book.space.loci
        if ctx.meaning.kind != THING or referent is None or not loci:
            return None
        for locus in self.loci.values():
            if locus.referent == referent:
                if ctx.commit:
                    locus.used = next(self._order)
                    self.record("catchment", {"referent": referent, "cell": locus.cell,
                                              "how": "again"})  # fmt: skip
                return locus.cell
        if not ctx.commit:
            return None
        free = [c for c in loci if c not in self.loci]
        oldest = min(self.loci.values(), key=lambda locus: locus.used) if not free else None
        cell = free[0] if oldest is None else oldest.cell
        self._take(cell, referent, "evicted")
        self.record("catchment", {"referent": referent, "cell": cell, "how": "new"})
        return cell

    def _count_cell(self, ctx: Context) -> str | None:
        """The n-th cell of the loci for ``{count:n}``."""
        arg, loci = ctx.meaning.arg, ctx.book.space.loci
        if arg is None or not arg.isdigit() or not 1 <= int(arg) <= len(loci):
            return None
        return loci[int(arg) - 1]

    def _take(self, cell: str, referent: str, how: str) -> None:
        """``cell`` given to ``referent``, whoever sat there let go (logged ``how``)."""
        old = self.loci.get(cell)
        if old is not None and old.referent != referent:
            self.record("catchment", {"referent": old.referent, "cell": cell, "how": how})
        self.loci[cell] = Locus(referent, cell, next(self._order))


def _seen_from(origin: tuple[float, float], point: tuple[float, float]) -> float:
    """The bearing of ``point`` from ``origin`` (cart frame, degrees, + left)."""
    return math.degrees(math.atan2(point[1] - origin[1], point[0] - origin[0]))


def _same(a: Pose, b: Pose) -> bool:
    return all(abs(a.get(j, math.inf) - v) < 1e-6 for j, v in b.items())


def _travel(a: Pose, b: Pose) -> float:
    return max((abs(a[j] - b[j]) for j in b if j in a), default=0.0)


def _leg_s(degrees: float) -> float:
    """A two-key leg of ``degrees`` at the caps (the book's cubic)."""
    if degrees <= 0.0:
        return 0.0
    return max(1.5 * degrees / CAPS_DEG_S, math.sqrt(6.0 * degrees / CAPS_DEG_S2))


class WeightedChooser:
    """Variants drawn by weight among those the scene allows (:func:`allowed`) and that are no
    floor (the scheduler's last rung, :mod:`pepin.expression.schedule`), never the one drawn last
    time for the same intent; a composition of floors alone draws among them. ``fits`` narrows
    the draw to the variants written for the marker's argument whose places resolve."""

    def __init__(self) -> None:
        """No intent performed yet."""
        self.last: dict[str, int] = {}

    def choose(
        self,
        composition: Composition,
        scene: Scene,
        rng: random.Random,
        fits: Callable[[Variant], bool] | None = None,
    ) -> int:
        """The index of the variant to perform."""
        variants = composition.variants
        everyone = [i for i in range(len(variants)) if fits is None or fits(variants[i])]
        everyone = everyone or list(range(len(variants)))
        able = [i for i in everyone if allowed(variants[i], scene)] or everyone
        drawable = [i for i in able if not variants[i].floor] or able
        weights = [variants[i].weight for i in drawable]
        last = self.last.get(composition.name)
        if len(drawable) > 1 and last in drawable:
            weights[drawable.index(last)] = 0.0
        pick = rng.choices(drawable, weights)[0]
        self.last[composition.name] = pick
        return pick


def allowed(variant: Variant, scene: Scene) -> bool:
    """Whether ``scene`` allows ``variant``: every part it needs is ready, and the arm is where
    it needs it (``any``: anywhere; ``rest``: at rest or unknown; ``up``: holding a pose)."""
    for part, state in variant.needs.items():
        if part == ARG_NEED:
            continue
        if part not in scene.ready:
            return False
        if part == "arm" and state == "rest" and scene.arm_at not in (None, "rest"):
            return False
        if part == "arm" and state == "up" and scene.arm_at in (None, "rest"):
            return False
    return True


@dataclass(frozen=True)
class Drawn:
    """One step as drawn: whether it plays by its odds, where its stroke lands from the sync
    point, its parameters, its names and whether it is the mirror image."""

    step: Step
    atom: Atom
    plays: bool
    at_s: float
    params: Mapping[str, float]
    names: Mapping[str, str]
    mirrored: bool


Keyed = list[tuple[Key, float, Pose]]  # an atom's keys with their shares and resolved poses


class BookRenderer:
    """:class:`pepin.expression.animation.Renderer` over a book (with the arm's rest pose among
    its poses, :func:`pepin.expression.book.with_rest`) and the gesture space's
    :class:`Resolver` (a fresh one, knowing no person and no place, when none is given)."""

    def __init__(
        self,
        book: Book,
        chooser: WeightedChooser | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        """``chooser`` defaults to a fresh :class:`WeightedChooser`."""
        self.book = book
        self.chooser = chooser or WeightedChooser()
        self.resolver = resolver or Resolver()

    def render(
        self,
        intent: str,
        scene: Scene,
        seed: int,
        cue: Cue | None = None,
        *,
        variant: int | None = None,
    ) -> Performance:
        """The performance of ``intent`` (``variant`` forces one, for a replay or a rung of the
        ladder); ``KeyError`` for an intent the book does not hold. The brain's marker rendered
        first (``cue`` and no ``variant``) places its catchment; a repeat of it in the same
        reply, or a variant written for another argument, renders no clips."""
        composition = self.book.compositions[intent]
        rng = random.Random(seed)
        resolver = self.resolver
        meaning = resolver.meaning(composition, cue)
        first = variant is None and cue is not None
        ctx = Context(self.book, composition, meaning, commit=first)

        def fits(v: Variant) -> bool:
            return resolver.fits(v, meaning) and self._resolvable(v, ctx)

        index = variant if variant is not None else self.chooser.choose(composition, scene, rng,
                                                                        fits)  # fmt: skip
        chosen = composition.variants[index]
        if not resolver.fits(chosen, meaning) or (
            first and resolver.repeated(self.book, composition, meaning)
        ):
            return Performance(intent, seed, index, (), cue)
        drawn = [self._aimed(self._draw(step, rng), ctx) for step in chosen.steps]
        if any(self._unplaced(d, ctx) for d in drawn):  # the arm's strokes need their place
            drawn = [d for d in drawn if d.atom.part != "arm"]
        able = [d for d in drawn if d.atom.part in scene.ready and self._can(d, ctx)]
        playing = [d for d in able if d.plays]
        if not playing and able:
            playing = [max(able, key=lambda d: d.step.p)]
        keyed = self._keyed_all(playing, ctx)  # the brain's first rendering places here
        clips = merge([self._unroll(d, *keyed[id(d)]) for d in playing if id(d) in keyed],
                      "gesture")  # fmt: skip
        return Performance(intent, seed, index, clips, cue)

    def render_atom(self, name: str, seed: int, layer: Layer = "idle") -> Clip:
        """One atom alone, its parameters drawn from its own ranges (the idle layer's); a
        looping atom's clip is marked to repeat on its owner."""
        drawn = self._draw(Step(name), random.Random(seed))
        return self._alone(drawn, layer)

    def atom_clip(
        self,
        name: str,
        params: Mapping[str, float],
        *,
        mirrored: bool = False,
        layer: Layer = "gesture",
    ) -> Clip:
        """One atom alone at the given parameters (every one of its own, the reserved ones
        included when it has them), mirrored or not: the extremes the authoring checks play."""
        atom = self.book.atoms[name]
        return self._alone(Drawn(Step(name), atom, True, 0.0, params, {}, mirrored), layer)

    def _alone(self, drawn: Drawn, layer: Layer) -> Clip:
        composition = Composition("", (Variant((drawn.step,)),))
        ctx = Context(self.book, composition, Meaning(None, None), commit=False)
        keyed = self._keyed(drawn, ctx, {})
        if keyed is None:
            raise KeyError(f"{drawn.atom.name}: a pose it names is not in the book")
        (clip,) = merge([self._unroll(drawn, keyed, None)], layer)
        return replace(clip, loop=drawn.atom.loop)

    def _draw(self, step: Step, rng: random.Random) -> Drawn:
        atom = self.book.atoms[step.atom]
        plays = rng.random() < step.p
        at_s = rng.uniform(step.at_s.lo, step.at_s.hi)
        ranges = {**atom.params, **step.params}
        params = {k: rng.uniform(r.lo, r.hi) for k, r in sorted(ranges.items())}
        names = {k: rng.choice(options) for k, options in sorted(step.choices.items())}
        mirrored = rng.random() < 0.5 if atom.mirror else False
        return Drawn(step, atom, plays, at_s, params, names, mirrored)

    # -- the gesture space -----------------------------------------------------------------------

    def _resolvable(self, variant: Variant, ctx: Context) -> bool:
        """Whether every place a sure arm step of ``variant`` names resolves (a head step's
        direction that does not is left out alone, as a step left to chance may be)."""
        for step in variant.steps:
            atom = self.book.atoms[step.atom]
            if step.p < 1.0 or atom.part != "arm":
                continue
            refs = [str(k.pose) for k in atom.keys if k.pose and not k.pose.startswith("$")]
            refs += list(step.choices.get("pose", ()))
            if not all(self.resolver.can(ref, ctx) for ref in refs if ref.startswith(REF)):
                return False
        return True

    def _can(self, drawn: Drawn, ctx: Context) -> bool:
        """Whether a drawn step can play: its head's direction resolved, every pose its arm
        keys name taught or resolvable."""
        if drawn.atom.part == "head":
            return drawn.names.get("pose", None) != ""
        if drawn.atom.part != "arm":
            return True
        for key in drawn.atom.keys:
            name = pose_name(key.pose, drawn.names) if key.pose else None
            if name is None:
                continue
            if not name or not self.resolver.can(name, ctx):
                return False
        return True

    def _unplaced(self, drawn: Drawn, ctx: Context) -> bool:
        """Whether ``drawn`` moves the arm to a place of the space that does not resolve (a
        point outside the pans, no place): the variant's other arm steps are left out with it."""
        if drawn.atom.part != "arm" or drawn.atom.mode != "absolute":
            return False
        names = [pose_name(k.pose, drawn.names) for k in drawn.atom.keys if k.pose]
        spatial = any(n.startswith(REF) and n not in (FROM, VIA) for n in names)
        return spatial and not self._can(drawn, ctx)

    def _aimed(self, drawn: Drawn, ctx: Context) -> Drawn:
        """A head step naming a place of the space turned toward it: its ``pan_deg`` and
        ``tilt_deg`` the place's direction, clamped into the step's ranges."""
        ref = drawn.names.get("pose", "")
        if drawn.atom.part != "head" or not ref.startswith(REF):
            return drawn
        direction = self.resolver.head(ref, ctx)
        if direction is None:
            return replace(drawn, names={**drawn.names, "pose": ""})  # unresolved: left out
        ranges = {**drawn.atom.params, **drawn.step.params}
        params = dict(drawn.params)
        for name, value in zip(("pan_deg", "tilt_deg"), direction, strict=True):
            if name in ranges:
                params[name] = min(max(value, ranges[name].lo), ranges[name].hi)
        return replace(drawn, params=params)

    def _keyed_all(
        self, drawn: Sequence[Drawn], ctx: Context
    ) -> dict[int, tuple[Keyed, Pose | None]]:
        """Each step's keys resolved (a step that cannot resolve is absent) and the pose its
        arm goes back to after its hold (home from the middle: :meth:`Resolver.back`; None:
        none), the arm's steps in the order they start, each from where the one before leaves
        the arm."""
        out: dict[int, tuple[Keyed, Pose | None]] = {}
        start: Pose = {}
        arm = sorted(
            (d for d in drawn if d.atom.part == "arm"),
            key=lambda d: d.at_s - d.atom.stroke * d.params[PERIOD],
        )
        for d in [*arm, *(d for d in drawn if d.atom.part != "arm")]:
            keyed = self._keyed(d, ctx, start)
            if keyed is None:
                continue
            back = None
            if d.atom.part == "arm" and d.atom.mode == "absolute" and keyed:
                last = keyed[-1]
                start = {**last[2], **_numbers(last[0], d.params)}
                final = pose_name(d.atom.keys[-1].pose, d.names)
                back = self.resolver.back(ctx, start, final)
                start = back or start
            out[id(d)] = (keyed, back)
        return out

    def _keyed(self, drawn: Drawn, ctx: Context, start: Pose) -> Keyed | None:
        """The atom's keys with their shares and poses (a key without a pose: none), the via
        placed or left out; None when a pose cannot be resolved."""
        atom = drawn.atom
        if atom.part != "arm":
            return [(key, key.at, {}) for key in atom.keys]
        raw: list[tuple[Key, str]] = [(k, pose_name(k.pose, drawn.names)) for k in atom.keys]
        poses: list[Pose | None] = []
        for key, name in raw:
            if not key.pose or name == VIA:
                poses.append({})
                continue
            if not name:
                return None
            found = self.resolver.pose(name, ctx, start)
            if found is None:
                return None
            poses.append(found)
        out: Keyed = []
        for i, (key, name) in enumerate(raw):
            if name != VIA:
                out.append((key, key.at, poses[i] or {}))
                continue
            before = out[-1] if out else None
            later = zip(raw[i + 1 :], poses[i + 1 :], strict=True)
            after = next(((k, n, p) for (k, n), p in later if n != VIA), None)
            if before is None or after is None or not before[2] or not after[2]:
                continue
            via = self.resolver.via(ctx, before[2], after[2], after[1])
            if via is None:
                continue
            first, second = _leg_s(_travel(before[2], via)), _leg_s(_travel(via, after[2]))
            if first + second <= 0.0:
                continue
            share = before[1] + (after[0].at - before[1]) * first / (first + second)
            out.append((key, share, via))
        paths = [n for _, n in raw if n.startswith(PATH)]
        return _traced(out, len(paths)) if len(paths) > 1 else out

    def _unroll(self, drawn: Drawn, keyed: Keyed, back: Pose | None) -> Unrolled:
        """A step's atom over its cycles, and the arm's way ``back`` home after its hold at
        :data:`BACK_SHARE` of the caps (None: none): (part, start from the sync point, stroke
        from its start, duration, tracks, events)."""
        atom, params = drawn.atom, drawn.params
        period = params[PERIOD]
        repeats = max(1, round(params.get(REPEATS, 1.0)))
        hold = params.get(HOLD, 0.0)
        stroke = atom.stroke * period
        start = drawn.at_s - stroke
        times: dict[str, list[float]] = {}
        values: dict[str, list[float]] = {}
        events: list[Event] = []
        for cycle in range(repeats):
            for key, at, pose in keyed:
                if cycle > 0 and key.at == 0.0 and atom.keys[-1].at == 1.0:
                    continue  # the previous cycle's last key stands there already
                t = (cycle + at) * period
                joints = {**pose, **{k: v for k, v in key.values.items() if k != "expression"}}
                for dof, raw in joints.items():
                    value = number(raw, params)
                    if drawn.mirrored and dof in atom.mirror:
                        value = -value
                    times.setdefault(dof, []).append(t)
                    values.setdefault(dof, []).append(value)
                face = key.values.get("expression")
                if isinstance(face, str):
                    name = drawn.names.get("name", "") if face == "$name" else face
                    events.append(Event(t, name, hold if hold > 0 else None))
        end = repeats * period
        if hold > 0 and atom.part != "face":
            for dof in times:
                times[dof].append(end + hold)
                values[dof].append(values[dof][-1])
        duration = end + hold
        if back is not None:
            here = {dof: vals[-1] for dof, vals in values.items()}
            duration += _leg_s(_travel(here, back)) / BACK_SHARE
            for dof in times:
                if dof in back:
                    times[dof].append(duration)
                    values[dof].append(back[dof])
        tracks = [Track(dof, atom.mode, tuple(times[dof]), tuple(values[dof])) for dof in times]
        return atom.part, start, stroke, duration, tracks, events


def _traced(keyed: Keyed, legs: int) -> Keyed:
    """An atom's last ``legs`` keys (a path's waypoints) re-timed over their span so each leg
    takes its share by its travel (one share of the caps for all), a waypoint the path repeats
    no key: a path shorter than the atom's keys fills its whole time."""
    head, path = keyed[:-legs], keyed[-legs:]
    kept = [path[0]]
    for item in path[1:]:
        if not _same(item[2], kept[-1][2]):
            kept.append(item)
    begin, end = path[0][1], path[-1][1]
    times = [0.0, *itertools.accumulate(
        _leg_s(_travel(a[2], b[2])) for a, b in itertools.pairwise(kept)
    )]  # fmt: skip
    total = times[-1]
    if total <= 0.0:
        return [*head, path[0]]
    return [*head, *((k, begin + (end - begin) * t / total, p)
                     for (k, _, p), t in zip(kept, times, strict=True))]  # fmt: skip


def _numbers(key: Key, params: Mapping[str, float]) -> dict[str, float]:
    """A key's own joint values as numbers."""
    return {k: number(v, params) for k, v in key.values.items() if k != "expression"}


Unrolled = tuple[PartName, float, float, float, list[Track], list[Event]]


def merge(unrolled: Sequence[Unrolled], layer: Layer) -> tuple[Clip, ...]:
    """The steps of each part as one clip, the part's first step setting its stroke."""
    clips = []
    for part in dict.fromkeys(u[0] for u in unrolled):
        steps = [u for u in unrolled if u[0] == part]
        offset = min(u[1] for u in steps)
        end = max(u[1] + u[3] for u in steps)
        tracks: list[Track] = []
        events: list[Event] = []
        for _, start, _, _, step_tracks, step_events in steps:
            shift = start - offset
            tracks += [
                Track(tr.dof, tr.mode, tuple(t + shift for t in tr.times), tr.values)
                for tr in step_tracks
            ]
            events += [Event(e.t_s + shift, e.name, e.hold_s) for e in step_events]
        first = steps[0]
        clips.append(
            Clip(
                part,
                offset,
                first[1] - offset + first[2],
                end - offset,
                tuple(tracks),
                tuple(sorted(events, key=lambda e: e.t_s)),
                layer,
            )
        )
    return tuple(clips)


def pose_name(pose: str | None, names: Mapping[str, str]) -> str:
    """A key's pose: its name, or the drawn one for ``$pose``."""
    if pose is None:
        return ""
    return names.get(pose[1:], "") if pose.startswith("$") else pose


def number(raw: float | str, params: Mapping[str, float]) -> float:
    """A key's value: a number, or a drawn parameter (``"$p"``, ``"-$p"``)."""
    if isinstance(raw, str):
        sign = -1.0 if raw.startswith("-") else 1.0
        return sign * params[raw.lstrip("-").lstrip("$")]
    return float(raw)
