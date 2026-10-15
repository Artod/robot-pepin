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
(``needs: {"arg": ...}``) renders only for that argument; another renders to no clips. One that
needs the person seen (``needs: {"person": "seen"}``: the waves of hello and bye) is never drawn
while the scene says why they are unseen (:attr:`pepin.expression.animation.Scene.unseen`), nor
a rung of the ladder: the floor plays.

A mark's arguments that set parameters (:attr:`pepin.expression.markers.Marker.args`: the
composition's ``settings``, an auto-wrapper's numbers in its atom's order, ``key=value``)
replace the ranges of the steps they name before anything is drawn; one that names nothing
here is left out (the parser left out the rest). A NESTED composition (a step that names one)
draws its own variant, by the same scene rules and with no memory of the last (the floor when
nothing else fits), and its own parameters, from ``random.Random`` of a seed derived from its
parent's and the step's index (:func:`sub_seed`): the parent's seed replays it. Its steps land
at the parent step's ``at_s`` plus their own, merge into the parts' clips as any step's (a later
one on the same part cuts the earlier), take no argument of the mark, and are listed in
:attr:`pepin.expression.animation.Performance.nested`.

THE RESOLVER (:class:`Resolver`; DESIGN.md §4.3-§4.6, the gesture space of config/animation.json's
``space``). A pose named ``@...`` in an atom's key or a step's ``pose`` is a place in the space:

* ``@from``: where the step starts: the previous arm step of the performance's last pose, else
  rest (a unit's held pose replaces rest when the arm part plans the clip from it);
* ``@via``: home (``space.home``), when the step's target (the key after it) is a place of the
  space inward of ``space.via_pan_deg`` or a pose of ``space.via``: straight from rest the hand
  passes the speaker box at 1.1 cm on its way to the middle, and the ways between ``stop``,
  ``tucked`` and the sweep's top pass the head (scratch/gesticulation_space/cells.py); its time
  divides the way so both legs run at one share of the caps; otherwise no key; and a
  performance's LAST arm step that ends there, inward of ``space.via_pan_deg``, goes back home
  after its hold (the board's own straight way to rest from the middle would pass the speaker
  box at 0.2-1.1 cm), or to the pose ``space.home_after`` names for its pose (``down`` to
  ``down_near``: straight to rest the hand would drag down the cart's front, 2026-10-14); an
  earlier arm step hands its pose to the next one as it is (``up_low``, then ``up``);
* ``@pose:NAME``: the book's pose NAME as a place of the space (the via applies); none while
  the head looks further right than ``space.head_pan_min_deg`` allows NAME (where the head
  looks: at the person engaged, as a head step's glance reckons it; the ways to and from
  ``back`` pass the head's pan sweep: 1.6 cm at -10 deg, 0.7 at -15: the variant's floor plays);
* ``@toward``: the open hand (``space.open_hand``) turned at the person engaged (the attention
  view's bearing and distance), ``@bearing``: the shut hand at the marker's place (the room's);
  a bearing past the clear pan range (``space.pan_deg``) gets the range's edge nearest it: the
  arm points as far as it reaches (logged ``reach``, with how far short), the head's glance
  carrying the rest (2026-10-13: {point:printer} at a place past the pans rendered nothing and
  the arm just rose for the posture);
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
pointed one is evicted for it; ``{count:n}`` takes the n-th cell for its item (``#n``),
overwriting what sat there. Forgotten at the conversation's close (:meth:`Resolver.forget`).
Logged as ``catchment`` rows (referent or item, cell, new / again / evicted / overwritten). A
mark the brain repeats in one reply with the same argument plays once (``repeat`` rows), but
those of ``space.again`` (pointing back at a referent, the accent).

Owners after slice 0: B (the chooser, the scenes, the ladder's renderings), C (the resolver).
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pepin.arm_motion import GRIPPER_ZERO_DEG
from pepin.expression.animation import (
    HOLD,
    NUMBER,
    PERIOD,
    PERSON,
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
    Range,
    Scene,
    Setting,
    Step,
    Track,
    Variant,
)
from pepin.expression.markers import Cue, number_of
from pepin.expression.places import Room
from pepin.expression.speaker import AttentionView

logger = logging.getLogger(__name__)

Record = Callable[[str, dict[str, Any]], None]
Pose = Mapping[str, float]
REF = "@"  # a pose that names a place of the gesture space
FROM, VIA, THERE, PATH = "@from", "@via", "@pose:", "@path:"
ARG_NEED = "arg"  # a variant's need of its marker's argument
ATTENTION = "attention"  # a variant's need of attention's phase
# The kinds of a pointing argument: the person, the robot, a place of the room, a referent, and
# a name no room was there to tell (its floor only: never a referent's cell for a place)
YOU, ME, PLACE, THING, NAME = "you", "me", "place", "thing", "name"
# A referent's spot is pointed at with the gripper shut (the finger): the jaw's closed end,
# config/arm.json's gripper limit (-10; 2026-10-14: the jaw pressed shut by hand reads -12, so
# the 0 of before left the tip a centimetre open; the book's shut poses and gripper_snap's
# shut_deg stand on the same number, a test holds them together)
SHUT = {"gripper": GRIPPER_ZERO_DEG}
# A place further than this from the person gets no head glance: behind their back the glance
# reads as looking away (90 until 2026-10-13: the printer at ~95 deg from Artem got none)
AWAY_DEG = 120.0
# A bearing this far past the clear pan range's edge still gets the edge (the arm as far as it
# reaches: the printer left of the pans); further is behind the robot, where an arm says nothing
REACH_DEG = 60.0
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
        away); ``room``: the places (None, or a room whose book was never read: a name is
        neither a place nor a referent, its floor plays); ``record(kind, fields)``: the
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

    def meaning(self, composition: Composition, cue: Cue | None, arg: str | None = None) -> Meaning:
        """The meaning of ``cue``'s argument for ``composition`` (without a cue: ``arg``, a
        person's ask's)."""
        if cue is not None:
            arg = cue.marker.arg if cue.marker.name == composition.name else None
        if arg is None:
            arg = next((a for a in composition.args if a != "*"), None)
        turn = cue.turn if cue is not None else None
        if arg is None or "*" not in composition.args or arg in composition.args:
            return Meaning(arg, arg, None, turn)
        if self.room is None or not self.room.known():  # no places to ask (yet): neither a
            return Meaning(arg, NAME, None, turn)  # place nor a referent for sure
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
            name = ref[len(THERE) :]
            least = space.head_pan_min_deg.get(name)
            if least is not None and _seen_from(space.neck_m, self._person(book)) < least:
                return None  # its ways pass the head turned this far right
            return book.poses.get(name)
        if ref.startswith(PATH):
            path = space.paths.get(f"{ctx.composition.name}:{meaning.arg}", ())
            k = ref[len(PATH) :]
            if not path or not k.isdigit():
                return None
            return book.poses.get(path[min(int(k), len(path) - 1)])
        if ref == "@toward":
            return self._pointing(book, space.open_hand, self._person(book), ctx.commit)
        if ref == "@bearing":
            return self._pointing(book, space.shut_hand, self._place(meaning), ctx.commit)
        if ref == "@locus":
            cell = self._locus(ctx)
            return None if cell is None else {**book.poses[cell], **SHUT}
        if ref == "@count":
            cell = self._count_cell(ctx)
            if cell is not None and ctx.commit:
                item = f"#{meaning.arg}"
                old = self.loci.get(cell)
                self._take(cell, item, "overwritten")
                how = "again" if old is not None and old.referent == item else "new"
                self.record("catchment", {"referent": item, "cell": cell, "how": how})
            return None if cell is None else book.poses.get(cell)
        return None

    def back(self, ctx: Context, final: Pose, final_ref: str) -> Pose | None:
        """Where the performance's last arm step goes after its hold when it ends on a place of
        the space: home for one inward of ``space.via_pan_deg`` (the board's own way to rest
        from there, a straight line when the hold lapses or the clip ends, would pass the
        speaker box at 0.2-1.1 cm; from home it keeps 1.7 cm), the pose ``space.home_after``
        names for a pose named there (``down``: straight to rest the hand would drag down the
        cart's front). None: it stays."""
        space, book = ctx.book.space, ctx.book
        home = book.poses.get(space.home)
        if home is None or not final_ref.startswith(REF) or final_ref in (FROM, VIA):
            return None
        named = final_ref[len(THERE) :] if final_ref.startswith(THERE) else ""
        after = book.poses.get(space.home_after.get(named, ""))
        if after is not None:
            return after
        if space.via_pan_deg is None or _same(final, home):
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
        self,
        book: Book,
        hand: str | None,
        there: tuple[float, float] | None,
        commit: bool = False,
    ) -> Pose | None:
        """``hand`` (a pose pointing straight ahead) turned on its pan at ``there``; a bearing
        past the clear pan range by up to :data:`REACH_DEG` gets the range's edge nearest it,
        as far as the arm reaches (logged ``reach`` on the brain's first rendering,
        ``commit``); None without a hand or a place, or for a bearing further behind."""
        if hand is None or there is None or hand not in book.poses:
            return None
        pose = book.poses[hand]
        bearing = _seen_from(book.space.arm_axis_m, there)
        pan = pose["shoulder_pan"] - bearing
        lo, hi = book.space.pan_deg
        reached = min(max(pan, lo), hi)
        if abs(pan - reached) > REACH_DEG:
            return None
        if commit and abs(reached - pan) > 1e-9:
            self.record("reach", {"hand": hand, "bearing_deg": round(bearing, 1),
                                  "pan_deg": round(reached, 1),
                                  "short_deg": round(abs(pan - reached), 1)})  # fmt: skip
        return {**pose, "shoulder_pan": round(reached, 1)}

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
    """Whether ``scene`` allows ``variant``: the arm is where it needs it (``rest``: at rest or
    unknown, ready or not: a variant that leaves it alone; ``up``: holding a pose; ``any``:
    anywhere), the face shows the expression it names (``any``: the face ready), attention is
    in the phase it names, the person is seen when it needs them (``person``: no
    ``scene.unseen``), and every other part it needs is ready."""
    for part, state in variant.needs.items():
        if part == ARG_NEED:
            continue
        if part == PERSON:
            if scene.unseen is not None:
                return False
            continue
        if part == ATTENTION:
            if scene.attention != state:
                return False
            continue
        if part == "face" and state not in ("any", "rest", "up"):
            if scene.face != state:
                return False
            continue
        down = part != "arm" or scene.arm_at in (None, "rest")
        if state == "rest":
            if not down:
                return False
            continue
        if part not in scene.ready or (state == "up" and down):
            return False
    return True


@dataclass(frozen=True)
class Drawn:
    """One step as drawn: whether it plays by its odds, where its stroke lands from the sync
    point, its parameters, its names, whether it is the mirror image, and the rendering it
    resolves its places in (None: the performance's own; a nested composition's)."""

    step: Step
    atom: Atom
    plays: bool
    at_s: float
    params: Mapping[str, float]
    names: Mapping[str, str]
    mirrored: bool
    ctx: Context | None = field(default=None, compare=False)


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
        arg: str | None = None,
        args: tuple[str, ...] = (),
        layer: Layer = "gesture",
    ) -> Performance:
        """The performance of ``intent`` (``variant`` forces one, for a replay or a rung of the
        ladder; ``arg`` and ``args``: a person's ask's arguments, with no cue; ``layer``: the
        idle layer's own); ``KeyError`` for an intent the book does not hold. The brain's
        marker rendered first (``cue`` and no ``variant``), or an ask with its argument, places
        its catchment; a repeat of the marker in the same reply, or a variant written for
        another argument, renders no clips."""
        composition = self.book.compositions[intent]
        if cue is not None and cue.marker.name == intent:
            args = cue.marker.args
        settings = self.settings(composition, args)
        rng = random.Random(seed)
        resolver = self.resolver
        meaning = resolver.meaning(composition, cue, arg)
        first = variant is None and (cue is not None or arg is not None)
        ctx = Context(self.book, composition, meaning, commit=first)
        # a variant written for the argument (its need) wins over one written for any: the
        # laugh's sinister variant for {laugh} in a sinister part, never a happy one
        exact = any(
            v.needs.get(ARG_NEED) is not None and resolver.fits(v, meaning)
            for v in composition.variants
        )

        def fits(v: Variant) -> bool:
            if exact and v.needs.get(ARG_NEED) is None:
                return False
            return resolver.fits(v, meaning) and self._resolvable(v, ctx)

        index = variant if variant is not None else self.chooser.choose(composition, scene, rng,
                                                                        fits)  # fmt: skip
        chosen = composition.variants[index]
        if not resolver.fits(chosen, meaning) or (exact and chosen.needs.get(ARG_NEED) is None):
            return Performance(intent, seed, index, (), cue)
        if first and resolver.repeated(self.book, composition, meaning):
            return Performance(intent, seed, index, (), cue)
        nested: list[tuple[str, int]] = []
        drawn = self._drawn(chosen.steps, scene, rng, seed, ctx, nested, settings)
        if any(self._unplaced(d, d.ctx or ctx) for d in drawn):  # the arm's strokes need their
            drawn = [d for d in drawn if d.atom.part != "arm"]  # place
        able = [d for d in drawn if d.atom.part in scene.ready and self._can(d, d.ctx or ctx)]
        playing = [d for d in able if d.plays]
        if not playing and able:
            playing = [max(able, key=lambda d: d.step.p)]
        keyed = self._keyed_all(playing, ctx)  # the brain's first rendering places here
        clips = merge([self._unroll(d, *keyed[id(d)]) for d in playing if id(d) in keyed],
                      layer)  # fmt: skip
        looping = {d.atom.part for d in playing if d.step.loop}
        clips = tuple(replace(c, loop=True) if c.part in looping else c for c in clips)
        return Performance(intent, seed, index, clips, cue, tuple(nested))

    def settings(self, composition: Composition, args: Sequence[str]) -> tuple[Setting, ...]:
        """What a mark's arguments set in ``composition``'s steps, in order (later wins): its
        words' settings, its number (an auto-wrapper's: one alone its repeats, several its
        atom's parameters in their order), then each ``key=value``; an argument that sets
        nothing here is left out."""
        out: list[Setting] = []
        numbers = [v for v in (number_of(a) for a in args if "=" not in a) if v is not None]
        for word in args:
            out += [s for s in composition.settings.get(word, ()) if s.value is not None]
        if composition.wraps is not None:
            order = list(self.book.atoms[composition.wraps].params)
            pairs = (
                [(REPEATS, numbers[0])]
                if len(numbers) == 1
                else list(zip(order, numbers, strict=False))
            )
            out += [Setting(param, None, Range(v, v)) for param, v in pairs]
        elif numbers:
            out += [replace(s, value=Range(numbers[0], numbers[0]))
                    for s in composition.settings.get(NUMBER, ())]  # fmt: skip
        for written in args:
            key, equals, raw = written.partition("=")
            value = number_of(raw) if equals else None
            if value is None:
                continue
            if composition.wraps is not None:
                out.append(Setting(key, None, Range(value, value)))
                continue
            named = {(s.param, s.atom) for found in composition.settings.values() for s in found}
            out += [Setting(p, a, Range(value, value)) for p, a in sorted(named, key=str)
                    if p == key]  # fmt: skip
        return tuple(out)

    def _drawn(
        self,
        steps: Sequence[Step],
        scene: Scene,
        rng: random.Random,
        seed: int,
        ctx: Context,
        nested: list[tuple[str, int]],
        settings: Sequence[Setting] = (),
    ) -> list[Drawn]:
        """``steps`` drawn in order (each atom's step aimed; a nested composition's variant and
        steps drawn from its own seed, placed from its step's ``at_s``, listed in ``nested``)."""
        out: list[Drawn] = []
        for index, step in enumerate(steps):
            if step.composition is None:
                out.append(self._aimed(self._draw(self._set(step, settings), rng), ctx))
                continue
            plays = rng.random() < step.p
            at_s = rng.uniform(step.at_s.lo, step.at_s.hi)
            child = self.book.compositions[step.composition]
            child_seed = sub_seed(seed, index)
            child_rng = random.Random(child_seed)
            child_ctx = Context(self.book, child, self.resolver.meaning(child, None), commit=False)
            k = self._child_variant(child, scene, child_rng, child_ctx)
            if k is None:
                continue
            nested.append((child.name, k))
            inner = self._drawn(child.variants[k].steps, scene, child_rng, child_seed, child_ctx,
                                nested)  # fmt: skip
            out += [replace(d, step=replace(d.step, p=d.step.p * step.p), at_s=d.at_s + at_s,
                            plays=d.plays and plays, ctx=d.ctx or child_ctx)
                    for d in inner]  # fmt: skip
        return out

    def _child_variant(
        self, composition: Composition, scene: Scene, rng: random.Random, ctx: Context
    ) -> int | None:
        """A nested composition's variant: drawn by weight among those the scene allows and
        that are no floor, else among its allowed floors, else among its floors (no memory of
        the last: the seed alone decides); None when it has none of these."""
        variants = composition.variants
        able = [i for i, v in enumerate(variants)
                if allowed(v, scene) and self.resolver.fits(v, ctx.meaning)
                and self._resolvable(v, ctx)]  # fmt: skip
        drawable = [i for i in able if not variants[i].floor]
        drawable = drawable or able or [i for i, v in enumerate(variants) if v.floor]
        if not drawable:
            return None
        return rng.choices(drawable, [variants[i].weight for i in drawable])[0]

    def _set(self, step: Step, settings: Sequence[Setting]) -> Step:
        """``step`` with the ranges a mark's arguments set on it (a setting for its atom, or
        for any step whose atom takes the parameter: ``repeats`` and ``hold_s`` any step)."""
        if not settings:
            return step
        atom = self.book.atoms[step.atom]
        params = dict(step.params)
        for s in settings:
            takes = s.param in atom.params or s.param in (REPEATS, HOLD)
            if s.value is not None and takes and s.atom in (None, step.atom):
                params[s.param] = s.value
        return replace(step, params=params) if params != step.params else step

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
        return replace(clip, loop=drawn.step.loop)

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
        direction that does not is left out alone, as a step left to chance may be; a nested
        composition resolves its own when it draws its variant)."""
        for step in variant.steps:
            if step.composition is not None:
                continue
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
        """Each step's keys resolved (a step that cannot resolve is absent) and, for the last
        absolute arm step, the pose the arm goes on to after its hold (home from the middle,
        ``space.home_after``'s: :meth:`Resolver.back`; None: none), the arm's steps in the
        order they start, each from where the one before leaves the arm (an earlier step's
        pose as it is: ``up_low``, then ``up``)."""
        out: dict[int, tuple[Keyed, Pose | None]] = {}
        start: Pose = {}
        tail: tuple[Drawn, Pose, str] | None = None
        arm = sorted(
            (d for d in drawn if d.atom.part == "arm"),
            key=lambda d: d.at_s - d.atom.stroke * d.params[PERIOD],
        )
        for d in [*arm, *(d for d in drawn if d.atom.part != "arm")]:
            keyed = self._keyed(d, d.ctx or ctx, start)
            if keyed is None:
                continue
            if d.atom.part == "arm" and d.atom.mode == "absolute" and keyed:
                last = keyed[-1]
                start = {**last[2], **_numbers(last[0], d.params)}
                tail = (d, start, pose_name(d.atom.keys[-1].pose, d.names))
            out[id(d)] = (keyed, None)
        if tail is not None:
            d, final, final_ref = tail
            out[id(d)] = (out[id(d)][0], self.resolver.back(d.ctx or ctx, final, final_ref))
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


def sub_seed(seed: int, index: int) -> int:
    """The seed a nested composition draws from: its parent's seed and its step's index mixed
    (the same pair, the same seed, on any machine)."""
    digest = hashlib.blake2b(f"{seed}:{index}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


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
