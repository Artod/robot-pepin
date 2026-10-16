"""config/animation.json read into a :class:`pepin.expression.animation.Book`, checked whole.

The file: ``atoms`` (name -> part, mode, stroke, params, keys, mirror, callable, note),
``compositions`` (name -> cooldown_s, bind, args, family, long, callable, variants of steps
with their weight, needs and floor, note), ``poses`` (name -> joint ->
degrees and an optional note; taught by hand with ``scripts/gesture.py teach`` or authored
from the URDF), ``idle`` (the idle layers, naming compositions),
``moments`` (a moment of the robot -> a composition), ``aliases`` (another spelling of a
marker -> its canonical ``name`` or ``name:arg``) and ``space`` (the gesture space:
:class:`pepin.expression.animation.Space`). A parameter is a number or ``[lo, hi]``;
in a step, a string or a list of strings is a named parameter (a face's or a pose's name, one
drawn); ``repeats`` and ``hold_s`` are a step's (an atom refuses them), and so is ``loop``
(``"loop": true``, only in the engine's own compositions). A step names an ``atom``, or a
``composition`` to nest (``at_s`` and ``p`` alone; no cycle, at most :data:`MAX_DEPTH` deep:
the error names the chain). A composition's ``args`` is a list of the arguments that choose a
variant or a place, or a map: an argument -> null (one that chooses), ``"number"`` -> the
parameter a mark's number sets (``"repeats"``, ``"gripper_snap.repeats"``: only that atom's
steps), a word -> parameter -> a number or ``[lo, hi]``. Every atom that renders bare (no
``$name``/``$pose``, no place of the gesture space but ``@from`` and ``@via``) and has no
composition of its name gets a generated one (:func:`wrapper`; a hand-written one of the name
wins). A key value is a number, ``"$param"`` or ``"-$param"``; on the face's expression and the
eyes' iris and effect (:data:`pepin.expression.animation.EVENT_DOFS`) a name or ``"$name"``; a
key's ``pose`` is a pose's name, ``"$pose"`` or a reference into the gesture space
(``"@from"``, ``"@via"``, ``"@path:0"``; :mod:`pepin.expression.render` resolves them, as it
does a step's ``pose`` that starts with ``@``). A step's ``name`` of ``"$arg"`` is its mark's
argument (only beside arguments that choose); a composition's ``bare`` lets a mark name it
without one (its first plays). Every mistake is a ``ValueError``
naming where it is, at load: a book that loads plays.

The arm's ``rest`` pose is not the book's: it is config/arm.json's ``motion.rest_deg``, the pose
the board returns to by itself (:func:`with_rest`); nor are the face's drawing styles: the style
mark's arguments beside ``default`` are config/face.json's ``styles`` (:func:`with_styles`). The
idle layer never switches the face's style.

Owners after slice 0: C (``space``, poses, compositions, ``aliases``), A (``moments``,
``idle``); a new block is read here by its owner, with defaults, so an older book still loads.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast, get_args

from pepin.expression.animation import (
    BINDS,
    EVENT_DOFS,
    FACE_STYLE,
    HOLD,
    IDLE_STATES,
    MARK_ARG,
    MOMENT_DOFS,
    NUMBER,
    PARTS,
    PERIOD,
    PERSON,
    REPEATS,
    SEEN,
    Atom,
    Bind,
    Book,
    Breath,
    Composition,
    Idle,
    IdleState,
    Key,
    Oscillation,
    PartName,
    Range,
    Setting,
    Space,
    Step,
    Value,
    Variant,
)
from pepin.expression.markers import ANY, KEYS, Takes, Vocabulary
from pepin.expression.markers import ARG as ARG_ANY
from pepin.expression.speaker import AttentionPhase
from pepin.expression.trajectory import JOINTS
from pepin.face import EYE_EFFECTS
from pepin.head_link import EYE_RANGES
from pepin.overlay import read_json
from pepin.sounds import SOUND, SOUND_LINE

BOOK_FILE = "animation.json"
REST = "rest"  # the arm's base pose, from config/arm.json
DOFS: dict[PartName, tuple[str, ...]] = {
    "arm": JOINTS,
    "head": ("pan_deg", "tilt_deg"),
    "face": ("expression", FACE_STYLE),
    "body": ("yaw_deg", "x_m"),
    "eyes": (*EYE_RANGES, *MOMENT_DOFS, "iris", "effect"),
}
RESERVED = (PERIOD, REPEATS, HOLD)
STEP_ONLY = (REPEATS, HOLD)  # a step's reserved parameters, never an atom's (2026-10-14)
NAMED = ("name", "pose")  # the named parameters a key may refer to
ARG = "arg"  # a variant's need of its marker's argument (beside the parts' states)
ATTENTION = "attention"  # a variant's need of attention's phase (Scene.attention)
PHASES: tuple[str, ...] = get_args(AttentionPhase)  # attention's phases a variant may need
SPELLING = re.compile(r"^[^\s:{}]+(?::[^:{}]+)?$")  # an alias: a name, or name:arg
MAX_DEPTH = 3  # a composition, one nested in it and one nested in that
BARE_REFS = ("@from", "@via")  # the gesture space's places an atom finds with no marker
# The words a wrapper's line names its atom's degrees of freedom by (the brain's vocabulary)
SAID = {
    "pan_deg": "pan", "tilt_deg": "tilt", "yaw_deg": "turn", "x_m": "forward and back",
    "expression": "expression", "shoulder_pan": "shoulder pan", "shoulder_lift": "shoulder lift",
    "elbow_flex": "elbow", "wrist_flex": "wrist flex", "wrist_roll": "wrist roll",
    "gripper": "gripper",
}  # fmt: skip
CLAUSE = re.compile(r"[:;,(]")  # a note's first clause ends at the first of these
SHORT = {param: key for key, param in KEYS.items()}  # hold_s -> s, as a mark line spells it
COLOUR = re.compile(r"#[0-9A-Fa-f]{6}")  # an eyes' iris colour


def load_book(path: str | Path | None = None) -> Book:
    """config/animation.json (wherever this library runs, or ``path``), checked."""
    if path is None:
        from pepin.deployment import config_file

        path = config_file(BOOK_FILE)
    return book_from(read_json(path))


def book_from(data: Mapping[str, Any]) -> Book:
    """The book in ``data``; ``ValueError`` naming the first mistake."""
    poses = {
        name: {str(j): float(v) for j, v in joints.items() if j != "note"}
        for name, joints in data.get("poses", {}).items()
        if isinstance(joints, dict)
    }
    for name, joints in poses.items():
        if set(joints) != set(JOINTS):
            raise ValueError(f"poses.{name}: one angle for each of {', '.join(JOINTS)}")
    atoms = {
        name: _atom(name, body)
        for name, body in data.get("atoms", {}).items()
        if isinstance(body, dict)
    }
    written = {n: b for n, b in data.get("compositions", {}).items() if isinstance(b, dict)}
    wrapped = [a for n, a in atoms.items() if n not in written and bare(a)]
    known = set(written) | {a.name for a in wrapped}
    compositions = {name: _composition(name, body, atoms, known) for name, body in written.items()}
    compositions.update((a.name, wrapper(a)) for a in wrapped)
    for name in compositions:
        _nesting(compositions, atoms, (name,))
    idle = tuple(_idle(i, body, compositions, atoms) for i, body in enumerate(data.get("idle", [])))
    moments = {k: str(v) for k, v in data.get("moments", {}).items() if k != "note"}
    for moment, intent in moments.items():
        if intent not in compositions:
            raise ValueError(f"moments.{moment}: no composition {intent!r}")
    aliases = _aliases(data.get("aliases", {}), compositions)
    space = _space(data.get("space", {}), poses)
    breath = _breaths(data.get("breath", {}), compositions)
    return Book(atoms, compositions, poses, idle, moments, aliases, space, breath)


def with_rest(book: Book, rest: Mapping[str, float]) -> Book:
    """``book`` with the arm's ``rest`` pose (config/arm.json's) among its poses."""
    return replace(book, poses={**book.poses, REST: dict(rest)})


def with_styles(book: Book, styles: Iterable[str]) -> Book:
    """``book`` with the face's drawing ``styles`` (config/face.json's: :func:`style_names`)
    among the arguments of its style mark (``{style:rose}``: its step's ``$arg`` names the
    style); a book without that mark as it is."""
    composition = book.compositions.get(FACE_STYLE)
    if composition is None:
        return book
    args = (*composition.args, *(s for s in styles if s not in composition.args))
    styled = replace(composition, args=args)
    return replace(book, compositions={**book.compositions, FACE_STYLE: styled})


def vocabulary(book: Book, faces: frozenset[str]) -> Vocabulary:
    """What a marker may name: the book's callable compositions (the engine's own ensembles
    are not a marker's) with the arguments each takes (those that choose, and those that set
    its parameters: :func:`takes`), those a bare mark may name, and their other spellings, the
    face's expressions, and the sound tool's reserved mark (:data:`pepin.sounds.SOUND`:
    ``{sound:NAME}``, any short name; the engine logs it until the tool lands)."""
    names = intents(book)
    args = {name: book.compositions[name].args for name in names if book.compositions[name].args}
    taken = {name: takes(book, book.compositions[name]) for name in names}
    args[SOUND] = (ANY,)
    bare = frozenset(name for name in names if book.compositions[name].bare)
    return Vocabulary(frozenset((*names, SOUND)), faces, args, dict(book.aliases),
                      {name: t for name, t in taken.items() if t != Takes()}, bare)  # fmt: skip


def takes(book: Book, composition: Composition) -> Takes:
    """The arguments of ``composition``'s mark that set its steps' parameters: an auto-wrapper's
    numbers fill its atom's parameters in their order (one alone: its repeats) and ``key=value``
    sets any of them or a step's own; a hand-written one's words and number are its
    ``settings``, and ``key=value`` sets a parameter those name."""
    if composition.wraps is not None:
        params = tuple(book.atoms[composition.wraps].params)
        return Takes(frozenset(), len(params), frozenset({*params, *STEP_ONLY}))
    settings = composition.settings
    keys = frozenset(s.param for found in settings.values() for s in found)
    words = frozenset(w for w in settings if w != NUMBER)
    return Takes(words, 1 if NUMBER in settings else 0, keys)


def mark_lines(book: Book, styles: Sequence[str] = ()) -> tuple[str, ...]:
    """The brain's vocabulary of gestures, one line per composition a mark reaches (callable,
    not another mark's spelling: ``emphasis`` is the accent's; hello and bye are the brain's
    own since 2026-10-14), in the book's order: its name and arguments (``a|b`` one of them,
    ``NAME`` any short name, ``[...]`` optional, ``N`` a number, an auto-wrapper's parameters
    in their order), a dash, and at most six words: the note's first clause, or for an
    auto-wrapper its part and what it moves; the style mark's place takes ``styles`` when
    given (the face's: :func:`pepin.face.style_lines`, one a style); last, the sound tool's
    reserved ``sound:NAME`` (:data:`pepin.sounds.SOUND_LINE`)."""
    lines: list[str] = []
    for name in intents(book):
        if name in book.aliases:
            continue
        if name == FACE_STYLE and styles:
            lines += styles
            continue
        composition = book.compositions[name]
        lines.append(f"{name}{_spelled(book, composition)} — {brief(book, composition)}")
    lines.append(SOUND_LINE)
    return tuple(lines)


def brief(book: Book, composition: Composition) -> str:
    """``composition`` in a few words: an auto-wrapper's part and the degrees of freedom its
    atom's keys move (``head: pan, tilt``); else its note's first clause."""
    if composition.wraps is not None:
        atom = book.atoms[composition.wraps]
        moved = dict.fromkeys(d for k in atom.keys for d in k.values)
        return f"{atom.part}: {', '.join(SAID.get(d, d) for d in moved)}"
    return CLAUSE.split(composition.note, maxsplit=1)[0].strip()


def _spelled(book: Book, composition: Composition) -> str:
    """A mark line's arguments (:func:`mark_lines`); beside an argument that chooses, a number
    that sets one parameter is spelled by its name (one positional argument at most:
    ``glow:red|...[,s=N]``)."""
    choose = "|".join("NAME" if a == ANY else a for a in composition.args)
    if composition.wraps is not None:
        sets = "N|" + ",".join(book.atoms[composition.wraps].params)
    else:
        words = "|".join(w for w in composition.settings if w != NUMBER)
        targets = {s.param for s in composition.settings.get(NUMBER, ())}
        number = "N"
        if choose and len(targets) == 1:
            (param,) = targets
            number = f"{SHORT.get(param, param)}=N"
        sets = ",".join(x for x in (number if NUMBER in composition.settings else "", words) if x)
    if choose:
        spelled = f":{choose}" + (f"[,{sets}]" if sets else "")
        return f"[{spelled}]" if composition.bare else spelled
    return f"[:{sets}]" if sets else ""


def face_names(path: str | Path | None = None) -> frozenset[str]:
    """The expressions of config/face.json."""
    if path is None:
        from pepin.deployment import config_file

        path = config_file("face.json")
    data = read_json(path)
    return frozenset(str(row["name"]) for row in data.get("expressions", []))


def style_names(path: str | Path | None = None) -> tuple[str, ...]:
    """The face's drawing styles (config/face.json's ``styles``, in the firmware's order)."""
    from pepin.face import load_face_table

    return tuple(s.name for s in load_face_table(path).styles)


def intents(book: Book) -> tuple[str, ...]:
    """The compositions the brain may name, in the book's order (the callable ones)."""
    return tuple(name for name, c in book.compositions.items() if c.callable)


def span(
    book: Book,
    intent: str,
    part: PartName,
    stretch: float = 1.0,
    fits: Callable[[Variant], bool] | None = None,
) -> tuple[float, float]:
    """How long before its sync point ``intent``'s steps on ``part`` may start moving it and how
    long after it they may end, at the widest their ranges allow over every variant (over those
    ``fits`` admits when given: one style of the dance with the arm where it is; a nested
    composition's steps from its step's ``at_s``), each step played ``stretch`` times slower
    round its stroke (the engine's speed scales); (0, 0) when no step moves the part (or the
    book lacks the intent)."""
    composition = book.compositions.get(intent)
    if composition is None:
        return 0.0, 0.0
    if fits is not None:
        composition = replace(composition, variants=tuple(filter(fits, composition.variants)))
    return _span(book, composition, part, stretch) or (0.0, 0.0)


def _span(
    book: Book, composition: Composition, part: PartName, stretch: float
) -> tuple[float, float] | None:
    """:func:`span` of one composition, None when none of its steps moves ``part``."""
    found: tuple[float, float] | None = None
    for variant in composition.variants:
        for step in variant.steps:
            if step.composition is not None:
                inner = _span(book, book.compositions[step.composition], part, stretch)
                if inner is None:
                    continue
                edges = (inner[0] - step.at_s.lo, step.at_s.hi + inner[1])
            else:
                atom = book.atoms[step.atom]
                if atom.part != part:
                    continue
                ranges = {**atom.params, **step.params}
                period = ranges[PERIOD].hi
                repeats = ranges[REPEATS].hi if REPEATS in ranges else 1.0
                hold = ranges[HOLD].hi if HOLD in ranges else 0.0
                late = (repeats - atom.stroke) * period + hold
                edges = (stretch * atom.stroke * period - step.at_s.lo,
                         step.at_s.hi + stretch * late)  # fmt: skip
            before, after = found or (0.0, 0.0)
            found = max(before, edges[0]), max(after, edges[1])
    return found


def step_parts(book: Book, step: Step) -> frozenset[PartName]:
    """The parts ``step`` may move: its atom's, or every part its nested composition's
    variants move."""
    if step.composition is None:
        return frozenset({book.atoms[step.atom].part})
    return composition_parts(book.compositions, book.atoms, step.composition)


def sure_parts(book: Book, step: Step) -> frozenset[PartName]:
    """The parts ``step`` moves whenever it plays by its odds alone (p 1): its atom's, or those
    every variant of its nested composition moves for sure (the floors aside, unless it has
    only floors)."""
    if step.p < 1.0:
        return frozenset()
    if step.composition is None:
        return frozenset({book.atoms[step.atom].part})
    variants = book.compositions[step.composition].variants
    drawn = [v for v in variants if not v.floor] or list(variants)
    each = [frozenset().union(*(sure_parts(book, s) for s in v.steps)) for v in drawn]
    return frozenset.intersection(*each) if each else frozenset()


def composition_parts(
    compositions: Mapping[str, Composition], atoms: Mapping[str, Atom], name: str
) -> frozenset[PartName]:
    """Every part a composition's variants may move, its nested ones' included."""
    parts: set[PartName] = set()
    for variant in compositions[name].variants:
        for step in variant.steps:
            if step.composition is None:
                parts.add(atoms[step.atom].part)
            else:
                parts |= composition_parts(compositions, atoms, step.composition)
    return frozenset(parts)


def loops(book: Book, name: str) -> bool:
    """Whether a step of composition ``name`` loops (its clip repeats until withdrawn)."""
    composition = book.compositions.get(name)
    return composition is not None and any(s.loop for v in composition.variants for s in v.steps)


def bare(atom: Atom) -> bool:
    """Whether ``atom`` renders with nothing chosen for it: no named parameter (a face's or a
    pose's name) and no place of the gesture space a marker's argument finds (``@from`` and
    ``@via`` aside)."""
    if _named_params(atom):
        return False
    return all(k.pose is None or k.pose in BARE_REFS for k in atom.keys)


def wrapper(atom: Atom) -> Composition:
    """``atom``'s auto-wrapper: one variant of one step, its own ranges (repeats 1, hold 0),
    callable as the atom is; an arm atom that moves a joint other than the gripper needs the
    arm up (it plays on a held pose, never from rest: the folded arm stands by the speaker
    box), and so does an absolute one (``gripper_snap``: from rest it shuts a jaw that stands
    shut); any other needs nothing."""
    moved = {d for k in atom.keys for d in k.values}
    up = atom.part == "arm" and (bool(moved - {"gripper"}) or atom.mode == "absolute")
    needs = {"arm": "up"} if up else {}
    variant = Variant((Step(atom.name),), needs=needs)
    return Composition(atom.name, (variant,), callable=atom.callable, wraps=atom.name)


def _range(where: str, value: Any) -> Range:
    """A number or ``[lo, hi]`` as a range."""
    if isinstance(value, bool):
        raise ValueError(f"{where}: a number or [lo, hi], not {value!r}")
    if isinstance(value, int | float):
        return Range(float(value), float(value))
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in value)
        and value[0] <= value[1]
    ):
        return Range(float(value[0]), float(value[1]))
    raise ValueError(f"{where}: a number or [lo, hi] with lo <= hi, not {value!r}")


def _atom(name: str, body: Mapping[str, Any]) -> Atom:
    where = f"atoms.{name}"
    part = body.get("part")
    if part not in PARTS:
        raise ValueError(f"{where}.part: one of {', '.join(PARTS)}")
    mode = body.get("mode", "additive")
    if mode not in ("absolute", "additive"):
        raise ValueError(f"{where}.mode: absolute or additive")
    params = {k: _range(f"{where}.params.{k}", v) for k, v in body.get("params", {}).items()}
    if PERIOD not in params or params[PERIOD].lo <= 0:
        raise ValueError(f"{where}.params.{PERIOD}: required, above 0")
    for reserved in STEP_ONLY:
        if reserved in params:
            raise ValueError(f"{where}.params.{reserved}: a step's, not an atom's (2026-10-14)")
    if "loop" in body:
        raise ValueError(f"{where}.loop: a step's, not an atom's (2026-10-14)")
    keys: list[Key] = []
    for i, raw in enumerate(body.get("keys", [])):
        at = float(raw.get("at", -1))
        if not 0.0 <= at <= 1.0 or (keys and at < keys[-1].at):
            raise ValueError(f"{where}.keys[{i}].at: 0..1, in order")
        values: dict[str, Value] = {}
        for dof, value in raw.items():
            if dof in ("at", "pose"):
                continue
            if dof not in DOFS[cast(PartName, part)]:
                raise ValueError(f"{where}.keys[{i}].{dof}: not a degree of freedom of the {part}")
            values[dof] = _value(f"{where}.keys[{i}].{dof}", value, params, dof in EVENT_DOFS)
            _eye_name(f"{where}.keys[{i}].{dof}", dof, values[dof])
        pose = raw.get("pose")
        if pose is not None and part != "arm":
            raise ValueError(f"{where}.keys[{i}].pose: only the arm has poses")
        keys.append(Key(at, values, None if pose is None else str(pose)))
    if not keys:
        raise ValueError(f"{where}.keys: at least one")
    mirror = tuple(str(d) for d in body.get("mirror", []))
    if not set(mirror) <= set(DOFS[cast(PartName, part)]):
        raise ValueError(f"{where}.mirror: degrees of freedom of the {part}")
    return Atom(
        name,
        cast(PartName, part),
        mode,
        tuple(keys),
        params,
        float(body.get("stroke", 0.5)),
        str(body.get("note", "")),
        mirror,
        bool(body.get("callable", True)),
    )


def _eye_name(where: str, dof: str, value: Value) -> None:
    """An eyes' named value checked (a ``$`` parameter aside): an ``iris`` colour is ``#RRGGBB``,
    an ``effect`` one of :data:`pepin.face.EYE_EFFECTS`."""
    if not isinstance(value, str) or value.startswith(("$", "-$")):
        return
    if dof == "iris" and not COLOUR.fullmatch(value):
        raise ValueError(f"{where}: an iris colour is #RRGGBB, not {value!r}")
    if dof == "effect" and value not in EYE_EFFECTS:
        raise ValueError(f"{where}: one of {', '.join(EYE_EFFECTS)}, not {value!r}")


def _value(where: str, value: Any, params: Mapping[str, Range], named: bool) -> Value:
    """A key's value, its parameter checked: a number, ``"$p"``/``"-$p"``, or a face's name."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(f"{where}: a finite number")
        return float(value)
    if isinstance(value, str):
        unsigned = value.removeprefix("-")
        if unsigned.startswith("$"):
            param = unsigned[1:]
            if param in params or (named and param in NAMED and unsigned == value):
                return value
        elif named:
            return value  # a face's name, checked against config/face.json by the face itself
    raise ValueError(
        f"{where}: a number, '$param' or '-$param' of the atom's params, not {value!r}"
    )


def _composition(
    name: str, body: Mapping[str, Any], atoms: Mapping[str, Atom], known: set[str]
) -> Composition:
    where = f"compositions.{name}"
    callable_ = bool(body.get("callable", True))
    variants = []
    for v, raw in enumerate(body.get("variants", [])):
        steps = []
        for s, step in enumerate(raw.get("steps", [])):
            at = f"{where}.variants[{v}].steps[{s}]"
            if "composition" in step:
                steps.append(_nested(at, step, known))
                continue
            atom = atoms.get(str(step.get("atom")))
            if atom is None:
                raise ValueError(f"{at}.atom: no atom {step.get('atom')!r}")
            if not atom.callable and callable_:
                raise ValueError(f"{at}.atom: {atom.name} is the idle layer's, not a callable "
                                 "composition's")  # fmt: skip
            loop = step.get("loop", False)
            if not isinstance(loop, bool) or (loop and callable_):
                raise ValueError(f"{at}.loop: true only in the engine's own (callable false)")
            params: dict[str, Range] = {}
            choices: dict[str, tuple[str, ...]] = {}
            for key, value in step.get("params", {}).items():
                if isinstance(value, str) or (
                    isinstance(value, list) and value and all(isinstance(x, str) for x in value)
                ):
                    if key not in NAMED:
                        raise ValueError(f"{at}.params.{key}: a name is given only to name or pose")
                    choices[key] = (value,) if isinstance(value, str) else tuple(value)
                    continue
                if key not in atom.params and key not in RESERVED:
                    raise ValueError(
                        f"{at}.params.{key}: {atom.name} takes {', '.join(atom.params)}"
                    )
                params[key] = _range(f"{at}.params.{key}", value)
            for needed in _named_params(atom):
                if needed not in choices:
                    raise ValueError(f"{at}.params.{needed}: {atom.name} needs one")
            for dof in {d for k in atom.keys for d, v in k.values.items() if v == "$name"}:
                for chosen in choices.get("name", ()):
                    _eye_name(f"{at}.params.name", dof, chosen)
            p = float(step.get("p", 1.0))
            if not 0.0 < p <= 1.0:
                raise ValueError(f"{at}.p: in (0, 1]")
            steps.append(Step(atom.name, _range(f"{at}.at_s", step.get("at_s", 0.0)), p, params,
                              choices, loop=loop))  # fmt: skip
        if not steps:
            raise ValueError(f"{where}.variants[{v}].steps: at least one")
        weight = float(raw.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"{where}.variants[{v}].weight: above 0")
        needs = raw.get("needs", {})
        if not isinstance(needs, dict) or not set(needs) <= {*PARTS, ARG, ATTENTION, PERSON}:
            raise ValueError(
                f"{where}.variants[{v}].needs: part -> state, parts of {PARTS}, {ARG}, "
                f"{ATTENTION} or {PERSON}"
            )
        needs = {str(k): str(x) for k, x in needs.items()}
        if needs.get(ATTENTION, PHASES[0]) not in PHASES:
            raise ValueError(f"{where}.variants[{v}].needs.{ATTENTION}: one of {PHASES}")
        if needs.get(PERSON, SEEN) != SEEN:
            raise ValueError(f"{where}.variants[{v}].needs.{PERSON}: {SEEN}")
        variants.append(Variant(tuple(steps), weight, needs, bool(raw.get("floor", False))))
    if not variants:
        raise ValueError(f"{where}.variants: at least one")
    bind = body.get("bind", "utterance")
    if bind not in BINDS:
        raise ValueError(f"{where}.bind: one of {', '.join(BINDS)}")
    family = body.get("family")
    args, settings = _args(where, body.get("args", []), variants, atoms)
    bare_ = bool(body.get("bare", False))
    if bare_ and not args:
        raise ValueError(f"{where}.bare: only beside arguments that choose (the first plays)")
    named_arg = any(MARK_ARG in s.choices.get("name", ()) for v in variants for s in v.steps)
    if named_arg and not args:
        raise ValueError(f"{where}: a step's name {MARK_ARG} needs arguments that choose")
    style_args = body.get("style_args", {})
    if not isinstance(style_args, dict):
        raise ValueError(f"{where}.style_args: arg -> the style words that choose it")
    written = {v.needs.get(ARG) for v in variants} | set(args)  # what a variant is written for
    for arg, words in style_args.items():
        if arg not in written or not isinstance(words, list) or not words:
            raise ValueError(f"{where}.style_args.{arg}: a variant's argument, a list of words")
    return Composition(
        name,
        tuple(variants),
        float(body.get("cooldown_s", 0.0)),
        str(body.get("note", "")),
        cast(Bind, bind),
        args,
        None if family is None else str(family),
        bool(body.get("long", False)),
        callable_,
        {str(a): tuple(str(w).lower() for w in ws) for a, ws in style_args.items()},
        settings,
        bare=bare_,
    )


def _nested(at: str, step: Mapping[str, Any], known: set[str]) -> Step:
    """A step that nests a composition: its name, ``at_s`` and ``p`` (nothing else)."""
    name = str(step.get("composition"))
    if name not in known:
        raise ValueError(f"{at}.composition: no composition {name!r}")
    extra = set(step) - {"composition", "at_s", "p"}
    if extra:
        raise ValueError(f"{at}: a nested composition's step takes at_s and p, not "
                         f"{', '.join(sorted(extra))}")  # fmt: skip
    p = float(step.get("p", 1.0))
    if not 0.0 < p <= 1.0:
        raise ValueError(f"{at}.p: in (0, 1]")
    return Step("", _range(f"{at}.at_s", step.get("at_s", 0.0)), p, composition=name)


def _nesting(
    compositions: Mapping[str, Composition], atoms: Mapping[str, Atom], chain: tuple[str, ...]
) -> None:
    """The compositions nested under ``chain``'s last: no cycle, at most :data:`MAX_DEPTH`
    deep, and none playing the idle layer's atoms under a callable one (the chain's first)."""
    root = compositions[chain[0]]
    for variant in compositions[chain[-1]].variants:
        for step in variant.steps:
            if step.composition is None:
                if len(chain) > 1 and root.callable and not atoms[step.atom].callable:
                    raise ValueError(f"compositions.{chain[0]}: {' > '.join(chain)} plays "
                                     f"{step.atom}, the idle layer's")  # fmt: skip
                continue
            nested = (*chain, step.composition)
            if step.composition in chain:
                raise ValueError(f"compositions.{chain[0]}: a cycle: {' > '.join(nested)}")
            if len(nested) > MAX_DEPTH:
                raise ValueError(f"compositions.{chain[0]}: nested deeper than {MAX_DEPTH}: "
                                 f"{' > '.join(nested)}")  # fmt: skip
            _nesting(compositions, atoms, nested)


def _args(
    where: str, raw: Any, variants: list[Variant], atoms: Mapping[str, Atom]
) -> tuple[tuple[str, ...], dict[str, tuple[Setting, ...]]]:
    """``args``: the arguments that choose (a list, or a map's keys to null) and the settings
    (a map: ``number`` -> the parameters a mark's number sets, a word -> parameter -> range);
    each parameter ``param`` (every step whose atom takes it) or ``atom.param``."""
    if isinstance(raw, list):
        return tuple(str(a) for a in raw), {}
    if not isinstance(raw, dict):
        raise ValueError(f"{where}.args: a list, or argument -> null | parameters")
    used = {s.atom for v in variants for s in v.steps if s.composition is None}
    choose: list[str] = []
    settings: dict[str, tuple[Setting, ...]] = {}
    for arg, value in raw.items():
        at = f"{where}.args.{arg}"
        if value is None:
            choose.append(str(arg))
        elif arg == NUMBER:
            targets = [value] if isinstance(value, str) else value
            if not isinstance(targets, list) or not targets:
                raise ValueError(f"{at}: a parameter or a list of them")
            settings[NUMBER] = tuple(_setting(at, str(t), None, used, atoms) for t in targets)
        elif isinstance(value, dict) and value and arg != ANY:
            settings[str(arg)] = tuple(
                _setting(f"{at}.{t}", str(t), _range(f"{at}.{t}", r), used, atoms)
                for t, r in value.items()
            )
        else:
            raise ValueError(f"{at}: null (it chooses) or parameter -> a number or [lo, hi]")
    return tuple(choose), settings


def _setting(
    at: str, target: str, value: Range | None, used: set[str], atoms: Mapping[str, Atom]
) -> Setting:
    """``param`` or ``atom.param`` as a setting, checked against the composition's atoms."""
    atom, dot, param = target.rpartition(".")
    if dot and atom not in used:
        raise ValueError(f"{at}: no step of the composition plays {atom!r}")
    takers = [atom] if dot else sorted(used)
    if param not in RESERVED and not any(param in atoms[a].params for a in takers):
        raise ValueError(f"{at}: no step's atom takes {param!r}")
    return Setting(param, atom if dot else None, value)


def _breaths(raw: Any, compositions: Mapping[str, Composition]) -> dict[PartName, Breath]:
    """``breath``: a part -> its layer (the arm's first; a part absent keeps its older way)."""
    if not isinstance(raw, dict):
        raise ValueError("breath: part -> its layer")
    out: dict[PartName, Breath] = {}
    for part, body in raw.items():
        if part == "note":
            continue
        where = f"breath.{part}"
        if part not in PARTS or not isinstance(body, dict):
            raise ValueError(f"{where}: one of {', '.join(PARTS)}, an object")
        dofs = DOFS[cast(PartName, part)]
        joints: dict[str, Oscillation] = {}
        for dof, osc in body.get("joints", {}).items():
            if dof not in dofs or not isinstance(osc, dict):
                raise ValueError(f"{where}.joints.{dof}: a degree of freedom of the {part}")
            amp = float(osc.get("amp_deg", 0.0))
            period = float(osc.get("period_s", 0.0))
            phase = float(osc.get("phase", 0.0))
            if amp < 0.0 or period <= 0.0 or not 0.0 <= phase < 1.0:
                raise ValueError(f"{where}.joints.{dof}: amp_deg >= 0, period_s > 0, phase 0..1")
            joints[str(dof)] = Oscillation(amp, period, phase)
        if not joints:
            raise ValueError(f"{where}.joints: at least one")
        gains = body.get("accent_gain", {})
        if not isinstance(gains, dict) or not set(gains) <= set(compositions):
            raise ValueError(f"{where}.accent_gain: a composition -> its gain")
        when = body.get("when", "up")
        if when != "up":
            raise ValueError(f"{where}.when: up (never at rest)")
        d = Breath({})
        out[cast(PartName, part)] = Breath(
            joints,
            "up",
            float(body.get("blend_s", d.blend_s)),
            {str(k): float(v) for k, v in gains.items()},
            float(body.get("accent_decay_s", d.accent_decay_s)),
            float(body.get("clip_s", d.clip_s)),
        )
    return out


def _named_params(atom: Atom) -> set[str]:
    """The named parameters an atom's keys refer to (``$name``, ``$pose``)."""
    needed = set()
    for key in atom.keys:
        if key.pose is not None and key.pose.startswith("$"):
            needed.add(key.pose[1:])
        for value in key.values.values():
            if isinstance(value, str) and value.startswith("$") and value[1:] in NAMED:
                needed.add(value[1:])
    return needed


def _idle(
    i: int,
    body: Mapping[str, Any],
    compositions: Mapping[str, Composition],
    atoms: Mapping[str, Atom],
) -> Idle:
    where = f"idle[{i}]"
    part = body.get("part")
    if part not in PARTS:
        raise ValueError(f"{where}.part: one of {', '.join(PARTS)}")
    names = tuple(str(a) for a in body.get("compositions", []))
    for name in names:
        if name not in compositions or composition_parts(compositions, atoms, name) != {part}:
            raise ValueError(f"{where}.compositions: {name!r} is no composition of the {part} "
                             "alone")  # fmt: skip
        if _styles_face(compositions, atoms, name):
            raise ValueError(f"{where}.compositions: {name} switches the face's style, never "
                             "the idle layer's")  # fmt: skip
    if not names:
        raise ValueError(f"{where}.compositions: at least one")
    when = body.get("when", "conversation")
    if when not in ("conversation", "always"):
        raise ValueError(f"{where}.when: conversation or always")
    unless = body.get("unless", list(IDLE_STATES))
    unless = [unless] if isinstance(unless, str) else unless
    if not isinstance(unless, list) or not set(unless) <= set(IDLE_STATES):
        raise ValueError(f"{where}.unless: some of {', '.join(IDLE_STATES)}")
    every = _range(f"{where}.every_s", body.get("every_s"))
    states = tuple(cast(IdleState, u) for u in unless)
    after = body.get("after_speech_s")
    if after is not None and (isinstance(after, bool) or not isinstance(after, (int, float))
                              or after < 0):  # fmt: skip
        raise ValueError(f"{where}.after_speech_s: seconds >= 0, or absent")
    return Idle(cast(PartName, part), names, every, when, states,
                None if after is None else float(after))  # fmt: skip


def _styles_face(
    compositions: Mapping[str, Composition], atoms: Mapping[str, Atom], name: str
) -> bool:
    """Whether a step of composition ``name`` (a nested one's included) switches the face's
    drawing style."""
    for variant in compositions[name].variants:
        for step in variant.steps:
            if step.composition is not None:
                if _styles_face(compositions, atoms, step.composition):
                    return True
            elif any(FACE_STYLE in key.values for key in atoms[step.atom].keys):
                return True
    return False


def _aliases(raw: Any, compositions: Mapping[str, Composition]) -> dict[str, str]:
    """``aliases``: a spelling (lower case, ``name`` or ``name:arg``) -> a composition, with an
    argument it takes when it takes any."""
    if not isinstance(raw, dict):
        raise ValueError("aliases: spelling -> name or name:arg")
    out: dict[str, str] = {}
    for spelling, target in raw.items():
        if spelling == "note":
            continue
        where = f"aliases.{spelling}"
        spelled = SPELLING.match(spelling) and spelling == spelling.lower()
        if not isinstance(target, str) or not spelled:
            raise ValueError(f"{where}: a lower-case name or name:arg -> a name or name:arg")
        name, _, arg = target.partition(":")
        composition = compositions.get(name)
        if composition is None:
            raise ValueError(f"{where}: no composition {name!r}")
        free = ANY in composition.args and ARG_ANY.match(arg) is not None
        if arg and arg not in composition.args and not free:
            raise ValueError(f"{where}: {name} takes no argument {arg!r}")
        out[spelling] = target
    return out


def _space(raw: Any, poses: Mapping[str, Mapping[str, float]]) -> Space:
    """``space``, every pose it names checked (``rest`` is config/arm.json's, never named)."""
    if not isinstance(raw, dict):
        raise ValueError("space: an object")

    def pose(where: str, name: Any) -> str:
        if not isinstance(name, str) or name not in poses:
            raise ValueError(f"space.{where}: no pose {name!r}")
        return name

    def pair(where: str, value: Any) -> tuple[float, float]:
        rng = _range(f"space.{where}", value)
        return rng.lo, rng.hi

    def two(where: str, value: Any, what: str) -> tuple[float, float]:
        if not (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(v, int | float) and not isinstance(v, bool) for v in value)
        ):
            raise ValueError(f"space.{where}: {what}")
        return float(value[0]), float(value[1])

    d = Space()
    if not raw:
        return d
    hand = raw.get("pointing", {})
    paths = {
        str(k): tuple(pose(f"paths.{k}", p) for p in v)
        for k, v in raw.get("paths", {}).items()
        if k != "note"
    }
    head = {
        str(k): two(f"head.{k}", v, "[pan, tilt] in degrees")
        for k, v in raw.get("head", {}).items()
        if k != "note"
    }
    via_pan = raw.get("via_pan_deg")
    home_after = {
        pose("home_after", k): pose(f"home_after.{k}", v)
        for k, v in raw.get("home_after", {}).items()
        if k != "note"
    }
    head_pan_min = {
        pose("head_pan_min_deg", k): float(v)
        for k, v in raw.get("head_pan_min_deg", {}).items()
        if k != "note"
    }
    return Space(
        home=pose("home", raw.get("home", d.home)),
        via_pan_deg=None if via_pan is None else float(via_pan),
        via=tuple(pose("via", p) for p in raw.get("via", [])),
        home_after=home_after,
        head_pan_min_deg=head_pan_min,
        loci=tuple(pose("loci", p) for p in raw.get("loci", [])),
        paths=paths,
        head=head,
        open_hand=pose("pointing.open", hand["open"]) if "open" in hand else None,
        shut_hand=pose("pointing.shut", hand["shut"]) if "shut" in hand else None,
        pan_deg=pair("pointing.pan_deg", hand.get("pan_deg", [0.0, 0.0])),
        arm_axis_m=two("pointing.arm_axis_m", hand.get("arm_axis_m", [0.0, 0.0]), "[x, y] m"),
        neck_m=two("pointing.neck_m", hand.get("neck_m", [0.0, 0.0]), "[x, y] m"),
        person_m=float(hand.get("person_m", d.person_m)),
        again=tuple(str(x) for x in raw.get("again", [])),
    )
