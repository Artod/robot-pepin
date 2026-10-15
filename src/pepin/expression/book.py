"""config/animation.json read into a :class:`pepin.expression.animation.Book`, checked whole.

The file: ``atoms`` (name -> part, mode, stroke, params, keys, mirror, callable, loop, note),
``compositions`` (name -> cooldown_s, bind, args, family, long, callable, variants of steps
with their weight, needs and floor, note), ``poses`` (name -> joint ->
degrees and an optional note; taught by hand with ``scripts/gesture.py teach`` or authored
from the URDF), ``idle`` (the idle layers),
``moments`` (a moment of the robot -> a composition), ``aliases`` (another spelling of a
marker -> its canonical ``name`` or ``name:arg``) and ``space`` (the gesture space:
:class:`pepin.expression.animation.Space`). A parameter is a number or ``[lo, hi]``;
in a step, a string or a list of strings is a named parameter (a face's or a pose's name, one
drawn). A key value is a number, ``"$param"`` or ``"-$param"``; on the face a name or
``"$name"``; a key's ``pose`` is a pose's name, ``"$pose"`` or a reference into the gesture space
(``"@from"``, ``"@via"``, ``"@path:0"``; :mod:`pepin.expression.render` resolves them, as it
does a step's ``pose`` that starts with ``@``). Every mistake is a ``ValueError``
naming where it is, at load: a book that loads plays.

The arm's ``rest`` pose is not the book's: it is config/arm.json's ``motion.rest_deg``, the pose
the board returns to by itself (:func:`with_rest`).

Owners after slice 0: C (``space``, poses, compositions, ``aliases``), A (``moments``,
``idle``); a new block is read here by its owner, with defaults, so an older book still loads.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from pepin.expression.animation import (
    BINDS,
    HOLD,
    IDLE_STATES,
    PARTS,
    PERIOD,
    REPEATS,
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
    Space,
    Step,
    Value,
    Variant,
)
from pepin.expression.markers import ANY, Vocabulary
from pepin.expression.markers import ARG as ARG_ANY
from pepin.expression.trajectory import JOINTS

BOOK_FILE = "animation.json"
REST = "rest"  # the arm's base pose, from config/arm.json
DOFS: dict[PartName, tuple[str, ...]] = {
    "arm": JOINTS,
    "head": ("pan_deg", "tilt_deg"),
    "face": ("expression",),
    "body": ("yaw_deg", "x_m"),
}
RESERVED = (PERIOD, REPEATS, HOLD)
NAMED = ("name", "pose")  # the named parameters a key may refer to
ARG = "arg"  # a variant's need of its marker's argument (beside the parts' states)
SPELLING = re.compile(r"^[^\s:{}]+(?::[^:{}]+)?$")  # an alias: a name, or name:arg


def load_book(path: str | Path | None = None) -> Book:
    """config/animation.json (wherever this library runs, or ``path``), checked."""
    if path is None:
        from pepin.deployment import config_file

        path = config_file(BOOK_FILE)
    return book_from(json.loads(Path(path).read_text()))


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
    compositions = {
        name: _composition(name, body, atoms)
        for name, body in data.get("compositions", {}).items()
        if isinstance(body, dict)
    }
    idle = tuple(_idle(i, body, atoms) for i, body in enumerate(data.get("idle", [])))
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


def vocabulary(book: Book, faces: frozenset[str]) -> Vocabulary:
    """What a marker may name: the book's callable compositions (the engine's own ensembles
    are not a marker's) with the arguments each takes and their other spellings, and the
    face's expressions."""
    names = intents(book)
    args = {name: book.compositions[name].args for name in names if book.compositions[name].args}
    return Vocabulary(frozenset(names), faces, args, dict(book.aliases))


def face_names(path: str | Path | None = None) -> frozenset[str]:
    """The expressions of config/face.json."""
    if path is None:
        from pepin.deployment import config_file

        path = config_file("face.json")
    data = json.loads(Path(path).read_text())
    return frozenset(str(row["name"]) for row in data.get("expressions", []))


def intents(book: Book) -> tuple[str, ...]:
    """The compositions the brain may name, in the book's order (the callable ones)."""
    return tuple(name for name, c in book.compositions.items() if c.callable)


def span(book: Book, intent: str, part: PartName, stretch: float = 1.0) -> tuple[float, float]:
    """How long before its sync point ``intent``'s steps on ``part`` may start moving it and how
    long after it they may end, at the widest their ranges allow over every variant, each step
    played ``stretch`` times slower round its stroke (the engine's speed scales); (0, 0) when no
    step moves the part (or the book lacks the intent)."""
    composition = book.compositions.get(intent)
    if composition is None:
        return 0.0, 0.0
    before, after = 0.0, 0.0
    for variant in composition.variants:
        for step in variant.steps:
            atom = book.atoms[step.atom]
            if atom.part != part:
                continue
            ranges = {**atom.params, **step.params}
            period = ranges[PERIOD].hi
            repeats = ranges[REPEATS].hi if REPEATS in ranges else 1.0
            hold = ranges[HOLD].hi if HOLD in ranges else 0.0
            before = max(before, stretch * atom.stroke * period - step.at_s.lo)
            after = max(after, step.at_s.hi + stretch * ((repeats - atom.stroke) * period + hold))
    return before, after


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
            values[dof] = _value(f"{where}.keys[{i}].{dof}", value, params, dof == "expression")
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
        bool(body.get("loop", False)),
    )


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


def _composition(name: str, body: Mapping[str, Any], atoms: Mapping[str, Atom]) -> Composition:
    where = f"compositions.{name}"
    variants = []
    for v, raw in enumerate(body.get("variants", [])):
        steps = []
        for s, step in enumerate(raw.get("steps", [])):
            at = f"{where}.variants[{v}].steps[{s}]"
            atom = atoms.get(str(step.get("atom")))
            if atom is None:
                raise ValueError(f"{at}.atom: no atom {step.get('atom')!r}")
            if not atom.callable:
                raise ValueError(f"{at}.atom: {atom.name} is the idle layer's, not a composition's")
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
            p = float(step.get("p", 1.0))
            if not 0.0 < p <= 1.0:
                raise ValueError(f"{at}.p: in (0, 1]")
            steps.append(Step(atom.name, _range(f"{at}.at_s", step.get("at_s", 0.0)), p, params,
                              choices))  # fmt: skip
        if not steps:
            raise ValueError(f"{where}.variants[{v}].steps: at least one")
        weight = float(raw.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"{where}.variants[{v}].weight: above 0")
        needs = raw.get("needs", {})
        if not isinstance(needs, dict) or not set(needs) <= {*PARTS, ARG}:
            raise ValueError(
                f"{where}.variants[{v}].needs: part -> state, parts of {PARTS}, or {ARG}"
            )
        needs = {str(k): str(x) for k, x in needs.items()}
        variants.append(Variant(tuple(steps), weight, needs, bool(raw.get("floor", False))))
    if not variants:
        raise ValueError(f"{where}.variants: at least one")
    bind = body.get("bind", "utterance")
    if bind not in BINDS:
        raise ValueError(f"{where}.bind: one of {', '.join(BINDS)}")
    family = body.get("family")
    args = tuple(str(a) for a in body.get("args", []))
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
        bool(body.get("callable", True)),
        {str(a): tuple(str(w).lower() for w in ws) for a, ws in style_args.items()},
    )


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


def _idle(i: int, body: Mapping[str, Any], atoms: Mapping[str, Atom]) -> Idle:
    where = f"idle[{i}]"
    part = body.get("part")
    if part not in PARTS:
        raise ValueError(f"{where}.part: one of {', '.join(PARTS)}")
    names = tuple(str(a) for a in body.get("atoms", []))
    for name in names:
        atom = atoms.get(name)
        if atom is None or atom.part != part or _named_params(atom):
            raise ValueError(f"{where}.atoms: {name!r} is no {part} atom without names")
    if not names:
        raise ValueError(f"{where}.atoms: at least one")
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
    return Space(
        home=pose("home", raw.get("home", d.home)),
        via_pan_deg=None if via_pan is None else float(via_pan),
        via=tuple(pose("via", p) for p in raw.get("via", [])),
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
