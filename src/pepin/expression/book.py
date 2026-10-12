"""config/animation.json read into a :class:`pepin.expression.animation.Book`, checked whole.

The file: ``atoms`` (name -> part, mode, stroke, params, keys, mirror, callable, loop, note),
``compositions`` (name -> cooldown_s, variants of steps, note), ``poses`` (name -> joint ->
degrees, taught by hand with ``scripts/gesture.py teach``), ``idle`` (the idle layers) and
``moments`` (a moment of the robot -> a composition). A parameter is a number or ``[lo, hi]``;
in a step, a string or a list of strings is a named parameter (a face's or a pose's name, one
drawn). A key value is a number, ``"$param"`` or ``"-$param"``; on the face a name or
``"$name"``; a key's ``pose`` is a pose's name or ``"$pose"``. Every mistake is a ``ValueError``
naming where it is, at load: a book that loads plays.

The arm's ``rest`` pose is not the book's: it is config/arm.json's ``motion.rest_deg``, the pose
the board returns to by itself (:func:`with_rest`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from pepin.expression.animation import (
    HOLD,
    PARTS,
    PERIOD,
    REPEATS,
    Atom,
    Book,
    Composition,
    Idle,
    Key,
    PartName,
    Range,
    Step,
    Value,
    Variant,
)
from pepin.expression.markers import Vocabulary
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


def load_book(path: str | Path | None = None) -> Book:
    """config/animation.json (wherever this library runs, or ``path``), checked."""
    if path is None:
        from pepin.deployment import config_file

        path = config_file(BOOK_FILE)
    return book_from(json.loads(Path(path).read_text()))


def book_from(data: Mapping[str, Any]) -> Book:
    """The book in ``data``; ``ValueError`` naming the first mistake."""
    poses = {
        name: {str(j): float(v) for j, v in joints.items()}
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
    return Book(atoms, compositions, poses, idle, moments)


def with_rest(book: Book, rest: Mapping[str, float]) -> Book:
    """``book`` with the arm's ``rest`` pose (config/arm.json's) among its poses."""
    return Book(book.atoms, book.compositions, {**book.poses, REST: dict(rest)}, book.idle,
                book.moments)  # fmt: skip


def vocabulary(book: Book, faces: frozenset[str]) -> Vocabulary:
    """What a marker may name: the book's compositions, and the face's expressions."""
    return Vocabulary(frozenset(book.compositions), faces)


def face_names(path: str | Path | None = None) -> frozenset[str]:
    """The expressions of config/face.json."""
    if path is None:
        from pepin.deployment import config_file

        path = config_file("face.json")
    data = json.loads(Path(path).read_text())
    return frozenset(str(row["name"]) for row in data.get("expressions", []))


def intents(book: Book) -> tuple[str, ...]:
    """The compositions the brain may name, in the book's order."""
    return tuple(book.compositions)


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
        variants.append(Variant(tuple(steps), weight))
    if not variants:
        raise ValueError(f"{where}.variants: at least one")
    return Composition(name, tuple(variants), float(body.get("cooldown_s", 0.0)),
                       str(body.get("note", "")))  # fmt: skip


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
    return Idle(cast(PartName, part), names, _range(f"{where}.every_s", body.get("every_s")), when)
