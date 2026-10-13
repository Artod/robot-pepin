"""An intent rendered into clips: the variant chosen, each step drawn (whether it plays, when
its stroke lands, its parameters, its names, its mirror image), its atom's cycle unrolled into
tracks, the steps of one part merged into that part's clip.

Every draw comes from one ``random.Random(seed)`` in a fixed order, whatever ends up playing, so
a logged seed (and variant) replays a performance exactly: ``scripts/gesture.py play laugh
--seed 1234``. The chooser never draws the variant it drew last for that intent (a composition
of one variant repeats it, its parameters still drawn afresh). A step whose part is not ready,
or whose pose has not been taught, is left out; when every optional step is left out by chance,
the most likely one that can play still plays.

The chooser draws among the variants a scene allows by their ``needs`` and never a floor; the
lead and the ladder are the scheduler's (:mod:`pepin.expression.schedule`), which renders a
given variant (``variant=``) for each rung. Owners after slice 0: B, then C (the resolver of
``@`` references: cells, loci, places, the person's bearing).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from pepin.expression.animation import (
    HOLD,
    PERIOD,
    REPEATS,
    Atom,
    Book,
    Clip,
    Composition,
    Event,
    Layer,
    PartName,
    Performance,
    Scene,
    Step,
    Track,
    Variant,
)
from pepin.expression.markers import Cue


class WeightedChooser:
    """Variants drawn by weight among those the scene allows (:func:`allowed`) and that are no
    floor (the scheduler's last rung, :mod:`pepin.expression.schedule`), never the one drawn last
    time for the same intent; a composition of floors alone draws among them."""

    def __init__(self) -> None:
        """No intent performed yet."""
        self.last: dict[str, int] = {}

    def choose(self, composition: Composition, scene: Scene, rng: random.Random) -> int:
        """The index of the variant to perform."""
        variants = composition.variants
        everyone = list(range(len(variants)))
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


class BookRenderer:
    """:class:`pepin.expression.animation.Renderer` over a book (with the arm's rest pose among
    its poses, :func:`pepin.expression.book.with_rest`)."""

    def __init__(self, book: Book, chooser: WeightedChooser | None = None) -> None:
        """``chooser`` defaults to a fresh :class:`WeightedChooser`."""
        self.book = book
        self.chooser = chooser or WeightedChooser()

    def render(
        self,
        intent: str,
        scene: Scene,
        seed: int,
        cue: Cue | None = None,
        *,
        variant: int | None = None,
    ) -> Performance:
        """The performance of ``intent`` (``variant`` forces one, for a replay); ``KeyError``
        for an intent the book does not hold."""
        composition = self.book.compositions[intent]
        rng = random.Random(seed)
        index = variant if variant is not None else self.chooser.choose(composition, scene, rng)
        drawn = [self._draw(step, rng) for step in composition.variants[index].steps]
        able = [d for d in drawn if d.atom.part in scene.ready and self._posed(d)]
        playing = [d for d in able if d.plays]
        if not playing and able:
            playing = [max(able, key=lambda d: d.step.p)]
        clips = merge([self._unroll(d) for d in playing], "gesture")
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
        (clip,) = merge([self._unroll(drawn)], layer)
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

    def _posed(self, drawn: Drawn) -> bool:
        """Every pose the step's keys name has been taught."""
        return all(
            pose_name(key.pose, drawn.names) in self.book.poses
            for key in drawn.atom.keys
            if key.pose is not None
        )

    def _unroll(self, drawn: Drawn) -> Unrolled:
        """A step's atom over its cycles: (part, start from the sync point, stroke from its
        start, duration, tracks, events)."""
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
            for key in atom.keys:
                if cycle > 0 and key.at == 0.0 and atom.keys[-1].at == 1.0:
                    continue  # the previous cycle's last key stands there already
                t = (cycle + key.at) * period
                pose = self.book.poses[pose_name(key.pose, drawn.names)] if key.pose else {}
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
        tracks = [Track(dof, atom.mode, tuple(times[dof]), tuple(values[dof])) for dof in times]
        return atom.part, start, stroke, duration, tracks, events


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
