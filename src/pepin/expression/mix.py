"""Motion math shared by the laptop's mixer and the board's arm driver: smooth curves through
keys, the layers of one part mixed into poses, joint trajectories sampled from a clip, and the
caps a trajectory must keep. Standard library only: the board imports it.

Between two keys a track moves on a cubic Hermite curve whose tangents are Catmull-Rom's (the
slope between the neighbours) where the keys climb or fall through a key, and zero at an
extreme, a plateau, the first and the last key (monotone: it never overshoots a key, so a
taught pose near a limit stays inside it): a wave through 0, +A, -A, 0 passes its zero
crossing at speed, and a clip starts and ends at rest. A move from one pose to another with
nothing in between (the board's approach to a trajectory's first knot, its way to rest) is
minimum jerk (Flash and Hogan), in the shortest time the speed and acceleration caps allow: a
peak speed of 1.875 d/T and a peak acceleration of 5.774 d/T^2 for a travel d in time T.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

from pepin.expression.animation import Clip, Track

MIN_JERK_SPEED = 1.875  # peak speed of a minimum-jerk move: this times distance over time
MIN_JERK_ACC = 10.0 / math.sqrt(3.0)  # peak acceleration: this times distance over time squared
CAP_SLACK = 1.0 + 1e-6  # float rounding a knot check forgives
STRETCH_MARGIN = (
    0.98  # a stretch aims this far under the caps: the wire rounds knots to 1 ms and 0.01 deg
)
# A preparation compressed to fit its lead runs at this share of the caps, never at them: an
# additive stroke over it (a wave, a wrist beat) then keeps the clip under the caps unstretched
PREP_CAPS = 0.9


def min_jerk(s: float) -> float:
    """The minimum-jerk profile at ``s`` in 0..1 (0 to 1, zero speed and acceleration at both
    ends)."""
    s = min(1.0, max(0.0, s))
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def min_jerk_time(distance: float, max_speed: float, max_acc: float) -> float:
    """The shortest minimum-jerk move over ``distance`` that keeps both caps (seconds)."""
    d = abs(distance)
    if d == 0.0:
        return 0.0
    return max(MIN_JERK_SPEED * d / max_speed, math.sqrt(MIN_JERK_ACC * d / max_acc))


def track_value(track: Track, t: float) -> float:
    """``track`` at ``t`` seconds from its clip's start: held before its first and after its
    last key, a Hermite curve with Catmull-Rom tangents between (zero at the ends)."""
    times, values = track.times, track.values
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    i = 0
    while times[i + 1] < t:
        i += 1
    t0, t1 = times[i], times[i + 1]
    span = t1 - t0
    if span <= 0.0:
        return values[i + 1]
    u = (t - t0) / span
    m0 = _tangent(times, values, i) * span
    m1 = _tangent(times, values, i + 1) * span
    u2, u3 = u * u, u * u * u
    return (
        (2 * u3 - 3 * u2 + 1) * values[i]
        + (u3 - 2 * u2 + u) * m0
        + (-2 * u3 + 3 * u2) * values[i + 1]
        + (u3 - u2) * m1
    )


def _tangent(times: Sequence[float], values: Sequence[float], i: int) -> float:
    """The slope at key ``i``: its neighbours' (Catmull-Rom) where the keys climb or fall
    through it, bounded by three times the gentler side's (Fritsch-Carlson: no overshoot);
    zero at an extreme, a plateau, and the first and last key."""
    if i == 0 or i == len(times) - 1:
        return 0.0
    left_dt, right_dt = times[i] - times[i - 1], times[i + 1] - times[i]
    if left_dt <= 0.0 or right_dt <= 0.0:
        return 0.0
    left = (values[i] - values[i - 1]) / left_dt
    right = (values[i + 1] - values[i]) / right_dt
    if left * right <= 0.0:
        return 0.0
    slope = (values[i + 1] - values[i - 1]) / (left_dt + right_dt)
    bound = 3.0 * min(abs(left), abs(right))
    return math.copysign(min(abs(slope), bound), slope)


def pose_at(clip: Clip, t: float, base: Mapping[str, float]) -> dict[str, float]:
    """One part's pose at ``t`` seconds into ``clip``: the base pose, replaced by each absolute
    track from its first key on (held at its last: the book's absolute atoms start on the base
    pose, and a pose held at a clip's end is the board's to take back to rest), every additive
    track added on top."""
    pose = dict(base)
    for track in clip.tracks:
        if track.mode == "absolute" and track.dof in pose and t >= track.times[0]:
            pose[track.dof] = track_value(track, t)
    for track in clip.tracks:
        if track.mode == "additive" and track.dof in pose:
            pose[track.dof] += track_value(track, t)
    return pose


def stretched(clip: Clip, factor: float) -> Clip:
    """``clip`` played ``factor`` times slower: every time scaled, the stroke kept on its key
    (the clip then starts earlier before the sync point by the stretch)."""
    if factor == 1.0:
        return clip
    tracks = tuple(
        Track(tr.dof, tr.mode, tuple(t * factor for t in tr.times), tr.values) for tr in clip.tracks
    )
    return replace(
        clip,
        offset_s=clip.offset_s - clip.stroke_s * (factor - 1.0),
        stroke_s=clip.stroke_s * factor,
        duration_s=clip.duration_s * factor,
        tracks=tracks,
    )


def rebased(clip: Clip, base: Mapping[str, float], rest: Mapping[str, float]) -> Clip:
    """``clip`` started where ``base`` holds the arm instead of at ``rest``: every absolute
    track's first key that stands on the rest pose (the book's ``arm_pose`` starts there) moves
    to the base's value, so a unit's next stroke goes from the held pose straight to its own."""
    tracks = []
    for track in clip.tracks:
        dof = track.dof
        start = rest.get(dof)
        if track.mode != "absolute" or start is None or dof not in base:
            tracks.append(track)
            continue
        if abs(track.values[0] - start) > 1e-6:
            tracks.append(track)
            continue
        tracks.append(Track(dof, track.mode, track.times, (base[dof], *track.values[1:])))
    return replace(clip, tracks=tuple(tracks))


def fit_preparation(clip: Clip, prep_s: float) -> Clip:
    """``clip`` with the part before its stroke re-timed into ``prep_s`` by one factor (never
    slowed: a ``prep_s`` at or over the stroke's time keeps it), the stroke and after untouched:
    the clip starts later, its stroke stays on its word (Laban's "sudden", the meaning kept)."""
    stroke = clip.stroke_s
    if stroke <= 0.0 or prep_s >= stroke:
        return clip
    factor = max(0.0, prep_s) / stroke
    saved = stroke - prep_s

    def warp(t: float) -> float:
        return t * factor if t <= stroke else t - saved

    tracks = tuple(
        Track(tr.dof, tr.mode, _increasing([warp(t) for t in tr.times]), tr.values)
        for tr in clip.tracks
    )
    events = tuple(replace(e, t_s=warp(e.t_s)) for e in clip.events)
    return replace(
        clip,
        offset_s=clip.offset_s + saved,
        stroke_s=prep_s,
        duration_s=clip.duration_s - saved,
        tracks=tracks,
        events=events,
    )


def _increasing(times: list[float]) -> tuple[float, ...]:
    """Key times a warp may have made equal kept strictly in order (1 us apart)."""
    out: list[float] = []
    for t in times:
        out.append(t if not out or t > out[-1] else out[-1] + 1e-6)
    return tuple(out)


def prep_min_s(
    clip: Clip,
    base: Mapping[str, float],
    joints: Sequence[str],
    max_speed: float,
    max_acc: float,
    hz: float = 50.0,
) -> float:
    """The least time from ``clip``'s start to its stroke that keeps the part before the stroke
    within ``max_speed`` and ``max_acc`` (one factor on that part's times, as
    :func:`fit_preparation` applies it); the clip's own time when that is already the least."""
    stroke = clip.stroke_s
    if stroke <= 0.0:
        return 0.0
    steps = max(2, math.ceil(stroke * hz))
    dt = stroke / steps
    poses = [pose_at(clip, k * dt, base) for k in range(steps + 1)]
    speeds = [[(b[j] - a[j]) / dt for j in joints] for a, b in itertools.pairwise(poses)]
    top_speed = max((abs(v) for row in speeds for v in row), default=0.0)
    top_acc = max(
        (abs(w - v) / dt for a, b in itertools.pairwise(speeds) for v, w in zip(a, b, strict=True)),
        default=0.0,
    )
    factor = 0.0
    if max_speed > 0:
        factor = max(factor, top_speed / max_speed)
    if max_acc > 0:
        factor = max(factor, math.sqrt(top_acc / max_acc))
    return stroke * min(1.0, factor)


def sample(
    clip: Clip, base: Mapping[str, float], joints: Sequence[str], hz: float
) -> list[tuple[float, tuple[float, ...]]]:
    """``clip`` as knots ``(t, one angle per joint)`` every ``1 / hz`` seconds from 0 to its
    end (the last knot exactly at the end)."""
    steps = max(1, math.ceil(clip.duration_s * hz - 1e-9))
    out = []
    for k in range(steps + 1):
        t = clip.duration_s * k / steps
        pose = pose_at(clip, t, base)
        out.append((t, tuple(pose[j] for j in joints)))
    return out


def stretch_to_caps(
    knots: Sequence[tuple[float, Sequence[float]]], max_speed: float, max_acc: float
) -> list[tuple[float, list[float]]]:
    """``knots`` with their times stretched by one factor so no step exceeds ``max_speed`` and no
    change of speed exceeds ``max_acc`` (deg/s, deg/s^2): a gesture asked slower than its caps
    allow takes longer instead of being refused. Angles are never changed; a factor under 1 is 1."""
    factor = 1.0
    previous: list[float] | None = None
    for k in range(1, len(knots)):
        t0, d0 = knots[k - 1]
        t, deg = knots[k]
        dt = t - t0
        if dt <= 0.0:
            continue
        speeds = [(b - a) / dt for a, b in zip(d0, deg, strict=True)]
        if max_speed > 0:
            factor = max(factor, max(abs(v) for v in speeds) / max_speed)
        if previous is not None and max_acc > 0:
            dt_mid = (t - knots[k - 2][0]) / 2.0
            if dt_mid > 0:
                acc = max(abs(v1 - v0) for v0, v1 in zip(previous, speeds, strict=True)) / dt_mid
                factor = max(factor, math.sqrt(acc / max_acc))
        previous = speeds
    if factor <= 1.0 + 1e-9:
        return [(t, list(d)) for t, d in knots]
    t_first = knots[0][0] if knots else 0.0
    return [(t_first + (t - t_first) * factor, list(d)) for t, d in knots]


def check_knots(
    knots: Sequence[tuple[float, Sequence[float]]],
    limits: Sequence[tuple[float, float]],
    max_speed: float,
    max_acc: float,
    names: Sequence[str] = (),
    slack: float = CAP_SLACK,
) -> str | None:
    """Why a trajectory may not be played, None when it may: a knot outside a joint's limits,
    times that do not grow, a step faster than ``max_speed`` or a change of speed between two
    steps faster than ``max_acc`` (finite differences, deg/s and deg/s^2; ``slack`` times the
    caps forgiven). Never clamps."""

    def name(j: int) -> str:
        return names[j] if j < len(names) else f"joint {j}"

    previous: list[float] | None = None
    for k, (t, deg) in enumerate(knots):
        if len(deg) != len(limits):
            return f"knot {k} has {len(deg)} angles for {len(limits)} joints"
        for j, (value, (low, high)) in enumerate(zip(deg, limits, strict=True)):
            if not (math.isfinite(value) and low <= value <= high):
                where = f"knot {k} at {t:.2f} s: {name(j)} {value:+.1f} deg"
                return f"{where} outside {low:+.1f}..{high:+.1f}"
        if k == 0:
            continue
        t0, d0 = knots[k - 1]
        dt = t - t0
        if dt <= 0.0:
            return f"knot {k}: its time {t:.3f} s does not follow {t0:.3f} s"
        speeds = [(b - a) / dt for a, b in zip(d0, deg, strict=True)]
        for j, v in enumerate(speeds):
            if abs(v) > max_speed * slack:
                return (
                    f"knot {k} at {t:.2f} s: {name(j)} at {abs(v):.0f} deg/s, over {max_speed:.0f}"
                )
        if previous is not None:
            dt_mid = (t - knots[k - 2][0]) / 2.0
            for j, (v0, v1) in enumerate(zip(previous, speeds, strict=True)):
                acc = abs(v1 - v0) / dt_mid
                if acc > max_acc * slack:
                    return (
                        f"knot {k} at {t:.2f} s: {name(j)} at {acc:.0f} deg/s^2, over {max_acc:.0f}"
                    )
        previous = speeds
    return None


def approach(
    start: Sequence[float],
    goal: Sequence[float],
    max_speed: float,
    max_acc: float,
    hz: float,
) -> list[tuple[float, tuple[float, ...]]]:
    """A minimum-jerk move of every joint at once from ``start`` to ``goal`` in the shortest
    time the caps allow for the joint that travels furthest, as knots every ``1 / hz`` s (at
    least the two ends)."""
    distance = max((abs(g - s) for s, g in zip(start, goal, strict=True)), default=0.0)
    seconds = min_jerk_time(distance, max_speed, max_acc)
    steps = max(1, math.ceil(seconds * hz))
    out = []
    for k in range(steps + 1):
        share = min_jerk(k / steps)
        t = seconds * k / steps
        out.append((t, tuple(s + share * (g - s) for s, g in zip(start, goal, strict=True))))
    return out
