"""The body's gestures from the laptop: read the arm, rest it, halt it, play a gesture of the book
at a chosen speed and seed, teach a pose by hand.

    uv run python scripts/gesture.py list                     # the book's intents and poses
    uv run python scripts/gesture.py read                     # the arm's six angles (no motion)
    uv run python scripts/gesture.py rest                     # to rest, then torque off
    uv run python scripts/gesture.py halt                     # stop where it is (held, then rest)
    uv run python scripts/gesture.py play greeting --speed 0.2
    uv run python scripts/gesture.py play laugh --seed 1234 --variant 0 --dry-run
    uv run python scripts/gesture.py teach raised             # torque off: place the arm, Enter
    uv run python scripts/gesture.py nudge shoulder_pan 5     # one joint +5 deg and back, slowly

``play`` renders the intent from config/animation.json (seed and variant printed, so a run is
replayed with them), stretches its arm clip by 1/speed, checks it against config/arm.json's
limits and caps scaled by the speed, and sends it to the base server, which plays it by itself
and rests; the face's expressions go to the head server (source ``expression``); head steps
wait for the gaze arbiter's overlay and are only printed. Ctrl-C during a play sends a halt.
``--dry-run`` prints the plan and sends nothing. ``teach`` reads the arm while its torque is off
(move it by hand first) and writes the six angles as ``poses.NAME`` of config/animation.json.
The board is ``PEPIN_HOST`` (or --host).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from pepin.arm_motion import ArmMotion, load_arm_motion
from pepin.audio_link import board_host
from pepin.expression.animation import Clip, Scene, Track
from pepin.expression.arm_link import ArmClient
from pepin.expression.book import book_from, load_book, with_rest
from pepin.expression.mix import stretched
from pepin.expression.parts import ArmPart
from pepin.expression.render import BookRenderer
from pepin.expression.trajectory import JOINTS, Trajectory

REPO = Path(__file__).resolve().parents[1]
BOOK = REPO / "config" / "animation.json"
ARM = REPO / "config" / "arm.json"
POSES = re.compile(r'^  "poses": \{\n.*?^  \},\n', re.MULTILINE | re.DOTALL)
DONE = ("done", "refused", "preempted", "halted")


def angles(deg: list[float] | tuple[float, ...]) -> str:
    """Six angles, named."""
    return "  ".join(f"{j} {d:+7.2f}" for j, d in zip(JOINTS, deg, strict=True))


def connect(host: str) -> ArmClient:
    """The arm's link, once the base server's state lines carry the arm (or exit 2)."""
    client = ArmClient(host).start()
    if client.wait_status(3.0) is None:
        client.close()
        sys.exit(f"no arm in {host}'s state lines: is pepin-base running with config/arm.json?")
    return client


def show_list(motion: ArmMotion) -> int:
    book = load_book(BOOK)
    for name, composition in book.compositions.items():
        parts = sorted({book.atoms[s.atom].part for v in composition.variants for s in v.steps})
        print(f"{name:13} {len(composition.variants)} variant(s), {', '.join(parts)}: "
              f"{composition.note}")  # fmt: skip
    taught = ", ".join(book.poses) or "none (scripts/gesture.py teach NAME)"
    print(f"poses: {taught}; rest is config/arm.json's: {angles(motion.rest())}")
    return 0


def read(host: str) -> int:
    client = connect(host)
    try:
        reply = client.ask("arm_read")
    finally:
        client.close()
    if reply is None:
        print("no answer to arm_read")
        return 3
    if "deg" in reply:
        print(angles(reply["deg"]))
    print(json.dumps({k: v for k, v in reply.items() if k not in ("type", "deg")}))
    return 0


def simple(host: str, cmd: str) -> int:
    client = connect(host)
    try:
        reply = client.ask(cmd)
    finally:
        client.close()
    print(reply if reply is not None else f"no answer to {cmd}")
    return 0 if reply is not None else 3


def play(args: argparse.Namespace, motion: ArmMotion) -> int:
    book = with_rest(load_book(BOOK), motion.rest_deg)
    if args.name not in book.compositions:
        print(f"no gesture {args.name!r}; the gestures are {', '.join(book.compositions)}")
        return 2
    if not 0.0 < args.speed <= 1.0:
        print("--speed: above 0, at most 1")
        return 2
    seed = args.seed if args.seed is not None else int(time.time()) % 2**31
    scene = Scene(frozenset({"arm", "face", "head"}))
    performance = BookRenderer(book).render(args.name, scene, seed, variant=args.variant)
    print(f"{args.name}: seed {seed}, variant {performance.variant}"
          f" (replay: --seed {seed} --variant {performance.variant})")  # fmt: skip
    part = ArmPart(_NoLink(), motion, speed_scale=args.speed, knot_hz=args.knot_hz)
    trajectory: Trajectory | None = None
    clips = [stretched(c, 1.0 / args.speed) if c.part == "arm" else c for c in performance.clips]
    start = min(c.offset_s for c in clips)
    for clip in clips:
        at = clip.offset_s - start
        if clip.part == "arm":
            slow = clip
            planned = part.plan(slow, args.name)
            if isinstance(planned, str):
                print(f"  arm: refused here: {planned}")
                return 3
            trajectory = planned
            print(f"  arm at +{at:.2f} s: {len(planned.knots)} knots over {slow.duration_s:.1f} s"
                  f" at speed {args.speed}, then rest; peak {_peak(planned, motion)}")  # fmt: skip
        elif clip.part == "face":
            shown = [f"{e.name} for {e.hold_s or 2.0:.1f} s" for e in clip.events]
            print(f"  face at +{at:.2f} s: {', '.join(shown)}")
        else:
            print(f"  {clip.part} at +{at:.2f} s for {clip.duration_s:.1f} s: not played (no"
                  f" {clip.part} executor yet)")  # fmt: skip
    if args.dry_run:
        return 0
    face = None
    if not args.no_face and any(c.part == "face" for c in clips):
        from pepin.head_link import HeadClient

        face = HeadClient(args.host, source="expression").start()
    client = connect(args.host) if trajectory is not None else None
    try:
        began = time.monotonic()
        for clip in sorted(clips, key=lambda c: c.offset_s):
            wait = clip.offset_s - start - (time.monotonic() - began)
            if wait > 0:
                time.sleep(wait)
            if clip.part == "arm" and client is not None and trajectory is not None:
                client.send(trajectory)
                answer = client.wait_answer(trajectory.id, ("accepted", *DONE), 3.0)
                print(f"  board: {answer.status if answer else 'no answer'}"
                      f"{': ' + answer.reason if answer and answer.reason else ''}")  # fmt: skip
            elif clip.part == "face" and face is not None:
                for event in clip.events:
                    face.express(event.name, hold_s=event.hold_s or 2.0)
        if client is not None and trajectory is not None:
            end = client.wait_answer(trajectory.id, DONE, trajectory.knots[-1].t_s + 15.0)
            print(f"  board: {end.status if end else 'no answer'}"
                  f"{': ' + end.reason if end and end.reason else ''}")  # fmt: skip
    except KeyboardInterrupt:
        if client is not None:
            client.halt()
            print("\nhalt sent: the arm holds where it is, then rests by itself")
        return 130
    finally:
        if client is not None:
            client.close()
        if face is not None:
            face.close()
    return 0


def _peak(trajectory: Trajectory, motion: ArmMotion) -> str:
    """The joint that travels furthest from rest, and how far."""
    rest = motion.rest()
    far = max(
        ((j, max(abs(k.deg[i] - rest[i]) for k in trajectory.knots)) for i, j in enumerate(JOINTS)),
        key=lambda x: x[1],
    )
    return f"{far[0]} {far[1]:.1f} deg from rest"


class _NoLink:
    """The planner's link when only planning: it sends nothing."""

    def send(self, trajectory: Trajectory) -> None:
        """Nothing."""

    def rest(self) -> None:
        """Nothing."""

    def halt(self) -> None:
        """Nothing."""

    def status(self) -> None:
        """Nothing."""

    def answer(self, trajectory_id: str) -> None:
        """Nothing."""


def nudge(args: argparse.Namespace, motion: ArmMotion) -> int:
    """One joint ``deg`` from rest and back over ``seconds``, at ``speed``: the first motion
    of a live test (no jump at torque-on, rest, torque off)."""
    if args.joint not in JOINTS:
        print(f"a joint: one of {', '.join(JOINTS)}")
        return 2
    seconds = args.seconds
    track = Track(args.joint, "additive", (0.0, 0.4 * seconds, 0.6 * seconds, seconds),
                  (0.0, args.deg, args.deg, 0.0))  # fmt: skip
    part = ArmPart(_NoLink(), motion, speed_scale=args.speed)
    clip = stretched(Clip("arm", 0.0, 0.0, seconds, (track,)), 1.0 / args.speed)
    planned = part.plan(clip, f"nudge-{args.joint}")
    if isinstance(planned, str):
        print(f"refused here: {planned}")
        return 3
    print(f"{args.joint} {args.deg:+.1f} deg and back over {clip.duration_s:.1f} s at speed"
          f" {args.speed} ({len(planned.knots)} knots), then rest and torque off")  # fmt: skip
    if args.dry_run:
        return 0
    client = connect(args.host)
    try:
        client.send(planned)
        for wanted in (("accepted", *DONE), DONE):
            answer = client.wait_answer(planned.id, wanted, seconds + 15.0)
            print(f"  board: {answer.status if answer else 'no answer'}"
                  f"{': ' + answer.reason if answer and answer.reason else ''}")  # fmt: skip
            if answer is None or answer.status in DONE:
                break
    except KeyboardInterrupt:
        client.halt()
        print("\nhalt sent: the arm holds where it is, then rests by itself")
        return 130
    finally:
        client.close()
    return 0


def teach(args: argparse.Namespace, motion: ArmMotion) -> int:
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", args.name) or args.name == "rest":
        print("a pose's name: lower_snake_case, not rest (config/arm.json's)")
        return 2
    client = connect(args.host)
    try:
        status = client.status()
        if status is not None and status.torque:
            print("the arm is energised: run `scripts/gesture.py rest` and wait for torque off")
            return 3
        input(f"torque is off: place the arm in pose {args.name!r} by hand, then press Enter ")
        reply = client.ask("arm_read")
    finally:
        client.close()
    if reply is None or "deg" not in reply:
        print(f"no reading: {reply}")
        return 3
    deg = [round(float(d), 1) for d in reply["deg"]]
    print(angles(deg))
    outside = [j for j, d, (lo, hi) in zip(JOINTS, deg, motion.limits(), strict=True)
               if not lo <= d <= hi]  # fmt: skip
    if outside:
        print(f"outside config/arm.json's limits: {outside}; not written")
        return 3
    if input(f"write poses.{args.name} into config/animation.json? [y/N] ").strip().lower() != "y":
        return 0
    write_pose(args.name, dict(zip(JOINTS, deg, strict=True)))
    print(f"poses.{args.name} written; `scripts/gesture.py play greeting --dry-run` shows it")
    return 0


def write_pose(name: str, joints: dict[str, float]) -> None:
    """``poses.NAME`` of config/animation.json set, the rest of the file as it was; checked."""
    text = BOOK.read_text()
    data = json.loads(text)
    poses: dict[str, Any] = dict(data.get("poses", {}))
    poses[name] = joints
    lines = [f"    {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}" for k, v in poses.items()]
    block = '  "poses": {\n' + ",\n".join(lines) + "\n  },\n"
    updated = POSES.sub(lambda _: block, text, count=1)
    book_from(json.loads(updated))  # a book that does not load is never written
    BOOK.write_text(updated)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--host", default=board_host(), help="the board (PEPIN_HOST)")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the book's intents and the taught poses")
    sub.add_parser("read", help="the arm's angles, torque and phase (no motion)")
    sub.add_parser("rest", help="to rest at the caps, then torque off")
    sub.add_parser("halt", help="stop where it is, held, then rest by itself")
    p = sub.add_parser("play", help="perform an intent of the book")
    p.add_argument("name")
    p.add_argument("--speed", type=float, default=0.2, help="share of the caps (0.2: 5x slower)")
    p.add_argument("--seed", type=int, help="replay a performance (default: a fresh one)")
    p.add_argument("--variant", type=int, help="force a variant (with --seed: a replay)")
    p.add_argument("--knot-hz", type=float, default=25.0, help="knots a second at speed 1")
    p.add_argument("--no-face", action="store_true", help="leave the face alone")
    p.add_argument("--dry-run", action="store_true", help="print the plan, send nothing")
    n = sub.add_parser("nudge", help="one joint some degrees from rest and back, slowly")
    n.add_argument("joint", help=", ".join(JOINTS))
    n.add_argument("deg", type=float)
    n.add_argument("--seconds", type=float, default=2.0, help="the motion at speed 1")
    n.add_argument("--speed", type=float, default=0.2, help="share of the caps")
    n.add_argument("--dry-run", action="store_true", help="print the plan, send nothing")
    t = sub.add_parser("teach", help="write the arm's present angles as a pose of the book")
    t.add_argument("name")
    args = ap.parse_args()
    motion = load_arm_motion(ARM)
    if args.command == "list":
        return show_list(motion)
    if args.command == "read":
        return read(args.host)
    if args.command in ("rest", "halt"):
        return simple(args.host, f"arm_{args.command}")
    if args.command == "nudge":
        return nudge(args, motion)
    if args.command == "play":
        return play(args, motion)
    return teach(args, motion)


if __name__ == "__main__":
    sys.exit(main())
