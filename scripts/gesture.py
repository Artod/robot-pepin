"""The body's gestures from the laptop: read the arm, rest it, halt it, play a gesture of the book
at a chosen speed and seed, teach a pose by hand.

    uv run python scripts/gesture.py list                     # the book's intents and poses
    uv run python scripts/gesture.py read                     # the arm's six angles (no motion)
    uv run python scripts/gesture.py rest                     # to rest, then torque off
    uv run python scripts/gesture.py halt                     # stop where it is (held, then rest)
    uv run python scripts/gesture.py play greeting --speed 0.2
    uv run python scripts/gesture.py play laugh --seed 1234 --variant 0 --dry-run
    uv run python scripts/gesture.py play point:printer --dry-run   # a mark with its argument
    uv run python scripts/gesture.py teach raised             # torque off: place the arm, Enter
    uv run python scripts/gesture.py nudge shoulder_pan 5     # one joint +5 deg and back, slowly
    uv run python scripts/gesture.py turn 10                  # the wheels: 10 deg left in place
    uv run python scripts/gesture.py rock 2 --times 1         # 2 cm forward and back
    uv run python scripts/gesture.py turn -10 --dry-run       # every check, nothing moves

``play`` renders the intent from config/animation.json (seed and variant printed, so a run is
replayed with them; a mark's argument after a colon, as the brain writes it, a place's name
read once from the goal server's places book and the robot's pose, the person straight
ahead), stretches its arm clip by 1/speed, checks it against config/arm.json's
limits and caps scaled by the speed, and sends it to the base server, which plays it by itself
and rests; the face's expressions go to the head server (source ``expression``); the head's
clip goes to the gaze arbiter's door on this Mac as an overlay on whatever holds the head
(``--no-head`` leaves it out). Ctrl-C during a play sends a halt and releases the overlay.
``--dry-run`` prints the plan and sends nothing. ``teach`` reads the arm while its torque is off
(move it by hand first) and writes the six angles as ``poses.NAME`` of config/animation.json.
``turn`` and ``rock`` are the wheels' expressive motions exactly as the voice cascade makes them
(pepin.expression.wheels: twists to the base server judged by its odometry, the goal server on
this Mac saying nothing navigates and depth_fusion's obstacle marks clear the arc), at
``--speed`` (a share of config/expression.json's wheels caps, its ``wheels.speed_scale`` by
default); Ctrl-C halts. ``ros/stop.sh`` stays the red button: the motion ends on the first
state line of a stop it did not send.
The board is ``PEPIN_HOST`` (or --host).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

from pepin.arm_motion import ArmMotion, load_arm_motion
from pepin.audio_link import board_host
from pepin.expression.animation import Book, Clip, Scene, Track
from pepin.expression.arm_link import ArmClient
from pepin.expression.book import book_from, load_book, vocabulary, with_rest
from pepin.expression.door import OverlayDoor
from pepin.expression.head import GESTURE, ClipLook
from pepin.expression.markers import ANY, gesture
from pepin.expression.mix import stretched
from pepin.expression.parts import ArmPart, NeckLimits, clip_look
from pepin.expression.places import GoalPlaces, Room
from pepin.expression.render import BookRenderer, Resolver
from pepin.expression.trajectory import JOINTS, Trajectory
from pepin.neck import NeckConfig
from pepin.tools.clients import GoalServerLink
from pepin.tools.robot import Endpoints

REPO = Path(__file__).resolve().parents[1]
BOOK = REPO / "config" / "animation.json"
ARM = REPO / "config" / "arm.json"
NECK = REPO / "config" / "neck.json"
HEAD_FADE_S = 0.35  # config/expression.json's layers.fade_s.head
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


def asked(text: str, book: Book) -> tuple[str, str | None] | None:
    """``play``'s NAME as a composition and its argument: a composition's name as it is (its
    first argument when it takes one), else as the brain writes a mark (``point:printer``,
    ``count:second``, ``!``)."""
    if text in book.compositions:
        return text, None
    named = gesture(text, vocabulary(book, frozenset()))
    return None if named is None else (named[1], named[2])


def room_for(book: Book, name: str, arg: str | None) -> Room | None:
    """The goal server's places, read once, for a mark that may name one (None: none needed);
    where the argument points, printed."""
    takes = book.compositions[name].args
    if arg is None or ANY not in takes or arg in takes:
        return None
    where = Endpoints.from_env()
    room = GoalPlaces(GoalServerLink(where.goal_host, where.goal_port))
    room.refresh()
    found = room.find(arg)
    if not room.known():
        print(f"  {arg}: the goal server did not answer: no place is known (its floor)")
    elif found is None:
        print(f"  {arg}: no place of that name: a referent (the first spot)")
    else:
        there = room.where(found)
        how = "no fresh pose" if there is None else f"{there[0]:+.0f} deg, {there[1]:.1f} m"
        print(f"  {arg}: the place {found}, {how}")
    return room


def play(args: argparse.Namespace, motion: ArmMotion) -> int:
    book = with_rest(load_book(BOOK), motion.rest_deg)
    found = asked(args.name, book)
    if found is None:
        print(f"no gesture {args.name!r}; the gestures are {', '.join(book.compositions)}"
              " (an argument after a colon: point:printer)")  # fmt: skip
        return 2
    name, arg = found
    if not 0.0 < args.speed <= 1.0:
        print("--speed: above 0, at most 1")
        return 2
    seed = args.seed if args.seed is not None else int(time.time()) % 2**31
    scene = Scene(frozenset({"arm", "face", "head"}))
    renderer = BookRenderer(book, resolver=Resolver(room=room_for(book, name, arg)))
    performance = renderer.render(name, scene, seed, variant=args.variant, arg=arg)
    mark = name if arg is None else f"{name}:{arg}"
    print(f"{mark}: seed {seed}, variant {performance.variant}"
          f" (replay: --seed {seed} --variant {performance.variant})")  # fmt: skip
    if not performance.clips:
        print("  nothing plays: a variant written for another argument, or a place past both the"
              " arm's and the head's reach")  # fmt: skip
        return 2
    part = ArmPart(_NoLink(), motion, speed_scale=args.speed, knot_hz=args.knot_hz)
    limits = NeckLimits.of(NeckConfig.from_json(NECK).motion)
    trajectory: Trajectory | None = None
    overlay: ClipLook | None = None
    clips = [stretched(c, 1.0 / args.speed) if c.part == "arm" else c for c in performance.clips]
    start = min(c.offset_s for c in clips)
    for clip in clips:
        at = clip.offset_s - start
        if clip.part == "arm":
            slow = clip
            plan = part.plan(slow, name)
            if isinstance(plan, str):
                print(f"  arm: refused here: {plan}")
                return 3
            planned = trajectory = plan.trajectory
            seconds = planned.knots[-1].t_s
            stretched_by = (
                f" (the clip's {slow.duration_s:.1f} s stretched by the caps x{plan.stretch:.2f})"
                if seconds > slow.duration_s + 0.05
                else ""
            )
            print(
                f"  arm at +{at:.2f} s: {len(planned.knots)} knots over {seconds:.1f} s"
                f"{stretched_by} at speed {args.speed}, stroke at {plan.stroke_s:.2f} s, then"
                f" rest; peak {_peak(planned, motion)}"
            )
        elif clip.part == "face":
            shown = [f"{e.name} for {e.hold_s or 2.0:.1f} s" for e in clip.events]
            print(f"  face at +{at:.2f} s: {', '.join(shown)}")
        elif clip.part == "head":
            overlay = None if args.no_head else clip_look(clip, GESTURE, limits)
            plan = "left out (--no-head)" if args.no_head else _overlay(overlay)
            print(f"  head at +{at:.2f} s: {plan}")
        else:
            print(f"  {clip.part} at +{at:.2f} s for {clip.duration_s:.1f} s: not played here"
                  " (the wheels: scripts/gesture.py turn | rock)")  # fmt: skip
    if args.dry_run:
        return 0
    face = None
    if not args.no_face and any(c.part == "face" for c in clips):
        from pepin.head_link import HeadClient

        face = HeadClient(args.host, source="expression").start()
    client = connect(args.host) if trajectory is not None else None
    door = OverlayDoor() if overlay is not None else None
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
            elif clip.part == "head" and door is not None and overlay is not None:
                refusal = door.look(overlay)
                print(f"  gaze arbiter: {'playing' if refusal is None else 'refused: ' + refusal}")
        if client is not None and trajectory is not None:
            end = client.wait_answer(trajectory.id, DONE, trajectory.knots[-1].t_s + 15.0)
            print(f"  board: {end.status if end else 'no answer'}"
                  f"{': ' + end.reason if end and end.reason else ''}")  # fmt: skip
    except KeyboardInterrupt:
        if door is not None:
            door.withdraw(GESTURE, HEAD_FADE_S)
            print("\nthe head's overlay released")
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


def _overlay(look: ClipLook | None) -> str:
    """A head overlay in one line: its keys, its furthest offsets and its fastest key."""
    if look is None:
        return "no head motion"
    keys = look["target"]["keys"]
    pan = max(abs(k["pan_rad"]) for k in keys)
    tilt = max(abs(k["tilt_rad"]) for k in keys)
    fastest = max(k["speed_deg_s"] for k in keys)
    return (
        f"{len(keys)} keys over the holder's aim, pan up to {math.degrees(pan):.1f} deg, tilt"
        f" up to {math.degrees(tilt):.1f} deg, fastest key {fastest:.0f} deg/s"
    )


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
    plan = part.plan(clip, f"nudge-{args.joint}")
    if isinstance(plan, str):
        print(f"refused here: {plan}")
        return 3
    planned = plan.trajectory
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


def wheels(args: argparse.Namespace) -> int:
    """``turn DEG`` or ``rock CM``: the checks, then the motion (none with ``--dry-run``)."""
    import threading
    from dataclasses import replace

    from pepin.expression.executors import WheelMove, WheelReport
    from pepin.expression.wheels import WheelsSettings, robot_wheels

    if args.speed is None:
        args.speed = WheelsSettings.load().speed_scale
    if not 0.0 < args.speed <= 1.0:
        print("--speed: above 0, at most 1")
        return 2
    settings = replace(WheelsSettings.load(), speed_scale=args.speed)
    if args.command == "turn":
        request = WheelMove(turn_deg=args.deg)
        rate, acc = settings.yaw_caps()
        print(f"turn {args.deg:+.1f} deg in place at speed {args.speed:g}: up to"
              f" {math.degrees(rate):.1f} deg/s, ramp {math.degrees(acc):.1f} deg/s^2")  # fmt: skip
    else:
        request = WheelMove(rock_m=args.cm / 100.0, rocks=args.times)
        speed, acc = settings.drive_caps()
        print(f"rock {args.cm:+.1f} cm and back x{args.times} at speed {args.speed:g}: up to"
              f" {speed * 100:.1f} cm/s, ramp {acc * 100:.1f} cm/s^2")  # fmt: skip
    executor = robot_wheels(args.host, load=lambda: replace(WheelsSettings.load(),
                                                            speed_scale=args.speed))  # fmt: skip
    try:
        refusal = executor.preview(request)
        retries = bool(request.turn_deg) and str(refusal).startswith("the arc")
        then = " (the turn would step away and try once more)" if retries else ""
        print(f"checks: {'ok' if refusal is None else refusal}{then}")
        if args.dry_run or (refusal is not None and not retries):
            return 0 if refusal is None else 3
        ended = threading.Event()
        reports: list[WheelReport] = []

        def done(report: WheelReport) -> None:
            reports.append(report)
            ended.set()

        if args.command == "turn":
            executor.turn(args.deg, done, lambda deg: print(f"  turned {deg:+.1f} deg"))
        else:
            executor.rock(args.cm / 100.0, args.times, done)
        try:
            while not ended.wait(0.1):
                pass
        except KeyboardInterrupt:
            executor.halt("Ctrl-C")
            ended.wait(2.0)
            print("\nhalted: the stop went to the base server")
            return 130
        report = reports[0]
        return 0 if report.done else 3
    finally:
        executor.close()


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
    p.add_argument("--no-head", action="store_true", help="leave the head alone")
    p.add_argument("--dry-run", action="store_true", help="print the plan, send nothing")
    n = sub.add_parser("nudge", help="one joint some degrees from rest and back, slowly")
    n.add_argument("joint", help=", ".join(JOINTS))
    n.add_argument("deg", type=float)
    n.add_argument("--seconds", type=float, default=2.0, help="the motion at speed 1")
    n.add_argument("--speed", type=float, default=0.2, help="share of the caps")
    n.add_argument("--dry-run", action="store_true", help="print the plan, send nothing")
    t = sub.add_parser("teach", help="write the arm's present angles as a pose of the book")
    t.add_argument("name")
    w = sub.add_parser("turn", help="the wheels: turn in place (+ left), as the gaze does")
    w.add_argument("deg", type=float)
    r = sub.add_parser("rock", help="the wheels: forward and back (backward first when negative)")
    r.add_argument("cm", type=float, nargs="?", default=2.0)
    r.add_argument("--times", type=int, choices=range(1, 6), default=1, metavar="1..5")
    for wheel in (w, r):
        wheel.add_argument("--speed", type=float, help="share of the wheels' caps (default:"
                           " config/expression.json's wheels.speed_scale)")  # fmt: skip
        wheel.add_argument("--dry-run", action="store_true", help="every check, nothing moves")
    args = ap.parse_args()
    if args.command in ("turn", "rock"):
        return wheels(args)
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
