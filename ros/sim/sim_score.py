"""Drive the simulated cart and score it: the goal goes through the goal server's socket with the
very client ros/goto.sh uses (pepin.goal_link), and the score is the world's own odometer.

Runs beside the world (ros/sim.sh execs it in the world's container, which shares the sim's
loopback): the goal server on 127.0.0.1:3337, the world's control socket on 127.0.0.1:3390.

    python3 /sim/sim_score.py state
    python3 /sim/sim_score.py place NAME | X Y [YAW_DEG]
    python3 /sim/sim_score.py boxes FILE | none
    python3 /sim/sim_score.py goal NAME | X Y [YAW_DEG]
    python3 /sim/sim_score.py scenario FILE [--repeat N] [--settle S] [--tag TEXT]

A leg's score: Nav2's status, seconds on the world's clock (sim seconds under --rate), the
recoveries the behaviour tree reported, the hull's path against the straight line, contacts,
and the arrival against the goal. Every score is also appended to /maps/rec/sim_scores.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from pepin.goal_link import NAV2_CANCELED, NAV2_SUCCEEDED, ask, events, goal_request
from pepin.odometry import Pose2D
from pepin.sim import Box, LegScore, Scenario, Target, places_from_payload, score_leg

GOAL_PORT = 3337  # pepin_bringup.goal_server.PORT
WORLD_PORT = 3390  # sim_world.CONTROL_PORT
HOST = "127.0.0.1"
WORLDS = Path("/sim/worlds")
SCORES = Path("/maps/rec/sim_scores.jsonl")
SETTLE_S = 3.0  # after a place: the costmaps refill from a few scans before the goal


def world(request: dict[str, Any]) -> dict[str, Any]:
    """One request to the world's control socket."""
    return ask(request, HOST, WORLD_PORT, timeout_s=5.0)


def book(world_name: str) -> dict[str, Pose2D]:
    """The world's places, as the goal server hears them on /places."""
    return places_from_payload((WORLDS / f"{world_name}.places.json").read_text())


def target_of(tokens: list[str], places: dict[str, Pose2D]) -> tuple[str, Pose2D]:
    """``NAME`` or ``X Y [YAW_DEG]`` as a label and a pose in the map."""
    request = goal_request(tokens)
    if "place" in request:
        name = str(request["place"])
        return name, Target(place=name).resolve(places)
    pose = Pose2D(request["x"], request["y"], math.radians(request["yaw_deg"]))
    return f"({pose.x:+.2f}, {pose.y:+.2f}, {request['yaw_deg']:+.0f})", pose


def place(pose: Pose2D, settle_s: float = SETTLE_S) -> dict[str, Any]:
    """Teleport the cart and give the costmaps ``settle_s`` to see the room from there."""
    answer = world({"cmd": "place", "x": pose.x, "y": pose.y, "yaw_deg": math.degrees(pose.theta)})
    time.sleep(settle_s)
    return answer


def drive(label: str, request: dict[str, Any], target: Pose2D, timeout_s: float) -> LegScore:
    """One goal through the goal server, printed as it goes; cancelled past ``timeout_s``."""
    before = world({"cmd": "state"})
    recoveries, status, cancelled = 0, "NO RESULT", False
    wall0 = time.monotonic()
    for event in events(request, HOST, GOAL_PORT, 5.0, 120.0):
        kind = event.get("event")
        if kind == "feedback":
            recoveries = max(recoveries, int(event.get("recoveries", 0)))
            print(
                f"  t+{event.get('t', 0):5.1f}s  {float(event.get('distance', math.nan)):5.2f} m"
                f" left, recoveries {recoveries}",
                flush=True,
            )
        elif kind == "done":
            code = int(event.get("status", 0))
            status = {NAV2_SUCCEEDED: "SUCCEEDED", NAV2_CANCELED: "CANCELED", 6: "ABORTED"}.get(
                code, f"STATUS {code}"
            )
            if cancelled:
                status = f"TIMEOUT ({status})"
        elif kind == "error":
            status = f"REFUSED: {event.get('detail')}"
        elif kind == "accepted":
            print(
                f"  accepted: planner {event.get('planner')}, run {event.get('run')},"
                f" taped {event.get('recording')}",
                flush=True,
            )
        if not cancelled and time.monotonic() - wall0 > timeout_s:
            cancelled = True
            print(f"  {timeout_s:.0f} s: cancelling", flush=True)
            ask({"cmd": "cancel"}, HOST, GOAL_PORT)
    after = world({"cmd": "state"})
    score = score_leg(label, target, before, after, status, recoveries)
    print(score.line() + f" [wall {time.monotonic() - wall0:.1f} s]", flush=True)
    return score


def record(score: LegScore, **context: Any) -> None:
    """Append the score with its context to the run directory's score file."""
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    with SCORES.open("a") as out:
        out.write(json.dumps({"stamp": time.time(), **context, **score.to_dict()}) + "\n")


def run_scenario(path: Path, repeat: int, settle_s: float, tag: str) -> int:
    """Place, furnish and drive every leg of a scenario ``repeat`` times; 0 when every leg
    succeeded (a slow leg is reported, not failed: showing it is what a scenario is for)."""
    scenario = Scenario.load(path)
    places = book(scenario.world)
    start = scenario.start_pose(places)
    boxes = scenario.resolved_boxes(places)
    world({"cmd": "boxes", "boxes": [b.to_dict() for b in boxes]})
    print(
        f"scenario {scenario.name}: {len(scenario.legs)} leg(s), boxes {[b.name for b in boxes]},"
        f" slow past {scenario.slow_s} s",
        flush=True,
    )
    failures = 0
    for round_no in range(1, repeat + 1):
        placed = place(start, settle_s)
        print(
            f"round {round_no}: cart at ({placed['x']:+.3f}, {placed['y']:+.3f},"
            f" {placed['yaw_deg']:+.1f} deg), hull overlap {placed['overlap']}",
            flush=True,
        )
        for leg in scenario.legs:
            target = leg.resolve(places, start)
            request: dict[str, Any]
            if leg.place is not None:
                label, request = leg.place, {"cmd": "go", "place": leg.place}
            else:
                label = f"({target.x:+.2f}, {target.y:+.2f})"
                request = {"cmd": "go", "x": target.x, "y": target.y,
                           "yaw_deg": math.degrees(target.theta)}  # fmt: skip
            print(f"leg -> {label}", flush=True)
            score = drive(label, request, target, scenario.timeout_s)
            slow = scenario.slow_s is not None and score.seconds > scenario.slow_s
            print(
                f"  {'SLOW' if slow else 'ok'}: {score.seconds:.1f} s against {scenario.slow_s} s",
                flush=True,
            )
            record(score, scenario=scenario.name, round=round_no, slow=slow, tag=tag)
            failures += score.status != "SUCCEEDED"
    world({"cmd": "boxes", "boxes": []})
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run one command."""
    parser = argparse.ArgumentParser(prog="score", description="drive the sim and score it")
    parser.add_argument("command", choices=("state", "place", "boxes", "goal", "scenario"))
    parser.add_argument("args", nargs="*")
    parser.add_argument("--world", default="flat")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--settle", type=float, default=SETTLE_S)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--tag", default="")
    options = parser.parse_args(argv)
    if options.command == "state":
        print(json.dumps(world({"cmd": "state"})))
        return 0
    if options.command == "boxes":
        if options.args in ([], ["none"]):
            boxes: list[dict[str, Any]] = []
        else:
            import yaml

            boxes = [
                Box.from_dict(b).to_dict()
                for b in yaml.safe_load(Path(options.args[0]).read_text())
            ]
        print(json.dumps(world({"cmd": "boxes", "boxes": boxes})))
        return 0
    if options.command == "scenario":
        return run_scenario(Path(options.args[0]), options.repeat, options.settle, options.tag)
    places = book(options.world)
    label, target = target_of(options.args, places)
    if options.command == "place":
        print(json.dumps(place(target, settle_s=0.0)))
        return 0
    score = drive(label, goal_request(options.args), target, options.timeout)
    record(score, goal_cmd=options.args, tag=options.tag)
    return 0 if score.status == "SUCCEEDED" else 1


if __name__ == "__main__":
    sys.exit(main())
