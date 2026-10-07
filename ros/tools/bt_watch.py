#!/usr/bin/env python3
"""Print the behaviour tree's decisive transitions, one line each, for as long as it runs.

Nav2's own log says that a recovery ran, not which node of the tree failed first and what the
tree did about it. This reads ``/behavior_tree_log`` (nav2_msgs/BehaviorTreeLog) and prints every
transition into FAILURE, every RUNNING -> SUCCESS, and the start of every recovery, of the nodes
that decide a drive (planning, following, the recoveries); the tree's plumbing is left out:

    bt| 512.34 ComputePathToPose: RUNNING -> FAILURE

The time is the transition's own stamp, seconds modulo 1000. One watcher per Nav2 container,
started by ros/laptop.sh nav (or by ros/goto.sh when it is missing) into
ros/maps/rec/bt_live.log, which goto streams into each goal's log:

    docker exec -d pepin-macnav /pepin_entrypoint.sh sh -c \\
        'exec python3 -u /tools/bt_watch.py >> /maps/rec/bt_live.log 2>&1'
"""

from __future__ import annotations

import contextlib
from typing import Any

import rclpy
from nav2_msgs.msg import BehaviorTreeLog

# The tree's control nodes and decorators: every tick passes through them, and their transitions
# only repeat what the nodes under them did.
QUIET = {
    "RateController",
    "PipelineSequence",
    "ReactiveSequence",
    "Sequence",
    "Fallback",
    "Inverter",
    "ControllerSelector",
    "PlannerSelector",
    "GoalCheckerSelector",
    "ForgetStaleObstacles",
    "ForgetGlobalGhosts",
    "NavigateWithReplanning",
    "ReactiveFallback",
    "RoundRobin",
    "GoalUpdated",
    "KeepPathWhileValid",
    "TruncatePath",  # the validity check's view of the plan, one line per check (2026-10-07)
}
# A node whose name holds one of these is a recovery: its start is a line of its own.
RECOVERY = ("Wait", "Spin", "BackUp", "DriveOnHeading", "Clear", "Rock", "Retreat")


def line(name: str, old: str, new: str, stamp_s: float) -> str | None:
    """The line for one transition, or ``None`` when it is not worth one."""
    if name in QUIET:
        return None
    starting_recovery = new == "RUNNING" and old == "IDLE" and any(r in name for r in RECOVERY)
    if new == "FAILURE" or (new == "SUCCESS" and old == "RUNNING") or starting_recovery:
        return f"bt| {stamp_s % 1000:7.2f} {name}: {old} -> {new}"
    return None


def main() -> None:
    """Subscribe and print until interrupted."""
    rclpy.init()
    node = rclpy.create_node("bt_watch")

    def on_log(msg: Any) -> None:
        for event in msg.event_log:
            stamp = event.timestamp.sec + event.timestamp.nanosec * 1e-9
            text = line(event.node_name, event.previous_status, event.current_status, stamp)
            if text is not None:
                print(text, flush=True)

    node.create_subscription(BehaviorTreeLog, "/behavior_tree_log", on_log, 50)
    with contextlib.suppress(KeyboardInterrupt, SystemExit):
        rclpy.spin(node)


if __name__ == "__main__":
    main()
