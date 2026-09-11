#!/usr/bin/env python3
"""Is a navigation goal running right now? One rclpy pass, for a guard that must not guess.

Run it inside the container the action server lives in, piped in from the laptop so nothing
has to be deployed first:

    ssh root@BOARD 'docker exec -i pepin-ros /pepin_entrypoint.sh python3 - \
        navigate_to_pose navigate_through_poses' < ros/tools/nav_goal_running.py

It prints one line, one word per action:

    navigate_to_pose=no navigate_through_poses=yes

``yes`` a goal is ACCEPTED or EXECUTING, ``no`` none is, ``?`` this pass could not tell. A
caller guarding the robot against a switch it should not get while driving (ros/sensor.sh)
refuses on ``?`` exactly as it refuses on ``yes``.

Why a node and not ``ros2 topic echo --once``: an action's status is latched
(transient_local), so the echo prints the current one and exits — until no goal has run since
the server started, when there is nothing latched and the echo hangs. It hangs the same way
when the CLI is merely too slow to discover anything on a loaded board (a successful echo
measured 8.1 s there on 2026-09-11, idle), and ``timeout`` reports both as 124. One is "no
goal is running", the other is a blind guard. A subscription separates them: it knows when it
is matched to the publisher — a latched sample follows within milliseconds of that — and it
can ask the graph whether it sees any other node at all, so "this action has no server" is
not confused with "this pass never saw the graph".
"""

from __future__ import annotations

import sys
from typing import Any

NAV_ACTIONS = ("navigate_to_pose", "navigate_through_poses")
RUNNING_STATUSES = (1, 2)  # action_msgs/GoalStatus: STATUS_ACCEPTED, STATUS_EXECUTING
WINDOW_S = 8.0  # the whole pass once rclpy is up; discovery on the board is a second or two of it
MATCH_GRACE_S = 2.0  # matched to the publisher and still silent: nothing is latched, no goal ran
GRAPH_GRACE_S = 4.0  # before this, "no publisher anywhere" may still be discovery in progress


def verdict(
    running: bool, status_seen: bool, matched_for_s: float | None, graph_for_s: float
) -> str:
    """One action's answer from what the pass saw: ``yes`` (a goal is ACCEPTED or EXECUTING),
    ``no`` (none is), ``?`` (this pass could not tell, and the caller must refuse).
    ``matched_for_s`` is how long the action's status publisher has been matched (None: never
    seen), ``graph_for_s`` how long this node has seen any other node on the graph (0.0: never)."""
    if running:
        return "yes"
    if status_seen:
        return "no"  # the server answered: its newest status list carries no running goal
    if matched_for_s is not None:
        # The server is up and nothing was latched: no goal has run since it started. Under the
        # grace the sample may simply still be in flight, which is not an answer yet.
        return "no" if matched_for_s >= MATCH_GRACE_S else "?"
    if graph_for_s >= GRAPH_GRACE_S:
        return "no"  # discovery works, and this action has no server: no goal can be running
    return "?"  # nothing was seen at all; a guard that cannot see waves nothing through


def main(actions: list[str]) -> int:
    """Subscribe to every action's status, spin until each has a firm answer or ``WINDOW_S``
    runs out, print the line and return 0 (1 when ROS itself refused to come up)."""
    import time

    import rclpy
    from action_msgs.msg import GoalStatusArray
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    running = dict.fromkeys(actions, False)
    status_seen = dict.fromkeys(actions, False)
    matched_since: dict[str, float | None] = dict.fromkeys(actions, None)
    graph_since: float | None = None

    def on_status(action: str, msg: Any) -> None:
        status_seen[action] = True
        running[action] = any(s.status in RUNNING_STATUSES for s in msg.status_list)

    def answer(action: str, now: float) -> str:
        since = matched_since[action]
        return verdict(
            running[action],
            status_seen[action],
            None if since is None else now - since,
            0.0 if graph_since is None else now - graph_since,
        )

    rclpy.init()
    node = Node("pepin_nav_goal_guard")
    latched = QoSProfile(
        depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
    )  # the action's status is offered latched: the current one arrives on connection
    topics = {action: f"/{action}/_action/status" for action in actions}
    for action, topic in topics.items():
        node.create_subscription(
            GoalStatusArray, topic, lambda m, a=action: on_status(a, m), latched
        )
    start = time.monotonic()
    while True:
        rclpy.spin_once(node, timeout_sec=0.05)
        now = time.monotonic()
        for action, topic in topics.items():
            if matched_since[action] is None and node.count_publishers(topic) > 0:
                matched_since[action] = now
        if graph_since is None and len(node.get_node_names()) > 1:  # anyone but ourselves
            graph_since = now
        if now - start >= WINDOW_S or all(answer(a, now) != "?" for a in actions):
            break
    now = time.monotonic()
    print(" ".join(f"{action}={answer(action, now)}" for action in actions))
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or list(NAV_ACTIONS)))
