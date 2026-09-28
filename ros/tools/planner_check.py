#!/usr/bin/env python3
"""Does the planner plan: the global costmap is updating, and one path is computed from here.

A Nav2 that reports itself active can still plan nothing: a global costmap wedged in its first
update, or one that stopped being current, answers every plan with "Costmap timed out waiting for
update" (drive 0499, 2026-09-28: 41 plans aborted while nobody re-ran the board's checks). So the
proof is the planner's own work, asked the way the navigator asks it:

1. /global_costmap/costmap (the full grid each publish, ros/params/nav2_params.yaml
   ``always_send_full_costmap``) or its /costmap_updates must deliver something NEW within
   ``--costmap-s``: the grid is latched, so its first copy proves nothing and is not counted.
2. One ComputePathToPose from the cart's pose (the planner's own, ``use_start`` false) to a
   point ``--ahead-m`` in front of it. A plan is not motion: nothing is sent to the wheels. A
   goal the planner finds occupied or unreachable is tried behind, left and right as well.

    python3 /tools/planner_check.py [--costmap-s 10] [--plan-s 20] [--ahead-m 0.5]
                                    [--planner GridBased]

Prints one line and exits 0 when both hold; 3 when the planner answered but found no path in any
of the four directions (the cart boxed in, or a costmap full of marks: a restart does not change
that); 1 for anything a restart can repair — a silent costmap, no pose in ``map``, a planner that
is absent, inactive, timed out or failed its transform. Run it on the laptop (pepin-vslam): the
pose costs a /tf subscription, which CLAUDE.md rule 20 keeps off the board.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PoseStamped
from map_msgs.msg import OccupancyGridUpdate
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

MAP_FRAME = "map"
BASE_FRAME = "base_link"
COSTMAP_TOPIC = "/global_costmap/costmap"
ACTION = "/compute_path_to_pose"
# nav2_msgs/action/ComputePathToPose error codes (Jazzy) that mean "the planner worked and the
# goal is the problem" — the next direction is tried; every other code is a broken planner.
GOAL_CODES = {204: "goal outside the map", 206: "goal occupied", 208: "no valid path"}
DIRECTIONS = (("ahead", 0.0), ("behind", math.pi), ("left", math.pi / 2), ("right", -math.pi / 2))
OK, BROKEN, BOXED = 0, 1, 3


@dataclass
class CostmapCount:
    """What reached us from the global costmap: grids (the first is the latched copy) and
    incremental updates, and the size of the last grid."""

    grids: int = 0
    updates: int = 0
    size: str = "?"

    def fresh(self) -> int:
        """Messages published after we subscribed: every update, every grid but the first."""
        return self.updates + max(self.grids - 1, 0)


def _yaw(rotation: object) -> float:
    """The heading of a quaternion in the plane, radians."""
    x = float(getattr(rotation, "x", 0.0))
    y = float(getattr(rotation, "y", 0.0))
    z = float(getattr(rotation, "z", 0.0))
    w = float(getattr(rotation, "w", 1.0))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _options(argv: list[str]) -> dict[str, str]:
    """``--name value`` pairs."""
    return {argv[i]: argv[i + 1] for i in range(0, len(argv) - 1, 2) if argv[i].startswith("--")}


def _spin_until(node: object, deadline: float, done: Callable[[], bool]) -> bool:
    """Spin until ``done()`` is true or the monotonic ``deadline`` passes; returns ``done()``."""
    while time.monotonic() < deadline:
        if done():
            return True
        rclpy.spin_once(node, timeout_sec=0.1)
    return bool(done())


def main() -> int:
    """Run the two proofs, print one line, return the exit code."""
    options = _options(sys.argv[1:])
    costmap_s = float(options.get("--costmap-s", 10))
    plan_s = float(options.get("--plan-s", 20))
    ahead = float(options.get("--ahead-m", 0.5))
    planner = options.get("--planner", "GridBased")
    rclpy.init()
    node = rclpy.create_node("pepin_planner_check")
    buffer = Buffer()
    TransformListener(buffer, node)
    count = CostmapCount()

    def on_grid(msg: OccupancyGrid) -> None:
        count.grids += 1
        count.size = f"{msg.info.width}x{msg.info.height}"

    def on_update(_msg: OccupancyGridUpdate) -> None:
        count.updates += 1

    latched = QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
    )
    node.create_subscription(OccupancyGrid, COSTMAP_TOPIC, on_grid, latched)
    node.create_subscription(OccupancyGridUpdate, f"{COSTMAP_TOPIC}_updates", on_update, 10)
    client = ActionClient(node, ComputePathToPose, ACTION)
    try:
        start = time.monotonic()
        _spin_until(node, start + costmap_s, lambda: count.fresh() > 0)
        costmap = f"costmap {count.grids} grid(s) {count.size} + {count.updates} update(s)"
        if count.fresh() == 0:
            why = "only its latched copy" if count.grids else "nothing"
            print(f"planner: BROKEN — {COSTMAP_TOPIC} sent {why} in {costmap_s:.0f} s ({costmap})")
            return BROKEN
        if not _spin_until(
            node,
            time.monotonic() + plan_s,
            lambda: buffer.can_transform(MAP_FRAME, BASE_FRAME, Time(), Duration()),
        ):
            print(f"planner: BROKEN — no {MAP_FRAME} -> {BASE_FRAME} in TF ({costmap})")
            return BROKEN
        pose = buffer.lookup_transform(MAP_FRAME, BASE_FRAME, Time()).transform
        if not client.wait_for_server(timeout_sec=5.0):
            print(f"planner: BROKEN — no {ACTION} action server ({costmap})")
            return BROKEN
        refused: list[str] = []
        for name, turn in DIRECTIONS:
            heading = _yaw(pose.rotation) + turn
            goal = ComputePathToPose.Goal()
            goal.goal = PoseStamped()
            goal.goal.header.frame_id = MAP_FRAME
            goal.goal.pose.position.x = pose.translation.x + ahead * math.cos(heading)
            goal.goal.pose.position.y = pose.translation.y + ahead * math.sin(heading)
            goal.goal.pose.orientation = pose.rotation
            goal.planner_id = planner
            goal.use_start = False
            asked = time.monotonic()
            sent = client.send_goal_async(goal)
            if not _spin_until(node, asked + plan_s, sent.done) or not sent.result().accepted:
                print(f"planner: BROKEN — the goal was not accepted (Nav2 inactive?) ({costmap})")
                return BROKEN
            result = sent.result().get_result_async()
            if not _spin_until(node, asked + plan_s, result.done):
                print(f"planner: BROKEN — no answer in {plan_s:.0f} s ({costmap})")
                return BROKEN
            answer = result.result().result
            poses = len(answer.path.poses)
            code = int(getattr(answer, "error_code", 0))
            if poses >= 2:
                print(
                    f"planner: OK — path of {poses} poses to {ahead:.2f} m {name} ({planner})"
                    f" in {time.monotonic() - asked:.1f} s; {costmap}"
                )
                return OK
            if code not in GOAL_CODES:
                message = str(getattr(answer, "error_msg", "")) or f"error code {code}"
                print(f"planner: BROKEN — {name}: {message} ({costmap})")
                return BROKEN
            refused.append(f"{name} {GOAL_CODES[code]}")
        print(
            f"planner: BOXED — it answers, but no path {ahead:.2f} m around: {', '.join(refused)}"
        )
        return BOXED
    finally:
        client.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
