"""Board-side watch over the laptop link: a drive with no planner behind it is stopped.

Runs only on a split stack (``side:=board``). Listens to the laptop's heartbeat and, when it
falls silent during a goal for longer than the patience, cancels every navigate_to_pose and
navigate_through_poses goal through each action's own cancel service (both trees load, and a
tour is a through-poses goal). The controller and the local costmap stay on the
board, so the cart is never blind — but with no one to replan it must not keep going. The
decision lives in ``pepin.deployment.LinkWatch``; this file only feeds it and obeys.

    python3 -m pepin_bringup.link_watch
"""

from __future__ import annotations

import time

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from rclpy.client import Client
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header

from pepin.deployment import BOARD_ACTIONS, HEARTBEAT_TOPIC, LinkWatch


class LinkWatchNode(Node):
    """Cancels the running goal when the laptop's heartbeat stops."""

    def __init__(self) -> None:
        super().__init__("link_watch")
        self._watch = LinkWatch(patience_s=float(self.declare_parameter("patience_s", 2.5).value))
        self._navigating: dict[str, bool] = dict.fromkeys(BOARD_ACTIONS, False)
        self.create_subscription(Header, HEARTBEAT_TOPIC, self._on_beat, 10)
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )  # the action's status is offered latched: the current one arrives on connection
        self._cancel: dict[str, Client] = {}
        for action in BOARD_ACTIONS:  # navigate_to_pose and navigate_through_poses
            self.create_subscription(
                GoalStatusArray,
                f"/{action}/_action/status",
                lambda msg, action=action: self._on_status(action, msg),
                latched,
            )
            self._cancel[action] = self.create_client(CancelGoal, f"/{action}/_action/cancel_goal")
        self.create_timer(0.5, self._check)
        self.get_logger().info(f"watching {HEARTBEAT_TOPIC}; a silent laptop cancels the drive")

    def _on_beat(self, _msg: Header) -> None:
        self._watch.beat(time.monotonic())

    def _on_status(self, action: str, msg: GoalStatusArray) -> None:
        self._navigating[action] = any(
            s.status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING)
            for s in msg.status_list
        )

    def _check(self) -> None:
        if not self._watch.should_cut(any(self._navigating.values()), time.monotonic()):
            return
        self.get_logger().error("laptop heartbeat lost during a drive: cancelling the goal")
        sent = True
        for action, running in self._navigating.items():
            if not running:
                continue
            if not self._cancel[action].wait_for_service(timeout_sec=1.0):
                self.get_logger().error(
                    f"the {action} cancel service is not up; the base deadman is the last line"
                )
                sent = False
                continue
            # an all-zero goal id cancels every goal of the action
            self._cancel[action].call_async(CancelGoal.Request())
        if sent:
            self._watch.cut_sent()  # this outage is handled; a missed service is asked again


def main() -> None:
    rclpy.init()
    node = LinkWatchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
