"""Board-side watch over the laptop link: a drive with no planner behind it is stopped.

Runs only on a split stack (``side:=board``). Listens to the laptop's heartbeat and, when it
falls silent during a goal for longer than the patience, cancels every navigate_to_pose goal
through the action's own cancel service. The controller and the local costmap stay on the
board, so the cart is never blind — but with no one to replan it must not keep going. The
decision lives in ``pepin.deployment.LinkWatch``; this file only feeds it and obeys.

    python3 -m pepin_bringup.link_watch
"""

from __future__ import annotations

import time

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header

from pepin.deployment import HEARTBEAT_TOPIC, LinkWatch


class LinkWatchNode(Node):
    """Cancels the running goal when the laptop's heartbeat stops."""

    def __init__(self) -> None:
        super().__init__("link_watch")
        self._watch = LinkWatch(patience_s=float(self.declare_parameter("patience_s", 2.5).value))
        self._navigating = False
        self.create_subscription(Header, HEARTBEAT_TOPIC, self._on_beat, 10)
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )  # the action's status is offered latched: the current one arrives on connection
        self.create_subscription(
            GoalStatusArray, "/navigate_to_pose/_action/status", self._on_status, latched
        )
        self._cancel = self.create_client(CancelGoal, "/navigate_to_pose/_action/cancel_goal")
        self.create_timer(0.5, self._check)
        self.get_logger().info(f"watching {HEARTBEAT_TOPIC}; a silent laptop cancels the drive")

    def _on_beat(self, _msg: Header) -> None:
        self._watch.beat(time.monotonic())

    def _on_status(self, msg: GoalStatusArray) -> None:
        self._navigating = any(
            s.status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING)
            for s in msg.status_list
        )

    def _check(self) -> None:
        if not self._watch.should_cut(self._navigating, time.monotonic()):
            return
        self.get_logger().error("laptop heartbeat lost during a drive: cancelling the goal")
        if not self._cancel.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(
                "the cancel service is not up; the base deadman is the last line"
            )
            return
        self._cancel.call_async(CancelGoal.Request())  # an all-zero goal id cancels every goal


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
