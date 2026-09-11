"""Turn the cart in place through a full circle, judged by the gyro, not by the clock: publish
RATE rad/s on /cmd_vel at 20 Hz until the odometry's yaw has swept TARGET_DEG (a little past
360, so the cart returns to its heading or overshoots slightly — a timed 19 s at 0.35 rad/s
came up short on the carpet), then stop. Recorded as a tape by the run recorder under the
given name. Run through ros/go.sh round from the laptop; on the board itself:
  docker exec -i pepin-ros /pepin_entrypoint.sh python3 - NAME < /tools/turn_full.py

Guards, because this writes /cmd_vel past every one of Nav2's: it refuses to turn while a
navigation goal is running (the action's latched status), it aborts when no odometry arrives
within ODOM_WAIT_S (a turn nobody measures is a turn nobody stops), and MAX_S caps the whole
turn at one circle's worth. The EKF's /odometry/filtered judges the sweep; /odom (the wheels)
stands in when the EKF is not running (no IMU).
"""

import json
import math
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

RATE, TARGET_DEG = 0.35, 372.0
MAX_S = 30.0  # 372 deg at 0.35 rad/s is 18.5 s; a turn still running at 30 s is stuck
ODOM_WAIT_S = 2.0
NAV_ACTIONS = ("navigate_to_pose", "navigate_through_poses")
name = sys.argv[1] if len(sys.argv) > 1 else "turn"

rclpy.init()
node = Node("turn_full")
cmd = node.create_publisher(Twist, "/cmd_vel", 10)
run = node.create_publisher(String, "/pepin/run", 10)


class Sweep:
    """How far the yaw has swept since the last reset, from consecutive odometry headings."""

    def __init__(self) -> None:
        self.deg = 0.0
        self.last: float | None = None
        self.source: str | None = None  # which odometry judges: the EKF once heard, else wheels
        self.heard = 0

    def feed(self, source: str, yaw: float) -> None:
        if self.source is None or (source == "filtered" and self.source == "odom"):
            self.source, self.last = source, None  # the EKF takes over from the wheels
        if source != self.source:
            return
        self.heard += 1
        if self.last is not None:
            d = (yaw - self.last + math.pi) % (2 * math.pi) - math.pi
            self.deg += abs(math.degrees(d))
        self.last = yaw


swept = Sweep()
navigating = {action: False for action in NAV_ACTIONS}


def yaw_of(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def on_status(action: str, msg: GoalStatusArray) -> None:
    navigating[action] = any(
        s.status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING)
        for s in msg.status_list
    )


latched = QoSProfile(
    depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
)  # the action's status is offered latched: the current one arrives on connection
for action in NAV_ACTIONS:
    node.create_subscription(
        GoalStatusArray, f"/{action}/_action/status", lambda m, a=action: on_status(a, m), latched
    )
node.create_subscription(
    Odometry, "/odometry/filtered", lambda m: swept.feed("filtered", yaw_of(m)), 20
)
node.create_subscription(Odometry, "/odom", lambda m: swept.feed("odom", yaw_of(m)), 20)
t0 = time.time()
while time.time() - t0 < ODOM_WAIT_S:  # the status and the first odometry arrive here
    rclpy.spin_once(node, timeout_sec=0.05)
if any(navigating.values()):
    running = ", ".join(a for a, on in navigating.items() if on)
    print(f"refused: a navigation goal is running ({running}); ros/go.sh cancel first")
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(2)
if swept.heard == 0:
    print(f"aborted: no odometry on /odometry/filtered or /odom within {ODOM_WAIT_S:.0f} s")
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(3)
print(f"judging the sweep by /{'odometry/filtered' if swept.source == 'filtered' else 'odom'}")
run.publish(String(data=json.dumps({"cmd": "start", "name": name})))
t0 = time.time()
while time.time() - t0 < 4.0:  # rest before, on the tape
    rclpy.spin_once(node, timeout_sec=0.05)
swept.deg = 0.0
t1 = time.time()
while swept.deg < TARGET_DEG and time.time() - t1 < MAX_S:
    tw = Twist()
    tw.angular.z = RATE
    cmd.publish(tw)
    rclpy.spin_once(node, timeout_sec=0.05)
cmd.publish(Twist())
t2 = time.time()
while time.time() - t2 < 4.0:  # rest after
    rclpy.spin_once(node, timeout_sec=0.05)
run.publish(String(data=json.dumps({"cmd": "stop"})))
time.sleep(0.5)
verdict = "" if swept.deg >= TARGET_DEG else f" (STOPPED BY THE {MAX_S:.0f} s CAP)"
print(f"turned {swept.deg:.0f} deg in {t2 - t1:.1f} s ({name}){verdict}")
node.destroy_node()
rclpy.shutdown()
