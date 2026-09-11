"""Turn the cart in place through a full circle, judged by the gyro, not by the clock: publish
RATE rad/s on /cmd_vel at 20 Hz until the odometry's yaw has swept TARGET_DEG (a little past
360, so the cart returns to its heading or overshoots slightly — a timed 19 s at 0.35 rad/s
came up short on the carpet), then stop. Recorded as a tape by the run recorder under the
given name. Run through ros/round.sh from the laptop; on the board itself:
  docker exec -i pepin-ros /pepin_entrypoint.sh python3 - NAME < /tools/turn_full.py
"""

import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

RATE, TARGET_DEG, MAX_S = 0.35, 372.0, 45.0
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

    def feed(self, yaw: float) -> None:
        if self.last is not None:
            d = (yaw - self.last + math.pi) % (2 * math.pi) - math.pi
            self.deg += abs(math.degrees(d))
        self.last = yaw


swept = Sweep()


def on_odom(msg: Odometry) -> None:
    q = msg.pose.pose.orientation
    swept.feed(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


node.create_subscription(Odometry, "/odometry/filtered", on_odom, 20)
time.sleep(1.0)
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
print(f"turned {swept.deg:.0f} deg in {t2 - t1:.1f} s ({name})")
node.destroy_node()
rclpy.shutdown()
