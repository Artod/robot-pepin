"""Move the cart by measured segments, without the planner: straight legs judged by the
odometry's travelled distance, turns judged by its swept heading, /cmd_vel at 20 Hz, recorded
as one tape by the run recorder under the given name. Meant for calibration drives where Nav2's
planning (its spins, reversals and recoveries) is exactly what must not happen. Run from the
laptop through ros/go.sh move; on the board itself:
  docker exec -i pepin-ros /pepin_entrypoint.sh python3 - NAME SEG... < /tools/move.py
Segments, executed in order with a short rest between: ``f0.40`` drives 0.40 m straight
(negative = backwards), ``t90`` turns 90 deg to the left (negative = right).
Guards, because this writes /cmd_vel past every one of Nav2's: it refuses to move while a
navigation goal is running (the action's latched status); it aborts when no odometry arrives
within ODOM_WAIT_S; each straight leg stops the moment the lidar reads anything nearer than
STOP_M inside a +-SECTOR_DEG cone in the direction of travel — beams are turned into the cart's
own axes through the laser's mount from TF, and without that transform nothing moves — and a
forward leg is refused outright when the cone's nearest return is not at least
MARGIN_M beyond the leg's length; every leg is capped in time at twice its nominal duration.
"""

import json
import math
import sys
import time

import rclpy
import rclpy.time
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

SPEED, RATE = 0.15, 0.35  # m/s straight, rad/s turning
STOP_M, MARGIN_M, SECTOR_DEG = 0.40, 0.30, 25.0
ODOM_WAIT_S, REST_S = 2.0, 2.0
NAV_ACTIONS = ("navigate_to_pose", "navigate_through_poses")

name = sys.argv[1] if len(sys.argv) > 1 else "move"
segments = sys.argv[2:]
if not segments:
    print("usage: NAME SEG... (f0.40 = 0.40 m straight, t90 = 90 deg left)")
    sys.exit(2)

rclpy.init()
node = Node("move")
cmd = node.create_publisher(Twist, "/cmd_vel", 10)
run = node.create_publisher(String, "/pepin/run", 10)


class Travel:
    """Distance and signed heading swept since the last reset, from consecutive odometry
    poses; the EKF's odometry judges once heard, the wheels' until then."""

    def __init__(self) -> None:
        self.m = 0.0
        self.deg = 0.0
        self.last: tuple[float, float, float] | None = None
        self.source: str | None = None
        self.heard = 0

    def feed(self, source: str, x: float, y: float, yaw: float) -> None:
        if self.source is None or (source == "filtered" and self.source == "odom"):
            self.source, self.last = source, None
        if source != self.source:
            return
        self.heard += 1
        if self.last is not None:
            lx, ly, lyaw = self.last
            self.m += math.hypot(x - lx, y - ly)
            self.deg += math.degrees((yaw - lyaw + math.pi) % (2 * math.pi) - math.pi)
        self.last = (x, y, yaw)

    def reset(self) -> None:
        self.m, self.deg = 0.0, 0.0


class Cone:
    """The nearest lidar return inside +-SECTOR_DEG of a direction IN THE CART'S AXES (0 = front,
    pi = back).

    The scan's own angle 0 is not the cart's front: the LD19 hangs upside down and turned 87.5
    degrees (config/lidar.json), so a beam is first turned through the laser's mount, read once
    from TF. From 2026-09-15 to 2026-09-19 this guard took the scan's angles for the cart's and so
    watched the cart's SIDES while it drove forwards and backwards; it was found when the "front"
    reading shrank after a metre in reverse."""

    def __init__(self) -> None:
        self.scan: LaserScan | None = None
        self.mount: tuple[float, float, float, float] | None = None  # quaternion base <- laser

    def _in_base(self, angle: float) -> float:
        """A beam's direction in the laser's frame as an angle in the cart's frame."""
        x, y, z, w = self.mount  # type: ignore[misc]
        vx, vy = math.cos(angle), math.sin(angle)  # the beam lies in the laser's own plane
        bx = (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - z * w) * vy
        by = 2 * (x * y + z * w) * vx + (1 - 2 * (x * x + z * z)) * vy
        return math.atan2(by, bx)

    def nearest(self, direction: float) -> float | None:
        s = self.scan
        if s is None or self.mount is None:
            return None
        best = None
        for i, r in enumerate(s.ranges):
            if not (s.range_min < r < s.range_max) or math.isnan(r):
                continue
            a = self._in_base(s.angle_min + i * s.angle_increment)
            d = (a - direction + math.pi) % (2 * math.pi) - math.pi
            if abs(d) <= math.radians(SECTOR_DEG) and (best is None or r < best):
                best = r
        return best


travel, cone = Travel(), Cone()
navigating = {action: False for action in NAV_ACTIONS}


def feed_odom(source: str, msg: Odometry) -> None:
    q = msg.pose.pose.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    travel.feed(source, msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)


def on_status(action: str, msg: GoalStatusArray) -> None:
    navigating[action] = any(
        s.status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING)
        for s in msg.status_list
    )


def on_scan(msg: LaserScan) -> None:
    cone.scan = msg


def spin(seconds: float) -> None:
    t = time.time()
    while time.time() - t < seconds:
        rclpy.spin_once(node, timeout_sec=0.05)


def finish(code: int, why: str) -> None:
    cmd.publish(Twist())
    print(why)
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(code)


latched = QoSProfile(
    depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
)
for action in NAV_ACTIONS:
    node.create_subscription(
        GoalStatusArray, f"/{action}/_action/status", lambda m, a=action: on_status(a, m), latched
    )
node.create_subscription(Odometry, "/odometry/filtered", lambda m: feed_odom("filtered", m), 20)
node.create_subscription(Odometry, "/odom", lambda m: feed_odom("odom", m), 20)
node.create_subscription(LaserScan, "/scan", on_scan, 10)
tf_buffer = Buffer()
tf_listener = TransformListener(tf_buffer, node)
spin(ODOM_WAIT_S)
# Discovery under rmw_zenoh on the board answers a fresh node in up to 10 s (interest timeouts,
# 2026-09-23): keep listening until the odometry AND the scan are both heard, up to 20 s.
for _extra in range(30):
    if travel.heard and cone.scan is not None:
        break
    spin(0.5)
if any(navigating.values()):
    finish(2, "refused: a navigation goal is running; ros/go.sh cancel first")
if travel.heard == 0:
    finish(3, f"aborted: no odometry on /odometry/filtered or /odom within {ODOM_WAIT_S:.0f} s")
if cone.scan is None:
    finish(3, "aborted: no /scan within the wait; the lidar guard cannot run")
assert cone.scan is not None  # finish() above does not return
laser_frame = cone.scan.header.frame_id
try:
    # A static edge is a one-shot transient-local sample; a fresh listener under rmw_zenoh can
    # miss the first query (2026-09-23: the guard refused every turn after a board restart while
    # the edge was there), so the lookup is retried for a few seconds before it counts as absent.
    for _attempt in range(12):
        if tf_buffer.can_transform("base_link", laser_frame, rclpy.time.Time()):
            break
        spin(0.5)
    edge = tf_buffer.lookup_transform("base_link", laser_frame, rclpy.time.Time())
    q = edge.transform.rotation
    cone.mount = (q.x, q.y, q.z, q.w)
except Exception as error:
    finish(
        3,
        f"aborted: no base_link <- {laser_frame} in TF ({error}); a guard that"
        " does not know where the lidar points cannot run",
    )
front, back = cone.nearest(0.0), cone.nearest(math.pi)
print(
    f"judging by /{'odometry/filtered' if travel.source == 'filtered' else 'odom'};"
    f" lidar cone: front {front if front is None else round(front, 2)} m,"
    f" back {back if back is None else round(back, 2)} m"
)
for seg in segments:  # every leg is checked before the wheels turn
    if seg[0] not in "ft" or not seg[1:].lstrip("-").replace(".", "", 1).isdigit():
        finish(2, f"bad segment {seg!r}: f<metres> or t<degrees>")
    if seg[0] == "f":
        metres = float(seg[1:])
        ahead = cone.nearest(0.0 if metres > 0 else math.pi)
        if ahead is not None and ahead < abs(metres) + MARGIN_M:
            finish(
                4,
                f"refused: {seg} would end {ahead - abs(metres):.2f} m from the nearest return"
                f" ({ahead:.2f} m in the cone), under the {MARGIN_M:.2f} m margin",
            )

run.publish(String(data=json.dumps({"cmd": "start", "name": name})))
spin(REST_S)
report = []
for seg in segments:
    value = float(seg[1:])
    travel.reset()
    t1 = time.time()
    stopped = ""
    if seg[0] == "f":
        cap = 2.0 * abs(value) / SPEED + 1.0
        direction = 0.0 if value > 0 else math.pi
        while travel.m < abs(value) and time.time() - t1 < cap:
            near = cone.nearest(direction)
            if near is not None and near < STOP_M:
                stopped = f" STOPPED: lidar {near:.2f} m in the cone"
                break
            tw = Twist()
            tw.linear.x = math.copysign(SPEED, value)
            cmd.publish(tw)
            rclpy.spin_once(node, timeout_sec=0.05)
        cmd.publish(Twist())
        done = f"{seg}: drove {travel.m:.3f} m in {time.time() - t1:.1f} s"
    else:
        cap = 2.0 * abs(math.radians(value)) / RATE + 1.0
        while abs(travel.deg) < abs(value) and time.time() - t1 < cap:
            tw = Twist()
            tw.angular.z = math.copysign(RATE, value)
            cmd.publish(tw)
            rclpy.spin_once(node, timeout_sec=0.05)
        cmd.publish(Twist())
        done = f"{seg}: turned {travel.deg:+.1f} deg in {time.time() - t1:.1f} s"
    if not stopped and time.time() - t1 >= cap:
        stopped = " (STOPPED BY THE TIME CAP)"
    report.append(done + stopped)
    print(report[-1])
    spin(REST_S)
    if stopped:
        break
run.publish(String(data=json.dumps({"cmd": "stop"})))
time.sleep(0.5)
print(f"done ({name}): " + "; ".join(report))
node.destroy_node()
rclpy.shutdown()
