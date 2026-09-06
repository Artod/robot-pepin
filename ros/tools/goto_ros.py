#!/usr/bin/env python3
"""Send the robot somewhere through Nav2 and report progress — the goal client we drive with.

Runs inside the container (rclpy + nav2_simple_commander):

    goto_ros.py X Y [YAW_DEG]     drive to map coordinates and print feedback until done
    goto_ros.py home              drive to the map origin, facing +x (the marked start spot)
    goto_ros.py seed X Y [YAW]    tell AMCL where the robot was put down by hand
    goto_ros.py cancel            cancel the current navigation task

A goal pose published once on /goal_pose can be lost to discovery timing and gives
no feedback; the action client here waits for Nav2, watches the task and prints
distance remaining, recoveries and the final result.
"""

import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult


def pose(nav: BasicNavigator, x: float, y: float, yaw_deg: float) -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = "map"
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.orientation.z = math.sin(math.radians(yaw_deg) / 2.0)
    p.pose.orientation.w = math.cos(math.radians(yaw_deg) / 2.0)
    return p


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)
    rclpy.init()
    nav = BasicNavigator()
    try:
        if args[0] == "cancel":
            nav.cancelTask()
            print("cancel requested")
            return
        if args[0] == "seed":
            x, y = float(args[1]), float(args[2])
            yaw = float(args[3]) if len(args) > 3 else 0.0
            nav.setInitialPose(pose(nav, x, y, yaw))
            time.sleep(1.0)
            print(f"AMCL seeded at ({x:.2f}, {y:.2f}) yaw {yaw:.0f} deg")
            return
        if args[0] == "home":
            x, y, yaw = 0.0, 0.0, 0.0
        else:
            x, y = float(args[0]), float(args[1])
            yaw = float(args[2]) if len(args) > 2 else 0.0
        nav.waitUntilNav2Active(localizer="amcl")
        nav.goToPose(pose(nav, x, y, yaw))
        print(f"goal ({x:.2f}, {y:.2f}) yaw {yaw:.0f} deg accepted", flush=True)
        started = time.monotonic()
        last = 0.0
        while not nav.isTaskComplete():
            fb = nav.getFeedback()
            now = time.monotonic()
            if fb is not None and now - last >= 2.0:
                last = now
                print(
                    f"  t+{now - started:5.1f}s  {fb.distance_remaining:5.2f} m left,"
                    f" recoveries {fb.number_of_recoveries}",
                    flush=True,
                )
            time.sleep(0.2)
        result = nav.getResult()
        name = {TaskResult.SUCCEEDED: "SUCCEEDED", TaskResult.CANCELED: "CANCELED"}.get(
            result, "FAILED"
        )
        print(f"result: {name} after {time.monotonic() - started:.0f} s")
    except KeyboardInterrupt:
        # The goal lives on the board's action server, not in this client: dying silently
        # would leave Nav2 driving toward it (2026-09-05: Ctrl-C on the laptop, robot kept going).
        print("\ninterrupted: cancelling the navigation task...", flush=True)
        nav.cancelTask()
        deadline = time.monotonic() + 5.0
        while not nav.isTaskComplete() and time.monotonic() < deadline:
            time.sleep(0.1)
        print("cancelled" if nav.isTaskComplete() else "cancel NOT confirmed — use ros/stop.sh")
    finally:
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
