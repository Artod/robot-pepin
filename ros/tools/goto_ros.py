#!/usr/bin/env python3
"""Send the robot somewhere through Nav2 and report progress — the goal client we drive with.

Runs inside the container (rclpy + nav2_simple_commander):

    goto_ros.py X Y [YAW_DEG]     drive to map coordinates and print feedback until done
    goto_ros.py home              drive to the map origin, facing +x (the marked start spot)
    goto_ros.py seed X Y [YAW]    tell AMCL where the robot was put down by hand
    goto_ros.py cancel            cancel the current navigation task
    goto_ros.py mark NAME         remember where the robot stands now as place NAME
    goto_ros.py NAME              drive to a remembered place
    goto_ros.py places            list the remembered places
Places live in the file named by --places (default /maps/places.yaml), one per map.

A goal pose published once on /goal_pose can be lost to discovery timing and gives
no feedback; the action client here waits for Nav2, watches the task and prints
distance remaining, recoveries and the final result.
"""

import json
import math
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

LOST_FIT, LOST_FOR_S = 0.30, 15.0  # fit below this for this long...
LOST_TRAVEL_M = 1.0  # ...while the wheels carried it this far: that is driving blind. Spinning on a
# stuck wheel with a lost fit is not — the recoveries (odom frame) can still work it free.


def pose(nav: BasicNavigator, x: float, y: float, yaw_deg: float) -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = "map"
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.orientation.z = math.sin(math.radians(yaw_deg) / 2.0)
    p.pose.orientation.w = math.cos(math.radians(yaw_deg) / 2.0)
    return p


def note(nav: BasicNavigator, text: str) -> None:
    """Say it here and on /pepin/note, which the relocalizer copies into the board log."""
    print(text, flush=True)
    pub = nav.create_publisher(String, "/pepin/note", 1)
    pub.publish(String(data=text))
    time.sleep(0.2)  # let the message leave before a cancel or an exit


def where_am_i(nav: BasicNavigator) -> tuple[float, float, float, float] | None:
    """The tracker's pose (x, y, yaw_deg) and its scan-to-map fit; None when it does not answer."""
    client = nav.create_client(Trigger, "/where_am_i")
    if not client.wait_for_service(timeout_sec=5.0):
        return None
    future = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(nav, future, timeout_sec=10.0)
    result = future.result()
    if result is None:
        return None
    text = result.message  # "x +1.23 m, y -0.45 m, yaw +12 deg; scan-to-map fit 0.57 (...)"
    try:
        px = float(text.split("x ")[1].split(" m")[0])
        py = float(text.split("y ")[1].split(" m")[0])
        pyaw = float(text.split("yaw ")[1].split(" deg")[0])
        fit = float(text.split("fit ")[1].split(" ")[0])
    except (IndexError, ValueError):
        return None
    return px, py, pyaw, fit


def load_places(path: Path) -> dict[str, dict[str, float]]:
    """Named places of this map: {name: {x, y, yaw_deg, fit}}; an absent file is an empty book."""
    if not path.exists():
        return {}
    data: dict[str, dict[str, float]] = json.loads(path.read_text())
    return data


def mark_place(nav: BasicNavigator, path: Path, name: str) -> str:
    """Store the robot's current tracked pose under ``name``; refuses a weak fit."""
    pose = where_am_i(nav)
    if pose is None:
        return "cannot mark: the relocalizer is not answering"
    px, py, pyaw, fit = pose
    if fit < 0.45:
        return (
            f"NOT marked: the fit here is only {fit:.2f}; stand still 2 s, or run relocalize first"
        )
    places = load_places(path)
    places[name] = {
        "x": round(px, 3),
        "y": round(py, 3),
        "yaw_deg": round(pyaw, 1),
        "fit": round(fit, 2),
    }
    path.write_text(json.dumps(places, indent=2, sort_keys=True) + "\n")
    return (
        f"marked {name!r} at x {px:+.2f} m, y {py:+.2f} m, yaw {pyaw:+.0f} deg "
        f"(fit {fit:.2f}) in {path}"
    )


def ensure_localized(nav: BasicNavigator) -> bool:
    """A weak fit before driving (the robot was carried, or just switched on) gets one whole-map
    search first; drive only when the scan fits the map."""
    pose = where_am_i(nav)
    if pose is not None and pose[3] >= 0.45:
        print(f"localized: fit {pose[3]:.2f}", flush=True)
        return True
    print(
        f"fit {pose[3] if pose else float('nan'):.2f}: searching the whole map first...", flush=True
    )
    client = nav.create_client(Trigger, "/relocalize")
    if not client.wait_for_service(timeout_sec=5.0):
        return False
    future = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(nav, future, timeout_sec=60.0)
    result = future.result()
    print(result.message if result else "no answer from /relocalize", flush=True)
    time.sleep(2.0)
    pose = where_am_i(nav)
    return pose is not None and pose[3] >= 0.45


def describe(x: float, y: float, yaw_deg: float, home: dict[str, float] | None = None) -> str:
    """The goal in words, relative to home: the place named home, else the map origin."""
    hx, hy, hyaw = (
        (home["x"], home["y"], math.radians(home["yaw_deg"])) if home else (0.0, 0.0, 0.0)
    )
    dx, dy = x - hx, y - hy
    ahead = math.cos(hyaw) * dx + math.sin(hyaw) * dy
    left = -math.sin(hyaw) * dx + math.cos(hyaw) * dy
    yaw_deg = (yaw_deg - math.degrees(hyaw) + 180.0) % 360.0 - 180.0
    parts = []
    if abs(ahead) >= 0.05:
        parts.append(f"{abs(ahead):.1f} m {'ahead of' if ahead > 0 else 'behind'} home")
    if abs(left) >= 0.05:
        parts.append(f"{abs(left):.1f} m to the {'left' if left > 0 else 'right'} of it")
    where = ", ".join(parts) or "at home"
    facing = (
        "facing as at the start"
        if abs(yaw_deg) < 5
        else f"facing {yaw_deg:+.0f} deg from the start heading"
    )
    return f"goal in words: {where}, {facing} (left/right as seen from the start heading)"


def arrival(nav: BasicNavigator, x: float, y: float, yaw_deg: float) -> str:
    """Where the tracker says the robot ended, against the goal."""
    pose = where_am_i(nav)
    if pose is None:
        return "arrival: the relocalizer is not answering"
    px, py, pyaw, fit = pose
    off = math.hypot(px - x, py - y)
    turn = (pyaw - yaw_deg + 180.0) % 360.0 - 180.0
    return (
        f"arrival: x {px:+.2f} m, y {py:+.2f} m, yaw {pyaw:+.0f} deg, fit {fit:.2f}\n"
        f"         {off:.2f} m from the goal, heading off by {turn:+.0f} deg"
    )


def main() -> None:
    args = sys.argv[1:]
    places_path = Path("/maps/places.yaml")
    if len(args) >= 2 and args[0] == "--places":
        places_path, args = Path(args[1]), args[2:]
    if not args:
        print(__doc__)
        sys.exit(2)
    rclpy.init()
    nav = BasicNavigator()
    try:
        if args[0] == "mark":
            print(mark_place(nav, places_path, args[1]))
            return
        if args[0] == "places":
            for name, p in sorted(load_places(places_path).items()):
                print(
                    f"{name:12s} x {p['x']:+.2f} m, y {p['y']:+.2f} m, "
                    f"yaw {p['yaw_deg']:+.0f} deg (fit {p['fit']:.2f})"
                )
            return
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
        name = None
        home = load_places(places_path).get("home")
        if args[0] == "home":
            x, y, yaw = (home["x"], home["y"], home["yaw_deg"]) if home else (0.0, 0.0, 0.0)
            name = "home"
        elif not args[0].lstrip("-").replace(".", "", 1).isdigit():
            places = load_places(places_path)
            if args[0] not in places:
                known = ", ".join(sorted(places)) or "none yet (goto_ros.py mark NAME)"
                print(f"unknown place {args[0]!r}; known: {known}")
                sys.exit(2)
            name, place = args[0], places[args[0]]
            x, y, yaw = place["x"], place["y"], place["yaw_deg"]
        else:
            x, y = float(args[0]), float(args[1])
            yaw = float(args[2]) if len(args) > 2 else 0.0
        nav.waitUntilNav2Active(
            localizer="robot_localization"
        )  # no AMCL, the tracker owns the frame
        print(describe(x, y, yaw, home), flush=True)
        if not ensure_localized(nav):
            print("not localized: not driving. Stand the robot still for 5 s and try again.")
            sys.exit(1)
        lost_since: list[float | None] = [None]
        odom_xy: list[tuple[float, float] | None] = [None]
        lost_at_xy: list[tuple[float, float] | None] = [None]

        def on_fit(msg: Float32) -> None:
            if msg.data < LOST_FIT:
                if lost_since[0] is None:
                    lost_since[0], lost_at_xy[0] = time.monotonic(), odom_xy[0]
            else:
                lost_since[0] = None

        def on_odom(msg: Odometry) -> None:
            odom_xy[0] = (msg.pose.pose.position.x, msg.pose.pose.position.y)

        def travelled_while_lost() -> float:
            if lost_at_xy[0] is None or odom_xy[0] is None:
                return 0.0
            return math.hypot(odom_xy[0][0] - lost_at_xy[0][0], odom_xy[0][1] - lost_at_xy[0][1])

        nav.create_subscription(Float32, "/localization_fit", on_fit, 10)
        nav.create_subscription(Odometry, "/odom", on_odom, 10)
        nav.goToPose(pose(nav, x, y, yaw))
        print(
            f"goal {name + ' ' if name else ''}({x:.2f}, {y:.2f}) yaw {yaw:.0f} deg accepted",
            flush=True,
        )
        started = time.monotonic()
        last = 0.0
        while not nav.isTaskComplete():
            fb = nav.getFeedback()
            now = time.monotonic()
            if (
                lost_since[0] is not None
                and now - lost_since[0] > LOST_FOR_S
                and travelled_while_lost() > LOST_TRAVEL_M
            ):
                note(
                    nav,
                    f"goto: localization lost for {now - lost_since[0]:.0f} s while the wheels "
                    f"travelled {travelled_while_lost():.1f} m: cancelling, not driving blind",
                )
                nav.cancelTask()
                break
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
        print(arrival(nav, x, y, yaw), flush=True)
    except KeyboardInterrupt:
        # The goal lives on the board's action server, not in this client: dying silently
        # would leave Nav2 driving toward it (2026-09-05: Ctrl-C on the laptop, robot kept going).
        note(nav, "goto: interrupted by the operator, cancelling the goal")
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
