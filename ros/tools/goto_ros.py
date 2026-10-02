#!/usr/bin/env python3
"""Nav2's side of ros/goto.sh: the cancel fallback, marks, places and the RTAB-Map seed.

Runs inside the Nav2 container (rclpy + nav2_simple_commander); drives go through the goal
server (pepin.goal_link), never through here:

    goto_ros.py cancel            cancel every goal on Nav2's navigators
    goto_ros.py seed X Y [YAW]    tell RTAB-Map where the robot stands
    goto_ros.py mark NAME         remember where the robot stands now as place NAME
    goto_ros.py places            list the remembered places

A PLACE LIVES IN THE GRAPH. The map is RTAB-Map's loop-closed graph and it BENDS when a loop
closes, so a place is the pose the cart had relative to a labelled graph node:
pepin_bringup.places keeps those resolved into coordinates on the latched ``/places`` and takes a
mark on ``/places/mark``. ``places`` prints the graph's book and, beside it, any coordinates in
the file named by --places (default /maps/places.yaml), which do not follow the graph.
"""

import contextlib
import json
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import rclpy
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_simple_commander.robot_navigator import BasicNavigator
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from pepin.places import (
    MARK_TOPIC,
    MARKED_TOPIC,
    PLACES_TOPIC,
    Place,
    places_from_json,
)
from pepin.watch import PLACEMENT_TOPIC, Placement

# How long the latched placement word is given to arrive over the transport.
PLACEMENT_WAIT_S = 3.0
# A cancel must be confirmed inside the patience ros/goto.sh gives it (timeout 5): the operator
# who typed "cancel" is watching the cart move. It is the budget for the WHOLE cancel, every
# navigator in it: spent per action it was 9 s (a discovery wait plus a spin, twice) under a
# shell timeout of 5, and the second navigator was never asked at all.
CANCEL_CONFIRM_S = 30.0  # as pepin.goal_link's, and for its reason
NAV_ACTIONS = ("navigate_to_pose", "navigate_through_poses")
# Where RTAB-Map (the node /rtabmap/rtabmap) takes an operator's pose in localisation mode.
RTABMAP_INITIAL_POSE = "/rtabmap/initialpose"
# How long the latched /places is given to land. A latched publisher delivers as soon as the two
# endpoints match, so this covers discovery over the bridge; 2 s missed the book under load
# (2026-09-24).
PLACES_WAIT_S = 10.0
# ...and how long a MARK is given to be answered. Longer, because it is the whole round trip: the
# request over the bridge, three of RTAB-Map's own services on the laptop, a file written, and the
# answer back.
MARK_WAIT_S = 8.0


def cancel_all(node: Node) -> str:
    """Cancel every goal on the board's navigators and say what came of it.

    Never through the commander's ``cancelTask``: that cancels the goal THIS PROCESS sent, and a
    process started to type "cancel" has sent none — it held no goal handle, cancelled nothing,
    printed "cancel requested" and then died in rclpy's teardown ("the given context is not
    valid", 2026-09-15), so the only stop that ever worked was ros/stop.sh. An action server
    also answers a plain service, ``<action>/_action/cancel_goal``, and a request with a zero
    goal id and a zero stamp means EVERY goal: no handle needed, and the answer says how many
    goals are cancelling.

    :data:`CANCEL_CONFIRM_S` is one deadline for all of them, shared: each navigator gets an
    equal share of what is LEFT (half of it to find the service, the rest to be answered), so
    one absent server cannot spend the patience the operator's shell gives the whole command.
    """
    said = []
    deadline = time.monotonic() + CANCEL_CONFIRM_S
    for index, action in enumerate(NAV_ACTIONS):
        share = max(deadline - time.monotonic(), 0.0) / (len(NAV_ACTIONS) - index)
        if share <= 0.0:
            said.append(f"{action}: not asked — the {CANCEL_CONFIRM_S:.0f} s was spent above")
            continue
        client = node.create_client(CancelGoal, f"/{action}/_action/cancel_goal")
        if not client.wait_for_service(timeout_sec=share / 2.0):
            said.append(f"{action}: no server answered")
            continue
        future = client.call_async(CancelGoal.Request())
        rclpy.spin_until_future_complete(
            node, future, timeout_sec=max(deadline - time.monotonic(), 0.0)
        )
        answer = future.result()
        if answer is None:
            said.append(f"{action}: NOT confirmed in {CANCEL_CONFIRM_S:.0f} s — use ros/stop.sh")
            continue
        codes = {0: "accepted", 1: "rejected", 2: "no such goal", 3: "the goal had already ended"}
        outcome = codes.get(int(answer.return_code), str(answer.return_code))
        said.append(f"{action}: {outcome}, {len(answer.goals_canceling)} cancelling")
    return "cancel — " + "; ".join(said)


def pose(nav: BasicNavigator, x: float, y: float, yaw_deg: float) -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = "map"
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.orientation.z = math.sin(math.radians(yaw_deg) / 2.0)
    p.pose.orientation.w = math.cos(math.radians(yaw_deg) / 2.0)
    return p


def guarded(what: str, step: Callable[[], Any]) -> bool:
    """Run one step of a shutdown and say whether it worked; never raise, so the next step runs."""
    try:
        step()
        return True
    except Exception as exc:  # a shutdown step may fail; the next one must still run
        print(f"!! {what} failed: {exc.__class__.__name__}: {exc}", flush=True)
        return False


def load_places(path: Path) -> dict[str, dict[str, float]]:
    """Named places of this map: {name: {x, y, yaw_deg, fit}}; an absent file is an empty book."""
    if not path.exists():
        return {}
    data: dict[str, dict[str, float]] = json.loads(path.read_text())
    return data


class Vocabulary:
    """The room's places as the GRAPH answers for them: the latched ``/places``.

    Each is a pose relative to a labelled node, so it rides the node when the graph bends, while a
    coordinate written into a file stays where the room used to be.
    """

    def __init__(self, nav: BasicNavigator, path: Path) -> None:
        self._nav = nav
        self._path = path
        self._graph: dict[str, Place] = {}
        self._heard = False
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        nav.create_subscription(String, PLACES_TOPIC, self._on_places, latched)

    def _on_places(self, msg: String) -> None:
        self._graph, self._heard = places_from_json(msg.data), True

    def wait(self, seconds: float = PLACES_WAIT_S) -> None:
        """Spin until the latched vocabulary lands or the patience runs out. Short on purpose: a
        latched publisher delivers on match, so what is not here quickly is not coming — the laptop
        is asleep, or this room has no graph book yet."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self._heard:
            rclpy.spin_once(self._nav, timeout_sec=0.1)

    def graph_places(self) -> dict[str, Place]:
        """What the graph answered with, for ``places`` to print beside the file's own."""
        return dict(self._graph)


def ask_mark(nav: BasicNavigator, name: str, timeout_s: float = MARK_WAIT_S) -> str:
    """Ask pepin_bringup.places on the laptop to mark where the cart stands as ``name``, and say
    what came of it.

    A topic and not a service because this process runs in the board's container and services do
    not cross the bridge here. The request carries an id of its own and the answer echoes it, which
    is what tells this mark's answer from a latched one of an earlier mark — the answer topic is
    latched so it survives a WiFi hiccup, and a latched message is by definition an old one until
    the id matches.
    """
    answers: list[dict[str, Any]] = []
    latched = QoSProfile(
        depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
    )

    def heard(msg: String) -> None:
        with contextlib.suppress(ValueError):  # an answer nobody can read says nothing
            answers.append(json.loads(msg.data))

    nav.create_subscription(String, MARKED_TOPIC, heard, latched)
    publisher = nav.create_publisher(
        String, MARK_TOPIC, QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
    )
    request = {"name": name, "id": f"{os.getpid()}-{time.monotonic_ns()}"}
    deadline = time.monotonic() + timeout_s
    # Published more than once on purpose: the subscription on the laptop may not be matched yet
    # when the first one goes out, and a request nobody received is a mark that silently never
    # happened. The id makes a repeat free — the node answers one request once.
    while time.monotonic() < deadline:
        publisher.publish(String(data=json.dumps(request)))
        for _ in range(10):
            rclpy.spin_once(nav, timeout_sec=0.1)
            for answer in answers:
                if answer.get("id") == request["id"]:
                    return ("marked: " if answer.get("ok") else "NOT marked: ") + str(
                        answer.get("detail", "no reason given")
                    )
    return (
        f"no answer on {MARKED_TOPIC} in {timeout_s:.0f} s: is the laptop's places node up"
        f" (ros/laptop.sh vslam) and does {MARK_TOPIC} cross the bridge?"
    )


def placement_now(nav: BasicNavigator, wait_s: float = PLACEMENT_WAIT_S) -> Placement | None:
    """What this start of RTAB-Map is placed by, as pepin_bringup.rtabmap_frame says it on the
    latched :data:`pepin.watch.PLACEMENT_TOPIC`; ``None`` when nothing arrived within ``wait_s``
    or it did not parse. Latched, so the laptop's last word is in the first callback once the two
    ends have matched over the transport — the wait covers that and nothing else."""
    heard: list[Placement | None] = []
    latched = QoSProfile(
        depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
    )
    subscription = nav.create_subscription(
        String, PLACEMENT_TOPIC, lambda msg: heard.append(Placement.from_json(msg.data)), latched
    )
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline and not heard:
        rclpy.spin_once(nav, timeout_sec=0.1)
    nav.destroy_subscription(subscription)
    return heard[-1] if heard else None


def main() -> None:
    args = sys.argv[1:]
    places_path = Path("/maps/places.yaml")
    if len(args) >= 2 and args[0] == "--places":
        places_path, args = Path(args[1]), args[2:]
    if not args or args[0] not in ("cancel", "mark", "places", "seed"):
        print(__doc__)
        sys.exit(2)
    startup = time.monotonic()
    # SIGINT stays Python's: rclpy's own handler shuts the context down the moment Ctrl-C lands,
    # and `finally` below must still be able to tear the node down.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    if args[0] == "cancel":
        # Before the navigator exists: this process sends no goal, so it needs no commander —
        # and building one is what used to end the cancel in rclpy's teardown instead of on the
        # board's action server.
        node = rclpy.create_node("goto_cancel")
        try:
            print(cancel_all(node), flush=True)
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        return
    nav = BasicNavigator()
    print(
        f"startup: rclpy and the navigator ready at +{time.monotonic() - startup:.1f} s", flush=True
    )
    try:
        if args[0] == "mark":
            print(ask_mark(nav, args[1]))
            return
        if args[0] == "places":
            vocabulary = Vocabulary(nav, places_path)
            vocabulary.wait()
            graph = vocabulary.graph_places()
            for name, place in sorted(graph.items()):
                print(
                    f"{name:12s} x {place.x:+.2f} m, y {place.y:+.2f} m, "
                    f"yaw {place.theta_deg or 0.0:+.0f} deg (from the graph)"
                )
            for name, p in sorted(load_places(places_path).items()):
                if name in graph:
                    continue
                print(
                    f"{name:12s} x {p['x']:+.2f} m, y {p['y']:+.2f} m, "
                    f"yaw {p['yaw_deg']:+.0f} deg (from {places_path}, does not follow the graph)"
                )
            if not graph:
                print(
                    f"nothing on {PLACES_TOPIC}: the laptop's places node is not up, or this room"
                    " has no graph book yet (ros/goto.sh mark NAME makes one)"
                )
            return
        if args[0] == "seed":
            x, y = float(args[1]), float(args[2])
            yaw = float(args[3]) if len(args) > 3 else 0.0
            seed = pose(nav, x, y, yaw)
            # RTAB-Map listens in its own namespace; it owns map -> odom (on 2026-09-23 a restart
            # with the cart at the bookshelf left it "at home", 0 places recognised in 198
            # updates, and a seed is the cure). Sent a few times over ~3 s: a fresh publisher is
            # matched over the transport in seconds, and a repeated seed is harmless.
            rtab = PoseWithCovarianceStamped()
            rtab.header = seed.header
            rtab.pose.pose = seed.pose
            rtab.pose.covariance[0] = rtab.pose.covariance[7] = 0.05**2
            rtab.pose.covariance[35] = math.radians(5.0) ** 2
            to_rtabmap = nav.create_publisher(PoseWithCovarianceStamped, RTABMAP_INITIAL_POSE, 1)
            for _ in range(6):
                rtab.header.stamp = nav.get_clock().now().to_msg()
                to_rtabmap.publish(rtab)
                time.sleep(0.5)
            print(f"seeded at ({x:.2f}, {y:.2f}) yaw {yaw:.0f} deg on {RTABMAP_INITIAL_POSE}")
            heard = placement_now(nav)  # the seed is what places this start: say whether it did
            print(
                "RTAB-Map's start, as rtabmap_frame says it: "
                + (heard.how() if heard else f"nothing on {PLACEMENT_TOPIC}")
            )
            return
    finally:
        guarded("destroying the node", nav.destroy_node)
        if rclpy.ok():  # a context already shut down refuses every call made on it
            guarded("shutting rclpy down", rclpy.shutdown)


if __name__ == "__main__":
    main()
