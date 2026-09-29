#!/usr/bin/env python3
"""Send the robot somewhere through Nav2 and report progress — the goal client we drive with.

Runs inside the container (rclpy + nav2_simple_commander):

    goto_ros.py X Y [YAW_DEG]     drive to map coordinates and print feedback until done
    goto_ros.py home              drive to the map origin, facing +x (the marked start spot)
    goto_ros.py seed X Y [YAW]    tell RTAB-Map where the robot stands
    goto_ros.py cancel            cancel the current navigation task
    goto_ros.py mark NAME         remember where the robot stands now as place NAME
    goto_ros.py NAME              drive to a remembered place
    goto_ros.py places            list the remembered places

A PLACE LIVES IN THE GRAPH, and this client asks the graph first. Under World R the map is
RTAB-Map's loop-closed graph and it BENDS when a loop closes, so a place is the pose the cart had
relative to a labelled graph node — pepin_bringup.places keeps those resolved into coordinates on
the latched ``/places`` and takes a mark on ``/places/mark``. Both are topics, because this process
runs inside the board's container and topics are what cross the zenoh bridge.

Latched is what makes it usable: the last vocabulary the laptop published is still in this client's
first callback after a WiFi drop, so ``go.sh printer`` works with the laptop asleep. When nothing
answers there at all, the coordinates in the file named by --places (default /maps/places.yaml) are
used with a plain warning — they are a frozen grid's numbers and the graph may have moved the room
since. An unknown name is refused either way, as before.
--no-tape drives without asking the run recorder for a numbered tape (the behaviour before
2026-09-14, when every drive through this door went unrecorded by it).

A goal pose published once on /goal_pose can be lost to discovery timing and gives
no feedback; the action client here waits for Nav2, watches the task and prints
distance remaining, recoveries and the final result.

Before any goal is sent, two checks are printed: whether ``map -> base_link`` is fresh (RTAB-Map
owns ``map -> odom``), and whether this start of RTAB-Map has been PLACED — a node of the loaded
map recognised, or an operator's seed — rather than still publishing the pose it saved at its last
shutdown (:meth:`pepin.watch.Preflight.placement`, latched on /localization/placement).
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
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
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
from pepin.runlink import (
    RUN_COMMAND_TOPIC,
    RUN_STATUS_TOPIC,
    RunLink,
    RunStatus,
    start_command,
    stop_command,
)
from pepin.watch import PLACEMENT_TOPIC, Placement, Preflight

# How old the map frame may be: RTAB-Map re-broadcasts map -> odom at 20 Hz, so anything past a
# second means its correction or the board's odometry is not arriving; 1 s is Nav2's own order of
# patience for the frame it plans in.
MAP_FRAME_FRESH_S = 1.0
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
# THE GOAL SERVER'S SWITCH FOR THE PLACEMENT REFUSAL, obeyed here too: one
# `ros/flags.sh set goal_server start_needs_placement false` lifts it for both goal paths, even
# with the laptop's rtabmap_frame down or on code that never publishes the word. Read from the
# goal server's parameter service (wherever the board's side runs it); a goal server that does not
# answer within the wait, or holds no such flag (a build from before it), leaves the default, on.
GOAL_SERVER = "/goal_server"
PLACEMENT_FLAG = "start_needs_placement"
FLAG_WAIT_S = 2.0
# How long the latched /places is given to land before the file beside the map answers instead. A
# latched publisher delivers as soon as the two endpoints match, so this covers discovery over the
# bridge and nothing else: the same 2 s every other "has the route come up" wait here uses, and a
# drive must not pay more than that for a vocabulary that may simply not exist in this room.
# 10 s since 2026-09-24: 2 s missed the latched book under load and the drive went to another map's
# shelf (Vocabulary.resolve now refuses instead of falling back to the file).
PLACES_WAIT_S = 10.0
# ...and how long a MARK is given to be answered. Longer, because it is the whole round trip: the
# request over the bridge, three of RTAB-Map's own services on the laptop, a file written, and the
# answer back. The same patience the recorder gets, for the same reason — it answers over a bridge.
MARK_WAIT_S = 8.0


RECORDER_PATIENCE_S = 8.0  # the recorder may answer over a bridge; the goal server waits as long


class Tape:
    """The numbered tape of this drive, asked of the run recorder (pepin_bringup.run_recorder).

    The recorder is a node where the sensors are and it opens a tape on one word published on
    ``pepin/run``; the goal server has always sent that word, and this client never did — so
    every drive started here went unrecorded by it while its own session log kept running
    (2026-09-13: the numbered tapes stop at 0248 and the goto tapes continue). Same protocol,
    second caller: :meth:`open` names the tape in this run's log, :meth:`close` ends it.
    """

    def __init__(self, nav: BasicNavigator) -> None:
        self._nav = nav
        self._runs = RunLink()
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pub = nav.create_publisher(String, RUN_COMMAND_TOPIC, 10)
        nav.create_subscription(String, RUN_STATUS_TOPIC, self._heard, latched)
        self.path: str | None = None

    def _heard(self, msg: String) -> None:
        status = RunStatus.from_json(msg.data)
        if status is not None:
            self._runs.observe(status)

    def _await(self, done) -> bool:  # type: ignore[no-untyped-def]
        """Spin this client until ``done()`` or the recorder's patience runs out."""
        deadline = time.monotonic() + RECORDER_PATIENCE_S
        while not done() and time.monotonic() < deadline:
            rclpy.spin_once(self._nav, timeout_sec=0.1)
        return bool(done())

    def open(self, name: str) -> None:
        """Ask for a tape called ``name`` and say which numbered one came back.

        A drive is never held hostage by its recorder: after the patience it goes anyway, and
        says so, exactly as the goal server does.
        """
        self._pub.publish(String(data=start_command(name)))
        if not self._await(lambda: self._runs.started(name)):
            print(
                f"no recorder confirmed run {name!r} in {RECORDER_PATIENCE_S:.0f} s:"
                " driving unrecorded",
                flush=True,
            )
            return
        self.path = self._runs.recording
        print(f"run {self._runs.run:04d}: taped {self.path}", flush=True)

    def close(self) -> None:
        """Close whatever tape the recorder says is open; harmless when none is.

        On the recorder's own word, never on ours: a start it heard but confirmed too late for
        :meth:`open`'s patience leaves a tape this client believes it never got, and a tape
        nobody closes keeps the board deserialising every scan onto the card until the run
        limit (900 s) runs out. The goal server closes on the same condition.
        """
        if self._runs.stopped():
            return
        self._pub.publish(String(data=stop_command()))
        closed = self._await(self._runs.stopped)
        print(
            f"run {self._runs.run:04d}: tape {'closed' if closed else 'NOT confirmed closed'}"
            f" {self.path or self._runs.recording}",
            flush=True,
        )


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


def note(text: str) -> None:
    """Say it here, flushed: the operator's terminal and ros/goto.sh's log are the record."""
    print(text, flush=True)


def guarded(what: str, step: Callable[[], Any]) -> bool:
    """Run one step of a shutdown and say whether it worked; never raise.

    Every step of an interrupt is independent: the goal's cancel, the recorder's stop word, the
    note, the teardown. Chaining them in one try block let the first failure swallow all the rest —
    which is exactly how a Ctrl-C left a goal running on the board.
    """
    try:
        step()
        return True
    except Exception as exc:  # a shutdown step may fail; the next one must still run
        print(f"!! {what} failed: {exc.__class__.__name__}: {exc}", flush=True)
        return False


def interrupted(nav: BasicNavigator, tape: Tape | None) -> None:
    """Ctrl-C: stop the robot first, then say so, in the order that matters if only one works.

    The goal lives on the board's action server, not in this client, so dying silently leaves Nav2
    driving toward it (2026-09-05: Ctrl-C on the laptop, the robot kept going; 2026-09-17: the same
    thing again, this time because the handler's own first line raised). Hence: the cancel, then the
    recorder's stop word, then the words — each :func:`guarded`, so a failure costs its own step and
    nothing else. The context is still alive here because rclpy was told not to install its own
    SIGINT handler (see :func:`main`), which is what makes any of this possible.
    """
    cancelled = guarded("cancelling the goal", nav.cancelTask)
    if tape is not None:
        guarded("closing the tape", tape.close)
    guarded("the note", lambda: note("goto: interrupted by the operator, cancelling the goal"))
    if not cancelled:
        print("cancel NOT sent — run ros/stop.sh NOW", flush=True)
        return
    deadline, done = time.monotonic() + 5.0, False
    while not done and time.monotonic() < deadline:
        time.sleep(0.1)
        try:
            done = bool(nav.isTaskComplete())
        except Exception as exc:  # the cancel is sent; this is only its confirmation
            print(f"!! cannot confirm the cancel: {exc.__class__.__name__}: {exc}", flush=True)
            break
    print("cancelled" if done else "cancel NOT confirmed — use ros/stop.sh", flush=True)


def load_places(path: Path) -> dict[str, dict[str, float]]:
    """Named places of this map: {name: {x, y, yaw_deg, fit}}; an absent file is an empty book."""
    if not path.exists():
        return {}
    data: dict[str, dict[str, float]] = json.loads(path.read_text())
    return data


class Vocabulary:
    """The room's places as the GRAPH answers for them: the latched ``/places``, with the
    coordinates beside the map as the fallback.

    The graph's book is asked first because it is the only one that is still true after a loop
    closure: each of its places is a pose relative to a labelled node, so it rides the node when the
    graph bends, while a coordinate written into a file stays where the room used to be. The
    fallback is announced in one plain line rather than used in silence — a drive that reached the
    wrong shelf must say which book sent it there.
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

    def resolve(self, name: str) -> tuple[float, float, float] | None:
        """``(x, y, yaw_deg)`` for a name — from the graph if it can answer, else from the file —
        or ``None`` when neither knows it. Says which book answered, every time."""
        if not self._heard:
            # The file beside the map was written for a FROZEN map in another frame: on 2026-09-24
            # a 2 s miss of the latched book sent the cart to (-10.06, -0.43), flat3_straight's
            # shelf. No answer from the graph is a refusal, never a silent fallback.
            print(
                f"!! no answer on {PLACES_TOPIC} within {PLACES_WAIT_S:.0f} s: refusing"
                f" {name!r} rather than fall back to {self._path} (a frozen map's"
                " coordinates); run the goal again",
                flush=True,
            )
            return None
        place = self._graph.get(name)
        if place is not None:
            print(
                f"place {name!r} from the graph ({PLACES_TOPIC}, {len(self._graph)} known): it"
                " rides its labelled node, so a loop closure moves it with the room",
                flush=True,
            )
            return place.x, place.y, place.theta_deg or 0.0
        entry = load_places(self._path).get(name)
        if entry is None:
            return None
        print(
            f"!! place {name!r} is not in {PLACES_TOPIC}"
            + ("" if self._heard else " (nothing published there at all)")
            + f": falling back to the coordinates in {self._path}, which were written for a frozen"
            " map and do not follow a loop closure",
            flush=True,
        )
        return entry["x"], entry["y"], entry["yaw_deg"]

    def known(self) -> str:
        """Every name either book has, for a refusal that names what IS known."""
        names = sorted(set(self._graph) | set(load_places(self._path)))
        return ", ".join(names) or "none yet"

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


def goal_server_flag(nav: BasicNavigator, name: str, wait_s: float = FLAG_WAIT_S) -> bool | None:
    """The goal server's live bool flag ``name`` as it holds it now (its GetParameters service);
    ``None`` when the goal server does not answer within ``wait_s`` or holds no such bool flag,
    and the caller keeps the flag's table default."""
    from rcl_interfaces.msg import ParameterType
    from rcl_interfaces.srv import GetParameters

    client = nav.create_client(GetParameters, f"{GOAL_SERVER}/get_parameters")
    try:
        if not client.wait_for_service(timeout_sec=wait_s):
            return None
        request = GetParameters.Request()
        request.names = [name]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(nav, future, timeout_sec=wait_s)
        answer = future.result()
    finally:
        nav.destroy_client(client)
    values = list(getattr(answer, "values", None) or [])
    if len(values) != 1 or values[0].type != ParameterType.PARAMETER_BOOL:
        return None  # not answered, or not declared there (PARAMETER_NOT_SET)
    return bool(values[0].bool_value)


def map_frame_age_s(nav: BasicNavigator, wait_s: float = 5.0) -> float | None:
    """Age in seconds of the newest ``map -> base_link`` transform, None while there is none:
    RTAB-Map's map -> odom composed with the board's own odometry."""
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformException, TransformListener

    buffer = Buffer()
    listener = TransformListener(buffer, nav)  # named: the subscription dies with the reference
    deadline = time.monotonic() + wait_s
    age: float | None = None
    while time.monotonic() < deadline:
        rclpy.spin_once(nav, timeout_sec=0.1)
        try:
            heard = buffer.lookup_transform("map", "base_link", Time())
        except TransformException:
            continue
        stamp = Time.from_msg(heard.header.stamp)
        age = float((nav.get_clock().now() - stamp).nanoseconds) * 1e-9
        break
    del listener
    return age


def ensure_localized(nav: BasicNavigator) -> str | None:
    """The preflight: may this goal be sent at all. ``None`` when it may; otherwise the hint to
    print under the refusal (empty when the refusing line already says what to do).

    Two checks. The map frame's own freshness (:func:`map_frame_age_s`). And, because RTAB-Map
    publishes a frame from the moment it starts, at the pose it saved at its last shutdown, that
    this start is PLACED — recognised or seeded (:meth:`pepin.watch.Preflight.placement`;
    2026-09-23, the cart "at home" at the bookshelf, then 76 cm off inside the table) — unless the
    goal server's flag ``start_needs_placement`` is off (:func:`goal_server_flag`).
    """
    age = map_frame_age_s(nav)
    if age is None:
        print(
            "no map -> base_link: RTAB-Map's map -> odom (ros/laptop.sh vslam) is not up",
            flush=True,
        )
        return ""
    if age > MAP_FRAME_FRESH_S:
        print(
            f"preflight frame     REFUSED  map -> base_link is {age:.1f} s old (over"
            f" {MAP_FRAME_FRESH_S:.1f} s): RTAB-Map's map -> odom or the board's odometry"
            " is not arriving",
            flush=True,
        )
        return ""
    print(f"preflight frame     ok       map -> base_link {age * 1e3:.0f} ms old", flush=True)
    asked = goal_server_flag(nav, PLACEMENT_FLAG)
    placed = Preflight.placement(placement_now(nav), asked=asked is not False)
    print(placed.line(), flush=True)
    return None if placed.ok else ""


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


def arrival(nav: BasicNavigator) -> str:
    """How fresh the pose is at the end of the drive; ros/go.sh where prints the pose itself."""
    age = map_frame_age_s(nav, wait_s=2.0)
    return "arrival: " + (
        f"map -> base_link {age * 1e3:.0f} ms old (the pose: ros/go.sh where)"
        if age is not None
        else "no map frame"
    )


def main() -> None:
    args = sys.argv[1:]
    name: str | None = None
    places_path = Path("/maps/places.yaml")
    if len(args) >= 2 and args[0] == "--places":
        places_path, args = Path(args[1]), args[2:]
    taping = "--no-tape" not in args  # the switch back to the unrecorded drive
    args = [a for a in args if a != "--no-tape"]
    if not args:
        print(__doc__)
        sys.exit(2)
    startup = time.monotonic()
    tape: Tape | None = None
    # SIGINT stays PYTHON's: rclpy's own handler shuts the context down the moment Ctrl-C lands, and
    # then every call the interrupt path needs — create_publisher, cancelTask, publish — raises
    # "context is invalid". On 2026-09-17 that turned an operator's Ctrl-C into a goal left running
    # on the board twice (ros/maps/rec/20260917_192935_goto.log, ..._201425_goto.log). With NO the
    # context outlives the signal, KeyboardInterrupt arrives as an ordinary exception, and the
    # cancel goes out over a live link; `finally` still shuts the context down afterwards.
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
                    " has no graph book yet (ros/go.sh mark NAME makes one)"
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
        name = None
        # The graph's own book first, the file beside the map second (:class:`Vocabulary`): a
        # coordinate written for a frozen grid is not where the furniture is once a loop has closed.
        vocabulary = Vocabulary(nav, places_path)
        vocabulary.wait()
        home_at = vocabulary.resolve("home") if args[0] == "home" else None
        home = (
            {"x": home_at[0], "y": home_at[1], "yaw_deg": home_at[2]}
            if home_at is not None
            else load_places(places_path).get("home")
        )
        if args[0] == "home":
            # An unmarked home used to fall back to the MAP ORIGIN in silence. On this map the
            # origin sits 6.6 m beyond the right-hand edge, so `goto.sh home` sent the cart on a
            # straight line out of the map and through the furniture in the way (2026-09-17,
            # Artem watching). A place that was never marked is a place nobody can drive to, and
            # the only honest answer is to say so.
            if home_at is None:
                print(
                    f"no place 'home' in {PLACES_TOPIC} or {places_path}: it was never marked on"
                    f" this map (known here: {vocabulary.known()}). Stand the cart where home is"
                    " and run ros/go.sh mark home."
                )
                sys.exit(2)
            x, y, yaw = home_at
            name = "home"
        elif not args[0].lstrip("-").replace(".", "", 1).isdigit():
            at = vocabulary.resolve(args[0])
            if at is None:
                print(f"unknown place {args[0]!r}; known: {vocabulary.known()}")
                sys.exit(2)
            name, (x, y, yaw) = args[0], at
        else:
            x, y = float(args[0]), float(args[1])
            yaw = float(args[2]) if len(args) > 2 else 0.0
        nav.waitUntilNav2Active(localizer="robot_localization")  # no AMCL: RTAB-Map owns the frame
        print(f"startup: Nav2 answered at +{time.monotonic() - startup:.1f} s", flush=True)
        print(describe(x, y, yaw, home), flush=True)
        checked = time.monotonic()
        refused = ensure_localized(nav)
        if refused is not None:
            print("not driving: the preflight refused above." + (f" {refused}" if refused else ""))
            sys.exit(1)
        print(
            f"startup: localization checked in {time.monotonic() - checked:.1f} s, "
            f"{time.monotonic() - startup:.1f} s since this client began",
            flush=True,
        )
        # The numbered tape, the one the replays and the reports are named by: opened before the
        # goal so its prelude holds the seconds before the cart moves, closed in `finally`.
        if taping:
            tape = Tape(nav)
            tape.open(name or f"{x:.0f}_{y:.0f}")
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
        print(arrival(nav), flush=True)
    except KeyboardInterrupt:
        interrupted(nav, tape)
        tape = None  # its stop word went out above; `finally` must not send a second one
    finally:
        if tape is not None:
            guarded("closing the tape", tape.close)
        guarded("destroying the node", nav.destroy_node)
        if rclpy.ok():  # a context already shut down refuses every call made on it
            guarded("shutting rclpy down", rclpy.shutdown)


if __name__ == "__main__":
    main()
