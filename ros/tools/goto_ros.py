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
--no-tape drives without asking the run recorder for a numbered tape (the behaviour before
2026-09-14, when every drive through this door went unrecorded by it).

A goal pose published once on /goal_pose can be lost to discovery timing and gives
no feedback; the action client here waits for Nav2, watches the task and prints
distance remaining, recoveries and the final result.

Before any goal is sent, three checks are printed, one line each, and any failure refuses the
drive with the reading behind it (:class:`pepin.watch.Preflight`): has any source spoken to the
tracker at all, is the pose sure enough to drive on (the fusion's own sigma —
/localization/sigma — not the lidar's fit, which is 0.00 on a camera-only drive), and, where no
lidar is holding the pose, does RTAB-Map's graph recognise the room and agree with the tracker
about the place in it.
"""

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import rclpy
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

from pepin.runlink import (
    RUN_COMMAND_TOPIC,
    RUN_STATUS_TOPIC,
    RunLink,
    RunStatus,
    start_command,
    stop_command,
)
from pepin.watch import (
    ADMIT_FIT,
    BLIND_FIT,
    DRIVE_SIGMA_M,
    SIGMA_TOPIC,
    BlindDriveWatch,
    Preflight,
    Sigma,
    SourceWord,
    source_words,
)

# How long the tracker's service is given to appear before this client decides the board is in
# online SLAM and runs no tracker at all (pepin.deployment.runs_here). The same 5 s every other
# service wait here uses: over the bridge a client that gives up sooner gives up on a live board.
TRACKER_PATIENCE_S = 5.0
# ...and how old the map frame may be there. pepin_bringup.slam_frame re-stamps the correction at
# 10 Hz, so anything past a tenth of a second means that node is not running; 1 s is Nav2's own
# order of patience for the frame it plans in.
MAP_FRAME_FRESH_S = 1.0

# The drive's own watch: the pose uncertain past pepin.watch.LOST_SIGMA_M (or, on a board that
# publishes no sigma, the fit under BLIND_FIT) for this long...
LOST_FOR_S = 15.0
LOST_TRAVEL_M = 1.0  # ...while the wheels carried it this far: that is driving blind. Spinning on a
# stuck wheel with a lost reading is not — the recoveries (odom frame) can still work it free.
# How long the preflight waits for the board's first word about the pose. The tracker publishes
# its sigma every check period (1 s) whatever the sensors do and its per-source report on every
# update, so three seconds of nothing is the tracker itself being absent — which is a refusal
# with a reason, not a reason to wait longer.
CERTAINTY_WAIT_S = 3.0
# A cancel must be confirmed inside the patience ros/goto.sh gives it (timeout 5): the operator
# who typed "cancel" is watching the cart move. It is the budget for the WHOLE cancel, every
# navigator in it: spent per action it was 9 s (a discovery wait plus a spin, twice) under a
# shell timeout of 5, and the second navigator was never asked at all.
CANCEL_CONFIRM_S = 3.0
NAV_ACTIONS = ("navigate_to_pose", "navigate_through_poses")


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


class Certainty:
    """What the board says about the pose, for the preflight and for the drive's own watch.

    Three topics, in the order they are trusted. ``/localization/sigma`` is how sure the
    tracker's FUSION is — the one number a drive is judged on, whichever source spoke into it
    (:class:`pepin.watch.Sigma`). ``/localization/sources`` is who is holding the pose: every
    source's health and the word it put in, which is what tells a camera-only drive apart from a
    lidar one. ``/localization_fit`` is the lidar's own scan-to-map metric, kept as the fallback
    for a board whose build predates the sigma — and never trusted over it, because it is 0.00
    by construction wherever no lidar scan scored the pose.
    """

    def __init__(self, nav: BasicNavigator) -> None:
        self._nav = nav
        self.fit: float | None = None
        self._sigma: Sigma | None = None
        self._sigma_at: float | None = None
        self._report: dict[str, object] | None = None
        nav.create_subscription(Float32, "/localization_fit", self._on_fit, 10)
        nav.create_subscription(String, SIGMA_TOPIC, self._on_sigma, 10)
        nav.create_subscription(String, "/localization/sources", self._on_sources, 5)

    def _on_fit(self, msg: Float32) -> None:
        self.fit = float(msg.data)

    def _on_sigma(self, msg: String) -> None:
        heard = Sigma.from_json(msg.data, 0.0)
        if heard is not None:  # a message that does not parse says nothing about the pose
            self._sigma, self._sigma_at = heard, time.monotonic()

    def _on_sources(self, msg: String) -> None:
        try:
            heard = json.loads(msg.data)
        except ValueError:
            return  # a message that does not parse is counted by the tracker, not obeyed here
        if isinstance(heard, dict):
            self._report = heard

    def wait(self, seconds: float = CERTAINTY_WAIT_S) -> None:
        """Spin until both words have arrived or the patience runs out. Neither is a heartbeat
        this client can ask for: the sigma comes every check period and the source report on
        every update, so what is not here within the patience is not coming."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not (self._sigma and self._report):
            rclpy.spin_once(self._nav, timeout_sec=0.1)

    def refresh(self, seconds: float = 2.0) -> None:
        """Spin for ``seconds`` whatever has already arrived. After a whole-map search the
        numbers held here are the ones from BEFORE it until the board's next publications land,
        and a plain sleep processes no callback at all."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self._nav, timeout_sec=0.1)

    def sigma(self) -> Sigma | None:
        """The fused uncertainty and how long ago it landed here; ``None`` where this board
        publishes none at all, and the fit rules answer instead."""
        if self._sigma is None:
            return None
        age = time.monotonic() - (self._sigma_at or time.monotonic())
        return replace(self._sigma, age_s=max(0.0, age))

    def words(self) -> list[SourceWord]:
        """Every source's line of the tracker's last report; empty when none has arrived."""
        return [] if self._report is None else source_words(self._report)

    def heard(self) -> bool:
        """Whether this board says anything at all about how sure the pose is. False in online
        SLAM, where no tracker runs: there is nothing there for a drive's watch to read, and a
        watch that took the silence for 0.00 would cut every healthy drive of that mode."""
        return self._sigma is not None or self.fit is not None


def preflight(nav: BasicNavigator, certainty: Certainty) -> bool:
    """Print one line per check and answer whether the goal may be sent.

    A refusal always names the reading that caused it. The one refusal that buys something first
    is a pose that is simply not sure enough: standing still, that is exactly what a whole-map
    search fixes, so the search is run once (the tracker's own /relocalize, as before) and the
    three checks are asked again — after which a refusal is final.
    """
    certainty.wait()
    checks = Preflight().checks(certainty.words(), certainty.sigma(), certainty.fit)
    for check in checks:
        print(check.line(), flush=True)
    if Preflight.passed(checks):
        return True
    if not all(check.ok for check in checks if check.name != "certainty"):
        print("not driving: the refusals above are not something a search can fix", flush=True)
        return False
    print("an unsure pose buys one whole-map search: asking the tracker...", flush=True)
    client = nav.create_client(Trigger, "/relocalize")
    if not client.wait_for_service(timeout_sec=5.0):
        print("no /relocalize on this board: nothing to search with", flush=True)
        return False
    future = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(nav, future, timeout_sec=60.0)
    result = future.result()
    print(result.message if result else "no answer from /relocalize", flush=True)
    certainty.refresh()  # the search's own seed lands at the very end of it
    checks = Preflight().checks(certainty.words(), certainty.sigma(), certainty.fit)
    for check in checks:
        print(check.line(), flush=True)
    return Preflight.passed(checks)


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


def mark_place(nav: BasicNavigator, path: Path, name: str, certainty: Certainty) -> str:
    """Store the robot's current tracked pose under ``name``, on the same evidence a drive
    starts on: the fused sigma where the tracker publishes one, its fit where it does not.

    A place marked while the cart does not know where it stands is a place nobody can drive to
    afterwards, which is why this refuses at all — and the fit alone refused every mark on a
    camera-only stack, where no lidar scan scores the pose (2026-09-15).
    """
    pose = where_am_i(nav)
    if pose is None:
        return "cannot mark: the relocalizer is not answering"
    px, py, pyaw, fit = pose
    certainty.wait()
    sigma = certainty.sigma()
    if sigma is not None:
        if sigma.xy_m > DRIVE_SIGMA_M:
            return (
                f"NOT marked: the pose here is known to {sigma.phrase()}, over the"
                f" {DRIVE_SIGMA_M:.2f} m a mark needs; stand still 2 s, or run relocalize first"
            )
    elif fit < ADMIT_FIT:
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
        f"(fit {fit:.2f}" + ("" if sigma is None else f", sigma {sigma.phrase()}") + f") in {path}"
    )


def tracker_here(nav: BasicNavigator, timeout_s: float = TRACKER_PATIENCE_S) -> bool:
    """Whether the board runs the scan-matching tracker (``/where_am_i`` answers within
    ``timeout_s``). False in online SLAM: there is no saved map to match a scan against, so
    pepin.deployment.runs_here keeps the relocalizer off and neither of its services exists."""
    return bool(nav.create_client(Trigger, "/where_am_i").wait_for_service(timeout_sec=timeout_s))


def map_frame_age_s(nav: BasicNavigator, wait_s: float = 5.0) -> float | None:
    """Age in seconds of the newest ``map -> base_link`` transform, None while there is none.

    This is the whole localisation evidence online SLAM has: no tracker runs, so no fit is
    published, and what says the cart has a place in the map is that the frame exists at all —
    pepin_bringup.slam_frame broadcasts it on the board at 10 Hz from the laptop's correction
    (identity until RTAB-Map's first graph, which is the truth at the start of a session).
    """
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


def ensure_localized(nav: BasicNavigator, certainty: Certainty) -> bool:
    """The preflight: may this goal be sent at all.

    Where a tracker runs, three checks are printed and any failure refuses the drive
    (:func:`preflight`). The old rule this replaces read ``/localization_fit`` alone — the
    LIDAR's scan-to-map fit — and so refused every camera-only drive out of hand, because
    nothing there scores a scan against the map and the number is 0.00 by construction.

    In online SLAM there is no tracker at all: the map is being built on the laptop, so
    ``/where_am_i`` and ``/relocalize`` do not exist and this check used to refuse every goal of
    the mode after ten seconds of waiting for two absent services ("fit nan: searching the whole
    map first..." then "not localized", 2026-09-14 20:03 and 20:04). What is judged there
    instead is the map frame's own freshness — see :func:`map_frame_age_s`; a drive whose
    correction then goes stale is cut by the goal server's own watch (pepin.watch).
    """
    if not tracker_here(nav):
        age = map_frame_age_s(nav)
        if age is None:
            print(
                "no /where_am_i and no map -> base_link: neither the tracker (a known map) nor"
                " the SLAM frame (ros/thin.sh slam) is up on the board",
                flush=True,
            )
            return False
        if age > MAP_FRAME_FRESH_S:
            print(
                f"online SLAM: map -> base_link is {age:.1f} s old (over {MAP_FRAME_FRESH_S:.1f}"
                " s): the board's slam_frame is not broadcasting",
                flush=True,
            )
            return False
        print(
            f"preflight slam      ok       online SLAM, map -> base_link {age * 1e3:.0f} ms old"
            " (no tracker here: nothing matches a scan against a map still being built)",
            flush=True,
        )
        return True
    return preflight(nav, certainty)


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
    """Where the tracker says the robot ended, against the goal; in online SLAM, where no tracker
    runs, the map frame's own word (map -> base_link) with no fit beside it."""
    if not tracker_here(nav, timeout_s=1.0):
        age = map_frame_age_s(nav, wait_s=2.0)
        return "arrival: online SLAM, no tracker to ask" + (
            f"; map -> base_link {age * 1e3:.0f} ms old" if age is not None else "; no map frame"
        )
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
    rclpy.init()
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
            print(mark_place(nav, places_path, args[1], Certainty(nav)))
            return
        if args[0] == "places":
            for name, p in sorted(load_places(places_path).items()):
                print(
                    f"{name:12s} x {p['x']:+.2f} m, y {p['y']:+.2f} m, "
                    f"yaw {p['yaw_deg']:+.0f} deg (fit {p['fit']:.2f})"
                )
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
        print(f"startup: Nav2 answered at +{time.monotonic() - startup:.1f} s", flush=True)
        print(describe(x, y, yaw, home), flush=True)
        checked = time.monotonic()
        certainty = Certainty(nav)
        if not ensure_localized(nav, certainty):
            print(
                "not driving: the preflight refused above."
                " Stand the cart still and run ros/goto.sh relocalize, or ros/goto.sh where."
            )
            sys.exit(1)
        print(
            f"startup: localization checked in {time.monotonic() - checked:.1f} s, "
            f"{time.monotonic() - startup:.1f} s since this client began",
            flush=True,
        )
        # The drive's own watch, on the same rule the goal server uses (pepin.watch): the fused
        # sigma where the board publishes one, the lidar's fit where it does not — plus this
        # client's extra condition, that the wheels actually carried the cart while it was lost.
        blind = (
            BlindDriveWatch(lost_fit=BLIND_FIT, patience_s=LOST_FOR_S)
            if certainty.heard()
            else None  # online SLAM: no tracker speaks here, and the goal server watches instead
        )
        odom_xy: list[tuple[float, float] | None] = [None]
        lost_at_xy: list[tuple[float, float] | None] = [None]

        def on_odom(msg: Odometry) -> None:
            odom_xy[0] = (msg.pose.pose.position.x, msg.pose.pose.position.y)

        def travelled_while_lost() -> float:
            if lost_at_xy[0] is None or odom_xy[0] is None:
                return 0.0
            return math.hypot(odom_xy[0][0] - lost_at_xy[0][0], odom_xy[0][1] - lost_at_xy[0][1])

        nav.create_subscription(Odometry, "/odom", on_odom, 10)
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
            lost = blind is not None and blind.observe(
                certainty.fit or 0.0, now, sigma=certainty.sigma()
            )
            if blind is not None and blind.lost_since is None:
                lost_at_xy[0] = None  # the pose is healthy again: the clock starts over
            elif blind is not None and lost_at_xy[0] is None:
                lost_at_xy[0] = odom_xy[0]  # where the wheels were when it first went bad
            if lost and blind is not None and travelled_while_lost() > LOST_TRAVEL_M:
                note(
                    nav,
                    f"goto: the pose has been uncertain for {now - (blind.lost_since or now):.0f}"
                    f" s ({blind.phrase()}, judged by the {blind.rule}) while the wheels"
                    f" travelled {travelled_while_lost():.1f} m: cancelling, not driving blind",
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
        if tape is not None:
            tape.close()
        nav.destroy_node()
        if rclpy.ok():  # a context already shut down refuses every call made on it
            rclpy.shutdown()


if __name__ == "__main__":
    main()
