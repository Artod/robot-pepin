"""The robot waits for orders instead of being started for each one.

Every goal used to boot its own client on the board: ssh, docker exec, import rclpy, import the
Nav2 commander, build a node, discover the action server — 8 to 15 seconds before the wheels
could move, paid again for every command (measured 2026-09-08). This node is already running,
already connected to Nav2 and already holding the places book, so a command costs a socket write.

It speaks JSON lines on a TCP port (like the base and ToF servers), one connection at a time:

    {"cmd": "go", "place": "printer"}      {"cmd": "go", "x": -11.4, "y": 0.8, "yaw_deg": 140}
    {"cmd": "cancel"}                      {"cmd": "where"}
    {"cmd": "mark", "name": "printer"}     {"cmd": "places"}

and answers with one JSON line per event: accepted, feedback, arrival, done. It also owns the
run's recording: it starts one when a goal starts and closes it when the goal ends, so a
recording can no longer outlive its run.
"""

from __future__ import annotations

import contextlib
import json
import math
import socket
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.srv import ChangeState, GetState
from nav2_msgs.action import NavigateToPose, Spin
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float32, Header, String
from std_srvs.srv import Trigger

from pepin.deployment import (
    BOARD_NAV_NODES,
    HEARTBEAT_HZ,
    HEARTBEAT_TOPIC,
    next_transition,
)
from pepin.places import heading_residual_deg
from pepin.runlink import (
    RUN_COMMAND_TOPIC,
    RUN_STATUS_TOPIC,
    RunLink,
    RunStatus,
    start_command,
    stop_command,
)
from pepin.watch import DRIVE_FIT, BlindDriveWatch

PORT = 3337
GOOD_FIT = DRIVE_FIT  # below this the robot is told to find itself before it drives (pepin.watch)
# The planner to select, and the controller that follows it. One controller now: the lattice
# planner no longer expands in reverse, so there is nothing a reversing controller would add.
PIVOT_TOLERANCE_DEG = 11.5  # the goal checker's 0.20 rad: below this the heading is met
PIVOT_ALLOWANCE_S = 15.0
RECORDER_PATIENCE_S = (
    8.0  # the recorder answers over the bridge; 3 s once named a drive after the previous tape
)
BRINGUP_ROUND_S = 10.0  # a lifecycle query or transition that has not answered by then is abandoned

PLANNERS = {
    "navfn": ("GridBased", "FollowPath"),
    "lattice": ("Lattice", "FollowPath"),  # experimental: see nav2_params.yaml
    "theta": ("ThetaStar", "FollowPath"),
    "smac": ("Smac2D", "FollowPath"),
    "hybrid": ("Hybrid", "FollowPathRS"),  # footprint-aware; the reversing RPP
}


class GoalServer(Node):
    """Takes goals over a socket and drives them through Nav2, with the run recorded."""

    def __init__(self) -> None:
        super().__init__("goal_server")
        self._places_path = Path(str(self.declare_parameter("places", "/maps/places.yaml").value))
        self._record_dir = Path(str(self.declare_parameter("record_dir", "/maps/rec").value))
        self._port = int(self.declare_parameter("port", PORT).value)
        self._client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self._spin = ActionClient(self, Spin, "spin")
        self._relocalize = self.create_client(Trigger, "relocalize")
        self._where = self.create_client(Trigger, "where_am_i")
        self.fit = 0.0
        self.create_subscription(Float32, "localization_fit", self._on_fit, 10)
        # Latched: the behaviour tree reads its selector once, whenever it next ticks.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._planner_pick = self.create_publisher(String, "planner_selector", latched)
        self._controller_pick = self.create_publisher(String, "controller_selector", latched)
        # Remembered across restarts: a container that comes back with a different planner than
        # the one being tested makes every comparison a lie.
        # Under maps/rec, which sync.sh excludes: kept in maps/ the file was deleted by the very
        # next deploy (rsync --delete), so every restart silently went back to the default planner
        # and a drive was credited to a planner that never ran.
        self._planner_path = self._record_dir / ".planner"
        self.planner = "navfn"
        with contextlib.suppress(OSError):
            self.pick_planner(self._planner_path.read_text().strip())
        self._goal_handle: Any = None
        self._driving = (
            False  # from before send_goal until the drive is finally over: cancel() clears it
        )
        # The laptop's pulse: on a split stack the board's link watch cancels a drive when this
        # stops. Harmless on one machine, where nothing listens.
        self._beat = self.create_publisher(Header, HEARTBEAT_TOPIC, 10)
        self.create_timer(1.0 / HEARTBEAT_HZ, self._heartbeat)
        # The laptop half brings the board's Nav2 up (pepin.deployment.next_transition): the
        # board's tree cannot load before this side's costmap service exists, and the board's
        # own manager gives up after one failure. Every few seconds: read the four states, send
        # the one transition that is due, read again. Idempotent, so a restart on either side
        # simply continues.
        self._side = str(self.declare_parameter("side", "all").value)
        self._board_state: dict[str, str] = {}
        self._board_up_logged = False
        if self._side == "laptop":
            self._state_clients = {
                node: self.create_client(GetState, f"/{node}/get_state") for node in BOARD_NAV_NODES
            }
            self._change_clients = {
                node: self.create_client(ChangeState, f"/{node}/change_state")
                for node in BOARD_NAV_NODES
            }
            self._bringup_busy = False
            self._bringup_since = 0.0
            self.create_timer(3.0, self._bring_board_up)
        # The recorder is a node where the sensors are (run_recorder, on the board): one command
        # opens a tape, the latched status names it (pepin.runlink).
        self._runs = RunLink()
        self._run_word = threading.Event()  # set on every status heard
        self._run_pub = self.create_publisher(String, RUN_COMMAND_TOPIC, 10)
        self.create_subscription(
            String,
            RUN_STATUS_TOPIC,
            self._on_run_status,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._lock = threading.Lock()
        threading.Thread(target=self._serve, daemon=True).start()
        self.get_logger().info(f"goal server ready on port {self._port}")

    def _heartbeat(self) -> None:
        self._beat.publish(Header(stamp=self.get_clock().now().to_msg(), frame_id="laptop"))

    def _bring_board_up(self) -> None:
        """Every 3 s on the laptop: read the board's lifecycle states and send the due step."""
        now = time.monotonic()
        if self._bringup_busy:
            # A call whose answer never comes (the board restarted under it, the bridge re-routing)
            # must not hold the bring-up for good: after BRINGUP_ROUND_S the round is abandoned.
            if now - self._bringup_since < BRINGUP_ROUND_S:
                return
            self.get_logger().warning("board bring-up: a round got no answer; asking afresh")
            self._board_state.clear()
        self._bringup_busy = True
        self._bringup_since = now
        pending = set(BOARD_NAV_NODES)
        for node, client in self._state_clients.items():
            if not client.service_is_ready():
                self._board_state.pop(node, None)
                pending.discard(node)
                continue
            future = client.call_async(GetState.Request())
            future.add_done_callback(lambda f, n=node: self._board_state_read(n, f, pending))
        if not pending:
            self._bringup_busy = False

    def _board_state_read(self, node: str, future: Any, pending: set[str]) -> None:
        try:
            self._board_state[node] = str(future.result().current_state.label)
        except Exception:  # a dropped call: the next round asks again
            self._board_state.pop(node, None)
        pending.discard(node)
        if pending:
            return
        step = next_transition(self._board_state)
        if step is None:
            self._bringup_busy = False
            if all(self._board_state.get(n) == "active" for n in BOARD_NAV_NODES):
                if not self._board_up_logged:
                    self.get_logger().info("board Nav2 is up")
                    self._board_up_logged = True
            else:
                self._board_up_logged = False
            return
        node, transition = step
        self._board_up_logged = False
        request = ChangeState.Request()
        request.transition.id = transition
        self.get_logger().info(f"board bring-up: {node} <- transition {transition}")
        future = self._change_clients[node].call_async(request)
        future.add_done_callback(lambda f: self._board_transition_done(node, transition, f))

    def _board_transition_done(self, node: str, transition: int, future: Any) -> None:
        try:
            ok = bool(future.result().success)
        except Exception:
            ok = False
        if not ok:
            self.get_logger().warning(
                f"board bring-up: {node} refused transition {transition}; asking again in 3 s"
            )
        self._bringup_busy = False

    def _on_fit(self, msg: Float32) -> None:
        self.fit = float(msg.data)

    @staticmethod
    def _wait(future: Any, timeout: float) -> Any:
        """Wait for a future from a session thread; the node's own spin drives it to completion.

        Spinning here instead raises "executor is already spinning": a process may spin in one
        place only, and that place is main().
        """
        done = threading.Event()
        future.add_done_callback(lambda _future: done.set())
        return future.result() if done.wait(timeout) else None

    # -- the run's recording ---------------------------------------------------

    def _on_run_status(self, msg: String) -> None:
        status = RunStatus.from_json(msg.data)
        if status is not None:
            self._runs.observe(status)
            self._run_word.set()

    def _await_recorder(self, done: Any) -> bool:
        """Wait up to RECORDER_PATIENCE_S for the recorder's status to satisfy ``done``."""
        deadline = time.monotonic() + RECORDER_PATIENCE_S
        while not done():
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            self._run_word.clear()
            self._run_word.wait(left)
        return True

    def start_recording(self, name: str) -> Path | None:
        """Ask the recorder for this run's tape: its path once confirmed, None if nobody answered.

        A drive is not held hostage by its recorder: after RECORDER_PATIENCE_S it goes anyway,
        loudly, with no recording named in its events.
        """
        self._run_pub.publish(String(data=start_command(name)))
        if not self._await_recorder(lambda: self._runs.started(name)):
            self.get_logger().warning(
                f"no recorder confirmed run '{name}' in {RECORDER_PATIENCE_S:.0f} s: "
                "driving unrecorded"
            )
            return None
        path = Path(str(self._runs.recording))
        self.get_logger().info(f"run {self._runs.run}: recording {path}")
        return path

    def stop_recording(self) -> None:
        """Close the run's tape (the recorder flushes and syncs it); harmless when none is open."""
        if self._runs.stopped():
            return
        self._run_pub.publish(String(data=stop_command()))
        if not self._await_recorder(self._runs.stopped):
            self.get_logger().warning("the recorder did not confirm the tape closed")

    def _serve(self) -> None:
        """One connection at a time: read a command, stream its events back, close."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", self._port))
        listener.listen(1)
        while rclpy.ok():
            try:
                connection, _ = listener.accept()
            except OSError:
                continue
            threading.Thread(target=self._session, args=(connection,), daemon=True).start()

    def _session(self, connection: socket.socket) -> None:
        with connection:
            connection.settimeout(600.0)
            try:
                line = connection.makefile("r").readline()
                request = json.loads(line) if line.strip() else {}
            except (OSError, ValueError) as error:
                self._send(connection, {"event": "error", "detail": str(error)[:120]})
                return
            try:
                self._handle(request, connection)
            except Exception as error:  # a bad command must not take the server down
                self._send(connection, {"event": "error", "detail": str(error)[:200]})

    @staticmethod
    def _send(connection: socket.socket, payload: dict[str, Any]) -> None:
        with contextlib.suppress(OSError):  # the caller may have hung up mid-run
            connection.sendall((json.dumps(payload) + "\n").encode())

    # -- commands --------------------------------------------------------------

    def _handle(self, request: dict[str, Any], connection: socket.socket) -> None:
        command = str(request.get("cmd", ""))
        if command == "places":
            self._send(connection, {"event": "places", "places": self.places()})
        elif command == "where":
            self._send(
                connection,
                {"event": "where", "fit": self.fit, "planner": self.planner, **self._pose_now()},
            )
        elif command == "mark":
            self._send(connection, self.mark(str(request.get("name", ""))))
        elif command == "planner":
            self._send(connection, self.pick_planner(str(request.get("name", ""))))
        elif command == "cancel":
            self._send(connection, {"event": "cancelled", "had_goal": self.cancel()})
        elif command == "go":
            self._go(request, connection)
        else:
            self._send(connection, {"event": "error", "detail": f"unknown command {command!r}"})

    def places(self) -> dict[str, dict[str, float]]:
        """The named places of the map in use; an absent book is an empty one."""
        try:
            data: dict[str, dict[str, float]] = json.loads(self._places_path.read_text())
            return data
        except (OSError, ValueError):
            return {}

    def _pose_now(self) -> dict[str, float]:
        """The tracker's pose, asked over its own service (empty when it does not answer)."""
        if not self._where.wait_for_service(timeout_sec=1.0):
            return {}
        result = self._wait(self._where.call_async(Trigger.Request()), 5.0)
        if result is None:
            return {}
        text = result.message
        try:
            return {
                "x": float(text.split("x ")[1].split(" m")[0]),
                "y": float(text.split("y ")[1].split(" m")[0]),
                "yaw_deg": float(text.split("yaw ")[1].split(" deg")[0]),
                "fit": float(text.split("fit ")[1].split(" ")[0]),
            }
        except (IndexError, ValueError):
            return {}

    def mark(self, name: str) -> dict[str, Any]:
        """Remember where the robot stands as ``name``; a weak fit is refused."""
        pose = self._pose_now()
        if not name or not pose:
            return {"event": "error", "detail": "no name, or the tracker did not answer"}
        if pose.get("fit", 0.0) < GOOD_FIT:
            return {"event": "error", "detail": f"fit {pose['fit']:.2f}: stand still or relocalize"}
        places = self.places()
        places[name] = {k: round(pose[k], 3) for k in ("x", "y", "yaw_deg")} | {
            "fit": round(pose["fit"], 2)
        }
        self._places_path.write_text(json.dumps(places, indent=2, sort_keys=True) + "\n")
        return {"event": "marked", "name": name, **places[name]}

    def pick_planner(self, name: str) -> dict[str, Any]:
        """Choose the planner and the controller that can follow it; both, or neither."""
        pair = PLANNERS.get(name.lower())
        if pair is None:
            return {"event": "error", "detail": f"planner must be one of {sorted(PLANNERS)}"}
        planner, controller = pair
        self._planner_pick.publish(String(data=planner))
        self._controller_pick.publish(String(data=controller))
        self.planner = name.lower()
        with contextlib.suppress(OSError):
            self._planner_path.write_text(f"{self.planner}\n")
        self.get_logger().info(f"planner {planner} with controller {controller}")
        return {"event": "planner", "planner": planner, "controller": controller}

    def cancel(self) -> bool:
        """Stop the running drive, if any; True when there was one.

        Clears ``_driving`` as well as the handle: during the lost -> relocalise -> resume window
        there is no handle to cancel, and the flag is what stops the resume from re-sending the
        goal the operator just cancelled.
        """
        with self._lock:
            handle, self._goal_handle = self._goal_handle, None
            was_driving, self._driving = self._driving, False
        if handle is not None:
            handle.cancel_goal_async()
        return was_driving

    def _go(
        self,
        request: dict[str, Any],
        connection: socket.socket,
        resume: bool = True,
        record: Path | None = None,
    ) -> None:
        """Send one goal and stream its progress until it ends or the caller hangs up.

        ``resume``: a drive stopped for being lost is sent again after a relocalisation, once —
        as a continuation of the same drive: ``record`` is the tape already open, so the resumed
        leg keeps the run's number and file instead of becoming a second run.
        """
        target = self._target_of(request)
        if target is None:
            self._send(connection, {"event": "error", "detail": "no such place"})
            return
        x, y, yaw_deg, name = target
        if self.fit < GOOD_FIT and not self._find_myself(connection):
            return
        if not self._client.wait_for_server(timeout_sec=5.0):
            self._send(connection, {"event": "error", "detail": "Nav2 is not up"})
            return
        goal = NavigateToPose.Goal()
        goal.pose = self._pose_msg(x, y, yaw_deg)
        started = time.monotonic()
        feedback: dict[str, Any] = {}
        with self._lock:
            if self._driving and record is None:
                self._send(
                    connection, {"event": "error", "detail": "already driving: cancel first"}
                )
                return
            self._driving = True
        if record is None:
            record = self.start_recording(name or f"{x:.0f}_{y:.0f}")
        self.get_logger().info(
            f"run {self._runs.run}: planner {PLANNERS[self.planner][0]} "
            f"-> {name or 'coordinates'} ({x:.2f}, {y:.2f}, {yaw_deg:.0f} deg)"
        )
        try:
            send = self._client.send_goal_async(
                goal,
                lambda f: feedback.update(
                    distance=f.feedback.distance_remaining,
                    recoveries=f.feedback.number_of_recoveries,
                ),
            )
            handle = self._wait(send, 10.0)
            if handle is None or not handle.accepted:
                self._send(connection, {"event": "error", "detail": "the goal was refused"})
                return
            with self._lock:
                self._goal_handle = handle
            self._send(
                connection,
                {
                    "event": "accepted",
                    "run": self._runs.run,
                    "planner": PLANNERS[self.planner][0],
                    # early: a drive that never reaches "done" is still fetched
                    "recording": None if record is None else str(record),
                    "place": name,
                    "x": x,
                    "y": y,
                    "yaw_deg": yaw_deg,
                    "sent_in_ms": round((time.monotonic() - started) * 1000),
                },
            )
            result_future = handle.get_result_async()
            last = 0.0
            blind = BlindDriveWatch()
            stopped_lost = False
            while rclpy.ok() and not result_future.done():
                time.sleep(0.05)  # the node's own spin serves the action; this thread only reports
                now = time.monotonic()
                if blind.observe(self.fit, now):  # a blind drive is stopped, not finished
                    stopped_lost = True
                    self._send(
                        connection, {"event": "lost", "fit": self.fit, "t": round(now - started, 1)}
                    )
                    self.get_logger().warning(
                        f"lost mid-drive (fit {self.fit:.2f}): stopping to relocalise"
                    )
                    handle.cancel_goal_async()
                    break
                if feedback and now - last > 1.0:
                    last = now
                    self._send(
                        connection, {"event": "feedback", "t": round(now - started, 1), **feedback}
                    )
            with self._lock:
                self._goal_handle = None
            if stopped_lost:
                # Stopped on purpose: find ourselves standing still, then the same goal, once.
                self._wait(result_future, 10.0)
                with self._lock:
                    still_wanted = self._driving
                if still_wanted and self._find_myself(connection) and resume:
                    self._send(connection, {"event": "resuming", "fit": self.fit})
                    self.get_logger().info(
                        f"resuming the goal after relocalising (fit {self.fit:.2f})"
                    )
                    self._go(request, connection, resume=False)
                return
            outcome = result_future.result()
            status = getattr(outcome, "status", 0) if outcome else 0
            if status == 4:  # position met: now the heading, on the tape still
                self._pivot_to(yaw_deg, connection)
            self.stop_recording()  # closed before the answer: the caller fetches it on reading
            self._send(
                connection,
                {
                    "event": "done",
                    "run": self._runs.run,
                    "planner": self.planner,
                    "status": int(status),
                    "seconds": round(time.monotonic() - started, 1),
                    "arrival": self._pose_now(),
                    "recording": None if record is None else str(record),
                },
            )
        finally:  # a refused goal or a broken connection must not leave a recorder running
            if (
                resume
            ):  # the outermost call owns the tape and the flag; a resumed leg is the same drive
                self.stop_recording()
                with self._lock:
                    self._driving = False

    def _pivot_to(self, yaw_deg: float, connection: socket.socket) -> None:
        """Turn in place to the mark's heading once the drive has met the position.

        The controller that can reverse (RPP, FollowPathRS) cannot rotate in place, and at a
        mark Hybrid-A* writes a 10 cm cusp plan whose carrot sits straight ahead: three printer
        approaches spent 35-61 s shuttling for the last 30 degrees. The drive therefore ends on
        position alone and the behaviour server's Spin does the heading: 40 degrees in ~1.5 s.
        """
        residual = heading_residual_deg(yaw_deg, self._pose_now()["yaw_deg"])
        if abs(residual) <= PIVOT_TOLERANCE_DEG:
            return
        event: dict[str, Any] = {"event": "pivot", "residual_deg": round(residual, 1)}
        if not self._spin.wait_for_server(timeout_sec=2.0):
            self._send(connection, event | {"status": 0, "detail": "no spin behaviour"})
            return
        goal = Spin.Goal()
        goal.target_yaw = math.radians(residual)
        goal.time_allowance = Duration(sec=int(PIVOT_ALLOWANCE_S))
        handle = self._wait(self._spin.send_goal_async(goal), 5.0)
        if handle is None or not handle.accepted:
            self._send(connection, event | {"status": 0, "detail": "the spin was refused"})
            return
        outcome = self._wait(handle.get_result_async(), PIVOT_ALLOWANCE_S + 5.0)
        status = getattr(outcome, "status", 0) if outcome else 0
        after = heading_residual_deg(yaw_deg, self._pose_now()["yaw_deg"])
        self._send(connection, event | {"status": int(status), "after_deg": round(after, 1)})
        self.get_logger().info(f"pivot {residual:+.0f} deg: status {status}, {after:+.0f} deg left")

    def _target_of(self, request: dict[str, Any]) -> tuple[float, float, float, str | None] | None:
        """The goal asked for: a named place, or plain coordinates."""
        if "place" in request:
            place = self.places().get(str(request["place"]))
            if place is None:
                return None
            return place["x"], place["y"], place["yaw_deg"], str(request["place"])
        if "x" in request and "y" in request:
            return (
                float(request["x"]),
                float(request["y"]),
                float(request.get("yaw_deg", 0.0)),
                None,
            )
        return None

    def _find_myself(self, connection: socket.socket) -> bool:
        """A weak fit before (or during) a drive buys one whole-map search; blind driving is never
        allowed. The tracker may already be searching on its own — then its answer is awaited
        rather than asked for twice — and its fit is published once a second, so the verdict
        waits for a fresh reading instead of reading a stale one (run 0054: relocalised at 0.61,
        judged "still lost" at the 0.31 published a moment earlier, never resumed)."""
        self._send(connection, {"event": "searching", "fit": self.fit})
        self.get_logger().info(f"searching the whole map (fit {self.fit:.2f})")
        if not self._relocalize.wait_for_service(timeout_sec=2.0):
            self._send(connection, {"event": "error", "detail": "the tracker is not up"})
            return False
        result = self._wait(self._relocalize.call_async(Trigger.Request()), 60.0)
        detail = result.message if result else "no answer"
        deadline = time.monotonic() + 15.0  # an episode is two whole-map searches, 4-7 s each here
        while self.fit < GOOD_FIT and time.monotonic() < deadline:
            time.sleep(0.2)
        self._send(connection, {"event": "searched", "detail": detail, "fit": self.fit})
        if self.fit >= GOOD_FIT:
            self.get_logger().info(f"found myself: fit {self.fit:.2f}")
            return True
        self.get_logger().warning(f"still lost after the search: fit {self.fit:.2f}")
        self._send(connection, {"event": "error", "detail": f"still lost (fit {self.fit:.2f})"})
        return False

    def _pose_msg(self, x: float, y: float, yaw_deg: float) -> PoseStamped:
        """A goal pose in the map frame."""
        message = PoseStamped()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.position.x = x
        message.pose.position.y = y
        message.pose.orientation.z = math.sin(math.radians(yaw_deg) / 2.0)
        message.pose.orientation.w = math.cos(math.radians(yaw_deg) / 2.0)
        return message


def main() -> None:
    rclpy.init()
    node = GoalServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_recording()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
