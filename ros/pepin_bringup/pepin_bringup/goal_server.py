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
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

from pepin_bringup.run_recorder import RunRecorder

PORT = 3337
GOOD_FIT = 0.45  # below this the robot is told to find itself before it drives
# The planner to select, and the controller that follows it. One controller now: the lattice
# planner no longer expands in reverse, so there is nothing a reversing controller would add.
PLANNERS = {
    "navfn": ("GridBased", "FollowPath"),
    "lattice": ("Lattice", "FollowPath"),
}


class GoalServer(Node):
    """Takes goals over a socket and drives them through Nav2, with the run recorded."""

    def __init__(self) -> None:
        super().__init__("goal_server")
        self._places_path = Path(str(self.declare_parameter("places", "/maps/places.yaml").value))
        self._record_dir = Path(str(self.declare_parameter("record_dir", "/maps/rec").value))
        self._port = int(self.declare_parameter("port", PORT).value)
        self._client = ActionClient(self, NavigateToPose, "navigate_to_pose")
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
        self._planner_path = self._record_dir.parent / "planner.txt"
        self.planner = "navfn"
        with contextlib.suppress(OSError):
            self.pick_planner(self._planner_path.read_text().strip())
        self._goal_handle: Any = None
        self._recorder = RunRecorder(self, self._record_dir)
        self._lock = threading.Lock()
        threading.Thread(target=self._serve, daemon=True).start()
        self.get_logger().info(f"goal server ready on port {self._port}")

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

    def start_recording(self, name: str) -> Path:
        """Open this run's tape and return its path; the seconds before the goal are already on it.

        The recorder is this node, not a child process: rclpy takes about four seconds to come up
        on this board, so a per-goal recorder missed exactly the first turn of every drive.
        """
        path = self._recorder.start(name)
        self.get_logger().info(f"run {self._recorder.number}: recording {path}")
        return path

    def stop_recording(self) -> None:
        """Close the run's tape (flushed and synced); harmless when no run is open."""
        self._recorder.stop()

    # -- the socket ------------------------------------------------------------

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
        """Stop the running goal, if any; True when there was one."""
        with self._lock:
            handle, self._goal_handle = self._goal_handle, None
        if handle is None:
            return False
        handle.cancel_goal_async()
        return True

    def _go(self, request: dict[str, Any], connection: socket.socket) -> None:
        """Send one goal and stream its progress until it ends or the caller hangs up."""
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
        record = self.start_recording(name or f"{x:.0f}_{y:.0f}")
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
                    "place": name,
                    "x": x,
                    "y": y,
                    "yaw_deg": yaw_deg,
                    "sent_in_ms": round((time.monotonic() - started) * 1000),
                },
            )
            result_future = handle.get_result_async()
            last = 0.0
            while rclpy.ok() and not result_future.done():
                time.sleep(0.05)  # the node's own spin serves the action; this thread only reports
                now = time.monotonic()
                if feedback and now - last > 1.0:
                    last = now
                    self._send(
                        connection, {"event": "feedback", "t": round(now - started, 1), **feedback}
                    )
            with self._lock:
                self._goal_handle = None
            outcome = result_future.result()
            status = getattr(outcome, "status", 0) if outcome else 0
            self.stop_recording()  # closed before the answer: the caller fetches it on reading
            self._send(
                connection,
                {
                    "event": "done",
                    "status": int(status),
                    "seconds": round(time.monotonic() - started, 1),
                    "arrival": self._pose_now(),
                    "recording": str(record),
                },
            )
        finally:  # a refused goal or a broken connection must not leave a recorder running
            self.stop_recording()

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
        """A weak fit before driving buys one whole-map search; blind driving is never allowed."""
        self._send(connection, {"event": "searching", "fit": self.fit})
        if not self._relocalize.wait_for_service(timeout_sec=2.0):
            self._send(connection, {"event": "error", "detail": "the tracker is not up"})
            return False
        result = self._wait(self._relocalize.call_async(Trigger.Request()), 60.0)
        self._send(
            connection, {"event": "searched", "detail": result.message if result else "no answer"}
        )
        if self.fit >= GOOD_FIT:
            return True
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
