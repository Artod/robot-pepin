"""Watch the bridge from the laptop: whose bridge it is, and whether the topics really flow.

Two failures, two answers.

*A different bridge.* Subscriptions made against one bridge do not follow it through a restart:
after the board rebooted, the laptop's costmap kept an old transform listener that never saw the
new bridge's /tf, no plan was ever produced and the tree spun the cart for 139 s (run 0148).
This node polls the board bridge's REST admin; when its zenoh id changes, or the admin has
answered nothing for a minute, or it answers for the first time to a watch that started without
it (``pepin.deployment.BridgeIdentity``), it waits for the routes to settle and ends the
process. The launch is told to shut down on that exit and the container's restart policy brings
the whole half back with fresh subscriptions.

*A route that carries nothing.* The route count says nothing about flow: a route exists on both
admins, the far side's publisher exists, and not one message crosses — /depth_scan on
2026-09-12, /imu/data_raw on 2026-09-13, each cured by restarting a bridge by hand. The cause is
in ``pepin.deployment.BRIDGED_QOS``: a route's DDS QoS is fixed by whichever declaration created
it and is never revised, so a route built from the far bridge's announcement can carry a reader
that never matches the local writer. So this watch counts messages. It subscribes to every topic
BOTH bridges agree should arrive here — the far side has a publisher, some node here is waiting
— and when the count stands still for ``flow_silence_s`` it repairs, least destructive first:
the laptop's bridge container alone (the fusion model and RTAB-Map's database live on), and the
whole half only if that did not bring the topic back.

The watch never subscribes to a topic no local node has asked for: its subscription would
otherwise create the route, pin that route's QoS to the watch's own, and pay for a topic to
cross the wireless hop that nobody reads. That is also why its own name is not in
``pepin.deployment.laptop_launch_nodes``: it joins only routes that other nodes hold, so its
ghost can never be the last local node of one, and a ghost wait for a name that starts beside
the wait would never end.

Threads: every admin query and every repair is slow (seconds of HTTP, up to two minutes of
waiting for routes) and runs on a worker thread; the executor thread only counts messages and
attaches new subscriptions, so a stalled admin never costs a single message.

    python3 -m pepin_bringup.bridge_watch <board host> [<local bridge admin>]
"""

from __future__ import annotations

import http.client
import os
import socket
import sys
import time
from typing import Any, Protocol

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from pepin.deployment import (
    BridgeIdentity,
    FlowWatch,
    TopicFlow,
    allowed_names,
    bridge_routes,
    bridge_zid,
    bridged_qos,
    routes_settled,
    topic_flows,
)
from pepin.flags import UNMEASURED, Flag, FlagSet
from pepin_bringup.node_kit import Switches, Worker, spin_main

BRIDGE_CHANGED_EXIT = 3
POLL_S = 5.0
ATTACH_S = 1.0
REPORT_S = 60.0
SETTLE_S = 15.0  # routes stable this long after the change: the board's nodes are all declared
SETTLE_PATIENCE_S = 120.0
SILENCE_S = 60.0  # an admin silent this long is a wedged or absent bridge: restart the half
# No repair in the first minute of this half's life: the routes are still being built, and
# for the first ten seconds the admin still lists the previous containers' nodes as the local
# nodes of every route (the DDS lease; pepin_bringup.ghost_wait waits them out for the ROS
# nodes, but this watch starts beside it, not after it).
STARTUP_GRACE_S = 60.0
WATCH_NODE = "bridge_watch"
LAPTOP_BRIDGE_CONTAINER = "pepin-zenoh"
DOCKER_SOCKET = "/var/run/docker.sock"

FLAGS = FlagSet(
    Flag(
        "flow_watch",
        True,
        description="count the messages of every topic that should arrive on this side and"
        " repair a topic that carries nothing; off, this watch sees only the board bridge's"
        " identity and its route count, as before",
        why="a route count cannot see the failure that cost two evenings: /depth_scan"
        " (2026-09-12) and /imu/data_raw (2026-09-13) each had a route on both admins and a live"
        " publisher on the far side, and carried zero messages until a bridge was restarted by"
        " hand. The mechanism is read off the bridge's own source and its admin"
        " (pepin.deployment.BRIDGED_QOS): a route's DDS QoS is whatever the declaration that"
        " created it carried, and it is never revised. The subscriptions this flag adds are"
        " free — every one of them is on a topic some other node here already receives",
        on_when="always on a split or vision stack: it is the only thing that can tell a dead"
        " route from a quiet one",
        off_when="while bisecting the bridge by hand, so the watch takes no action of its own",
    ),
    Flag(
        "flow_silence_s",
        20.0,
        range=(5.0, 300.0),
        description="seconds a topic both bridges say should flow may carry nothing before it"
        " counts as a dead route",
        why="the topics watched here are the periodic ones — /scan at 10 Hz, /tf at 40, /imu at"
        " 48, /localization/sources at 1 — so twenty seconds is twenty missed messages of the"
        " slowest of them and four polls of this watch, while still shorter than a goal. A"
        " topic published on a change only (/pepin/run_status) is never starved: nothing"
        " publishes it between changes, and the watch only counts what both bridges call"
        f" live. {UNMEASURED}",
        on_when="shorten it when a dead route must be caught inside a drive",
        off_when="lengthen it on a congested link, where a ten-second stall is the wireless hop"
        " and not the bridge",
    ),
    Flag(
        "bridge_restart",
        True,
        description="repair a dead route by restarting the laptop's bridge container alone"
        " (Docker Engine API over /var/run/docker.sock); off, the repair is the old one — this"
        " whole half restarts, which throws away the fusion model and RTAB-Map's working set",
        why="the bridge offers nothing gentler: its REST admin is read-only in 1.7.0 (the"
        " running config prints permissions { read: true, write: false }), so there is no reload"
        " and no way to drop a single route. Restarting the container re-creates every route in"
        " a few seconds and leaves pepin-vslam alive. It falls back by itself when the docker"
        " socket is not mounted, and escalates to the whole half when the silence returns after"
        f" a restart. {UNMEASURED}",
        on_when="always: it is strictly less destructive than the fallback",
        off_when="when the laptop's bridge must not be touched — bisecting it by hand, or"
        " running without the docker socket mounted",
    ),
)


class Repair(Protocol):
    """Something that can put the bridge's routes back: the watch's least destructive action."""

    def available(self) -> bool:
        """Whether this repair can be attempted at all right now."""

    def restart(self) -> str:
        """Do it; returns one line saying what happened, for the log."""


class DockerRestart:
    """Restarts one container through the Docker Engine API on the mounted docker socket.

    The bridge is a container beside this one, not a process in it, so the only handle on it is
    the daemon: ``POST /containers/<name>/restart``. Plain HTTP over a UNIX socket — no docker
    CLI in the image, no shell. ``connect`` is injectable so a test drives it with a fake.
    """

    def __init__(
        self,
        container: str = LAPTOP_BRIDGE_CONTAINER,
        socket_path: str = DOCKER_SOCKET,
        timeout_s: float = 30.0,
        connect: Any = None,
    ) -> None:
        self._container = container
        self._socket_path = socket_path
        self._timeout_s = timeout_s
        self._connect = connect

    def available(self) -> bool:
        """Whether the docker socket is mounted into this container."""
        return self._connect is not None or os.path.exists(self._socket_path)

    def restart(self) -> str:
        """Restart the container; the daemon's answer as one line."""
        connection = (
            self._connect()
            if self._connect is not None
            else _UnixSocketConnection(self._socket_path, self._timeout_s)
        )
        try:
            connection.request("POST", f"/containers/{self._container}/restart?t=5")
            reply = connection.getresponse()
            body = reply.read().decode("utf-8", "replace").strip()
            return f"restarted {self._container}: HTTP {reply.status} {body or 'ok'}"
        finally:
            connection.close()


class _UnixSocketConnection(http.client.HTTPConnection):
    """An HTTP connection over a UNIX socket path instead of a TCP port."""

    def __init__(self, path: str, timeout_s: float) -> None:
        super().__init__("localhost", timeout=timeout_s)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


def fetch(url: str, timeout_s: float = 3.0) -> str | None:
    """The body at ``url``, or ``None`` when it did not answer within ``timeout_s``."""
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as reply:
            return str(reply.read().decode("utf-8", "replace"))
    except Exception:
        return None


def route_count(board: str) -> int:
    """How many routes the bridges declare, counted the cheap way: the settle test only needs
    the number to stop moving."""
    text = fetch(f"http://{board}:8000/@/*/ros2/route/**", 6.0) or ""
    return text.count('"key"')


def wait_for_routes(board: str, expected: int | None) -> int:
    """Poll the new bridge's route count until it settles (pepin.deployment.routes_settled)
    or the patience runs out; the count reached."""
    last, since, stable_since = -1, time.monotonic(), time.monotonic()
    while time.monotonic() - since < SETTLE_PATIENCE_S:
        count = route_count(board)
        if count != last:
            last, stable_since = count, time.monotonic()
        elif routes_settled(count, expected, time.monotonic() - stable_since, SETTLE_S):
            break
        time.sleep(POLL_S)
    return last


class BridgeWatch(Node):
    """Watches the board bridge's identity and the flow of every topic that should arrive here.

    ``repair`` is the gentle repair (a fake in tests), ``exit_with`` how the node ends the
    process when only a restart of this half will do, ``grace_s`` how long after the first round
    no repair is attempted (:data:`STARTUP_GRACE_S`).
    """

    def __init__(
        self,
        board: str,
        local_admin: str,
        repair: Repair | None = None,
        exit_with: Any = None,
        grace_s: float = STARTUP_GRACE_S,
    ) -> None:
        super().__init__(WATCH_NODE)
        self._board = board
        self._local_admin = local_admin.rstrip("/")
        self._repair: Repair = repair if repair is not None else DockerRestart()
        self._exit = exit_with if exit_with is not None else os._exit
        self._identity = BridgeIdentity(silence_s=SILENCE_S)
        self._flow = FlowWatch()
        self._expected: int | None = None  # the healthy bridge's route count, at first contact
        self._local_zid: str | None = None
        self._allowed: tuple[str, ...] = ()
        self._wanted: dict[str, str] = {}  # topic -> ROS type, filled by the worker
        self._probes: dict[str, Any] = {}  # topic -> subscription
        self._unknown: set[str] = set()  # topics whose ROS type this image cannot resolve
        self._counts: dict[str, int] = {}
        self._reported: dict[str, int] = {}
        self._reported_at = 0.0
        self._attempts = 0
        self._grace_s = grace_s
        self._started: float | None = None
        self._switches = Switches(self, FLAGS)
        self._worker = Worker[float](
            self.round, name="bridge_watch", on_error=self.get_logger().error
        ).start()
        self.create_timer(POLL_S, self.tick)
        self.create_timer(ATTACH_S, self.attach)
        self.get_logger().info(
            f"bridge watch: {board}:8000, local {self._local_admin}; {self._switches.state()}"
        )

    def close(self) -> None:
        """Stop the worker thread and wait for it (``node_kit.spin_main``)."""
        self._worker.stop()

    # ---- the executor thread: counting, and attaching to what the worker asked for --------

    def tick(self) -> None:
        """Ask the worker for a round. A round still running keeps its tick; the next one
        replaces it, so a slow admin cannot build a backlog."""
        self._worker.offer(time.monotonic())

    def attach(self) -> None:
        """Subscribe to every topic the worker asked for and is not being counted yet.

        BEST_EFFORT unless the topic has a pinned QoS: a best-effort reader matches a writer of
        either reliability, so the probe itself can never be the endpoint that fails to match,
        and it asks the bridge for no retransmission of its own.
        """
        for topic, type_name in list(self._wanted.items()):
            if topic in self._probes:
                continue
            message = self.message_class(type_name)
            if message is None:
                self._unknown.add(topic)
                self._wanted.pop(topic, None)
                continue
            pinned = bridged_qos(topic)
            reliability = (
                ReliabilityPolicy.RELIABLE
                if pinned and pinned[0] == "reliable"
                else ReliabilityPolicy.BEST_EFFORT
            )
            self._counts.setdefault(topic, 0)
            self._probes[topic] = self.create_subscription(
                message,
                topic,
                lambda _msg, topic=topic: self.count(topic),
                QoSProfile(depth=pinned[1] if pinned else 1, reliability=reliability),
            )
            self.get_logger().info(f"bridge watch: following {topic} ({type_name})")

    def message_class(self, type_name: str) -> Any:
        """The message class of a ROS type name ("sensor_msgs/msg/Imu"), or ``None`` with a
        warning: one unknown type must not stop the watch following the rest."""
        try:
            from rosidl_runtime_py.utilities import get_message

            return get_message(type_name)
        except Exception as exc:
            self.get_logger().warning(f"bridge watch: no message class for {type_name}: {exc}")
            return None

    def count(self, topic: str) -> None:
        """One message arrived on ``topic``."""
        self._counts[topic] = self._counts.get(topic, 0) + 1

    # ---- the worker thread: the admins, the verdict, the repair --------------------------

    def round(self, now: float) -> None:
        """One round: the board bridge's identity, then the flow of everything due here."""
        if self._started is None:
            self._started = now
        zid = bridge_zid(fetch(f"http://{self._board}:8000/@/local/router") or "")
        if zid is not None and self._expected is None:
            self._expected = route_count(self._board) or None
            self.get_logger().info(f"bridge watch: {zid} has {self._expected or 0} routes")
        if self._identity.observe(zid, now):
            why = f"is a new one ({zid})" if zid else f"answered nothing for {SILENCE_S:.0f} s"
            self.restart_half(f"the board's bridge {why}")
            return
        if self._switches.on("flow_watch") and zid is not None:
            self.check_flow(now)

    def check_flow(self, now: float) -> None:
        """Ask both bridges what should arrive here, count it, repair what is silent."""
        self._flow.silence_s = float(self._switches["flow_silence_s"])
        flows = self.flows()
        for flow in flows:
            expected = flow.should_flow
            if expected:
                self._wanted.setdefault(flow.topic, flow.type_name)
            counted = flow.topic not in self._unknown  # an unreadable type is never starved
            self._flow.observe(
                flow.topic, self._counts.get(flow.topic, 0), expected and counted, now
            )
        starved = self._flow.starved(now)
        if now - self._reported_at >= REPORT_S:
            self.report(flows, starved, now)
        if starved:
            self.mend(starved, now)
        elif self._flow.settled(now):
            self._attempts = 0  # a link healthy well past the last repair earns the gentle one

    def flows(self) -> tuple[TopicFlow, ...]:
        """What both bridges say about every allowed incoming topic. One query: the zenoh admin
        space is network-wide, so the board's admin answers for the laptop's bridge too."""
        if self._local_zid is None:
            self._local_zid = bridge_zid(fetch(f"{self._local_admin}/@/local/router") or "")
        if not self._allowed:
            self._allowed = self.allowed()
        if self._local_zid is None or not self._allowed:
            return ()
        routes = bridge_routes(fetch(f"http://{self._board}:8000/@/*/ros2/route/**", 6.0) or "")
        return topic_flows(routes, self._local_zid, self._allowed, watcher=WATCH_NODE)

    def allowed(self) -> tuple[str, ...]:
        """The topics this side's bridge may bring in, read from the bridge's own config through
        its admin: the mode's list, from the bridge that is actually running it."""
        import json

        try:
            rows = json.loads(fetch(f"{self._local_admin}/@/local/ros2/config") or "")
            allow = rows[0]["value"]["allow"]["subscribers"]
        except (ValueError, TypeError, KeyError, IndexError):
            return ()
        return allowed_names(allow if isinstance(allow, str) else str(allow[0]))

    def mend(self, starved: tuple[str, ...], now: float) -> None:
        """Put the routes back, least destructive first: the laptop's bridge container alone,
        and the whole half only when a restart has already failed to bring the topics back."""
        topics = " ".join(starved)
        if self._started is not None and now - self._started < self._grace_s:
            self.get_logger().warning(
                f"bridge watch: {topics} silent while this half is still coming up; waiting"
            )
            return
        if self._switches.on("bridge_restart") and not self._attempts and self._repair.available():
            self._attempts += 1
            try:
                said = self._repair.restart()
            except Exception as exc:
                self.restart_half(
                    f"{topics} carried nothing and the bridge would not restart: {exc}"
                )
                return
            self._flow.repaired(now)
            self._local_zid = None  # a new bridge has a new id; its routes are new too
            self.get_logger().error(f"bridge watch: {topics} carried nothing; {said}")
            return
        why = "again after a bridge restart" if self._attempts else "and the gentle repair is off"
        self.restart_half(f"{topics} carried nothing {why}")

    def restart_half(self, why: str) -> None:
        """The old action, kept: wait for the routes to settle, then end the process so the
        launch shuts this half down and the container's restart policy brings it back."""
        self.get_logger().error(f"bridge watch: {why}; waiting for the routes")
        count = wait_for_routes(self._board, self._expected)
        self.get_logger().error(f"bridge watch: routes settled at {count}; restarting this half")
        self._exit(BRIDGE_CHANGED_EXIT)

    def report(self, flows: tuple[TopicFlow, ...], starved: tuple[str, ...], now: float) -> None:
        """One line: what each watched topic carried since the last line, what is silent, and
        the switches (CLAUDE.md rule 19)."""
        window = max(1e-3, now - self._reported_at)
        self._reported_at = now
        carried = []
        for flow in flows:
            if not flow.should_flow:
                continue
            total = self._counts.get(flow.topic, 0)
            since, self._reported[flow.topic] = total - self._reported.get(flow.topic, 0), total
            carried.append(f"{flow.topic.lstrip('/')} {since / window:.1f}")
        silent = f"; SILENT {' '.join(starved)}" if starved else ""
        self.get_logger().info(
            f"bridge watch: {len(carried)} topics Hz [{', '.join(carried)}]{silent};"
            f" {self._switches.state()}"
        )


def main(args: list[str] | None = None) -> None:
    """``python3 -m pepin_bringup.bridge_watch <board host> [<local bridge admin>]``."""
    argv = list(sys.argv[1:] if args is None else args)
    board = argv[0] if argv else os.environ.get("PEPIN_HOST", "10.0.0.187")
    admin = argv[1] if len(argv) > 1 else "http://pepin-zenoh:8000"
    spin_main(lambda: BridgeWatch(board, admin))


if __name__ == "__main__":
    main()
