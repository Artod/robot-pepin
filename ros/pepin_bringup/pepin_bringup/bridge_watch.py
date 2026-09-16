"""Watch the bridge from the laptop: whose bridge it is, and whether the topics really flow.

Three failures, one repair.

*A different bridge.* Subscriptions made against one bridge do not follow it through a restart:
after the board rebooted, the laptop's costmap kept an old transform listener that never saw the
new bridge's /tf, no plan was ever produced and the tree spun the cart for 139 s (run 0148).
This node polls the board bridge's REST admin; when its zenoh id changes, or the admin has
answered nothing for a minute, or it answers for the first time to a watch that started without
it (``pepin.deployment.BridgeIdentity``), it waits for the routes to settle and repairs.

*A route with no endpoint of its own.* Restarting this whole half on that identity change is
what broke the next thing (2026-09-14): the laptop bridge came back with a pub route for /vo
that had the publisher, had the board's matching route, and had ``dds_reader ""`` — no DDS
reader, so the EKF received not one visual-odometry message while every count in the admin
looked right. The bridge builds a pub route's reader once and never revises it, so a publisher
that declared after the bridge started leaves it readerless for the bridge's life. That is a
fact about the admin's JSON, not a wait (:attr:`pepin.deployment.BridgeRoute.dead`), and it is
read from the same reply this watch already fetches.

*A route with no endpoint on the OTHER side.* The same fault, read from the same reply, about
the board's own routes (2026-09-15): a five-second wireless stall on a reliable 50 Hz topic
filled the board bridge's transmission queue, it closed the transport itself ("Unable to push
non droppable network message ... Closing transport!") and reconnected with the same zenoh id
and thirteen pub routes whose ``dds_reader`` was empty. Not one message crossed from the board
for the rest of the evening, while this watch printed "dead routes 0" — it judged the laptop's
routes alone (:func:`pepin.deployment.far_dead_routes` is the other half). This side cannot mend
it: the board's bridge is what must be restarted, and it must be restarted AFTER this side's,
because of two bridges the one that starts last is the one that gets working routes. So the
repair ladder ends with a kick to the board (:mod:`pepin_bringup.bridge_kick`), never with ssh.

*A route that carries nothing.* The route count says nothing about flow: a route exists on both
admins, the far side's publisher exists, and not one message crosses — /depth_scan on
2026-09-12, /imu/data_raw on 2026-09-13, each cured by restarting a bridge by hand. The cause is
in ``pepin.deployment.BRIDGED_QOS``: a route's DDS QoS is fixed by whichever declaration created
it and is never revised, so a route built from the far bridge's announcement can carry a reader
that never matches the local writer. So this watch counts messages. It subscribes to every topic
BOTH bridges agree should arrive here — the far side has a publisher, some node here is waiting
— and when the count stands still for ``flow_silence_s`` it repairs, least destructive first:
the laptop's bridge container alone (the fusion model and RTAB-Map's database live on), and the
whole half only if that did not bring the topic back. All three faults take that same repair,
and the whole half stays one ``bridge_restart false`` away.

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
from std_msgs.msg import String

from pepin.deployment import (
    BRIDGE_KICK_TOPIC,
    CONTAINER_STOP_TIMEOUT_S,
    BridgeIdentity,
    FlowWatch,
    TopicFlow,
    allowed_names,
    bridge_routes,
    bridge_zid,
    bridged_qos,
    dead_routes,
    far_dead_routes,
    routes_settled,
    topic_flows,
)
from pepin.flags import UNMEASURED, Flag, FlagSet
from pepin_bringup.node_kit import Switches, Worker, bridged_qos_profile, spin_main

BRIDGE_CHANGED_EXIT = 3
REPAIR_COOLDOWN_S = (
    120.0  # after a gentle repair that did not help, wait this long before the next one
)
# The kick to the board's bridge (pepin_bringup.bridge_kick), the step after the gentle repair.
# Its cooldown is the long one: a board bridge restart takes ~25 s (its unit waits for the
# stack's last node and 20 s more) and costs the robot every bridged topic for that time, so it
# is asked for at most once in five minutes however long the fault lasts.
KICK_COOLDOWN_S = 300.0
# ...and for this long after a kick a NEW board bridge is the answer to it, not a fault: the
# board's bridge comes back with a new zenoh id by construction, and repairing on that would be
# an endless round of restarts.
KICK_PATIENCE_S = 120.0
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
# This watch's repair is a stop like any other here: the daemon is told the same window
# ros/lib.sh and board/pepin-ros.service use, so a container it restarts gets its stop signal
# and the seconds to answer it instead of a five-second kill. The bridge itself is out in one,
# and `docker stop` returns as soon as the container is down — the window is a ceiling, not a
# wait — so the number costs nothing here and keeps one way to stop a container in the repo.
HTTP_SLACK_S = 15.0  # the socket must outlive the restart the daemon is doing on it

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
        "dead_routes",
        True,
        description="a route wired at both ends and missing its own DDS endpoint — a pub route"
        " with local publishers and a remote route but no dds_reader, a sub route with local"
        " subscribers and a remote route but no dds_writer — is repaired without waiting for the"
        " silence to be counted; off, only the message counters of flow_watch can find it",
        why="2026-09-14, after the board's bridge changed identity: the laptop bridge's pub route"
        " for /vo had local_nodes ['/visual_odometry'] and a remote route and dds_reader \"\","
        " while /depth_scan's route beside it had its reader and carried. The EKF got no visual"
        " odometry and nothing in the admin said so — the route count was right. The only cure"
        " was restarting the laptop's bridge alone, with every node alive. The endpoints are in"
        " the same REST reply this watch already fetches, so the test costs no query: a dead"
        " route is a fact about the JSON, not a twenty-second wait for a counter that will never"
        " move (a live dump on 2026-09-14 15:5x, 62 routes, had 0 dead)",
        on_when="always on a split stack: it is the earliest and cheapest signal there is",
        off_when="while bisecting the bridge by hand, or if a bridge version ever built a route's"
        " endpoint lazily enough that a healthy route reads as dead",
    ),
    Flag(
        "board_routes",
        True,
        description="the BOARD's own routes are judged too: a pub route of the board's bridge"
        " with publishers, a remote route naming this bridge and no dds_reader carries nothing"
        " and is counted in the report line as 'board routes without a reader N'; off, only"
        " this side's routes are judged, as before",
        why="2026-09-15, the evening every topic stopped: this watch printed 'dead routes 0' for"
        " an hour while thirteen of the board's pub routes had an empty dds_reader and nothing"
        " crossed at all. dead_routes judged this bridge's routes alone, and the board's are in"
        " the same network-wide admin reply this watch already fetches (the admin space is"
        " network-wide: either bridge answers for both). The fault is invisible from every other"
        " signal — the route count is right, the far side's publishers are alive, and the"
        " message counters only say 'silent', which a starved wifi link says too",
        on_when="always on a split or vision stack: it is the difference between 'the link is"
        " slow' and 'the board's bridge must be restarted'",
        off_when="while bisecting the bridge by hand",
    ),
    Flag(
        "bridge_kick",
        True,
        description="when a fault survives the gentle repair, ask the BOARD to restart its own"
        " bridge (one String on /bridge/kick; the board's run recorder touches a flag file and a"
        " systemd path unit there does the restart); off, the ladder ends at the gentle repair"
        " and the log, as before",
        why="of two bridges the one that starts LAST gets working routes — a route's DDS"
        " endpoint is built when the route is created and only while the far bridge is already"
        " announcing — which is why ros/laptop.sh restarts the board's bridge over ssh"
        " (settle_bridge) right after the laptop's, and why restarting this side alone could not"
        " cure 2026-09-15: the readerless routes were the board's. This container has no ssh key"
        " and must not have one, so the request crosses as a topic and the board's own systemd"
        " does the restart. It is sent only after a restart of this side's bridge, so the order"
        f" that works is the order that happens. {UNMEASURED}",
        on_when="always once the board carries pepin-bridge-kick.path: it is the only repair"
        " for the board's own routes that does not need a human",
        off_when="on a board without the kick units installed (the message is then published"
        " into nothing), or while bisecting the bridge by hand",
    ),
    Flag(
        "bridge_restart",
        True,
        description="repair a dead route, a starved topic or a board bridge that changed identity"
        " by restarting the laptop's bridge container alone (Docker Engine API over"
        " /var/run/docker.sock); off, the repair is the old one — this whole half restarts, which"
        " throws away the fusion model and RTAB-Map's working set",
        why="the bridge offers nothing gentler: its REST admin is read-only in 1.7.0 (the"
        " running config prints permissions { read: true, write: false }), so there is no reload"
        " and no way to drop a single route. Restarting the container re-creates every route in"
        " a few seconds and leaves pepin-vslam alive. It falls back by itself when the docker"
        " socket is not mounted, and escalates to the whole half when the fault returns after a"
        " restart. It is also the answer to a board bridge that changed identity: on 2026-09-14"
        " restarting this whole half on that event left the laptop bridge's /vo route without a"
        " DDS reader, and what cured it was a restart of the laptop's bridge alone with the nodes"
        f" up. {UNMEASURED}",
        on_when="always: it is strictly less destructive than the fallback",
        off_when="when the laptop's bridge must not be touched — bisecting it by hand, or"
        " running without the docker socket mounted",
    ),
    Flag(
        "half_restart",
        False,
        description="when the gentle repair (the bridge alone) did not bring the routes back,"
        " end this process so the launch restarts the whole laptop half; off: say so in the"
        " log, keep everything alive, and retry the gentle repair after a cooldown",
        why="off since 2026-09-15: after every board restart the escalation killed the whole"
        " half within minutes (RestartCount 3 -> 6 in 45 min: rgbd_odometry, the fusion model,"
        " Foxglove's channels and RTAB-Map's writes all died with it) because a restarted bridge"
        " does not always re-match the nodes' subscriptions (/scan silent). A half that keeps"
        " running with one silent topic beats one that dies whole; ros/restart.sh laptop is"
        " the hand repair",
        on_when="a half whose nodes cannot be kicked one by one and whose routes never come back",
        off_when="always while the escalation costs more than the fault (today)",
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
    CLI in the image, no shell. ``t=`` is the same stop window every other stop here uses
    (:data:`pepin.deployment.CONTAINER_STOP_TIMEOUT_S`), so this repair is not a shortcut past
    it. ``connect`` is injectable so a test drives it with a fake.
    """

    def __init__(
        self,
        container: str = LAPTOP_BRIDGE_CONTAINER,
        socket_path: str = DOCKER_SOCKET,
        stop_timeout_s: float = CONTAINER_STOP_TIMEOUT_S,
        connect: Any = None,
    ) -> None:
        self._container = container
        self._socket_path = socket_path
        self._stop_timeout_s = stop_timeout_s
        self._timeout_s = stop_timeout_s + HTTP_SLACK_S
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
            connection.request(
                "POST", f"/containers/{self._container}/restart?t={self._stop_timeout_s:g}"
            )
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
        self._dead_since: dict[str, float] = {}  # topic -> when its route first read as dead
        self._far_since: dict[str, float] = {}  # the same, for the board's own routes
        self._reported: dict[str, int] = {}
        self._reported_at = 0.0
        self._attempts = 0
        self._kicked_first = False
        self._cooldown_until = 0.0
        self._kick_until = 0.0  # no second kick to the board before this
        self._expect_board_bridge = 0.0  # ...and until here a new board bridge is our own doing
        self._grace_s = grace_s
        self._started: float | None = None
        self._switches = Switches(self, FLAGS)
        # The only message this watch sends. Created here, on the executor thread, published
        # from the worker (rclpy's publish is thread-safe); the QoS is the topic's pinned one
        # so the route cannot be built from a race (pepin.deployment.BRIDGED_QOS).
        self._kick_pub = self.create_publisher(
            String, f"/{BRIDGE_KICK_TOPIC}", bridged_qos_profile(BRIDGE_KICK_TOPIC)
        )
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
            if zid is not None and now < self._expect_board_bridge:
                # We asked for this one: the board restarted its bridge because this watch
                # kicked it, and that is the repair finishing, not a fault. Every clock starts
                # again on the new routes, and the ladder starts from the top if they are wrong.
                self._expect_board_bridge = 0.0
                self._cooldown_until = now
                self._attempts = 0
                self._flow.repaired(now)
                self._dead_since.clear()
                self._far_since.clear()
                self.get_logger().info(
                    f"bridge watch: the board's bridge came back after our kick ({zid});"
                    " judging its routes again from now"
                )
                return
            why = f"is a new one ({zid})" if zid else f"answered nothing for {SILENCE_S:.0f} s"
            # The board's bridge changed under us and this side's routes are the ones that come
            # back wrong: the gentle repair is the one that cured it on 2026-09-14, and the whole
            # half stays one bridge_restart=false away.
            self.mend(f"the board's bridge {why}", now, settle=True)
            return
        watching = (
            self._switches.on("flow_watch")
            or self._switches.on("dead_routes")
            or self._switches.on("board_routes")
        )
        if watching and zid is not None:
            self.check_flow(now)

    def check_flow(self, now: float) -> None:
        """Ask both bridges what should arrive here, count it, repair what is silent or dead."""
        self._flow.silence_s = float(self._switches["flow_silence_s"])
        flows, dead, far_dead = self.flows()
        if self._switches.on("flow_watch"):
            for flow in flows:
                expected = flow.judged  # periodic and due; a latched or on-demand topic is never
                if expected:
                    self._wanted.setdefault(flow.topic, flow.type_name)
                counted = flow.topic not in self._unknown  # an unreadable type is never starved
                self._flow.observe(
                    flow.topic, self._counts.get(flow.topic, 0), expected and counted, now
                )
        starved = self._flow.starved(now) if self._switches.on("flow_watch") else ()
        rotten = self.rotten(
            self._dead_since, dead if self._switches.on("dead_routes") else (), now
        )
        far_rotten = self.rotten(
            self._far_since, far_dead if self._switches.on("board_routes") else (), now
        )
        if now - self._reported_at >= REPORT_S:
            self.report(flows, starved, dead, far_dead, now)
        if starved or rotten or far_rotten:
            self.mend(
                self.fault(starved, rotten, far_rotten),
                now,
                grace=True,
                far=bool(far_rotten) and not bool(rotten),
            )
        elif self._flow.settled(now):
            self._attempts = 0  # a link healthy well past the last repair earns the gentle one

    def flows(self) -> tuple[tuple[TopicFlow, ...], tuple[str, ...], tuple[str, ...]]:
        """What both bridges say about every allowed incoming topic, the topics whose route on
        this side is dead by its own endpoints (:func:`pepin.deployment.dead_routes`) and the
        topics the BOARD publishes to us with no reader of its own
        (:func:`pepin.deployment.far_dead_routes`).

        One query for all three: the zenoh admin space is network-wide, so the board's admin
        answers for the laptop's bridge too — and for its own routes, which is what makes the
        far side judgeable from here at no extra cost.
        """
        if self._local_zid is None:
            self._local_zid = bridge_zid(fetch(f"{self._local_admin}/@/local/router") or "")
        if not self._allowed:
            self._allowed = self.allowed()
        if self._local_zid is None:
            return (), (), ()
        routes = bridge_routes(fetch(f"http://{self._board}:8000/@/*/ros2/route/**", 6.0) or "")
        dead = dead_routes(routes, self._local_zid)
        far_dead = far_dead_routes(routes, self._local_zid)
        if not self._allowed:
            return (), dead, far_dead
        flows = topic_flows(routes, self._local_zid, self._allowed, watcher=WATCH_NODE)
        return flows, dead, far_dead

    def rotten(self, since: dict[str, float], dead: tuple[str, ...], now: float) -> tuple[str, ...]:
        """The dead routes of ``since`` (this side's clocks or the board's) that have stayed
        dead for the patience (``flow_silence_s``), so a route caught in the seconds between its
        creation and its endpoint is not a fault. ``since`` is updated in place: a route that
        came back loses its clock."""
        for topic in dead:
            since.setdefault(topic, now)
        for topic in [t for t in since if t not in dead]:
            del since[topic]
        return tuple(t for t, first in since.items() if now - first >= self._flow.silence_s)

    @staticmethod
    def fault(
        starved: tuple[str, ...], rotten: tuple[str, ...], far_rotten: tuple[str, ...] = ()
    ) -> str:
        """One line naming what is wrong, for the log and for the escalation."""
        said = []
        if starved:
            said.append(f"{' '.join(starved)} carried nothing")
        if rotten:
            said.append(f"{' '.join(rotten)} has a route with no DDS endpoint")
        if far_rotten:
            said.append(f"the board's route for {' '.join(far_rotten)} has no reader")
        return " and ".join(said)

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

    def mend(
        self,
        why: str,
        now: float,
        grace: bool = False,
        settle: bool = False,
        far: bool = False,
    ) -> None:
        """Put the routes back, least destructive first: the laptop's bridge container alone,
        then a kick to the board's bridge, and the whole half only after both.

        One ladder for every fault — a starved topic, a dead route here, a readerless route on
        the board, a board bridge that changed identity — because one thing cured all of them by
        hand: a bridge restarted, and the board's restarted AFTER this side's. That order is the
        ladder's shape, not a coincidence: of two bridges the one that starts last is the one
        that gets working routes, so the kick is only ever sent once this side's bridge is new.

        ``grace`` holds off while this half is younger than :data:`STARTUP_GRACE_S` (the routes
        are still being built, so a fault read off them is not one); ``settle`` waits for the
        board's route count to stop moving first, which a new bridge needs and a dead route does
        not.
        """
        if grace and self._started is not None and now - self._started < self._grace_s:
            self.get_logger().warning(
                f"bridge watch: {why} while this half is still coming up; waiting"
            )
            return
        # Whose bridge must restart is not a preference, it is the mechanism: a route's DDS
        # endpoint is built when the route is created and only if the far bridge is already
        # announcing, so THE BRIDGE THAT STARTS LAST is the one that ends up with working routes.
        # When it is the BOARD's own pub routes that carry no reader, restarting this side first
        # makes this side last and leaves the board's routes exactly as dead as they were — and
        # the kick that follows makes the board last again, which is the loop seen on 2026-09-16
        # after every `restart.sh board --deploy`. So the far side's fault is kicked FIRST.
        if (
            far
            and not self._kicked_first
            and self._switches.on("bridge_kick")
            and now >= self._kick_until
        ):
            self._kicked_first = True
            self.kick(why, now)
            return
        if (
            self._switches.on("bridge_restart")
            and not self._attempts
            and now >= self._cooldown_until
            and self._repair.available()
        ):
            self._attempts += 1
            if settle:
                count = wait_for_routes(self._board, self._expected)
                self.get_logger().info(f"bridge watch: routes settled at {count}")
            try:
                said = self._repair.restart()
            except Exception as exc:
                self.restart_half(f"{why} and the bridge would not restart: {exc}")
                return
            self._flow.repaired(now)
            self._kicked_first = False
            self._dead_since.clear()  # fresh routes: every clock starts again
            self._far_since.clear()
            self._local_zid = None  # a new bridge has a new id; its routes are new too
            self.get_logger().error(f"bridge watch: {why}; {said}")
            return
        if self._attempts and self._switches.on("bridge_kick") and now >= self._kick_until:
            self.kick(why, now)
            return
        if self._switches.on("half_restart"):
            self.restart_half(f"{why} {self.stalled(now)}")
            return
        self.get_logger().error(
            f"bridge watch: {why} {self.stalled(now)}; half_restart is off — this half stays up;"
            f" the hand repair is ros/restart.sh laptop; the gentle repair may run again in"
            f" {max(0.0, self._cooldown_until - now):.0f} s"
        )
        self._attempts = 0
        # Armed only when it is not already running: re-arming it on every failing round (five
        # seconds apart) pushed it forever into the future, so the gentle repair, which is
        # allowed once per cooldown, never ran a second time at all (2026-09-15).
        if now >= self._cooldown_until:
            self._cooldown_until = now + REPAIR_COOLDOWN_S

    def stalled(self, now: float) -> str:
        """Why nothing gentler is happening, for the log: which rung of the ladder is spent,
        which cooldown is running, or which switch is off."""
        if self._attempts and not self._switches.on("bridge_kick"):
            return "again after a bridge restart, and the kick to the board is off"
        if self._attempts:
            return (
                "again after a bridge restart, and the board was kicked less than"
                f" {KICK_COOLDOWN_S:.0f} s ago ({self._kick_until - now:.0f} s to go)"
            )
        if now < self._cooldown_until:
            return f"and the gentle repair is cooling down ({self._cooldown_until - now:.0f} s)"
        return "and the gentle repair is off"

    def kick(self, why: str, now: float) -> None:
        """Ask the board to restart its own bridge: one message, and the clocks that keep this
        watch from asking again or from reading the answer as a new fault.

        The board's bridge comes back with a new zenoh id ~25 s later (its unit waits for the
        stack), and that identity change is the repair landing, not a bridge that changed under
        us — :data:`KICK_PATIENCE_S` is how long this watch remembers that it asked.
        """
        self._kick_until = now + KICK_COOLDOWN_S
        self._expect_board_bridge = now + KICK_PATIENCE_S
        self._cooldown_until = now + REPAIR_COOLDOWN_S  # nothing else while the board comes back
        self._attempts = 0
        self._flow.repaired(now)
        self._dead_since.clear()
        self._far_since.clear()
        self._kick_pub.publish(String(data=why))
        self.get_logger().error(
            f"bridge watch: {why} after a bridge restart; asked the board to restart its own"
            f" bridge (/{BRIDGE_KICK_TOPIC}); it is gone for ~25 s and comes back with a new"
            f" zenoh id. Nothing else here for {REPAIR_COOLDOWN_S:.0f} s, no second kick for"
            f" {KICK_COOLDOWN_S:.0f} s (board: journalctl -u pepin-bridge-kick -u pepin-bridge)"
        )

    def restart_half(self, why: str) -> None:
        """The old action, kept: wait for the routes to settle, then end the process so the
        launch shuts this half down and the container's restart policy brings it back."""
        self.get_logger().error(f"bridge watch: {why}; waiting for the routes")
        count = wait_for_routes(self._board, self._expected)
        self.get_logger().error(f"bridge watch: routes settled at {count}; restarting this half")
        self._exit(BRIDGE_CHANGED_EXIT)

    def report(
        self,
        flows: tuple[TopicFlow, ...],
        starved: tuple[str, ...],
        dead: tuple[str, ...],
        far_dead: tuple[str, ...],
        now: float,
    ) -> None:
        """One line: what each watched topic carried since the last line, what is silent, how
        many routes have no DDS endpoint on this side and how many of the board's have no reader
        of their own, and the switches (CLAUDE.md rule 19)."""
        window = max(1e-3, now - self._reported_at)
        self._reported_at = now
        carried = []
        for flow in flows:
            if not flow.judged:
                continue
            total = self._counts.get(flow.topic, 0)
            since, self._reported[flow.topic] = total - self._reported.get(flow.topic, 0), total
            carried.append(f"{flow.topic.lstrip('/')} {since / window:.1f}")
        silent = f"; SILENT {' '.join(starved)}" if starved else ""
        rotten = f"; DEAD ROUTES {len(dead)} [{' '.join(dead)}]" if dead else "; dead routes 0"
        far = (
            f"; BOARD ROUTES WITHOUT A READER {len(far_dead)} [{' '.join(far_dead)}]"
            if far_dead
            else "; board routes without a reader 0"
        )
        self.get_logger().info(
            f"bridge watch: {len(carried)} topics Hz [{', '.join(carried)}]{silent}{rotten}{far};"
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
