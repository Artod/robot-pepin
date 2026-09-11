"""Which part of the stack runs where: the board keeps the reflexes, the laptop takes the rest.

The board is four Cortex-A53 cores. Whatever closes a control loop or owns a frame stays on
it: the sensors, the EKF, the tracker (map->odom), the controller with its local costmap, the
behaviours and the tree that orders them. Whatever answers once a second and tolerates a
wireless hop moves to the laptop: the planner with the global costmap, the goal server with its
recorder. The split is data, so a test can hold it and the launch file merely reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SIDES = ("all", "board", "laptop")

# Nav2 lifecycle nodes by side. "all" is the union: one machine, as before the split.
# In bring-up order: the tree last, because loading it needs the planner side's costmap service.
BOARD_NAV_NODES = ("controller_server", "behavior_server", "velocity_smoother", "bt_navigator")
LAPTOP_NAV_NODES = ("planner_server",)
MAP_NODES = ("map_server",)  # the map is served from the board: the tracker needs it there

HEARTBEAT_TOPIC = "laptop/heartbeat"
HEARTBEAT_HZ = 2.0

# The base's speed caps (config/base.json, the base server's own clamp). The C++ bridge on the
# board clamps /cmd_vel too, at 0.25 m/s by default: for half a day every tape sat at 0.20 and
# the bridge would have cut anything faster — one cap, the base's, passed to it at launch.
BASE_MAX_LINEAR_M_S = 0.30
BASE_MAX_ANGULAR_RAD_S = 1.0


def config_file(name: str) -> Path:
    """The path of ``config/<name>`` wherever this library runs: ``$PEPIN_CONFIG_DIR`` when
    set; else the ``config`` directory beside the ``pepin`` package (the board's container:
    /ws/pepin_src/config, put there by ros/sync.sh — the container mounts no /ws/config); else
    the one beside the source tree (a checkout: src/pepin/../../config; the laptop's containers:
    /ws/config). Raises ``FileNotFoundError`` naming every place looked, never a guess."""
    import os

    package = Path(__file__).resolve().parent
    homes = [Path(os.environ["PEPIN_CONFIG_DIR"])] if os.environ.get("PEPIN_CONFIG_DIR") else []
    homes += [package.parent / "config", package.parents[1] / "config"]
    for home in homes:
        if (home / name).is_file():
            return home / name
    raise FileNotFoundError(f"config/{name} is in none of {[str(h) for h in homes]}")


def nav_nodes(side: str) -> tuple[str, ...]:
    """The Nav2 lifecycle nodes the navigation manager on ``side`` must bring up."""
    if side == "all":
        return BOARD_NAV_NODES + LAPTOP_NAV_NODES
    if side == "board":
        return BOARD_NAV_NODES
    if side == "laptop":
        return LAPTOP_NAV_NODES
    raise ValueError(f"side must be one of {SIDES}, not {side!r}")


def runs_here(side: str, node: str) -> bool:
    """Whether a named piece runs on ``side``: Nav2 nodes, the map, the tracker, the goal server."""
    if node in MAP_NODES or node == "relocalizer":
        return side in ("all", "board")
    if node == "goal_server":  # it carries the laptop's heartbeat too
        return side in ("all", "laptop")
    if node == "run_recorder":  # the tape is written where the sensors are
        return side in ("all", "board")
    if node == "link_watch":
        return side == "board"  # only a split stack has a link to watch
    return node in nav_nodes(side)


@dataclass
class LinkWatch:
    """Cuts a drive when the laptop's heartbeat stops: a plan may never arrive, so stop now.

    With the planner on the laptop, a lost link means the tree on the board keeps following
    its last path with no one to replan around whatever appears. The controller and the local
    costmap still avoid what the lidar sees, so the cart is not blind — but it is deaf, and a
    deaf cart stops. ``patience_s`` covers a wireless hiccup; a link that stays silent longer
    is gone. The verdict is armed only while a goal is running, and fires once per outage.
    """

    patience_s: float = 2.5
    _last_beat: float | None = field(default=None, init=False)
    _cut: bool = field(default=False, init=False)

    def beat(self, now: float) -> None:
        """A heartbeat arrived."""
        self._last_beat = now
        self._cut = False

    def should_cut(self, navigating: bool, now: float) -> bool:
        """True while a running drive has had no heartbeat for the patience and the cut has not
        been sent yet. It does not consume itself: a node that could not reach the cancel
        service must be told again on the next tick, so ``cut_sent`` latches, not this."""
        if not navigating or self._cut:
            return False
        if self._last_beat is None:
            return False  # never heard the laptop: the stack is not split, nothing to watch
        return now - self._last_beat > self.patience_s

    def cut_sent(self) -> None:
        """The cancel went out: this outage is handled until the next heartbeat."""
        self._cut = True

    @property
    def alive(self) -> bool:
        return self._last_beat is not None and not self._cut


# Lifecycle transitions and states (lifecycle_msgs), by name so a test needs no ROS.
TRANSITION_CONFIGURE = 1
TRANSITION_ACTIVATE = 3


def autostart_for(side: str) -> bool:
    """Whether the navigation lifecycle manager on ``side`` activates its nodes by itself.

    A whole stack does. The board half does not: its tree cannot load until the planner side's
    global costmap answers, and a bring-up that fails once is aborted for good by the manager
    (2026-09-09, "Action server is inactive"). The laptop brings the board up instead, node by
    node, when it is there to answer — see :func:`next_transition`.
    """
    return side != "board"


def next_transition(states: dict[str, str]) -> tuple[str, int] | None:
    """The one lifecycle transition to send next so the board's Nav2 comes up, or ``None``.

    ``states`` maps each board node to its lifecycle state label. Nodes are walked in
    :data:`BOARD_NAV_NODES` order and the first that is not active gets its next step:
    unconfigured -> configure, inactive -> activate. A node in transit (activating, ...) or
    missing answers ``None``: wait and ask again. Sending one step at a time and re-reading the
    states makes the bring-up idempotent — a half-failed earlier attempt is simply continued.
    """
    for node in BOARD_NAV_NODES:
        state = states.get(node)
        if state == "active":
            continue
        if state == "unconfigured":
            return node, TRANSITION_CONFIGURE
        if state == "inactive":
            return node, TRANSITION_ACTIVATE
        return None
    return None


# What crosses the bridge, by direction. Each bridge is allowed only the publishers that live on
# its own side and the subscribers that live on the other: a bridge that may route a topic both
# ways discovers its own writer as a local publisher and loops the topic back until nothing
# crosses at all (scan and tf died the moment a second laptop launch subscribed, 2026-09-09).
BOARD_PUBLISHES = (
    "scan",
    "tf",
    "tf_static",
    "map",
    "odom",
    "odometry/filtered",
    "imu/data_raw",
    "tof/front",
    "tof/left",
    "tof/right",
    "tracker_pose",
    "localization_fit",
    "dynamic_obstacles",
    "local_costmap/costmap",
    "pepin/run_status",
    "neck/state",  # the neck's joint angles (pepin_bringup.neck_state); its transform rides /tf
)
LAPTOP_PUBLISHES = (
    "plan",
    "pepin/run",
    HEARTBEAT_TOPIC,
    "planner_selector",
    "controller_selector",
    "rtabmap/map",
    "rtabmap/mapGraph",
    "rtabmap/mapPath",
    "rtabmap/info",
    "depth_scan",  # the camera's depth folded onto the plane, for the board's local costmap
    "contact_scan",  # the same depth read at the floor: where bodies touch it (pepin.contact)
)
BOARD_SERVES = (
    "relocalize",
    "where_am_i",
    *(f"{node}/{srv}" for node in BOARD_NAV_NODES for srv in ("get_state", "change_state")),
)
LAPTOP_SERVES = ("global_costmap/clear_entirely_global_costmap",)
# Both navigators' trees load at activation and look for both planner actions: a missing
# through-poses server failed the board's bring-up ("Action server ... not available").
BOARD_ACTIONS = ("navigate_to_pose", "navigate_through_poses")
LAPTOP_ACTIONS = ("compute_path_to_pose", "compute_path_through_poses")


_Names = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]


def _names_regex(names: tuple[str, ...]) -> str:
    """One anchored regex over ROS names (with their leading slash) for the bridge's allow-list.
    No names at all is a regex no name matches: an empty list could read as "everything"."""
    return "^/(" + "|".join(sorted(names)) + ")$" if names else "^$"


# The bridge's two modes. "split" (ros/thin.sh on): the board keeps the reflexes, the laptop
# plans and takes goals. "vision" (ros/thin.sh vision): every drive stays on the board and the
# bridge carries topics only — RTAB-Map and the camera on the laptop, the operator's Foxglove
# there too — so what the laptop half would publish in the split (the plan, the global costmap)
# now comes FROM the board, and the laptop side publishes none of it (a topic allowed as a
# publisher on both sides loops). No services or actions cross in vision mode: actions over the
# bridge aborted the navigation container ("Failed to accept new goal", 2026-09-10 16:06).
BRIDGE_MODES = ("split", "vision")
VISION_BOARD_PUBLISHES = (
    *BOARD_PUBLISHES,
    "plan",
    "local_plan",
    "global_costmap/costmap",
    "local_costmap/published_footprint",
    "goal_pose",
    "amcl_path",
)
VISION_LAPTOP_PUBLISHES = (
    "rtabmap/map",
    "rtabmap/mapGraph",
    "rtabmap/mapPath",
    "rtabmap/info",
    "depth_scan",
    "contact_scan",
)


def bridge_allow(side: str, mode: str = "split") -> dict[str, list[str]]:
    """The bridge's ``allow`` block for ``side`` ("board" or "laptop") in ``mode``: its own
    publishers, servers and action servers; the other side's as its subscribers and clients."""
    if mode == "split":
        board: _Names = (BOARD_PUBLISHES, BOARD_SERVES, BOARD_ACTIONS)
        laptop: _Names = (LAPTOP_PUBLISHES, LAPTOP_SERVES, LAPTOP_ACTIONS)
    elif mode == "vision":
        board, laptop = (VISION_BOARD_PUBLISHES, (), ()), (VISION_LAPTOP_PUBLISHES, (), ())
    else:
        raise ValueError(f"a bridge mode is one of {BRIDGE_MODES}, not {mode!r}")
    if side == "board":
        mine, theirs = board, laptop
    elif side == "laptop":
        mine, theirs = laptop, board
    else:
        raise ValueError(f"a bridge sits on the board or the laptop, not {side!r}")
    return {
        "publishers": [_names_regex(mine[0])],
        "subscribers": [_names_regex(theirs[0])],
        "service_servers": [_names_regex(mine[1])],
        "service_clients": [_names_regex(theirs[1])],
        "action_servers": [_names_regex(mine[2])],
        "action_clients": [_names_regex(theirs[2])],
    }


def bridge_config(side: str, mode: str = "split") -> dict[str, object]:
    """zenoh-bridge-ros2dds's configuration file for ``side`` in ``mode`` (its own strict
    schema: nothing but its keys). Written to ros/<bridge_config_name(side, mode)>; a test
    keeps the files equal to this."""
    return {
        "plugins": {
            "ros2dds": {"allow": bridge_allow(side, mode), "queries_timeout": {"default": 5.0}}
        }
    }


def bridge_config_name(side: str, mode: str = "split") -> str:
    """The file under ros/ that carries :func:`bridge_config` for ``side`` and ``mode``: the
    board's unit reads it as ``$PEPIN_BRIDGE_CONFIG`` (ros/thin.sh sets it with the mode),
    ros/laptop.sh picks its own by the side the board reports."""
    if mode not in BRIDGE_MODES:
        raise ValueError(f"a bridge mode is one of {BRIDGE_MODES}, not {mode!r}")
    return f"zenoh-bridge-{side}.json" if mode == "split" else f"zenoh-bridge-{side}-vision.json"


def bridge_admin_for(side: str) -> str:
    """The REST admin of the bridge a launch on ``side`` shares a host with: the laptop's
    bridge container on the Docker network, or the board's on the host network (the board's
    container runs with ``--network host``)."""
    return "http://pepin-zenoh:8000" if side == "laptop" else "http://127.0.0.1:8000"


def bridge_zid(admin_json: str) -> str | None:
    """The zenoh id of a bridge from its REST admin reply for ``@/local/router``, or ``None``.

    The id changes whenever the bridge process restarts — a board reboot, a stack restart — and
    a laptop that keeps its old subscriptions past that moment is deaf (run 0148: no plan in
    139 s because the laptop's costmap never saw the new bridge's transforms).
    """
    import json

    try:
        entries = json.loads(admin_json)
        key = str(entries[0]["key"])
    except (ValueError, TypeError, KeyError, IndexError):
        return None
    parts = key.split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "@" and parts[2] == "router" else None


class BridgeIdentity:
    """Remembers which bridge the laptop last talked to; says when this half must restart.

    Three verdicts, each once: a different id than the last seen (the board's bridge was
    restarted); no answer for ``silence_s`` after contact (a wedged bridge stays "Up" and
    answers nothing, 2026-09-09 — the half restarts so it comes back against whatever the
    unit's liveness check brings up); and the first id seen by a watch that started with no
    bridge to talk to (its nodes subscribed against nothing: a bridge restarted AFTER the
    containers breaks exactly those subscriptions, run 0148). A brief silence is not a change.
    """

    def __init__(self, silence_s: float = 60.0) -> None:
        self._silence_s = silence_s
        self._zid: str | None = None
        self._last_answer: float | None = None
        self._started_blind = False
        self._polls = 0

    def observe(self, zid: str | None, now: float = 0.0) -> bool:
        """One poll of the admin (``zid`` or ``None`` when it did not answer) at ``now``
        seconds: True when this half should restart."""
        self._polls += 1
        if zid is None:
            if self._polls == 1:
                self._started_blind = True
            if self._last_answer is None or now - self._last_answer < self._silence_s:
                return False
            self._last_answer = None  # once per silence: the next answer is a first contact
            return True
        first, self._zid = self._zid, zid
        heard_before = self._last_answer is not None
        self._last_answer = now
        if first is None:
            blind, self._started_blind = self._started_blind, False
            return blind and not heard_before
        return first != zid


def routes_settled(count: int, expected: int | None, stable_s: float, settle_s: float) -> bool:
    """Whether a new bridge has finished declaring its routes: the count has not moved for
    ``settle_s`` seconds and is at least half of ``expected`` — the previous bridge's count at
    first contact, the only measure of "all" there is (a fixed 20 made vision mode, which
    routes fewer topics, wait out the whole patience). Unknown ``expected``: any route at all."""
    floor = 1 if expected is None else max(1, expected // 2)
    return stable_s >= settle_s and count >= floor


# Fully qualified names of the ROS nodes the laptop's SLAM launch creates
# (ros/pepin_bringup/launch/vslam.launch.py): what its restart must see gone from the bridge.
LAPTOP_SLAM_NODES = (
    "/camera_stream",
    "/depth_stream",
    "/contact_scan",
    "/depth_fusion",
    "/rtabmap/rtabmap",
    "/rtabmap_frame",
    "/foxglove_bridge",
)


def nav_container_nodes(side: str) -> tuple[str, ...]:
    """Fully qualified names of the ROS nodes the Nav2 container on ``side`` creates: the
    container itself, its lifecycle nodes, the costmap each planner/controller creates inside,
    the map server with its manager where the map lives, and the navigation manager. What a
    respawn of the container must see gone from the bridge first (2026-09-11 03:07: respawned
    within its predecessor's lease, the board's bridge dropped /map and /navigate_to_pose)."""
    names = [f"/nav2_container_{side}" if side != "all" else "/nav2_container"]
    names += [f"/{node}" for node in nav_nodes(side)]
    if "controller_server" in nav_nodes(side):
        names.append("/local_costmap/local_costmap")
    if "planner_server" in nav_nodes(side):
        names.append("/global_costmap/global_costmap")
    if runs_here(side, "map_server"):
        names += ["/map_server", "/lifecycle_manager_localization"]
    names.append(f"/lifecycle_manager_navigation_{side}")
    return tuple(names)


def laptop_launch_nodes(launch: str) -> tuple[str, ...]:
    """Fully qualified names of the ROS nodes the laptop's ``launch`` ("nav" or "slam") creates.

    The navigation half is the planner side's container (:func:`nav_container_nodes`) and the
    goal server.
    """
    if launch == "slam":
        return LAPTOP_SLAM_NODES
    if launch == "nav":
        return (*nav_container_nodes("laptop"), "/goal_server")
    raise ValueError(f"launch must be 'nav' or 'slam', not {launch!r}")


def lingering_nodes(admin_json: str, names: tuple[str, ...]) -> set[str]:
    """Which of ``names`` the bridge still lists, from its REST admin reply for
    ``@/local/ros2/node/**`` (keys read ``@/<zid>/ros2/node/<participant>/<node name>``).

    The bridge keeps a route's local nodes by name, not by process. A container replaced within
    the DDS lease of its killed predecessor (ten seconds) shows the bridge two /rtabmap/rtabmap;
    when the ghost expires, the bridge drops the name from every route and the live node is left
    with a deaf /scan route and no map route out (RTAB-Map fed only while some other subscriber
    happened to exist, 2026-09-10). A launch waits until its own names are gone before it starts.
    """
    import json

    try:
        rows = json.loads(admin_json)
    except ValueError:
        return set()
    seen: set[str] = set()
    for row in rows if isinstance(rows, list) else []:
        try:
            tail = str(row["key"]).split("/ros2/node/", 1)[1]
        except (TypeError, KeyError, IndexError):
            continue
        _participant, _, node = tail.partition("/")
        seen.add(f"/{node}")
    return set(names) & seen


def node_host(node: str) -> tuple[str, str]:
    """Where a node's process lives in the split stack, as ``(side, container)``: the laptop's
    SLAM container for the camera nodes, its navigation container for the planner and the goal
    server, the board's ``pepin-ros`` for everything else (the sensors, the tracker, the
    reflexes) — what ros/flags.sh execs into to reach the node's parameters."""
    name = f"/{node.lstrip('/')}"
    if name in laptop_launch_nodes("slam"):
        return "laptop", "pepin-vslam"
    if name in laptop_launch_nodes("nav"):
        return "laptop", "pepin-laptop"
    return "board", "pepin-ros"
