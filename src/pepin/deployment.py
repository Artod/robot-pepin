"""Which part of the stack runs where: the board keeps the reflexes, the laptop takes the rest.

The board is four Cortex-A53 cores. Whatever closes a control loop or owns a frame stays on
it: the sensors, the EKF, the tracker (map->odom), the controller with its local costmap, the
behaviours and the tree that orders them. Whatever answers once a second and tolerates a
wireless hop moves to the laptop: the planner with the global costmap, the goal server with its
recorder. The split is data, so a test can hold it and the launch file merely reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """One anchored regex over ROS names (with their leading slash) for the bridge's allow-list."""
    return "^/(" + "|".join(sorted(names)) + ")$"


def bridge_allow(side: str) -> dict[str, list[str]]:
    """The bridge's ``allow`` block for ``side`` ("board" or "laptop"): its own publishers,
    servers and action servers; the other side's as its subscribers and clients."""
    board: _Names = (BOARD_PUBLISHES, BOARD_SERVES, BOARD_ACTIONS)
    laptop: _Names = (LAPTOP_PUBLISHES, LAPTOP_SERVES, LAPTOP_ACTIONS)
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


def bridge_config(side: str) -> dict[str, object]:
    """zenoh-bridge-ros2dds's configuration file for ``side`` (its own strict schema: nothing
    but its keys). Written to ros/zenoh-bridge-<side>.json; a test keeps the files equal to this."""
    return {
        "plugins": {"ros2dds": {"allow": bridge_allow(side), "queries_timeout": {"default": 5.0}}}
    }


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
    """Remembers which bridge the laptop last talked to; says when it is a different one."""

    def __init__(self) -> None:
        self._zid: str | None = None

    def observe(self, zid: str | None) -> bool:
        """True once: the first id seen after another one — the board's bridge was restarted.
        An unreachable bridge (``None``) is not a change; the same id again is not either."""
        if zid is None:
            return False
        first, self._zid = self._zid, zid
        return first is not None and first != zid


# Fully qualified names of the ROS nodes the laptop's SLAM launch creates
# (ros/pepin_bringup/launch/vslam.launch.py): what its restart must see gone from the bridge.
LAPTOP_SLAM_NODES = (
    "/camera_stream",
    "/depth_stream",
    "/rtabmap/rtabmap",
    "/rtabmap_frame",
    "/foxglove_bridge",
)


def laptop_launch_nodes(launch: str) -> tuple[str, ...]:
    """Fully qualified names of the ROS nodes the laptop's ``launch`` ("nav" or "slam") creates.

    The navigation half is the planner side's lifecycle nodes in their own container, its
    manager, the global costmap the planner creates, and the goal server.
    """
    if launch == "slam":
        return LAPTOP_SLAM_NODES
    if launch == "nav":
        return (
            *(f"/{node}" for node in LAPTOP_NAV_NODES),
            "/global_costmap/global_costmap",
            "/lifecycle_manager_navigation_laptop",
            "/nav2_container_laptop",
            "/goal_server",
        )
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
