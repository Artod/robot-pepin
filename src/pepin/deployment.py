"""Which part of the stack runs where: the board keeps the reflexes, the laptop takes the rest.

The board is four Cortex-A53 cores. Whatever closes a control loop or owns a frame stays on
it: the sensors, the EKF, the tracker (map->odom), the controller with its local costmap, the
behaviours and the tree that orders them. Whatever answers once a second and tolerates a
wireless hop moves to the laptop: the planner with the global costmap, the goal server with its
recorder. The split is data, so a test can hold it and the launch file merely reads it.
"""

from __future__ import annotations

from collections.abc import Sequence
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

# How long a container of this robot is given to stop before it is killed, everywhere: ros/lib.sh
# (pepin_stop_container, which every ros/*.sh goes through), board/pepin-ros.service's ExecStop,
# the launches' sigterm_timeout (ros/pepin_bringup/launch/*.launch.py) and the bridge watch's
# own repair (pepin_bringup.bridge_watch). The slowest thing inside that window is RTAB-Map
# closing its database: 20-28 GB of visual memory, and the 5 s a launch escalates in by default
# is not enough for it. Eight SIGKILLs of a crash loop on 2026-09-13 left ros/maps/rtabmap.db
# "database disk image is malformed" — the window is what keeps a stop from being the ninth.
CONTAINER_STOP_TIMEOUT_S = 30

# The camera's own odometry (rtabmap_odom's rgbd_odometry, gated by pepin_bringup.visual_odometry
# on the laptop) on its way to the board's EKF, which fuses it as odom1 (ros/params/ekf.yaml).
# The laptop's, by CLAUDE.md rule 20: it consumes the camera, it costs a quarter of a core at
# 9 Hz (measured 2026-09-14, scratch/vo_probe.py), and a cart that loses the laptop loses one of
# three odometry inputs and drives on the wheels and the gyro exactly as it does today.
VO_TOPIC = "vo"

# The laptop's one word to the board's own systemd: "restart your zenoh bridge, mine is newer".
# A route's DDS endpoint is built when the route is created and only while the far bridge is
# already announcing, so of two bridges the one that started LAST gets working routes and the
# one that started first keeps routes with an empty endpoint — which is why ros/laptop.sh
# restarts the board's bridge (settle_bridge, over ssh) right after it starts the laptop's. The
# bridge watch has no ssh and must not have one: it publishes this topic instead, the board's
# run recorder (pepin_bringup.bridge_kick) touches :data:`BRIDGE_KICK_FLAG`, and a systemd path
# unit on the board (board/pepin-bridge-kick.path) does the restart with the board's own
# privileges. One String, at most once per repair.
BRIDGE_KICK_TOPIC = "bridge/kick"
# The flag file, on the board, at the same path inside the container and on the host: the ROS
# container bind-mounts /run/pepin (ros/run.sh), which is tmpfs — a kick never survives a reboot
# and nothing of ours lives on the SD card for it. Not under /root/pepin-ros: ros/sync.sh rsyncs
# that whole tree with --delete, which removes anything the board made there.
BRIDGE_KICK_FLAG = "/run/pepin/bridge_kick"

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


def runs_here(side: str, node: str, slam: bool = False) -> bool:
    """Whether a named piece runs on ``side``: Nav2 nodes, the map, the tracker, the goal server.

    ``slam`` is the online-SLAM mode, where the map does not exist before the drive: RTAB-Map on
    the laptop builds it and owns the correction, so the board serves no saved map and runs no
    scan-matching tracker, and one node of its own (``slam_frame``) puts that correction on the
    board's ``map -> odom``. Exactly one owner of that edge in either mode.
    """
    if node in MAP_NODES or node == "relocalizer":
        return side in ("all", "board") and not slam
    if node == "slam_frame":
        return side in ("all", "board") and slam
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
    # How sure the tracker's fusion is, whichever source spoke into it: the one number a goal
    # gate and a blind-drive watch read (pepin.watch). It crosses for the operator's view — the
    # judges that matter run on the board, beside the tracker.
    "localization/sigma",
    "localization/sources",  # every scan source's word on each update, JSON (the tracker)
    # ...and how sure of itself the tracker is after fusing them (sigma_xy m, sigma_yaw deg,
    # JSON). The laptop's fusion reads it before it paints: a pose whose sigma has grown is not
    # a pose to write a wall with (pepin.watch.PaintTrust, depth_fusion's paint_sigma_m).
    "localization/sigma",
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
    VO_TOPIC,  # the camera's own odometry, gated here, fused by the board's EKF as odom1
    BRIDGE_KICK_TOPIC,  # "restart your bridge after mine": the watch's last repair, not a drive
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
# "slam" (ros/thin.sh slam) is vision mode with the map turned around: RTAB-Map on the laptop
# IS the map, built while the cart drives, so the board serves no saved map and runs no tracker.
# The grid crosses laptop -> board as /map (the global costmap's static layer reads it,
# transient local) and the graph's correction as /rtabmap/mapGraph, which pepin_bringup.slam_frame
# turns into map -> odom ON THE BOARD. /tf itself still crosses one way only (board -> laptop):
# a topic allowed as a publisher on both sides loops until nothing crosses at all, so the
# correction travels as a message and becomes a transform where the reflexes look it up.
BRIDGE_MODES = ("split", "vision", "slam")
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
    # The laptop's whole-map watchdog (pepin_bringup.laptop_localizer) proposing a place to the
    # board's tracker, once a second, as JSON. Vision mode only: this is where the laptop sees
    # the board's /scan and /map, and in SLAM mode there is no saved map to search.
    "localization/candidate",
    # ...and the camera's poses, measured on the laptop out of /depth_scan and /contact_scan and
    # fused by the board's tracker (pepin.measurements). The scans themselves still cross for the
    # costmap; what used to cross for the POSE was the matching, and that cost the board 147 ms a
    # scan and 50 cm p90 of live error (scratch/drive_bisect.py, run 0238).
    "localization/measurement",
    # ...and RTAB-Map's pose graph's own answer about where the cart is on the SAME map, sent
    # whenever the graph moves (pepin_bringup.rtabmap_frame, flag graph_measurement) and fused
    # by a gate of its own on the board. A topic apart from the camera's because the board's
    # measurement gate fuses everything on one topic into one word under one name.
    "localization/graph_measurement",
    # ...and the lidar layer of that same volume, on a topic of its own (depth_fusion's
    # lidar_map flag): the map the board's tracker matches on when its map_topic flag names it.
    # NOT /map — the board's map_server owns that in this mode, and two publishers of one /map
    # is the failure of 2026-09-10. This one nobody else publishes, so it needs no owner rule.
    "map_lidar",
    VO_TOPIC,
    BRIDGE_KICK_TOPIC,
)
# The saved map's own topics, the ones SLAM mode has no publisher for: /map is the laptop's here,
# and the rest are the tracker's, which does not run because nothing matches a scan against a map
# that does not exist yet. Named, not spelled out below, so a topic added to the vision list
# reaches SLAM mode by itself unless it is one of these.
_NOT_IN_SLAM = (
    "map",
    "tracker_pose",
    "localization_fit",
    "localization/sigma",
    "localization/sources",
)
# The board drives exactly as in vision mode, minus those.
SLAM_BOARD_PUBLISHES = tuple(n for n in VISION_BOARD_PUBLISHES if n not in _NOT_IN_SLAM)
# RTAB-Map's grid is remapped onto /map in this mode (there is no second map to fight), so
# /rtabmap/map is not published at all; the graph and its correction still are.
SLAM_LAPTOP_PUBLISHES = (
    "map",
    "localization/measurement",  # the camera's poses; harmless where no tracker listens
    "map_odom",  # RTAB-Map's correction as a message; the board broadcasts it as map -> odom
    "rtabmap/mapGraph",
    "rtabmap/mapPath",
    "rtabmap/info",
    "depth_scan",
    "contact_scan",
    VO_TOPIC,
    BRIDGE_KICK_TOPIC,
)


# Who publishes /map in each mode — one owner, always. In "split" and "vision" the board serves
# the saved map (:data:`MAP_NODES` run there, and the tracker beside them needs it local); in
# "slam" there is no saved map and the laptop's own grid IS /map. A second publisher would feed
# the costmaps two maps and make the tracker rebuild on whichever arrived last, so anything on
# the laptop that can publish a map (pepin_bringup.depth_fusion with map_source=volume) asks
# here first and stays quiet where the answer is "board".
MAP_OWNER = {"split": "board", "vision": "board", "slam": "laptop"}


def map_owner(mode: str = "split") -> str:
    """The side that publishes /map in ``mode`` ("board" or "laptop")."""
    if mode not in BRIDGE_MODES:
        raise ValueError(f"a bridge mode is one of {BRIDGE_MODES}, not {mode!r}")
    return MAP_OWNER[mode]


def bridge_allow(side: str, mode: str = "split") -> dict[str, list[str]]:
    """The bridge's ``allow`` block for ``side`` ("board" or "laptop") in ``mode``: its own
    publishers, servers and action servers; the other side's as its subscribers and clients."""
    if mode == "split":
        board: _Names = (BOARD_PUBLISHES, BOARD_SERVES, BOARD_ACTIONS)
        laptop: _Names = (LAPTOP_PUBLISHES, LAPTOP_SERVES, LAPTOP_ACTIONS)
    elif mode == "vision":
        board, laptop = (VISION_BOARD_PUBLISHES, (), ()), (VISION_LAPTOP_PUBLISHES, (), ())
    elif mode == "slam":
        board, laptop = (SLAM_BOARD_PUBLISHES, (), ()), (SLAM_LAPTOP_PUBLISHES, (), ())
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


# How much of each topic may cross the wireless hop, in Hz, on the side that PUBLISHES it
# (zenoh-bridge-ros2dds downsamples a pub route, so an entry only does something in the config
# of the bridge whose ROS graph holds the writer — all four of these are the board's). Measured
# at rest on 2026-09-15, 19:27, with the link healthy (scratch/bridge_logs_1927): the laptop
# received tf 51.6 Hz, imu/data_raw 46.6, odometry/filtered 18.6, odom 16.2. In a drive that
# same evening the link starved and the laptop saw tf 27.8, imu 19, scan 4 of 9.6, odom ~8 —
# the radio cannot carry what the board offers, so the board offers less, and what it does
# offer arrives instead of being dropped somewhere in the middle:
PUB_MAX_FREQUENCY_HZ: dict[str, float] = {
    # /tf is the big one: 51.6 Hz of it is the EKF's odom -> base_link at 50. Everything here
    # reads it through tf2, which interpolates between samples — the depth stream's carry to a
    # frame's stamp (pepin_bringup.depth_stream), RTAB-Map's odometry (odom_frame_id: odom) and
    # the costmaps — so 50 ms between transforms is a smaller error than the 5 s of nothing a
    # stalled link produced. /tf_static is NOT capped (it is latched and published once).
    "tf": 20.0,
    # The gyro, read by node_kit.LeanFeed for the camera's lean alone (the EKF that integrates
    # it runs on the board, on the local copy, and never sees this route). A lean estimate
    # rides a few degrees a second: 20 Hz is oversampling it already.
    "imu/data_raw": 20.0,
    # The wheels, read here only by the visual odometry's rest watch (is the cart standing
    # still) — 16.2 Hz measured, so this cap is a ceiling on a burst, not a cut.
    "odom": 20.0,
    # The EKF's pose, read here by the laptop localizer, which decides once a second (18.6 Hz
    # measured: again a ceiling, not a cut).
    "odometry/filtered": 20.0,
}


def pub_max_frequencies(names: Sequence[str]) -> list[str]:
    """The bridge's ``pub_max_frequencies`` entries for the topics of ``names`` that have a cap
    in :data:`PUB_MAX_FREQUENCY_HZ`, in its own ``"<regex>=<float>"`` form.

    The regex is anchored by us because the plugin does not anchor it and matches with
    ``is_match`` (1.7.0 ``config.rs``): a bare ``/tf`` would cap ``/tf_static`` too.
    """
    wanted = {n.lstrip("/") for n in names}
    return [
        f"^/{name}$={PUB_MAX_FREQUENCY_HZ[name]:g}"
        for name in sorted(wanted & set(PUB_MAX_FREQUENCY_HZ))
    ]


def bridge_config(side: str, mode: str = "split") -> dict[str, object]:
    """zenoh-bridge-ros2dds's configuration file for ``side`` in ``mode`` (its own strict
    schema: nothing but its keys). Written to ros/<bridge_config_name(side, mode)>; a test
    keeps the files equal to this.

    Two settings beside the allow-list, both about a link that stalls. ``reliable_routes_blocking``
    is the bridge's default (true), and it is what killed the link on 2026-09-15: a RELIABLE DDS
    writer's publications are pushed to zenoh with ``CongestionControl::Block``, so a 5 s wireless
    stall on a 50 Hz topic filled the transmission queue and the board's bridge ended the
    transport itself — "Unable to push non droppable network message ... Closing transport!" —
    and reconnected with the same zenoh id and routes that were never rebuilt. False drops the
    samples that do not fit instead, which is what every consumer here already tolerates (the
    topics are periodic; a dropped /tf is one the next one replaces 50 ms later). The caps of
    :data:`PUB_MAX_FREQUENCY_HZ` are the other half: they keep the queue from filling at all.
    """
    return {
        "plugins": {
            "ros2dds": {
                "allow": bridge_allow(side, mode),
                "pub_max_frequencies": pub_max_frequencies(
                    allowed_names(bridge_allow(side, mode)["publishers"][0])
                ),
                "queries_timeout": {"default": 5.0},
                "reliable_routes_blocking": False,
            }
        }
    }


def bridge_config_name(side: str, mode: str = "split") -> str:
    """The file under ros/ that carries :func:`bridge_config` for ``side`` and ``mode``: the
    board's unit reads it as ``$PEPIN_BRIDGE_CONFIG`` (ros/thin.sh sets it with the mode),
    ros/laptop.sh picks its own by the side and the SLAM flag the board reports."""
    if mode not in BRIDGE_MODES:
        raise ValueError(f"a bridge mode is one of {BRIDGE_MODES}, not {mode!r}")
    return f"zenoh-bridge-{side}.json" if mode == "split" else f"zenoh-bridge-{side}-{mode}.json"


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


# ---- what crosses, and whether it really does --------------------------------------------------

# The QoS every endpoint of a bridged topic must use, on BOTH sides. Not a style rule: the bridge
# fixes a route's DDS QoS at the moment the route is created and never revises it. A route is
# created by whichever declaration arrives first — the local ROS endpoint, or the remote bridge's
# announcement of its own — and the two carry different QoS ("those are either the QoS announced
# by a remote bridge on a Reader discovery, either the QoS adapted from a local discovered
# Writer", zenoh-plugin-ros2dds 1.7.0 route_publisher.rs), while the route itself is keyed by
# topic name alone, so the loser's QoS is simply never used. Measured on 2026-09-13
# (scratch/bridge_state_182255_*): /imu/data_raw is written RELIABLE, KEEP_LAST 10 by the board's
# C++ bridge and read BEST_EFFORT, KEEP_LAST 5 by all three laptop nodes (node_kit.LeanFeed used
# qos_profile_sensor_data). Whenever the laptop's announcement won the race, the board's side of
# the route became a BEST_EFFORT reader five deep on a loaded Orange Pi and dropped four samples
# in five: 48 Hz on the board, 10-11 Hz on the laptop, a burst of 58 when it flushed. With both
# sides equal there is no race left to lose. The DDS legs are inside one host — the wireless hop
# is zenoh's, not DDS's — so RELIABLE there costs a memcpy, not a retransmission.
BRIDGED_QOS: dict[str, tuple[str, int]] = {
    "/imu/data_raw": ("reliable", 10),  # base_bridge.cpp publishes RELIABLE, KEEP_LAST 10
    # robot_localization subscribes to every odomN with rclcpp's default (RELIABLE) at the
    # depth of its odomN_queue_size, which ros/params/ekf.yaml sets to 10 for /vo: the laptop's
    # publisher matches it exactly, so the route's QoS cannot depend on which side announced it
    # first (the failure that starved /imu/data_raw at 10 Hz on 2026-09-13).
    f"/{VO_TOPIC}": ("reliable", 10),
    # base_bridge.cpp publishes /odom with create_publisher(..., 10): RELIABLE, KEEP_LAST 10.
    # It became a bridged topic the day a laptop node started reading it (the visual odometry's
    # rest watch, which needs the wheels to know the cart stands still), and the laptop has two
    # endpoints on it — that node and bridge_watch's own flow probe — which would otherwise ask
    # for different reliabilities and let the route's QoS be decided by whichever declaration
    # the bridge saw first. A RELIABLE reader does not match a BEST_EFFORT writer at all, and
    # the loser of that race receives nothing, in silence.
    "/odom": ("reliable", 10),
    # The graph's word to the board's tracker: RELIABLE, five deep on both ends (the tracker
    # keeps a few tenths of a second of history rather than only the newest, since every
    # measurement carries the stamp it was made at). A word that arrives late is dropped by the
    # gate's own age rule; one that never arrives because the route lost a QoS race is invisible.
    "/localization/graph_measurement": ("reliable", 5),
    # The laptop's kick to the board's bridge: one String, and the one message of this robot
    # that must not be dropped by a QoS race — it is sent exactly once per repair, while the
    # link is already sick. RELIABLE on both ends, five deep (pepin_bringup.bridge_kick).
    f"/{BRIDGE_KICK_TOPIC}": ("reliable", 5),
}


def bridged_qos(topic: str) -> tuple[str, int] | None:
    """The reliability ("reliable" or "best_effort") and history depth every endpoint of
    ``topic`` must use, on both sides of the bridge, or ``None`` for a topic with no rule."""
    return BRIDGED_QOS.get(topic if topic.startswith("/") else f"/{topic}")


def incoming_topics(side: str, mode: str = "split") -> tuple[str, ...]:
    """The topics that ARRIVE on ``side`` in ``mode`` (the other side's publishers), each with
    its leading slash: what a watch there must see messages on for the link to be working."""
    other = "laptop" if side == "board" else "board"
    return tuple(sorted(allowed_names(bridge_allow(other, mode)["publishers"][0])))


def allowed_names(regex: str) -> tuple[str, ...]:
    """The ROS names an allow-list regex admits, in the shape :func:`_names_regex` writes them
    (``^/(a|b)$``, and ``^^/(a|b)$$`` as the bridge echoes it back through its admin). A regex
    of any other shape, or the empty ``^$``, is no names at all."""
    body = regex.strip()
    while body.startswith("^"):
        body = body[1:]
    while body.endswith("$"):
        body = body[:-1]
    if not body.startswith("/(") or not body.endswith(")"):
        return ()
    return tuple(sorted(f"/{name}" for name in body[2:-1].split("|") if name))


@dataclass(frozen=True)
class BridgeRoute:
    """One route as the bridge's REST admin describes it: which bridge owns it (``zid``), which
    way it carries (``direction`` "pub" — local publications out to zenoh — or "sub" — zenoh
    traffic into the local DDS), the ROS topic and type, the local ROS nodes it serves, the
    routes the other bridges hold for the same topic (``remote_routes``, "<zid>:<key>"), whether
    the bridge has an endpoint for it at all (``active``; a pub route builds its DDS reader only
    once some remote subscriber wants the topic) and that endpoint's GUID — the DDS reader of a
    pub route, the DDS writer of a sub route, empty when the bridge never built one."""

    zid: str
    direction: str
    topic: str
    type_name: str
    local_nodes: tuple[str, ...]
    active: bool
    endpoint: str
    remote_routes: tuple[str, ...] = ()

    @property
    def dead(self) -> bool:
        """Whether this route is wired at both ends and carries nothing by construction: a local
        ROS endpoint to serve, a bridge on the far side that holds the matching route, and no
        DDS endpoint of its own — so not a byte can cross it.

        2026-09-14: after the board's bridge changed identity, the laptop bridge's
        ``topic/pub/vo`` route had ``local_nodes ['/visual_odometry']`` and a remote route, and
        ``dds_reader ""``; ``topic/pub/depth_scan`` beside it had its reader and flowed. The
        bridge builds a pub route's reader when the route is created and never revises it, so a
        publisher that undeclared and declared again after the bridge started leaves the route
        readerless for as long as the bridge lives. Only restarting the bridge made one.
        """
        return bool(self.local_nodes) and bool(self.remote_routes) and not self.endpoint


def bridge_routes(admin_json: str) -> tuple[BridgeRoute, ...]:
    """Every topic route in a REST admin reply for ``@/*/ros2/route/**``.

    The admin space is network-wide once two bridges are linked: BOTH bridges answer this query
    with the same set, their own routes and the other's, told apart by the zid in the key. That
    is why the laptop's admin lists ``topic/sub/depth_scan`` although the laptop's allow-list has
    depth_scan as a publisher only — the sub route is the board's, read through the laptop's
    admin (2026-09-13: the two answers were byte-for-byte the same length).
    """
    import json

    try:
        rows = json.loads(admin_json)
    except ValueError:
        return ()
    routes: list[BridgeRoute] = []
    for row in rows if isinstance(rows, list) else []:
        parts = str(row.get("key", "")).split("/")  # @ zid ros2 route topic pub|sub name...
        if len(parts) < 7 or parts[3] != "route" or parts[4] != "topic":
            continue
        value = row.get("value")
        if not isinstance(value, dict):
            continue
        # The endpoint of the direction: a pub route carries the local publications out through a
        # DDS reader, a sub route brings zenoh traffic in through a DDS writer. Which one is
        # missing is the whole point (:attr:`BridgeRoute.dead`), so they are not collapsed.
        key = "dds_writer" if parts[5] == "sub" else "dds_reader"
        endpoint = str(value.get(key) or "")
        routes.append(
            BridgeRoute(
                zid=parts[1],
                direction=parts[5],
                topic=str(value.get("ros2_name") or "/" + "/".join(parts[6:])),
                type_name=str(value.get("ros2_type") or ""),
                local_nodes=tuple(str(n) for n in value.get("local_nodes") or ()),
                active=bool(value.get("is_active", bool(endpoint))),
                endpoint=endpoint,
                remote_routes=tuple(str(r) for r in value.get("remote_routes") or ()),
            )
        )
    return tuple(routes)


def dead_routes(routes: Sequence[BridgeRoute], zid: str) -> tuple[str, ...]:
    """The topics of the bridge ``zid`` whose route is wired at both ends and has no DDS
    endpoint of its own (:attr:`BridgeRoute.dead`), sorted, each topic once.

    Only that one bridge's own routes are judged: the repair is a restart of that bridge, and a
    dead route on the far side is not something this side can mend.
    """
    return tuple(sorted({route.topic for route in routes if route.zid == zid and route.dead}))


def far_dead_routes(routes: Sequence[BridgeRoute], zid: str) -> tuple[str, ...]:
    """The topics the OTHER bridge publishes to us and has built no DDS reader for: its pub
    routes with local publishers, no ``dds_reader``, and a remote route naming our own ``zid``.

    The mirror of :func:`dead_routes`, and the fault that was invisible until 2026-09-15: the
    watch judged the laptop's own routes only and printed "dead routes 0" while thirteen of the
    board's pub routes had an empty reader, so nothing crossed from the board at all. It is the
    same reading of the same network-wide admin reply — the board's routes are in it, keyed by
    the board's zid — and the cure is the other one: the board's bridge must be restarted, which
    this side cannot do by itself (:data:`BRIDGE_KICK_TOPIC`).

    Only pub routes: a far sub route without its writer is the same class of fault, but it is
    the far side's word about topics WE publish, and the watch already sees those starve from
    the other end. The routes counted here are the ones measured empty on 2026-09-15.
    """
    mine = f"{zid}:"
    return tuple(
        sorted(
            {
                route.topic
                for route in routes
                if route.zid != zid
                and route.direction == "pub"
                and route.dead
                and any(remote.startswith(mine) for remote in route.remote_routes)
            }
        )
    )


# Topics that are latched or event-driven on purpose: a map published once with transient
# durability, the static transforms, a path only while a drive runs, a run status on change.
# Their silence is not a dead route. A watch that judged them restarted a healthy bridge twenty
# seconds after every start (2026-09-13 19:12: "/map /plan /tf_static carried nothing" — and
# the restart it made rebuilt the routes from the far side's announcements, leaving /scan at
# 0.8 Hz and /imu/data_raw at 3.7 Hz on the laptop until the next ordered restart).
ON_DEMAND_TOPICS: frozenset[str] = frozenset(
    {
        "/map",
        "/map_camera",
        "/map_lidar",
        "/tf_static",
        "/plan",
        "/local_plan",
        "/amcl_path",
        "/goal_pose",
        "/pepin/run_status",
        "/rtabmap/info",
        "/rtabmap/map",
        "/rtabmap/mapGraph",
        "/rtabmap/mapPath",
        f"/{BRIDGE_KICK_TOPIC}",  # one message per repair, and none at all on a healthy link
    }
)


@dataclass(frozen=True)
class TopicFlow:
    """What both bridges say about one topic: the ROS type to subscribe with, whether some node
    on the other side really publishes it, and which local nodes are waiting for it here."""

    topic: str
    type_name: str
    published_there: bool
    subscribers_here: tuple[str, ...]

    @property
    def should_flow(self) -> bool:
        """Whether messages must be arriving: somebody publishes it there, somebody wants it
        here. Neither half alone is a fault — an unwanted topic is never routed at all."""
        return self.published_there and bool(self.subscribers_here)

    @property
    def judged(self) -> bool:
        """Whether silence on this topic means a dead route: it should flow AND it is periodic
        by nature — the latched and event-driven topics of :data:`ON_DEMAND_TOPICS` are never
        judged, however long they stay quiet."""
        return self.should_flow and self.topic not in ON_DEMAND_TOPICS


def topic_flows(
    routes: Sequence[BridgeRoute], local_zid: str, allowed: Sequence[str], watcher: str = ""
) -> tuple[TopicFlow, ...]:
    """The two bridges' word on every topic of ``allowed`` that should arrive at the bridge
    ``local_zid``: its type, whether the far side publishes it, which nodes here subscribe
    (``watcher``, the watch's own node name, never counts as one of them)."""
    wanted = {t if t.startswith("/") else f"/{t}" for t in allowed}
    flows: dict[str, TopicFlow] = {}
    for route in routes:
        if route.topic not in wanted:
            continue
        flow = flows.get(route.topic) or TopicFlow(route.topic, "", False, ())
        if route.direction == "sub" and route.zid == local_zid:
            here = tuple(n for n in route.local_nodes if n.lstrip("/") != watcher.lstrip("/"))
            flow = TopicFlow(
                route.topic, route.type_name or flow.type_name, flow.published_there, here
            )
        elif route.direction == "pub" and route.zid != local_zid:
            flow = TopicFlow(
                route.topic,
                route.type_name or flow.type_name,
                bool(route.local_nodes),
                flow.subscribers_here,
            )
        flows[route.topic] = flow
    return tuple(flows[t] for t in sorted(flows))


class FlowWatch:
    """Which bridged topics have a publisher on the far side and carry nothing on this one.

    A route count cannot see this: the route exists, the far side's publisher exists, and the
    QoS the route was built with never matched it (see :data:`BRIDGED_QOS`), so the admin looks
    perfect while the topic is dead — /depth_scan on 2026-09-12, /imu/data_raw on 2026-09-13,
    both cured by restarting a bridge. So the watch counts messages instead: a topic that
    should flow (:attr:`TopicFlow.should_flow`) and whose count has not moved for ``silence_s``
    is starved. After a repair every clock is reset and nothing is starved for ``cooldown_s``:
    fresh routes take seconds to carry their first message, and a repair loop is worse than a
    dead topic.
    """

    def __init__(self, silence_s: float = 20.0, cooldown_s: float = 90.0) -> None:
        self.silence_s = silence_s  # live: the node's flow_silence_s flag writes it
        self.cooldown_s = cooldown_s
        self._counts: dict[str, int] = {}
        self._moved_at: dict[str, float] = {}
        self._seen_at: dict[str, float] = {}
        self._quiet_until = 0.0

    def observe(self, topic: str, delivered: int, expected: bool, now: float) -> None:
        """One reading of ``topic``'s message counter. ``expected`` is whether messages are due
        at all (:attr:`TopicFlow.should_flow`, and the watch really is counting them): a topic
        nobody publishes or nobody here reads is never starved, and its clock stays fresh so it
        is not declared dead the moment it becomes due again."""
        if not expected or delivered != self._counts.get(topic, -1):
            self._moved_at[topic] = now
        self._counts[topic] = delivered
        self._seen_at[topic] = now

    def starved(self, now: float) -> tuple[str, ...]:
        """The topics silent for longer than the patience, outside the cooldown after a repair.

        Only the topics of the round just observed (the same ``now``) can be named: a round that
        could not reach the admins observes nothing and must accuse no one — an unreachable
        bridge is the identity watch's business, not this one's.
        """
        if now < self._quiet_until:
            return ()
        return tuple(
            topic
            for topic, moved in sorted(self._moved_at.items())
            if self._seen_at.get(topic) == now and now - moved >= self.silence_s
        )

    def repaired(self, now: float) -> None:
        """A repair was just attempted: every clock restarts and the cooldown begins."""
        self._quiet_until = now + self.cooldown_s
        self._moved_at = dict.fromkeys(self._moved_at, now)

    def settled(self, now: float) -> bool:
        """Whether the link has been healthy since well past the last repair — when the next
        failure deserves the gentle repair again rather than the escalation."""
        return now >= self._quiet_until + self.cooldown_s


# Fully qualified names of the ROS nodes the laptop's SLAM launch creates
# (ros/pepin_bringup/launch/vslam.launch.py): what its restart must see gone from the bridge.
LAPTOP_SLAM_NODES = (
    "/camera_stream",
    "/depth_stream",
    "/contact_scan",
    "/depth_fusion",
    "/laptop_localizer",
    "/rtabmap/rtabmap",
    "/rtabmap_frame",
    "/foxglove_bridge",
    "/rgbd_odometry",  # the camera's odometry (vo:=true, the default)
    "/visual_odometry",  # and the node that gates it for the board's EKF
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
