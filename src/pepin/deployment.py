"""Which part of the stack runs where: the board keeps the reflexes, the laptop takes the rest.

The board is four Cortex-A53 cores. Whatever closes a control loop or owns a frame stays on
it: the sensors, the EKF, the controller with its local costmap, the behaviours and the tree
that orders them. Whatever answers once a second and tolerates a
wireless hop moves to the laptop: the planner with the global costmap, the goal server with its
recorder. The split is data, so a test can hold it and the launch file merely reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SIDES = ("all", "board", "laptop")

# WHO OWNS map -> odom: RTAB-Map on the laptop publishes it (vslam.launch.py's publish_tf) and
# every consumer composes it with the board's own odom -> base_link; the board keeps odometry
# only (one localiser, 2026-09-22). The board's scan-matching tracker that owned the edge before
# is on the tag alt/tracker-2026-09-22.

# Nav2 lifecycle nodes by side. "all" is the union: one machine, as before the split.
# In bring-up order: the tree last, because loading it needs the planner side's costmap service.
BOARD_NAV_NODES = ("controller_server", "behavior_server", "velocity_smoother", "bt_navigator")
LAPTOP_NAV_NODES = ("planner_server",)
# The pgm server, on the board and off by default: under World R the one map is RTAB-Map's live
# grid and the board's own cache of it (pepin.mapcache), so a file is only ever the seed of a room
# nobody has mapped yet (ros/nav.launch.py's map_server argument).
MAP_NODES = ("map_server",)

HEARTBEAT_TOPIC = "laptop/heartbeat"
HEARTBEAT_HZ = 2.0

# How long a container of this robot is given to stop before it is killed, everywhere: ros/lib.sh
# (pepin_stop_container, which every ros/*.sh goes through), board/pepin-ros.service's ExecStop,
# and the launches' sigterm_timeout (ros/pepin_bringup/launch/*.launch.py). The slowest thing
# inside that window is RTAB-Map closing its database: 20-28 GB of visual memory, and the 5 s a
# launch escalates in by default is not enough for it. Eight SIGKILLs of a crash loop on
# 2026-09-13 left ros/maps/rtabmap.db "database disk image is malformed" — the window is what
# keeps a stop from being the ninth.
CONTAINER_STOP_TIMEOUT_S = 30

# The camera's own odometry (rtabmap_odom's rgbd_odometry, gated by pepin_bringup.visual_odometry
# on the laptop) on its way to the board's EKF, which fuses it as odom1 (ros/params/ekf.yaml).
# The laptop's, by CLAUDE.md rule 20: it consumes the camera, it costs a quarter of a core at
# 9 Hz (measured 2026-09-14, scratch/vo_probe.py), and a cart that loses the laptop loses one of
# three odometry inputs and drives on the wheels and the gyro exactly as it does today.
VO_TOPIC = "vo"

# LASER ODOMETRY (rf2o, on the board): the cart's planar motion from consecutive scans, scan to
# scan and against no map at all. It replaces what the lidar tracker used to give the EKF by
# implication — a second opinion on how far and how fast the cart really went, which a wheel
# spinning on carpet cannot fool — and it is on the board because it must survive a WiFi loss
# (CLAUDE.md rule 20: real-time, wifi-loss, and it consumes no camera).
LASER_ODOM_TOPIC = "odom_laser"
# The node's loop rate. It consumes the newest scan each turn, so above the lidar's own rate it
# only finds nothing to do, and below it it drops scans: the LD19 is configured for 10 Hz
# (robot.launch.py, lidar.bins 455) and measures 9.6-10 at the board.
LASER_ODOM_HZ = 10.0
# What the node CLAIMS about its own twist, as the diagonal of twist.covariance. It is stamped
# here and not computed by rf2o, which publishes an empty matrix — and an empty matrix is not
# "unknown" to robot_localization, it is a variance of 1e-9 (ekf.cpp: a measurement covariance
# under 1e-9 is raised to 1e-9) and therefore a source that outvotes the wheels AND the gyro on
# every sample. The same lesson as the camera's, and the same answer: a constant, deliberately
# weak, stamped where the message is born (ros/params/ekf.yaml's /vo block, visual_odometry.py).
#
# vx = 0.0009 (m/s)^2, sigma 3 cm/s: the wheels' own 0.001 at half their rate, so a scan match
# carries about half the wheels' information per second — a real second opinion on distance,
# which is the input the 2026-09-16 carpet slip had none of, and not a replacement for them.
# vyaw = 0.0025 (rad/s)^2, sigma 0.05 rad/s: about 3 % of the gyro's information per second
# (0.0004 at 47 Hz). The gyro still owns heading; this is the vote that survives a dead MPU6050.
# vy is published (the patched node measures it) and NOT fused: the wheels' vy = 0 is a
# kinematic truth and a scan matcher's lateral estimate is the noisiest of the three.
# NONE OF THE THREE IS MEASURED ON THIS CART YET. They are sized to be quiet; the drive that
# measures them is in ros/README.md, and until it exists nobody may tighten them.
LASER_ODOM_TWIST_VARIANCE: dict[str, float] = {"vx": 0.0009, "vy": 0.01, "vyaw": 0.0025}

# The base's speed caps (config/base.json, the base server's own clamp). The C++ bridge on the
# board clamps /cmd_vel too, at 0.25 m/s by default: for half a day every tape sat at 0.20 and
# the bridge would have cut anything faster — one cap, the base's, passed to it at launch.
BASE_MAX_LINEAR_M_S = 0.45  # raised 0.30 -> 0.45 on 2026-09-30, with config/base.json
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
    """Whether a named piece runs on ``side``: Nav2 nodes, the map, the goal server, the watches."""
    if node in MAP_NODES:
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


# Both navigators' actions on the board: what the link watch reads the goal status of.
BOARD_ACTIONS = ("navigate_to_pose", "navigate_through_poses")


# The QoS every endpoint of these cross-machine topics uses, on BOTH sides. A reader and a writer
# that disagree on reliability do not match at all, and the loser receives nothing, in silence.
# Pinned under CycloneDDS, where the bridge fixed a route's QoS from whichever side declared first
# and starved /imu/data_raw to 10 Hz on 2026-09-13 (tag alt/cyclone-bridges-2026-09-20 has the
# bridge); kept under rmw_zenoh so both ends stay one declaration apart from a mismatch.
BRIDGED_QOS: dict[str, tuple[str, int]] = {
    "/imu/data_raw": ("reliable", 10),  # base_bridge.cpp publishes RELIABLE, KEEP_LAST 10
    # robot_localization subscribes to every odomN with rclcpp's default (RELIABLE) at the
    # depth of its odomN_queue_size, which ros/params/ekf.yaml sets to 10 for /vo.
    f"/{VO_TOPIC}": ("reliable", 10),
    # base_bridge.cpp publishes /odom with create_publisher(..., 10): RELIABLE, KEEP_LAST 10.
    "/odom": ("reliable", 10),
}


def bridged_qos(topic: str) -> tuple[str, int] | None:
    """The reliability ("reliable" or "best_effort") and history depth every endpoint of
    ``topic`` must use on both machines, or ``None`` for a topic with no rule."""
    return BRIDGED_QOS.get(topic if topic.startswith("/") else f"/{topic}")


# Fully qualified names of the ROS nodes the laptop's SLAM launch creates
# (ros/pepin_bringup/launch/vslam.launch.py): where ros/flags.sh finds them (:func:`node_host`).
LAPTOP_SLAM_NODES = (
    "/camera_stream",
    "/depth_stream",
    "/contact_scan",
    "/depth_fusion",
    "/sensor_pack",  # the one input RTAB-Map reads: a snapshot of whatever sensor is alive
    "/rtabmap/rtabmap",
    "/rtabmap_frame",
    "/places",  # the room's vocabulary, resolved against the graph
    "/marks_audit",  # who painted the costmap's lethal cells, live (marks_audit:=false: absent)
    "/foxglove_bridge",
    "/rgbd_odometry",  # the camera's odometry (vo:=true, the default)
    "/visual_odometry",  # and the node that gates it for the board's EKF
)


def nav_container_nodes(side: str) -> tuple[str, ...]:
    """Fully qualified names of the ROS nodes the Nav2 container on ``side`` creates: the
    container itself, its lifecycle nodes, the costmap each planner/controller creates inside,
    the map server with its manager where the map lives, and the navigation manager."""
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


def node_host(node: str, split: bool = False) -> tuple[str, str]:
    """Where a node's process lives, as ``(side, container)``: what ros/flags.sh execs into to
    reach the node's parameters.

    The camera nodes are always in the laptop's SLAM container (``pepin-vslam``). The planner and
    the goal server follow the board's ``PEPIN_SIDE``: ``split`` (``PEPIN_SIDE=board``) puts them
    in the laptop's navigation container (``pepin-laptop``), which ros/laptop.sh starts in that
    mode ONLY; a whole board (no ``PEPIN_SIDE`` line, ros/thin.sh vision) runs them itself, and
    there is no ``pepin-laptop`` to exec into (2026-09-23: ``ros/flags.sh set goal_server ...``
    failed on "No such container"). Everything else — the sensors and the reflexes — is the
    board's ``pepin-ros``."""
    name = f"/{node.lstrip('/')}"
    if name in laptop_launch_nodes("slam"):
        return "laptop", "pepin-vslam"
    if split and name in laptop_launch_nodes("nav"):
        return "laptop", "pepin-laptop"
    return "board", "pepin-ros"
