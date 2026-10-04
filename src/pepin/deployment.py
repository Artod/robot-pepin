"""Which part of the stack runs where: the board is a sensor box, the Mac runs the rest.

The board is four Cortex-A53 cores and carries what is wired to its pins and what must survive a
WiFi loss: the sensors, the base bridge and the EKF. Nav2 (planner, controller, costmaps,
behaviour tree), the goal server and the run recorder run on the Mac in one container
(``ros/laptop.sh nav``), the camera's mapping beside them (``ros/laptop.sh vslam``). Where a node
lives is data here, so a test can hold it and the scripts merely read it.
"""

from __future__ import annotations

from pathlib import Path

# WHO OWNS map -> odom: RTAB-Map on the laptop publishes it (vslam.launch.py's publish_tf) and
# every consumer composes it with the board's own odom -> base_link; the board keeps odometry
# only (one localiser, 2026-09-22). The board's scan-matching tracker that owned the edge before
# is on the tag alt/tracker-2026-09-22.

# The Nav2 lifecycle nodes, all in the Mac's one navigation container, in bring-up order.
NAV_NODES = (
    "controller_server",
    "behavior_server",
    "velocity_smoother",
    "bt_navigator",
    "planner_server",
)
# The container ros/laptop.sh nav runs them in, with the goal server and the run recorder.
NAV_CONTAINER = "pepin-macnav"

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

# The MPU6050's READ rate in the C++ bridge (its imu_rate_hz: the read loop and the bias tracker's
# block), fused by the board's EKF as imu0. 100 Hz since 2026-10-01, 50 before. The chip's low-pass
# stays at the ~44 Hz it was (DLPF_CFG 3, mpu6050.hpp), the filter every gyro number of the bridge
# was measured through (the zero-velocity update's quiet threshold is 7.9 sigma of THAT noise), and
# at 100 Hz it sits under the Nyquist limit where at 50 it did not: no new noise, no aliasing,
# twice the samples. The chip's own OUTPUT rate (SMPLRT_DIV) is config/imu.json's
# timing.output_rate_hz, 1 kHz since 2026-10-02 (this rate before): the sample read is <= 1 ms
# old. A read is one 14-byte burst, ~0.4 ms of the 400 kHz bus the three ToFs share
# (board/i2c3-400k.dts since 2026-10-02; ~1.6 ms at the 100 kHz before), 4 % of it.
IMU_RATE_HZ = 100.0


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
    # The head IMU (base_bridge.cpp from head_server's 3340 stream, 200 Hz), read by the
    # laptop's VIO (pepin-vio): RELIABLE, KEEP_LAST 10 like the base IMU.
    "/head/imu": ("reliable", 10),
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
    "/marks_audit",  # who painted the costmap's lethal cells, live
    "/foxglove_bridge",
    "/rgbd_odometry",  # the camera's odometry (vo:=true, the default)
    "/stereo_odometry",  # the same role from the two eyes (vo_input:=stereo; one of the two runs)
    "/visual_odometry",  # and the node that gates it for the board's EKF
)


# The visual-inertial odometry's own container (ros/laptop.sh vio, vio.launch.py): OpenVINS's
# subscriber node, kept apart from pepin-vslam so it can be kicked and its memory counted alone.
VIO_CONTAINER = "pepin-vio"
# vio.launch.py's nodes: OpenVINS in its namespace ov_msckf, and the gated feed in front of it
LAPTOP_VIO_NODES = ("/ov_msckf/run_subscribe_msckf", "/vio_feed", "/vio_keeper")


# Fully qualified names of the ROS nodes of the Mac's navigation container
# (ros/pepin_bringup/launch/nav.launch.py): the composed container, its lifecycle nodes and the
# costmaps they create, the navigation manager, and the processes beside them.
LAPTOP_NAV_NODES = (
    "/nav2_container",
    *(f"/{node}" for node in NAV_NODES),
    "/local_costmap/local_costmap",
    "/global_costmap/global_costmap",
    "/lifecycle_manager_navigation",
    "/goal_server",
    "/run_recorder",
    "/bag_recorder",
    "/gaze",  # the gaze arbiter: every head decision, the stall look, path and reverse gaze
)


def node_host(node: str) -> tuple[str, str]:
    """Where a node's process lives, as ``(side, container)``: what ros/flags.sh execs into to
    reach the node's parameters. The camera nodes are in the Mac's SLAM container
    (``pepin-vslam``), Nav2 with the goal server and the recorder in its navigation container
    (:data:`NAV_CONTAINER`); everything else — the sensors — is the board's ``pepin-ros``."""
    name = f"/{node.lstrip('/')}"
    if name in LAPTOP_SLAM_NODES:
        return "laptop", "pepin-vslam"
    if name in LAPTOP_NAV_NODES:
        return "laptop", NAV_CONTAINER
    if name in LAPTOP_VIO_NODES:
        return "laptop", VIO_CONTAINER
    return "board", "pepin-ros"
