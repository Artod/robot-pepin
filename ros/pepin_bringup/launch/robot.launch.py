"""Sensors and bridges in as few processes as the board can afford.

Two component containers (high CPU priority, each respawned with fresh nodes when it
dies: pepin_bringup.launch_kit): the lidar's holds the LD19 driver, the hull box filter that
turns its scan into /scan, the static base_link->laser transform and the lifecycle manager that
activates the driver; the base's holds the C++ base bridge (``pepin_base_cpp``: odometry, the
/cmd_vel sink, the IMU, and the neck's encoders as /neck/state and base_link -> camera_link at
the state line's stamp, its geometry from config/neck.json and the camera's link frame from
config/camera.json, pepin.neck.bridge_parameters). Every extra ROS process costs ~140 MB on this
1.5 GB board, so
composition is not a nicety here. The board runs no Foxglove bridge: the laptop's
(vslam.launch.py) sees these topics through zenoh.

The sensor mounts are not arguments: base_link -> laser comes from config/lidar.json (the LD19
hangs upside down, roll pi, yaw -87.5 deg: the calibration's one home), read through
pepin.mounts.Mounts — the one loader every publisher of a sensor frame uses — and found by
pepin.deployment.config_file at launch time: on the board under /ws/pepin_src/config, which
ros/sync.sh keeps beside the library. The IMU's readings are stamped in base_link (the bridge
rotates each one into base_link's axes itself, base_bridge.cpp to_base_axes, the mount of
config/imu.json), so no imu_link frame is published.

Arguments:

- ``tof`` (default true): the ToF bridge (its scan fans reach the laptop's costmaps).
- ``imu`` (default true): read the MPU6050 on /dev/i2c-2 inside the base bridge and publish
  /imu/data_raw. The IMU is a sensor of the filter below, never its precondition: with
  ``imu:=false`` the EKF still runs.
- ``ekf`` (default true): fuse whatever odometry sources are alive (ros/params/ekf.yaml) and
  own odom -> base_link. ``ekf:=false`` is the way back to the bridge's own transform with no
  filter in the chain.
- ``laser_odom`` (default **true**): laser odometry (rf2o, in the image) matching each scan
  against the one before it — no map, no graph — and publishing /odom_laser, which the EKF fuses
  as a twist. It is a source of the filter, never its precondition: ``laser_odom:=false`` leaves
  the wheels, the gyro and the camera exactly as they were. ``ros/feature.sh laser_odom on|off``
  flips it; the node publishes NO transform (the EKF owns odom -> base_link).
- ``board_bag`` (default false): the board's raw sensors recorded on the board itself, always,
  into minute MCAP files under /maps/board_rec, capped at 20 GB with 10 GB of the card always
  left free (pepin.board_bag: one long-lived ``ros2 bag record`` and its supervisor, niced under
  every sensor). ros/feature.sh board_bag on|off flips it.
"""

import math

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode
from pepin_bringup.launch_kit import respawned_container

from pepin.base_link import STATE_HZ
from pepin.deployment import (
    BASE_MAX_ANGULAR_RAD_S,
    BASE_MAX_LINEAR_M_S,
    IMU_RATE_HZ,
    LASER_ODOM_HZ,
    LASER_ODOM_TOPIC,
    LASER_ODOM_TWIST_VARIANCE,
    config_file,
)
from pepin.footprint import hull_box
from pepin.head_imu import HeadImuConfig, camera_from_imu
from pepin.mounts import Mounts
from pepin.neck import JOINT_NAMES, NeckConfig, bridge_parameters
from pepin.sensor_timing import imu_timing

# Our own Python nodes come back by themselves after this pause (a code change is one kicked
# process: ros/board.sh kick <node>).
RESPAWN = {"respawn": True, "respawn_delay": 2.0}


MOUNTS = Mounts.load()
LASER = MOUNTS.lidar.transform()
# The cart's own body, with 5 cm of margin: returns just outside the exact hull are its own
# posts and cables, they travel with it, and the costmap turned them into a wall that made
# every in-place turn "a collision ahead" (measured 2026-09-08: |y| 0.28-0.34 m in 41-71%
# of the scans on the home legs). Anything real at the contact band around the hull is inside the
# swing
# circle anyway and is handled by the ToF sensors and the inflation.
HULL = (
    hull_box()
)  # the hull plus the contact band: what the cart is parked against is not an obstacle


def quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """x, y, z, w of the rotation Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def lidar_parts(context: LaunchContext) -> list:  # type: ignore[type-arg]
    """The lidar's nodes, described anew on every call (a respawn loads fresh ones, see
    pepin_bringup.launch_kit): the LD19 driver, the hull filter that makes /scan of its scan,
    base_link -> laser and the driver's lifecycle manager."""
    laser_x, laser_y, laser_z, laser_roll, laser_pitch, laser_yaw = LASER
    qx, qy, qz, qw = quaternion(laser_roll, laser_pitch, laser_yaw)
    return [
        ComposableNode(
            package="ldlidar_component",
            plugin="ldlidar::LdLidarComponent",
            name="ldlidar_node",
            parameters=[
                {
                    "general.debug_mode": False,
                    "comm.serial_port": "/dev/lidar",  # board/99-pepin-usb.rules
                    "comm.baudrate": 230400,
                    "comm.timeout_msec": 1000,
                    "lidar.model": "LD19",
                    "lidar.rot_verse": "CCW",
                    "lidar.units": "M",
                    "lidar.frame_id": "laser",
                    "lidar.bins": 455,  # a fixed beam count, scan after scan
                    "lidar.range_min": 0.05,
                    "lidar.range_max": 12.0,
                    "lidar.enable_angle_crop": False,
                }
            ],
            # No remap on the driver: it gates publishing on count_subscribers() of its own
            # topic name. The filter below subscribes to it and republishes /scan.
            extra_arguments=[{"use_intra_process_comms": False}],
        ),
        ComposableNode(
            package="laser_filters",
            plugin="ScanToScanFilterChain",
            name="scan_filter",
            parameters=[
                {
                    "filter1.name": "hull",
                    "filter1.type": "laser_filters/LaserScanBoxFilter",
                    "filter1.params.box_frame": "base_link",
                    "filter1.params.min_x": HULL["min_x"],
                    "filter1.params.max_x": HULL["max_x"],
                    "filter1.params.min_y": HULL["min_y"],
                    "filter1.params.max_y": HULL["max_y"],
                    "filter1.params.min_z": -1.0,
                    "filter1.params.max_z": 1.0,
                    "filter1.params.invert": False,
                }
            ],
            remappings=[("scan", "/ldlidar_node/scan"), ("scan_filtered", "/scan")],
        ),
        ComposableNode(
            package="tf2_ros",
            plugin="tf2_ros::StaticTransformBroadcasterNode",
            name="base_to_laser",
            parameters=[
                {
                    "frame_id": "base_link",
                    "child_frame_id": "laser",
                    "translation.x": laser_x,
                    "translation.y": laser_y,
                    "translation.z": laser_z,
                    "rotation.x": qx,
                    "rotation.y": qy,
                    "rotation.z": qz,
                    "rotation.w": qw,
                }
            ],
        ),
        ComposableNode(
            package="nav2_lifecycle_manager",
            plugin="nav2_lifecycle_manager::LifecycleManager",
            name="lifecycle_manager_sensors",
            parameters=[{"autostart": True, "node_names": ["ldlidar_node"], "bond_timeout": 0.0}],
        ),
    ]


def base_parts(context: LaunchContext) -> list:  # type: ignore[type-arg]
    """The base's own node, described anew on every call: the C++ base bridge (wheels, IMU, the
    gyro-bias tracker). Kept apart from the lidar's so a sensor that dies cannot take the
    actuator down with it."""
    imu_on = LaunchConfiguration("imu").perform(context).lower() == "true"
    ekf_on = LaunchConfiguration("ekf").perform(context).lower() == "true"
    head_on = LaunchConfiguration("head_imu").perform(context).lower() == "true"
    return [
        ComposableNode(
            package="pepin_base_cpp",
            plugin="pepin::BaseBridge",
            name="base_bridge",
            # The transform follows the FILTER, not the IMU (2026-09-15): with the EKF up it
            # owns odom -> base_link and the bridge publishes /odom only, whether or not the
            # gyro is there to be fused. Keyed on the IMU, `imu off` left the stack with no
            # /odometry/filtered at all — the tracker of the day read that topic and never
            # localised.
            # The speed caps are the base's own, not the bridge's defaults (0.25 m/s).
            parameters=[
                {
                    "imu_enable": imu_on,
                    "imu_rate_hz": IMU_RATE_HZ,
                    "publish_tf": not ekf_on,
                    "max_linear_m_s": BASE_MAX_LINEAR_M_S,
                    "max_angular_rad_s": BASE_MAX_ANGULAR_RAD_S,
                    **imu_parameters(),
                    **neck_parameters(),
                    **head_imu_parameters(head_on),
                }
            ],
        )
    ]


def head_imu_parameters(enable: bool) -> dict[str, object]:
    """The head IMU's link, publish cap and mast filter from config/head_imu.json, and the IMU's
    rotation into camera_link from config/camera.json's head_imu block (pepin.head_imu), read
    anew at every (re)spawn. ``enable`` is the head_imu launch argument: off, the bridge opens no
    link to head_server and nothing about the head exists. Without the json the bridge keeps its
    own defaults and says so; without the extrinsics the mast filter does not run."""
    try:
        config = HeadImuConfig.load()
    except (OSError, KeyError, ValueError) as exc:
        print(
            f"[robot.launch] no head IMU config ({exc}): the bridge's defaults, head_imu {enable}"
        )
        return {"head_imu_enable": enable}
    try:
        rotation = camera_from_imu(config_file("camera.json").parent)
    except (OSError, KeyError, ValueError) as exc:
        print(f"[robot.launch] no head IMU extrinsics ({exc}): the mast filter stays off")
        rotation = None
    if enable and rotation is None:
        print("[robot.launch] head_imu on without config/camera.json's head_imu: /head/imu only")
    return config.bridge_parameters(rotation, enable)


def imu_parameters() -> dict[str, float]:
    """The MPU6050's output rate and filter delay from config/imu.json's timing block
    (pepin.sensor_timing), read anew at every (re)spawn; without the block the bridge keeps its
    own defaults, 1 kHz and a stamp at the read, and this says so."""
    try:
        return imu_timing().bridge_parameters()
    except (OSError, KeyError, ValueError) as exc:
        print(f"[robot.launch] no IMU timing ({exc}): /imu/data_raw is stamped at the read")
        return {}


def neck_parameters() -> dict[str, object]:
    """The bridge's neck: config/neck.json's geometry and config/camera.json's link frame, read
    anew at every (re)spawn so a re-measured mount is a restart, never a rebuild. The bridge
    publishes base_link -> camera_link alone; the laptop's static copy stays off (ros/laptop.sh
    vslam without --fixed-head). Without config/neck.json there is no camera edge from here."""
    try:
        neck = NeckConfig.from_json(config_file("neck.json"))
    except (OSError, KeyError, ValueError) as exc:
        print(f"[robot.launch] no neck ({exc}): no /neck/state, no camera transform")
        return {"neck_camera_frame": ""}
    return {
        **bridge_parameters(neck),
        "neck_parent_frame": "base_link",
        "neck_camera_frame": MOUNTS.camera.link_frame,
        "neck_joint_names": list(JOINT_NAMES),
        "neck_publish_hz": STATE_HZ,  # every state line: the camera's pose at the odometry's rate
    }


def sensors_container(context: LaunchContext) -> list:  # type: ignore[type-arg]
    """The sensing containers once the launch arguments have values: the lidar's and the base's,
    each respawned with fresh nodes."""
    # TWO FAILURE DOMAINS (2026-09-24). Until then the lidar driver, its filter AND the base
    # bridge shared one process, and at 00:11Z on 2026-09-24 the LD19 driver aborted on a
    # deactivate ("*** bit out of range 0 - FD_SETSIZE on fd_set ***", exit -6: a select() on a
    # descriptor its own close had just invalidated) and took the wheels, the IMU and the gyro's
    # bias tracker down with it, unrespawned: the EKF coasted on the camera alone for hours and the
    # 'camera-only' leg that followed had no base to drive. A dead lidar is exactly the failure a
    # camera-only cart must ride out, so the lidar gets a process of its own and the base another,
    # and each comes back by itself two seconds after it dies (RESPAWN). The rest of the stack
    # already degrades on its own: RTAB-Map's registration follows what the snapshots carry
    # (pepin.graphmode), the costmaps' lidar layer goes quiet, the EKF loses rf2o and keeps the
    # wheels, the gyro and the camera.
    # sensing first: a starved driver ships scans seconds late; the wheels and the gyro alike
    return respawned_container("lidar_container", lidar_parts, "nice -n -10") + (
        respawned_container("base_container", base_parts, "nice -n -10")
    )


def generate_launch_description() -> LaunchDescription:
    # Wheels, gyro and the camera's odometry fused in the plane (ros/params/ekf.yaml): the wheels
    # over-report rotation on carpet, the gyro does not; the filter publishes odom -> base_link
    # instead of the bridge, and /odometry/filtered, which is the odometry the recorder reads.
    #
    # THE IMU IS NOT A PRECONDITION (2026-09-15): the bridge hands ``publish_tf`` over whenever
    # the filter runs (base_parts). The IMU is one of three sources and robot_localization needs
    # none of them in particular: it initialises on the first measurement of ANY configured
    # source and then publishes at ``frequency`` forever, a silent imu0 costing nothing but its
    # own weight. Gated on ``imu`` instead, `ros/feature.sh imu
    # off` took the whole filter down with the gyro: /odometry/filtered went to zero messages, the
    # tracker of the day carried its scans on an odometry that never arrived and goto refused
    # with "not localized". Without the gyro the heading comes off the wheels (ekf.yaml's odom0
    # index 11).
    ekf = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        prefix="nice -n -5",
        parameters=["/params/ekf.yaml"],
        condition=IfCondition(LaunchConfiguration("ekf")),
    )
    # A rclpy process costs ~140 MB on this board; ros/feature.sh tof off saves it.
    # Respawned like every other node of this launch: on 2026-09-21 the bridge died once (a
    # logging call rclpy refuses) and stayed dead — a near-field sensor that silently never comes
    # back is worse than one that was never on.
    tof = Node(
        package="pepin_bringup",
        executable="tof_bridge",
        output="screen",
        condition=IfCondition(LaunchConfiguration("tof")),
        **RESPAWN,
    )
    # LASER ODOMETRY: the cart's own motion from consecutive scans, scan to scan and with no map
    # (rf2o, built into the image from a pinned commit, ros/Dockerfile). It is the odometry input
    # that a wheel spinning on carpet cannot fool and the board keeps when the WiFi goes, which is
    # why it lives here; the EKF fuses its TWIST as odom3 (ros/params/ekf.yaml) and nothing else
    # reads it. ``publish_tf`` is FALSE and must stay so: the filter owns odom -> base_link, and a
    # second publisher of that edge is the oldest bug in this stack. ``init_pose_from_topic`` is
    # emptied because upstream's default makes the node wait for /base_pose_ground_truth, a
    # simulator topic nothing here publishes, before it processes a single scan.
    laser_odom = Node(
        package="rf2o_laser_odometry",
        executable="rf2o_laser_odometry_node",
        output="screen",
        # Niced with the filter it feeds: a scan matched late is a velocity measured late.
        # TWO names, because this one process holds two rclcpp nodes: the matcher itself is a
        # Node too, which is also why no ``name=`` is passed here — that is a process-wide
        # ``__node:=`` remap and it would give both of them the same name. The names are the
        # patch's (ros/patches/rf2o-base-twist.patch renames the outer one); the parameters
        # reach them through launch_ros's ``/**`` wildcard, which needs no name either.
        prefix="nice -n -5",
        parameters=[
            {
                "laser_scan_topic": "/scan",
                "odom_topic": f"/{LASER_ODOM_TOPIC}",
                "base_frame_id": "base_link",
                "odom_frame_id": "odom",
                "publish_tf": False,
                "init_pose_from_topic": "",
                "freq": LASER_ODOM_HZ,
                # Both patched in (see ros/Dockerfile): the twist in base_link instead of in the
                # laser's own frame, and a covariance the filter can weigh.
                "base_frame_twist": True,
                "twist_covariance_vx": LASER_ODOM_TWIST_VARIANCE["vx"],
                "twist_covariance_vy": LASER_ODOM_TWIST_VARIANCE["vy"],
                "twist_covariance_vyaw": LASER_ODOM_TWIST_VARIANCE["vyaw"],
                # Measured on the robot 2026-09-23 against the wheels and the gyro (0.40 m
                # reverse, 60 deg turn): the differenced pose's yaw rate carries the robot's sign,
                # its forward speed the opposite. Live parameters: `ros2 param set /laser_odometry
                # twist_sign_linear 1.0` is the way back.
                "twist_sign_linear": -1.0,
                "twist_sign_angular": 1.0,
            }
        ],
        condition=IfCondition(LaunchConfiguration("laser_odom")),
        **RESPAWN,
    )
    # THE BOARD'S OWN RECORDING (pepin.board_bag): every raw sensor topic into minute MCAP files on
    # the card, for as long as the stack runs, so a drive whose WiFi stalled is still whole here.
    # One recorder opened once (a zenoh session opened per goal stalled the board's delivery,
    # 2026-09-25), its supervisor deleting the oldest minutes past the cap. The least important
    # process on the board: niced under every sensor, and a supervisor that dies takes its
    # recorder with it (PR_SET_PDEATHSIG) before the respawn starts a new one.
    board_bag = ExecuteProcess(
        cmd=["python3", "-m", "pepin.board_bag", "--dir", "/maps/board_rec"],
        output="screen",
        prefix="nice -n 10",
        condition=IfCondition(LaunchConfiguration("board_bag")),
        **RESPAWN,
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("tof", default_value="true"),
            DeclareLaunchArgument("imu", default_value="true"),
            DeclareLaunchArgument("ekf", default_value="true"),
            DeclareLaunchArgument("laser_odom", default_value="true"),
            DeclareLaunchArgument("board_bag", default_value="false"),
            # The head IMU (head_server's 3340 stream) into the bridge: off until the head exists
            # and head_server runs (board/pepin-ros.service's PEPIN_HEAD_IMU, ros/feature.sh).
            DeclareLaunchArgument("head_imu", default_value="false"),
            OpaqueFunction(function=sensors_container),
            ekf,
            tof,
            laser_odom,
            board_bag,
        ]
    )
