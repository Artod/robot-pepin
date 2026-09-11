"""Sensors and bridges in as few processes as the board can afford.

One component container (high CPU priority) holds the LD19 driver, the hull
box filter that turns its scan into /scan, the static base_link->laser
transform, the lifecycle manager that activates the driver, and the Foxglove
bridge. A separate Python process runs ``base_bridge`` (odometry, TF, /cmd_vel
to the wheels). Every extra ROS process costs ~140 MB on this 1.5 GB board, so
composition is not a nicety here.

The sensor mounts are not arguments: base_link -> laser comes from config/lidar.json (the LD19
hangs upside down, roll pi, yaw -87.5 deg: the calibration's one home) and base_link -> imu_link
from config/imu.json, both read through pepin.mounts.Mounts — the one loader every publisher of
a sensor frame uses — and found by pepin.deployment.config_file at launch time: on the board
under /ws/pepin_src/config, which ros/sync.sh keeps beside the library.

Arguments:

- ``lidar_port`` (default /dev/lidar), ``lidar_debug`` (default false),
- ``foxglove`` (default true) and ``foxglove_port`` (default 8765),
- ``tof`` (default false): the ToF bridge, once Nav2 has a layer that reads it.
- ``base_bridge_cpp`` (default false): run the C++ base bridge (``pepin_base_cpp``, ~25 MB)
  instead of the Python one. Same node, parameters and wire protocol; the default flips
  once it has driven the cart.
- ``imu`` (default false): read the MPU6050 on /dev/i2c-2 inside the C++ bridge and publish
  /imu/data_raw. Needs ``base_bridge_cpp:=true``; the Python bridge has no IMU.
- ``neck`` (default false): the neck's encoders as /neck/state and, behind the node's live
  ``neck_tf`` switch, base_link -> camera_link from them (pepin_bringup.neck_state, a Python
  process, ~150 MB). The laptop's camera node must then keep its static edge off
  (ros/laptop.sh vslam --neck); ros/feature.sh neck on|off flips this one.
"""

import math

from launch import Condition, LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from pepin.deployment import BASE_MAX_ANGULAR_RAD_S, BASE_MAX_LINEAR_M_S, bridge_admin_for
from pepin.footprint import hull_box
from pepin.mounts import Mounts

# Our own Python nodes come back by themselves after this pause (a code change is one kicked
# process: ros/thin.sh kick <node>), through a ghost wait of their own name first, as in
# nav.launch.py: a crashed node's name outlives it in the bridge by the DDS lease.
RESPAWN = {"respawn": True, "respawn_delay": 2.0}


def _after_ghost(*names: str) -> str:
    """A command prefix that waits until the bridge on this host lists none of ``names`` and
    then becomes the command (pepin_bringup.ghost_wait; an unreachable admin is not waited for)."""
    return f"python3 -m pepin_bringup.ghost_wait {bridge_admin_for('board')} {' '.join(names)} --"


MOUNTS = Mounts.load()
LASER = MOUNTS.lidar.transform()
# The GY-521 sits with its Y axis up (gravity reads +9.8 on Y, 2026-09-07): roll +90 deg maps
# the chip's Y onto base_link's Z, so its Y gyro is our yaw rate. The numbers live in the file.
IMU = MOUNTS.imu.transform()
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


def sensors_container(context: LaunchContext) -> list:  # type: ignore[type-arg]
    """Build the container once the launch arguments have values."""
    laser_x, laser_y, laser_z, laser_roll, laser_pitch, laser_yaw = LASER
    qx, qy, qz, qw = quaternion(laser_roll, laser_pitch, laser_yaw)
    imu_x, imu_y, imu_z, imu_roll, imu_pitch, imu_yaw = IMU
    ix, iy, iz, iw = quaternion(imu_roll, imu_pitch, imu_yaw)
    debug = LaunchConfiguration("lidar_debug").perform(context).lower() == "true"
    port = int(LaunchConfiguration("foxglove_port").perform(context))
    components = [
        ComposableNode(
            package="ldlidar_component",
            plugin="ldlidar::LdLidarComponent",
            name="ldlidar_node",
            parameters=[
                {
                    "general.debug_mode": debug,
                    "comm.serial_port": LaunchConfiguration("lidar_port").perform(context),
                    "comm.baudrate": 230400,
                    "comm.timeout_msec": 1000,
                    "lidar.model": "LD19",
                    "lidar.rot_verse": "CCW",
                    "lidar.units": "M",
                    "lidar.frame_id": "laser",
                    "lidar.bins": 455,  # fixed size: slam_toolbox wants it
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
    if LaunchConfiguration("imu").perform(context).lower() == "true":
        components.append(
            ComposableNode(
                package="tf2_ros",
                plugin="tf2_ros::StaticTransformBroadcasterNode",
                name="base_to_imu",  # documentation only: the bridge publishes in base_link
                parameters=[
                    {
                        "frame_id": "base_link",
                        "child_frame_id": "imu_link",
                        "translation.x": imu_x,
                        "translation.y": imu_y,
                        "translation.z": imu_z,
                        "rotation.x": ix,  # config/imu.json: roll +90 deg, the chip's Y up
                        "rotation.y": iy,
                        "rotation.z": iz,
                        "rotation.w": iw,
                    }
                ],
            )
        )
    if LaunchConfiguration("base_bridge_cpp").perform(context).lower() == "true":
        imu_on = LaunchConfiguration("imu").perform(context).lower() == "true"
        components.append(
            ComposableNode(
                package="pepin_base_cpp",
                plugin="pepin::BaseBridge",
                name="base_bridge",
                # With the IMU the EKF owns odom -> base_link; the bridge then publishes /odom only.
                # The speed caps are the base's own, not the bridge's defaults (0.25 m/s).
                parameters=[
                    {
                        "imu_enable": imu_on,
                        "publish_tf": not imu_on,
                        "max_linear_m_s": BASE_MAX_LINEAR_M_S,
                        "max_angular_rad_s": BASE_MAX_ANGULAR_RAD_S,
                    }
                ],
            )
        )
    if LaunchConfiguration("foxglove").perform(context).lower() == "true":
        components.append(
            ComposableNode(
                package="foxglove_bridge",
                plugin="foxglove_bridge::FoxgloveBridge",
                name="foxglove_bridge",
                parameters=[{"port": port, "send_buffer_limit": 10000000}],
            )
        )
    container = ComposableNodeContainer(
        name="sensors_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_isolated",
        output="screen",
        prefix="nice -n -10",  # sensing first: a starved driver ships scans seconds late
        composable_node_descriptions=components,
    )
    return [container]


def base_bridge(
    package: str,
    condition: Condition,
    parameters: list | None = None,  # type: ignore[type-arg]
) -> Node:
    """The base bridge from ``package`` (the Python or the C++ build), niced above Nav2."""
    return Node(
        package=package,
        executable="base_bridge",
        output="screen",
        prefix="nice -n -5",
        parameters=parameters or [],
        condition=condition,
    )


def generate_launch_description() -> LaunchDescription:
    use_cpp = LaunchConfiguration("base_bridge_cpp")
    # The Python bridge keeps the transform: without the C++ bridge there is no EKF to own it.
    base = base_bridge("pepin_bringup", UnlessCondition(use_cpp), [{"publish_tf": True}])
    # Only the C++ bridge reads the IMU: the Python one has no such parameter.
    imu = LaunchConfiguration("imu")
    # Wheels + gyro fused in the plane (ros/params/ekf.yaml): the wheels over-report rotation
    # on carpet, the gyro does not; the filter publishes odom -> base_link instead of the bridge.
    # The EKF owns odom -> base_link only when the C++ bridge is what feeds it: with the Python
    # bridge there is no /imu/data_raw to fuse and both would broadcast the same transform.
    ekf = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        prefix="nice -n -5",
        parameters=["/params/ekf.yaml"],
        condition=IfCondition(
            PythonExpression(["'", imu, "' == 'true' and '", use_cpp, "' == 'true'"])
        ),
    )
    # Off by default for now: a rclpy process costs ~140 MB and Nav2 does not read Range yet.
    tof = Node(
        package="pepin_bringup",
        executable="tof_bridge",
        output="screen",
        condition=IfCondition(LaunchConfiguration("tof")),
    )
    # The neck's encoders and the live camera transform (pepin_bringup.neck_state). As a module,
    # like the recorder: the image's console scripts are generated at build time and the sources
    # are mounted over them. Off by default until the switch-over is measured: the laptop's
    # static edge must go off in the same breath (two publishers of one edge fight).
    neck = ExecuteProcess(
        cmd=["python3", "-m", "pepin_bringup.neck_state"],
        output="screen",
        prefix=_after_ghost("/neck_state"),
        condition=IfCondition(LaunchConfiguration("neck")),
        **RESPAWN,
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("lidar_port", default_value="/dev/lidar"),
            DeclareLaunchArgument("lidar_debug", default_value="false"),
            DeclareLaunchArgument("foxglove", default_value="true"),
            DeclareLaunchArgument("foxglove_port", default_value="8765"),
            DeclareLaunchArgument("tof", default_value="false"),
            DeclareLaunchArgument("base_bridge_cpp", default_value="false"),
            DeclareLaunchArgument("imu", default_value="false"),
            DeclareLaunchArgument("neck", default_value="false"),
            OpaqueFunction(function=sensors_container),
            base,
            ekf,
            tof,
            neck,
        ]
    )
