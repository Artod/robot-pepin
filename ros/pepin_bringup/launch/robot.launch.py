"""Sensors and bridges in as few processes as the board can afford.

One component container (high CPU priority) holds the LD19 driver, the hull
box filter that turns its scan into /scan, the static base_link->laser
transform, the lifecycle manager that activates the driver, and the Foxglove
bridge. A separate Python process runs ``base_bridge`` (odometry, TF, /cmd_vel
to the wheels). Every extra ROS process costs ~140 MB on this 1.5 GB board, so
composition is not a nicety here.

Arguments:

- ``laser_roll`` (default pi): the LD19 hangs upside down; the driver emits a
  standard counter-clockwise scan for an upright sensor, so a roll of pi mirrors
  it back. If a wall in front of the cart draws mirrored left/right, pass 0.0.
- ``laser_yaw`` (default -1.5272 rad = -87.5 deg, from config/lidar.json).
- ``lidar_port`` (default /dev/lidar), ``lidar_debug`` (default false),
- ``foxglove`` (default true) and ``foxglove_port`` (default 8765),
- ``tof`` (default false): the ToF bridge, once Nav2 has a layer that reads it.
- ``base_bridge_cpp`` (default false): run the C++ base bridge (``pepin_base_cpp``, ~25 MB)
  instead of the Python one. Same node, parameters and wire protocol; the default flips
  once it has driven the cart.
- ``imu`` (default false): read the MPU6050 on /dev/i2c-2 inside the C++ bridge and publish
  /imu/data_raw. Needs ``base_bridge_cpp:=true``; the Python bridge has no IMU.
"""

import math

from launch import Condition, LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from pepin.deployment import BASE_MAX_ANGULAR_RAD_S, BASE_MAX_LINEAR_M_S
from pepin.footprint import hull_box

LASER_X, LASER_Y, LASER_Z = 0.005, 0.0, 0.20
# The MPU6050 sits flat on the chassis over base_link, Z up and its X arrow forward:
# no rotation, only the height of the deck it is glued to.
IMU_X, IMU_Y, IMU_Z = 0.0, 0.0, 0.10
# The cart's own body, with 5 cm of margin: returns just outside the exact hull are its own
# posts and cables, they travel with it, and the costmap turned them into a wall that made
# every in-place turn "a collision ahead" (measured 2026-09-08: |y| 0.28-0.34 m in 41-71%
# of the scans on the home legs). Anything real at 5 cm from the body is inside the swing
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
    """Build the container once the launch arguments have values (the quaternion needs numbers)."""
    roll = float(LaunchConfiguration("laser_roll").perform(context))
    yaw = float(LaunchConfiguration("laser_yaw").perform(context))
    qx, qy, qz, qw = quaternion(roll, 0.0, yaw)
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
                    "translation.x": LASER_X,
                    "translation.y": LASER_Y,
                    "translation.z": LASER_Z,
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
                        "translation.x": IMU_X,
                        "translation.y": IMU_Y,
                        "translation.z": IMU_Z,
                        # The GY-521 sits with its Y axis up (gravity reads +9.8 on Y,
                        # 2026-09-07): roll +90 deg maps the chip's Y onto base_link's Z,
                        # so its Y gyro is our yaw rate.
                        "rotation.x": 0.7071068,
                        "rotation.y": 0.0,
                        "rotation.z": 0.0,
                        "rotation.w": 0.7071068,
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
    return LaunchDescription(
        [
            DeclareLaunchArgument("laser_roll", default_value="3.14159265"),
            DeclareLaunchArgument("laser_yaw", default_value="-1.5272"),
            DeclareLaunchArgument("lidar_port", default_value="/dev/lidar"),
            DeclareLaunchArgument("lidar_debug", default_value="false"),
            DeclareLaunchArgument("foxglove", default_value="true"),
            DeclareLaunchArgument("foxglove_port", default_value="8765"),
            DeclareLaunchArgument("tof", default_value="false"),
            DeclareLaunchArgument("base_bridge_cpp", default_value="false"),
            DeclareLaunchArgument("imu", default_value="false"),
            OpaqueFunction(function=sensors_container),
            base,
            ekf,
            tof,
        ]
    )
