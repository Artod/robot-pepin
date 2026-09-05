"""Sensors and bridges: lidar, base (odom + cmd_vel), ToF, static transforms, Foxglove bridge.

Nothing here plans or moves the robot on its own; it is the layer every other
launch file (Nav2, SLAM) sits on. Arguments:

- ``laser_roll`` (default pi): our LD19 hangs upside down; the driver emits a
  standard counter-clockwise scan for an upright sensor, so a roll of pi mirrors
  it back. If a wall in front of the cart draws mirrored left/right, pass 0.0.
- ``laser_yaw`` (default -1.5272 rad = -87.5 deg, from config/lidar.json).
- ``lidar_port`` (default /dev/lidar), ``foxglove_port`` (default 8765),
  ``lidar_debug`` (default false: the driver logs every frame when true).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

LASER_X, LASER_Y, LASER_Z = "0.005", "0.0", "0.20"


def generate_launch_description() -> LaunchDescription:
    laser_roll = LaunchConfiguration("laser_roll")
    laser_yaw = LaunchConfiguration("laser_yaw")
    lidar_port = LaunchConfiguration("lidar_port")
    foxglove_port = LaunchConfiguration("foxglove_port")

    # The driver ships as a composable lifecycle component, not an executable.
    lidar = ComposableNodeContainer(
        name="ldlidar_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_isolated",
        output="screen",
        composable_node_descriptions=[
            ComposableNode(
                package="ldlidar_component",
                plugin="ldlidar::LdLidarComponent",
                name="ldlidar_node",
                parameters=[
                    {
                        "general.debug_mode": LaunchConfiguration("lidar_debug"),
                        "comm.serial_port": lidar_port,
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
                # No remap: the driver gates publishing on count_subscribers() of its own
                # topic name, and a remapped name left it convinced nobody listens. Nav2 is
                # pointed at /ldlidar_node/scan in ros/params/nav2_params.yaml instead.
                extra_arguments=[{"use_intra_process_comms": False}],
            )
        ],
    )
    # The driver is a lifecycle node: this brings it to active and keeps it there.
    lidar_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_sensors",
        output="screen",
        parameters=[{"autostart": True, "node_names": ["ldlidar_node"], "bond_timeout": 0.0}],
    )
    base = Node(package="pepin_bringup", executable="base_bridge", output="screen")
    tof = Node(package="pepin_bringup", executable="tof_bridge", output="screen")
    laser_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_laser",
        arguments=[
            *("--x", LASER_X, "--y", LASER_Y, "--z", LASER_Z),
            *("--roll", laser_roll, "--pitch", "0.0", "--yaw", laser_yaw),
            *("--frame-id", "base_link", "--child-frame-id", "laser"),
        ],
    )
    foxglove = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        output="screen",
        parameters=[{"port": foxglove_port, "send_buffer_limit": 10000000}],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("laser_roll", default_value="3.14159265"),
            DeclareLaunchArgument("laser_yaw", default_value="-1.5272"),
            DeclareLaunchArgument("lidar_port", default_value="/dev/lidar"),
            DeclareLaunchArgument("lidar_debug", default_value="false"),
            DeclareLaunchArgument("foxglove_port", default_value="8765"),
            lidar,
            lidar_manager,
            base,
            tof,
            laser_tf,
            foxglove,
        ]
    )
