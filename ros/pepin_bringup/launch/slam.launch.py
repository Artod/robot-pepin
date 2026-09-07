"""Build a map while driving: slam_toolbox (online, asynchronous) over robot.launch.py's sensors.

The ROS answer to "does the robot improve its map while it drives?": AMCL only
localises on a frozen map; mapping is a separate mode. Drive the flat with
``ros/teleop.sh``, watch the map grow in Foxglove (/map), then save it with
``ros/savemap.sh NAME`` and navigate on it with ``nav.launch.py map:=/maps/NAME.yaml``.
slam_toolbox also has a localization mode that keeps refining a saved map; that
is the next step once a clean map exists.

slam_toolbox is a lifecycle node in Jazzy: left alone it sits unconfigured and
never publishes. The event handlers below configure it at start and activate it
once it reports inactive — the pattern of slam_toolbox's own launch file.
"""

from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description() -> LaunchDescription:
    slam = LifecycleNode(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        namespace="",
        output="screen",
        prefix="nice -n 5",
        parameters=[
            {
                "use_sim_time": False,
                "odom_frame": "odom",
                "map_frame": "map",
                "base_frame": "base_link",
                "scan_topic": "/scan",
                "mode": "mapping",
                "resolution": 0.05,
                "max_laser_range": 12.0,
                "minimum_travel_distance": 0.15,
                # Turns in place must reach the mapper: by default slam_toolbox forwards a scan
                # on translation alone and dead-reckons every turn by the wheels, which over-report
                # rotation by 10-25% here. That was the 40 deg bend in every map before 2026-09-06.
                "minimum_travel_heading": 0.1,
                "check_min_dist_and_heading_precisely": True,
                "minimum_time_interval": 0.0,  # was 0.5 s: keyframes capped at 2 Hz
                "use_scan_barycenter": True,
                "map_update_interval": 2.0,
                "transform_publish_period": 0.05,
                "do_loop_closing": True,
                "loop_search_maximum_distance": 3.0,
                "throttle_scans": 2,  # every 2nd scan: the A53 must keep up or the map tears
                "scan_buffer_size": 10,
                "use_scan_matching": True,
                "ceres_linear_solver": "SPARSE_NORMAL_CHOLESKY",
                "ceres_preconditioner": "SCHUR_JACOBI",
                "solver_plugin": "solver_plugins::CeresSolver",
            }
        ],
    )
    configure = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam),
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )
    activate_when_inactive = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam,
            start_state="configuring",
            goal_state="inactive",
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(slam),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    )
                )
            ],
        )
    )
    return LaunchDescription([slam, activate_when_inactive, configure])
