"""The visual-inertial odometry on the laptop: OpenVINS's ROS 2 subscriber in its own container.

``ros/laptop.sh vio`` starts this in ``pepin-vio`` on ``pepin-laptop:vio`` (ros/Dockerfile.vio),
beside ``pepin-vslam`` and behind the same zenoh router: OpenVINS reads the two rectified eyes
(``/camera/image``, ``/camera/right/image``, camera_stream's) and the head IMU (``/head/imu``, the
board's base bridge), and publishes ``/ov_msckf/poseimu`` (the IMU's pose in its gravity frame,
per image update), which pepin_bringup.visual_odometry reads under ``vo_input:=vio``
(``ros/laptop.sh vslam --vo-vio``) and turns into ``/vo`` for the board's EKF. Nothing here talks
to the EKF directly, and nothing here owns a transform.

The config is generated, never edited: ``ros/tools/vio_config.py`` (run by ``ros/laptop.sh vio``
inside this image at every start) writes estimator_config.yaml and the two Kalibr chains into
ros/maps/vio (``/maps/vio`` here) from
config/stereo_calibration.json, config/camera.json's head_imu block and config/head_imu.json. The
node is respawned when it dies; it does not reset itself when it diverges (the relay marks it
lost), so ``ros/laptop.sh vio kick`` restarts it, AT REST: its static initialisation needs
stillness, then motion.

``executor`` (default single) is run_subscribe_msckf's executor
(ros/patches/openvins-executor.patch, built in by Dockerfile.vio's EXECUTOR=1): upstream's multi
froze for good seconds after init on rmw_zenoh (2026-10-04); an EXECUTOR=0 image ignores it.
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

RESPAWN = {"respawn": True, "respawn_delay": 2.0}
DEFAULT_CONFIG = "/maps/vio/estimator_config.yaml"
EXECUTORS = ("single", "multi")


def _describe(context):  # type: ignore[no-untyped-def]
    config = LaunchConfiguration("config").perform(context)
    if not os.path.isfile(config):
        return [
            LogInfo(
                msg=f"vio NOT started: {config} does not exist. ros/laptop.sh vio writes it (with"
                " ros/tools/vio_config.py, which needs config/camera.json's head_imu)"
            )
        ]
    shas = Path("/opt/openvins/SHAS")
    pinned = shas.read_text().strip().replace("\n", "; ") if shas.is_file() else "unknown"
    executor = LaunchConfiguration("executor").perform(context).strip()
    if executor not in EXECUTORS:
        raise ValueError(f"executor {executor!r}: one of {', '.join(EXECUTORS)}")
    node = Node(
        package="ov_msckf",
        executable="run_subscribe_msckf",
        namespace="ov_msckf",
        output="screen",
        # a terminal, so OpenVINS's printf lines are line-buffered: through a pipe they reached
        # docker logs in 4 kB bursts, minutes late at rest (2026-10-04)
        emulate_tty=True,
        parameters=[
            {
                "config_path": config,
                "verbosity": LaunchConfiguration("verbosity").perform(context),
                "use_stereo": True,
                "max_cameras": 2,
                "save_total_state": False,
                # OpenVINS broadcasts global -> imu -> cam0/cam1 on /tf at the IMU's rate by
                # default (600 transforms/s), and /tf reaches every listener in the graph, the
                # board's included: off, so nothing here owns a transform
                "publish_global_to_imu_tf": False,
                "publish_calibration_tf": False,
                "executor": executor,
                "multi_threading_subs": True,
            }
        ],
        **RESPAWN,
    )
    return [
        LogInfo(
            msg=f"vio up: OpenVINS ({pinned}, executor {executor}) on {config}: /camera/image +"
            " /camera/right/image +"
            " /head/imu -> /ov_msckf/poseimu for visual_odometry (vo_input vio)"
        ),
        node,
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=DEFAULT_CONFIG),
            DeclareLaunchArgument("verbosity", default_value="INFO"),
            DeclareLaunchArgument("executor", default_value="single"),
            OpaqueFunction(function=_describe),
        ]
    )
