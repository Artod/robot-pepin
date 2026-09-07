"""map -> odom composition of the relocalizer's tracker, read out of its module without rclpy."""

import ast
import math
from pathlib import Path

from pepin.odometry import Pose2D

SOURCE = Path(__file__).resolve().parents[2] / "ros/pepin_bringup/pepin_bringup/relocalizer.py"


def _load():  # type: ignore[no-untyped-def]
    """Compile only the pure function: the module imports rclpy at import time."""
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "map_to_odom")
    namespace = {"math": math, "Pose2D": Pose2D}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["map_to_odom"]


def _apply(t, p):  # type: ignore[no-untyped-def]
    x, y, yaw = t
    c, s = math.cos(yaw), math.sin(yaw)
    return Pose2D(x + c * p.x - s * p.y, y + s * p.x + c * p.y, yaw + p.theta)


def test_identity_when_the_frames_coincide() -> None:
    x, y, yaw = _load()(Pose2D(1.0, 2.0, 0.3), Pose2D(1.0, 2.0, 0.3))
    assert abs(x) < 1e-9 and abs(y) < 1e-9 and abs(yaw) < 1e-9


def test_the_transform_maps_the_odom_pose_onto_the_map_pose() -> None:
    map_pose, odom_pose = Pose2D(-2.0, 1.5, 2.0), Pose2D(0.7, -0.4, -1.1)
    back = _apply(_load()(map_pose, odom_pose), odom_pose)
    assert abs(back.x - map_pose.x) < 1e-9 and abs(back.y - map_pose.y) < 1e-9
    assert (
        abs(
            math.atan2(math.sin(back.theta - map_pose.theta), math.cos(back.theta - map_pose.theta))
        )
        < 1e-9
    )
