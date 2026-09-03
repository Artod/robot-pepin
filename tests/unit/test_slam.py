"""Graph SLAM front end: chain of keyframes, loop detection and correction."""

import math

import pytest
from synthetic import raycast_room

from pepin.mapping import GridSpec
from pepin.odometry import Pose2D
from pepin.posegraph import Edge
from pepin.slam import GraphSlam, LoopClosure, LoopClosureConfig

SPEC = GridSpec(0.05, -4, -3, 8, 6)


def drive(slam: GraphSlam, poses: list[Pose2D], odom_poses: list[Pose2D] | None = None) -> None:
    for truth, odom in zip(poses, odom_poses or poses, strict=True):
        slam.process(odom, raycast_room(truth))


def square(n_per_side: int = 6, side: float = 1.6) -> list[Pose2D]:
    """Poses along a square path, heading along each side."""
    poses = []
    corners = [
        (-side / 2, -side / 2),
        (side / 2, -side / 2),
        (side / 2, side / 2),
        (-side / 2, side / 2),
    ]
    for c in range(4):
        (x0, y0), (x1, y1) = corners[c], corners[(c + 1) % 4]
        heading = math.atan2(y1 - y0, x1 - x0)
        for i in range(n_per_side):
            f = i / n_per_side
            poses.append(Pose2D(x0 + f * (x1 - x0), y0 + f * (y1 - y0), heading))
    return poses


def test_keyframes_follow_motion_and_skip_standing_still() -> None:
    slam = GraphSlam(SPEC)
    assert slam.process(Pose2D(), raycast_room(Pose2D())) is not None
    assert slam.process(Pose2D(0.005, 0, 0), raycast_room(Pose2D(0.005, 0, 0))) is None
    assert slam.process(Pose2D(0.1, 0, 0), raycast_room(Pose2D(0.1, 0, 0))) is not None
    assert len(slam.graph.edges) == 1


def test_loop_closure_detected_and_end_pose_corrected() -> None:
    poses = square()
    slam = GraphSlam(SPEC, loop=LoopClosureConfig(min_index_gap=16, min_path_m=3.0))
    drive(slam, poses)  # around the square with truthful odometry
    closures_before = len(slam.closures)  # the last side already revisits the start: legitimate
    # Come back to the start, but with odometry claiming a pose 0.25 m / 6 deg off.
    truth = poses[0]
    drifted = Pose2D(truth.x + 0.25, truth.y - 0.1, truth.theta + math.radians(6.0))
    slam = GraphSlam(
        SPEC, loop=LoopClosureConfig(min_index_gap=16, min_path_m=3.0, cooldown_keyframes=0)
    )
    drive(slam, poses)
    slam.process(drifted, raycast_room(truth))
    assert len(slam.closures) >= max(1, closures_before)
    assert slam.closures[-1].inlier_fraction > 0.6
    end = slam.pose
    assert end.x == pytest.approx(truth.x, abs=0.05)
    assert end.y == pytest.approx(truth.y, abs=0.05)
    assert abs(end.theta - truth.theta) < math.radians(1.5)


def test_no_closure_against_recent_keyframes() -> None:
    slam = GraphSlam(SPEC, loop=LoopClosureConfig(min_index_gap=100))
    drive(slam, square())
    slam.process(square()[0], raycast_room(square()[0]))
    assert not slam.closures


def test_no_closure_without_enough_path_between_the_keyframes() -> None:
    # A slow wiggle in place accumulates keyframes but no distance: never a revisit.
    slam = GraphSlam(SPEC, loop=LoopClosureConfig(min_index_gap=5, min_path_m=3.0))
    for i in range(20):
        p = Pose2D(0.0, 0.0, math.radians(3.0 * i))
        slam.process(p, raycast_room(p))
    assert not slam.closures and len(slam.keyframes) == 20


def test_inconsistent_closure_is_rolled_back() -> None:
    slam = GraphSlam(SPEC, loop=LoopClosureConfig(min_index_gap=100))
    drive(slam, square())
    nodes_before = list(slam.graph.nodes)
    edges_before = len(slam.graph.edges)
    # Claim the last keyframe sits 2 m away from the first — contradicts every chain edge.
    bogus = LoopClosure(Edge(len(nodes_before) - 1, 0, Pose2D(2.0, 0.0, 0.0)), inlier_fraction=0.9)
    assert not slam._accept_closure(bogus)
    assert len(slam.graph.edges) == edges_before
    assert slam.graph.nodes == nodes_before
