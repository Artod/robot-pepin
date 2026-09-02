"""Pose graph: a square with drifted edges and one loop closure snaps back together."""

import math

import pytest

from pepin.odometry import Pose2D
from pepin.posegraph import Edge, PoseGraph
from pepin.scanmatch import apply_motion, relative_motion

SIDE = Pose2D(2.0, 0.0, math.pi / 2)  # drive 2 m, turn left 90 deg: four of these close a square
ODOMETRY_INFO = (2500.0, 2500.0, 365.0)  # 2 cm, 2 cm, 3 deg: chain edges are the less trusted ones


def chain(motions: list[Pose2D]) -> PoseGraph:
    """Graph whose node guesses are integrated from the given motions, with odometry-grade edges."""
    graph = PoseGraph()
    graph.add_node(Pose2D())
    for k, motion in enumerate(motions):
        graph.add_node(apply_motion(graph.nodes[k], motion))
        graph.add_edge(Edge(k, k + 1, motion, ODOMETRY_INFO))
    return graph


def test_consistent_chain_is_already_optimal() -> None:
    graph = chain([SIDE] * 4)
    before = [*graph.nodes]
    assert graph.optimize() == pytest.approx(0.0, abs=1e-12)
    assert all(
        n.x == pytest.approx(o.x) and n.theta == pytest.approx(o.theta)
        for n, o in zip(graph.nodes, before, strict=True)
    )


def test_loop_closure_removes_accumulated_drift() -> None:
    drifted = Pose2D(2.0, 0.0, math.pi / 2 + math.radians(3.0))  # each turn over-counted by 3 deg
    graph = chain([drifted] * 4)
    end_before = graph.nodes[4]
    assert math.hypot(end_before.x, end_before.y) > 0.2  # the square does not close on its own
    # Loop closure: the matcher recognises the start and measures the true relative pose.
    # Its default 1 deg weight outranks the 3 deg chain edges, so it absorbs almost no residual:
    # least squares splits the 12 deg discrepancy in inverse proportion to the weights.
    graph.add_edge(Edge(4, 0, relative_motion(Pose2D(0, 0, 0), Pose2D(0, 0, 0))))
    graph.optimize()
    end = graph.nodes[4]
    assert math.hypot(end.x, end.y) < 0.02
    assert abs(end.theta) < math.radians(0.5)


def test_anchor_never_moves() -> None:
    graph = chain([Pose2D(1.0, 0.0, 0.3)] * 3)
    graph.add_edge(Edge(3, 0, Pose2D(-2.5, 0.2, -0.9)))
    graph.optimize()
    assert graph.nodes[0] == Pose2D()
