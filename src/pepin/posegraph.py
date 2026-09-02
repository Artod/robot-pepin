"""Pose-graph optimisation on SE(2): make a chain of relative measurements globally consistent.

Nodes are keyframe poses, edges are measured relative motions between two
nodes (from odometry, scan matching, or a loop closure) with a weight
(information) per axis. Optimisation moves every node except the anchor
(node 0, pinned to the origin) to minimise the weighted squared mismatch
between what each edge measured and what the current poses imply. With
only chain edges nothing changes; one loop-closure edge redistributes the
accumulated drift along the whole chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import relative_motion


@dataclass(frozen=True)
class Edge:
    """Measured motion from node ``i`` to node ``j`` (in node i's frame) and its per-axis weight."""

    i: int
    j: int
    measured: Pose2D
    information: tuple[float, float, float] = (
        2500.0,
        2500.0,
        3283.0,
    )  # 1/sigma^2: 2 cm, 2 cm, 1 deg


@dataclass
class PoseGraph:
    """Keyframe poses plus the measurements that tie them together."""

    nodes: list[Pose2D] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, pose: Pose2D) -> int:
        """Append a node with its initial guess; returns its index."""
        self.nodes.append(pose)
        return len(self.nodes) - 1

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def residual(self, edge: Edge) -> NDArray[np.float64]:
        """Mismatch (dx, dy, dtheta) between the measured motion and the one the poses imply."""
        predicted = relative_motion(self.nodes[edge.i], self.nodes[edge.j])
        m = edge.measured
        return np.array(
            [m.x - predicted.x, m.y - predicted.y, wrap_angle(m.theta - predicted.theta)]
        )

    def total_error(self) -> float:
        """Sum of weighted squared residuals over all edges; the quantity optimisation minimises."""
        return float(sum(float(np.dot(self.residual(e) ** 2, e.information)) for e in self.edges))

    def optimize(self, iterations: int = 10, tolerance: float = 1e-9) -> float:
        """Gauss-Newton with numerical Jacobians, node 0 pinned; returns the final total error."""
        n = len(self.nodes)
        if n < 2 or not self.edges:
            return self.total_error()
        for _ in range(iterations):
            h = np.zeros((3 * n, 3 * n))
            b = np.zeros(3 * n)
            for edge in self.edges:
                r = self.residual(edge)
                jac = self._jacobian(edge)  # (3, 6): d residual / d (node i, node j)
                w = np.diag(edge.information)
                idx = [
                    3 * edge.i,
                    3 * edge.i + 1,
                    3 * edge.i + 2,
                    3 * edge.j,
                    3 * edge.j + 1,
                    3 * edge.j + 2,
                ]
                h[np.ix_(idx, idx)] += jac.T @ w @ jac
                b[idx] += jac.T @ w @ r
            # Pin the anchor: solve only for nodes 1..n-1.
            free = slice(3, 3 * n)
            delta = np.linalg.solve(h[free, free] + 1e-9 * np.eye(3 * n - 3), -b[free])
            for k in range(1, n):
                p = self.nodes[k]
                d = delta[3 * (k - 1) : 3 * k]
                self.nodes[k] = Pose2D(p.x + d[0], p.y + d[1], wrap_angle(p.theta + d[2]))
            if float(np.abs(delta).max()) < tolerance:
                break
        return self.total_error()

    def _jacobian(self, edge: Edge, eps: float = 1e-6) -> NDArray[np.float64]:
        """Numerical derivative of the residual w.r.t. the six coordinates of nodes i and j."""
        jac = np.zeros((3, 6))
        base = self.residual(edge)
        for col in range(6):
            node = edge.i if col < 3 else edge.j
            axis = col % 3
            saved = self.nodes[node]
            bumped = [saved.x, saved.y, saved.theta]
            bumped[axis] += eps
            self.nodes[node] = Pose2D(*bumped)
            jac[:, col] = (self.residual(edge) - base) / eps
            self.nodes[node] = saved
        return jac


def yaw_error_deg(a: Pose2D, b: Pose2D) -> float:
    """Heading difference in degrees, wrapped; handy for tests and reports."""
    return math.degrees(wrap_angle(a.theta - b.theta))
