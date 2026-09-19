"""The room's own movement, told apart from the cart's being found (:mod:`pepin.graphbend`).

The whole value of this module is that a re-localisation of the CART does not drag the painted
volume, so every test here is written as the pair of situations the old signal (``map -> odom``)
could not distinguish.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pepin.graphbend import GraphBend, identity_pose
from pepin.tsdf import PlanarShift, RigidPose
from pepin.worldmap import CorrectionFollower


def _pose(x: float, y: float, yaw_deg: float = 0.0) -> RigidPose:
    """A node's optimised pose in the map."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return RigidPose(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.array([x, y, 0.0]))


def test_the_first_graph_owes_nothing() -> None:
    """There is no previous graph to compare with, so nothing is stale and nothing moves."""
    bend = GraphBend()
    assert bend.observe({1: _pose(0.0, 0.0), 2: _pose(1.0, 0.0)}) is None
    assert bend.bends == 0 and bend.nodes == 2
    assert PlanarShift.between(identity_pose(), bend.drift).nothing


def test_a_graph_that_did_not_move_owes_nothing() -> None:
    """Every message of a session that is only LOCALISING: nothing is written, so nothing is
    optimised and every node comes back where it was."""
    bend = GraphBend()
    poses = {1: _pose(0.0, 0.0), 2: _pose(1.0, 0.0)}
    bend.observe(poses)
    for _ in range(5):
        assert bend.observe(dict(poses)) is None
    assert bend.bends == 0
    assert bend.text() == "no bend yet (2 nodes)"


def test_a_bend_of_the_newest_shared_node_is_what_the_volume_owes() -> None:
    """A closure landed and the graph moved the room: the increment is read at the newest node
    both graphs hold, which is where the cart has just been painting."""
    bend = GraphBend()
    bend.observe({1: _pose(0.0, 0.0), 7: _pose(2.0, 0.0)})
    increment = bend.observe({1: _pose(0.0, 0.0), 7: _pose(2.0, 0.30)})
    assert increment is not None
    assert increment.dx == pytest.approx(0.0) and increment.dy == pytest.approx(0.30)
    assert bend.node == 7, "the newest shared node, not the oldest"
    assert bend.bends == 1


def test_the_increments_accumulate_so_small_bends_move_the_volume_together() -> None:
    """The follower measures against ONE anchor, so a run of bends under its threshold must add
    up in the drift and move the volume when they are worth it."""
    bend = GraphBend()
    bend.observe({5: _pose(0.0, 0.0)})
    for step in range(1, 5):
        bend.observe({5: _pose(0.0, 0.02 * step)})
    assert bend.bends == 4
    total = PlanarShift.between(identity_pose(), bend.drift)
    assert total.dy == pytest.approx(0.08), "four 2 cm bends are 8 cm of stale paint"


def test_a_graph_sharing_no_node_is_another_room_and_owes_nothing() -> None:
    """A database swapped under the node: an increment between two different rooms is nonsense,
    and the new graph simply becomes the reference."""
    bend = GraphBend()
    bend.observe({1: _pose(0.0, 0.0)})
    assert bend.observe({900: _pose(9.0, 9.0)}) is None
    assert bend.bends == 0
    assert bend.observe({900: _pose(9.0, 9.2)}) is not None, "and the new room is followed"


def test_the_drift_drives_the_follower_the_way_map_to_odom_used_to() -> None:
    """The point of accumulating into a pose rather than into a shift: the existing
    :class:`pepin.worldmap.CorrectionFollower` takes it unchanged, with its thresholds, its anchor
    and its 'owed until it is worth it' behaviour all meaning what they meant."""
    bend, follower = GraphBend(), CorrectionFollower()
    bend.observe({3: _pose(0.0, 0.0)})
    follower.anchor(bend.drift)
    bend.observe({3: _pose(0.0, 0.02)})
    assert follower.pending(bend.drift, 0.05, 1.0) is None, "2 cm is under one voxel: owed"
    bend.observe({3: _pose(0.0, 0.08)})
    shift = follower.pending(bend.drift, 0.05, 1.0)
    assert shift is not None and shift.dy == pytest.approx(0.08), "both bends, against one anchor"
    follower.moved(bend.drift, shift)
    assert follower.pending(bend.drift, 0.05, 1.0) is None, "and the debt is paid"


def test_a_turn_of_the_graph_is_carried_as_a_turn_about_the_maps_origin() -> None:
    """A closure that rotates the room: the turn is the same wherever it is read, but a
    :class:`pepin.tsdf.PlanarShift` pivots about the MAP's origin, so a node that turned in place
    away from that origin also carries the translation that keeps it where the graph puts it.

    That is the move the volume needs, not a bookkeeping quirk: every voxel of the room turns with
    the graph, and this is the one rigid transform that turns them all and leaves the node itself
    where the new graph says it is."""
    bend = GraphBend()
    bend.observe({2: _pose(1.0, 0.0, 0.0)})
    increment = bend.observe({2: _pose(1.0, 0.0, 5.0)})
    assert increment is not None
    assert increment.yaw_deg == pytest.approx(5.0)
    moved_x, moved_y = increment.moved(1.0, 0.0)
    assert (moved_x, moved_y) == pytest.approx((1.0, 0.0)), "the node stays where the graph puts it"
    _far_x, far_y = increment.moved(5.0, 0.0)
    assert far_y == pytest.approx(4.0 * math.sin(math.radians(5.0))), (
        "4 m further out the same turn is 35 cm — which is why the threshold has an angular half"
    )
