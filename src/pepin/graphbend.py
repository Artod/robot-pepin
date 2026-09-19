"""How far RTAB-Map's optimised graph has moved THE ROOM — which is not the same question as how
far it has moved the cart, and the difference is the whole of this module.

WHAT MAKES OLD PAINT STALE. The fused volume (:mod:`pepin.worldmap`) is written in the ``map``
frame at the poses the BOARD's tracker publishes, and nothing localises against it: it is painted
open-loop. When RTAB-Map optimises its graph, every node's pose in ``map`` moves, so the room's own
expression in that frame moves — and every voxel painted before the optimisation is stale by
exactly that move. That is what the volume must follow.

WHY ``map_to_odom`` ALONE CANNOT SAY IT, although the message carries it. A correction between
``map`` and ``odom`` changes for two reasons that look identical in the transform and are opposite
in meaning:

* THE ROOM MOVED — a closure landed and the graph bent. Old paint is stale; the volume must follow.
* THE CART WAS FOUND — the same room, recognised after some drift, so the correction jumps by the
  drift. Old paint is exactly where it belongs; a volume that followed THAT would be dragged off
  the room by the size of the recovery, which after a carry is metres.

The node poses tell the two apart, and nothing else in the message does: a re-localisation of the
cart moves ``map_to_odom`` and leaves every node's pose where it was, while an optimisation moves
the poses. So the signal here is the CHANGE IN THE OPTIMISED POSES between two
``/rtabmap/mapGraph`` messages (``poses_id`` beside ``poses``,
rtabmap_msgs/msg/MapGraph.msg:11-12 — the optimised graph, MapData.msg:5), accumulated into one
transform a caller can hand to :class:`pepin.worldmap.CorrectionFollower` exactly as it used to
hand it ``map -> odom``.

WHICH NODE'S MOVE, since a closure bends a graph and does not translate it. A rigid shift of a whole
volume can only ever be right in one neighbourhood, and the one that matters is where the cart is
painting: so the increment is read at the NEWEST node the two messages share. That is where the cart
has just been — RTAB-Map adds a node every ``RGBD/LinearUpdate`` of travel — and it is the same
neighbourhood ``map_to_odom`` itself describes, so this is the old behaviour's intent with the
cart's own recoveries taken out of it. Far corners of the volume are carried by the same rigid move
and are wrong by however much the graph bent between there and here; the sensors repaint what they
see within a second of driving, and a map left behind the graph never comes back
(:mod:`pepin_bringup.depth_fusion`'s ``follow_correction``).

BESIDE A LOADED DATABASE NOTHING SHOULD MOVE AT ALL, and that is a prediction this makes rather
than a special case it carries: with ``Mem/IncrementalMemory`` false no node and no link is written,
so there is no new constraint to optimise and every node's pose comes back identical. The volume
then never follows, because there is nothing to follow — and the moment the memory rule lets
RTAB-Map learn again (:class:`pepin.graphmode.ModeRule`), closures start landing and the moves
begin. One mechanism, no mode.

Nothing here is ROS: a table of poses in, one accumulated transform out.
"""

from __future__ import annotations

import numpy as np

from pepin.tsdf import PlanarShift, RigidPose

__all__ = ["GraphBend", "identity_pose"]


def identity_pose() -> RigidPose:
    """A :class:`pepin.tsdf.RigidPose` that moves nothing: what an unbent graph's accumulated
    drift is, and what :class:`GraphBend` starts from."""
    return RigidPose(np.eye(3), np.zeros(3))


class GraphBend:
    """The accumulated move of the ROOM across ``/rtabmap/mapGraph`` messages.

    Fed one message's optimised node poses at a time (:meth:`observe`), it answers with the
    increment the room moved by, and keeps the running total in :attr:`drift` — a transform of the
    same kind a ``map -> odom`` correction is, so :class:`pepin.worldmap.CorrectionFollower` takes
    it unchanged and its thresholds, its rate and its resampling law all mean what they meant.
    """

    def __init__(self) -> None:
        self._poses: dict[int, RigidPose] = {}
        self.drift = identity_pose()  # everything the room has moved by since this run began
        self.bends = 0  # ...over this many increments
        self.node = 0  # the node the last increment was read at
        self.last = PlanarShift(0.0, 0.0, 0.0)  # ...and what it was
        self.nodes = 0  # how many nodes the last message carried

    def observe(self, poses: dict[int, RigidPose]) -> PlanarShift | None:
        """One ``/rtabmap/mapGraph``'s optimised poses in; the move the room made since the last
        one, or ``None`` when it made none.

        ``None`` covers three cases, and all three mean "nothing is owed": the first message ever
        (there is no previous graph to compare with), a graph that shares no node with the previous
        one (a database was swapped under us, and an increment between two different rooms is
        nonsense), and a graph whose newest shared node has not moved — which is every message of a
        session that is only localising.
        """
        self.nodes = len(poses)
        shared = sorted(set(poses) & set(self._poses))
        previous, self._poses = self._poses, dict(poses)
        if not shared:
            return None
        node = shared[-1]  # the newest node both graphs hold: where the cart has just been
        increment = PlanarShift.between(previous[node], poses[node])
        if increment.nothing:
            return None
        self.node = node
        self.last = increment
        self.bends += 1
        self.drift = increment.applied_to(self.drift)
        return increment

    def text(self) -> str:
        """The room's own movement for a report line: ``3 bends, last +1.2, -0.4 cm, +0.31 deg at
        node 88429 of 1204``."""
        if not self.bends:
            return f"no bend yet ({self.nodes} nodes)"
        return f"{self.bends} bends, last {self.last.text()} at node {self.node} of {self.nodes}"
