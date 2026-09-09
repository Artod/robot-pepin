"""Which part of the stack runs where: the board keeps the reflexes, the laptop takes the rest.

The board is four Cortex-A53 cores. Whatever closes a control loop or owns a frame stays on
it: the sensors, the EKF, the tracker (map->odom), the controller with its local costmap, the
behaviours and the tree that orders them. Whatever answers once a second and tolerates a
wireless hop moves to the laptop: the planner with the global costmap, the goal server with its
recorder. The split is data, so a test can hold it and the launch file merely reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SIDES = ("all", "board", "laptop")

# Nav2 lifecycle nodes by side. "all" is the union: one machine, as before the split.
BOARD_NAV_NODES = ("controller_server", "behavior_server", "bt_navigator", "velocity_smoother")
LAPTOP_NAV_NODES = ("planner_server",)
MAP_NODES = ("map_server",)  # the map is served from the board: the tracker needs it there

HEARTBEAT_TOPIC = "laptop/heartbeat"
HEARTBEAT_HZ = 2.0

# The base's speed caps (config/base.json, the base server's own clamp). The C++ bridge on the
# board clamps /cmd_vel too, at 0.25 m/s by default: for half a day every tape sat at 0.20 and
# the bridge would have cut anything faster — one cap, the base's, passed to it at launch.
BASE_MAX_LINEAR_M_S = 0.30
BASE_MAX_ANGULAR_RAD_S = 1.0


def nav_nodes(side: str) -> tuple[str, ...]:
    """The Nav2 lifecycle nodes the navigation manager on ``side`` must bring up."""
    if side == "all":
        return BOARD_NAV_NODES + LAPTOP_NAV_NODES
    if side == "board":
        return BOARD_NAV_NODES
    if side == "laptop":
        return LAPTOP_NAV_NODES
    raise ValueError(f"side must be one of {SIDES}, not {side!r}")


def runs_here(side: str, node: str) -> bool:
    """Whether a named piece runs on ``side``: Nav2 nodes, the map, the tracker, the goal server."""
    if node in MAP_NODES or node == "relocalizer":
        return side in ("all", "board")
    if node == "goal_server":  # it carries the laptop's heartbeat too
        return side in ("all", "laptop")
    if node == "link_watch":
        return side == "board"  # only a split stack has a link to watch
    return node in nav_nodes(side)


@dataclass
class LinkWatch:
    """Cuts a drive when the laptop's heartbeat stops: a plan may never arrive, so stop now.

    With the planner on the laptop, a lost link means the tree on the board keeps following
    its last path with no one to replan around whatever appears. The controller and the local
    costmap still avoid what the lidar sees, so the cart is not blind — but it is deaf, and a
    deaf cart stops. ``patience_s`` covers a wireless hiccup; a link that stays silent longer
    is gone. The verdict is armed only while a goal is running, and fires once per outage.
    """

    patience_s: float = 2.5
    _last_beat: float | None = field(default=None, init=False)
    _cut: bool = field(default=False, init=False)

    def beat(self, now: float) -> None:
        """A heartbeat arrived."""
        self._last_beat = now
        self._cut = False

    def should_cut(self, navigating: bool, now: float) -> bool:
        """True while a running drive has had no heartbeat for the patience and the cut has not
        been sent yet. It does not consume itself: a node that could not reach the cancel
        service must be told again on the next tick, so ``cut_sent`` latches, not this."""
        if not navigating or self._cut:
            return False
        if self._last_beat is None:
            return False  # never heard the laptop: the stack is not split, nothing to watch
        return now - self._last_beat > self.patience_s

    def cut_sent(self) -> None:
        """The cancel went out: this outage is handled until the next heartbeat."""
        self._cut = True

    @property
    def alive(self) -> bool:
        return self._last_beat is not None and not self._cut
