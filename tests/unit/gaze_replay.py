"""A recorded drive's commands replayed through the gaze node (pepin_bringup.gaze) under the ROS
stubs, on a clock of its own: the base server's state lines carry each twist, a head that runs
to every neck_target at 300 deg/s answers through /neck/state, depth_fusion fuses a frame every
0.1 s. What comes out is every neck write the arbiter made, so two builds or two knob sets can
be compared write for write.

Only names the gaze node has had since 2026-10-05 are used here, so the same replay runs on an
older checkout (scratch/gaze_follow/baseline_capture.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import ros_stubs

ros_stubs.install()

from pepin_bringup import gaze as gaze_node  # noqa: E402
from ros_stubs import GoalStatus, GoalStatusArray, Header, Path_, PoseStamped, Time  # noqa: E402

T0 = 1_000_000.0  # the replay's clock at the drive's start
TICK_S = 0.05  # the node's step period
HEAD_DEG_S = 300.0  # the neck's measured top speed (config/neck.json motion)
FRAME_EVERY = 2  # a fused frame every second tick: 10 Hz
NAV = "navigate_to_pose"

# Drive 306 (2026-10-05 19:36:35.1-41.7Z, the tape's cmd rows at 10 Hz, t+4.5 to t+11.1 s): the
# end of the reverse out of home, forward 0.9 s, a 0.6 s reverse, forward 0.3 s, a 1.1 s reverse,
# forward. With path gaze muted while reversing, the path look lapsed home twice here.
# fmt: off
DRIVE_306_V = (
    -0.06, -0.11, -0.12, -0.12, -0.12, -0.12, -0.13, -0.13, -0.14, -0.14, -0.12, -0.12, -0.12,
    -0.12, -0.12, -0.12, -0.12, -0.09, -0.1, -0.1, -0.1, -0.1, -0.08, 0.07, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1, -0.05, -0.1, -0.1, -0.1, -0.1, -0.1, 0.05, 0.1, 0.1, -0.05, -0.1, -0.1,
    -0.1, -0.1, -0.09, -0.08, -0.07, -0.06, -0.06, -0.06, 0.09, 0.1, 0.11, 0.11, 0.12, 0.12, 0.12,
    0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12,
)
DRIVE_306_W = (
    0.09, 0.29, 0.49, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.3, 0.1, -0.1, 0.1, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.1, -0.1,
    -0.3, -0.5, -0.3, -0.1, 0.1, -0.1, -0.3, -0.1, 0.1, 0.3, 0.5, 0.5, 0.5, 0.5, 0.58, 0.64, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
)
# fmt: on
# A plan that turns left 0.3 m ahead: path gaze's point 0.6 m along it is 45 deg left and near,
# so the path look is never home and a lapse shows as a write home.
TURN_LEFT = ((0.0, 0.0), (0.3, 0.0), (0.3, 1.5))


@dataclass(frozen=True)
class Write:
    """One neck write: seconds since the drive's start, where to, and at what speed."""

    t: float
    pan_deg: float
    tilt_deg: float
    speed_deg_s: float | None

    def home(self, home_tilt_deg: float) -> bool:
        """Whether it sends the head home."""
        return abs(self.pan_deg) < 0.05 and abs(self.tilt_deg - home_tilt_deg) < 0.05


def plan(points: tuple[tuple[float, float], ...] = TURN_LEFT) -> Any:
    """A /plan message through ``points`` in map."""
    msg = Path_()
    msg.header.frame_id = "map"
    for x, y in points:
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y = x, y
        msg.poses.append(pose)
    return msg


def goals(*uuids: int) -> Any:
    """A navigator's status list with these goals executing."""
    msg = GoalStatusArray()
    for uuid in uuids:
        status = GoalStatus(status=2)
        status.goal_info.goal_id.uuid = [uuid] * 16
        msg.status_list.append(status)
    return msg


def stamp(t: float) -> Any:
    """A ROS time of ``t`` seconds."""
    whole = math.floor(t)
    return Time(sec=whole, nanosec=round((t - whole) * 1e9))


def replay(
    node: Any,
    v: tuple[float, ...] = DRIVE_306_V,
    w: tuple[float, ...] = DRIVE_306_W,
    *,
    knobs: dict[str, Any] | None = None,
    after_s: float = 3.0,
) -> list[Write]:
    """Drive the node through one drive of these 10 Hz commands with path and reverse gaze on,
    ``knobs`` set live, then ``after_s`` at rest after the drive's end; every write it made."""
    clock = [T0]
    node._now = lambda: clock[0]
    for name, value in {"path_gaze": True, "reverse_gaze": True, **(knobs or {})}.items():
        node._switches.set(name, value)
    home_tilt = math.degrees(node._arbiter.home.tilt_rad)
    head = [0.0, home_tilt]
    sent = node._link.sent
    writes: list[Write] = []
    node._on_plan(plan())
    node._link.on_message({"type": "state", "moving": False, "v": 0.0, "w": 0.0, **_ticks()})
    node._on_nav_status(NAV, goals(9))
    ticks = 2 * len(v) + round(after_s / TICK_S)
    for k in range(ticks):
        if k == 2 * len(v):
            node._on_nav_status(NAV, goals())  # the drive's end
        twist = (v[k // 2], w[k // 2]) if k < 2 * len(v) else (0.0, 0.0)
        node._link.on_message(
            {"type": "state", "moving": k < 2 * len(v), "v": twist[0], "w": twist[1], **_ticks()}
        )
        before = node._arbiter.counts["writes"]
        node._step()
        if node._arbiter.counts["writes"] > before:
            last = sent[-1]
            writes.append(
                Write(
                    round(clock[0] - T0, 2),
                    round(math.degrees(last["pan_rad"]), 1),
                    round(math.degrees(last["tilt_rad"]), 1),
                    last.get("speed_deg_s"),
                )
            )
        targets = [m for m in sent if m.get("cmd") == "neck_target"]
        if targets:
            goal = (math.degrees(targets[-1]["pan_rad"]), math.degrees(targets[-1]["tilt_rad"]))
            speed = targets[-1].get("speed_deg_s") or HEAD_DEG_S
            for axis in (0, 1):
                step = speed * TICK_S
                head[axis] += max(-step, min(step, goal[axis] - head[axis]))
        clock[0] += TICK_S
        joints = gaze_node.JointState()
        joints.header.stamp = stamp(clock[0])
        joints.name = ["neck_pan", "head_tilt"]
        joints.position = [math.radians(head[0]), math.radians(head[1])]
        node._on_neck(joints)
        if k % FRAME_EVERY == 0:
            node._on_frame(Header(stamp=stamp(clock[0])))
    return writes


def _ticks() -> dict[str, int]:
    """A state line's neck encoders: their presence picks the neck_target driver."""
    return {"pan_ticks": 2048, "tilt_ticks": 2311}
