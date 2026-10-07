"""A recorded drive's commands replayed through the gaze node (pepin_bringup.gaze) under the ROS
stubs, on a clock of its own: the base server's state lines carry each twist, a head that runs
to every neck_target at 300 deg/s answers through /neck/state, depth_fusion fuses a frame every
0.1 s. What comes out is every neck write the arbiter made, so two builds or two knob sets can
be compared write for write.

:func:`replay` drives a fixed plan with a list of twists (only names the gaze node has had since
2026-10-05 are used, so it runs on an older checkout: scratch/gaze_follow/baseline_capture.py);
:func:`replay_tape` replays a whole taped drive (:class:`Tape`): its twists, plans, poses, local
costmaps, stall looks and the controller's status (who drives the wheels), each at its own time.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import ros_stubs

ros_stubs.install()

from pepin_bringup import gaze as gaze_node  # noqa: E402
from ros_stubs import (  # noqa: E402
    GoalStatus,
    GoalStatusArray,
    Header,
    OccupancyGrid,
    Path_,
    PoseStamped,
    String,
    Time,
)

from pepin.gaze import NAVIGATION, Aim, Look  # noqa: E402
from pepin.tsdf import RigidPose  # noqa: E402

T0 = 1_000_000.0  # the replay's clock at the drive's start
TICK_S = 0.05  # the node's step period
HEAD_DEG_S = 300.0  # the neck's measured top speed (config/neck.json motion)
FRAME_EVERY = 2  # a fused frame every second tick: 10 Hz
NAV = "navigate_to_pose"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

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
    source: str = ""  # the request that made it (replay_tape only)

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
    route: tuple[tuple[float, float], ...] = TURN_LEFT,
) -> list[Write]:
    """Drive the node through one drive of these 10 Hz commands with path and reverse gaze on,
    ``knobs`` set live, along a fixed plan through ``route`` (the cart stays at its start,
    facing +x), then ``after_s`` at rest after the drive's end; every write it made."""
    clock = [T0]
    node._now = lambda: clock[0]
    for name, value in {"path_gaze": True, "reverse_gaze": True, **(knobs or {})}.items():
        node._switches.set(name, value)
    home_tilt = math.degrees(node._arbiter.home.tilt_rad)
    head = [0.0, home_tilt]
    sent = node._link.sent
    writes: list[Write] = []
    node._on_plan(plan(route))
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


@dataclass(frozen=True)
class Tape:
    """A drive's inputs to the gaze node, from its tape (tests/fixtures/drive_<run>_gaze.json):
    seconds from the goal's acceptance; ``writes`` are the real head's, for comparison."""

    end_s: float
    first_plan: list[list[float]]
    cmd: list[list[float]]
    plan: list[list[Any]]
    map_pose: list[list[float]]
    odom_pose: list[list[float]]
    costmap: list[list[Any]]
    stall: list[list[float]]
    writes: list[list[Any]]
    follow: list[list[float]] = field(default_factory=list)  # [t, 1 while FollowPath runs]
    controller: list[list[Any]] = field(default_factory=list)  # [t, the selector's controller]

    @classmethod
    def load(cls, run: str, folder: Path = FIXTURES) -> Tape:
        """``tests/fixtures/drive_<run>_gaze.json`` (or ``folder``'s)."""
        data = json.loads((folder / f"drive_{run}_gaze.json").read_text())
        return cls(**{k: v for k, v in data.items() if not k.startswith("_")})


class TapeTf:
    """TF as the tape had it at the replay's clock: map and odom -> base_link, the newest row at
    or before that time (the first one before it)."""

    def __init__(self, tape: Tape, clock: list[float]) -> None:
        self._rows = {"map": np.array(tape.map_pose), "odom": np.array(tape.odom_pose)}
        self._clock = clock

    def pose(
        self, target: str, source: str, stamp: Any = None, timeout_s: float = 0.0
    ) -> RigidPose | None:
        """``target <- source`` now; None for any pair but map or odom <- base_link."""
        rows = self._rows.get(target)
        if rows is None or source != "base_link":
            return None
        i = max(int(np.searchsorted(rows[:, 0], self._clock[0] - T0, side="right")) - 1, 0)
        _t, x, y, yaw = rows[i]
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return RigidPose(rotation, np.array([x, y, 0.0]))

    def close(self) -> None:
        """Nothing to close."""


def grid(origin_x: float, origin_y: float, runs: list[int]) -> Any:
    """A taped local costmap: 60 x 60 cells of 0.05 m in odom, lethal along ``runs`` (start,
    length pairs of the flat index), free elsewhere."""
    msg = OccupancyGrid()
    msg.header.frame_id = "odom"
    msg.info.resolution = 0.05
    msg.info.width = msg.info.height = 60
    msg.info.origin.position.x, msg.info.origin.position.y = origin_x, origin_y
    data = np.zeros(60 * 60, dtype=int)
    for start, length in zip(runs[::2], runs[1::2], strict=True):
        data[start : start + length] = 100
    msg.data = list(data)
    return msg


def replay_tape(
    node: Any,
    tape: Tape,
    *,
    knobs: dict[str, Any] | None = None,
    after_s: float = 3.0,
    frame_every: int = FRAME_EVERY,
    head_deg_s: float = HEAD_DEG_S,
    head: Head | None = None,
    on_tick: Callable[[float, tuple[float, float], float, float], None] | None = None,
) -> list[Write]:
    """Drive the node through a taped drive with path gaze, reverse gaze and the stall look on,
    ``knobs`` set live: every input at its own time (a stall look submitted as the node's
    ``_stall`` submits it, without the columns; the controller the selector names), a frame
    every ``frame_every`` ticks, a head of ``head_deg_s`` (or ``head``'s profile), the drive's
    end at ``end_s``, then ``after_s`` at rest; every write it made, with the source that made
    it. ``on_tick``: seconds from the start, the head (pan, tilt deg) and the command (v, w)
    after every step."""
    clock = [T0]
    node._now = lambda: clock[0]
    node._tf = TapeTf(tape, clock)
    flags = {"path_gaze": True, "reverse_gaze": True, "stall_look": True}
    for name, value in {**flags, **(knobs or {})}.items():
        node._switches.set(name, value)
    at = [0.0, math.degrees(node._arbiter.home.tilt_rad)]
    sent = node._link.sent
    writes: list[Write] = []
    events: list[tuple[float, str, Any]] = sorted(
        [(row[0], "a plan", row[1]) for row in tape.plan]
        + [(row[0], "b grid", row) for row in tape.costmap]
        + [(row[0], "c stall", row) for row in tape.stall]
        + [(row[0], "a follow", row[1]) for row in tape.follow]
        + [(row[0], "a controller", row[1]) for row in tape.controller]
        + [(tape.end_s, "d end", None)],
        key=lambda event: (event[0], event[1]),
    )
    cmd = np.array(tape.cmd)
    node._on_plan(plan(tuple((x, y) for x, y in tape.first_plan)))
    node._link.on_message({"type": "state", "moving": False, "v": 0.0, "w": 0.0, **_ticks()})
    node._on_nav_status(NAV, goals(9))
    driving = True
    for k in range(round((tape.end_s + after_s) / TICK_S)):
        t = clock[0] - T0
        while events and events[0][0] <= t:
            _at, what, row = events.pop(0)
            if what == "a plan":
                node._on_plan(plan(tuple((x, y) for x, y in row)))
            elif what == "a follow":
                if hasattr(node, "_on_driver_status"):  # the node since 2026-10-06
                    node._on_driver_status("follow_path", goals(1) if row else goals())
            elif what == "a controller":
                if hasattr(node, "_on_controller"):  # the node since 2026-10-07
                    node._on_controller(String(data=row))
            elif what == "b grid":
                node._on_costmap(grid(row[1], row[2], row[3]))
            elif what == "c stall":
                node._arbiter.submit(_stall_look(node, Aim(row[1], row[2])), clock[0])
            else:
                node._on_nav_status(NAV, goals())  # the drive's end
                driving = False
        i = int(np.searchsorted(cmd[:, 0], t, side="right")) - 1
        v, w = (float(cmd[i, 1]), float(cmd[i, 2])) if driving and i >= 0 else (0.0, 0.0)
        node._link.on_message({"type": "state", "moving": driving, "v": v, "w": w, **_ticks()})
        before = node._arbiter.counts["writes"]
        node._step()
        if node._arbiter.counts["writes"] > before:
            last = sent[-1]
            writes.append(
                Write(
                    round(t, 2),
                    round(math.degrees(last["pan_rad"]), 1),
                    round(math.degrees(last["tilt_rad"]), 1),
                    last.get("speed_deg_s"),
                    node._arbiter.state(clock[0]).source,
                )
            )
        if head is None:
            _follow(at, sent, head_deg_s)
        else:
            head.follow(at, sent)
        if on_tick is not None:
            on_tick(t, (at[0], at[1]), v, w)
        clock[0] += TICK_S
        joints = gaze_node.JointState()
        joints.header.stamp = stamp(clock[0])
        joints.name = ["neck_pan", "head_tilt"]
        joints.position = [math.radians(at[0]), math.radians(at[1])]
        node._on_neck(joints)
        if k % frame_every == 0:
            node._on_frame(Header(stamp=stamp(clock[0])))
    return writes


def _stall_look(node: Any, aim: Aim) -> Look:
    """The stall look as the node's ``_stall`` asks it, at a taped aim."""
    ttl = float(node._knob("ttl_navigation_s"))
    return Look(
        gaze_node.STALL_SOURCE,
        (aim,),
        NAVIGATION,
        int(node._switches["frames"]),
        0.0,
        ttl,
        kind="point",
        hold_s=ttl if node._knob("glance_dwell_s") > 0.0 else 0.0,
    )


class Head:
    """A neck that runs each axis on the board's profile (config/neck.json's motion block): a
    trapezoid of ``top_deg_s`` (the servos' measured peak) and the axis's ramp, from rest to
    rest, toward the newest neck_target (at its own ``speed_deg_s`` when it asks for one)."""

    def __init__(
        self, top_deg_s: float = 289.0, pan_acc: float = 2232.0, tilt_acc: float = 600.0
    ) -> None:
        self.top = top_deg_s
        self.acc = (pan_acc, tilt_acc)
        self.speed = [0.0, 0.0]

    def follow(self, at: list[float], sent: list[dict[str, Any]]) -> None:
        """One tick of both axes toward the newest neck_target."""
        targets = [m for m in sent if m.get("cmd") == "neck_target"]
        if not targets:
            return
        goal = (math.degrees(targets[-1]["pan_rad"]), math.degrees(targets[-1]["tilt_rad"]))
        top = targets[-1].get("speed_deg_s") or self.top
        for axis in (0, 1):
            gap = goal[axis] - at[axis]
            if abs(gap) < 1e-9:
                self.speed[axis] = 0.0
                continue
            acc = self.acc[axis]
            cap = min(top, math.sqrt(2.0 * acc * abs(gap)))  # still stops at the goal from here
            speed = min(cap, self.speed[axis] + acc * TICK_S)
            step = min(abs(gap), 0.5 * (self.speed[axis] + speed) * TICK_S)
            at[axis] += math.copysign(step, gap)
            self.speed[axis] = speed if step < abs(gap) else 0.0


def _follow(head: list[float], sent: list[dict[str, Any]], top_deg_s: float) -> None:
    """One tick of the head toward the newest neck_target, at its speed or ``top_deg_s``."""
    targets = [m for m in sent if m.get("cmd") == "neck_target"]
    if not targets:
        return
    goal = (math.degrees(targets[-1]["pan_rad"]), math.degrees(targets[-1]["tilt_rad"]))
    step = (targets[-1].get("speed_deg_s") or top_deg_s) * TICK_S
    for axis in (0, 1):
        head[axis] += max(-step, min(step, goal[axis] - head[axis]))
