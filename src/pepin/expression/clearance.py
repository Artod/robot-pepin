"""Whether a small motion of the cart is clear: the footprint swept through a turn in place or a
straight leg, against the lidar's returns.

The cart turns about the middle of its drive axle (``base_link``'s origin), which sits at its
front: the rectangle of config/base.json's ``footprint`` reaches 0.0625 m ahead of it and 0.30 m
behind, 0.275 m to each side, so a turn in place swings the rear corners round a circle of
0.41 m while the nose barely moves (a turn to the left sweeps the rear to the right and the
front-right corner forward). A return BLOCKS a motion when, somewhere along it, the footprint
comes within ``margin_m`` of it AND closer than it stood at the motion's origin: a wall already
near the cart blocks a motion toward it, not one away from it (a back-off from what stopped a
turn), and going back to where the cart stood never counts. A return inside the footprint itself
is the cart (a post, a cable) and never blocks.

The returns come from the goal server beside Nav2 (``{"cmd": "clearance"}``,
:func:`near_points` on its newest scan through config/lidar.json's mount). A check is made in
the frame of the motion's ORIGIN (where the cart stood when it began); one made while the cart
moves is extended ahead by what the cart may have turned or driven since the scan.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

Points = NDArray[np.float64]  # (n, 2), metres in base_link
TURN_SAMPLE_RAD = math.radians(1.0)  # the sweep's poses: a corner at 0.41 m moves 7 mm a degree
DRIVE_SAMPLE_M = 0.005
CLOSER_M = 1e-3  # a return must come this much closer than at the origin to count as approached


@dataclass(frozen=True)
class Footprint:
    """The cart's rectangle in ``base_link`` (config/base.json ``footprint``): how far it
    reaches ahead of the axle, behind it, and to each side."""

    front_m: float = 0.0625
    rear_m: float = 0.30
    half_width_m: float = 0.275

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> Footprint:
        """config/base.json's object (its ``footprint`` block)."""
        block = data.get("footprint", {})
        d = cls()
        return cls(
            float(block.get("front_m", d.front_m)),
            float(block.get("rear_m", d.rear_m)),
            float(block.get("half_width_m", d.half_width_m)),
        )

    @property
    def reach_m(self) -> float:
        """The farthest corner from the axle's middle: what a turn in place sweeps."""
        return math.hypot(max(self.front_m, self.rear_m), self.half_width_m)


@dataclass(frozen=True)
class Blocker:
    """The return that blocks a motion: where it is (the origin's frame, metres), how near the
    footprint comes to it, and where on the motion (degrees or metres from the origin)."""

    x_m: float
    y_m: float
    gap_m: float
    at: float

    def text(self, unit: str) -> str:
        """``a return at (-0.42, -0.20) m comes within 0.04 m at 12 deg`` (``unit`` deg or cm:
        the motion's own)."""
        at = self.at if unit == "deg" else self.at * 100.0
        return (
            f"a return at ({self.x_m:+.2f}, {self.y_m:+.2f}) m comes within {self.gap_m:.2f} m"
            f" at {at:+.0f} {unit}"
        )


def gaps(points: Points, footprint: Footprint) -> NDArray[np.float64]:
    """Each point's distance to the footprint's rectangle (0 inside it)."""
    if len(points) == 0:
        return np.zeros(0)
    x, y = points[:, 0], points[:, 1]
    dx = np.maximum(np.maximum(-footprint.rear_m - x, x - footprint.front_m), 0.0)
    dy = np.maximum(np.abs(y) - footprint.half_width_m, 0.0)
    out: NDArray[np.float64] = np.hypot(dx, dy)
    return out


def _first_hit(
    footprint: Footprint, margin_m: float, origin: Points, poses: list[Points]
) -> tuple[int, int, float] | None:
    """The first (point, pose) whose gap falls under ``margin_m`` and below its gap at the
    origin, given the points as each pose of the motion sees them; (index, pose, gap)."""
    start = gaps(origin, footprint)
    for k, seen in enumerate(poses):
        gap = gaps(seen, footprint)
        hit = (gap < margin_m) & (gap < start - CLOSER_M) & (start > 0.0)
        if np.any(hit):
            i = int(np.argmax(hit))
            return i, k, float(gap[i])
    return None


def turn_blocker(
    points: Points, footprint: Footprint, start_rad: float, end_rad: float, margin_m: float
) -> Blocker | None:
    """The return the footprint would come within ``margin_m`` of while turning in place from
    ``start_rad`` to ``end_rad`` (headings from the motion's origin, + left; ``points`` as the
    cart at the origin sees them); None when the arc is clear."""
    sweep = end_rad - start_rad
    if len(points) == 0 or sweep == 0.0:
        return None
    near = points[np.hypot(points[:, 0], points[:, 1]) <= footprint.reach_m + margin_m]
    if len(near) == 0:
        return None
    angles = np.linspace(start_rad, end_rad, max(1, math.ceil(abs(sweep) / TURN_SAMPLE_RAD)) + 1)
    x, y = near[:, 0], near[:, 1]
    poses = []
    for phi in angles:  # the cart turned by phi: a fixed point turns by -phi in its frame
        c, s = math.cos(phi), math.sin(phi)
        poses.append(np.column_stack((c * x + s * y, -s * x + c * y)))
    found = _first_hit(footprint, margin_m, near, poses)
    if found is None:
        return None
    i, k, gap = found
    return Blocker(float(x[i]), float(y[i]), gap, math.degrees(float(angles[k])))


def drive_blocker(
    points: Points, footprint: Footprint, start_m: float, end_m: float, margin_m: float
) -> Blocker | None:
    """The return the footprint would come within ``margin_m`` of while driving straight from
    ``start_m`` to ``end_m`` (travel from the motion's origin, + forward; ``points`` as the cart
    at the origin sees them); None when the way is clear."""
    run = end_m - start_m
    if len(points) == 0 or run == 0.0:
        return None
    shifts = np.linspace(start_m, end_m, max(1, math.ceil(abs(run) / DRIVE_SAMPLE_M)) + 1)
    poses = [points - np.array([shift, 0.0]) for shift in shifts]
    found = _first_hit(footprint, margin_m, points, poses)
    if found is None:
        return None
    i, k, gap = found
    return Blocker(float(points[i, 0]), float(points[i, 1]), gap, float(shifts[k]))


def near_points(
    angles: NDArray[np.float64],
    ranges: NDArray[np.float64],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
    radius_m: float,
) -> Points:
    """A scan's returns (laser frame: bearings and ranges, NaN where none) in ``base_link``
    through the laser's mount, the ones within ``radius_m`` of the axle's middle."""
    good = np.isfinite(ranges)
    a, r = angles[good], ranges[good]
    if len(r) == 0:
        return np.zeros((0, 2))
    xyz = np.column_stack((r * np.cos(a), r * np.sin(a), np.zeros(len(r))))
    placed = xyz @ np.asarray(rotation, dtype=float).T + np.asarray(translation, dtype=float)
    xy: Points = placed[:, :2]
    out: Points = xy[np.hypot(xy[:, 0], xy[:, 1]) <= radius_m]
    return out


# ---- the goal server's word ------------------------------------------------------------------
@dataclass(frozen=True)
class View:
    """The goal server's word on the surroundings: when it arrived (this process's clock),
    whether anything navigates (None: no word), how old its newest lidar scan was, that scan's
    returns near the cart (base_link, metres), why there is no usable view ("" when there is
    one), and how long ago its last cancel came (ros/stop.sh sends one; None: none)."""

    at: float
    navigating: bool | None = None
    scan_age_s: float | None = None
    points: Points = field(default_factory=lambda: np.zeros((0, 2)), compare=False, repr=False)
    why: str = ""
    cancel_age_s: float | None = None


class Surroundings(Protocol):
    """What says the way is clear: asked continuously only while a motion runs."""

    def watch(self, on: bool) -> None:
        """Start (or stop) keeping :meth:`latest` fresh; never blocks."""
        ...

    def latest(self) -> View | None:
        """The newest view; never blocks."""
        ...


def view_of(answer: Mapping[str, Any], at: float) -> View:
    """The goal server's ``clearance`` answer as a :class:`View`."""
    if answer.get("event") != "clearance":
        detail = answer.get("detail") or answer.get("event")
        return View(at, why=f"the goal server has no clearance ({detail}): an older build,"
                            " `ros/laptop.sh kick goal_server`")  # fmt: skip
    nav = answer.get("navigating")
    age = answer.get("scan_age_s")
    cancel = answer.get("cancel_age_s")
    raw = answer.get("points") or []
    points = np.asarray(raw, dtype=np.float64).reshape(-1, 2) if raw else np.zeros((0, 2))
    lidar = str(answer.get("lidar", ""))
    why = "" if lidar == "ok" else f"the lidar: {lidar or 'no word'}"
    if age is None and not why:
        why = "no lidar scan has reached the goal server"
    return View(
        at,
        bool(nav) if isinstance(nav, bool) else None,
        float(age) if isinstance(age, (int, float)) else None,
        points,
        why,
        float(cancel) if isinstance(cancel, (int, float)) else None,
    )


class GoalServerView:
    """:class:`Surroundings` from the goal server beside Nav2 on this Mac (``{"cmd":
    "clearance"}`` on 127.0.0.1:3337: ``navigating`` as ``where`` says it, and the newest scan's
    returns within ``radius_m``), asked every ``every_s`` on a daemon thread of its own while
    watched; idle otherwise."""

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        radius_m: float = 1.0,
        every_s: float = 0.1,
        timeout_s: float = 0.3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """The goal server's address (pepin.goal_link's by default) and how often it is asked."""
        from pepin import goal_link

        self.host = host or goal_link.HOST
        self.port = port or goal_link.PORT
        self.radius_m, self.every_s, self.timeout_s = radius_m, every_s, timeout_s
        self.clock = clock
        self._wanted = threading.Event()
        self._lock = threading.Lock()
        self._latest: View | None = None
        self._thread: threading.Thread | None = None

    def watch(self, on: bool) -> None:
        """Ask (or stop asking) the goal server."""
        if on:
            with self._lock:
                if self._thread is None:
                    self._thread = threading.Thread(
                        target=self._run, name="wheels-view", daemon=True
                    )
                    self._thread.start()
            self._wanted.set()
        else:
            self._wanted.clear()

    def latest(self) -> View | None:
        """The newest answer."""
        with self._lock:
            return self._latest

    def look(self) -> View:
        """One request (blocks up to ``timeout_s``)."""
        from pepin import goal_link

        request = {"cmd": "clearance", "radius_m": self.radius_m}
        try:
            answer = goal_link.ask(request, self.host, self.port, self.timeout_s)
        except OSError as error:
            return View(self.clock(), why=f"the goal server does not answer ({error}):"
                                          " `ros/laptop.sh nav`")  # fmt: skip
        return view_of(answer, self.clock())

    def _run(self) -> None:
        while True:
            self._wanted.wait()
            began = self.clock()
            view = self.look()
            with self._lock:
                self._latest = view
            time.sleep(max(0.0, self.every_s - (self.clock() - began)))
