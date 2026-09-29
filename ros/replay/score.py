"""The fixed score of one replayed drive: what the costmaps put in the cart's way, and where.

QUESTION: with a given set of costmap parameters, how much did the obstacle pipeline block this
drive? The numbers Fable read off the live robot by hand on 2026-09-25/26 (scratch/costmap/
corridor_check.py, stuck_snap.py, cell_owner.py, marks_rear.py), now one row per drive.

METHOD, over the replayed costmaps (ros/replay/engine, the snapshot file) while the goal was active:
    * ``goal_max`` / ``goal_end``: LETHAL cells (254) of the GLOBAL costmap — the planner's map —
      whose centre lies within 0.35 m of the goal pose; the worst snapshot, and the last one;
    * ``corr_p50`` / ``corr_max``: LETHAL and INSCRIBED cells (254, 253) of the global costmap
      within 0.45 m of the RECORDED /plan polyline, one count per recorded plan, against the
      newest global snapshot at or before that plan; median and worst;
    * ``cam_mean`` / ``cam_max``: "camera-only" — LETHAL cells of the LOCAL costmap within 2 m of
      the cart that no lidar return confirms within 1.5 cells (pepin.marks_audit, the live node's
      arithmetic: the newest /scan placed in the grid's frame at its own stamp); per snapshot;
    * ``stuck_s``: seconds the RECORDED /cmd_vel held zero while the goal was active — the ground
      truth of "stuck", unchanged by any parameter (a silence longer than the base's 0.5 s
      cmd_timeout is a zero: the wheels stopped);
    * ``fid``: how close the replay is to what the board had — F1 of the lethal cells of each
      recorded /local_costmap/costmap against the replayed local snapshot nearest in time, a cell
      matching within 1.5 cells; the mean over the drive.
Each costmap is scored from its first clear inside the bag (:func:`scored_from`): before it, the
board's costmap held marks from before the recording began and the replay's cannot.
ANSWER: :func:`score_drive` returns the row; :func:`table` and :func:`diff_table` print them.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.marks_audit import MATCH_CELLS, RADIUS_M, audit_marks, nearest_distance

Array = npt.NDArray[np.float64]
Costs = npt.NDArray[np.uint8]

LETHAL = 254  # nav2_costmap_2d LETHAL_OBSTACLE: a sensor (or the map) marked this cell
INSCRIBED = 253  # INSCRIBED_INFLATED_OBSTACLE: the cart's centre may not stand here
GOAL_RADIUS_M = 0.35  # "lethal within 0.35 m of the goal" (corridor_check.py)
CORRIDOR_HALF_WIDTH_M = 0.45  # the corridor along the plan (corridor_check.py)
CMD_TIMEOUT_S = 0.5  # the base bridge's cmd_timeout_s: a silence this long stops the wheels
ZERO_TWIST = 1e-3  # |v| (m/s) and |w| (rad/s) below this are a zero command
FIDELITY_MATCH_CELLS = 1.5
WARMUP_S = 2.0  # never scored: the first seconds of a bag (TF with no past, costmaps empty)
PERSISTENCE_S = 1.0  # the longest observation_persistence of the costmaps' sources
LOCAL, GLOBAL = 0, 1
LIDAR, CAMERA = 0, 1

COLUMNS = (
    "run",
    "goal",
    "active_s",
    "unseen_s",
    "stuck_s",
    "goal_max",
    "goal_end",
    "corr_p50",
    "corr_max",
    "cam_mean",
    "cam_max",
    "fid",
    "wall_s",
    "x_rt",
)
# The columns a parameter can move: the ones a diff reports.
REPLAYED = ("goal_max", "goal_end", "corr_p50", "corr_max", "cam_mean", "cam_max", "fid")


@dataclass
class Snapshot:
    """One costmap update of the replay: the costs (row-major from the origin), where the cart
    stood, and — for the audited costmap — the newest sensor returns placed in its frame."""

    costmap: int
    t: float
    robot: tuple[float, float, float]
    origin: tuple[float, float]
    resolution: float
    costs: Costs
    points: dict[int, Array] = field(default_factory=dict)


def read_snapshots(path: Path) -> list[Snapshot]:
    """Every record of a costmap_replay snapshot file, in the order written."""
    buf = path.read_bytes()
    out: list[Snapshot] = []
    i = 0
    while i < len(buf):
        kind = buf[i : i + 1]
        if kind == b"C":
            costmap = buf[i + 1]
            t, x, y, yaw, ox, oy, res = struct.unpack_from("<7d", buf, i + 2)
            sx, sy = struct.unpack_from("<2I", buf, i + 58)
            start = i + 66
            costs = np.frombuffer(buf, dtype=np.uint8, count=sx * sy, offset=start)
            out.append(Snapshot(costmap, t, (x, y, yaw), (ox, oy), res, costs.reshape(sy, sx)))
            i = start + sx * sy
        elif kind == b"P":
            sensor = buf[i + 1]
            (n,) = struct.unpack_from("<I", buf, i + 10)
            xy = np.frombuffer(buf, dtype=np.float32, count=2 * n, offset=i + 14)
            out[-1].points[sensor] = xy.reshape(n, 2).astype(np.float64)
            i += 14 + 8 * n
        else:
            raise ValueError(f"{path}: unknown record {kind!r} at byte {i}")
    return out


def cell_centres(
    mask: npt.NDArray[np.bool_], origin: tuple[float, float], resolution: float
) -> Array:
    """Centres (n, 2) of the cells ``mask`` selects, in the grid's frame."""
    rows, cols = np.nonzero(mask)
    return np.column_stack(
        (origin[0] + (cols + 0.5) * resolution, origin[1] + (rows + 0.5) * resolution)
    ).astype(np.float64)


def cells_near_point(
    costs: Costs,
    origin: tuple[float, float],
    resolution: float,
    point: tuple[float, float],
    radius_m: float,
    floor: int = LETHAL,
) -> int:
    """How many cells of cost >= ``floor`` have their centre within ``radius_m`` of ``point``."""
    centres = cell_centres(costs >= floor, origin, resolution)
    if not len(centres):
        return 0
    d = np.hypot(centres[:, 0] - point[0], centres[:, 1] - point[1])
    return int((d <= radius_m).sum())


def distance_to_polyline(points: Array, line: Array) -> Array:
    """For every point (n, 2) its distance to the polyline ``line`` (m, 2); a single vertex is
    a point."""
    if len(points) == 0:
        return np.zeros(0)
    if len(line) == 1:
        return np.asarray(np.hypot(points[:, 0] - line[0, 0], points[:, 1] - line[0, 1]))
    a, b = line[:-1], line[1:]
    ab = b - a
    length2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
    ap = points[:, None, :] - a[None, :, :]
    u = np.clip((ap * ab[None, :, :]).sum(axis=2) / length2[None, :], 0.0, 1.0)
    nearest = a[None, :, :] + u[:, :, None] * ab[None, :, :]
    d = np.hypot(points[:, None, 0] - nearest[:, :, 0], points[:, None, 1] - nearest[:, :, 1])
    return np.asarray(d.min(axis=1))


def corridor_cells(
    costs: Costs,
    origin: tuple[float, float],
    resolution: float,
    line: Array,
    half_width_m: float = CORRIDOR_HALF_WIDTH_M,
) -> int:
    """How many lethal or inscribed cells (>= 253) lie within ``half_width_m`` of the polyline."""
    if len(line) == 0:
        return 0
    centres = cell_centres(costs >= INSCRIBED, origin, resolution)
    return int((distance_to_polyline(centres, line) <= half_width_m).sum())


def unbacked_lethal(snapshot: Snapshot) -> int:
    """Lethal cells within :data:`pepin.marks_audit.RADIUS_M` of the cart with no lidar return
    within :data:`pepin.marks_audit.MATCH_CELLS` cells — the live marks_audit's arithmetic on the
    ROS scale (254 -> 100), its camera-only and unexplained together."""
    ros_scale = np.where(snapshot.costs == LETHAL, 100, 0)
    verdict = audit_marks(
        ros_scale,
        snapshot.origin,
        snapshot.resolution,
        (snapshot.robot[0], snapshot.robot[1]),
        snapshot.points.get(LIDAR, np.zeros((0, 2))),
        snapshot.points.get(CAMERA, np.zeros((0, 2))),
        radius_m=RADIUS_M,
        match_cells=MATCH_CELLS,
    )
    return verdict.lethal - verdict.lidar_backed


def stuck_seconds(
    cmds: list[tuple[float, float, float]],
    start: float,
    end: float,
    timeout_s: float = CMD_TIMEOUT_S,
) -> float:
    """Seconds of [start, end] during which the cart was commanded to stand: the newest
    (t, v, w) command is a zero twist, or none has come for ``timeout_s`` (the base stops by
    itself then), or none has come yet."""
    if end <= start:
        return 0.0
    stuck = 0.0
    last_t, last_zero = -math.inf, True
    t = start
    for c_t, v, w in sorted(cmds):
        if c_t <= start:
            last_t, last_zero = c_t, abs(v) < ZERO_TWIST and abs(w) < ZERO_TWIST
            continue
        if c_t >= end:
            break
        stuck += _zero_time(t, c_t, last_t, last_zero, timeout_s)
        t, last_t, last_zero = c_t, c_t, abs(v) < ZERO_TWIST and abs(w) < ZERO_TWIST
    stuck += _zero_time(t, end, last_t, last_zero, timeout_s)
    return stuck


def _zero_time(a: float, b: float, last_t: float, last_zero: bool, timeout_s: float) -> float:
    """Zero-command seconds in [a, b] when the newest command (at ``last_t``) is ``last_zero``."""
    if last_zero:
        return b - a
    expires = last_t + timeout_s
    return max(0.0, b - max(a, expires))


def lethal_f1(
    replayed: Snapshot,
    recorded: npt.NDArray[np.int8],
    origin: tuple[float, float],
    resolution: float,
    match_cells: float = FIDELITY_MATCH_CELLS,
) -> float | None:
    """F1 of the lethal cells of a replayed snapshot against a recorded OccupancyGrid (ROS
    scale, lethal = 100) of the same frame, a cell matching within ``match_cells`` cells;
    None when neither has a lethal cell."""
    ours = cell_centres(replayed.costs == LETHAL, replayed.origin, replayed.resolution)
    theirs = cell_centres(recorded == 100, origin, resolution)
    if not len(ours) and not len(theirs):
        return None
    if not len(ours) or not len(theirs):
        return 0.0
    tolerance = match_cells * resolution
    precision = float((nearest_distance(ours, theirs) <= tolerance).mean())
    recall = float((nearest_distance(theirs, ours) <= tolerance).mean())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _latest_before(snaps: list[Snapshot], t: float) -> Snapshot | None:
    best = None
    for s in snaps:
        if s.t <= t:
            best = s
        else:
            break
    return best


def _nearest(snaps: list[Snapshot], t: float, within_s: float) -> Snapshot | None:
    if not snaps:
        return None
    best = min(snaps, key=lambda s: abs(s.t - t))
    return best if abs(best.t - t) <= within_s else None


def scored_from(drive: dict[str, Any], which: str, clears_applied: bool = True) -> float:
    """When the replayed ``which`` costmap ("local" / "global") starts holding what the board's
    held: the replay starts empty with a TF buffer that has no past, the board's costmap had
    marks from before the bag began. From the first clear of that costmap inside the bag (at
    least :data:`PERSISTENCE_S` in, so the observations it re-marks from are the replay's too)
    both are the same; never before :data:`WARMUP_S`, nor before the goal."""
    bag_start = float(drive["bag_start"])
    t = bag_start + WARMUP_S
    if clears_applied:
        inside = [
            c for c, w in drive.get("clears", []) if w == which and c >= bag_start + PERSISTENCE_S
        ]
        if inside:
            t = max(t, min(inside))
    return max(t, float(drive["goal_accepted"]))


def score_drive(
    drive: dict[str, Any],
    snapshots: list[Snapshot],
    recorded: dict[str, npt.NDArray[Any]],
    local_period_s: float,
    clears_applied: bool = True,
) -> dict[str, Any]:
    """One drive's row: the prepared description (ros/replay/prepare.py's drive.json), the
    replay's snapshots and the board's own recorded local costmaps."""
    start = max(float(drive["goal_accepted"]), float(drive["bag_start"]))
    end = min(float(drive["goal_end"]), float(drive["bag_end"]))
    local_from = scored_from(drive, "local", clears_applied)
    global_from = scored_from(drive, "global", clears_applied)
    local = [s for s in snapshots if s.costmap == LOCAL and local_from <= s.t <= end]
    glob = [s for s in snapshots if s.costmap == GLOBAL and global_from <= s.t <= end]
    row: dict[str, Any] = {
        "run": int(drive["run"]),
        "goal": drive["goal_name"],
        "active_s": round(end - start, 1),
        "unseen_s": round(max(0.0, float(drive["bag_start"]) - float(drive["goal_accepted"])), 1),
        "stuck_s": round(
            stuck_seconds([(c[0], c[1], c[2]) for c in drive["cmd_vel"]], start, end), 1
        ),
    }
    goal = drive.get("goal")
    at_goal = (
        [
            cells_near_point(s.costs, s.origin, s.resolution, (goal[0], goal[1]), GOAL_RADIUS_M)
            for s in glob
        ]
        if goal
        else []
    )
    row["goal_max"] = max(at_goal) if at_goal else None
    row["goal_end"] = at_goal[-1] if at_goal else None
    corridor = []
    for plan in drive["plans"]:
        if not global_from <= plan["t"] <= end or plan["frame"] != "map" or not plan["points"]:
            continue
        snap = _latest_before(glob, plan["t"])
        if snap is not None:
            line = np.asarray(plan["points"], dtype=np.float64)
            corridor.append(corridor_cells(snap.costs, snap.origin, snap.resolution, line))
    row["corr_p50"] = float(np.median(corridor)) if corridor else None
    row["corr_max"] = max(corridor) if corridor else None
    unbacked = [unbacked_lethal(s) for s in local if LIDAR in s.points]
    row["cam_mean"] = round(float(np.mean(unbacked)), 1) if unbacked else None
    row["cam_max"] = max(unbacked) if unbacked else None
    scores = []
    offset = 0
    for k, t in enumerate(recorded["t"]):
        h, w = (int(v) for v in recorded["shape"][k])
        grid = recorded["data"][offset : offset + h * w].reshape(h, w)
        offset += h * w
        if not local_from <= t <= end:
            continue
        snap = _nearest(local, float(t), local_period_s / 2.0 + 1e-3)
        if snap is None:
            continue
        origin = (float(recorded["origin"][k][0]), float(recorded["origin"][k][1]))
        f1 = lethal_f1(snap, grid, origin, float(recorded["resolution"][k]))
        if f1 is not None:
            scores.append(f1)
    row["fid"] = round(float(np.mean(scores)), 2) if scores else None
    return row


DECIMALS = {
    "active_s": 1,
    "unseen_s": 1,
    "stuck_s": 1,
    "corr_p50": 1,
    "cam_mean": 1,
    "fid": 2,
    "wall_s": 2,
}


def _cell(value: Any, column: str = "") -> str:
    """One value as the tables print it: '-' for none, the column's decimals, counts whole."""
    if value is None:
        return "-"
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float(value):.{DECIMALS.get(column, 0)}f}"
    return str(value)


def table(rows: list[dict[str, Any]]) -> str:
    """The rows as a fixed-width text table, one line per drive, then the totals."""
    lines = [" ".join(f"{c:>9}" for c in COLUMNS)]
    for r in [*rows, totals(rows)]:
        lines.append(" ".join(f"{_cell(r.get(c), c):>9}" for c in COLUMNS))
    return "\n".join(lines)


def totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sums over the drives (means for fid and x_rt), labelled in the first two columns."""
    out: dict[str, Any] = {"run": "all", "goal": f"{len(rows)}"}
    for c in COLUMNS[2:]:
        values = [r[c] for r in rows if r.get(c) is not None]
        if not values:
            out[c] = None
        elif c in ("fid", "x_rt"):
            out[c] = round(float(np.mean(values)), 2)
        else:
            out[c] = round(float(np.sum(values)), 1)
    return out


def diff_table(rows: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> str:
    """Per drive the change of every replayed column against ``baseline`` (candidate minus
    baseline), and the change of the totals; a drive missing from either side is said so."""
    base = {int(r["run"]): r for r in baseline}
    head = ["run", "goal", "stuck_s", *REPLAYED]
    lines = [" ".join(f"{c:>9}" for c in head)]

    def line(r: dict[str, Any], b: dict[str, Any]) -> str:
        cells = [_cell(r["run"]), _cell(r["goal"]), _cell(r.get("stuck_s"), "stuck_s")]
        for c in REPLAYED:
            new, old = r.get(c), b.get(c)
            if new is None or old is None:
                cells.append("-")
                continue
            delta = float(new) - float(old)
            decimals = max(1, DECIMALS.get(c, 0))
            cells.append("0" if abs(delta) < 1e-9 else f"{delta:+.{decimals}f}")
        return " ".join(f"{v:>9}" for v in cells)

    matched = []
    for r in rows:
        b = base.get(int(r["run"]))
        if b is None:
            lines.append(f"{r['run']:>9} not in the baseline")
            continue
        matched.append((r, b))
        lines.append(line(r, b))
    if matched:
        lines.append(line(totals([r for r, _ in matched]), totals([b for _, b in matched])))
    return "\n".join(lines)
