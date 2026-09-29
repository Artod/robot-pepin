"""The replay stand's score (ros/replay/score.py) on synthetic grids: every number of a baseline row
means what its column says, and the snapshot file reads back what costmap_replay writes.

The stand exists so a costmap change is judged on the recorded drives before it is driven; a
score that counted the wrong cells would send that judgement the wrong way on every drive at
once. Each test builds the smallest grid or command tape whose right answer is obvious.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _score_module() -> Any:
    """ros/replay/score.py, loaded by path: ros/replay is not a package."""
    spec = importlib.util.spec_from_file_location("replay_score", REPO / "ros/replay/score.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("replay_score", module)
    spec.loader.exec_module(module)
    return module


S = _score_module()
RES = 0.05


def grid(cells: dict[tuple[int, int], int], size: int = 40) -> Any:
    """A size x size cost grid, free but for ``{(row, col): cost}``."""
    costs = np.zeros((size, size), dtype=np.uint8)
    for (row, col), cost in cells.items():
        costs[row, col] = cost
    return costs


def centre(row: int, col: int, origin: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """The metric centre of cell (row, col)."""
    return (origin[0] + (col + 0.5) * RES, origin[1] + (row + 0.5) * RES)


def snapshot(costs: Any, robot: tuple[float, float] = (1.0, 1.0), **points: Any) -> Any:
    """A local snapshot at the origin with ``lidar=`` / ``camera=`` returns."""
    snap = S.Snapshot(S.LOCAL, 10.0, (robot[0], robot[1], 0.0), (0.0, 0.0), RES, costs)
    if "lidar" in points:
        snap.points[S.LIDAR] = np.asarray(points["lidar"], dtype=np.float64).reshape(-1, 2)
    if "camera" in points:
        snap.points[S.CAMERA] = np.asarray(points["camera"], dtype=np.float64).reshape(-1, 2)
    return snap


# ---- the goal: lethal cells within 0.35 m ---------------------------------------------------
def test_the_goal_counts_lethal_cells_by_their_centre_inside_the_radius() -> None:
    """A lethal cell 0.30 m from the goal counts, one 0.40 m away does not, and an inscribed or
    inflated cell at the goal itself is not lethal."""
    goal = centre(20, 20)
    costs = grid({(20, 26): S.LETHAL, (20, 28): S.LETHAL, (20, 20): S.INSCRIBED, (21, 20): 128})
    assert S.cells_near_point(costs, (0.0, 0.0), RES, goal, S.GOAL_RADIUS_M) == 1


def test_the_goal_floor_is_the_cost_asked_for() -> None:
    """With the inscribed floor the 253 band is counted too — what the corridor asks."""
    goal = centre(10, 10)
    costs = grid({(10, 11): S.INSCRIBED, (10, 12): S.LETHAL})
    assert S.cells_near_point(costs, (0.0, 0.0), RES, goal, 0.2, floor=S.INSCRIBED) == 2


# ---- the corridor along the recorded plan ---------------------------------------------------
def test_distance_to_a_polyline_is_to_the_nearest_segment_not_the_nearest_vertex() -> None:
    """Midway along a 2 m segment a point 0.3 m off it is 0.3 m away, though both vertices are
    a metre off; past the end it is the distance to the end vertex."""
    line = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    points = np.array([[1.0, 0.3], [2.5, -0.4], [2.2, 1.0]])
    np.testing.assert_allclose(S.distance_to_polyline(points, line), [0.3, np.hypot(0.5, 0.4), 0.2])


def test_a_one_pose_plan_is_a_point() -> None:
    """A plan of one pose (a goal on the cart) is judged as a disc, not refused."""
    d = S.distance_to_polyline(np.array([[3.0, 4.0]]), np.array([[0.0, 0.0]]))
    np.testing.assert_allclose(d, [5.0])


def test_the_corridor_counts_lethal_and_inscribed_within_its_half_width() -> None:
    """Along a straight plan: the lethal cell and the inscribed cell inside 0.45 m count, the
    one 0.55 m off and the inflated ramp do not."""
    line = np.array([centre(20, 2), centre(20, 38)])
    costs = grid({(20, 10): S.LETHAL, (28, 12): S.INSCRIBED, (31, 14): S.LETHAL, (21, 16): 200})
    assert S.corridor_cells(costs, (0.0, 0.0), RES, line) == 2
    assert S.corridor_cells(costs, (0.0, 0.0), RES, line[:0]) == 0


# ---- camera-only: lethal cells no lidar return confirms -------------------------------------
def test_a_lethal_cell_with_a_lidar_return_on_it_is_not_camera_only() -> None:
    """Two lethal cells near the cart; the lidar lands on one within 1.5 cells, so one is left."""
    costs = grid({(20, 30): S.LETHAL, (30, 20): S.LETHAL})
    lidar = [centre(20, 30)[0] + 0.06, centre(20, 30)[1]]
    assert S.unbacked_lethal(snapshot(costs, lidar=lidar, camera=[centre(30, 20)])) == 1


def test_a_lidar_return_two_cells_off_does_not_confirm() -> None:
    """1.5 cells is the tolerance: a return 0.10 m (two cells) away confirms nothing."""
    costs = grid({(20, 30): S.LETHAL})
    lidar = [centre(20, 30)[0] + 0.10, centre(20, 30)[1]]
    assert S.unbacked_lethal(snapshot(costs, lidar=lidar)) == 1


def test_cells_beyond_two_metres_of_the_cart_are_not_judged() -> None:
    """Out of the lidar's own marking range nothing is expected of it."""
    costs = grid({(39, 39): S.LETHAL}, size=60)
    assert S.unbacked_lethal(snapshot(costs, robot=(0.0, 0.0), lidar=[])) == 0


def test_camera_only_counts_what_no_sensor_explains_too() -> None:
    """The column is 'no lidar return confirms it': a cell the camera's fan does not land on
    (a ToF mark, a stale one) is counted as well — marks_audit's camera-only plus unexplained."""
    costs = grid({(20, 30): S.LETHAL, (30, 20): S.LETHAL})
    assert S.unbacked_lethal(snapshot(costs, lidar=[], camera=[centre(30, 20)])) == 2


# ---- stuck: the recorded /cmd_vel at zero while the goal is active --------------------------
def at_10_hz(t0: float, t1: float, v: float, w: float) -> list[tuple[float, float, float]]:
    """The controller's tape: one (t, v, w) every 0.1 s over [t0, t1)."""
    return [(t0 + 0.1 * k, v, w) for k in range(round((t1 - t0) / 0.1))]


def test_stuck_is_the_time_the_newest_command_is_a_zero_twist() -> None:
    """Moving 0-2 s, zero 2-5 s, turning in place 5-8 s (a turn is not stuck), zero after."""
    cmds = at_10_hz(0, 2, 0.2, 0) + at_10_hz(2, 5, 0, 0) + at_10_hz(5, 8, 0, 0.5)
    cmds += at_10_hz(8, 10, 0, 0)
    assert abs(S.stuck_seconds(cmds, 0.0, 10.0) - (3.0 + 2.0)) < 1e-9


def test_a_silent_controller_is_a_stop_after_the_base_timeout() -> None:
    """One command at t=1 s and nothing after: the base stops 0.5 s later, and the time before
    the first command is a standing cart as well."""
    assert S.stuck_seconds([(1.0, 0.2, 0.0)], 0.0, 4.0) == 1.0 + 2.5


def test_stuck_starts_from_the_command_in_force_at_the_window() -> None:
    """A zero sent before the goal's window counts from the window's start, not from its own."""
    assert S.stuck_seconds([(-3.0, 0.0, 0.0), (1.0, 0.1, 0.0)], 0.0, 1.2) == 1.0


def test_an_empty_window_is_never_stuck() -> None:
    assert S.stuck_seconds([(0.0, 0.0, 0.0)], 5.0, 5.0) == 0.0


# ---- fidelity: the replay against the board's own local costmap ----------------------------
def test_fidelity_is_one_for_the_same_cells_and_forgives_one_cell_of_offset() -> None:
    """The board's grid on its own lattice, one cell shifted: F1 is still 1 within 1.5 cells."""
    replayed = snapshot(grid({(10, 10): S.LETHAL, (12, 14): S.LETHAL}))
    recorded = np.zeros((40, 40), dtype=np.int8)
    recorded[10, 11] = recorded[12, 15] = 100
    assert S.lethal_f1(replayed, recorded, (0.0, 0.0), RES) == 1.0


def test_fidelity_halves_with_a_cell_the_board_never_had() -> None:
    """One matched cell and one only the replay has: precision 1/2, recall 1, F1 2/3."""
    replayed = snapshot(grid({(10, 10): S.LETHAL, (30, 30): S.LETHAL}))
    recorded = np.zeros((40, 40), dtype=np.int8)
    recorded[10, 10] = 100
    assert abs(S.lethal_f1(replayed, recorded, (0.0, 0.0), RES) - 2.0 / 3.0) < 1e-9


def test_fidelity_is_undefined_when_neither_has_a_lethal_cell_and_zero_when_one_has() -> None:
    empty = np.zeros((40, 40), dtype=np.int8)
    assert S.lethal_f1(snapshot(grid({})), empty, (0.0, 0.0), RES) is None
    assert S.lethal_f1(snapshot(grid({(1, 1): S.LETHAL})), empty, (0.0, 0.0), RES) == 0.0


# ---- the snapshot file ----------------------------------------------------------------------
def write_costmap(index: int, t: float, costs: Any) -> bytes:
    """One 'C' record as costmap_replay writes it."""
    h, w = costs.shape
    head = b"C" + bytes([index]) + struct.pack("<7d", t, 1.0, 2.0, 0.5, -1.5, -1.5, RES)
    return head + struct.pack("<2I", w, h) + costs.tobytes()


def write_points(sensor: int, t: float, xy: list[tuple[float, float]]) -> bytes:
    """One 'P' record."""
    flat = [v for p in xy for v in p]
    return (
        b"P"
        + bytes([sensor])
        + struct.pack("<dI", t, len(xy))
        + struct.pack(f"<{len(flat)}f", *flat)
    )


def test_the_snapshot_file_reads_back_costmaps_and_their_points(tmp_path: Path) -> None:
    costs = np.arange(12, dtype=np.uint8).reshape(3, 4)
    data = write_costmap(0, 5.0, costs) + write_points(0, 5.0, [(1.0, 2.0), (3.0, 4.0)])
    data += write_points(1, 5.0, []) + write_costmap(1, 5.5, costs[:2])
    path = tmp_path / "x.snap"
    path.write_bytes(data)
    local, glob = S.read_snapshots(path)
    assert (local.costmap, local.t, local.robot, local.origin) == (
        0,
        5.0,
        (1.0, 2.0, 0.5),
        (-1.5, -1.5),
    )
    np.testing.assert_array_equal(local.costs, costs)
    np.testing.assert_allclose(local.points[S.LIDAR], [[1.0, 2.0], [3.0, 4.0]])
    assert local.points[S.CAMERA].shape == (0, 2)
    assert glob.costmap == 1 and glob.costs.shape == (2, 4) and not glob.points


# ---- where scoring starts -------------------------------------------------------------------
def test_each_costmap_is_scored_from_its_first_clear_inside_the_bag() -> None:
    """Before its first clear inside the bag the board's costmap still held marks from before the
    recording; a clear in the bag's first second re-marks from observations the replay lacks."""
    drive = {
        "bag_start": 100.0,
        "goal_accepted": 95.0,
        "clears": [[95.0, "local"], [100.5, "global"], [103.4, "local"], [105.1, "global"]],
    }
    assert S.scored_from(drive, "local") == 103.4
    assert S.scored_from(drive, "global") == 105.1
    assert S.scored_from(drive, "global", clears_applied=False) == 100.0 + S.WARMUP_S


def test_without_a_clear_in_the_bag_only_the_warm_up_is_skipped() -> None:
    drive = {"bag_start": 100.0, "goal_accepted": 101.0, "clears": [[101.0, "local"]]}
    assert S.scored_from(drive, "local") == 100.0 + S.WARMUP_S
    assert S.scored_from({**drive, "goal_accepted": 104.0}, "local") == 104.0


# ---- one drive, end to end ------------------------------------------------------------------
def test_a_drive_is_scored_only_inside_its_goal_after_the_warm_up() -> None:
    """Snapshots before the warm-up or after the goal ended are not scored; the plan and the
    goal are judged on the planner's snapshot, camera-only on the controller's."""
    lethal_at_goal = grid({(20, 20): S.LETHAL, (20, 21): S.LETHAL})
    drive = {
        "run": 7,
        "goal_name": "home",
        "goal": [*centre(20, 20), 0.0],
        "goal_accepted": 99.0,
        "bag_start": 100.0,
        "bag_end": 120.0,
        "goal_end": 110.0,
        "cmd_vel": [[100.0, 0.0, 0.0], *[list(c) for c in at_10_hz(105.0, 110.0, 0.2, 0.0)]],
        "plans": [{"t": 104.0, "frame": "map", "points": [centre(20, 5), centre(20, 35)]}],
    }
    snaps = [
        S.Snapshot(S.GLOBAL, 101.0, (0.5, 0.5, 0.0), (0.0, 0.0), RES, grid({(20, 20): S.LETHAL})),
        S.Snapshot(S.GLOBAL, 103.0, (0.5, 0.5, 0.0), (0.0, 0.0), RES, lethal_at_goal),
        S.Snapshot(S.GLOBAL, 112.0, (0.5, 0.5, 0.0), (0.0, 0.0), RES, grid({})),
        snapshot(grid({(20, 30): S.LETHAL}), lidar=[]),
    ]
    snaps[-1].t = 104.0
    recorded = {
        "t": np.array([104.0]),
        "origin": np.array([[0.0, 0.0]]),
        "resolution": np.array([RES]),
        "shape": np.array([[40, 40]]),
        "data": np.where(snaps[-1].costs == S.LETHAL, 100, 0).astype(np.int8).ravel(),
    }
    row = S.score_drive(drive, snaps, recorded, local_period_s=0.2)
    assert row["active_s"] == 10.0 and row["unseen_s"] == 1.0 and row["stuck_s"] == 5.0
    assert row["goal_max"] == 2 and row["goal_end"] == 2
    assert row["corr_p50"] == 2.0 and row["corr_max"] == 2
    assert row["cam_mean"] == 1.0 and row["cam_max"] == 1
    assert row["fid"] == 1.0


# ---- the tables -----------------------------------------------------------------------------
def test_the_diff_is_candidate_minus_baseline_per_drive_and_in_total() -> None:
    base = [{"run": 1, "goal": "home", "stuck_s": 2.0, "goal_max": 10, "fid": 0.9}]
    cand = [{"run": 1, "goal": "home", "stuck_s": 2.0, "goal_max": 4, "fid": 0.95}]
    lines = S.diff_table(cand, base).splitlines()
    assert lines[1].split()[:4] == ["1", "home", "2.0", "-6.0"]
    assert "+0.05" in lines[1] and lines[2].split()[0] == "all"
    assert "not in the baseline" in S.diff_table([{"run": 2, "goal": "x"}], base)


def test_the_table_has_a_line_per_drive_and_the_totals() -> None:
    rows = [{"run": 1, "goal": "home", "stuck_s": 1.25, "fid": 0.5}, {"run": 2, "goal": "x"}]
    lines = S.table(rows).splitlines()
    assert len(lines) == 4 and lines[-1].split()[:2] == ["all", "2"]
    assert "1.2" in lines[1] and "0.50" in lines[1]
