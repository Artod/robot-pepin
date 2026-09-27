"""The kinematic world under ros/sim: the unicycle base, the raycast lidar, the grid's files."""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from pepin.footprint import HULL, hull_box
from pepin.geometry import BaseConfig
from pepin.mounts import load_lidar
from pepin.odometry import Pose2D
from pepin.recording import scan_record_from_ros
from pepin.sim import (
    DEADMAN_S,
    FREE,
    OCCUPIED,
    UNKNOWN,
    Box,
    Grid,
    LidarModel,
    Scenario,
    SimWorld,
    Target,
    UnicycleBase,
    arc,
    cast_boxes,
    cast_grid,
    graph_poses_from_rtabmap_db,
    grid_from_rtabmap_db,
    places_from_payload,
    score_leg,
)

REPO = Path(__file__).resolve().parents[2]
RES = 0.05


def empty_grid(width: int = 40, height: int = 40, value: int = FREE) -> Grid:
    """A free square room of ``width`` x ``height`` cells at 5 cm, origin (0, 0)."""
    return Grid(np.full((height, width), value, dtype=np.int8), RES, 0.0, 0.0)


def with_wall_column(grid: Grid, col: int) -> Grid:
    """The same grid with one occupied column (a wall along y at x = col * res)."""
    cells = grid.cells.copy()
    cells[:, col] = OCCUPIED
    return Grid(cells, grid.resolution, grid.origin_x, grid.origin_y)


def world(grid: Grid, boxes: tuple[Box, ...] = (), pose: Pose2D | None = None) -> SimWorld:
    """The cart at ``pose`` in ``grid``, with the repo's own lidar mount and base limits."""
    base = UnicycleBase.from_config(BaseConfig.from_json(REPO / "config/base.json"), pose)
    return SimWorld(grid, base, LidarModel(load_lidar(REPO / "config")), HULL, boxes)


# -- the base -------------------------------------------------------------------------------------


def test_arc_is_a_straight_line_or_the_exact_circle() -> None:
    assert arc(Pose2D(1.0, 2.0, 0.0), 0.3, 0.0, 2.0) == Pose2D(1.6, 2.0, 0.0)
    quarter = arc(Pose2D(), 0.2, math.pi / 2, 1.0)  # radius 0.2 / (pi / 2)
    r = 0.2 / (math.pi / 2)
    assert (quarter.x, quarter.y, quarter.theta) == pytest.approx((r, r, math.pi / 2))
    back = arc(Pose2D(0.0, 0.0, math.pi / 2), -0.1, 0.0, 1.0)
    assert (back.x, back.y) == pytest.approx((0.0, -0.1))


def test_the_base_clamps_each_axis_to_its_own_limits() -> None:
    base = UnicycleBase.from_config(BaseConfig.from_json(REPO / "config/base.json"))
    assert (base.max_linear, base.max_angular) == (0.30, 1.0)  # config/base.json
    base.command(0.5, -2.0, now=0.0)
    assert base.twist(0.0) == (0.30, -1.0)


def test_the_deadman_stops_the_base_half_a_second_after_the_last_command() -> None:
    base = UnicycleBase(0.3, 1.0)
    base.command(0.2, 0.0, now=10.0)
    assert base.twist(10.0 + DEADMAN_S - 0.01) == (0.2, 0.0)
    assert base.twist(10.0 + DEADMAN_S + 0.01) == (0.0, 0.0)
    moved = base.step(0.1, now=10.1)
    assert moved.travel_m == pytest.approx(0.02)
    assert base.step(0.1, now=11.0).travel_m == 0.0  # silent controller: the wheels stop
    base.halt()
    assert base.twist(11.0) == (0.0, 0.0)


def test_a_cart_driven_into_a_box_stops_there_and_counts_one_contact() -> None:
    # The nose at x 0.0625; a box whose face is 0.10 m ahead of it.
    box = Box("wall", x=0.1625 + 0.05, y=0.0, length=0.10, width=1.0)
    sim = world(empty_grid(80, 80), (box,), Pose2D(-1.0, 0.0, 0.0))
    sim.place(Pose2D(0.0, 0.0, 0.0))
    t = 0.0
    for _ in range(60):  # 1.2 s at 0.3 m/s would be 0.36 m: far past the face
        sim.base.command(0.3, 0.0, t)
        t += 0.02
        sim.step(0.02, t)
    assert 0.08 <= sim.pose.x <= 0.10 + 1e-9, "stopped with the nose at the face, not through it"
    assert sim.odometer.contacts == 1  # one run of refused steps is one contact
    assert sim.odometer.blocked_s > 0.5
    sim.base.command(-0.2, 0.0, t)
    assert not sim.step(0.1, t).blocked  # backing off is never refused


def test_a_cart_standing_in_the_grids_noise_may_drive_out_but_not_deeper() -> None:
    grid = empty_grid(80, 80)
    cells = grid.cells.copy()
    cells[40, 40] = OCCUPIED  # one noisy cell under the hull at (2.0..2.05, 2.0..2.05)
    sim = world(Grid(cells, RES, 0.0, 0.0), pose=Pose2D(2.05, 2.02, 0.0))
    assert sim.overlap() > 0
    sim.base.command(0.2, 0.0, 0.0)
    assert not sim.step(0.1, 0.0).blocked  # forward takes the hull off the cell
    assert (
        sim.overlap() <= world(Grid(cells, RES, 0.0, 0.0), pose=Pose2D(2.05, 2.02, 0.0)).overlap()
    )


# -- the rays -------------------------------------------------------------------------------------


def test_a_ray_meets_a_wall_at_the_boundary_of_the_cell_it_enters() -> None:
    grid = with_wall_column(empty_grid(), 10)  # the wall's cells span x 0.50 .. 0.55
    ranges = cast_grid(grid, 0.125, 0.51, np.array([0.0, math.pi / 4, math.pi]), 12.0)
    assert ranges[0] == pytest.approx(0.375)
    assert ranges[1] == pytest.approx(0.375 * math.sqrt(2.0))
    assert ranges[2] == math.inf  # nothing behind: out of the grid, no return


def test_a_ray_never_hits_its_own_cell_and_goes_blind_past_its_range() -> None:
    grid = with_wall_column(empty_grid(), 10)
    assert cast_grid(grid, 0.52, 0.5, np.array([0.0]), 12.0)[0] == math.inf  # starts in the wall
    assert cast_grid(grid, 0.125, 0.5, np.array([0.0]), 0.3)[0] == math.inf  # wall at 0.375


def test_a_ray_meets_the_near_face_of_a_box_even_turned() -> None:
    box = Box("b", x=1.0, y=0.0, length=0.2, width=0.4)
    assert cast_boxes([box], 0.0, 0.0, np.array([0.0]), 12.0)[0] == pytest.approx(0.9)
    diamond = Box("d", x=1.0, y=0.0, length=0.2, width=0.2, yaw_deg=45.0)
    assert cast_boxes([diamond], 0.0, 0.0, np.array([0.0]), 12.0)[0] == pytest.approx(
        1.0 - 0.1 * math.sqrt(2.0)
    )
    assert cast_boxes([box], 1.0, 0.0, np.array([0.0]), 12.0)[0] == math.inf  # from inside
    assert cast_boxes([box], 0.0, 0.0, np.array([math.pi]), 12.0)[0] == math.inf


def test_the_beams_leave_the_mount_as_the_tape_places_them() -> None:
    mount = load_lidar(REPO / "config")
    lidar = LidarModel(mount)
    # pepin.recording turns the driver's laser-frame angles into robot bearings for every tape
    record = scan_record_from_ros(
        0.0, 0.0, lidar.angle_increment, [1.0] * lidar.beams, [], 0.05, 12.0, 0.1,
        mount_yaw_rad=math.radians(mount.yaw_offset_deg), mount_x_m=mount.x_m,
    )  # fmt: skip
    taped = np.array(record["angles"])
    ours = np.mod(lidar.bearings(), 2.0 * math.pi)
    assert np.max(np.abs(np.angle(np.exp(1j * (ours - taped))))) < 1e-3


def test_the_masked_beams_are_the_posts_shadows() -> None:
    lidar = LidarModel(load_lidar(REPO / "config"))
    masked = np.degrees(np.mod(lidar.bearings()[lidar.masked()], 2.0 * math.pi))
    # config/lidar.json: sensor 192-218 and 317-343 deg at a 87.5 deg offset
    left = (masked >= 104.5 - 1) & (masked <= 130.5 + 1)
    right = (masked >= 229.5 - 1) & (masked <= 255.5 + 1)
    assert masked.size > 0 and np.all(left | right) and left.any() and right.any()


def test_the_scan_cuts_what_the_board_filter_cuts() -> None:
    lidar = LidarModel(load_lidar(REPO / "config"))
    band = hull_box()
    ahead_far = Box("far", x=2.0, y=2.5, length=0.1, width=0.6)
    grid = empty_grid(100, 100)
    sim = world(grid, (ahead_far,), Pose2D(1.0, 2.5, 0.0))
    ranges = sim.scan()
    bearings = lidar.bearings()
    forward = int(np.argmin(np.abs(np.angle(np.exp(1j * bearings)))))
    assert ranges[forward] == pytest.approx(2.0 - 0.05 - 1.0 - lidar.mount.x_m, abs=0.02)
    assert np.all(np.isnan(ranges[lidar.masked()]))
    # a box touching the hull's side, inside the contact band: the board's filter cuts it
    touching = Box("touch", x=1.0, y=2.5 + band["max_y"] - 0.02, length=0.2, width=0.02)
    beside = world(grid, (touching,), Pose2D(1.0, 2.5, 0.0)).scan()
    left = int(np.argmin(np.abs(np.angle(np.exp(1j * (bearings - math.pi / 2))))))
    assert math.isnan(beside[left])


# -- the grid's files -----------------------------------------------------------------------------


def test_a_grid_survives_the_pgm_with_its_three_values_and_its_orientation(tmp_path: Path) -> None:
    cells = np.full((3, 4), FREE, dtype=np.int8)
    cells[0, 0] = OCCUPIED  # row 0 is the lowest y
    cells[2, 3] = UNKNOWN
    grid = Grid(cells, 0.05, -1.5, 2.25)
    grid.save(tmp_path / "room.yaml", note="a test room")
    back = Grid.load(tmp_path / "room.yaml")
    assert np.array_equal(back.cells, cells)
    assert (back.resolution, back.origin_x, back.origin_y) == pytest.approx((0.05, -1.5, 2.25))
    assert (tmp_path / "room.pgm").read_bytes().startswith(b"P5\n4 3\n255\n")
    assert back.occupied_at(np.array([-1.49]), np.array([2.26]))[0]


def compressed(array: np.ndarray, cvtype: int) -> bytes:
    """RTAB-Map's compressData of ``array`` (rows, cols, cv type trailer)."""
    rows, cols = array.shape
    return zlib.compress(array.tobytes()) + struct.pack("<iii", rows, cols, cvtype)


def test_the_grid_and_the_graph_come_out_of_an_rtabmap_database(tmp_path: Path) -> None:
    cells = np.array([[0, 100, -1], [0, 0, 100]], dtype=np.int8)
    poses = np.array(
        [[1, 0, 0, 0.5, 0, 1, 0, -0.25, 0, 0, 1, 0], [0, -1, 0, 2.0, 1, 0, 0, 1.0, 0, 0, 1, 0]],
        dtype=np.float32,
    )
    db = tmp_path / "rtabmap.db"
    with sqlite3.connect(db) as con:
        con.execute(
            "CREATE TABLE Admin (opt_ids BLOB, opt_poses BLOB, opt_map BLOB, opt_map_x_min FLOAT,"
            " opt_map_y_min FLOAT, opt_map_resolution FLOAT)"
        )
        con.execute(
            "INSERT INTO Admin VALUES (?, ?, ?, ?, ?, ?)",
            (
                compressed(np.array([[7, 9]], dtype=np.int32), 4),
                compressed(poses, 5),
                compressed(cells, 1),
                -4.5,
                -4.9,
                0.05,
            ),
        )
    grid = grid_from_rtabmap_db(db)
    assert np.array_equal(grid.cells, cells)
    assert (grid.origin_x, grid.origin_y, grid.resolution) == pytest.approx((-4.5, -4.9, 0.05))
    nodes = graph_poses_from_rtabmap_db(db)
    assert sorted(nodes) == [7, 9]
    assert (nodes[7].x, nodes[7].y, nodes[7].theta) == pytest.approx((0.5, -0.25, 0.0))
    assert (nodes[9].x, nodes[9].y, nodes[9].theta) == pytest.approx((2.0, 1.0, math.pi / 2))


# -- scenarios, scores and the control socket -----------------------------------------------------


def test_a_scenario_written_from_the_carts_seat_resolves_into_the_map() -> None:
    scenario = Scenario.from_dict(
        {
            "name": "pocket",
            "start": {"x": 1.0, "y": 2.0, "yaw_deg": 90.0},
            "boxes": [{"name": "ahead", "frame": "start", "x": 0.5, "y": 0.0, "length": 0.2,
                       "width": 0.8}],
            "legs": [{"frame": "start", "x": -1.0, "y": 1.0, "yaw_deg": 45.0}, "home"],
            "slow_s": 60,
        }
    )  # fmt: skip
    places = {"home": Pose2D(3.0, 3.0, 0.0)}
    start = scenario.start_pose(places)
    (ahead,) = scenario.resolved_boxes(places)
    assert (ahead.x, ahead.y, ahead.yaw_deg) == pytest.approx((1.0, 2.5, 90.0))
    first = scenario.legs[0].resolve(places, start)
    assert (first.x, first.y, math.degrees(first.theta)) == pytest.approx((0.0, 1.0, 135.0))
    assert scenario.legs[1].resolve(places, start) == places["home"]
    assert scenario.slow_s == 60.0
    with pytest.raises(ValueError):
        Target(place="kitchen").resolve(places)


def test_every_scenario_in_the_repo_parses() -> None:
    files = sorted((REPO / "ros/sim/scenarios").glob("*.yaml"))
    assert files, "ros/sim/scenarios holds the drives the sim is judged on"
    places = places_from_payload((REPO / "ros/sim/worlds/flat.places.json").read_text())
    for path in files:
        scenario = Scenario.load(path)
        start = scenario.start_pose(places)
        for leg in scenario.legs:
            leg.resolve(places, start)
        scenario.resolved_boxes(places)


def test_a_leg_is_scored_on_the_odometers_truth() -> None:
    before = {
        "t": 100.0,
        "x": 0.0,
        "y": 0.0,
        "yaw_deg": 0.0,
        "path_m": 1.0,
        "contacts": 2,
        "blocked_s": 0.5,
    }
    after = {"t": 130.5, "x": 2.95, "y": 0.1, "yaw_deg": 178.0, "path_m": 4.5, "contacts": 3,
             "blocked_s": 1.5}  # fmt: skip
    score = score_leg("far", Pose2D(3.0, 0.0, math.radians(-179.0)), before, after, "SUCCEEDED", 7)
    assert score.seconds == pytest.approx(30.5)
    assert (score.path_m, score.straight_m, score.detour) == pytest.approx((3.5, 3.0, 3.5 / 3.0))
    assert (score.contacts, score.blocked_s) == (1, pytest.approx(1.0))
    assert score.heading_error_deg == pytest.approx(-3.0)  # across the +-180 seam
    assert json.loads(json.dumps(score.to_dict()))["recoveries"] == 7
    assert score.line().startswith("SCORE far: SUCCEEDED in 30.5 s, recoveries 7")


def test_the_world_answers_state_place_and_boxes_on_its_socket() -> None:
    sim = world(empty_grid(80, 80), pose=Pose2D(1.0, 1.0, 0.0))
    sim.base.command(0.2, 0.0, 0.0)
    sim.step(0.1, 0.05)
    placed = sim.answer({"cmd": "place", "x": 2.0, "y": 2.5, "yaw_deg": 90.0}, 5.0)
    assert placed["event"] == "placed" and (placed["x"], placed["yaw_deg"]) == (2.0, 90.0)
    assert sim.base.twist(5.0) == (0.0, 0.0), "a placed cart stands still"
    boxes = sim.answer({"cmd": "boxes", "boxes": [{"name": "c", "x": 3, "y": 3, "length": 0.4,
                                                  "width": 0.4}]}, 5.0)  # fmt: skip
    assert boxes["boxes"] == ["c"]
    assert sim.answer({"cmd": "state"}, 6.0)["path_m"] == pytest.approx(0.02)
    assert sim.answer({"cmd": "place", "x": 1.0}, 6.0)["event"] == "error"
    assert sim.answer({"cmd": "fly"}, 6.0)["event"] == "error"


def test_the_places_payload_reads_back_as_poses() -> None:
    payload = json.dumps({"home": {"x": -0.387, "y": 2.922, "yaw_deg": 96.1, "node": 661}})
    home = places_from_payload(payload)["home"]
    assert (home.x, home.y, math.degrees(home.theta)) == pytest.approx((-0.387, 2.922, 96.1))
