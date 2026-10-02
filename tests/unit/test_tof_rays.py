"""The whiskers write the volume: a ToF fan as a tiny depth image through the camera's own
integrator (pepin.tof_rays), and the weight arithmetic the module promises, proven on the grid
config/fusion.json ships (max_weight 20, min_weight 2.0, 5 cm voxels, 10 cm truncation)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pepin.depth import Intrinsics
from pepin.tof_rays import (
    SILENT_M,
    TOF_WEIGHT,
    fan_image,
    fans_to_clear,
    fans_to_speak,
    optical_pose,
    tof_law,
)
from pepin.tsdf import DepthLaw, GridSpec, RigidPose
from pepin.volume_scan import MarksLaw, marks_ranges
from pepin.worldmap import TOF, WorldMap

SPEC = GridSpec(origin=(-3.0, -3.0, -0.15), shape=(120, 120, 34), max_weight=20.0)
FOV = 0.47  # the VL53L1X cone the bridge draws (pepin_bringup.tof_bridge)
BEAMS = 7  # the low whiskers' fan (pepin.tof_horizon.cone_beams at a 0.57 m ceiling)
HEIGHT_M = 0.16  # the low whiskers' mount (config/tof.json)
REACH_M = 0.57  # their trusted range: the fan's own range_max
HIT_M = 0.4
BASE = RigidPose(np.eye(3), np.zeros(3))  # the cart at the origin, facing +x
SENSOR = RigidPose(np.eye(3), np.array([0.0, 0.0, HEIGHT_M]))  # level, facing +x
LAW = MarksLaw()


def fan(value: float) -> tuple[np.ndarray, float, float]:
    """One fan of ``BEAMS`` equal beams: ``(ranges, angle_min, angle_increment)``."""
    return np.full(BEAMS, value), -FOV / 2.0, FOV / (BEAMS - 1)


def integrate(world: WorldMap, value: float) -> int:
    depth, intr = fan_image(*fan(value))
    return world.integrate_depth(
        depth, None, intr, optical_pose(SENSOR), stamp=1.0, law=tof_law(REACH_M), sensor=TOF
    )


def ahead(world: WorldMap) -> float:
    """What the costmaps' marks read inside the whisker's cone ahead of the cart — the nearest
    range over the bearings within fov / 2 of bearing 0 — metres; NaN for nothing. (The fan
    starts at -pi, so bearing 0 is the middle bin; the crossings sit at voxel centres, a
    half-voxel either side of the axis, never on the one bin at exactly zero.)"""
    ranges = marks_ranges(world.volume, BASE, LAW)
    half = round((FOV / 2) / LAW.step)
    cone = ranges[LAW.bins // 2 - half : LAW.bins // 2 + half + 1]
    return float(np.nanmin(cone)) if np.isfinite(cone).any() else math.nan


def spot(world: WorldMap) -> tuple[int, int, int]:
    """The voxel the hit lands in: on the sensor's axis, at its own height."""
    idx, inside = world.volume.voxel_of(np.array([[HIT_M, 0.0, HEIGHT_M]]))
    assert inside[0]
    return int(idx[0, 0]), int(idx[0, 1]), int(idx[0, 2])


def test_a_fan_is_a_disc_of_one_range_seen_through_a_pinhole() -> None:
    ranges, angle_min, inc = fan(HIT_M)
    depth, intr = fan_image(ranges, angle_min, inc)
    assert depth.shape == (BEAMS, BEAMS) and (intr.width, intr.height) == (BEAMS, BEAMS)
    centre = (BEAMS - 1) / 2
    assert depth[int(centre), int(centre)] == HIT_M
    assert depth[0, 0] == SILENT_M, "a corner of the square lies outside the cone"
    inside = depth == HIT_M
    assert 0.6 < inside.mean() < 0.9, "a disc, not the whole square"
    # the pixels' outer edges are the cone's edges: half the image width over f is tan(fov / 2)
    assert math.atan((BEAMS / 2) / intr.fx) == pytest.approx(FOV / 2)
    miss, _ = fan_image(*fan(math.inf))
    assert np.isnan(miss[int(centre), int(centre)]), "+inf is a depthless pixel: it carves"
    silent, _ = fan_image(*fan(math.nan))
    assert silent[int(centre), int(centre)] == SILENT_M, "NaN says nothing"


def test_a_hit_marks_the_sensor_s_height_once_two_fans_agree() -> None:
    world = WorldMap(SPEC)
    speak = fans_to_speak(LAW.min_weight)
    assert speak == 2  # the test law's min_weight 2.0 at TOF_WEIGHT 1.0
    for _ in range(speak - 1):
        integrate(world, HIT_M)
    assert math.isnan(ahead(world)), "three fans are not yet agreement (min_weight 2.0)"
    integrate(world, HIT_M)
    assert ahead(world) == pytest.approx(HIT_M, abs=SPEC.voxel_m)
    ix, iy, iz = spot(world)
    assert world.volume.weight[ix, iy, iz] == pytest.approx(speak * TOF_WEIGHT)
    assert abs(float(world.volume.sdf[ix, iy, iz])) < 0.5, "the surface is in that voxel"
    assert world.frames[-1][1] == TOF


def test_a_hit_weighs_the_same_near_and_far() -> None:
    """A ranger's error does not shrink with distance: (ref / d)^2 would weigh a 0.3 m return
    at the cap, 4.0, and five fans would freeze the voxel."""
    for value in (0.3, 0.45):
        world = WorldMap(SPEC)
        integrate(world, value)
        idx, _ = world.volume.voxel_of(np.array([[value, 0.0, HEIGHT_M]]))
        assert world.volume.weight[tuple(idx[0])] == pytest.approx(TOF_WEIGHT)


def test_misses_carve_a_saturated_hit_within_two_seconds() -> None:
    """40 fans saturate the pillow (20 / 0.5); then the pillow is gone and the fan says +inf.
    The marks stop reading it within fans_to_clear misses — 15 at W 20, w 1.0 — which is under
    the 30 fans two seconds hold at 15 Hz."""
    world = WorldMap(SPEC)
    for _ in range(40):
        integrate(world, HIT_M)
    ix, iy, iz = spot(world)
    assert world.volume.weight[ix, iy, iz] == SPEC.max_weight, "saturated"
    promised = fans_to_clear(SPEC.max_weight)
    assert promised == 15 and promised <= 30
    misses = 0
    while not math.isnan(ahead(world)):
        integrate(world, math.inf)
        misses += 1
        assert misses <= promised, "the arithmetic in pepin.tof_rays must hold on the grid"
    assert misses >= 2, "and a single miss does not erase a saturated hit"
    # the formula itself, against a scalar walk of the integration law
    t, w, n = -1.0, TOF_WEIGHT, 0
    while t <= 0.0:
        t = (t * SPEC.max_weight + 1.0 * w) / (SPEC.max_weight + w)
        n += 1
    assert n == promised


def test_the_camera_s_own_rays_carve_the_whisker_s_mark() -> None:
    """A pillow the ToF saturated, then carried away: the camera looking through the spot at
    the wall 2 m behind carves it at its own weight (1.0 per frame at 2 m), with the shipped
    law — no depthless carving needed, the pixels measured something."""
    world = WorldMap(SPEC)
    for _ in range(40):
        integrate(world, HIT_M)
    assert ahead(world) == pytest.approx(HIT_M, abs=SPEC.voxel_m)
    intr = Intrinsics(fx=30.0, fy=30.0, cx=16.0, cy=12.0, width=32, height=24)
    wall = np.full((24, 32), 2.0)
    camera = optical_pose(SENSOR)  # the camera standing where the whisker is, for simplicity
    frames = 0
    while ahead(world) == pytest.approx(HIT_M, abs=SPEC.voxel_m):
        world.integrate_depth(wall, None, intr, camera, law=DepthLaw())
        frames += 1
        assert frames <= fans_to_clear(SPEC.max_weight, 1.0)
    assert fans_to_clear(SPEC.max_weight, 1.0) == 15


def farthest_written_x(world: WorldMap) -> float:
    written = world.volume.weight > 0.0
    assert written.any()
    far = np.flatnonzero(written.any(axis=(1, 2))).max()
    return float((far + 0.5) * SPEC.voxel_m) + SPEC.origin[0]


def test_a_silent_fan_writes_nothing_and_a_miss_carves_to_the_reach_and_not_beyond() -> None:
    world = WorldMap(SPEC)
    assert integrate(world, math.nan) == 0
    assert float(world.volume.weight.sum()) == 0.0
    integrate(world, math.inf)
    assert REACH_M - SPEC.voxel_m < farthest_written_x(world) <= REACH_M
    carved = world.volume.weight > 0.0
    assert (world.volume.sdf[carved] == 1.0).all(), "free space, a whole truncation from anything"


def test_a_hit_near_the_reach_leaves_no_halo_the_misses_cannot_carve() -> None:
    """A return at 0.52 m with a 0.57 m ceiling: its back halo would reach 0.62 m, past where
    any miss carves, and the mark would migrate there and stay for ever. The law writes the
    hit no farther than the reach, so the same 29 misses clear it."""
    near_reach = REACH_M - SPEC.voxel_m
    world = WorldMap(SPEC)
    for _ in range(40):
        integrate(world, near_reach)
    assert farthest_written_x(world) <= REACH_M
    assert ahead(world) == pytest.approx(near_reach, abs=SPEC.voxel_m)
    misses = 0
    while not math.isnan(ahead(world)):
        integrate(world, math.inf)
        misses += 1
        assert misses <= fans_to_clear(SPEC.max_weight)


def test_a_hand_fifteen_centimetres_ahead_is_a_mark() -> None:
    """The ToF's law cuts at its own near limit (0.08 m), not the camera's 0.20: a return at
    0.15 m writes a surface; at 0.06 m, inside the sensor's dead zone, nothing."""
    from pepin.tof_rays import NEAR_M, fan_image, optical_pose, tof_law
    from pepin.tsdf import RigidPose, Tsdf

    assert tof_law(0.96).near_m == NEAR_M == 0.08
    pose = optical_pose(RigidPose(np.eye(3), np.array([0.027, 0.0, 0.27])))
    for range_m, expected in ((0.15, True), (0.06, False)):
        volume = Tsdf(SPEC)
        depth, intr = fan_image(np.array([range_m] * 3), -0.235, 0.235)
        for _ in range(4):
            volume.integrate(depth.astype(np.float32), None, intr, pose, tof_law(0.96))
        points, _ = volume.surface(min_weight=2.0)
        assert (len(points) > 0) == expected, range_m
