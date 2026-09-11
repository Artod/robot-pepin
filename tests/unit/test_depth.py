"""The lidar sets the law of the camera's depth: projection, the pairs, the pooled fit, the
scan the costmap reads, the floor anchor and the tilt behind it, the saved law."""

import math
from pathlib import Path

import numpy as np
import pytest

from pepin.depth import (
    MIN_SAMPLES,
    CameraPose,
    Intrinsics,
    project,
    scan_points,
    to_base,
)

INTR = Intrinsics(fx=457.0, fy=457.0, cx=320.0, cy=180.0, width=640, height=360)
CAM = CameraPose(x=0.0, y=0.0, z=1.23)


def test_a_point_on_the_optical_axis_lands_on_the_principal_point() -> None:
    hit = project(np.array([[2.0, 0.0, 1.23]]), CAM, INTR)
    assert hit.shape == (1, 3)
    assert hit[0, 0] == pytest.approx(320.0) and hit[0, 1] == pytest.approx(180.0)
    assert hit[0, 2] == pytest.approx(2.0)


def test_the_lidar_plane_appears_low_in_the_image_and_only_far_enough_away() -> None:
    """15 cm above the floor, 1.08 m below a level camera: at 3 m it is 165 px below the
    centre; at 2 m it falls out of a 360-px-high image (a tilt down would bring it in)."""
    far = project(np.array([[3.0, 0.0, 0.15]]), CAM, INTR)
    assert far[0, 1] == pytest.approx(180.0 + 457.0 * 1.08 / 3.0)
    near = project(np.array([[2.0, 0.0, 0.15]]), CAM, INTR)
    assert near.shape[0] == 0
    tilted = project(
        np.array([[2.0, 0.0, 0.15]]), CameraPose(0.0, 0.0, 1.23, math.radians(15)), INTR
    )
    assert tilted.shape[0] == 1 and 180.0 < tilted[0, 1] < 360.0


def test_left_is_left_and_behind_is_dropped() -> None:
    left = project(np.array([[2.0, 0.5, 1.23]]), CAM, INTR)
    assert left[0, 0] < 320.0  # a point to the robot's left is on the left of the image
    assert project(np.array([[-1.0, 0.0, 1.23]]), CAM, INTR).shape[0] == 0


def test_scan_points_and_the_lidar_mount() -> None:
    ranges = np.array([1.0, np.inf, 2.0, 0.0])
    xy = scan_points(ranges, angle_min=0.0, angle_increment=math.pi / 2, range_max=6.0)
    assert xy.shape == (2, 2)
    assert xy[0] == pytest.approx([1.0, 0.0]) and xy[1] == pytest.approx([-2.0, 0.0], abs=1e-9)
    base = to_base(xy, np.eye(3), np.array([0.10, 0.0, 0.15]))
    assert base[0] == pytest.approx([1.10, 0.0, 0.15])


def test_a_quaternion_becomes_the_rotation_tf_means() -> None:
    from pepin.depth import rotation_matrix

    assert rotation_matrix(0.0, 0.0, 0.0, 1.0) == pytest.approx(np.eye(3))
    quarter = rotation_matrix(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    assert quarter @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])  # +90 deg yaw


def test_a_rotation_survives_the_trip_through_a_quaternion_and_inverts() -> None:
    from pepin.depth import invert, quaternion_from_matrix, rotation_matrix

    q = (0.1, -0.2, 0.3, math.sqrt(1 - 0.14))
    rot = rotation_matrix(*q)
    assert quaternion_from_matrix(rot) == pytest.approx(q, abs=1e-9)
    back_rot, back_t = invert(rot, np.array([1.0, 2.0, 3.0]))
    assert back_rot @ rot == pytest.approx(np.eye(3), abs=1e-12)
    assert back_rot @ np.array([1.0, 2.0, 3.0]) + back_t == pytest.approx([0.0, 0.0, 0.0])


def _tilted_room(cam: CameraPose) -> tuple[np.ndarray, np.ndarray]:
    """(wall, floor): the optical depth at which each pixel's ray of a 640x360 image meets a
    wall two metres ahead and the floor (inf above the horizon)."""
    rows, _cols = np.mgrid[0:360, 0:640]
    up = -(rows + 0.5 - INTR.cy) / INTR.fy  # each ray in camera_link, per unit optical depth
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    fwd_base = c * 1.0 + s * up  # base x per unit optical depth
    up_base = -s * 1.0 + c * up
    wall = 2.0 / fwd_base
    with np.errstate(divide="ignore"):
        floor = -cam.z / up_base
    floor[up_base >= 0] = np.inf
    return wall, floor


def test_the_depth_image_becomes_a_scan_of_what_stands_above_the_floor() -> None:
    """A tilted camera looking at a wall two metres ahead: the wall's pixels between 8 cm and
    1.3 m mark the central bearings at 2 m, the floor pixels below it mark nothing."""
    from pepin.depth import depth_to_scan

    cam = CameraPose(0.0, 0.0, 1.23, math.radians(28.0))
    wall, floor = _tilted_room(cam)
    depth = np.where(wall < floor, wall, floor)
    angle_min, step, ranges = depth_to_scan(depth, INTR, cam)
    centre = round((0.0 - angle_min) / step)
    assert ranges[centre] == pytest.approx(2.0, abs=0.02)
    assert ranges[centre - 20] == pytest.approx(2.0 / math.cos(20 * step), abs=0.03)
    assert np.isfinite(ranges).sum() > 100  # the wall spans most of the view


def test_a_bearing_with_no_finite_pixel_is_unknown_and_one_with_only_floor_is_clear() -> None:
    """Nav2 reads inf as "clear out to the raytrace range" (inf_is_valid) and drops NaN: a
    frame the law could not place must not wipe the costmap, while a bearing where the camera
    saw only floor really is clear."""
    from pepin.depth import depth_to_scan

    cam = CameraPose(0.0, 0.0, 1.23, math.radians(28.0))
    unknown = depth_to_scan(np.full((360, 640), np.nan), INTR, cam)[2]
    assert np.isnan(unknown).all()
    _wall, floor = _tilted_room(cam)
    angle_min, step, clear = depth_to_scan(floor, INTR, cam)  # nothing but floor and sky
    centre = round((0.0 - angle_min) / step)
    assert np.isinf(clear[centre]) and not np.isfinite(clear).any()
    assert np.isinf(clear).sum() > 100  # the floor's near pixels reach every bearing of the fan
    half = floor.copy()
    half[:, :320] = np.nan  # the left half of the picture unplaceable: its bearings unknown
    blind = depth_to_scan(half, INTR, cam)[2]
    assert np.isnan(blind[centre + 20]) and np.isinf(blind[centre - 20])
    assert np.isnan(blind).sum() > 30 and np.isinf(blind).sum() > 30
    # a table top 1.5 m ahead in front of the same floor marks its bearings and nothing else
    wall, _ = _tilted_room(cam)
    table = np.where(wall * 0.75 < floor, wall * 0.75, floor)
    marked = depth_to_scan(table, INTR, cam)[2]
    assert marked[centre] == pytest.approx(1.5, abs=0.02)
    assert not np.isnan(marked[np.isinf(clear)]).any()  # the floor's bearings stay clear


def test_the_floor_s_depth_follows_the_camera_s_height_tilt_and_the_cart_s_lean() -> None:
    """A camera 1.23 m up, tilted 26 degrees down: the ray through the principal point meets the
    floor 1.23 / sin(26 deg) along the axis; the cart pitched 5 degrees nose-down brings it to
    1.23 / sin(31 deg); rows at and above the horizon never meet the floor."""
    from pepin.depth import floor_depth

    intr = Intrinsics(fx=400.0, fy=400.0, cx=320.0, cy=180.0, width=640, height=360)
    cam = CameraPose(0.0, 0.0, 1.23, math.radians(26.0))
    level = floor_depth(intr, cam)
    assert level[180, 320] == pytest.approx(1.23 / math.sin(math.radians(26.0)), rel=1e-6)
    assert level[359, 320] < level[180, 320]  # the bottom row looks at nearer floor
    # a shallower tilt puts the horizon inside the picture: above it no ray meets the floor
    shallow = floor_depth(intr, CameraPose(0.0, 0.0, 1.23, math.radians(10.0)))
    horizon_row = 180 - math.tan(math.radians(10.0)) * 400  # where the ray runs level
    assert np.isnan(shallow[int(horizon_row) - 5, 320]) and np.isfinite(
        shallow[int(horizon_row) + 5, 320]
    )
    nose_down = np.array([-math.sin(math.radians(5.0)), 0.0, math.cos(math.radians(5.0))])
    leaning = floor_depth(intr, cam, up=nose_down)
    # the camera stands 1.23 cos 5 deg above the leaning plane and the ray meets it at 31 deg
    expected = 1.23 * math.cos(math.radians(5.0)) / math.sin(math.radians(31.0))
    assert leaning[180, 320] == pytest.approx(expected, rel=1e-6)


def test_the_floor_anchor_is_a_height_test_so_a_shoe_stays_a_shoe() -> None:
    """A point at depth d on a ray that meets the floor at E stands h (1 - d / E) above it:
    2.5 cm of height snaps to the floor, a 5 cm shoe and a box half-way up the ray do not — at
    any range, where the old 15 % of depth was 18 cm and swallowed the shoe."""
    from pepin.depth import FLOOR_HEIGHT_TOLERANCE, floor_anchor, floor_depth

    intr = Intrinsics(fx=400.0, fy=400.0, cx=320.0, cy=180.0, width=640, height=360)
    cam = CameraPose(0.0, 0.0, 1.23, math.radians(26.0))
    level = floor_depth(intr, cam)
    guess = level * (1.0 - 0.025 / 1.23)  # the network sees the floor 2.5 cm high everywhere
    guess[300:340, 200:260] = level[300:340, 200:260] * 0.5  # a box half-way to the floor
    guess[100:140, 400:460] = level[100:140, 400:460] * (1.0 - 0.05 / 1.23)  # a shoe, far
    guess[:50, :] = np.nan
    fixed, anchored = floor_anchor(guess, level, cam.z)
    assert FLOOR_HEIGHT_TOLERANCE == 0.04
    assert anchored > 100_000
    assert np.allclose(fixed[350, 100:600], level[350, 100:600])
    assert np.allclose(fixed[320, 230], guess[320, 230])  # the box is left to the network
    assert np.allclose(fixed[120, 430], guess[120, 430])  # so is the shoe
    assert level[120, 430] > 3.0  # and the shoe is far: 5 cm at 3 m is 1.6 % of depth


def _chip_reading(pitch_deg: float) -> np.ndarray:
    """The MPU6050's accelerometer (Y up, roll +90 mount) when the cart is nose down by
    ``pitch_deg``: gravity leans onto the chip's z (base x)."""
    from pepin.depth import GRAVITY

    p = math.radians(pitch_deg)
    return np.array([-GRAVITY * math.sin(p), GRAVITY * math.cos(p), 0.0])


def test_the_tilt_reads_gravity_through_the_mount_and_ignores_a_bump_and_a_push() -> None:
    """Level at the first sample; a bump (not 1 g) is ignored; a push (the cart accelerating
    leans the apparent gravity 3 degrees at the same 1 g) is ignored for as long as any push
    lasts; the same lean outlasting the time constant is a slope and the up vector follows."""
    from pepin.depth import GRAVITY, TILT_TAU_S, Tilt, imu_mount_rotation

    tilt = Tilt(imu_mount_rotation(90.0, 0.0, 0.0))
    tilt.observe(np.array([0.0, GRAVITY, 0.0]), 0.0)  # the chip's Y up: level
    assert np.allclose(tilt.up, [0.0, 0.0, 1.0])
    assert tilt.roll_pitch_deg == pytest.approx((0.0, 0.0), abs=1e-9)
    tilt.observe(np.array([0.0, GRAVITY, 6.0]), 0.5)  # a bump: not 1 g, ignored
    assert np.allclose(tilt.up, [0.0, 0.0, 1.0])
    t = 0.5
    while t < 3.0:  # a push: 3 degrees for 2.5 s
        t += 0.1
        tilt.observe(_chip_reading(3.0), t)
    assert tilt.roll_pitch_deg[1] == pytest.approx(0.0, abs=1e-9)
    while t < 3.0 + TILT_TAU_S + 0.5:  # the lean persists past the time constant: a slope
        t += 0.1
        tilt.observe(_chip_reading(3.0), t)
    assert tilt.roll_pitch_deg[1] == pytest.approx(3.0, abs=1e-6)


def test_a_small_lean_is_followed_slowly_and_a_nan_sample_is_ignored() -> None:
    """Half a degree is inside the gate and is low-passed with the 10 s time constant: after
    30 s the filter has come 95 % of the way. A NaN reading passes no gate and leaves the up
    vector finite and unchanged."""
    from pepin.depth import GRAVITY, TILT_TAU_S, Tilt, imu_mount_rotation

    tilt = Tilt(imu_mount_rotation(90.0, 0.0, 0.0))
    tilt.observe(np.array([0.0, GRAVITY, 0.0]), 0.0)
    for i in range(1, 301):
        tilt.observe(_chip_reading(0.5), 0.1 * i)
    assert tilt.roll_pitch_deg[1] == pytest.approx(
        0.5 * (1 - math.exp(-30.0 / TILT_TAU_S)), abs=0.01
    )
    before = tilt.up.copy()
    tilt.observe(np.array([np.nan, GRAVITY, 0.0]), 31.0)
    assert np.array_equal(tilt.up, before) and np.isfinite(tilt.up).all()


def test_a_dead_accelerometer_leaves_the_lean_alone_instead_of_dividing_by_its_zero() -> None:
    """A bridge that publishes zeros (the chip unplugged, a read that failed) must not turn the
    up vector into NaN — every pixel of the floor would then stop being the floor."""
    from pepin.depth import GRAVITY, Tilt, imu_mount_rotation

    tilt = Tilt(imu_mount_rotation(90.0, 0.0, 0.0))
    tilt.observe(np.array([0.0, GRAVITY, 0.0]), 0.0)
    for i in range(10):
        tilt.observe(np.zeros(3), 0.1 * (i + 1))
    assert np.allclose(tilt.up, [0.0, 0.0, 1.0])
    assert np.isfinite(tilt.up).all()


def test_the_imu_mount_of_the_config_is_the_rotation_the_cpp_bridge_applies() -> None:
    """config/imu.json says roll +90 deg; base_bridge.cpp's to_base_axes for a Y-up chip maps
    base (x, y, z) <- chip (x, -z, y). The two must agree, or a reading published in base_link
    would be rotated twice."""
    import json

    from pepin.depth import imu_mount_rotation

    mount = json.loads((Path(__file__).parents[2] / "config/imu.json").read_text())["mount"]
    rot = imu_mount_rotation(mount["roll_deg"], mount["pitch_deg"], mount["yaw_deg"])
    chip = np.array([1.0, 2.0, 3.0])
    assert rot @ chip == pytest.approx([1.0, -3.0, 2.0])


def test_a_floor_the_camera_cannot_see_anchors_nothing() -> None:
    """Pointed at the ceiling no ray meets the floor: the expected image is all NaN and the
    network's depth must come back untouched."""
    from pepin.depth import floor_anchor

    guess = np.full((8, 8), 1.5)
    fixed, anchored = floor_anchor(guess, np.full((8, 8), np.nan), 1.23)
    assert anchored == 0 and np.array_equal(fixed, guess)


def test_a_stale_scan_is_carried_to_the_frame_s_moment() -> None:
    """The cart turned 2 degrees left between the scan and the frame: a point dead ahead at 2 m
    at the scan's moment sits 2 degrees to the right at the frame's moment."""
    from pepin.depth import carry, rotation_matrix

    ahead = np.array([[2.0, 0.0, 0.2]])
    yaw = math.radians(-2.0)  # base_link@frame <- base_link@scan: the world turned right
    rot = rotation_matrix(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    moved = carry(ahead, rot, np.zeros(3))
    assert moved[0, 1] == pytest.approx(-2.0 * math.sin(math.radians(2.0)), rel=1e-6)
    assert moved[0, 2] == 0.2


def test_the_frame_takes_the_scan_nearest_its_exposure_not_the_newest() -> None:
    """The network answers 0.25 s after the picture: the newest scan is younger than the frame,
    the right one was taken at the exposure; nothing within half a second means no verdict."""
    from pepin.depth import SCAN_MAX_AGE_S, nearest_stamp

    stamps = [10.0, 10.1, 10.2, 10.3]  # the last one arrived while the network ran
    assert nearest_stamp(stamps, 10.06) == 1
    assert nearest_stamp(stamps, 10.3 + SCAN_MAX_AGE_S + 0.01) is None
    assert nearest_stamp([], 10.0) is None


def test_edge_pixels_are_the_blurred_outline_of_an_object_and_the_border() -> None:
    """A box at 1 m in front of a wall at 2 m: the pixels on both sides of its outline are
    edges, the flat inside of either surface is not, a NaN neighbour makes an edge, and the
    image's border rows and columns are always edges; the mask is the same after a scale."""
    from pepin.depth import drop_edges, edge_mask

    d = np.full((12, 12), 2.0)
    d[4:8, 4:8] = 1.0
    d[2, 9] = np.nan
    edge = edge_mask(d)
    assert edge[0].all() and edge[-1].all() and edge[:, 0].all() and edge[:, -1].all()
    assert edge[3, 5] and edge[4, 5] and edge[7, 5] and edge[8, 5]  # both sides of the outline
    assert not edge[5:7, 5:7].any() and not edge[2, 2] and not edge[9, 2]
    assert edge[2, 9] and edge[1, 9] and edge[3, 9] and edge[2, 8] and edge[2, 10]
    assert np.array_equal(edge_mask(d * 0.63), edge)
    dropped, n = drop_edges(d, edge)
    assert n == int(edge.sum()) and np.isnan(dropped[4, 5]) and dropped[5, 5] == 1.0


def _frame(
    z_lo: float, z_hi: float, a: float, b: float, n: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """A depth image seen through the law 1/z = a/D + b along one row, and the beams that hit
    it (column, row, true depth) spanning ``z_lo``..``z_hi``."""
    z_true = np.linspace(z_lo, z_hi, n)
    d_net = 1.0 / ((1.0 / z_true - b) / a)
    depth = np.tile(d_net, (10, 1))
    samples = np.stack([np.arange(n, dtype=float), np.full(n, 5.0), z_true], axis=1)
    return depth, samples


def test_the_beams_pooled_over_frames_fit_the_network_s_depth_as_an_affine_law() -> None:
    """The network sees the room too far, the far end more than the near: true 1/z = 1.2/D +
    0.05. One frame's beams span 1.3-2 m and fit a scale only; six frames pooled span 1-4 m
    and recover both numbers; a frame without beams keeps the law."""
    from pepin.depth import AffineScale, apply_affine, beam_pairs, fit_affine

    a_true, b_true = 1.2, 0.05
    depth, samples = _frame(1.3, 2.0, a_true, b_true)
    pairs = beam_pairs(depth, samples)
    assert pairs is not None and pairs[0].size == 40
    assert fit_affine(*pairs)[1] == 0.0  # too narrow a spread for a shift
    law = AffineScale()
    assert not law.ready  # nothing pooled: the raw network must not be published
    for lo, hi in ((1.0, 1.6), (1.4, 2.2), (2.0, 3.0), (2.8, 4.0), (1.0, 4.0), (1.2, 3.5)):
        law.observe(beam_pairs(*_frame(lo, hi, a_true, b_true)))
    assert law.ready and law.fitted
    assert law.a == pytest.approx(a_true, rel=1e-3) and law.b == pytest.approx(b_true, abs=1e-3)
    assert law.pooled == 240
    held = law.observe(None)
    assert held == (law.a, law.b) and law.held == 1
    corrected = apply_affine(_frame(1.0, 4.0, a_true, b_true)[0], *held)
    assert np.allclose(corrected[5], np.linspace(1.0, 4.0, 40), atol=0.01)
    assert beam_pairs(depth, samples[: MIN_SAMPLES - 1]) is None


def test_beams_landing_on_edge_pixels_stay_out_of_the_fit() -> None:
    """A beam on a blurred edge pairs the lidar with a depth between two surfaces; the mask
    of the raw depth removes those pairs, and too few clean pairs are no verdict."""
    from pepin.depth import beam_pairs, edge_mask

    depth, samples = _frame(1.0, 4.0, 1.2, 0.05)
    depth[:, 20] *= 1.5  # a step: column 20 and its neighbours are edges
    edge = edge_mask(depth)
    clean = beam_pairs(depth, samples, edge)
    assert clean is not None and clean[0].size == 40 - int(edge[5, :].sum())
    assert not np.isin(clean[1], samples[19:22, 2]).any()
    # 23 beams minus the border column and the step's three: 19, one short of MIN_SAMPLES
    assert beam_pairs(depth, samples[:23], edge) is None
    assert beam_pairs(depth, samples[:24], edge) is not None


def test_the_fit_regresses_the_noisy_network_on_the_exact_lidar() -> None:
    """Least squares puts all the noise in the response: with the network's 1/D noisy and the
    lidar's 1/z exact, regressing 1/z on 1/D attenuates the slope (regression dilution) while
    regressing 1/D on 1/z and inverting recovers the law."""
    from pepin.depth import fit_affine

    a_true, b_true = 1.2, 0.05
    rng = np.random.default_rng(0)
    z = np.linspace(1.0, 4.0, 400)
    d = 1.0 / ((1.0 / z - b_true) / a_true) * (1.0 + 0.10 * rng.standard_normal(z.size))
    a, b = fit_affine(d, z)
    assert a == pytest.approx(a_true, rel=0.03) and b == pytest.approx(b_true, abs=0.02)
    a_diluted, _ = np.polyfit(1.0 / d, 1.0 / z, 1)  # the other way round
    assert abs(a_diluted - a_true) > 2 * abs(a - a_true)


def test_one_far_reflection_does_not_open_the_shift_on_a_near_pool() -> None:
    """300 beams at 1.3-2 m and five at 6 m: min / max says x4.6, the 5th / 95th percentiles
    say x1.5 — the shift stays off."""
    from pepin.depth import MIN_DEPTH_SPREAD, fit_affine

    z = np.concatenate([np.linspace(1.3, 2.0, 300), np.full(5, 6.0)])
    d = 1.0 / ((1.0 / z - 0.05) / 1.2)
    assert z.max() / z.min() > MIN_DEPTH_SPREAD
    assert fit_affine(d, z)[1] == 0.0


def test_a_binding_bound_refits_the_other_parameter_instead_of_clipping_both() -> None:
    """A pool whose joint fit says a 0.2, b 0.1: a binds at 0.3, and b is refitted with a
    fixed (the median residual) — the law then reproduces the pool's middle depth exactly,
    where the independently clipped (0.3, 0.1) would have squeezed 2.5 m to 1.8 m."""
    from pepin.depth import A_BOUNDS, B_BOUNDS, apply_affine, fit_affine

    z = np.linspace(1.0, 4.0, 301)
    d = 1.0 / ((1.0 / z - 0.1) / 0.2)
    a, b = fit_affine(d, z)
    assert a == A_BOUNDS[0] and B_BOUNDS[0] <= b <= B_BOUNDS[1]
    assert b == pytest.approx(float(np.median(1.0 / z - a / d)))
    refit = np.abs(apply_affine(d, a, b) - z)
    clipped = np.abs(apply_affine(d, a, 0.1) - z)
    assert np.median(refit) < np.median(clipped) and refit[150] < 1e-9


def test_the_law_is_bounded_and_the_pool_forgets_old_frames() -> None:
    from pepin.depth import A_BOUNDS, B_BOUNDS, AffineScale, fit_affine

    d = np.linspace(1.0, 4.0, 300)
    a, b = fit_affine(d, d / 10.0)  # a network ten times too far: clipped to the bound
    assert a == A_BOUNDS[1] and B_BOUNDS[0] <= b <= B_BOUNDS[1]
    law = AffineScale(pool_frames=2)
    for _ in range(3):
        law.observe((d, d))
    assert (
        law.pooled == 600 and law.a == pytest.approx(1.0) and law.b == pytest.approx(0.0, abs=1e-9)
    )


def test_a_saved_law_is_applied_at_once_and_holds_until_the_live_pool_can_replace_it() -> None:
    from pepin.depth import POOL_MIN_SAMPLES, AffineScale, beam_pairs

    law = AffineScale()
    law.seed(1.4, 0.01)
    assert law.ready and not law.fitted and law.observe(None) == (1.4, 0.01)
    frames = [beam_pairs(*_frame(lo, hi, 1.2, 0.05)) for lo, hi in ((1.0, 2.0), (2.0, 4.0))]
    for pairs in frames * 2:  # 160 pairs: the seed still rules
        assert law.observe(pairs) == (1.4, 0.01)
    assert law.pooled < POOL_MIN_SAMPLES
    law.observe(frames[0])  # 200: the live pool takes over
    assert (
        law.fitted
        and law.a == pytest.approx(1.2, rel=1e-3)
        and law.b == pytest.approx(0.05, abs=1e-3)
    )


def test_the_law_is_saved_atomically_and_loaded_only_while_fresh_and_plausible(
    tmp_path: Path,
) -> None:
    from pepin.depth import LAW_MAX_AGE_S, load_law, save_law

    path = tmp_path / "depth_law.json"
    assert load_law(path, now=1000.0) is None  # no file
    save_law(path, 1.45, 0.012, 3200, now=1000.0)
    assert not (tmp_path / "depth_law.json.tmp").exists()
    assert load_law(path, now=1000.0 + 3600.0) == (1.45, 0.012, 3200)
    assert load_law(path, now=1000.0 + LAW_MAX_AGE_S + 1.0) is None  # another day's room
    save_law(path, 1.45, 0.012, 50, now=1000.0)
    assert load_law(path, now=1000.0) is None  # too few beams behind it
    save_law(path, 9.0, 0.0, 3200, now=1000.0)
    assert load_law(path, now=1000.0) is None  # outside the bounds
    path.write_text("{not json")
    assert load_law(path, now=1000.0) is None


def test_a_sideways_imu_mount_reads_as_a_roll_of_ninety_degrees_and_no_floor() -> None:
    """The chip's Y up through an identity mount (as if config/imu.json said the chip were
    not turned): the up vector lands on base_link's y, a 90 degree roll, and a floor
    perpendicular to that passes through the wheels edge-on — no ray meets it, every pixel's
    floor depth is NaN. The mount's roll +90 is what turns this into a level floor."""
    from pepin.depth import GRAVITY, Tilt, floor_depth

    tilt = Tilt(np.eye(3))
    tilt.observe(np.array([0.0, GRAVITY, 0.0]), 0.0)
    assert tilt.up == pytest.approx([0.0, 1.0, 0.0])
    assert tilt.roll_pitch_deg == pytest.approx((90.0, 0.0))
    cam = CameraPose(0.0, 0.0, 1.23, math.radians(26.0))
    assert np.isnan(floor_depth(INTR, cam, up=tilt.up)).all()


def test_the_edge_mask_of_a_five_by_five_is_the_spike_its_cross_and_the_border() -> None:
    """One pixel twice as far as its neighbours: it is an edge, so are the four pixels beside
    it (their step against it is 100 %), the diagonals are not, the border always is. A NaN in
    the same place marks exactly the same cross."""
    from pepin.depth import edge_mask

    expected = np.ones((5, 5), dtype=bool)
    expected[1:4, 1:4] = False
    expected[2, 1:4] = expected[1:4, 2] = True
    spike = np.ones((5, 5))
    spike[2, 2] = 2.0
    assert np.array_equal(edge_mask(spike), expected)
    hole = np.ones((5, 5))
    hole[2, 2] = np.nan
    assert np.array_equal(edge_mask(hole), expected)


def _blank() -> np.ndarray:
    """A 640x360 depth image with nothing placed in it."""
    return np.full((360, 640), np.nan)


def test_the_scan_keeps_right_on_the_right_and_a_lone_pixel_marks_nothing() -> None:
    """A post at (1, -0.5) in base_link — ahead and to the RIGHT — is the scan's bearing of
    -26.6 degrees (bin -26.5) at 1.12 m, and the mirror bearing on the left stays unknown; one
    flying pixel dead ahead clears its bearing (it was seen) but marks nothing: SCAN_KTH
    pixels must agree before a bearing carries a range."""
    from pepin.depth import SCAN_KTH, depth_to_scan

    cam = CameraPose(0.0, 0.0, 0.5)  # level, half a metre up: the post's column is in the band
    post = _blank()
    post[:, 548] = 1.0  # u = 548.5: left = -(548.5 - 320) / 457 m at 1 m, half a metre right
    angle_min, step, ranges = depth_to_scan(post, INTR, cam)
    right = round((math.atan2(-0.5, 1.0) - angle_min) / step)
    assert math.degrees(angle_min + right * step) == pytest.approx(-26.5)
    assert ranges[right] == pytest.approx(math.hypot(1.0, 0.5), abs=0.01)
    assert np.isnan(ranges[round((math.atan2(0.5, 1.0) - angle_min) / step)])
    lone = _blank()
    lone[180, 320] = 2.0
    ranges = depth_to_scan(lone, INTR, cam)[2]
    assert np.isinf(ranges[round((0.0 - angle_min) / step)])
    assert not np.isfinite(ranges).any() and SCAN_KTH == 3


def test_far_and_high_pixels_clear_their_bearing_and_unseen_bearings_stay_unknown() -> None:
    """A column at 5 m (past max_range) dead ahead and a column standing 1.5 m up (past max_z)
    ahead-right: both bearings were looked at and read inf, clear that far; every bearing no
    finite pixel fell on stays NaN, which the costmap neither marks nor clears."""
    from pepin.depth import depth_to_scan

    cam = CameraPose(0.0, 0.0, 1.23)
    depth = _blank()
    depth[:, 320] = 5.0
    depth[:60, 548] = 1.0  # rows 0-59 at 1 m: 1.50-1.62 m above the floor
    angle_min, step, ranges = depth_to_scan(depth, INTR, cam)
    centre = round((0.0 - angle_min) / step)
    right = round((math.atan2(-0.5, 1.0) - angle_min) / step)
    assert np.isinf(ranges[centre]) and np.isinf(ranges[right])
    assert not np.isfinite(ranges).any()
    assert np.isnan(ranges).sum() == ranges.size - 2


def test_the_pool_forgets_a_view_that_lied() -> None:
    """A frame whose beams landed on a mirror (the network twice too far against the lidar)
    bends the law while it sits in the pool; with a two-frame pool, two honest frames later it
    is gone and the law is the identity again."""
    from pepin.depth import AffineScale

    z = np.linspace(1.0, 4.0, 300)
    law = AffineScale(pool_frames=2)
    law.observe((2.0 * z, z))
    assert law.a == pytest.approx(2.0)  # d = 2 z: 1 / z = 2 / d
    law.observe((z, z))
    assert 1.0 < law.a < 2.0  # both views in the pool: the fit sits between them
    law.observe((z, z))
    assert law.a == pytest.approx(1.0) and law.b == pytest.approx(0.0, abs=1e-9)


def test_camera_info_slots_the_bgr_flip_and_a_half_turn_through_a_quaternion() -> None:
    """CameraInfo's row-major K puts fx, cx, fy, cy at 0, 2, 4, 5; a bgr8 image comes out RGB
    and an encoding we cannot read comes out None; a 180 degree rotation has trace -1 (the
    other branch of the conversion) and still comes back as the same matrix."""
    from pepin.depth import decode_rgb, quaternion_from_matrix, rotation_matrix

    k = [457.0, 0.0, 320.0, 0.0, 460.0, 180.0, 0.0, 0.0, 1.0]
    assert Intrinsics.from_camera_info(k, 640, 360) == Intrinsics(
        457.0, 460.0, 320.0, 180.0, 640, 360
    )
    px = bytes([1, 2, 3, 4, 5, 6])  # two pixels
    bgr = decode_rgb(px, 1, 2, "bgr8")
    assert bgr is not None and bgr.shape == (1, 2, 3) and bgr[0, 0].tolist() == [3, 2, 1]
    rgb = decode_rgb(px, 1, 2, "rgb8")
    assert rgb is not None and rgb[0, 1].tolist() == [4, 5, 6]
    assert decode_rgb(px, 1, 2, "mono8") is None
    for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]):
        n = np.array(axis) / np.linalg.norm(axis)
        rot = rotation_matrix(n[0], n[1], n[2], 0.0)  # a half turn about n
        assert np.trace(rot) == pytest.approx(-1.0)
        assert rotation_matrix(*quaternion_from_matrix(rot)) == pytest.approx(rot, abs=1e-12)


def test_a_law_file_that_is_not_a_law_is_refused(tmp_path: Path) -> None:
    """A key missing, a list instead of a dict, a word where a number belongs, a bare null:
    each is refused as no law, never an exception on the node's start-up."""
    from pepin.depth import load_law

    path = tmp_path / "law.json"
    for text in (
        '{"a": 1.0}',
        "[1.0, 0.0, 300, 0.0]",
        '{"a": "one", "b": 0.0, "pooled": 300, "saved_at": 0.0}',
        "null",
    ):
        path.write_text(text)
        assert load_law(path, now=0.0) is None, text


def test_the_camera_pose_is_read_off_the_optical_edge_pitch_kept_pan_reported() -> None:
    """base_link <- camera_optical as TF carries it (the neck's pitch and pan, then REP 103's
    optical turn): the pose the pipeline wants has the translation as is and the pitch of the
    optical axis; the pan is not carried, but optical_heading says how far the head is turned."""
    from pepin.camera import OPTICAL_RPY
    from pepin.depth import optical_heading
    from pepin.mounts import rotation_from_rpy

    optical = rotation_from_rpy(*OPTICAL_RPY)
    straight = rotation_from_rpy(0.0, math.radians(31.5), 0.0) @ optical
    cam = CameraPose.from_optical(straight, np.array([0.02, -0.01, 1.2]))
    assert (cam.x, cam.y, cam.z) == (0.02, -0.01, 1.2)
    assert cam.pitch == pytest.approx(math.radians(31.5))
    assert optical_heading(straight) == pytest.approx((math.radians(31.5), 0.0))
    panned = rotation_from_rpy(0.0, math.radians(31.5), math.radians(-20.0)) @ optical
    assert CameraPose.from_optical(panned, np.zeros(3)).pitch == pytest.approx(math.radians(31.5))
    assert optical_heading(panned) == pytest.approx((math.radians(31.5), math.radians(-20.0)))
    # the optical axis of the level, unturned camera is base_link's x: no pitch, no pan
    assert optical_heading(optical) == pytest.approx((0.0, 0.0))
