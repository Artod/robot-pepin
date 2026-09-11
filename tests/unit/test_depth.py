"""The lidar sets the scale of the camera's depth: projection, the verdict, the running scale."""

import math

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


def test_the_depth_image_becomes_a_scan_of_what_stands_above_the_floor() -> None:
    """A tilted camera looking at a wall two metres ahead: the wall's pixels between 8 cm and
    1.3 m mark the central bearings at 2 m, the floor pixels below it mark nothing."""
    from pepin.depth import depth_to_scan

    cam = CameraPose(0.0, 0.0, 1.23, math.radians(28.0))
    rows, cols = np.mgrid[0:360, 0:640]
    # each pixel's ray, in camera_link: forward = 1, left, up (per unit depth)
    left = -(cols + 0.5 - INTR.cx) / INTR.fx
    up = -(rows + 0.5 - INTR.cy) / INTR.fy
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    fwd_base = c * 1.0 + s * up  # base x per unit optical depth
    up_base = -s * 1.0 + c * up
    depth = np.full((360, 640), np.inf)
    wall = 2.0 / fwd_base  # optical depth at which the ray meets the wall plane x = 2 m
    floor = -cam.z / up_base  # optical depth at which the ray meets the floor z = 0
    floor[up_base >= 0] = np.inf
    depth = np.where(wall < floor, wall, floor)
    angle_min, step, ranges = depth_to_scan(depth, INTR, cam)
    centre = round((0.0 - angle_min) / step)
    assert ranges[centre] == pytest.approx(2.0, abs=0.02)
    assert ranges[centre - 20] == pytest.approx(2.0 / math.cos(20 * step), abs=0.03)
    assert np.isfinite(ranges).sum() > 100  # the wall spans most of the view
    empty = depth_to_scan(np.full((360, 640), np.inf), INTR, cam)[2]
    assert not np.isfinite(empty).any()
    _ = left  # the ray geometry above is what the function inverts


def test_the_floor_s_depth_follows_the_camera_s_height_tilt_and_the_cart_s_lean() -> None:
    """A camera 1.23 m up, tilted 26 degrees down: the ray through the principal point meets the
    floor 1.23 / sin(26 deg) along the axis; the cart pitched 5 degrees nose-down brings it to
    1.23 / sin(31 deg); rows at and above the horizon never meet the floor."""
    import math

    from pepin.depth import CameraPose, floor_anchor, floor_depth

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
    # the anchor: floor within 15 % snaps, a box on the floor does not
    guess = level * 1.06
    guess[300:340, 200:260] = level[300:340, 200:260] * 0.5  # a box half-way to the floor
    guess[:100, :] = np.nan
    fixed, anchored = floor_anchor(guess, level)
    assert anchored > 100_000
    assert np.allclose(fixed[350, 100:600], level[350, 100:600])
    assert np.allclose(fixed[320, 230], guess[320, 230])  # the box is left to the network


def test_the_tilt_reads_gravity_through_the_mount_and_ignores_a_bump() -> None:
    from pepin.depth import GRAVITY, Tilt, imu_mount_rotation

    tilt = Tilt(imu_mount_rotation(90.0, 0.0, 0.0))
    tilt.observe(np.array([0.0, GRAVITY, 0.0]), 0.0)  # the chip's Y up: level
    assert np.allclose(tilt.up, [0.0, 0.0, 1.0])
    assert tilt.roll_pitch_deg == pytest.approx((0.0, 0.0), abs=1e-9)
    tilt.observe(np.array([0.0, GRAVITY, 6.0]), 0.5)  # a bump: not 1 g, ignored
    assert np.allclose(tilt.up, [0.0, 0.0, 1.0])
    # nose down 5 degrees: gravity leans onto the chip's z (base x); after a few seconds it shows
    import math

    a = np.array(
        [-GRAVITY * math.sin(math.radians(5.0)), GRAVITY * math.cos(math.radians(5.0)), 0.0]
    )
    for i in range(1, 60):
        tilt.observe(a, 0.5 + 0.1 * i)
    _roll, pitch = tilt.roll_pitch_deg
    assert pitch == pytest.approx(5.0, abs=0.2)


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


def test_a_floor_the_camera_cannot_see_anchors_nothing() -> None:
    """Pointed at the ceiling no ray meets the floor: the expected image is all NaN and the
    network's depth must come back untouched."""
    from pepin.depth import floor_anchor

    guess = np.full((8, 8), 1.5)
    fixed, anchored = floor_anchor(guess, np.full((8, 8), np.nan))
    assert anchored == 0 and np.array_equal(fixed, guess)


def test_a_stale_scan_is_carried_to_the_frame_s_moment() -> None:
    """The cart turned 2 degrees left between the scan and the frame: a point dead ahead at 2 m
    at the scan's moment sits 2 degrees to the right at the frame's moment."""
    import math

    from pepin.depth import carry, rotation_matrix

    ahead = np.array([[2.0, 0.0, 0.2]])
    yaw = math.radians(-2.0)  # base_link@frame <- base_link@scan: the world turned right
    rot = rotation_matrix(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    moved = carry(ahead, rot, np.zeros(3))
    assert moved[0, 1] == pytest.approx(-2.0 * math.sin(math.radians(2.0)), rel=1e-6)
    assert moved[0, 2] == 0.2


def test_the_beams_pooled_over_frames_fit_the_network_s_depth_as_an_affine_law() -> None:
    """The network sees the room too far, the far end more than the near: true 1/z = 1.2/D +
    0.05. One frame's beams span 1.3-2 m and fit a scale only; ten frames pooled span 1-4 m
    and recover both numbers; a frame without beams keeps the law."""
    from pepin.depth import AffineScale, apply_affine, beam_pairs, fit_affine

    a_true, b_true = 1.2, 0.05

    def frame(z_lo: float, z_hi: float) -> tuple[np.ndarray, np.ndarray]:
        z_true = np.linspace(z_lo, z_hi, 40)
        d_net = 1.0 / ((1.0 / z_true - b_true) / a_true)
        depth = np.tile(d_net, (10, 1))
        samples = np.stack([np.arange(40, dtype=float), np.full(40, 5.0), z_true], axis=1)
        return depth, samples

    depth, samples = frame(1.3, 2.0)
    pairs = beam_pairs(depth, samples)
    assert pairs is not None and pairs[0].size == 40
    assert fit_affine(*pairs)[1] == 0.0  # too narrow a spread for a shift
    law = AffineScale()
    for lo, hi in ((1.0, 1.6), (1.4, 2.2), (2.0, 3.0), (2.8, 4.0), (1.0, 4.0), (1.2, 3.5)):
        law.observe(beam_pairs(*frame(lo, hi)))
    assert law.a == pytest.approx(a_true, rel=1e-3) and law.b == pytest.approx(b_true, abs=1e-3)
    assert law.pooled == 240
    held = law.observe(None)
    assert held == (law.a, law.b) and law.held == 1
    corrected = apply_affine(frame(1.0, 4.0)[0], *held)
    assert np.allclose(corrected[5], np.linspace(1.0, 4.0, 40), atol=0.01)
    assert beam_pairs(depth, samples[: MIN_SAMPLES - 1]) is None


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
