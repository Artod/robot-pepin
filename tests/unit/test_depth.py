"""The lidar sets the scale of the camera's depth: projection, the verdict, the running scale."""

import math

import numpy as np
import pytest

from pepin.depth import (
    MIN_SAMPLES,
    CameraPose,
    DepthScale,
    Intrinsics,
    project,
    scale_from_samples,
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


def test_the_verdict_is_the_median_ratio_and_needs_enough_samples() -> None:
    depth = np.full((360, 640), 2.0)  # the network says 2 m everywhere
    n = MIN_SAMPLES + 5
    samples = np.stack([np.linspace(10, 600, n), np.full(n, 300.0), np.full(n, 1.0)], axis=1)
    samples[0, 2] = 40.0  # one wild lidar return does not move the median
    verdict = scale_from_samples(depth, samples)
    assert verdict is not None
    scale, count = verdict
    assert scale == pytest.approx(0.5) and count == n
    assert scale_from_samples(depth, samples[: MIN_SAMPLES - 1]) is None
    assert scale_from_samples(depth, samples[:0]) is None
    depth[300, :] = np.nan  # no prediction on that row
    assert scale_from_samples(depth, samples) is None


def test_the_running_scale_steps_boundedly_and_holds_without_the_lidar() -> None:
    scale = DepthScale()
    assert scale.observe(None) == 1.0 and scale.held == 1
    assert scale.observe((0.5, 30)) == 0.5  # the first verdict is taken whole
    assert scale.observe((5.0, 30)) == pytest.approx(0.625)  # then at most +25 % a frame
    assert scale.observe((0.1, 30)) == pytest.approx(0.625 * 0.75)
    held = scale.observe(None)
    assert held == pytest.approx(0.625 * 0.75) and scale.held == 1 and scale.frames == 3


def test_a_quaternion_becomes_the_rotation_tf_means() -> None:
    from pepin.depth import rotation_matrix

    assert rotation_matrix(0.0, 0.0, 0.0, 1.0) == pytest.approx(np.eye(3))
    quarter = rotation_matrix(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    assert quarter @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])  # +90 deg yaw
