"""The fused model: one wall from many frames, sharper from near, and the frame that turns
itself to fit the model before it is let in."""

import math

import numpy as np
import pytest

from pepin.depth import Intrinsics
from pepin.tsdf import GridSpec, RigidPose, Tsdf, align_yaw, backproject

INTR = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=45.0, width=160, height=90)


def _spec() -> GridSpec:
    return GridSpec(origin=(-1.0, -3.0, -0.2), shape=(80, 120, 20), range_max_m=4.0)


def _optical_pose(x: float, y: float, yaw: float, z: float = 0.6) -> RigidPose:
    """A camera at (x, y, z) looking along the map heading ``yaw``, level: map <- optical
    (optical x right, y down, z forward)."""
    c, s = math.cos(yaw), math.sin(yaw)
    forward = np.array([c, s, 0.0])
    left = np.array([-s, c, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    rotation = np.stack([-left, -up, forward], axis=1)  # columns: optical x, y, z in map
    return RigidPose(rotation, np.array([x, y, z]))


def _render_wall(pose: RigidPose, wall_x: float) -> np.ndarray:
    """The depth image a camera at ``pose`` sees of the plane x = wall_x (metres, optical)."""
    return _render_plane(pose, (1.0, 0.0, 0.0), wall_x)


def _render_plane(pose: RigidPose, normal: tuple[float, float, float], offset: float) -> np.ndarray:
    """The depth image of the plane normal . p = offset seen from ``pose``; NaN off the plane."""
    rows, cols = np.mgrid[0 : INTR.height, 0 : INTR.width]
    dirs = np.stack(
        [(cols - INTR.cx) / INTR.fx, (rows - INTR.cy) / INTR.fy, np.ones_like(rows, dtype=float)],
        axis=-1,
    ).reshape(-1, 3)
    n = np.array(normal, dtype=float)
    denom = (dirs @ pose.rotation.T) @ n
    t = np.full(denom.shape, np.nan)
    np.divide(offset - pose.translation @ n, denom, out=t, where=np.abs(denom) > 1e-3)
    # depth = t because dirs z == 1; a ray grazing the plane runs off to infinity, not to a room
    depth = np.where((t > 0) & (t < 50.0), t, np.nan)
    return depth.reshape(INTR.height, INTR.width)


def test_two_frames_of_one_wall_make_one_surface_on_the_wall() -> None:
    model = Tsdf(_spec())
    for yaw in (0.0, math.radians(3.0)):
        pose = _optical_pose(0.0, 0.0, yaw)
        touched = model.integrate(_render_wall(pose, 2.0), None, INTR, pose)
        assert touched > 100
    points, _ = model.surface(min_weight=1.0)
    assert points.shape[0] > 50
    assert np.abs(points[:, 0] - 2.0).max() < 0.03  # on the wall to well under a voxel


def test_a_near_observation_outweighs_a_far_one_and_the_model_sharpens() -> None:
    spec = _spec()
    model = Tsdf(spec)
    far = _optical_pose(-0.9, 0.0, 0.0)  # 2.9 m from the wall the far view believes at x=2.1
    model.integrate(_render_wall(far, 2.1), None, INTR, far)
    near = _optical_pose(1.2, 0.0, 0.0)  # 0.8 m from the true wall at x=2.0
    model.integrate(_render_wall(near, 2.0), None, INTR, near)
    points, _ = model.surface(min_weight=1.0)
    centre = points[np.abs(points[:, 1]) < 0.2]
    assert abs(float(np.median(centre[:, 0])) - 2.0) < 0.03  # the near view decided
    assert (
        spec.observation_weight(np.array([0.8]))[0]
        > spec.observation_weight(np.array([2.9]))[0] * 5
    )


def test_the_frame_turns_itself_to_fit_the_model() -> None:
    """A wall seen obliquely (its normal 40 degrees off the view): a heading error slides the
    frame's points off the plane, and the alignment turns them back."""
    model = Tsdf(_spec())
    true_pose = _optical_pose(0.0, 0.0, 0.0)
    normal = (math.cos(math.radians(40.0)), math.sin(math.radians(40.0)), 0.0)
    for _ in range(3):
        model.integrate(_render_plane(true_pose, normal, 2.0), None, INTR, true_pose)
    # the tracker says the cart faces 2 degrees left of where it really does
    wrong = true_pose.turned_about((0.0, 0.0), math.radians(2.0))
    band = backproject(_render_plane(true_pose, normal, 2.0), INTR, stride=2)
    band_map = band @ wrong.rotation.T + wrong.translation
    found = align_yaw(model, band_map, (0.0, 0.0))
    assert found is not None
    yaw, gain, judged = found
    assert judged >= 200 and gain > 0.02
    assert yaw == pytest.approx(math.radians(-2.0), abs=math.radians(0.4))
    # a frame that already fits asks for no turn
    assert align_yaw(model, band @ true_pose.rotation.T + true_pose.translation, (0.0, 0.0)) is None


def test_an_empty_model_has_no_surface_and_judges_nothing() -> None:
    """The publisher runs once a second from the first one, before any frame is in."""
    model = Tsdf(_spec())
    points, colours = model.surface()
    assert points.shape == (0, 3) and colours.shape == (0, 3) and colours.dtype == np.uint8
    assert model.score(np.array([[0.0, 0.0, 0.5]])) == (0.0, 0)


def test_the_trilinear_read_stops_at_the_grid_s_edge() -> None:
    """A point in the outermost voxel has no eight neighbours: it must be dropped, not wrapped."""
    spec = _spec()
    model = Tsdf(spec)
    model.weight[:] = 10.0
    model.sdf[:] = 0.0
    nx, ny, nz = spec.shape
    far = np.array(spec.origin) + np.array([nx, ny, nz]) * spec.voxel_m
    outside = np.array([far, np.array(spec.origin) - 0.01, far - 0.4 * spec.voxel_m])
    assert model.score(outside) == (0.0, 0)
    inside = np.array([np.array(spec.origin) + np.array([nx, ny, nz]) * spec.voxel_m / 2])
    assert model.score(inside)[1] == 1


def test_a_heading_error_past_the_search_answers_at_the_bound_or_not_at_all() -> None:
    """Past +-4 degrees the best candidate is the last one tried: the true turn may lie beyond
    the search, and a turn to the bound would bake the remainder into the model, so the frame is
    refused. Further out the model stops knowing the band's voxels at all. Inside, it answers."""
    model = Tsdf(_spec())
    true_pose = _optical_pose(0.0, 0.0, 0.0)
    normal = (math.cos(math.radians(40.0)), math.sin(math.radians(40.0)), 0.0)
    depth = _render_plane(true_pose, normal, 2.0)
    for _ in range(3):
        model.integrate(depth, None, INTR, true_pose)
    band = backproject(depth, INTR, stride=2)

    def turned(error_deg: float) -> tuple[float, float, int] | None:
        wrong = true_pose.turned_about((0.0, 0.0), math.radians(error_deg))
        return align_yaw(model, band @ wrong.rotation.T + wrong.translation, (0.0, 0.0))

    assert turned(5.0) is None  # the best candidate is the last one tried: refused, not baked in
    assert turned(7.0) is None  # too few of the band's points land on voxels the model knows
    inside = turned(3.0)
    assert inside is not None and inside[0] == pytest.approx(
        math.radians(-3.0), abs=math.radians(0.4)
    )


def test_the_grid_config_is_the_one_the_node_loads() -> None:
    from pathlib import Path

    spec = GridSpec.load(Path(__file__).resolve().parents[2] / "config/fusion.json")
    assert spec.voxel_m == 0.05 and spec.truncation_m == 2 * spec.voxel_m
    x0, y0, z0 = spec.origin
    nx, ny, nz = spec.shape
    # the served map (flat3_straight, 239x215 cells at 5 cm from -18.53, -4.38) lies inside
    assert x0 <= -18.53 and x0 + nx * spec.voxel_m >= -18.53 + 239 * 0.05
    assert y0 <= -4.38 and y0 + ny * spec.voxel_m >= -4.38 + 215 * 0.05
    assert z0 <= 0.0 and z0 + nz * spec.voxel_m >= 1.4
    assert spec.weight_cap > 1.0 > spec.observation_weight(np.array([spec.range_max_m]))[0]


def test_the_slow_correction_follows_a_jump_with_its_time_constant_and_composes() -> None:
    from pepin.tsdf import SlowCorrection

    def yawed(deg: float, x: float = 0.0) -> RigidPose:
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        return RigidPose(
            np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.array([x, 0.0, 0.0])
        )

    slow = SlowCorrection(tau_s=10.0)
    first = slow.observe(yawed(0.0), 0.0)
    assert first.rotation[0, 0] == pytest.approx(1.0)  # the first value is taken whole
    after = slow.observe(yawed(2.0, 0.1), 10.0)  # one time constant later: 63 % of the way
    yaw = math.degrees(math.atan2(after.rotation[1, 0], after.rotation[0, 0]))
    assert yaw == pytest.approx(2.0 * (1 - math.exp(-1.0)), abs=0.01)
    assert after.translation[0] == pytest.approx(0.1 * (1 - math.exp(-1.0)), abs=1e-6)
    composed = SlowCorrection.compose(
        yawed(90.0, 1.0), yawed(0.0, 1.0)
    )  # a step east, turned north
    assert composed.translation == pytest.approx([1.0, 1.0, 0.0], abs=1e-9)
