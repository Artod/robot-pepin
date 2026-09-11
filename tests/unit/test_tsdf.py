"""The fused model: one wall from many frames, sharper from near, and the frame that turns
itself to fit the model before it is let in."""

import math

import numpy as np
import pytest

from pepin.depth import Intrinsics
from pepin.tsdf import (
    ALIGN_MIN_GAIN,
    Alignment,
    AlignReason,
    GridSpec,
    RigidPose,
    Tsdf,
    align_yaw,
    backproject,
)

INTR = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=45.0, width=160, height=90)
RANGE_M = 4.0


def _spec() -> GridSpec:
    return GridSpec(origin=(-1.0, -3.0, -0.2), shape=(80, 120, 20), range_max_m=RANGE_M)


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


OBLIQUE = (math.cos(math.radians(40.0)), math.sin(math.radians(40.0)), 0.0)


def _oblique_model(frames: int = 8) -> tuple[Tsdf, RigidPose, np.ndarray]:
    """A model of a wall seen obliquely (its normal 40 degrees off the view) from the origin,
    fed ``frames`` times so the whole wall carries ``min_weight`` — with three frames the far
    end (a frame from 3 m weighs 0.44) stays unknown and the score's edge noise decides."""
    model = Tsdf(_spec())
    true_pose = _optical_pose(0.0, 0.0, 0.0)
    depth = _render_plane(true_pose, OBLIQUE, 2.0)
    for _ in range(frames):
        model.integrate(depth, None, INTR, true_pose)
    return model, true_pose, backproject(depth, INTR, stride=2, range_max=RANGE_M)


def _turned(model: Tsdf, true_pose: RigidPose, band: np.ndarray, error_deg: float) -> Alignment:
    """The alignment of ``band`` when the tracker's heading is ``error_deg`` off the truth."""
    wrong = true_pose.turned_about((0.0, 0.0), math.radians(error_deg))
    return align_yaw(model, band @ wrong.rotation.T + wrong.translation, (0.0, 0.0))


def test_two_frames_of_one_wall_make_one_surface_on_the_wall() -> None:
    model = Tsdf(_spec())
    for yaw in (0.0, math.radians(3.0)):
        pose = _optical_pose(0.0, 0.0, yaw)
        touched = model.integrate(_render_wall(pose, 2.0), None, INTR, pose)
        assert touched > 100
    points, _ = model.surface(min_weight=1.0)
    assert points.shape[0] > 50
    assert np.abs(points[:, 0] - 2.0).max() < 0.03  # on the wall to well under a voxel


def test_the_frame_integrates_its_whole_view_and_nothing_behind_it() -> None:
    """The integration box is the frustum's: the wall shows across the full field of view
    (half-angle 21.8 degrees at fx 200 over 160 px) and nothing is written behind the camera."""
    model = Tsdf(_spec())
    pose = _optical_pose(0.0, 0.0, 0.0)
    model.integrate(_render_wall(pose, 2.0), None, INTR, pose)
    points, _ = model.surface(min_weight=1.0)
    half_width = 2.0 * math.tan(math.atan2(INTR.cx, INTR.fx))
    assert points[:, 1].max() > half_width - 0.05 and points[:, 1].min() < -half_width + 0.05
    assert not np.any(model.weight[: int(1.0 / model.spec.voxel_m)] > 0)  # x < 0: untouched


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


def test_min_weight_hides_a_wall_seen_once_from_two_metres() -> None:
    """One look from 2 m weighs 1: below the default min_weight of 2 the wall is not shown,
    a second look (or one from 1 m, weighing 4) makes it known."""
    model = Tsdf(_spec())
    pose = _optical_pose(0.0, 0.0, 0.0)
    model.integrate(_render_wall(pose, 2.0), None, INTR, pose)
    assert model.surface(min_weight=2.0)[0].shape[0] == 0
    assert model.surface(min_weight=1.0)[0].shape[0] > 50
    model.integrate(_render_wall(pose, 2.0), None, INTR, pose)
    assert model.surface(min_weight=2.0)[0].shape[0] > 50


def test_a_moved_wall_is_overwritten_within_max_weight() -> None:
    """A wall looked at from 1 m until its weight saturates (4 a frame, cap 60), then moved
    30 cm back and looked at from 1 m again: the model's memory is max_weight / 4 = 15 frames,
    and the slowest voxel (the old wall's back, sdf -0.75) flips after ln(1.75) / ln(64/60) =
    9 of them; twelve frames put the surface at the new place and nothing at the old."""
    model = Tsdf(_spec())
    pose = _optical_pose(1.0, 0.0, 0.0)
    for _ in range(15):
        model.integrate(_render_wall(pose, 2.0), None, INTR, pose)
    assert float(model.weight.max()) == pytest.approx(model.spec.max_weight)
    moved = _optical_pose(1.3, 0.0, 0.0)
    for _ in range(12):
        model.integrate(_render_wall(moved, 2.3), None, INTR, moved)
    points, _ = model.surface()
    # the core both cameras saw (the old wall's rims outside the new view are rightly kept)
    core = points[(np.abs(points[:, 1]) < 0.2) & (points[:, 2] > 0.5) & (points[:, 2] < 0.7)]
    assert core.shape[0] > 20
    assert np.abs(core[:, 0] - 2.3).max() < 0.03


def test_colour_stays_on_its_own_surface() -> None:
    """A red face 1 m ahead in the right half of the view, a white wall at 2 m behind it: the
    wall's rays run past the face's edge through free space and must not tint it; the surface
    reads red at 1 m and white at 2 m."""
    model = Tsdf(_spec())
    pose = _optical_pose(0.0, 0.0, 0.0)
    wall = _render_wall(pose, 2.0)
    face = _render_wall(pose, 1.0)
    cols = np.arange(INTR.width)[None, :]
    on_face = cols >= INTR.width // 2
    depth = np.where(on_face, face, wall)
    rgb = np.where(on_face[:, :, None], [255, 0, 0], [255, 255, 255]).astype(np.uint8)
    rgb = np.broadcast_to(rgb, (INTR.height, INTR.width, 3)).copy()
    for _ in range(3):
        model.integrate(depth, rgb, INTR, pose)
    points, colours = model.surface()
    at_face = np.abs(points[:, 0] - 1.0) < 0.03
    at_wall = np.abs(points[:, 0] - 2.0) < 0.03
    assert at_face.sum() > 20 and at_wall.sum() > 20
    assert np.all(colours[at_face] == [255, 0, 0])
    assert np.all(colours[at_wall] == [255, 255, 255])
    assert model.rgb.dtype == np.uint8


def test_the_frame_turns_itself_to_fit_the_model() -> None:
    """A wall seen obliquely: a heading error slides the frame's points off the plane, and the
    alignment turns them back; a frame that already fits asks for no turn."""
    model, true_pose, band = _oblique_model()
    # the tracker says the cart faces 2 degrees left of where it really does
    found = _turned(model, true_pose, band, 2.0)
    assert found.reason is AlignReason.ALIGNED and found.aligned
    assert found.judged >= 200 and found.gain > ALIGN_MIN_GAIN
    assert found.yaw == pytest.approx(math.radians(-2.0), abs=math.radians(0.15))
    fits = _turned(model, true_pose, band, 0.0)
    assert fits.reason is AlignReason.FITS and fits.yaw == 0.0 and not fits.aligned


def test_the_refinement_lands_between_the_lattice_points() -> None:
    """A 2.25 degree error lies halfway between two candidates (0.5 degree lattice): the
    parabola through the best three must answer near -2.25, not the lattice's -2.5 nudged
    away from the truth (the sign of the vertex formula)."""
    model, true_pose, band = _oblique_model()
    for error in (2.25, -2.25, 1.1):
        found = _turned(model, true_pose, band, error)
        assert found.reason is AlignReason.ALIGNED
        assert found.yaw == pytest.approx(math.radians(-error), abs=math.radians(0.15))


def test_a_far_wall_decides_the_turn_over_a_near_cluster() -> None:
    """A sofa 0.5 m ahead fills seven eighths of the view and only slides along itself when
    the frame turns; the far oblique wall in the rest is what the turn moves. Weighted by lever
    arm the wall decides, and its gain is about twice what a plain mean over the points gives
    (0.12 against 0.07 here; a longer sofa would push the plain mean under the 0.02 gate)."""
    model = Tsdf(_spec())
    true_pose = _optical_pose(0.0, 0.0, 0.0)
    sofa = _render_wall(true_pose, 0.5)
    wall = _render_plane(true_pose, OBLIQUE, 2.0)
    cols = np.arange(INTR.width)[None, :]
    on_wall = cols < INTR.width // 8  # a fifth of the pixels, four times farther out
    depth = np.where(on_wall, wall, sofa)
    for _ in range(8):
        model.integrate(depth, None, INTR, true_pose)
    band = backproject(depth, INTR, stride=2, range_max=RANGE_M)
    error = 1.5
    wrong = true_pose.turned_about((0.0, 0.0), math.radians(error))
    band_map = band @ wrong.rotation.T + wrong.translation
    found = align_yaw(model, band_map, (0.0, 0.0))
    assert found.reason is AlignReason.ALIGNED
    assert found.yaw == pytest.approx(math.radians(-error), abs=math.radians(0.3))
    # the same band, every point counting alike: the sofa's points halve the wall's signal
    plain_gain = (
        model.score(band @ true_pose.rotation.T + true_pose.translation)[0]
        - model.score(band_map)[0]
    )
    assert ALIGN_MIN_GAIN < plain_gain < found.gain / 1.5


def test_a_turn_cannot_win_by_dropping_points_off_the_model() -> None:
    """The score's denominator is the whole band: a point carried off known ground counts
    zero, so leaving the model costs, and points on known ground alone would have hidden it."""
    model = Tsdf(_spec())
    model.weight[20:60, 40:80, :] = 10.0  # a known slab of zero field: every point fits
    model.sdf[:] = 0.0
    on = np.array([[1.5, 0.0, 0.3], [1.6, 0.1, 0.3]])
    off = np.array([[-0.5, 2.5, 0.3], [-0.6, 2.4, 0.3]])
    assert model.score(on) == (1.0, 2)
    assert model.score(np.vstack([on, off])) == (0.5, 2)
    fit, known = model.fit_per_point(np.vstack([on, off]))
    assert float(fit[known].mean()) == 1.0  # the biased mean would not have told them apart
    # and a frame of a wall that fits already: no turn improves it, even where a turn would
    # only slide points along the wall
    wall_model, true_pose, band = _oblique_model()
    exact = align_yaw(wall_model, band @ true_pose.rotation.T + true_pose.translation, (0.0, 0.0))
    assert exact.reason is AlignReason.FITS


def test_a_heading_error_past_the_search_is_refused_with_its_reason() -> None:
    """Past +-4 degrees the best candidate is the last one tried: the true turn may lie beyond
    the search, and a turn to the bound would bake the remainder into the model, so the verdict
    is AT_BOUND. Further out the model stops knowing the band's voxels at all. Inside, it
    answers."""
    model, true_pose, band = _oblique_model()
    at_bound = _turned(model, true_pose, band, 5.0)
    assert at_bound.reason is AlignReason.AT_BOUND and not at_bound.aligned
    assert at_bound.yaw == pytest.approx(math.radians(-4.0))
    assert _turned(model, true_pose, band, 9.0).reason is AlignReason.UNJUDGED
    inside = _turned(model, true_pose, band, 3.0)
    assert inside.reason is AlignReason.ALIGNED
    assert inside.yaw == pytest.approx(math.radians(-3.0), abs=math.radians(0.15))


def test_the_candidates_must_include_no_turn() -> None:
    model = Tsdf(_spec())
    with pytest.raises(ValueError):
        align_yaw(model, np.zeros((1, 3)), (0.0, 0.0), candidates=(-0.1, 0.1))
    # a zero that is not an exact float zero still counts as no turn
    assert align_yaw(model, np.zeros((1, 3)), (0.0, 0.0), candidates=(-0.1, 1e-15, 0.1)).reason is (
        AlignReason.UNJUDGED
    )


def test_turned_about_a_pivot_away_from_the_origin() -> None:
    """The cart at (2, 1) with its camera 0.1 m ahead: a turn of 90 degrees about the cart
    moves the camera to the cart's left, not around the map's origin."""
    pose = _optical_pose(2.1, 1.0, 0.0)
    turned = pose.turned_about((2.0, 1.0), math.radians(90.0))
    assert turned.translation == pytest.approx([2.0, 1.1, 0.6], abs=1e-9)
    forward = turned.rotation[:, 2]  # optical z in the map
    assert forward == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    assert pose.turned_about((0.0, 0.0), math.radians(90.0)).translation == pytest.approx(
        [-1.0, 2.1, 0.6], abs=1e-9
    )


def test_an_empty_model_has_no_surface_and_judges_nothing() -> None:
    """The publisher runs once a second from the first one, before any frame is in."""
    model = Tsdf(_spec())
    points, colours = model.surface()
    assert points.shape == (0, 3) and colours.shape == (0, 3) and colours.dtype == np.uint8
    assert model.score(np.array([[0.0, 0.0, 0.5]])) == (0.0, 0)
    assert align_yaw(model, np.zeros((300, 3)), (0.0, 0.0)).reason is AlignReason.UNJUDGED


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


def test_the_snapshot_carries_what_the_surface_reads() -> None:
    model = Tsdf(_spec())
    pose = _optical_pose(0.0, 0.0, 0.0)
    model.integrate(_render_wall(pose, 2.0), None, INTR, pose)
    twin = model.snapshot()
    model.integrate(_render_wall(pose, 2.5), None, INTR, pose)
    points, _ = twin.surface(min_weight=1.0)
    assert np.abs(points[:, 0] - 2.0).max() < 0.03  # the twin kept the first wall
    assert not hasattr(twin, "colour_weight")  # only what surface() needs is copied


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


def test_the_surface_colour_leans_to_the_view_that_weighed_more() -> None:
    """The same wall seen white from 2.9 m (weight 0.48) and red from 1 m (weight 4): the
    surface's colour is the weighted mean, red with a ninth of white in it — the same average
    the field uses, on the colour's own weight."""
    spec = _spec()
    model = Tsdf(spec)
    far, near = _optical_pose(-0.9, 0.0, 0.0), _optical_pose(1.0, 0.0, 0.0)
    white = np.full((INTR.height, INTR.width, 3), 255, dtype=np.uint8)
    red = white.copy()
    red[:, :, 1:] = 0
    model.integrate(_render_wall(far, 2.0), white, INTR, far)
    model.integrate(_render_wall(near, 2.0), red, INTR, near)
    points, colours = model.surface(min_weight=1.0)
    centre = (np.abs(points[:, 1]) < 0.2) & (np.abs(points[:, 2] - 0.6) < 0.15)
    assert centre.sum() > 20
    w_far, w_near = (spec.observation_weight(np.array([d]))[0] for d in (2.9, 1.0))
    expected = round(255 * w_far / (w_far + w_near))
    assert 20 < expected < 35
    assert np.all(colours[centre, 0] == 255)
    assert np.abs(colours[centre, 1:].astype(int) - expected).max() <= 1


def test_a_depth_beyond_range_max_carves_nothing() -> None:
    """A wall past range_max is not information: the frame touches no voxel, and the free
    space in front of it is not carved either; five centimetres nearer it is a wall again."""
    model = Tsdf(_spec())
    pose = _optical_pose(-0.9, 0.0, 0.0)
    assert model.integrate(_render_wall(pose, -0.9 + RANGE_M + 0.05), None, INTR, pose) == 0
    assert not model.weight.any()
    assert model.integrate(_render_wall(pose, -0.9 + RANGE_M - 0.05), None, INTR, pose) > 0


def test_the_frustum_box_is_the_view_out_to_range_max_and_nothing_when_the_view_misses() -> None:
    """The voxels a frame may touch: from the camera out to its corner rays at range_max, and
    no farther sideways than the view reaches — a point well inside the grid but outside the
    view is not in the box, nor is one behind the camera. A camera outside the grid looking
    away from it has no box at all and its frame is a no-op."""
    spec = _spec()
    model = Tsdf(spec)
    pose = _optical_pose(0.0, 0.0, 0.0)
    box = model._frustum_box(INTR, pose)
    assert box is not None
    corners = np.array(
        [
            [(u - INTR.cx) / INTR.fx * 0.8, (v - INTR.cy) / INTR.fy * 0.8, 0.8]
            for u in (0, 159)
            for v in (0, 89)
        ]
    )
    inside = np.vstack([corners @ pose.rotation.T + pose.translation, [[2.99, 0.0, 0.6]]])
    outside = np.array([[-0.15, 0.0, 0.6], [1.0, 2.5, 0.6]])  # behind the lens; beside the view
    for points, expected in ((inside, True), (outside, False)):
        idx, in_grid = model.voxel_of(points)
        assert in_grid.all()
        in_box = np.all(
            [(box[k].start <= idx[:, k]) & (idx[:, k] < box[k].stop) for k in range(3)], axis=0
        )
        assert in_box.all() == expected and in_box.any() == expected
    away = _optical_pose(-2.0, 0.0, math.pi)  # outside the grid, looking away from it
    assert model._frustum_box(INTR, away) is None
    assert model.integrate(_render_wall(away, -4.0), None, INTR, away) == 0
