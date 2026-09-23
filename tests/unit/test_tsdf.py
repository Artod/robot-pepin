"""The fused model: one wall from many frames, sharper from near, and the frame that turns
itself to fit the model before it is let in."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from pepin.depth import Intrinsics
from pepin.tsdf import (
    ALIGN_MIN_GAIN,
    Alignment,
    AlignReason,
    DepthLaw,
    GridSpec,
    ObservedReach,
    RigidPose,
    Tsdf,
    align_yaw,
    backproject,
)

INTR = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=45.0, width=160, height=90)
RANGE_M = 4.0


def _spec() -> GridSpec:
    return GridSpec(origin=(-1.0, -3.0, -0.2), shape=(80, 120, 20), range_max_m=RANGE_M)


def _carve_spec() -> GridSpec:
    """The same grid at the live config's max_weight (config/fusion.json, 20.0 since
    2026-09-22): how fast the volume forgets is what a carving test measures."""
    return dataclasses.replace(_spec(), max_weight=20.0)


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
    # the colour's own weight travels with the colour since 2026-09-22: surface()'s
    # colour_fallback asks it which neighbour a camera ever painted, and a twin without it
    # would answer for the whole published cloud
    assert twin.colour_weight.shape == model.colour_weight.shape
    assert twin.colour_weight is not model.colour_weight
    assert twin.surface(min_weight=1.0, colour_fallback=True)[0].shape == points.shape


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


def test_the_band_s_half_width_is_a_config_value_with_a_fallback(tmp_path: Path) -> None:
    """The layer a frame is seated on is data, not a constant compiled into the fusion node
    (CLAUDE.md rule 19): ``band_half_z_m`` in config/fusion.json is where it starts, and the
    node makes it a live flag from there. An unreadable or silent file falls back, never
    raises — the node would otherwise not come up at all."""
    from pepin.deployment import config_file
    from pepin.tsdf import BAND_HALF_Z_M, band_half_z_m

    assert band_half_z_m() == band_half_z_m(config_file("fusion.json")) == 0.125
    named = tmp_path / "named.json"
    named.write_text('{"band_half_z_m": 0.2}')
    assert band_half_z_m(named) == 0.2
    silent = tmp_path / "silent.json"
    silent.write_text("{}")
    assert band_half_z_m(silent) == BAND_HALF_Z_M
    assert band_half_z_m(tmp_path / "missing.json") == BAND_HALF_Z_M


def test_a_slam_session_s_box_is_centred_on_where_the_cart_woke_up() -> None:
    """The served map's box covers its own coordinates; a session born in an unknown place
    puts the map origin under the cart, so the box must be centred there, height untouched."""
    from pepin.tsdf import GridSpec

    spec = GridSpec(origin=(-19.5, -5.5, -0.15), shape=(280, 250, 34), voxel_m=0.05)
    c = spec.centred_on_start()
    assert c.origin == (-7.0, -6.25, -0.15) and c.shape == spec.shape
    assert c.origin[0] + c.shape[0] * c.voxel_m == 7.0  # the cart at (0, 0) sits in the middle


def test_a_seeded_box_is_snapped_to_the_saved_map_s_own_cell_lattice() -> None:
    """A volume seeded from a file must have the file's cells, not cells a third of a cell
    aside: the slice a tracker matches on carries every wall wherever the lattice puts it, and
    on the four tapes of 2026-09-13 that third of a cell was a median 2.3-2.9 cm of live pose
    error against the very same map as a file (scratch/volume_vs_pgm.py + drive_bisect)."""
    from pepin.tsdf import GridSpec

    spec = GridSpec(origin=(-19.5, -5.5, -0.15), shape=(280, 250, 34), voxel_m=0.05)
    served = (-18.533, -4.382)  # ros/maps/flat3_straight.yaml
    snapped = spec.aligned_to(served, 0.05)
    for axis in (0, 1):
        offset = (snapped.origin[axis] - served[axis]) / spec.voxel_m
        assert abs(offset - round(offset)) < 1e-9, "a whole number of cells from the file's"
        assert abs(snapped.origin[axis] - spec.origin[axis]) < spec.voxel_m, "within one voxel"
    assert snapped.origin[2] == spec.origin[2] and snapped.shape == spec.shape
    assert spec.aligned_to(served, 0.10) == spec, "another resolution: nothing can be aligned"
    assert snapped.aligned_to(served, 0.05) == snapped, "already on the lattice"


# ---- the rolling window: a volume that is local memory follows the cart -----------------------
def test_the_window_slides_by_whole_voxels_and_keeps_the_overlap_where_it_was() -> None:
    """A volume painted in odom is LOCAL OBSTACLE MEMORY, so its box follows the cart: the slide
    is a whole number of voxels (the lattice may never move), what stays keeps the very metres it
    was painted at, and what leaves is gone — not carried, not resampled, forgotten."""
    spec = _spec()
    model = Tsdf(spec)
    pose = _optical_pose(0.0, 0.0, 0.0)
    model.integrate(_render_wall(pose, 2.0), None, INTR, pose)
    before, sdf_before = model.weight.copy(), model.sdf.copy()
    wall, _colours = model.surface(min_weight=1.0)
    assert wall.shape[0] > 100

    assert spec.centre_xy == pytest.approx((1.0, 0.0)), "where the cart stands now"
    move = model.recentre((2.02, -0.51))  # not on the lattice: the slide rounds to whole voxels
    assert (move.di, move.dj) == (20, -10) and not move.nothing
    assert model.spec.origin == pytest.approx((-1.0 + 1.0, -3.0 - 0.5, -0.2))
    assert model.spec.shape == spec.shape and model.spec.voxel_m == spec.voxel_m
    assert np.array_equal(model.weight[:-20, 10:], before[20:, :-10]), "what stayed, where it was"
    assert np.array_equal(model.sdf[:-20, 10:], sdf_before[20:, :-10])
    assert not model.weight[-20:].any() and not model.weight[:, :10].any(), "the new edges"
    assert (model.sdf[-20:] == 1.0).all(), "and unobserved is a whole truncation from a surface"
    # the surface that survived is at the same map metres it was: the world did not move
    after, _ = model.surface(min_weight=1.0)
    kept = wall[(wall[:, 0] < model.spec.origin[0] + 80 * 0.05) & (wall[:, 1] > -3.5 + 0.5)]
    assert after.shape[0] == pytest.approx(kept.shape[0], rel=0.05)
    assert np.abs(after[:, 0] - 2.0).max() < 0.05, "the wall is still at x = 2.0 m"


def test_a_window_that_jumps_clear_of_itself_keeps_nothing() -> None:
    """A cart carried across the flat, or an odometry that teleported: the box and its old self do
    not overlap at all, and the new window is simply empty rather than wrapping round."""
    model = Tsdf(_spec())
    model.weight[:] = 5.0
    move = model.recentre((100.0, 100.0))
    assert move.boxes(80, 120) is None
    assert not model.weight.any() and (model.sdf == 1.0).all()
    assert model.spec.shape == (80, 120, 20), "the grid is the same grid, elsewhere"


def test_the_cart_in_the_middle_of_the_window_slides_nothing() -> None:
    """The slide is asked for on every observation, so its no-op has to be free and exact: a cart
    within half a voxel of the centre moves nothing at all, and the arrays are the same arrays."""
    spec = GridSpec(origin=(-2.0, -2.0, -0.2), shape=(80, 80, 20))
    model = Tsdf(spec)
    weight = model.weight
    assert model.recentre((0.0, 0.0)).nothing and model.weight is weight
    assert model.recentre((0.02, -0.02)).nothing, "within half a voxel of the centre"
    assert model.spec == spec
    assert spec.off_centre_m((0.0, 3.0)) == pytest.approx(3.0)
    assert spec.centre_xy == pytest.approx((0.0, 0.0))


def test_what_one_slide_of_a_node_sized_window_costs() -> None:
    """The window is slid on the paint path, under the model's lock, so its cost is a number this
    robot has to know (CLAUDE.md rule 20). It is a copy of five channels and no arithmetic at all:
    measured 0.3-0.9 ms on the fusion node's own 120x120x34 test grid and 4.7 ms on the live
    280x250x34 one, against the 9-108 ms a graph correction's resample costs there
    (scratch/volume_shift_cost.py). The bound below is loose on purpose — it is a regression
    guard on a shared laptop, not the measurement — and the node prints the milliseconds of its
    own last slide in the report line."""
    import time

    model = Tsdf(GridSpec(origin=(-3.0, -3.0, -0.15), shape=(120, 120, 34)))
    model.weight[:] = 3.0
    model.sdf[:] = 0.2
    slides = []
    for step in (0.5, 1.0, 1.5, 2.0, 2.5):
        started = time.perf_counter()
        model.recentre((step, 0.0))
        slides.append((time.perf_counter() - started) * 1e3)
    assert max(slides) < 50.0, f"one slide of the node's window: {max(slides):.1f} ms"


def test_a_pixel_with_no_depth_carves_its_ray_only_when_the_law_says_so() -> None:
    """THE PHANTOM THAT NEVER DECAYED (2026-09-22). A NaN pixel used to touch nothing, so a
    surface standing in front of something beyond the rig's reach — where the published depth is
    NaN by construction — could never be carved by anything. With ``no_depth_free`` the depthless
    ray carves free space to ``reach - truncation`` and the surface goes; with the law off not one
    voxel moves, and beyond the carve the field is untouched either way."""
    model = Tsdf(_carve_spec())
    pose = _optical_pose(0.0, 0.0, 0.0)
    for _ in range(20):  # a face at 1 m, with nothing the rig can reach behind it
        model.integrate(_render_wall(pose, 1.0), None, INTR, pose)
    before_sdf, before_weight = model.sdf.copy(), model.weight.copy()
    blind = np.full((INTR.height, INTR.width), np.nan)  # every pixel silent
    off = DepthLaw(no_depth_free=False, reach_m=2.46)
    assert model.integrate(blind, None, INTR, pose, off) == 0
    assert np.array_equal(model.sdf, before_sdf) and np.array_equal(model.weight, before_weight)

    on = DepthLaw(no_depth_free=True, no_depth_weight=0.5, reach_m=2.46)
    assert on.carve_to_m(model.spec.truncation_m) == pytest.approx(2.36)
    face = np.abs(model.surface(min_weight=1.0)[0][:, 0] - 1.0) < 0.05
    assert face.sum() > 50, "the phantom is in the volume to start with"
    # at max_weight 20 and the carve's own 0.36 a saturated voxel needs ln(2)/ln(1 + 0.36/20)
    # = 39 frames to cross zero: 6.1 s of the node's 6.4 fps, and 60 leaves room for the rims
    for _ in range(60):
        assert model.integrate(blind, None, INTR, pose, on) > 0
    points, _ = model.surface(min_weight=1.0)
    ahead = points[(np.abs(points[:, 1]) < 0.3) & (points[:, 0] > 0.3) & (points[:, 0] < 2.3)]
    assert ahead.shape[0] == 0, "the depthless rays carved the phantom out of their own cone"
    carved = model.weight > before_weight
    assert carved.any(), "the carve is written on the voxels' own weight"


def test_a_depthless_ray_carves_no_further_than_the_source_answers() -> None:
    """The carve stops a truncation short of the source's reach: a surface standing AT the reach
    keeps its halo, and a voxel behind it — which no pixel of this rig ever measured — is not
    touched at all. A reach nobody knows (0) carves nothing, which is what an unmeasured source
    must do."""
    model = Tsdf(_carve_spec())
    pose = _optical_pose(0.0, 0.0, 0.0)
    for _ in range(20):
        model.integrate(_render_wall(pose, 2.8), None, INTR, pose)
    before = model.weight.copy()
    blind = np.full((INTR.height, INTR.width), np.nan)
    assert model.integrate(blind, None, INTR, pose, DepthLaw(True, 0.5, 0.0)) == 0
    assert np.array_equal(model.weight, before), "no reach, no carve"
    for _ in range(60):
        model.integrate(blind, None, INTR, pose, DepthLaw(True, 0.5, 2.46))
    grew = np.argwhere(model.weight > before)
    reached = (grew[:, 0] + 0.5) * model.spec.voxel_m + model.spec.origin[0]
    assert reached.max() < 2.36 + model.spec.voxel_m, "nothing past reach - truncation was touched"
    wall = np.abs(model.surface(min_weight=1.0)[0][:, 0] - 2.8) < 0.05
    assert wall.sum() > 50, "the wall beyond the carve is exactly where it was"


def test_the_surface_takes_a_colour_the_camera_wrote_rather_than_the_lidar_s_black() -> None:
    """The lidar writes field and weight but never a colour, and the readout takes the neighbour
    NEARER the surface — for a beam's own return always the uncoloured one, which is why 1532
    points of the live volume were pure black (2026-09-22). With ``colour_fallback`` the point
    takes the other neighbour's colour when the nearer one was never painted; a crossing between
    two voxels neither camera ever saw stays black, because that is the truth about it."""
    model = Tsdf(_spec())
    # two crossings along x: the nearer voxel uncoloured (the lidar's), the farther one painted
    model.weight[10:12, 10, 10] = 5.0
    model.sdf[10, 10, 10], model.sdf[11, 10, 10] = -0.1, 0.9  # the crossing sits near voxel 10
    model.rgb[11, 10, 10] = (10, 200, 30)
    model.colour_weight[11, 10, 10] = 2.0
    model.weight[20:22, 10, 10] = 5.0  # a second one nobody ever coloured
    model.sdf[20, 10, 10], model.sdf[21, 10, 10] = -0.1, 0.9

    plain, colours = model.surface(min_weight=1.0)
    assert colours.shape[0] == 2 and not colours.any(), "the old readout: both black"
    points, colours = model.surface(min_weight=1.0, colour_fallback=True)
    order = np.argsort(points[:, 0])
    assert tuple(colours[order[0]]) == (10, 200, 30), "the colour a camera did write"
    assert tuple(colours[order[1]]) == (0, 0, 0), "no camera ever painted either neighbour"
    assert plain.shape == points.shape, "a readout rule: the same points, the same places"


def test_the_source_s_own_reach_is_measured_off_its_frames() -> None:
    """Nothing on the wire carries the reach — the publisher's gate is 3.0 m and the stereo rig
    2.46 — and the depth is NaN above it by construction, so the largest finite metre in the last
    frames IS the reach, and can only ever under-state it."""
    reach = ObservedReach(frames=3)
    assert reach.m == 0.0 and reach.frames == 0, "nothing measured yet: carve nothing"
    assert reach.saw(np.array([[np.nan, 1.2], [2.46, np.nan]])) == pytest.approx(2.46)
    reach.saw(np.full((2, 2), np.nan))
    assert reach.m == pytest.approx(2.46), "a blind frame does not lower the reach"
    for _ in range(3):
        reach.saw(np.array([[1.0, 1.1], [1.2, 1.3]]))
    assert reach.m == pytest.approx(1.3) and reach.frames == 3, "the window forgets"
    assert DepthLaw(no_depth_free=True, reach_m=0.0).carve_to_m(0.1) == 0.0
    assert DepthLaw(no_depth_free=False, reach_m=2.46).carve_to_m(0.1) == 0.0
    # a ray that weighs nothing says nothing — and must never reach the weighted average, where
    # a voxel nobody has observed would divide zero by zero
    assert DepthLaw(no_depth_free=True, no_depth_weight=0.0, reach_m=2.46).carve_to_m(0.1) == 0.0
    model = Tsdf(_carve_spec())
    pose = _optical_pose(0.0, 0.0, 0.0)
    blind = np.full((INTR.height, INTR.width), np.nan)
    assert model.integrate(blind, None, INTR, pose, DepthLaw(True, 0.0, 2.46)) == 0
    assert np.isfinite(model.sdf).all() and not model.weight.any()


def test_a_nan_column_carves_its_own_rays_and_leaves_the_wall_beside_it() -> None:
    """The carve is a RAY, not a region: one band of columns goes silent while the rest of the
    frame still measures the wall, and only the wall behind those pixels is carved away. With
    the flag off the same frames change nothing at all."""
    band = slice(INTR.width // 2 - 10, INTR.width // 2 + 10)  # the pixels that go silent
    pose = _optical_pose(0.0, 0.0, 0.0)
    wall = _render_wall(pose, 1.0)
    holed = wall.copy()
    holed[:, band] = np.nan

    def wall_points(model: Tsdf) -> tuple[int, int]:
        """(points of the wall behind the silent band, points of the wall beside it)."""
        points, _ = model.surface(min_weight=1.0)
        at_wall = points[np.abs(points[:, 0] - 1.0) < 0.06]
        # the band's rays at 1 m: (10 px / fx) * 1 m either side of the optical axis
        inside = np.abs(at_wall[:, 1]) < 10.0 / INTR.fx
        return int(inside.sum()), int((~inside).sum())

    for law in (DepthLaw(), DepthLaw(True, 0.5, 2.46)):
        model = Tsdf(_carve_spec())
        for _ in range(20):
            model.integrate(wall, None, INTR, pose, law)
        behind, beside = wall_points(model)
        assert behind > 10 and beside > 50, "the whole wall is in the volume to start with"
        for _ in range(60):
            model.integrate(holed, None, INTR, pose, law)
        after_behind, after_beside = wall_points(model)
        if law.no_depth_free:
            assert after_behind == 0, "the silent rays carved the wall behind them away"
        else:
            assert after_behind == behind, "a NaN pixel touches nothing with the law off"
        assert after_beside >= beside * 0.9, "the wall beside the band is measured and stays"
