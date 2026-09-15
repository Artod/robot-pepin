"""The depth chain as a pipeline: today's stages give the node's numbers bit for bit, the
floor and the walls contribute pairs of their own, the laws read the elevation, and every
stage switches by name."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from pepin.contact import DEPTH_NOISE
from pepin.depth import (
    POOL_MIN_SAMPLES,
    AffineScale,
    CameraPose,
    Intrinsics,
    apply_affine,
    beam_pairs,
    drop_edges,
    edge_mask,
    fit_frame,
    fit_node,
    floor_anchor,
    floor_depth,
    project,
)
from pepin.depth_pipeline import (
    FLOOR_PAIR_STRIDE,
    AffineLaw,
    DepthPipeline,
    EdgeFilter,
    ElevationLaw,
    FloorAnchor,
    FloorGeometry,
    FloorPairs,
    Frame,
    FrameContext,
    FrameLaw,
    LidarAnchor,
    Pairs,
    RangeLawStage,
    RowLaw,
    ScaleField,
    WallAnchor,
    floor_sigma,
    left_of,
    lift_of,
    standard_pipeline,
)

INTR = Intrinsics(fx=457.0, fy=457.0, cx=320.0, cy=180.0, width=640, height=360)
CAM = CameraPose(0.0, 0.0, 1.23, math.radians(26.0))


def _rays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each pixel's ray in base_link per unit optical depth: (forward, left, up) arrays."""
    rows, cols = np.mgrid[0:360, 0:640]
    lift = -(rows - INTR.cy) / INTR.fy
    left = -(cols - INTR.cx) / INTR.fx
    c, s = math.cos(CAM.pitch), math.sin(CAM.pitch)
    return c + s * lift, left, -s + c * lift


def _scene(wall_x: float | None, floor: bool = True) -> np.ndarray:
    """The true optical depth of a room: a vertical wall across the view at ``wall_x`` metres
    ahead (``None`` for none) and the floor (NaN where a ray meets neither)."""
    fwd, _left, up = _rays()
    depth = np.full((360, 640), np.inf)
    if wall_x is not None:
        depth = np.minimum(depth, wall_x / fwd)
    if floor:
        with np.errstate(divide="ignore"):
            to_floor = np.where(up < 0, -CAM.z / up, np.inf)
        depth = np.minimum(depth, to_floor)
    return np.where(np.isfinite(depth), depth, np.nan)


def _network(z: np.ndarray, a: float, b: float, noise: float, seed: int) -> np.ndarray:
    """What a network obeying 1 / z = a / D + b (plus relative noise) says about ``z``."""
    rng = np.random.default_rng(seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 1.0 / ((1.0 / z - b) / a)
    return d * (1.0 + noise * rng.standard_normal(z.shape))


# This file builds its own little room; SCENE_LIDAR_Z is where that room's lidar hangs, not
# the cart's mount (config/lidar.json, measured 2026-09-12). The stages under test read the
# beams they are handed and never the mount, so the scene is free to put them anywhere.
SCENE_LIDAR_Z = 0.2


def _wall_returns(wall_x: float, n: int = 80) -> np.ndarray:
    """The lidar's returns on a wall ``wall_x`` ahead, in scan order, at the scene's lidar
    height."""
    y = np.linspace(-1.2, 1.2, n)
    return np.stack([np.full(n, wall_x), y, np.full(n, SCENE_LIDAR_Z)], axis=1)


def _context(lidar: np.ndarray | None) -> FrameContext:
    return FrameContext(INTR, CAM, lidar=lidar, stamp=1.0)


# ---- today's chain, bit for bit -----------------------------------------------------------------
def _node_chain(
    raw: np.ndarray, lidar: np.ndarray, law: AffineScale
) -> tuple[np.ndarray, np.ndarray | None]:
    """depth_stream._process as it stood: edges -> beam pairs -> law -> drop edges -> floor."""
    edge = edge_mask(raw)
    samples = project(lidar, CAM, INTR)
    pairs = beam_pairs(raw, samples, edge)
    a, b = law.observe(pairs)
    if not law.ready:
        return raw, None
    metric = apply_affine(raw, a, b)
    metric, _ = drop_edges(metric, edge)
    metric, _ = floor_anchor(metric, floor_depth(INTR, CAM), CAM.z)
    return metric, metric


def test_the_pipeline_reproduces_the_node_s_chain_bit_for_bit() -> None:
    """Four views of walls at 1-3.5 m through a noisy network: after every frame the pipeline's
    law is the node's law to the last bit, its output the node's output, and it withholds
    exactly the frames the node withheld."""
    node_law = AffineScale()
    # the affine law alone, fitted on the beams alone, as the node ran then: the floor's and the
    # parallax's pairs are on by default since 2026-09-16 and would feed a pool the node had not
    pipeline = standard_pipeline(
        range_law=False, frame_law=False, floor_pairs=False, parallax_anchor=False
    )
    beams = pipeline.stage("lidar_anchor")
    assert isinstance(beams, LidarAnchor)
    assert beams.sigma_m == 0.0, "and every beam weighing the same, as the node's law was fitted"
    law = pipeline.stage("affine_law")
    assert isinstance(law, AffineLaw)
    assert pipeline.names == [
        "edge_filter",
        "lidar_anchor",
        "floor_pairs",
        "wall_anchor",
        "parallax_anchor",
        "affine_law",
        "ray_law",
        "range_law",
        "frame_law",
        "wall_correct",
        "floor_anchor",
    ]
    assert not pipeline.on("floor_pairs") and not pipeline.on("wall_anchor")
    assert not pipeline.on("wall_correct")
    withheld = 0
    for k, wall_x in enumerate((1.0, 1.5, 2.5, 3.5, 2.0)):
        raw = _network(_scene(wall_x), 1.3, 0.03, noise=0.03, seed=k)
        lidar = _wall_returns(wall_x)
        node_out, published = _node_chain(raw, lidar, node_law)
        result = pipeline.run(raw, _context(lidar))
        assert (law.a, law.b) == (node_law.a, node_law.b)
        assert law.pooled == node_law.pooled and law.ready == node_law.ready
        if published is None:
            assert result.withheld
            withheld += 1
            assert "floor_anchor" not in result.after
        else:
            assert not result.withheld
            assert np.array_equal(result.depth, node_out, equal_nan=True)
    assert withheld >= 1 and law.fitted
    assert law.a == pytest.approx(1.3, rel=0.05) and law.b == pytest.approx(0.03, abs=0.02)


def test_switching_the_edge_filter_off_keeps_the_beams_off_the_edges() -> None:
    """The node computed the mask for the beams whatever the switch said; off, the pipeline's
    depth keeps its edge pixels while the lidar's pairs are the same as with it on."""
    raw = _network(_scene(2.0), 1.3, 0.0, noise=0.0, seed=0)
    raw[100:200, 300:340] *= 0.5  # a box: edges around it
    lidar = _wall_returns(2.0)
    on = DepthPipeline([EdgeFilter(), LidarAnchor()])
    off = DepthPipeline([EdgeFilter(), LidarAnchor()], off=["edge_filter"])
    r_on, r_off = on.run(raw, _context(lidar)), off.run(raw, _context(lidar))
    assert r_on.verdict("edge_filter").on and not r_off.verdict("edge_filter").on
    assert np.isnan(r_on.depth).sum() > np.isnan(r_off.depth).sum() == np.isnan(raw).sum()
    assert r_on.frame.pool is not None and r_off.frame.pool is not None
    assert np.array_equal(r_on.frame.pool.d, r_off.frame.pool.d)
    assert r_on.verdict("lidar_anchor").pairs == r_off.verdict("lidar_anchor").pairs > 20


# ---- the pipeline's own behaviour ----------------------------------------------------------
def test_stages_switch_by_name_and_the_report_counts_them() -> None:
    assert standard_pipeline().switches == {
        "edge_filter": True,
        "lidar_anchor": True,
        "floor_pairs": True,
        "wall_anchor": False,
        "parallax_anchor": True,
        "affine_law": True,
        "ray_law": False,
        "range_law": True,
        "frame_law": True,
        "wall_correct": False,
        "floor_anchor": True,
    }
    # the counting below is of a run that stops at the law, and the floor's pairs fit one with no
    # lidar at all: the two rulers that need no beams are off for it
    pipeline = standard_pipeline(floor_pairs=False, parallax_anchor=False)
    pipeline.set("floor_anchor", False)
    assert not pipeline.on("floor_anchor")
    with pytest.raises(KeyError):
        pipeline.set("no_such_stage", True)
    with pytest.raises(KeyError):
        pipeline.stage("no_such_stage")
    with pytest.raises(ValueError):
        DepthPipeline([EdgeFilter(), EdgeFilter()])
    raw = _network(_scene(2.0), 1.3, 0.0, noise=0.0, seed=0)
    result = pipeline.run(raw, _context(None))  # no lidar: no pairs, no law, withheld
    assert result.withheld and result.verdict("affine_law").withhold
    assert result.verdict("lidar_anchor").pairs == 0  # the run stopped at the law: no floor verdict
    assert [v.stage for v in result.verdicts] == pipeline.names[:6]  # stopped at the law
    with pytest.raises(KeyError):
        result.verdict("floor_anchor")
    stats = pipeline.stats
    assert stats["edge_filter"].frames == 1 and stats["affine_law"].frames == 1
    assert stats["floor_anchor"].frames == 0 and stats["edge_filter"].ms[0] >= 0.0
    report = pipeline.report()
    assert (
        "edge_filter on" in report and "floor_anchor off" in report and "no law yet" not in report
    )
    assert "affine_law on [a 1.00 b +0.000 on 0 pairs (none yet)]" in report
    pipeline.reset_stats()
    assert pipeline.stats["edge_filter"].frames == 0


def test_the_depth_before_a_stage_is_the_last_output_before_it_or_the_raw() -> None:
    """The node's scan is built from the depth as it stood before the floor anchor: the law's
    output by default, the wall correction's when that is on, the raw depth when nothing
    before the anchor ran; a stage the run never reached has no before."""
    law = AffineLaw()
    law.seed(1.4, 0.01)
    raw = _network(_scene(2.0), 1.4, 0.01, noise=0.0, seed=0)
    pipeline = standard_pipeline(law)
    result = pipeline.run(raw, _context(_wall_returns(2.0)))
    assert result.before("floor_anchor") is result.after["frame_law"]
    assert result.before("edge_filter") is result.frame.raw
    assert result.before("lidar_anchor") is result.after["edge_filter"]
    corrected = standard_pipeline(law, wall_correct=True)
    result = corrected.run(raw, _context(_wall_returns(2.0)))
    assert result.before("floor_anchor") is result.after["wall_correct"]
    bare = DepthPipeline([EdgeFilter(), FloorAnchor()], off=["edge_filter"])
    result = bare.run(raw, _context(None))
    assert result.before("floor_anchor") is result.frame.raw
    # withheld at the law: with no lidar and the floor's and the parallax's pairs off (on by
    # default since 2026-09-16) nothing pairs, so no stage after the law has a before
    stopped = standard_pipeline(floor_pairs=False, parallax_anchor=False).run(raw, _context(None))
    assert stopped.withheld
    with pytest.raises(KeyError):
        stopped.before("floor_anchor")


def test_a_seeded_law_publishes_at_once_and_pairs_join_and_carry_their_lift() -> None:
    law = AffineLaw()
    law.seed(1.4, 0.01)
    pipeline = DepthPipeline([law])
    raw = _network(_scene(2.0), 1.4, 0.01, noise=0.0, seed=0)
    result = pipeline.run(raw, _context(None))
    assert not result.withheld and law.held == 1
    assert np.allclose(result.depth[300], _scene(2.0)[300], equal_nan=True)
    assert "seed" in law.describe()
    one = Pairs.of([1.0, 2.0], [1.1, 2.2], lift_of([10, 20], INTR))
    two = Pairs.of([3.0], [3.3], lift_of([30], INTR), weight=[0.5])
    joined = Pairs.join([one, two])
    assert joined is not None and joined.size == 3 and Pairs.join([]) is None
    assert joined.weight.tolist() == [1.0, 1.0, 0.5]
    assert joined.lift[0] == pytest.approx((180.0 - 10.0) / 457.0)


# ---- the floor as a second hoop ------------------------------------------------------------------
def test_the_floor_alone_fits_the_law_with_no_lidar_and_a_box_stays_out() -> None:
    """A floor with a box on it, seen through 1 / z = 1.3 / D + 0.03: three frames in the law
    from the floor alone is within a few percent of the truth — no lidar anywhere. Every pair
    is (network, plane) at one pixel of the stage's lattice; the box's lattice pixels are
    candidates by every other test yet none reaches the pool, and against the same floor
    without the box the pool loses exactly the box's footprint and gains nothing."""
    truth = _scene(None)
    in_box = np.zeros(truth.shape, dtype=bool)
    in_box[220:300, 260:380] = True
    box = np.where(in_box, truth * 0.6, truth)  # a box half a metre high
    expected = floor_depth(INTR, CAM)
    stage = FloorPairs(AffineLaw())
    s = stage.stride
    pools: dict[bool, Pairs] = {}
    raws: dict[bool, np.ndarray] = {}
    for with_box, scene in ((False, truth), (True, box)):
        law = AffineLaw()
        geometry = FloorGeometry()
        stage = FloorPairs(law, geometry)
        pipeline = DepthPipeline([EdgeFilter(), stage, law, FloorAnchor(geometry)])
        for k in range(3):
            raw = _network(scene, 1.3, 0.03, noise=0.02, seed=k)
            result = pipeline.run(raw, _context(None))
        assert not result.withheld and law.fitted
        assert law.a == pytest.approx(1.3, rel=0.05) and law.b == pytest.approx(0.03, abs=0.02)
        assert result.frame.pool is not None and result.frame.pool.size > 500
        pools[with_box], raws[with_box] = result.frame.pool, result.frame.raw
    assert "stride 8" in pipeline.report() and s == 8
    pool, raw = pools[True], raws[True]
    assert 0.0 < pool.weight.max() < 0.1, "a floor pair weighs its own sigma, not a flat share"
    # every pair is the network's depth and the plane's depth at one and the same lattice pixel
    lattice_raw, lattice_z = raw[::s, ::s].ravel(), expected[::s, ::s].ravel()
    order = np.argsort(lattice_raw)
    at = order[np.searchsorted(lattice_raw[order], pool.d)]
    assert np.array_equal(lattice_raw[at], pool.d) and np.array_equal(lattice_z[at], pool.z)
    # the box's lattice pixels are finite, off the plane's edges by nothing but the box, and out
    sampled_box = raw[::s, ::s][in_box[::s, ::s]]
    assert sampled_box.size == 150 and np.isfinite(sampled_box).all()
    assert not np.isin(pool.d, sampled_box).any()
    # the same noise on the plain floor: the box removes its own footprint's pairs, no other
    plain, plain_raw = pools[False], raws[False]
    footprint = plain_raw[::s, ::s][in_box[::s, ::s]]
    footprint_pairs = footprint[np.isin(footprint, plain.d)]
    assert footprint_pairs.size >= 0.9 * sampled_box.size  # the plain floor pairs there
    assert np.isin(pool.d, plain.d).all()
    gone = plain.d[~np.isin(plain.d, pool.d)]
    assert np.array_equal(np.sort(gone), np.sort(footprint_pairs))


def test_the_floor_pairs_judge_floor_with_the_law_once_it_exists() -> None:
    """With a law ready the height test uses the metric depth: a network off by 1.5x is
    still read as floor everywhere the plane is; without the law and with the floor gone the
    stage has nothing to say."""
    law = AffineLaw()
    law.seed(1.5, 0.0)
    stage = FloorPairs(law)
    raw = 1.5 * _scene(None)
    frame = Frame(raw, _context(None))
    pairs = stage.pairs(frame)
    assert pairs is not None and pairs.size > 1000
    assert np.allclose(pairs.d / 1.5, pairs.z)
    sky = Frame(np.full((360, 640), np.nan), _context(None))
    assert stage.pairs(sky) is None


# ---- the walls as a third hoop -------------------------------------------------------------------
def test_the_wall_anchor_extrudes_the_lidar_row_up_a_wall_and_stops_at_a_chair_s_top() -> None:
    """A wall 2 m ahead seen 1.5x too far: the walked pixels' true depth is the wall's optical
    depth at every row above the return. A chair back 1.2 m ahead in the middle columns: the
    walk stops where the network's depth jumps to the wall behind, so no pair of those columns
    lies above the chair's top."""
    fwd, _left, up = _rays()
    truth = _scene(2.0, floor=False)
    chair = truth.copy()
    chair_cols = slice(280, 360)
    chair_depth = 1.2 / fwd
    chair_top = 0.6  # metres above the floor
    on_chair = (CAM.z + chair_depth * up) < chair_top
    chair[:, chair_cols] = np.where(on_chair, chair_depth, truth)[:, chair_cols]
    raw = 1.5 * chair
    returns = _wall_returns(2.0)
    u = project(returns, CAM, INTR)[:, 0]
    hits_chair = (u >= 280) & (u < 360)  # the scene's beam meets the chair there, not the wall
    c, s = math.cos(CAM.pitch), math.sin(CAM.pitch)
    lift = CAM.z - SCENE_LIDAR_Z
    same_column = (c * 1.2 + s * lift) / (c * 2.0 + s * lift)
    returns[hits_chair, 0] = 1.2
    returns[hits_chair, 1] *= same_column
    assert np.allclose(project(returns, CAM, INTR)[:, 0], u)
    anchor = WallAnchor(row_stride=1)
    frame = Frame(raw, _context(returns))
    walk = anchor.walk(frame)
    assert walk is not None and walk.count > 5000
    pairs = anchor.pairs(frame)
    assert pairs is not None and pairs.size == walk.count
    rows = np.rint(INTR.cy - pairs.lift * INTR.fy).astype(int)
    r, k = np.nonzero(walk.walked)
    cols = walk.cols[k]
    assert np.array_equal(np.sort(rows), np.sort(r))
    wall_cols = (cols < 280) | (cols >= 360)
    assert np.allclose(walk.depth[r[wall_cols], k[wall_cols]], truth[r[wall_cols], cols[wall_cols]])
    assert r[wall_cols].min() < 40  # the wall goes on to the top of the picture
    in_chair = ~wall_cols
    assert in_chair.any()
    assert np.allclose(
        walk.depth[r[in_chair], k[in_chair]], chair_depth[r[in_chair], cols[in_chair]], rtol=1e-6
    )
    assert on_chair[r[in_chair], cols[in_chair]].all()  # never above the chair's top
    assert pairs.weight[0] == 0.2 and np.allclose(np.sort(pairs.d), np.sort(raw[r, cols]))
    corrected, touched = WallAnchor(correct=True).correct(raw, frame)
    assert touched == walk.count and np.allclose(corrected[r, cols], walk.depth[r, k])
    assert corrected[359, 0] == raw[359, 0]  # untouched below the return
    assert WallAnchor().correct(raw, frame) == (raw, 0)
    assert "correcting" in WallAnchor(correct=True).describe()
    assert anchor.pairs(Frame(raw, _context(None))) is None
    z = SCENE_LIDAR_Z
    lonely = np.array([[2.0, 0.0, z], [2.0, 0.9, z], [2.0, -0.9, z]])  # no neighbours
    assert anchor.walk(Frame(raw, _context(lonely))) is None


def test_the_walk_stops_where_a_table_top_recedes_and_can_correct_without_pairs() -> None:
    """A table 1.5 m ahead: its front is vertical from the floor to 0.7 m, its top runs back
    to 2.5 m, and the network's depth is continuous across the edge (the front's depth grows
    into the top's without a step). The walk climbs the front and stops within the slope
    window of the top's start; the wall behind is walked to the picture's top. With ``pairs``
    off the anchor still corrects the walked pixels and contributes nothing."""
    fwd, _left, up = _rays()
    wall = _scene(2.0, floor=False)
    front = 1.5 / fwd
    z_front = CAM.z + front * up  # the height each ray meets the front's plane at
    on_front = z_front < 0.7
    # the top: the horizontal plane at 0.7 m, from the front's edge back to the wall
    with np.errstate(divide="ignore"):
        top = np.where(up < 0, (0.7 - CAM.z) / up, np.inf)
    on_top = ~on_front & (top < wall) & (top >= front)
    truth = np.where(on_front, front, np.where(on_top, top, wall))
    raw = 1.4 * truth
    returns = _wall_returns(1.5)  # the scene's beam meets the front across the whole view
    anchor = WallAnchor(row_stride=1)
    walk = anchor.walk(Frame(raw, _context(returns)))
    assert walk is not None
    r, k = np.nonzero(walk.walked)
    cols = walk.cols[k]
    edge_row = np.array([np.flatnonzero(on_front[:, c]).min() for c in cols])
    # the walk reaches the top's edge and passes it by no more than the slope window (the
    # window straddling the edge averages the top's climb with the front's); never the wall
    overshoot = edge_row - r  # rows above the top's edge (negative: still on the front)
    assert 0 <= overshoot.max() <= anchor.slope_window
    assert on_front[r, cols].mean() > 0.9 and not (truth[r, cols] >= wall[r, cols]).any()
    quiet = WallAnchor(pairs=False, correct=True)
    frame = Frame(raw, _context(returns))
    assert quiet.pairs(frame) is None
    corrected, touched = quiet.correct(raw, frame)
    assert touched == walk.count and np.allclose(corrected[r, cols], front[r, cols])
    assert "correcting" in quiet.describe() and "pairs" not in quiet.describe()
    law = AffineLaw()
    law.seed(1.4, 0.0)
    # the seeded law must still be the seed under the correction: the floor's and the parallax's
    # pairs (on by default since 2026-09-16) would refit it, and this test is about the walk
    pipeline = standard_pipeline(law, wall_correct=True, floor_pairs=False, parallax_anchor=False)
    result = pipeline.run(raw, _context(returns))
    assert not result.verdict("wall_anchor").on and result.verdict("wall_correct").pixels > 0
    assert result.verdict("wall_correct").pairs == 0
    # the correction lands after the law: the walked pixels are the plane's metric depth
    assert np.allclose(result.depth[r, cols], front[r, cols])
    assert np.allclose(result.after["affine_law"][r, cols], raw[r, cols] / 1.4)


def test_the_wall_pairs_teach_the_affine_law_the_room_above_the_lidar_row() -> None:
    """A network right at the lidar's row and 20 % too far above it (an error the beams cannot
    see): with the wall pairs in the pool the elevation law finds the term, the row law's
    upper bands differ from its lower ones, and the plain affine law lands in between."""
    lift = lift_of(np.arange(360), INTR)[:, None]
    factor = 1.0 + 0.6 * np.clip(lift - lift[200], 0.0, None)  # grows up the picture
    views = [(_scene(x, floor=False) / factor, _wall_returns(x)) for x in (1.0, 3.5, 2.0)]
    truth, raw, lidar = _scene(2.0, floor=False), views[-1][0], views[-1][1]
    laws = {"affine": AffineLaw(), "elevation": ElevationLaw(), "row": RowLaw(bands=4)}
    for law in laws.values():
        pipeline = DepthPipeline([LidarAnchor(), WallAnchor(), law])
        for view, returns in views:
            result = pipeline.run(view, _context(returns))
        assert not result.withheld
    elevation = laws["elevation"]
    assert isinstance(elevation, ElevationLaw) and elevation.c != 0.0
    assert "c " in elevation.describe()
    row = laws["row"]
    assert isinstance(row, RowLaw) and row.centres.size == 4
    assert row.a_of[-1] != row.a_of[0] and "bands a/b" in row.describe()
    # the corrected depth at the top rows: the elevation and row laws come nearer the truth
    ctx = _context(lidar)
    top = slice(20, 60)
    err = {
        name: float(np.nanmedian(np.abs(law.apply(raw, ctx)[top] - truth[top])))
        for name, law in laws.items()
    }
    assert err["elevation"] < err["affine"] and err["row"] < err["affine"]


def test_the_elevation_and_row_laws_fall_back_to_the_affine_law_on_a_flat_pool() -> None:
    """Pairs of one row only: no elevation spread, so the elevation term stays 0 and the row
    law has no bands — both apply the plain affine law, and a seed holds them too."""
    raw = _network(_scene(2.0), 1.3, 0.03, noise=0.0, seed=0)
    lidar = _wall_returns(2.0)
    for law in (ElevationLaw(), RowLaw()):
        pipeline = DepthPipeline([LidarAnchor(), law])
        for _ in range(3):
            result = pipeline.run(raw, _context(lidar))
        assert not result.withheld and law.fitted
        assert np.array_equal(law.apply(raw, _context(lidar)), apply_affine(raw, law.a, law.b))
    elevation = ElevationLaw()
    elevation.seed(1.3, 0.03)
    assert elevation.ready and elevation.c == 0.0
    elevation.fit(Pairs.of([1.0] * 30, [1.0] * 30, [0.0] * 30))
    assert (elevation.a, elevation.b, elevation.c) == (1.3, 0.03, 0.0)
    rows = RowLaw()
    rows.fit(None)
    assert rows.centres.size == 0 and rows.describe().endswith("(none yet)")
    assert POOL_MIN_SAMPLES == 200


def test_the_floor_geometry_is_recomputed_only_when_the_lean_moves() -> None:
    geometry = FloorGeometry()
    level = geometry.expected(_context(None))
    again = geometry.expected(FrameContext(INTR, CAM, up=np.array([0.001, 0.0, 1.0])))
    assert again is level  # a hair of lean: the cached image
    leaning = geometry.expected(FrameContext(INTR, CAM, up=np.array([0.05, 0.0, 1.0])))
    assert leaning is not level and not np.array_equal(leaning, level, equal_nan=True)
    other = geometry.expected(FrameContext(Intrinsics(400.0, 400.0, 320.0, 180.0, 640, 360), CAM))
    assert other is not leaning


# ---- the law's speed limit ----------------------------------------------------------------------
def _law_pairs(a: float, b: float, n: int = 400) -> Pairs:
    """``n`` pairs a network obeying 1 / z = a / D + b would give over 1-4 m of true depth."""
    z = np.linspace(1.0, 4.0, n)
    d = a / (1.0 / z - b)
    return Pairs.of(d, z, lift_of(np.full(n, INTR.cy), INTR))


def _reach(law: AffineLaw, a: float, b: float, pairs: Pairs) -> float:
    """The relative move of the inverse depth at the pool's ends between the law and (a, b)."""
    ends = np.percentile(pairs.d, (5, 95))
    return float(np.max(np.abs((a / ends + b) - (law.a / ends + law.b)) / (law.a / ends + law.b)))


def test_the_law_walks_to_a_far_fit_no_faster_than_the_slew_allows() -> None:
    """A law with a slew takes its first live fit whole and then moves at most ``slew_per_s``
    of inverse depth a second, saying in its report line where it is walking to; with the slew
    off it takes every fit whole, as the node always did."""
    now = [0.0]
    law = AffineLaw(pool_frames=1, slew_per_s=0.01, clock=lambda: now[0])
    law.fit(_law_pairs(1.75, 0.0))
    assert (law.a, law.b) == pytest.approx((1.75, 0.0), abs=0.02)  # the first fit, whole
    assert "slewing" not in law.describe()
    far = _law_pairs(2.30, -0.15)
    before, asked = (law.a, law.b), _reach(law, 2.30, -0.15, far)
    now[0] += 1.0
    law.fit(far)
    assert asked > 0.05  # the fit really is far away: worth a speed limit
    moved = np.max(
        np.abs(
            (law.a / np.percentile(far.d, (5, 95)) + law.b)
            - (before[0] / np.percentile(far.d, (5, 95)) + before[1])
        )
        / (before[0] / np.percentile(far.d, (5, 95)) + before[1])
    )
    assert moved == pytest.approx(0.01, rel=1e-6)  # one second of the allowance, no more
    assert "slewing to a 2.30 b -0.150" in law.describe()
    for _ in range(400):  # given the seconds, it arrives and stops saying so
        now[0] += 1.0
        law.fit(far)
    assert (law.a, law.b) == pytest.approx((2.30, -0.15), abs=0.01)
    assert "slewing" not in law.describe()
    quick = AffineLaw(pool_frames=1, clock=lambda: now[0])  # slew off: every fit whole
    quick.fit(_law_pairs(1.75, 0.0))
    now[0] += 1.0
    quick.fit(far)
    assert (quick.a, quick.b) == pytest.approx((2.30, -0.15), abs=0.01)


# ---- the law that follows the range ------------------------------------------------------------
def test_the_range_law_is_the_chain_s_live_law_and_the_flag_gives_the_affine_one_back() -> None:
    """The chain ships the range law on, behind the affine one: once its pool fills two bins
    its image is what goes out — not the affine law's — with the holes of the image it was
    handed, and the flag puts the affine law back without a restart."""
    live = standard_pipeline()
    assert live.on("range_law")
    assert live.names.index("affine_law") < live.names.index("range_law")
    stage = live.stage("range_law")
    assert isinstance(stage, RangeLawStage)
    result = None
    for k, wall_x in enumerate((1.0, 1.5, 2.0, 2.5, 3.0, 3.5) * 2):
        raw = _network(_scene(wall_x), 1.3, 0.03, noise=0.03, seed=k)
        raw[100:200, 300:340] = np.nan  # a hole the stages before the law left
        result = live.run(raw, _context(_wall_returns(wall_x)))
    assert result is not None and not result.withheld
    assert stage.fitted and stage.law is not None and stage.law.centres.size >= 2
    affine, ranged = result.after["affine_law"], result.after["range_law"]
    assert not np.array_equal(affine, ranged, equal_nan=True), "the range law's image goes out"
    assert np.array_equal(np.isnan(affine), np.isnan(ranged)), "and the same holes"
    assert "range_law on [D" in live.report() and "pairs]" in live.report()
    live.set("range_law", False)
    live.set("frame_law", False)
    back = live.run(raw, _context(_wall_returns(2.0)))
    assert back.before("floor_anchor") is back.after["affine_law"]
    assert "range_law off" in live.report()


def test_the_range_law_publishes_through_the_affine_law_until_two_bins_fill() -> None:
    """A cart facing one wall at one range fills one bin and learns nothing about range: the
    frame goes out through the affine law rather than being withheld, and the stage says so."""
    law = AffineLaw()
    law.seed(1.3, 0.0)
    stage = RangeLawStage(law)
    # one wall means one bin only while the beams are the whole pool: the floor's pairs (on by
    # default since 2026-09-16) span the range on their own and would fill the second bin
    pipeline = standard_pipeline(law, range_stage=stage, floor_pairs=False, parallax_anchor=False)
    raw = _network(_scene(2.0), 1.3, 0.0, noise=0.0, seed=0)
    result = pipeline.run(raw, _context(_wall_returns(2.0)))
    assert not result.withheld and stage.law is None and not stage.fitted
    assert np.array_equal(result.after["range_law"], result.after["affine_law"], equal_nan=True)
    assert stage.describe().startswith("affine fallback on ")
    assert stage.saved_state() is None
    bare = RangeLawStage(AffineLaw())  # nothing seeded, nothing pooled: nothing to publish
    _depth, verdict = bare.run(raw, Frame(raw, _context(None)))
    assert verdict.withhold and verdict.note == "no law yet"


def test_a_seeded_range_law_is_applied_at_once_and_written_back_whole() -> None:
    """The law the last run saved is the law of the first frame — no warm-up at the affine
    law — and it is written back until the live pool replaces it."""
    from pepin.depth import RangeLaw

    saved = RangeLaw.fit(np.linspace(1.0, 4.0, 2000) * 1.8, np.linspace(1.0, 4.0, 2000))
    assert saved is not None
    stage = RangeLawStage(AffineLaw())
    stage.seed(saved)
    assert stage.ready and not stage.fitted
    raw = _network(_scene(2.0), 1.3, 0.0, noise=0.0, seed=0)
    out, verdict = stage.run(raw, Frame(raw, _context(None)))
    assert not verdict.withhold
    assert np.array_equal(out, saved.apply(raw), equal_nan=True)
    assert stage.describe().endswith("(seed)") and stage.saved_state() == saved.state()


# ---- the law of the frame in hand --------------------------------------------------------------
def test_the_frame_law_corrects_what_the_pool_law_left_on_this_frame() -> None:
    """A camera whose scale changes between frames — a neck that tilts, a room that changes —
    is followed by the frame law and not by the pool's: after a pool fitted at one scale the
    next frame's own beams put its depth right, and the stage's numbers are the pool law's
    residual, near 1.0, not the raw network's 1.8."""
    live = standard_pipeline()
    assert live.on("frame_law")
    assert live.names.index("range_law") < live.names.index("frame_law")
    stage = live.stage("frame_law")
    assert isinstance(stage, FrameLaw)
    for k, wall_x in enumerate((1.0, 1.5, 2.0, 2.5, 3.0, 3.5) * 2):  # a pool at scale 1.3
        live.run(
            _network(_scene(wall_x), 1.3, 0.0, noise=0.01, seed=k), _context(_wall_returns(wall_x))
        )
    assert stage.fits > 0 and stage.a == pytest.approx(1.0, abs=0.1)
    moved = _network(_scene(2.0), 1.9, 0.0, noise=0.0, seed=99)  # the network's scale jumps
    result = live.run(moved, _context(_wall_returns(2.0)))
    beams = result.frame.ctx.beams
    assert beams is not None
    rows, cols = beams[:, 1].astype(int), beams[:, 0].astype(int)
    pooled = float(np.median(result.after["range_law"][rows, cols] / beams[:, 2]))
    own = float(np.median(result.after["frame_law"][rows, cols] / beams[:, 2]))
    assert abs(own - 1.0) < abs(pooled - 1.0) / 3.0, "the frame's own beams win on its own frame"
    assert "frame_law on [a " in live.report() and "frames held" in live.report()
    live.set("frame_law", False)
    back = live.run(moved, _context(_wall_returns(2.0)))
    assert back.before("floor_anchor") is back.after["range_law"]


# ---- two rulers in one fit -------------------------------------------------------------------
def test_a_beam_ships_weighing_a_flat_one_and_sigma_m_weighs_it_by_range_instead() -> None:
    """A beam ships weighing 1 whatever its range — the reference pair the corners are weighed
    against — and ``sigma_m`` above 0 is the live knob that weighs it 1 / sigma^2 in inverse
    depth instead, which puts the weight as z^4 and measured worse (LIDAR_SIGMA_M)."""
    frame = Frame(_network(_scene(2.0), 1.3, 0.0, noise=0.0, seed=0), _context(_wall_returns(2.0)))
    weighed = LidarAnchor(sigma_m=0.015).pairs(frame)
    flat = LidarAnchor().pairs(frame)
    assert weighed is not None and flat is not None
    assert np.array_equal(weighed.z, flat.z) and np.all(flat.weight == 1.0)
    near, far = np.argmin(weighed.z), np.argmax(weighed.z)
    ratio = (weighed.z[far] / weighed.z[near]) ** 4  # 1 / (sigma_m / z^2)^2
    assert weighed.weight[far] / weighed.weight[near] == pytest.approx(ratio, rel=1e-9)
    assert "sigma 1.5 cm" in LidarAnchor(sigma_m=0.015).describe()
    assert "flat" in LidarAnchor().describe()


def test_the_pool_of_two_rulers_is_read_by_weight_and_the_report_says_whose_it_was() -> None:
    """A frame carrying both rulers: 30 beams at weight 1 and 300 corners at 0.03 that claim a
    different scale. The fit follows the beams, the report line names the shares by weight (not
    by pairs), and the same frame with the corners alone fits on them instead."""
    raw = _network(_scene(2.0), 1.3, 0.0, noise=0.0, seed=0)
    frame = Frame(raw, _context(_wall_returns(2.0)))
    beams = LidarAnchor(sigma_m=0.0).pairs(frame)
    assert beams is not None
    keep = np.arange(beams.size) % (beams.size // 30) == 0
    beams = Pairs(
        beams.d[keep], beams.z[keep], beams.weight[keep], beams.lift[keep], beams.left[keep]
    )
    wrong = Pairs.of(  # 300 corners saying every depth is 30 % further than it is
        np.repeat(beams.d, 300 // beams.size + 1)[:300],
        np.repeat(beams.z, 300 // beams.size + 1)[:300] * 1.3,
        np.zeros(300),
        weight=0.03,
    )
    frame.add("lidar_anchor", beams)
    frame.add("parallax_anchor", wrong)
    shares = frame.rulers
    assert shares["lidar_anchor"] == pytest.approx(float(beams.weight.sum()))
    assert shares["parallax_anchor"] == pytest.approx(9.0)
    law = AffineLaw()
    law.seed(1.0, 0.0)
    stage = FrameLaw(law)
    stage.run(raw, frame)
    assert stage.pairs == beams.size + 300
    both = stage.a
    share = 9.0 / (9.0 + float(beams.weight.sum()))
    assert stage.rulers.startswith("rulers: lidar ") and f"parallax {share:.0%}" in stage.rulers
    assert f"{beams.size + 300} pts" in stage.rulers and stage.rulers in stage.describe()
    alone = Frame(raw, _context(None))
    alone.add("parallax_anchor", wrong)
    only = FrameLaw(law)
    only.run(raw, alone)
    assert only.fits == 1, "a frame with no beams at all still fits on the corners"
    assert only.rulers == f"rulers: parallax 100%, {wrong.size} pts"
    assert only.b == 0.0, "a pool with no beams in it fits a scale, never a shift"
    far = np.linspace(0.8, 4.0, 200)  # corners at every range of the room, as the anchor gives
    span = Pairs.of(1.3 * far + 0.05, far, np.zeros(far.size), weight=0.03)
    for gate, shift in ((True, 0.0), (False, None)):
        pool_frame = Frame(raw, _context(None))
        pool_frame.add("parallax_anchor", span)
        stage_gate = FrameLaw(law, shift_needs_beams=gate)
        stage_gate.run(raw, pool_frame)
        assert stage_gate.fits == 1
        if shift is None:
            assert stage_gate.b != 0.0, "the gate off, the spread alone opens the shift"
        else:
            assert stage_gate.b == shift, "the gate on, a beamless pool gets a scale only"
    truth = 1.3  # the network's own scale on this frame: what the beams measure
    assert only.a == pytest.approx(truth / 1.3, abs=0.02), "the corners alone fit their own claim"
    assert abs(both - truth) < abs(only.a - truth) / 3.0, "the beams write the law where they are"


# ---- the frame's law as a field over the picture -----------------------------------------------
def _at_node(
    field: ScaleField, node: tuple[int, int], scale: float, count: int = 24, seed: int = 0
) -> Pairs:
    """``count`` pairs sitting exactly on one node of ``field`` over a 360x640 picture, whose
    network depth is ``scale`` times their true depth (1.8-2.2 m apart, under the shift gate)."""
    rows, cols = field.grid
    row = node[0] / max(rows - 1, 1) * 359.0
    col = node[1] / max(cols - 1, 1) * 639.0
    z = np.linspace(1.8, 2.2, count) + 0.01 * np.random.default_rng(seed).standard_normal(count)
    return Pairs.of(
        scale * z,
        z,
        lift_of(np.full(count, row), INTR),
        1.0,
        left_of(np.full(count, col), INTR),
    )


def test_a_one_node_field_is_the_frame_law_as_it_was_to_the_bit() -> None:
    """The old behaviour reachable: a 1x1 grid is one node, it IS the frame's global fit, and
    the image it publishes is what apply_affine makes of the prior's depth — bit for bit."""
    law = AffineLaw()
    law.seed(1.0, 0.0)
    raw = _network(_scene(2.0), 1.6, 0.0, noise=0.01, seed=3)
    frame = Frame(raw, _context(_wall_returns(2.0)))
    beams = LidarAnchor().pairs(frame)
    assert beams is not None
    frame.add("lidar_anchor", beams)
    stage = FrameLaw(law, grid=(1, 1), clock=lambda: 0.0)  # frozen: no decay to blend in
    out, _verdict = stage.run(raw, frame)
    own = fit_frame(law.apply(beams.d, frame.ctx), beams.z, beams.weight)
    assert own is not None and (stage.a, stage.b) == own
    expected = apply_affine(law.apply(raw, frame.ctx), *own)
    assert np.array_equal(out, np.where(np.isfinite(raw), expected, np.nan), equal_nan=True)
    assert stage.field.grid == (1, 1) and "field 1x1 [" in stage.describe()


def test_the_field_is_its_own_scale_where_it_has_pairs_and_the_global_fit_where_it_has_none() -> (
    None
):
    """A network whose error depends on where in the picture a pixel is — 1.1x at the bottom
    left, 1.6x in the middle, 2.0x at the top — with pairs in two of the nine nodes only. Those
    two come back as their own scale; the nodes nobody measured are the frame's global fit to
    the bit, which is what makes a field safe where the anchors are sparse."""
    stage = FrameLaw(AffineLaw(), grid=(3, 3))
    stage.prior.seed(1.0, 0.0)  # type: ignore[attr-defined]  the identity, so the field is the law
    field = stage.field
    two_nodes = Pairs.join(
        [_at_node(field, (2, 0), 1.1, seed=1), _at_node(field, (1, 1), 1.6, seed=2)]
    )
    assert two_nodes is not None
    stage.fit(two_nodes, _context(None), beams=True)
    a, b = field.nodes
    assert a[2, 0] == pytest.approx(1.1, rel=0.03), "the node that saw the floor's regime"
    assert a[1, 1] == pytest.approx(1.6, rel=0.03), "the node that saw the lidar's row"
    assert a[0, 2] == pytest.approx(stage.a, rel=1e-9), "a node with no pairs IS the global fit"
    assert np.all(b == 0.0), "one depth band: the frame's gate kept the shift shut"
    assert 1.1 < stage.a < 1.6, "the global fit sits between the two regimes, as one law must"
    seen = field.seen
    assert seen[2, 0] == pytest.approx(24.0) and seen[0, 2] == 0.0
    assert field.describe().startswith("3x3 [") and " | " in field.describe()
    # and the picture it publishes: the true depth back at both regimes, which one law cannot do
    z = np.full((360, 640), 2.0)
    published = field.apply(np.where(np.arange(360)[:, None] > 180, 1.1 * z, 1.6 * z))
    assert published[359, 0] == pytest.approx(2.0, rel=0.03)
    assert published[180, 320] == pytest.approx(2.0, rel=0.03)


def test_a_node_starved_of_pairs_carries_its_value_and_decays_toward_the_global_fit() -> None:
    """The carry: a node fed on one frame and starved on the next keeps what it measured,
    decaying toward the frame's global fit over field_carry_tau_s — the only thing holding a
    node's scale on a frame whose anchors landed somewhere else."""
    now = [0.0]
    kept = []
    for gap in (0.0, 2.0, 20.0):
        stage = FrameLaw(
            AffineLaw(), grid=(3, 3), field_prior=1.0, field_carry=10.0, clock=lambda: now[0]
        )
        stage.prior.seed(1.0, 0.0)  # type: ignore[attr-defined]
        field = stage.field
        now[0] = 0.0
        fed = Pairs.join(
            [_at_node(field, (2, 0), 1.1, seed=1), _at_node(field, (1, 1), 1.6, seed=2)]
        )
        assert fed is not None
        stage.fit(fed, _context(None), beams=True)
        measured = float(field.nodes[0][2, 0])
        now[0] = gap  # the next frame, that long later, with nothing at the starved node
        starved = Pairs.join(
            [_at_node(field, (1, 1), 1.6, seed=4), _at_node(field, (0, 2), 1.6, seed=5)]
        )
        assert starved is not None
        stage.fit(starved, _context(None), beams=True)
        kept.append(abs(float(field.nodes[0][2, 0]) - stage.a) / abs(measured - stage.a))
    # with no pairs of its own a node is the carry and the prior averaged: carry / (carry + 1)
    # of the way from the global fit to what it last measured, the carry decaying as exp(-dt/tau)
    share = [10.0 * math.exp(-gap / 2.0) for gap in (0.0, 2.0, 20.0)]
    assert kept[0] == pytest.approx(share[0] / (1 + share[0]), rel=0.1), "no time: it keeps it"
    assert kept[1] == pytest.approx(share[1] / (1 + share[1]), rel=0.1), "one tau: a third gone"
    assert kept[2] < 0.05, "ten time constants later the node is the global fit again"


def test_the_field_prior_weighs_the_pairs_it_claims_to_wherever_the_node_sits() -> None:
    """``field_prior = N`` has to mean N pairs of weight 1 AT THIS NODE — the same N for a node
    looking at the cart's own bumper and for one looking down the corridor. The pseudo-
    observation prior it replaces did not: the skeptic's probe (scratch/_field_refutations.py,
    2026-09-15 — ten pairs of weight 1 saying a = 1.00 against one prior of weight 1 saying
    a = 2.00) measured an effective pull of 0.4 pairs with the cluster at 0.6 m and 36 pairs
    with it at 6 m, so "1.0" named nothing physical and the top nodes of a field, holding a few
    weak corners, could never leave the frame's global fit.

    Read in the fit's own linear parameter, alpha = 1 / a, where a prior IS a weighted average
    and the pull can be read straight off the answer (scratch/field_prior_units.py)."""
    n, a_prior = 10, 2.0
    for pull in (0.3, 1.0, 3.0):
        read = []
        for z0 in (0.6, 1.5, 3.0, 6.0):  # one tight cluster, near and far, the prior far away
            z = np.full(n, z0)
            got = fit_node(z, z, np.ones(n), [(a_prior, 0.0, pull)], 1.0, shift=False)
            assert got is not None
            frac = (1.0 / got[0] - 1.0) / (1.0 / a_prior - 1.0)
            read.append(n * frac / (1.0 - frac))
        assert read == pytest.approx([pull] * 4, rel=1e-9), f"{pull} pairs at every depth"
    # With the shift OPEN the same holds for the scale. A cluster at one depth cannot separate
    # slope from shift and the prior decides that split — but it decides it the same way at
    # every depth, where the pseudo rows read a flat ~15 pairs whatever was asked of them. The
    # SHIFT that comes out is not depth-free and must not be: b is in 1 / m, and the compromise
    # line between a prior and a cluster tilts about the cluster's own inverse depth.
    wide = [
        fit_node(np.full(n, z0), np.full(n, z0), np.ones(n), [(a_prior, 0.0, 1.0)], 1.0, True)
        for z0 in (3.0, 6.0, 9.0)  # under 3 m this cluster's compromise b lands on B_BOUNDS
    ]
    assert all(got is not None for got in wide)
    scales = [got[0] for got in wide if got is not None]
    assert scales == pytest.approx([scales[0]] * 3, rel=1e-9), "one scale, wherever it stood"


def test_a_node_with_no_pairs_is_its_prior_and_two_priors_are_averaged_in_their_weights() -> None:
    """A node that saw nothing must come back as what it was told, whatever the frame's own
    depths were — that is what makes a field safe where the anchors are sparse. And the frame's
    global fit and the node's own carry are two priors, averaged in the parameter they are held
    in (alpha = 1 / a), so the exchange rate cannot tilt one against the other."""
    nothing = (np.array([]), np.array([]), np.array([]))
    for unit in (0.01, 1.0, 100.0):  # the frame's own mean inverse depth squared, any of them
        assert fit_node(*nothing, [(1.7, 0.05, 1.0)], unit) == (1.7, 0.05)
    assert fit_node(*nothing, [], 1.0) is None, "no pairs and no prior constrains nothing"
    two = fit_node(*nothing, [(2.0, 0.0, 1.0), (1.0, 0.0, 3.0)], 0.25)
    assert two is not None and two[0] == pytest.approx(1.0 / ((0.5 + 3.0 * 1.0) / 4.0))


def test_a_floor_pair_weighs_its_own_sigma_and_a_tilted_plane_is_refused() -> None:
    """A floor pair's ruler is the mount's pitch: its sigma grows as the square of the range
    (z^2 / h * sigma_pitch) and its weight against a lidar beam's 1 says so. And the frame must
    prove its floor is one, either way the gate can judge it: the degree gate takes the same
    picture of a plane rolled 3 degrees at 5 degrees and refuses it at 1, and the band gate —
    the default, which asks in metres whether the plane leaves the band its own pixels were
    chosen in — takes the 3-degree roll at 0.6 of a band and refuses an 8-degree one. Both
    count the refusal in the report line."""
    sigma = floor_sigma(np.array([1.0, 2.0, 3.0]), CAM.z, math.radians(1.5), DEPTH_NOISE)
    assert sigma[0] < sigma[1] < sigma[2], "a floor pixel further out is a worse ruler"
    assert sigma[1] / sigma[0] > 2.0, "and worse as the square of the range, not the range"
    rolled = np.array([0.0, math.sin(math.radians(3.0)), math.cos(math.radians(3.0))])
    raw = 1.3 * floor_depth(INTR, CAM, rolled)
    frame = Frame(raw, _context(None))
    wide = FloorPairs(AffineLaw(), plane_band=False)
    pairs = wide.pairs(frame)
    assert pairs is not None and wide.gated == 0 and wide.frames == 1
    assert wide.tilt_deg == pytest.approx(3.0, abs=0.2)
    assert 0.0 < pairs.weight.max() < 0.2, "a floor pair is a fraction of a beam"
    near, far = int(np.argmin(pairs.z)), int(np.argmax(pairs.z))
    assert pairs.weight[near] != pairs.weight[far], "and not the same fraction at every range"
    tight = FloorPairs(AffineLaw(), normal_tol_deg=1.0, plane_band=False)
    assert tight.pairs(Frame(raw, _context(None))) is None
    assert tight.gated == 1 and "1/1 frames out" in tight.describe()
    assert "plane gate 1 deg" in tight.describe()
    level = FloorPairs(AffineLaw(), normal_tol_deg=1.0, plane_band=False)
    assert level.pairs(Frame(1.3 * floor_depth(INTR, CAM), _context(None))) is not None
    assert level.gated == 0 and level.tilt_deg < 0.1
    # the band gate reads the same rolls in metres of the pixels' own band
    band = FloorPairs(AffineLaw())
    assert band.pairs(Frame(raw, _context(None))) is not None
    assert band.tilt_deg == pytest.approx(3.0, abs=0.2) and 0.3 < band.plane_off < 0.9
    assert "plane gate band" in band.describe() and "band <= 20 cm" in band.describe()
    steep = np.array([0.0, math.sin(math.radians(8.0)), math.cos(math.radians(8.0))])
    hard = FloorPairs(AffineLaw())
    assert hard.pairs(Frame(1.3 * floor_depth(INTR, CAM, steep), _context(None))) is None
    assert hard.gated == 1 and hard.plane_off > 1.0


def test_a_door_two_metres_ahead_is_not_the_floor_however_wide_its_band_is() -> None:
    """The door tapes of 2026-09-15, in the small: a camera 1.2 m up pitched 24 degrees down, a
    closed door 2 m ahead. The floor is plainly in view — everything out to 2.3 m — but the rows
    just under the horizon look at the DOOR, and their floor depth runs to hundreds of metres,
    where the band that decides "is this pixel on the floor?" is metres wide and admits it. Those
    pixels stand over a metre up at the same 2 m of range, they own the plane fitted to the
    candidates, and both gates then throw the frame's real floor away — 15.5 degrees of lean and
    1.6 bands of departure, 25.1 degrees to the old degree gate. Capping the band at what still
    separates a floor from what stands on it takes the door out and leaves 1501 pairs of honest
    floor at 2.1 degrees."""
    intr = Intrinsics(fx=340.0, fy=340.0, cx=320.0, cy=180.0, width=640, height=360)
    cam = CameraPose(0.0, 0.0, 1.20, math.radians(24.0))  # the horizon at row 29, inside the view
    ctx = FrameContext(intr, cam)
    expected = floor_depth(intr, cam)
    rows = np.mgrid[0:360, 0:640][0]
    forward = math.cos(cam.pitch) + math.sin(cam.pitch) * lift_of(rows, intr)
    door = 2.0 / forward  # depth along each ray to a vertical plane 2 m ahead
    seen = np.fmin(np.where(np.isfinite(expected), expected, np.inf), door)
    raw = 1.3 * seen
    assert np.nanmax(np.where(expected <= door, expected, np.nan)) == pytest.approx(2.31, abs=0.05)
    assert np.nanmax(expected) > 100.0, "and rows whose floor is hundreds of metres out"

    uncapped = FloorPairs(AffineLaw(), band_max_m=0.0)
    assert uncapped.pairs(Frame(raw, ctx)) is None, "the door passes a band metres wide"
    assert uncapped.gated == 1 and uncapped.plane_off > 1.5 and uncapped.tilt_deg > 15.0
    old = FloorPairs(AffineLaw(), band_max_m=0.0, plane_band=False)
    assert old.pairs(Frame(raw, ctx)) is None and old.tilt_deg > 25.0
    capped = FloorPairs(AffineLaw())
    pairs = capped.pairs(Frame(raw, ctx))
    assert pairs is not None and capped.gated == 0
    assert capped.tilt_deg < 3.0 and capped.plane_off < 0.5 and pairs.size > 1400
    assert pairs.z.max() < 3.0, "and what it kept is the floor it can see, not the door"


def test_a_frame_without_beams_decays_back_to_the_law_behind_it() -> None:
    """A frame carrying too few pairs holds the last frame's law, which fades toward the pool's
    over the time constant; with no law of its own the prior's image goes out untouched, and
    with no law at all the frame is withheld."""
    now = [0.0]
    law = AffineLaw()
    law.seed(1.3, 0.0)
    stage = FrameLaw(law, tau_s=2.0, clock=lambda: now[0])
    raw = _network(_scene(2.0), 1.6, 0.0, noise=0.0, seed=0)
    blind = Frame(raw, _context(None))
    out, verdict = stage.run(raw, blind)  # no beams: the seeded affine law stands alone
    assert not verdict.withhold and stage.held == 1 and stage.weight == 0.0
    assert np.array_equal(out, law.apply(raw, blind.ctx), equal_nan=True)
    assert stage.describe() == "prior stands, 1/1 frames held"
    seen = Frame(raw, _context(_wall_returns(2.0)))
    seen.pairs.append(LidarAnchor().pairs(seen) or Pairs.of(np.empty(0), np.empty(0), np.empty(0)))
    own, _verdict = stage.run(raw, seen)
    assert stage.fits == 1 and stage.weight == 1.0 and stage.pairs > 0
    now[0] += 2.0  # one time constant later, with nothing to fit
    faded, _verdict = stage.run(raw, Frame(raw, _context(None)))
    assert stage.weight == pytest.approx(math.exp(-1.0), abs=1e-6)
    between = 1.0 / (stage.weight / own + (1.0 - stage.weight) / law.apply(raw, blind.ctx))
    assert np.allclose(faded, between, equal_nan=True)
    bare = FrameLaw(AffineLaw())  # nothing of its own, nothing behind it
    _depth, verdict = bare.run(raw, Frame(raw, _context(None)))
    assert verdict.withhold and verdict.note == "no law yet"


# ---- the parallax anchor may not wait inside the frame path -------------------------------------
from pepin.tsdf import RigidPose  # noqa: E402  the rigid transform a motion source hands back


class BlockingPoser:
    """A motion source whose map lookup blocks, the way TF does when it cannot cover a frame's
    stamp: the odometry answers at once, the map costs a whole timeout."""

    def __init__(self, timeout_s: float = 0.2) -> None:
        self.timeout_s = timeout_s
        self.waited = 0
        self.asked_recent = 0

    def motion(self, from_stamp: float, to_stamp: float) -> RigidPose:
        return RigidPose(np.eye(3), np.array([0.2 * (to_stamp - from_stamp), 0.0, 0.0]))

    def map_motion(self, from_stamp: float, to_stamp: float) -> None:
        self.waited += 1
        time.sleep(self.timeout_s)
        return None

    def map_motion_recent(self, from_stamp: float, to_stamp: float, max_age_s: float) -> None:
        self.asked_recent += 1
        return None  # the newest map pose is older than max_age_s: the odometry answers


def test_the_parallax_anchor_does_not_wait_for_a_map_pose_it_cannot_get() -> None:
    """The live stall of 2026-09-15, as a test: the blocking ask cost 0.2 s a window and took
    the stream from 8.7 to 1.5 frames/s. The default ask must return in under a millisecond and
    still produce a motion — the odometry's — and count the fallback."""
    from pepin.depth_pipeline import ParallaxAnchor

    anchor = ParallaxAnchor()
    assert anchor.map_wait is False
    source = BlockingPoser()
    ctx = FrameContext(INTR, CAM, motion=source, stamp=10.0)
    started = time.perf_counter()
    moved, whose = anchor._moved(ctx, 9.5, 10.0, ask_tracker=True)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.001  # the frame path waited for nothing at all
    assert whose == "odom" and moved is not None
    assert source.waited == 0 and source.asked_recent == 1
    assert anchor.stale == 1  # the report line's "map pose stale -> odom"

    anchor.map_wait = True  # the old behaviour, still reachable for an A/B
    started = time.perf_counter()
    anchor._moved(ctx, 9.5, 10.0, ask_tracker=True)
    assert time.perf_counter() - started >= 0.15
    assert source.waited == 1


# ---- what the field must never publish ----------------------------------------------------------
def test_a_law_that_is_not_a_number_never_reaches_the_picture() -> None:
    """A NaN law is not a bad law, it is a blind robot: the field blends its nodes with two
    matrix products, and a matmul does not skip a zero membership — 0 * NaN is NaN, so ONE bad
    node of nine takes every pixel of the image with it, not its own corner. So neither fit
    hands one back: a frame fit and a node fit that do not come back finite are "no law", which
    the callers already hold frames for."""
    blind = np.full(40, np.inf)  # a ruler and a network that both read infinitely far
    assert fit_frame(blind, blind, np.ones(40)) is None
    assert fit_node(blind, blind, np.ones(40), [(math.nan, math.nan, 1.0)], 0.25) is None
    assert fit_node(blind, blind, np.ones(40), [(1.5, 0.0, 1.0)], 0.25) == (1.5, 0.0)

    field = ScaleField((3, 3), prior=1.0, carry=0.0)
    rng = np.random.default_rng(5)
    z = rng.uniform(1.0, 3.0, 300)
    where = (rng.uniform(0, 359, 300), rng.uniform(0, 639, 300))
    field.fit(1.5 * z, z, np.ones(300), *where, (360, 640), (1.5, 0.0), 0.0, shift=False)
    assert np.isfinite(field.apply(np.full((360, 640), 2.0))).all()
    kept = field.nodes
    field.fit(1.5 * z, z, np.ones(300), *where, (360, 640), (math.nan, 0.0), 0.0, shift=False)
    assert np.array_equal(field.nodes[0], kept[0]), "a NaN global fit leaves the field alone"
    assert np.isfinite(field.apply(np.full((360, 640), 2.0))).all()


def test_the_carry_runs_out_even_with_nothing_to_decay_against() -> None:
    """The carry decays against the pull toward the global fit. At field_prior 0 there is no
    such pull, and a weight of 1e-13 holds an unconstrained node just as completely as a weight
    of 1: without a floor under the decay a starved node reads its own last value for ever.
    Past FIELD_CARRY_SPENT of itself the carry is dropped outright and the node is the global
    fit again, at either setting of the prior."""
    rng = np.random.default_rng(3)
    where = (rng.uniform(0, 359, 500), rng.uniform(0, 639, 500))
    z = rng.uniform(1.0, 3.0, 500)
    starved = (np.full(40, 10.0), np.full(40, 10.0))  # every pair in the top-left node
    held = {}
    for prior in (1.0, 0.0):
        field = ScaleField((3, 3), prior=prior, carry=1.0, carry_tau_s=2.0)
        field.fit(1.5 * z, z, np.ones(500), *where, (360, 640), (1.5, 0.0), 0.0, shift=False)
        field.fit(3.0 * z[:40], z[:40], np.ones(40), *starved, (360, 640), (3.0, 0.0), 1.0, False)
        held[prior] = float(field.nodes[0][2, 2])
        field.fit(3.0 * z[:40], z[:40], np.ones(40), *starved, (360, 640), (3.0, 0.0), 30.0, False)
        assert field.nodes[0][2, 2] == pytest.approx(3.0), "thirty seconds later it is the fit"
    assert 1.5 < held[1.0] < 3.0, "against a prior of 1 the carry is one of two pulls and decays"
    assert held[0.0] == pytest.approx(1.5), (
        "against no prior it is the only pull, whatever it weighs"
    )


def test_recutting_the_grid_live_keeps_the_frame_s_own_law() -> None:
    """field_grid is a live flag and a new grid starts every node again from the next frame's
    fit — but between the flag and that fit the field is a grid of ones, the identity, while the
    stage still says a law of its own stands. The stage's own global two numbers hold that gap,
    so flipping the flag changes the law's SHAPE and never whether the picture is corrected at
    all; on a cart whose beams have gone quiet the gap is however long the frames are held."""
    law = AffineLaw()
    law.seed(1.0, 0.0)
    stage = FrameLaw(law, grid=(3, 3))
    raw = _network(_scene(2.0), 1.6, 0.0, noise=0.0, seed=0)
    seen = Frame(raw, _context(_wall_returns(2.0)))
    seen.pairs.append(LidarAnchor().pairs(seen) or Pairs.of(np.empty(0), np.empty(0), np.empty(0)))
    before, _verdict = stage.run(raw, seen)
    assert stage.fits == 1 and stage.field.fitted
    stage.field.grid = (2, 2)  # the flag, between two frames
    assert not stage.field.fitted
    after = stage.apply(raw, seen.ctx)
    assert not np.allclose(
        after[np.isfinite(after)], law.apply(raw, seen.ctx)[np.isfinite(after)]
    ), "the prior's own depth must not go out as if no frame had ever spoken"
    assert np.allclose(after[np.isfinite(after)], before[np.isfinite(after)], rtol=0.05)


def test_the_report_line_says_which_nodes_are_empty_and_which_are_at_a_bound() -> None:
    """A field of nine laws had no way of saying what the single law has always said: this one
    is the bound, not a measurement. The report line counts both — the nodes that saw no pair
    (those are the frame's global fit) and the nodes that came back pinned."""
    field = ScaleField((3, 3), prior=1.0, carry=0.0)
    fed = _at_node(field, (1, 1), 1.6, seed=1)
    rows = INTR.cy - fed.lift * INTR.fy
    cols = INTR.cx - fed.left * INTR.fx
    field.fit(fed.d, fed.z, fed.weight, rows, cols, (360, 640), (1.6, 0.0), 0.0, shift=False)
    assert "empty" in field.describe() and "AT BOUND" not in field.describe()
    assert field.describe().endswith("8 empty"), field.describe()
    # A node whose pairs ask for a shift past what a lens can do: the bound does the fitting.
    z = np.linspace(1.2, 4.0, 60)
    beyond = 1.6 / (1.0 / z - 0.35)  # the law 1 / z = 1.6 / D + 0.35, past B_BOUNDS
    field.fit(
        beyond,
        z,
        np.ones(60),
        np.full(60, 180.0),
        np.full(60, 320.0),
        (360, 640),
        (1.6, 0.0),
        0.0,
        shift=True,
    )
    assert "AT BOUND" in field.describe(), field.describe()


def test_the_floor_bootstraps_from_the_floor_and_not_from_the_median_of_the_clutter() -> None:
    """Which pixels are floor is judged with the law as it stands, and before any law exists
    with a scale bootstrapped from the pixels themselves. That bootstrap cannot be started at
    the MEDIAN of every candidate below the horizon: everything down there that is not floor
    stands ON the floor and is therefore NEARER than it, so the clutter's network-over-plane
    ratio is always LOWER and the median of the mixture is biased low by construction — and the
    turns that follow cannot climb back out, the height band being about a tenth of the depth
    wide and admitting whatever scale it is handed.

    Here two thirds of the picture below the horizon is furniture — six things of six heights,
    none of them as large as the floor — which drags the median of the candidates well under the
    floor's own 1.60 while leaving the floor the biggest single population in the picture. The
    starts run from the floor's side as well, and the one that ends up holding the most pixels
    wins."""
    truth = _scene(None)
    scene = truth.copy()
    for k, factor in enumerate((0.30, 0.42, 0.54, 0.66, 0.78, 0.90)):
        scene[120:360, k * 106 : (k + 1) * 106] = truth[120:360, k * 106 : (k + 1) * 106] * factor
    raw = _network(scene, 1.6, 0.0, noise=0.01, seed=11)
    expected = floor_depth(INTR, CAM)
    s = FLOOR_PAIR_STRIDE
    seen = np.isfinite(expected[::s, ::s]) & np.isfinite(raw[::s, ::s])
    ratios = (raw[::s, ::s] / expected[::s, ::s])[seen]
    assert float(np.median(ratios)) < 1.3, "the median of the candidates is not the floor's 1.60"

    stage = FloorPairs(AffineLaw())  # no law anywhere: the bootstrap is all there is
    pairs = stage.pairs(Frame(raw, _context(None)))
    assert pairs is not None and stage.gated == 0
    assert float(np.median(pairs.d / pairs.z)) == pytest.approx(1.6, rel=0.03)
    assert pairs.size > 800, "and it is the floor it held, not a corner of it"
