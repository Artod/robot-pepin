"""The depth chain as a pipeline: today's stages give the node's numbers bit for bit, the
floor and the walls contribute pairs of their own, the laws read the elevation, and every
stage switches by name."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pepin.depth import (
    POOL_MIN_SAMPLES,
    AffineScale,
    CameraPose,
    Intrinsics,
    apply_affine,
    beam_pairs,
    drop_edges,
    edge_mask,
    floor_anchor,
    floor_depth,
    project,
)
from pepin.depth_pipeline import (
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
    WallAnchor,
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
    pipeline = standard_pipeline(range_law=False, frame_law=False)  # the affine law alone,
    # as the node ran then
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
    pipeline = standard_pipeline()
    assert pipeline.switches == {
        "edge_filter": True,
        "lidar_anchor": True,
        "floor_pairs": False,
        "wall_anchor": False,
        "parallax_anchor": False,
        "affine_law": True,
        "ray_law": False,
        "range_law": True,
        "frame_law": True,
        "wall_correct": False,
        "floor_anchor": True,
    }
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
    stopped = standard_pipeline().run(raw, _context(None))  # withheld at the law
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
    assert pool.weight[0] == 0.1
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
    pipeline = standard_pipeline(law, wall_correct=True)
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
    pipeline = standard_pipeline(law, range_stage=stage)
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
