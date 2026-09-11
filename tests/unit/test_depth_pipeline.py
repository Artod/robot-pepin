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
    LidarAnchor,
    Pairs,
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


def _wall_returns(wall_x: float, n: int = 80) -> np.ndarray:
    """The lidar's returns on a wall ``wall_x`` ahead, in scan order, at the lidar's height."""
    y = np.linspace(-1.2, 1.2, n)
    return np.stack([np.full(n, wall_x), y, np.full(n, 0.2)], axis=1)


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
    pipeline = standard_pipeline()
    law = pipeline.stage("affine_law")
    assert isinstance(law, AffineLaw)
    assert pipeline.names == [
        "edge_filter",
        "lidar_anchor",
        "floor_pairs",
        "wall_anchor",
        "affine_law",
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
        "affine_law": True,
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
    assert [v.stage for v in result.verdicts] == pipeline.names[:5]  # stopped at the law
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
    assert result.before("floor_anchor") is result.after["affine_law"]
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
    hits_chair = (u >= 280) & (u < 360)  # the lidar at 0.2 m meets the chair there, not the wall
    c, s = math.cos(CAM.pitch), math.sin(CAM.pitch)
    same_column = (c * 1.2 + s * (CAM.z - 0.2)) / (c * 2.0 + s * (CAM.z - 0.2))
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
    lonely = np.array([[2.0, 0.0, 0.2], [2.0, 0.9, 0.2], [2.0, -0.9, 0.2]])  # no neighbours
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
    returns = _wall_returns(1.5)  # the lidar meets the front at 0.2 m across the view
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
