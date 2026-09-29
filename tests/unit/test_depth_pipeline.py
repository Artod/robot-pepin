"""The depth chain as a pipeline: the stages give the node's numbers bit for bit, and every
stage switches by name."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pepin.depth import (
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
    FloorAnchor,
    FloorGeometry,
    Frame,
    FrameContext,
    LidarAnchor,
    Pairs,
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
    pipeline = standard_pipeline()
    beams = pipeline.stage("lidar_anchor")
    assert isinstance(beams, LidarAnchor)
    assert beams.sigma_m == 0.0, "and every beam weighing the same, as the node's law was fitted"
    law = pipeline.stage("affine_law")
    assert isinstance(law, AffineLaw)
    assert pipeline.names == ["edge_filter", "lidar_anchor", "affine_law", "floor_anchor"]
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
    assert [v.stage for v in result.verdicts] == pipeline.names[:3]  # stopped at the law
    with pytest.raises(KeyError):
        result.verdict("floor_anchor")
    stats = pipeline.stats
    assert stats["edge_filter"].frames == 1 and stats["affine_law"].frames == 1
    assert stats["floor_anchor"].frames == 0 and stats["edge_filter"].ms[0] >= 0.0
    report = pipeline.report()
    assert "edge_filter on" in report and "floor_anchor off" in report
    assert "affine_law on [a 1.00 b +0.000 on 0 pairs (none yet)]" in report
    pipeline.reset_stats()
    assert pipeline.stats["edge_filter"].frames == 0


def test_the_depth_before_a_stage_is_the_last_output_before_it_or_the_raw() -> None:
    """The node's scan is built from the depth as it stood before the floor anchor: the law's
    output by default, the raw depth when nothing before the anchor ran; a stage the run never
    reached has no before."""
    law = AffineLaw()
    law.seed(1.4, 0.01)
    raw = _network(_scene(2.0), 1.4, 0.01, noise=0.0, seed=0)
    pipeline = standard_pipeline(law)
    result = pipeline.run(raw, _context(_wall_returns(2.0)))
    assert result.before("floor_anchor") is result.after["affine_law"]
    assert result.before("edge_filter") is result.frame.raw
    assert result.before("lidar_anchor") is result.after["edge_filter"]
    bare = DepthPipeline([EdgeFilter(), FloorAnchor()], off=["edge_filter"])
    result = bare.run(raw, _context(None))
    assert result.before("floor_anchor") is result.frame.raw
    # withheld at the law: with no lidar nothing pairs, so no stage after the law has a before
    stopped = standard_pipeline().run(raw, _context(None))
    assert stopped.withheld
    with pytest.raises(KeyError):
        stopped.before("floor_anchor")


def test_a_watching_law_fits_and_reports_and_leaves_the_depth_as_it_was_measured() -> None:
    """Under a metric source the law is a witness: the lidar's pairs still fit it, the report
    says what it found, the depth goes out untouched and no frame waits for a law."""
    law = AffineLaw()
    law.watching = True
    pipeline = DepthPipeline([EdgeFilter(), LidarAnchor(), law], off=["edge_filter"])
    raw = _network(_scene(2.0), 1.4, 0.01, noise=0.0, seed=0)  # a depth the law WOULD rescale
    result = pipeline.run(raw, _context(_wall_returns(2.0)))
    assert not result.withheld, "a watching law never withholds, fitted or not"
    assert result.depth is result.after["lidar_anchor"], "the depth is not touched"
    assert law.pooled > 0 and "watching" in law.describe()
    law.watching = False
    applied = pipeline.run(raw, _context(_wall_returns(2.0)))
    assert applied.withheld or applied.depth is not applied.after["lidar_anchor"]


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


def test_the_pool_law_reads_the_beams_alone_or_a_capped_pool_without_them() -> None:
    """With beams in the frame the law gets the beams and nothing else; without them, every
    other anchor's block thinned to POOL_CAP_PER_SOURCE evenly spaced pairs with its total
    weight unchanged."""
    from pepin.depth_pipeline import POOL_CAP_PER_SOURCE, Frame, Pairs

    n = 10_000
    floor = Pairs.of(np.linspace(1, 3, n), np.linspace(1, 3, n), np.zeros(n), 0.1)
    frame = Frame(raw=np.ones((4, 4)), ctx=None)  # type: ignore[arg-type]
    frame.add("another_anchor", floor)
    capped = frame.pool_capped()
    assert capped is not None and frame.pool is not None
    assert capped.size == POOL_CAP_PER_SOURCE and frame.pool.size == n
    assert np.isclose(float(np.sum(capped.weight)), float(np.sum(frame.pool.weight)))
    assert float(capped.d[0]) == 1.0 and np.isclose(float(capped.d[-1]), 3.0)
    frame.add("lidar_anchor", Pairs.of(np.ones(30), np.ones(30), np.zeros(30), 1.0))
    beams = frame.pool_capped()
    assert beams is not None and beams.size == 30 and frame.pool.size == n + 30
