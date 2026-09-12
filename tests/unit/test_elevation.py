"""The law of the ray: a network error that grows with elevation is recovered where one affine
law cannot, the law is the camera's and not the head's (the same scene at two pitches fits the
same law), it survives a save and a load, it says when it sits at its bound, and it falls back
to the affine law wherever the pool cannot carry it."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pepin.depth import (
    A_BOUNDS,
    POOL_MIN_SAMPLES,
    CameraPose,
    Intrinsics,
    apply_affine,
    fit_affine,
    load_law,
    load_ray,
    project,
    save_law,
)
from pepin.depth_pipeline import (
    AffineLaw,
    FrameContext,
    Pairs,
    RayLaw,
    left_of,
    lift_of,
    standard_pipeline,
)
from pepin.elevation import (
    MIN_RAY_SPAN,
    RAY_SCALE,
    RayGain,
    fit_ray,
    ray_angles,
    separable,
    usable_degree,
)

INTR = Intrinsics(fx=395.0, fy=395.0, cx=320.0, cy=180.0, width=640, height=360)
CAM = CameraPose(0.0, 0.0, 1.23, math.radians(26.0))
SCALE = 1.6  # the network is that much too far on the optical axis
TILT = (0.35, 0.30)  # and more of it up the cone: the ratio's first and second order in radians


def _ratio(elevation: np.ndarray) -> np.ndarray:
    """The network's depth over the true depth for rays at these elevations (radians): a
    smooth rise up the cone that is not itself the law's shape (the law is a polynomial in the
    inverse-depth slope, this one in the ratio)."""
    return SCALE * (1.0 + TILT[0] * elevation + TILT[1] * elevation**2)


def _rays(cam: CameraPose = CAM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each pixel's ray in base_link per unit optical depth: (forward, left, up)."""
    rows, cols = np.mgrid[0:360, 0:640]
    lift, left = lift_of(rows, INTR), left_of(cols, INTR)
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    return c + s * lift, left, -s + c * lift


BEARINGS = np.linspace(-0.6, 0.6, 240)  # radians across the view
RADII = 1.0 + 2.4 * (BEARINGS - BEARINGS[0]) / (BEARINGS[-1] - BEARINGS[0])  # metres, near to far
LIDAR_Z = 0.383


def _radius(bearing: np.ndarray) -> np.ndarray:
    """How far the room's wall stands at these bearings (radians, left positive)."""
    return np.interp(bearing, BEARINGS, RADII)


def _scene() -> np.ndarray:
    """A room's true optical depth: a vertical wall standing 1 to 3.4 m away depending on the
    bearing (so the lidar's own returns land over a whole cone of elevations, as they do in a
    room) and the floor under it."""
    fwd, left, up = _rays()
    with np.errstate(divide="ignore", invalid="ignore"):
        flat = np.hypot(fwd, left)
        wall = _radius(np.arctan2(left, fwd)) / flat
        depth = np.minimum(wall, np.where(up < 0, -CAM.z / up, np.inf))
    return np.where(np.isfinite(depth) & (depth > 0.2) & (depth < 8.0), depth, np.nan)


def _returns() -> np.ndarray:
    """The lidar's returns on that wall, in scan order, at the lidar's height."""
    x = RADII * np.cos(BEARINGS)
    y = RADII * np.sin(BEARINGS)
    return np.stack([x, y, np.full(x.size, LIDAR_Z)], axis=1)


def _network(z: np.ndarray, noise: float = 0.01, seed: int = 7) -> np.ndarray:
    """What a network whose error grows with the ray's elevation says about the depth ``z``."""
    rng = np.random.default_rng(seed)
    elevation, _azimuth = ray_angles(lift_of(np.arange(z.shape[0]), INTR)[:, None])
    return z * _ratio(elevation) * (1.0 + noise * rng.standard_normal(z.shape))


def _pairs(stride: int = 6) -> Pairs:
    """Every ``stride``-th pixel of the scene as a (network, true, ray) pair."""
    z = _scene()
    d = _network(z)
    rows, cols = np.mgrid[0:360, 0:640]
    ok = np.isfinite(z[::stride, ::stride])
    return Pairs.of(
        d[::stride, ::stride][ok],
        z[::stride, ::stride][ok],
        lift_of(rows[::stride, ::stride][ok], INTR),
        1.0,
        left_of(cols[::stride, ::stride][ok], INTR),
    )


def _bands(elevation: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """The rays cut into a bottom, a middle and a top third of the picture's elevation."""
    lo, hi = elevation.min(), elevation.max()
    edges = np.linspace(lo, hi + 1e-9, 4)
    return [
        (name, (elevation >= edges[k]) & (elevation < edges[k + 1]))
        for k, name in enumerate(("bottom", "middle", "top"))
    ]


def _band_error(metric: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """The median relative error of a corrected image against the truth, by elevation third."""
    elevation, _azimuth = ray_angles(lift_of(np.arange(truth.shape[0]), INTR))
    out = {}
    for name, sel in _bands(elevation):
        rows = np.flatnonzero(sel)
        err = np.abs(metric[rows] / truth[rows] - 1.0)
        out[name] = float(np.nanmedian(err))
    return out


# ---- what the affine law cannot do --------------------------------------------------------------
def test_the_ray_law_recovers_an_elevation_error_the_affine_law_cannot() -> None:
    """A wall and a floor seen through a network whose error grows up the cone: the ray law is
    within 2 % of the truth in every third of the picture, one affine law in none of them."""
    pool = _pairs()
    assert pool.size > POOL_MIN_SAMPLES
    truth = _scene()
    raw = _network(truth)
    elevation, _azimuth = ray_angles(pool.lift, pool.left)
    gain = fit_ray(pool.d, pool.z, elevation, pool.weight)
    assert gain is not None and not gain.clipped
    rows = lift_of(np.arange(360), INTR)[:, None]
    ray_metric = gain.apply(raw, ray_angles(rows)[0])
    a, b = fit_affine(pool.d, pool.z, pool.weight)
    affine_metric = apply_affine(raw, a, b)
    ray_err = _band_error(ray_metric, truth)
    affine_err = _band_error(affine_metric, truth)
    assert max(ray_err.values()) < 0.02, (ray_err, affine_err)
    # one affine law splits the difference: right in the middle, wrong at both ends
    assert affine_err["bottom"] > 0.07 and affine_err["top"] > 0.10, affine_err
    # and the gain says what it found: more scale up the cone than down it
    assert gain.scale_at(20.0) > gain.scale_at(-20.0) > 0.5


def test_the_law_is_the_camera_s_and_not_the_head_s() -> None:
    """The same points seen at two head pitches: what the network does at a given world height
    changes with the pitch, what it does along a given ray does not — so the law fitted at one
    pitch is the law fitted at the other, inside the angles both of them saw."""
    grid = np.stack(
        np.meshgrid(
            np.array([1.0, 1.6, 2.4, 3.4]), np.linspace(-1.0, 1.0, 21), np.linspace(0.0, 2.0, 21)
        ),
        axis=-1,
    ).reshape(-1, 3)
    fits, heights = [], []
    for pitch_deg in (26.0, 8.0):
        cam = CameraPose(CAM.x, CAM.y, CAM.z, math.radians(pitch_deg))
        seen = project(grid, cam, INTR)
        rows, z = seen[:, 1], seen[:, 2]
        lift = lift_of(rows, INTR)
        elevation, _azimuth = ray_angles(lift)
        d = z * _ratio(elevation)
        fits.append(fit_ray(d, z, elevation, np.ones_like(d)))
        world = cam.z + z * (-math.sin(cam.pitch) + math.cos(cam.pitch) * lift)
        slice_ = (world > 0.6) & (world < 1.1)
        heights.append(float(np.median((d / z)[slice_])))
    first, second = fits
    assert first is not None and second is not None
    assert abs(heights[0] - heights[1]) > 0.05, heights  # the two views really differ
    common = np.linspace(
        max(first.lo, second.lo) + 1e-3, min(first.hi, second.hi) - 1e-3, 9, dtype=float
    )
    a_first, _b = first.law(common)
    a_second, _b2 = second.law(common)
    assert np.allclose(a_first, a_second, rtol=0.02), (a_first, a_second)


# ---- the shape's own guards ---------------------------------------------------------------------
def test_a_pool_of_one_elevation_carries_no_angular_law_and_the_affine_law_stands() -> None:
    """A pool spanning less than MIN_RAY_SPAN of elevation (one row of beams, a corridor seen
    end on) fits no angular law at all, and the stage applies the affine law it inherited."""
    pool = _pairs()
    flat = (np.abs(pool.lift - np.median(pool.lift)) < 0.5 * math.tan(MIN_RAY_SPAN)) & (pool.d > 0)
    elevation, _azimuth = ray_angles(pool.lift[flat])
    assert float(np.ptp(elevation)) < MIN_RAY_SPAN
    assert usable_degree(elevation, float(elevation.min()), float(elevation.max()), 2) == 0
    assert fit_ray(pool.d[flat], pool.z[flat], elevation, pool.weight[flat]) is None
    law = RayLaw()
    law.fit(Pairs(pool.d[flat], pool.z[flat], pool.weight[flat], pool.lift[flat], pool.left[flat]))
    assert law.fitted and not law.ray_ready
    ctx = FrameContext(INTR, CAM, stamp=1.0)
    raw = _network(_scene())
    assert np.allclose(
        law.apply(raw, ctx), apply_affine(raw, law.a, law.b), equal_nan=True, rtol=1e-12
    )
    assert "affine fallback" in law.describe()


def test_a_pool_whose_angle_is_the_depth_in_disguise_fits_no_ray_law() -> None:
    """The lidar's returns lie in one plane, so a beam's elevation in the picture is a curve of
    its range: such a pool reads 1.00 separable and fits no angular law at all, because any
    shape it found would be the affine law written twice. The wall anchor's pairs — many
    elevations at one depth — break that and the law stands."""
    raw, ctx = _frames()
    beams = ctx.beams
    assert beams is not None
    hits = np.arange(beams.shape[0])
    rows = beams[hits, 1].astype(int)
    lift = lift_of(rows, INTR)
    elevation, _azimuth = ray_angles(lift)
    y = 1.0 / beams[hits, 2]
    assert separable(elevation / RAY_SCALE, y) > 0.99
    pipeline = standard_pipeline(ray_law=True)  # the lidar alone
    for _ in range(3):
        pipeline.run(raw, ctx)
    law = pipeline.stage("ray_law")
    assert isinstance(law, RayLaw) and law.fitted and not law.ray_ready
    walled = standard_pipeline(ray_law=True, wall_anchor=True)
    for _ in range(3):
        walled.run(raw, ctx)
    with_walls = walled.stage("ray_law")
    assert isinstance(with_walls, RayLaw) and with_walls.ray_ready


def test_a_law_at_its_bound_says_so_and_is_clipped_there() -> None:
    """A network four times too far at the top of the cone asks for a scale past A_BOUNDS: the
    fit stands, the verdict says CLIPPED, and no pixel is corrected past the bound."""
    pool = _pairs()
    elevation, _azimuth = ray_angles(pool.lift)
    steep = pool.d * (1.0 + 1.6 * (elevation - elevation.min()))
    gain = fit_ray(steep, pool.z, elevation, pool.weight)
    assert gain is not None and gain.clipped
    assert "CLIPPED" in gain.describe()
    grid = np.linspace(gain.lo, gain.hi, 17, dtype=float)
    a, _b = gain.law(grid)
    assert float(a.max()) <= A_BOUNDS[1] + 1e-12 and float(a.min()) >= A_BOUNDS[0] - 1e-12


def test_the_gain_is_held_at_the_span_s_edge_above_what_the_pool_saw() -> None:
    """Above the highest ray the pool reached the gain stops rising: a polynomial free to
    extrapolate would put the ceiling anywhere."""
    pool = _pairs()
    elevation, _azimuth = ray_angles(pool.lift)
    gain = fit_ray(pool.d, pool.z, elevation, pool.weight)
    assert gain is not None
    edge = gain.scale_at(math.degrees(gain.hi))
    assert gain.scale_at(math.degrees(gain.hi) + 10.0) == pytest.approx(edge)
    assert gain.scale_at(math.degrees(gain.lo) - 10.0) == pytest.approx(
        gain.scale_at(math.degrees(gain.lo))
    )


def test_a_saved_law_comes_back_whole_beside_the_affine_one(tmp_path: Path) -> None:
    """Both laws live in one versioned file: the affine law reads as it always did, the ray
    law comes back to the bit, and a file written before the ray law existed has none."""
    pool = _pairs()
    elevation, _azimuth = ray_angles(pool.lift)
    gain = fit_ray(pool.d, pool.z, elevation, pool.weight)
    assert gain is not None
    path = tmp_path / "depth_law.json"
    save_law(path, 1.82, -0.03, pool.size, 1000.0, ray=gain.state())
    assert json.loads(path.read_text())["version"] == 2
    assert load_law(path, 1000.0) == (1.82, -0.03, pool.size)
    back = RayGain.restore(load_ray(path, 1000.0))
    assert back is not None
    assert np.array_equal(back.alpha, gain.alpha) and back.beta == gain.beta
    assert (back.lo, back.hi, back.clipped, back.pairs) == (gain.lo, gain.hi, False, gain.pairs)
    raw = _network(_scene())
    rows = ray_angles(lift_of(np.arange(360), INTR)[:, None])[0]
    assert np.allclose(back.apply(raw, rows), gain.apply(raw, rows), equal_nan=True)
    save_law(path, 1.82, -0.03, pool.size, 2000.0)
    assert load_ray(path, 2000.0) is None  # no ray law in the file
    assert RayGain.restore(None) is None
    assert RayGain.restore({"alpha": [1.0], "beta": 0.0, "lo": 0.0, "hi": 0.1, "pairs": 3}) is None
    # a law seeded from the file is applied before any live pool can replace it
    law = RayLaw()
    law.seed_gain(back)
    assert law.ray_ready and law.gain is back


# ---- the stage in the chain ---------------------------------------------------------------------
def _frames() -> tuple[np.ndarray, FrameContext]:
    """One frame of the synthetic room with the scan that measured its wall."""
    return _network(_scene()), FrameContext(INTR, CAM, lidar=_returns(), stamp=1.0)


def test_the_stage_is_off_by_default_and_leaves_the_affine_image_untouched() -> None:
    """The chain carries the ray law switched off: the published image is the affine law's, bit
    for bit, and the flag is what turns it into the ray law's without a restart."""
    raw, ctx = _frames()
    plain = standard_pipeline(wall_anchor=True)
    assert standard_pipeline().switches["ray_law"] is False
    switched = standard_pipeline(ray_law=True, wall_anchor=True)
    switched.set("ray_law", False)
    for _ in range(3):
        first = plain.run(raw, ctx)
        second = switched.run(raw, ctx)
    assert np.array_equal(first.depth, second.depth, equal_nan=True)
    switched.set("ray_law", True)
    third = switched.run(raw, ctx)
    assert not np.array_equal(first.depth, third.depth, equal_nan=True)
    assert "ray_law on" in switched.report() and "ray deg" in switched.report()


def test_the_stage_keeps_the_holes_of_the_image_it_is_handed() -> None:
    """The ray law corrects the frame's raw depth but publishes nothing the stages before it
    dropped: the edge filter's pixels stay NaN."""
    raw, ctx = _frames()
    pipeline = standard_pipeline(ray_law=True, wall_anchor=True)
    for _ in range(3):
        result = pipeline.run(raw, ctx)
    affine = result.after["affine_law"]
    ray = result.after["ray_law"]
    assert np.array_equal(np.isnan(affine), np.isnan(ray))
    assert np.isnan(ray).any() and np.isfinite(ray).any()


def test_the_stage_withholds_only_while_no_law_of_any_kind_exists() -> None:
    """With nothing pooled the ray law withholds the frame like the affine law does; once the
    affine law stands it never withholds again, angular fit or not."""
    raw, ctx = _frames()
    pipeline = standard_pipeline(ray_law=True, wall_anchor=True)
    blind = FrameContext(INTR, CAM, stamp=1.0)
    assert pipeline.run(raw, blind).withheld
    for _ in range(3):
        result = pipeline.run(raw, ctx)
    assert not result.withheld
    law = pipeline.stage("ray_law")
    assert isinstance(law, RayLaw) and law.ray_ready and isinstance(law, AffineLaw)


# ---- the guards read the whole cone, not the axis ---------------------------------------------
def _cone(az_gain: float, n: int = 4000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A pool of rays over a whole 70 deg cone, the network's inverse-depth slope rising with
    the elevation and by ``az_gain`` per normalised radian of azimuth: (network depth, true
    depth, elevation, azimuth)."""
    rng = np.random.default_rng(0)
    elevation = rng.uniform(-0.35, 0.30, n)
    azimuth = rng.uniform(-0.60, 0.60, n)
    z = rng.uniform(0.8, 4.0, n)
    alpha = 1.0 + 0.3 * elevation / RAY_SCALE + az_gain * azimuth / RAY_SCALE
    return z / alpha, z, elevation, azimuth


def test_an_azimuth_that_turns_the_slope_over_at_one_edge_is_refused() -> None:
    """The azimuth polynomial carries no constant term, so on the optical axis — where the
    guards used to be read — it is identically zero. A pool steep enough in azimuth for the
    fitted slope to go negative at one edge of the lens came back as a law with clipped false
    while every ray on that side silently took the A_BOUNDS ceiling; the guards now cross both
    spans and refuse it."""
    d, z, elevation, azimuth = _cone(1.2)
    assert fit_ray(d, z, elevation, azimuth=azimuth, azimuth_degree=1) is None
    on_axis = fit_ray(d, z, elevation, azimuth=azimuth, azimuth_degree=0)
    assert on_axis is not None, "without the azimuth term the same pool still carries a law"


def test_the_azimuth_is_held_at_its_own_span_s_edge_like_the_elevation() -> None:
    """A gentle azimuth dependence fits, and outside the azimuth the pool reached the gain
    stops moving: a polynomial free to extrapolate would run away past the lens' edge."""
    d, z, elevation, azimuth = _cone(0.2)
    gain = fit_ray(d, z, elevation, azimuth=azimuth, azimuth_degree=1)
    assert gain is not None and gain.azimuth.size == 1 and not gain.clipped
    assert math.degrees(gain.az_hi) == pytest.approx(33.0, abs=2.0)
    assert gain.az_lo == pytest.approx(-gain.az_hi, abs=0.02)
    grid = np.linspace(gain.lo, gain.hi, 9)
    edge = gain.slope(grid, np.full_like(grid, gain.az_hi))
    assert np.allclose(gain.slope(grid, np.full_like(grid, gain.az_hi + 0.5)), edge)
    assert f"az {math.degrees(gain.az_lo):+.0f}" in gain.describe()
    wide = np.linspace(-1.0, 1.0, 33)
    assert bool(np.all(gain.slope(grid[:, None], wide[None, :]) > 0.0))


def test_a_saved_gain_that_places_nothing_is_refused_like_a_saved_affine_law() -> None:
    """A record is judged on restore the way the fit judged it: a slope that turns
    non-positive anywhere on its own cone (every ray there would take the A_BOUNDS ceiling), an
    azimuth polynomial with no span of its own, and a record whose ``clipped`` denies a bound it
    meets are all refused, and the affine law stands instead."""
    d, z, elevation, azimuth = _cone(0.2)
    gain = fit_ray(d, z, elevation, azimuth=azimuth, azimuth_degree=1)
    assert gain is not None
    good = gain.state()
    assert RayGain.restore(good) is not None
    assert RayGain.restore({**good, "alpha": [-1.0, 0.0, 0.0]}) is None
    assert RayGain.restore({**good, "az_lo": 0.0, "az_hi": 0.0}) is None
    assert RayGain.restore({**good, "azimuth": [-5.0]}) is None  # positive on the axis alone
    bound = {**good, "alpha": [0.4, 0.0, 0.0]}  # positive, but a of 2.5-5.9: past A_BOUNDS
    assert RayGain.restore(bound) is None, "clipped false denies a bound it meets"
    assert RayGain.restore({**bound, "clipped": True}) is not None


def test_the_stage_saves_only_the_gain_it_fitted_itself() -> None:
    """What goes into the law file is a measurement, never the seed the last run left: a
    seeded stage offers nothing until its own pool carries a law."""
    pool = _pairs()
    elevation, _azimuth = ray_angles(pool.lift, pool.left)
    gain = fit_ray(pool.d, pool.z, elevation, pool.weight)
    assert gain is not None
    law = RayLaw()
    law.seed_gain(gain)
    assert law.ray_ready and not law.ray_fitted and law.saved_state() is None
    law.fit(pool)
    assert law.ray_fitted and law.saved_state() == law.gain.state()
