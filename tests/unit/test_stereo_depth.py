"""pepin.stereo_depth: the matcher's settings, the rig's metres, and what the matcher measures.

The scenes are rendered rectified pairs of KNOWN geometry (``stereo_scenes``), which is the only
way to judge a depth: the real head has no calibration yet, so the real frames can be timed and
looked at and nothing more. What is asserted here is what a costmap and RTAB-Map depend on — the
depth is right where the matcher answers, and the matcher does NOT answer over a blank wall, in
an occlusion band, past the rig's reach, or in the left band no right eye reaches into.
"""

from __future__ import annotations

import numpy as np
import pytest
import stereo_scenes as scenes
from stereo_scenes import BASELINE_M, FX, FX_B

from pepin.stereo import MIN_DISPARITY_PX, Rectifier, StereoCalibration
from pepin.stereo_depth import (
    DEPTH_SIGMA_M,
    DISPARITY_SIGMA_PX,
    Baseline,
    MatcherSettings,
    StereoDepth,
    StereoMatcher,
    StereoUnavailableError,
    near_m,
    reach_m,
)

RIG = Baseline(fx=FX, baseline_m=BASELINE_M)


@pytest.fixture(scope="module")
def planes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Textured planes at 0.5, 1, 2 and 4 m across the picture."""
    return scenes.planes_scene()


@pytest.fixture(scope="module")
def step() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A 1.0 -> 2.5 m depth step and the band of it the right eye cannot see."""
    left, right, truth = scenes.step_scene()
    return left, right, truth, scenes.occlusion_band(truth)


@pytest.fixture(scope="module")
def blank() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A wall at 1.2 m with a blank rectangle painted on it."""
    return scenes.blank_scene()


def source(reach: float = 99.0, **settings: object) -> StereoDepth:
    """A stereo source on the rendered rig, by default answering at any range."""
    return StereoDepth(RIG, StereoMatcher(MatcherSettings(**settings)), reach=reach)  # type: ignore[arg-type]


# ---- the settings ---------------------------------------------------------------------------
def test_the_settings_refuse_numbers_opencv_would_mishandle() -> None:
    """A search that is not a multiple of 16, an even block, an unknown mode and a downscale
    under 1 are all refused at construction: OpenCV would either throw deep inside compute or,
    worse, silently round."""
    for bad in (
        {"num_disparities": 100},
        {"num_disparities": 0},
        {"block_size": 4},
        {"mode": "elas"},
        {"downscale": 0},
    ):
        with pytest.raises(ValueError):
            MatcherSettings(**bad)  # type: ignore[arg-type]


def test_the_smoothness_rule_is_opencv_s_and_p2_never_falls_under_p1() -> None:
    """P1/P2 default to 8 and 32 times the block area (200 and 800 at block 5); a P2 stated
    under P1 is lifted above it, which OpenCV requires and does not check."""
    assert MatcherSettings().smoothness == (200, 800)
    assert MatcherSettings(block_size=7).smoothness == (392, 1568)
    assert MatcherSettings(p1=500, p2=100).smoothness == (500, 501)


def test_the_search_shrinks_with_the_picture_so_the_near_end_does_not_move() -> None:
    """``num_disparities`` is stated at full size. Matching at 1/2 halves fx too, so the search
    is halved with it and ``fx * B / num_disparities`` — the nearest measurable depth — stays
    put. A search that does not divide evenly is rounded UP, never further away than asked."""
    assert MatcherSettings().search_px == 128
    assert MatcherSettings(downscale=2).search_px == 64
    assert MatcherSettings(num_disparities=128, downscale=3).search_px == 48
    assert MatcherSettings(num_disparities=16, downscale=4).search_px == 16


# ---- the rig's metres -----------------------------------------------------------------------
def test_the_rig_comes_out_of_the_two_camera_infos() -> None:
    """The left eye's fx and the right eye's ``P[0, 3] = -fx * baseline`` are the whole rig."""
    rig = Baseline.from_projection(FX, -FX * BASELINE_M)
    assert rig.fx == FX and rig.baseline_m == pytest.approx(BASELINE_M)


def test_a_right_eye_that_never_said_its_baseline_is_refused() -> None:
    """``P[0, 3]`` zero is a right eye published as if it were a left one: believing it would
    make every disparity infinitely far, so it raises instead."""
    with pytest.raises(StereoUnavailableError):
        Baseline.from_projection(FX, 0.0)
    with pytest.raises(StereoUnavailableError):
        Baseline.from_projection(FX, +1.0)
    with pytest.raises(StereoUnavailableError):
        Baseline.from_projection(0.0, -1.0)


def test_the_rig_s_metres_are_the_rectifier_s_to_the_bit() -> None:
    """``z = fx * B / d`` is written twice — here for a node that has only the two numbers, and
    in pepin.stereo for a node that has the calibration. This is the pin that keeps them one
    formula: the same disparities through both, bit for bit, NaN for NaN."""
    k = ((FX, 0.0, 400.0), (0.0, FX, 300.0), (0.0, 0.0, 1.0))
    calibration = StereoCalibration(
        width=64, height=48, k_left=k, d_left=(0.0,) * 5, k_right=k, d_right=(0.0,) * 5,
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation_m=(-BASELINE_M, 0.0, 0.0), rms_px=0.2, date="2026-09-20", method="rendered",
    )  # fmt: skip
    rectifier = Rectifier.from_calibration(calibration)
    rig = Baseline(fx=rectifier.fx, baseline_m=rectifier.baseline_m)
    disparities = np.array(
        [[np.nan, -3.0, 0.0, MIN_DISPARITY_PX, 1.0, 11.75, 47.0, 1e6]], dtype=np.float32
    )
    assert np.array_equal(rig.depth_m(disparities), rectifier.depth_m(disparities), equal_nan=True)


def test_the_reach_is_where_the_error_model_crosses_ten_centimetres() -> None:
    """The reach is derived, not copied from the mono network's 3.0 m: sigma_z = z^2 / (fx*B) *
    sigma_d reaches 10 cm at 2.17 m on this rig, and the nearest measurable depth is 18 cm."""
    reach = reach_m(FX, BASELINE_M)
    assert reach == pytest.approx(2.17, abs=0.01)
    assert reach**2 / FX_B * DISPARITY_SIGMA_PX == pytest.approx(DEPTH_SIGMA_M)
    assert near_m(FX, BASELINE_M, 128) == pytest.approx(0.18, abs=0.005)
    assert reach_m(0.0, BASELINE_M) == 0.0 and near_m(FX, BASELINE_M, 0) == 0.0


def test_a_source_without_its_rig_refuses_instead_of_inventing_a_metre(planes: tuple) -> None:
    """camera_info arrives on a topic, so a source may be built before its rig is known."""
    left, right, _truth = planes
    depth = StereoDepth()
    with pytest.raises(StereoUnavailableError):
        depth(left, right)
    depth.geometry = RIG
    assert depth.reach == pytest.approx(2.17, abs=0.01), "a reach nobody stated is derived"
    assert np.isfinite(depth(left, right)).any()


def test_two_eyes_of_different_sizes_are_not_one_rig(planes: tuple) -> None:
    left, right, _truth = planes
    with pytest.raises(StereoUnavailableError):
        source()(left, right[:, :400])


# ---- what it measures -----------------------------------------------------------------------
def test_a_known_plane_comes_back_within_two_percent_at_every_range_it_answers_for(
    planes: tuple,
) -> None:
    """The whole point: where the matcher answers, the depth is the geometry's. Under 2 % of
    median relative error at 0.5, 1 and 2 m — the range this head is claimed over — and the 4 m
    band, past the reach, shows why the reach exists: 2.1 %, all of it the subpixel bias."""
    left, right, truth = planes
    depth = source()(left, right)
    answered = np.isfinite(depth)
    for z, bound in ((0.5, 0.02), (1.0, 0.02), (2.0, 0.02), (4.0, 0.025)):
        on = answered & (truth == z)
        assert on.sum() > 10_000, f"{z} m: the matcher answered for almost nothing"
        error = float(np.median(np.abs(depth[on] - z) / z))
        assert error < bound, f"{z} m: median error {error:.3%}"
    assert answered.mean() > 0.8, "a fully textured scene should answer nearly everywhere"


def test_the_left_band_no_right_eye_reaches_into_is_unknown_not_guessed(planes: tuple) -> None:
    """The search runs off the left of the right eye, so the leftmost num_disparities columns
    have nothing to be matched against. They must be NaN — 16 % of an 800 px picture at the
    shipped search, which is the price of measuring down to 0.18 m."""
    left, right, _truth = planes
    depth = source()(left, right)
    assert not np.isfinite(depth[:, :128]).any(), "the blind band answered something"
    assert np.isfinite(depth[:, 130:]).mean() > 0.9


def test_a_blank_wall_is_unknown_and_not_the_smoothness_term_talking(blank: tuple) -> None:
    """SGBM has no texture gate of its own: with the gate off it paints a confident plateau
    over 97 % of a blank rectangle, and that plateau would mark a costmap where nothing stands.
    With the gate on, under a tenth of the rectangle survives."""
    left, right, _truth, region = blank
    without = source(texture_threshold=0.0)(left, right)
    assert np.isfinite(without[region]).mean() > 0.9, "the scene does not provoke the plateau"
    with_gate = source()(left, right)
    assert np.isfinite(with_gate[region]).mean() < 0.1
    outside = ~region
    assert np.isfinite(with_gate[outside]).mean() > 0.7, "the gate ate the textured wall too"


def test_the_occlusion_band_is_unknown_and_the_step_itself_has_no_spikes(step: tuple) -> None:
    """The strip of far wall the near surface hides from the right eye cannot be measured by
    anyone: over 85 % of it comes back NaN. Away from it the step is measured to 2 %, and not
    one pixel in a thousand lands more than 25 cm from the truth — a costmap reads a spike as an
    obstacle in mid-air."""
    left, right, truth, band = step
    depth = source()(left, right)
    assert np.isfinite(depth)[band].mean() < 0.15, "the occlusion band was answered for"
    away = np.isfinite(depth) & ~band
    away[:, :128] = False  # the blind band is already NaN; this is about the step
    assert float(np.median(np.abs(depth[away] - truth[away]) / truth[away])) < 0.02
    assert float(np.mean(np.abs(depth[away] - truth[away]) > 0.25)) < 0.001


def test_nothing_is_answered_past_the_reach(planes: tuple) -> None:
    """The 4 m band is measurable — a disparity of 5.9 px — and refused anyway: a sensor states
    where it stops answering for itself, and NaN is the one value no consumer acts on. Not one
    published metre is past the reach, and the far band is blank but for the single column where
    the block straddles the 2 m plane's edge and reads it as near."""
    left, right, truth = planes
    depth = source(reach=2.17)(left, right)
    assert float(np.nanmax(depth)) <= 2.17
    far = truth > 2.17 + 0.3
    assert np.isfinite(depth[far]).mean() < 0.001
    near = np.isfinite(depth) & (truth <= 2.0)
    assert near.mean() > 0.3, "the reach ate the near planes too"


def test_a_thin_bar_survives_the_shipped_block_and_a_wide_one_swallows_it() -> None:
    """A chair leg at range is an 8 px bar. It is why the block is 5 and not 9: two thirds of
    the bar is recovered at 5 and under half at 9, while a plane's depth error is the same."""
    left, right, _truth, bars = scenes.bars_scene(bar_px=8)
    found = {}
    for block in (5, 9):
        depth = source(block_size=block)(left, right)
        on = bars & np.isfinite(depth) & (np.abs(depth - 1.0) < 0.15)
        found[block] = float(on.sum()) / float(bars.sum())
    assert found[5] > 0.6 and found[9] < 0.55 and found[5] > found[9] + 0.15


def test_matching_at_half_size_scales_the_disparity_back_and_costs_the_thin_things() -> None:
    """The half-size knob must still answer in metres of the full-size picture — a plane at 1 m
    reads 1 m — and it is not a default because a chair leg does not survive it."""
    left, right, truth = scenes.planes_scene()
    depth = source(downscale=2)(left, right)
    on = np.isfinite(depth) & (truth == 1.0)
    assert float(np.median(np.abs(depth[on] - 1.0))) < 0.03
    assert depth.shape == left.shape, "the disparity comes back at the picture's own size"
    bleft, bright, _btruth, bars = scenes.bars_scene(bar_px=8)
    half = source(downscale=2)(bleft, bright)
    kept = bars & np.isfinite(half) & (np.abs(half - 1.0) < 0.15)
    assert float(kept.sum()) / float(bars.sum()) < 0.1


# ---- the instrumentation --------------------------------------------------------------------
def test_every_pair_is_timed_and_the_valid_share_is_kept_for_the_report(planes: tuple) -> None:
    """Rule 15: what is latency-sensitive is instrumented. The matcher times itself, the texture
    gate separately, and the source keeps the share of pixels it answered for."""
    left, right, _truth = planes
    depth = source()
    depth(left, right)
    depth(left, right)
    assert depth.frames == 2
    assert depth.timing["total"].summary().count == 2
    assert depth.matcher.timing["match"].summary().median_ms > 0.0
    assert depth.matcher.timing["texture"].summary().median_ms > 0.0
    assert 0.5 < depth.valid_fraction < 1.0
    words = depth.describe()
    assert "128px/5px 3way" in words and "% valid" in words and "0.18-" in words
