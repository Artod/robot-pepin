"""The mast-sway filter (pepin.mast, the twin of mast.hpp): what vio.md section 5 promises."""

from __future__ import annotations

import math

import pytest

from pepin.mast import (
    MastFilter,
    compose_sway,
    head_rate_in_base,
    ring_rate,
    rotation_from_rpy,
)

DEG = math.pi / 180.0


def _ring_residual(arm_window_s: float = 0.5) -> tuple[float, float]:
    """A 5.3 Hz ring of 0.35 deg p-p armed at a 0.17 deg deflection, 200 Hz: the truth's p-p and
    the residual's (truth minus the published pitch) over the first full period after arming."""
    from pepin.mast import MastSettings

    mast = MastFilter(MastSettings(arm_window_s=arm_window_s))
    amplitude, f = 0.175 * DEG, 5.3
    phase = math.asin(0.17 / 0.175)
    truths, residuals = [], []
    for i in range(400):
        local = i / 200.0
        truth = amplitude * math.sin(2 * math.pi * f * local + phase)
        out = mast.update(50.0 + local, (0.0, ring_rate(local, amplitude, f, phase), 0.0), 0.0)
        if 1 / f <= local < 2 / f:
            truths.append(truth)
            residuals.append(truth - out.theta[1])
    return max(truths) - min(truths), max(residuals) - min(residuals)


def test_at_least_70_percent_of_the_ring_is_removed_from_its_first_full_period() -> None:
    raw, residual = _ring_residual()
    assert raw == pytest.approx(0.35 * DEG, rel=0.02)
    assert 1.0 - residual / raw >= 0.70, f"{100 * (1 - residual / raw):.0f} % removed"
    _raw0, residual0 = _ring_residual(arm_window_s=0.0)
    assert residual < residual0, "the arming window's mean is what buys the first period"


def test_a_base_turn_is_no_sway_a_hold_says_nan_and_arming_starts_from_zero() -> None:
    mast = MastFilter()
    for i in range(2000):
        out = mast.update(i / 200.0, (0.0, 0.0, 0.3), 0.3)
    assert out.theta[2] == pytest.approx(0.0, abs=1e-12) and not out.held
    held = mast.update(20.0, (math.nan, 0.0, 0.0), 0.0)
    assert held.held and all(math.isnan(v) for v in held.theta)
    armed = mast.update(20.01, (0.0, 1.0, 0.0), 0.0)
    assert not armed.held and armed.theta[1] == 0.0
    gap = mast.update(21.0, (0.0, 1.0, 0.0), 0.0)
    assert gap.theta[1] == 0.0, "a gap past max_dt_s re-arms"


def test_a_positive_sway_pitch_tilts_the_camera_down_about_the_hinge() -> None:
    hinge, neck = (-0.058, 0.0, 0.78), (0.03, 0.0, 1.20)
    xyz, rotation = compose_sway((0.0, 0.0, 0.0), hinge, neck, (0.0, 0.4154, 0.2))
    assert xyz == pytest.approx(neck) and rotation == pytest.approx(
        rotation_from_rpy(0, 0.4154, 0.2)
    )
    swayed, turned = compose_sway((0.0, 0.01, 0.0), hinge, neck, (0.0, 0.0, 0.0))
    assert turned[6] < 0.0, "the camera's x points below the horizon"
    assert swayed[0] > neck[0], "and the lens moves forward over a hinge below it"
    lever = math.hypot(neck[0] - hinge[0], neck[2] - hinge[2])
    assert math.hypot(swayed[0] - neck[0], swayed[2] - neck[2]) == pytest.approx(
        0.01 * lever, rel=1e-3
    )
    in_base = head_rate_in_base((0.0, 1.0, 0.0), rotation_from_rpy(0, 0, 0), 90 * DEG, 0.0)
    assert in_base[0] == pytest.approx(-1.0), "a head panned 90 deg left: its y rate is base -x"
