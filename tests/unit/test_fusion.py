"""Information fusion of pose measurements, and the covariance read off a score surface."""

import math

import numpy as np
import pytest
from synthetic import raycast_room
from test_localization import PILLAR, furnished_room_map

from pepin.fusion import (
    EDGE_INFLATION,
    FIT_FLOOR,
    PoseMeasurement,
    at_edge,
    covariance_from_score_surface,
    disagreement,
    fuse,
)
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import CorrelativeMatcher, ScoreSurface, SearchWindow

WINDOW = SearchWindow(xy_m=0.09, xy_step_m=0.03, theta_deg=9.0, theta_step_deg=1.5)


def measurement(
    x: float, y: float, yaw: float, sx: float, sy: float, st: float, source: str = "a"
) -> PoseMeasurement:
    return PoseMeasurement(x, y, yaw, np.diag([sx**2, sy**2, st**2]), source, 1.0, 0.8)


def lattice(scores: np.ndarray) -> ScoreSurface:
    """A synthetic surface on the node's window: ``scores`` (T, P) per beam in [-1, 1]."""
    n = round(WINDOW.xy_m / WINDOW.xy_step_m)
    offsets = np.arange(-n, n + 1) * WINDOW.xy_step_m
    positions = np.array([(dx, dy) for dx in offsets for dy in offsets])
    m = round(WINDOW.theta_deg / WINDOW.theta_step_deg)
    headings = np.radians(np.arange(-m, m + 1) * WINDOW.theta_step_deg)
    # ties break toward the centre, as the matcher's own lattice penalty breaks them to the guess
    scores = scores - 1e-9 * (np.abs(positions).sum(axis=1)[None, :] + np.abs(headings)[:, None])
    k, i = divmod(int(np.argmax(scores)), scores.shape[1])
    return ScoreSurface(scores * 100.0, positions, headings, k, i, n_points=100, top=1.0)


def surface_of(x_sharp: float, y_sharp: float, theta_sharp: float) -> ScoreSurface:
    """A peak at the centre whose score falls by ``*_sharp`` per metre / radian squared along
    each axis: 0 along an axis is a ridge, the scan fits alike anywhere along it."""
    n = round(WINDOW.xy_m / WINDOW.xy_step_m)
    offsets = np.arange(-n, n + 1) * WINDOW.xy_step_m
    m = round(WINDOW.theta_deg / WINDOW.theta_step_deg)
    headings = np.radians(np.arange(-m, m + 1) * WINDOW.theta_step_deg)
    scores = np.empty((len(headings), len(offsets) ** 2))
    for k, th in enumerate(headings):
        p = 0
        for dx in offsets:
            for dy in offsets:
                scores[k, p] = 1.0 - x_sharp * dx**2 - y_sharp * dy**2 - theta_sharp * th**2
                p += 1
    return lattice(scores)


# -- fuse ---------------------------------------------------------------------


def test_nothing_fuses_to_none_and_one_measurement_is_itself() -> None:
    assert fuse([]) is None
    m = measurement(1.0, 2.0, 0.3, 0.01, 0.01, 0.01)
    assert fuse([m]) is m


def test_two_isotropic_measurements_meet_at_their_information_weighted_mean() -> None:
    sharp = measurement(0.0, 0.0, 0.0, 0.01, 0.01, 0.01, "lidar")
    blunt = measurement(0.10, 0.10, 0.10, 0.03, 0.03, 0.03, "depth")
    fused = fuse([sharp, blunt], gate=math.inf)  # ten centimetres apart: the gate would object
    assert fused is not None and fused.source == "lidar+depth"
    weight = (1 / 0.03**2) / (1 / 0.01**2 + 1 / 0.03**2)  # the blunt one's share: a tenth
    assert (fused.x, fused.y, fused.yaw) == pytest.approx((0.1 * weight,) * 3)
    assert fused.sigmas[0] == pytest.approx(math.sqrt(1 / (1 / 0.01**2 + 1 / 0.03**2)))
    assert fused.stamp == 1.0 and fused.fit == pytest.approx(0.8)


def test_headings_are_fused_across_the_wrap_not_through_zero() -> None:
    a = measurement(0.0, 0.0, math.radians(179.0), 0.01, 0.01, math.radians(1.0))
    b = measurement(0.0, 0.0, math.radians(-179.0), 0.01, 0.01, math.radians(1.0), "b")
    fused = fuse([a, b])
    assert fused is not None
    assert abs(wrap_angle(fused.yaw - math.pi)) < math.radians(1e-6)


def test_a_ridge_along_a_wall_leaves_that_axis_to_the_other_source() -> None:
    """The camera sees one wall across y: it pins y and yaw, not x. Fused with a lidar that
    pins everything, the camera moves y (a little, by its weight) and x not at all."""
    lidar = measurement(0.0, 0.0, 0.0, 0.01, 0.01, 0.01, "lidar")
    camera = measurement(0.50, 0.02, 0.0, 1.0, 0.01, 0.01, "depth")  # x: anything; y: 2 cm
    fused = fuse([lidar, camera])
    assert fused is not None
    assert abs(fused.x) < 0.0001  # a metre of x uncertainty carries no vote on x
    assert fused.y == pytest.approx(0.01)  # two equally sure y answers meet half way


def test_a_measurement_that_disagrees_beyond_the_gate_is_left_out_and_named() -> None:
    """A camera scan of a table top the map has no wall for matches the map well half a metre
    away; only its disagreement with the lidar gives it away. Fused with the gate off it would
    drag the pose; with the gate it is a bystander, named in ``rejected``."""
    lidar = measurement(0.0, 0.0, 0.0, 0.01, 0.01, 0.01, "lidar")
    stray = measurement(0.50, 0.0, 0.0, 0.02, 0.02, 0.02, "depth")
    assert disagreement(lidar, stray) == pytest.approx(0.5**2 / (0.01**2 + 0.02**2))
    fused = fuse([lidar, stray])
    assert fused is not None and fused.rejected == ("depth",) and fused.source == "lidar"
    assert (fused.x, fused.y, fused.yaw) == (0.0, 0.0, 0.0)
    fused = fuse([lidar, stray], gate=math.inf)
    assert fused is not None and fused.rejected == () and fused.x > 0.05
    agreeing = measurement(0.02, 0.0, 0.0, 0.02, 0.02, 0.02, "depth")
    fused = fuse([lidar, agreeing])
    assert fused is not None and fused.rejected == () and fused.source == "lidar+depth"


# -- the covariance from a surface ------------------------------------------


def test_a_match_on_the_windows_edge_is_a_bound_not_a_measurement() -> None:
    """The best candidate on the lattice's border means the scan wanted to go farther than the
    window allowed: the answer is the window's, and its covariance is widened accordingly."""
    centred = surface_of(200.0, 200.0, 20.0)
    assert not at_edge(centred)
    n = round(WINDOW.xy_m / WINDOW.xy_step_m)
    shifted = surface_of(200.0, 200.0, 20.0)
    scores = shifted.scores.copy()
    # move the peak to the last position (dx = dy = +0.09): the window's corner
    scores[:, -1] = scores.max() + 1.0
    edge = ScoreSurface(
        scores, shifted.positions, shifted.headings, shifted.k, len(shifted.positions) - 1, 100, 1.0
    )
    assert at_edge(edge) and (2 * n + 1) ** 2 == len(shifted.positions)
    tight = covariance_from_score_surface(centred, fit=1.0)
    wide = covariance_from_score_surface(edge, fit=1.0)
    assert wide[0, 0] > EDGE_INFLATION * tight[0, 0] / 4  # the corner's spread plus the inflation
    turned = ScoreSurface(
        shifted.scores, shifted.positions, shifted.headings, 0, shifted.i, 100, 1.0
    )
    assert at_edge(turned)


def test_a_ridge_gives_an_anisotropic_covariance_and_a_peak_a_tight_one() -> None:
    ridge = covariance_from_score_surface(surface_of(200.0, 0.0, 20.0), fit=1.0)
    sx, sy, _ = np.sqrt(np.diag(ridge))
    assert sy > 3 * sx, f"along the ridge {sy * 100:.1f} cm vs across {sx * 100:.1f} cm"
    plateau = math.sqrt(np.mean(np.arange(-3, 4) ** 2) * 0.03**2 + 0.03**2 / 12)  # the window
    assert sy == pytest.approx(plateau, rel=0.05)
    peak = covariance_from_score_surface(surface_of(200.0, 200.0, 20.0), fit=1.0)
    px, py, _ = np.sqrt(np.diag(peak))
    assert px == pytest.approx(sx) and py == pytest.approx(sx)
    assert abs(ridge[0, 1]) < 1e-9  # axis-aligned ridge: no correlation


def test_a_flat_surface_is_a_wide_answer_in_every_direction() -> None:
    flat = covariance_from_score_surface(surface_of(0.0, 0.0, 0.0), fit=1.0)
    sx, sy, st = np.sqrt(np.diag(flat))
    assert sx == pytest.approx(sy) and sx > 0.05 and st > math.radians(4.0)


def test_a_poor_fit_and_a_low_trust_inflate_the_covariance() -> None:
    surface = surface_of(200.0, 200.0, 20.0)
    good = covariance_from_score_surface(surface, fit=1.0)
    half = covariance_from_score_surface(surface, fit=0.5)
    hopeless = covariance_from_score_surface(surface, fit=0.0)
    assert np.allclose(half, 4.0 * good) and np.allclose(hopeless, good / FIT_FLOOR**2)
    assert np.allclose(covariance_from_score_surface(surface, fit=1.0, trust=0.5), 2.0 * good)


def fan(points: np.ndarray, half_deg: float = 40.0) -> np.ndarray:
    """The returns inside a +-``half_deg`` fan ahead: what the camera's virtual scan sees."""
    bearing = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    return points[np.abs(bearing) <= half_deg]


def test_a_partial_fan_on_one_wall_is_sure_across_it_and_unsure_along_it() -> None:
    """The real matcher on the synthetic room: a +-40 degree fan facing the far wall (x = 3)
    from the middle sees only that wall. Its match pins x (the distance to the wall) and the
    heading, and says almost nothing about y; the whole lidar revolution pins both."""
    grid = furnished_room_map()
    matcher = CorrelativeMatcher(grid, max_points=200, interpolate=True)
    truth = Pose2D(0.5, 0.0, 0.0)
    whole = raycast_room(truth, beams=360, pillar=PILLAR)
    partial = fan(whole)
    assert 60 <= len(partial) <= 100
    _, full_surface = matcher.match_surface(truth, whole, WINDOW)
    _, fan_surface = matcher.match_surface(truth, partial, WINDOW)
    full = covariance_from_score_surface(full_surface, matcher.inlier_fraction(truth, whole))
    wall = covariance_from_score_surface(fan_surface, matcher.inlier_fraction(truth, partial))
    fx, fy, _ = np.sqrt(np.diag(full))
    wx, wy, wt = np.sqrt(np.diag(wall))
    assert wy > 2 * wx, f"along the wall {wy * 100:.1f} cm, across {wx * 100:.1f} cm"
    assert wy > 2 * fy and wx < 1.5 * fx  # along: the fan is blind; across: as sure as the lidar
    assert wt < math.radians(3.0)
