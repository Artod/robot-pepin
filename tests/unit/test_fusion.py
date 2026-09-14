"""Information fusion of pose measurements, and the covariance read off a score surface."""

import math
from dataclasses import replace

import numpy as np
import pytest
from synthetic import raycast_room
from test_localization import PILLAR, furnished_room_map

from pepin.fusion import (
    BOUND_INFLATION,
    FIT_FLOOR,
    MIN_SIGMA_XY_M,
    MIN_SIGMA_YAW_RAD,
    PoseMeasurement,
    at_edge,
    bound_directions,
    covariance_from_score_surface,
    disagreement,
    from_peak,
    fuse,
    peak_covariance,
    peak_temperature,
    sigma_from_fit,
)
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import CorrelativeMatcher, ScoreSurface, SearchWindow

WINDOW = SearchWindow(xy_m=0.09, xy_step_m=0.03, theta_deg=9.0, theta_step_deg=1.5)
TEST_T = peak_temperature("lidar")  # the calibrated number these synthetic peaks are read with


def measurement(
    x: float, y: float, yaw: float, sx: float, sy: float, st: float, source: str = "a"
) -> PoseMeasurement:
    return PoseMeasurement(x, y, yaw, np.diag([sx**2, sy**2, st**2]), source, 1.0, 0.8)


def lattice(scores: np.ndarray, window: SearchWindow = WINDOW) -> ScoreSurface:
    """A synthetic surface on the node's window: ``scores`` (T, P) per beam in [-1, 1]."""
    n = round(window.xy_m / window.xy_step_m)
    offsets = np.arange(-n, n + 1) * window.xy_step_m
    positions = np.array([(dx, dy) for dx in offsets for dy in offsets])
    m = round(window.theta_deg / window.theta_step_deg)
    headings = np.radians(np.arange(-m, m + 1) * window.theta_step_deg)
    # ties break toward the centre, as the matcher's own lattice penalty breaks them to the guess
    scores = scores - 1e-9 * (np.abs(positions).sum(axis=1)[None, :] + np.abs(headings)[:, None])
    k, i = divmod(int(np.argmax(scores)), scores.shape[1])
    return ScoreSurface(scores * 100.0, positions, headings, k, i, n_points=100, top=1.0)


def surface_of(
    x_sharp: float, y_sharp: float, theta_sharp: float, window: SearchWindow = WINDOW
) -> ScoreSurface:
    """A peak at the centre whose score falls by ``*_sharp`` per metre / radian squared along
    each axis: 0 along an axis is a ridge, the scan fits alike anywhere along it."""
    n = round(window.xy_m / window.xy_step_m)
    offsets = np.arange(-n, n + 1) * window.xy_step_m
    m = round(window.theta_deg / window.theta_step_deg)
    headings = np.radians(np.arange(-m, m + 1) * window.theta_step_deg)
    scores = np.empty((len(headings), len(offsets) ** 2))
    for k, th in enumerate(headings):
        p = 0
        for dx in offsets:
            for dy in offsets:
                scores[k, p] = 1.0 - x_sharp * dx**2 - y_sharp * dy**2 - theta_sharp * th**2
                p += 1
    return lattice(scores, window)


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
    assert wide[0, 0] > BOUND_INFLATION * tight[0, 0] / 4  # the corner's spread plus the inflation
    turned = ScoreSurface(
        shifted.scores, shifted.positions, shifted.headings, 0, shifted.i, 100, 1.0
    )
    assert at_edge(turned)


def test_a_ridge_gives_an_anisotropic_covariance_and_a_peak_a_tight_one() -> None:
    """Along the ridge the scan resolved nothing inside the window: that axis is a bound,
    the window's spread widened like an edge; across it the answer is as sharp as a peak's."""
    ridge = covariance_from_score_surface(surface_of(200.0, 0.0, 20.0), fit=1.0)
    sx, sy, _ = np.sqrt(np.diag(ridge))
    assert sy > 3 * sx, f"along the ridge {sy * 100:.1f} cm vs across {sx * 100:.1f} cm"
    plateau = math.sqrt(np.mean(np.arange(-3, 4) ** 2) * 0.03**2 + 0.03**2 / 12)  # the window
    assert sy == pytest.approx(plateau * math.sqrt(BOUND_INFLATION), rel=0.05)
    peak = covariance_from_score_surface(surface_of(200.0, 200.0, 20.0), fit=1.0)
    px, py, _ = np.sqrt(np.diag(peak))
    assert px == pytest.approx(sx) and py == pytest.approx(sx)
    assert abs(ridge[0, 1]) < 1e-9  # axis-aligned ridge: no correlation
    (along,) = bound_directions(surface_of(200.0, 0.0, 20.0))
    assert np.allclose(np.abs(along), [0.0, 1.0, 0.0])
    assert bound_directions(surface_of(200.0, 200.0, 20.0)) == []


def test_a_flat_surface_is_a_wide_answer_in_every_direction() -> None:
    """A scan that fits nothing: flat everywhere, a bound in every direction, no vote."""
    flat = covariance_from_score_surface(surface_of(0.0, 0.0, 0.0), fit=1.0)
    sx, sy, st = np.sqrt(np.diag(flat))
    assert sx == pytest.approx(sy) and sx > 0.5 and st > math.radians(40.0)
    assert len(bound_directions(surface_of(0.0, 0.0, 0.0))) == 3


def diagonal_ridge(across_sharp: float) -> ScoreSurface:
    """A peak flat along the map's (1, 1) diagonal — a wall at 45 degrees to the axes — whose
    score falls by ``across_sharp`` per metre squared across it."""
    n = round(WINDOW.xy_m / WINDOW.xy_step_m)
    offsets = np.arange(-n, n + 1) * WINDOW.xy_step_m
    m = round(WINDOW.theta_deg / WINDOW.theta_step_deg)
    headings = np.radians(np.arange(-m, m + 1) * WINDOW.theta_step_deg)
    scores = np.empty((len(headings), len(offsets) ** 2))
    for k, th in enumerate(headings):
        p = 0
        for dx in offsets:
            for dy in offsets:
                scores[k, p] = 1.0 - across_sharp * ((dx - dy) / math.sqrt(2.0)) ** 2 - 20.0 * th**2
                p += 1
    return lattice(scores)


def test_a_plateau_is_a_bound_the_tie_break_toward_the_guess_measures_nothing() -> None:
    """A wall at any angle: the direction along it is found by the spread's own eigenvectors,
    widened like an edge, while across it the answer stays sharp. The threshold is the
    temperature: a likelihood that falls to 1/e only at the window's edge is a plateau, one
    that falls twice as fast is resolved. The heading is judged the same way."""
    (along,) = bound_directions(diagonal_ridge(200.0))
    assert np.allclose(np.abs(along), [1.0, 1.0, 0.0] / np.sqrt(2.0))
    cov = covariance_from_score_surface(diagonal_ridge(200.0), fit=1.0)
    u, v = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0), np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    assert u @ cov @ u > BOUND_INFLATION / 2 * (v @ cov @ v)
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)  # widened by congruence: still a covariance
    # 1 - 12.35 * 0.09^2 = 0.9: the score drops by one temperature (a tenth) at the edge
    assert len(bound_directions(surface_of(200.0, 12.0, 20.0))) == 1  # flatter: a plateau
    assert bound_directions(surface_of(200.0, 30.0, 20.0)) == []  # steeper: resolved
    (turn,) = bound_directions(surface_of(200.0, 200.0, 0.0))
    assert np.allclose(turn, [0.0, 0.0, 1.0])
    st = covariance_from_score_surface(surface_of(200.0, 200.0, 0.0), fit=1.0)[2, 2]
    assert (
        st
        > BOUND_INFLATION
        / 2
        * covariance_from_score_surface(surface_of(200.0, 200.0, 20.0), fit=1.0)[2, 2]
    )


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


def test_a_pose_known_only_by_its_fit_is_still_a_measurement() -> None:
    """What the tracker's own belief is worth beside a fresh match: isotropic in position, its
    sigmas from the fit, so the two can be weighed by information instead of by a rule of
    thumb (pepin.watchdog re-seeds through exactly this)."""
    import math

    from pepin.fusion import from_fit, sigma_from_fit
    from pepin.odometry import Pose2D

    sharp_xy, sharp_yaw = sigma_from_fit(1.0)
    lost_xy, lost_yaw = sigma_from_fit(0.0)
    assert sharp_xy < lost_xy and sharp_yaw < lost_yaw
    assert sigma_from_fit(-5.0) == (lost_xy, lost_yaw), "a fit outside 0..1 is clamped"
    assert sigma_from_fit(9.0) == (sharp_xy, sharp_yaw)
    held = from_fit(Pose2D(1.0, 2.0, 0.5), 0.8, "tracker", stamp=7.0)
    assert held.pose == Pose2D(1.0, 2.0, 0.5) and held.fit == 0.8 and held.stamp == 7.0
    assert held.source == "tracker" and not held.edge
    sx, sy, syaw = held.sigmas
    assert sx == sy and math.isclose(sx, sigma_from_fit(0.8)[0])
    assert np.allclose(held.covariance, np.diag([sx**2, sy**2, syaw**2]))


# -- the covariance of the score peak -----------------------------------------

WIDE = SearchWindow(xy_m=0.30, xy_step_m=0.03, theta_deg=9.0, theta_step_deg=1.5)


def test_the_peak_of_a_corridor_is_a_ridge_along_the_corridor() -> None:
    """A scan that fits alike anywhere along a wall keeps its weight along it and loses it
    across: the covariance of that peak is an ellipse stretched along the wall, at any angle to
    the map's axes — not a circle, which is all an isotropic sigma could ever say."""
    across = peak_covariance(surface_of(200.0, 4.0, 20.0), TEST_T)
    sx, sy, _ = np.sqrt(np.diag(across))
    assert sy > 4 * sx, f"along {sy * 100:.1f} cm, across {sx * 100:.1f} cm"
    diagonal = peak_covariance(diagonal_ridge(200.0), TEST_T)
    along = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    tight = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    assert along @ diagonal @ along > 4 * (tight @ diagonal @ tight)
    assert np.all(np.linalg.eigvalsh(diagonal) > 0.0)


def test_a_colder_temperature_is_a_sharper_answer_and_the_scale_is_proportional() -> None:
    """The temperature is the only free scale, and around a quadratic peak the spread is
    proportional to it (``exp(-x^T H x / 2T)`` has covariance ``T H^-1``) — which is what lets
    the camera's be re-calibrated in one step from a recording (scratch/peak_temperature.py)."""
    broad = surface_of(4.0, 4.0, 200.0, WIDE)  # a peak the lattice resolves: 4 cm over a 3 cm step
    warm = peak_covariance(broad, 2 * TEST_T)
    cold = peak_covariance(broad, TEST_T)
    assert np.all(np.diag(cold) < np.diag(warm))
    assert np.trace(warm[:2, :2]) / np.trace(cold[:2, :2]) == pytest.approx(2.0, abs=0.3)


def test_a_peak_sharper_than_the_floor_stays_invertible() -> None:
    """A peak on one cell alone would read as zero spread and infinite information; the floor
    keeps it a covariance, a millimetre wide — a fifth of the lidar's measured error."""
    cov = peak_covariance(surface_of(1e6, 1e6, 1e8), TEST_T)
    sx, sy, st = np.sqrt(np.diag(cov))
    assert sx == pytest.approx(MIN_SIGMA_XY_M) and sy == pytest.approx(MIN_SIGMA_XY_M)
    assert st == pytest.approx(MIN_SIGMA_YAW_RAD)
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)


def test_a_sharp_lidar_is_not_moved_by_a_broad_camera_and_takes_its_word_when_lost() -> None:
    """What the peak covariance is for. The lidar's revolution pins the pose to half a
    centimetre; the camera's fan, matched 5 cm away with a peak a handful of centimetres wide,
    is weighed by its own spread and moves the fused pose by a fraction of a millimetre. Inflate
    the lidar's covariance to what a lost tracker is worth and the same camera measurement takes
    the pose over: nothing about the camera changed, only what it was weighed against.
    """
    lidar = from_peak(surface_of(80.0, 80.0, 20000.0), Pose2D(), 0.8, "lidar", temperature=TEST_T)
    camera = from_peak(
        surface_of(4.0, 4.0, 200.0, WIDE),
        Pose2D(0.05, 0.0, 0.0),
        0.5,
        "camera",
        temperature=TEST_T,
        trust=0.5,
    )
    sharp_cm, broad_cm = lidar.sigmas[0] * 100, camera.sigmas[0] * 100
    assert sharp_cm < 1.0 and broad_cm > 5.0, f"lidar {sharp_cm:.2f} cm, camera {broad_cm:.2f} cm"
    fused = fuse([lidar, camera])
    assert fused is not None and not fused.rejected
    assert abs(fused.x) < 0.001, f"the camera moved the sharp pose {fused.x * 1000:.2f} mm"
    sigma_xy, sigma_yaw = sigma_from_fit(0.0)
    lost = replace(lidar, covariance=np.diag([sigma_xy**2, sigma_xy**2, sigma_yaw**2]))
    taken = fuse([lost, camera])
    assert taken is not None and taken.x > 0.04, f"the camera moved a lost pose {taken.x:.3f} m"


def test_the_peak_is_read_between_the_candidates_not_on_them() -> None:
    """A peak that really lies between two candidates is not quantised to the lattice step: the
    parabola through the winner and its neighbours puts it where it is. The surface's own answer
    is the matcher's when the matcher interpolates."""
    shifted = surface_of(200.0, 200.0, 20.0)
    scores = shifted.scores.copy()
    scores[:, :] += 400.0 * shifted.positions[:, 0][None, :]  # tilt the peak off the centre in x
    tilted = ScoreSurface(
        scores,
        shifted.positions,
        shifted.headings,
        *divmod(int(np.argmax(scores)), scores.shape[1]),
        n_points=shifted.n_points,
        top=shifted.top,
    )
    peak = tilted.peak()
    on_lattice = tilted.positions[tilted.i]
    assert peak.x != on_lattice[0]
    assert abs(peak.x - on_lattice[0]) < WINDOW.xy_step_m / 2 + 1e-9
    assert peak.x == pytest.approx(0.01, abs=0.001)  # the tilt's own apex: 400 / (2 * 20000)


def test_a_real_fan_on_one_wall_peaks_as_a_ridge_along_that_wall() -> None:
    """The claim on a real match, not a synthetic surface: the +-40 degree fan of the furnished
    room facing the far wall is pinned across the wall and free to slide along it, so its peak
    covariance is a ridge — while the whole revolution's peak is tight in both directions. An
    isotropic sigma cannot say this, and it is what keeps a fan from dragging the pose sideways.
    """
    grid = furnished_room_map()
    matcher = CorrelativeMatcher(grid, max_points=200, interpolate=True)
    truth = Pose2D(0.5, 0.0, 0.0)
    whole = raycast_room(truth, beams=360, pillar=PILLAR)
    _, full_surface = matcher.match_surface(truth, whole, WINDOW)
    _, fan_surface = matcher.match_surface(truth, fan(whole), WINDOW)
    full = np.sqrt(np.diag(peak_covariance(full_surface, TEST_T)))
    wall = np.sqrt(np.diag(peak_covariance(fan_surface, TEST_T)))
    assert wall[1] > 3 * wall[0], f"along {wall[1] * 100:.1f} cm, across {wall[0] * 100:.1f} cm"
    assert wall[1] > 3 * full[1], f"fan {wall[1] * 100:.1f} cm, revolution {full[1] * 100:.1f} cm"
    assert full[0] < 0.02 and full[1] < 0.02, f"the revolution: {full[:2] * 100} cm"
