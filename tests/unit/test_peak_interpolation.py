"""The matcher answers between its candidates, not only on them."""

import math

import numpy as np
from synthetic import raycast_room
from test_localization import PILLAR, furnished_room_map

from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import CorrelativeMatcher, SearchWindow

WINDOW = SearchWindow(xy_m=0.09, xy_step_m=0.03, theta_deg=9.0, theta_step_deg=1.5)


def test_a_parabola_apex_is_found_between_samples() -> None:
    apex = CorrelativeMatcher._apex
    assert apex(1.0, 2.0, 1.0) == 0.0  # symmetric: the middle sample is the peak
    assert 0.2 < apex(1.0, 2.0, 1.8) < 0.5  # leaning right
    assert -0.5 < apex(1.8, 2.0, 1.0) < -0.2  # leaning left
    assert apex(1.0, 0.5, 1.0) == 0.0  # a minimum is not a peak
    assert apex(1.0, 1.0, 1.0) == 0.0  # flat


def test_the_correction_is_not_quantised_by_the_heading_step() -> None:
    """Half a lattice step of heading error must come back as about half a step, not zero or one."""
    grid = furnished_room_map()
    matcher = CorrelativeMatcher(grid, interpolate=True)
    truth = Pose2D(0.2, -0.4, math.radians(15.0))
    points = raycast_room(truth, pillar=PILLAR)
    off_lattice = []
    for error_deg in (0.7, -0.7, 1.1, -1.1):
        guess = Pose2D(truth.x, truth.y, truth.theta + math.radians(error_deg))
        found = matcher.match(guess, points, WINDOW).pose
        corrected = math.degrees(wrap_angle(found.theta - guess.theta))
        off_lattice.append(abs(corrected - 1.5 * round(corrected / 1.5)) > 0.15)
        assert abs(math.degrees(wrap_angle(found.theta - truth.theta))) < 1.0
    assert any(off_lattice), "every correction still landed on the lattice"


def test_the_position_correction_is_not_quantised_either() -> None:
    """The shift applied to the guess must be able to land between lattice points."""
    grid = furnished_room_map()
    matcher = CorrelativeMatcher(grid, interpolate=True)
    truth = Pose2D(0.31, -0.22, math.radians(-40.0))
    points = raycast_room(truth, pillar=PILLAR)
    off_lattice = 0
    for dx, dy in ((0.014, -0.011), (0.02, 0.0), (0.0, 0.012), (-0.016, 0.009)):
        guess = Pose2D(truth.x + dx, truth.y + dy, truth.theta)
        found = matcher.match(guess, points, WINDOW).pose
        for shift in (found.x - guess.x, found.y - guess.y):
            if abs(shift - 0.03 * round(shift / 0.03)) > 0.002:
                off_lattice += 1
    assert off_lattice >= 4, "the position correction still snapped to the lattice"


def test_interpolation_never_leaves_the_winning_cell() -> None:
    """A shifted pose must stay within half a step of the candidate that won."""
    matcher = CorrelativeMatcher(furnished_room_map(), interpolate=True)
    rng = np.random.default_rng(3)
    positions = np.array(
        [(dx, dy) for dx in np.arange(-2, 3) * 0.03 for dy in np.arange(-2, 3) * 0.03]
    )
    headings = np.radians(np.arange(-2, 3) * 1.5)
    for _ in range(50):
        scores = rng.normal(size=(len(headings), len(positions)))
        pose, _ = matcher._peak(scores, positions, headings)
        k, i = np.unravel_index(int(np.argmax(scores)), scores.shape)
        assert abs(pose.x - positions[i, 0]) <= 0.015 + 1e-9
        assert abs(pose.y - positions[i, 1]) <= 0.015 + 1e-9
        assert abs(wrap_angle(pose.theta - headings[k])) <= math.radians(0.75) + 1e-9
