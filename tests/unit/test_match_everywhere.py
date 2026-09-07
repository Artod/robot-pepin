"""The exhaustive whole-map search: any position, any heading, no start guess."""

import math
from itertools import pairwise

from synthetic import raycast_room
from test_localization import PILLAR, furnished_room_map

from pepin.localization import Localizer
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import CorrelativeMatcher


def test_finds_the_true_pose_from_nowhere_at_a_wide_heading_step() -> None:
    grid = furnished_room_map()
    truth = Pose2D(-1.1, 0.4, math.radians(137.0))
    points = raycast_room(truth, pillar=PILLAR)
    peaks = CorrelativeMatcher(grid).match_everywhere(points, theta_step_deg=8.0, top_k=4)
    # The pass is exhaustive but coarse (8 deg lattice): the truth must be among its peaks;
    # ranking them is the refine's job (see the pooling test below).
    assert any(
        math.hypot(p.pose.x - truth.x, p.pose.y - truth.y) < 0.25
        and abs(wrap_angle(p.pose.theta - truth.theta)) < math.radians(9.0)
        for p in peaks
    )


def test_peaks_are_distinct_places_and_come_best_first() -> None:
    grid = furnished_room_map()
    points = raycast_room(Pose2D(0.3, -0.5, 0.2), pillar=PILLAR)
    peaks = CorrelativeMatcher(grid).match_everywhere(points, theta_step_deg=8.0, top_k=6)
    assert 2 <= len(peaks) <= 6
    assert all(a.score >= b.score for a, b in pairwise(peaks))
    for i, a in enumerate(peaks):
        for b in peaks[i + 1 :]:
            far = math.hypot(a.pose.x - b.pose.x, a.pose.y - b.pose.y) >= 3 * grid.spec.resolution_m
            turned = abs(wrap_angle(a.pose.theta - b.pose.theta)) > math.radians(8.0)
            assert far or turned


def test_pooling_keeps_the_answer_and_the_field_score_ranks_the_truth_first() -> None:
    grid = furnished_room_map()
    truth = Pose2D(0.7, 0.9, math.radians(-60.0))
    points = raycast_room(truth, pillar=PILLAR)
    matcher = CorrelativeMatcher(grid)
    pooled_peaks = matcher.match_everywhere(points, theta_step_deg=8.0, top_k=4, pool=2)
    loc = Localizer(grid, Pose2D())
    refined = [loc.refine(p.pose, points)[0].pose for p in pooled_peaks]
    scores = [matcher.field_score(p, points) for p in refined]
    winner = refined[scores.index(max(scores))]
    assert math.hypot(winner.x - truth.x, winner.y - truth.y) < 0.1
    assert abs(wrap_angle(winner.theta - truth.theta)) < math.radians(3.0)


def test_global_search_localises_a_carried_robot_without_a_prior() -> None:
    grid = furnished_room_map()
    truth = Pose2D(-0.4, -1.1, math.radians(75.0))
    loc = Localizer(grid, Pose2D())  # believes it is at the origin: it was carried
    best, confidence = loc.global_search(raycast_room(truth, pillar=PILLAR))
    assert confidence > 0.7
    assert math.hypot(best.pose.x - truth.x, best.pose.y - truth.y) < 0.1
