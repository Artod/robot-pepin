"""One tracker, any sources: the lidar path unchanged, a camera's fan alone, and both fused."""

import math
from itertools import pairwise

import numpy as np
import pytest
from synthetic import raycast_room
from test_localization import PILLAR, furnished_room_map

from pepin.dynamic import StaticMask, voting_mask
from pepin.localization import Localizer, ScanObservation
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import CorrelativeMatcher, SearchWindow, apply_motion, relative_motion
from pepin.sources import CONTACT, DEPTH, LIDAR, SourceRegistry

WINDOW = SearchWindow(xy_m=0.09, xy_step_m=0.03, theta_deg=9.0, theta_step_deg=1.5)
HALF_FAN_DEG = 40.0  # pepin.depth.SCAN_HALF_FOV: the camera's virtual scan


def whole(truth: Pose2D) -> np.ndarray:
    """The lidar's revolution from ``truth`` in the furnished room."""
    return raycast_room(truth, beams=180, pillar=PILLAR)


def fan(truth: Pose2D) -> np.ndarray:
    """What the camera's virtual scan sees from ``truth``: the returns within +-40 degrees."""
    points = whole(truth)
    bearing = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    return points[np.abs(bearing) <= HALF_FAN_DEG]


def drive() -> tuple[list[Pose2D], list[Pose2D]]:
    """A curved drive and the odometry that over-counts its turns by 15 %."""
    truth = [Pose2D(0.0, 0.0, 0.0)]
    for i in range(1, 25):
        truth.append(Pose2D(0.08 * i, 0.03 * i, 0.06 * i))
    odom = [truth[0]]
    for a, b in pairwise(truth):
        m = relative_motion(a, b)
        odom.append(apply_motion(odom[-1], Pose2D(m.x, m.y, m.theta * 1.15)))
    return truth, odom


def tracker(**overrides: object) -> Localizer:
    """The node's tracker on the furnished room."""
    settings: dict[str, object] = {
        "window": WINDOW,
        "max_points": 120,
        "correction_gain": 0.5,
        "lost_after": 1_000_000,
        "global_retry": False,
    }
    settings.update(overrides)
    return Localizer(furnished_room_map(), Pose2D(), **settings)  # type: ignore[arg-type]


def error(loc: Localizer, truth: Pose2D) -> tuple[float, float]:
    """(metres, degrees) between the tracked pose and ``truth``."""
    return (
        math.hypot(loc.pose.x - truth.x, loc.pose.y - truth.y),
        abs(math.degrees(wrap_angle(loc.pose.theta - truth.theta))),
    )


def test_update_is_update_from_with_the_lidar_and_nothing_changes() -> None:
    """The single-lidar entry and the general one are one code path: the same drive through
    both gives the same floats, update for update."""
    truth, odom = drive()
    old, new = tracker(), tracker()
    for o, t in zip(odom, truth, strict=True):
        a = old.update(o, whole(t))
        b = new.update_from(o, [ScanObservation(LIDAR, whole(t))])
        assert (a.x, a.y, a.theta) == (b.x, b.y, b.theta)
        assert old.confidence == new.confidence and old.published_fit == new.published_fit
    assert old.report().summary() == new.report().summary()
    assert error(new, truth[-1])[0] < 0.05 and error(new, truth[-1])[1] < 2.0
    assert new.measurements[0].source == LIDAR and new.measurements[0].fit == new.confidence


def test_a_disabled_source_is_ignored_and_nothing_to_match_holds_the_prediction() -> None:
    loc = tracker()  # the flag: lidar only
    loc.update_from(Pose2D(), [ScanObservation(DEPTH, fan(Pose2D()))])
    assert loc.pose == Pose2D() and loc.measurements == [] and loc.report().thin == 1
    loc.sources.enable([LIDAR, DEPTH])
    loc.update_from(Pose2D(), [ScanObservation(DEPTH, fan(Pose2D()))])
    assert [m.source for m in loc.measurements] == [DEPTH] and loc.report().matched == 1
    thin = np.array([[0.3, 0.3], [0.4, -0.2], [-0.3, 0.1]])  # under the source's floor
    loc.update_from(Pose2D(), [ScanObservation(DEPTH, thin)])
    assert loc.measurements == [] and loc.report().thin == 1


def test_a_fan_alone_pins_the_heading_and_a_corner_pins_the_position() -> None:
    """Camera-only: the +-40 degree fan facing the pillar's corner sees two walls and the box,
    and the tracker started 5 cm and 5 degrees off converges onto the truth with it alone."""
    truth = Pose2D(-0.5, 0.0, math.radians(135.0))  # the pillar dead ahead at 1.8 m
    loc = tracker(sources=SourceRegistry(enabled=[DEPTH]))
    loc.pose = Pose2D(truth.x + 0.05, truth.y, truth.theta + math.radians(5.0))
    scan = fan(truth)
    assert 30 <= len(scan) <= 45
    for _ in range(12):
        loc.update_from(truth, [ScanObservation(DEPTH, scan)])
    metres, degrees = error(loc, truth)  # a cell of position: forty beams on a 5 cm map
    assert metres < 0.06 and degrees < 1.5, f"{metres * 100:.1f} cm, {degrees:.1f} deg"
    assert loc.confidence > 0.8 and loc.measurements[0].source == DEPTH
    assert "sources depth" in loc.report().summary()


def test_both_sources_fuse_on_the_drive_and_the_fan_cannot_wreck_the_lidar() -> None:
    truth, odom = drive()
    alone = tracker()
    both = tracker(sources=SourceRegistry(enabled=[LIDAR, DEPTH]))
    for o, t in zip(odom, truth, strict=True):
        alone.update(o, whole(t))
        both.update_from(o, [ScanObservation(LIDAR, whole(t)), ScanObservation(DEPTH, fan(t))])
    assert [m.source for m in both.measurements] == [LIDAR, DEPTH]
    metres, degrees = error(both, truth[-1])
    assert metres < 0.05 and degrees < 2.0
    assert error(alone, truth[-1])[0] < 0.05
    stats = both.report()
    assert stats.fused == len(truth) and stats.matched == len(truth)
    assert set(stats.source_fit) == {LIDAR, DEPTH} and "fused 25" in stats.summary()


def test_fusion_off_takes_the_widest_source_alone_but_still_measures_the_rest() -> None:
    truth, odom = drive()
    alone = tracker()
    off = tracker(sources=SourceRegistry(enabled=[LIDAR, DEPTH]), fusion=False)
    for o, t in zip(odom, truth, strict=True):
        a = alone.update(o, whole(t))
        b = off.update_from(o, [ScanObservation(DEPTH, fan(t)), ScanObservation(LIDAR, whole(t))])
        assert (a.x, a.y, a.theta) == (b.x, b.y, b.theta)
    assert [m.source for m in off.measurements] == [DEPTH, LIDAR]
    assert off.report().fused == 0
    off.fusion = True  # the switch is live
    off.update_from(
        odom[-1], [ScanObservation(DEPTH, fan(truth[-1])), ScanObservation(LIDAR, whole(truth[-1]))]
    )
    assert off.report().fused == 1


def test_the_anchors_bound_is_taken_alone_and_a_blind_fan_never_delays_a_carry() -> None:
    """The cart stands 12 cm off in y, beyond the window, so the lidar's match sits on the
    window's edge (a bound); the fan facing the far wall is blind along y, a plateau whose
    winner is the tie-break toward the guess. Left to the fusion that plateau out-voted the
    widened lidar and held the wrong pose for seconds under the rest lock (a review probe,
    2026-09-11). Now the anchor's bound is taken alone and the plateau is a bound too: the
    fused tracker carries when the lidar-only one does, and follows it within half a
    centimetre, at rest and while driving."""
    truth = Pose2D(0.5, 0.0, 0.0)  # the fan faces the far wall at x = 3: pins x, blind in y
    lidar, camera = whole(truth), fan(truth)
    for at_rest in (True, False):
        alone = tracker()
        both = tracker(sources=SourceRegistry(enabled=[LIDAR, DEPTH]))
        alone.pose = both.pose = Pose2D(truth.x, truth.y + 0.12, truth.theta)
        errors: list[tuple[float, float]] = []
        for _ in range(8):
            alone.update_from(truth, [ScanObservation(LIDAR, lidar)], at_rest=at_rest, dt_s=0.1)
            both.update_from(
                truth,
                [ScanObservation(LIDAR, lidar), ScanObservation(DEPTH, camera)],
                at_rest=at_rest,
                dt_s=0.1,
            )
            errors.append((error(alone, truth)[0], error(both, truth)[0]))
            if len(errors) == 1:  # the first match: on the edge, taken alone, the same floats
                assert both.measurements[0].edge and "edge" in both.measurements[0].text()
                assert (both.pose.x, both.pose.y) == (alone.pose.x, alone.pose.y)
        for k, (a, b) in enumerate(errors):
            assert b <= a + 0.005, (
                f"at_rest={at_rest} update {k}: {b * 100:.1f} vs {a * 100:.1f} cm"
            )
        assert errors[0][0] > 0.07 and errors[-1][1] < 0.04
        one, two = alone.report(), both.report()
        assert two.carries == one.carries == (1 if at_rest else 0)
        assert two.bound >= 1 and two.fused + two.bound == 8 and one.bound == 0
        assert "anchor bound" in two.summary()


def test_a_partial_fan_is_not_penalised_as_unexplained() -> None:
    """A fan explains the map as well as the revolution where it looks (the fit is a share of
    the returns, never of the field of view), and it may vote with its own floor: the lidar's
    sixty explained returns would silence every fan."""
    grid = furnished_room_map()
    matcher = CorrelativeMatcher(grid, max_points=120, interpolate=True)
    truth = Pose2D(-0.5, 0.0, math.radians(135.0))
    assert (
        matcher.inlier_fraction(truth, fan(truth))
        >= matcher.inlier_fraction(truth, whole(truth)) - 0.1
    )
    mask = StaticMask(grid)
    assert voting_mask(fan(truth), truth, mask) is None  # the lidar's floor: too few to vote
    vote = voting_mask(fan(truth), truth, mask, min_points=20)
    assert vote is not None and vote.all()


def test_settings_name_the_sources_and_the_fusion_switch() -> None:
    loc = tracker(sources=SourceRegistry(enabled=[LIDAR, CONTACT]), fusion=False)
    assert "sources lidar,contact, fusion off" in loc.settings()
    loc.sources.enable([])
    assert "sources none" in loc.settings()
    with pytest.raises(ValueError):
        loc.sources.enable(["radar"])
