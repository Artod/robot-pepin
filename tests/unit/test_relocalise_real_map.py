"""The whole-map search on the real map, with real scans: the base must be found at the base.

These are the two scans of 2026-09-09 01:33: one recorded at home by run 0046 (tracked at fit
0.94) and one taken live after the cart had been carried to the base — the scan on which the
board's search had settled in a corner four metres away. The map is the one in force.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pepin.localization import Localizer
from pepin.mapping import grid_from_pgm
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import CorrelativeMatcher

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"


@pytest.fixture(scope="module")
def flat():  # type: ignore[no-untyped-def]
    return grid_from_pgm(REPO / "ros" / "maps" / "flat3_straight.yaml")


def points_of(scan: dict) -> np.ndarray:  # type: ignore[type-arg]
    angles = np.array(scan["angles"], dtype=float)
    ranges = np.array([np.nan if r is None else r for r in scan["ranges"]], dtype=float)
    ok = np.isfinite(ranges) & (ranges > 0.05)
    return np.column_stack((ranges[ok] * np.cos(angles[ok]), ranges[ok] * np.sin(angles[ok])))


def pose_of(d: dict) -> Pose2D:  # type: ignore[type-arg]
    return Pose2D(d["x"], d["y"], math.radians(d["theta_deg"]))


@pytest.mark.parametrize("name", ["scan_home_run0046", "scan_base_after_carry"])
def test_the_whole_map_search_finds_the_base_from_a_scan_at_the_base(flat, name: str) -> None:  # type: ignore[no-untyped-def]
    fixture = json.loads((FIXTURES / f"{name}.json").read_text())
    points = points_of(fixture["scan"])
    localizer = Localizer(flat, Pose2D(0.0, 0.0, 0.0), max_points=120, global_retry=False)
    # exactly the node's call: coarse heading step, thinned scan, no prior, twins allowed
    found, confidence = localizer.global_search(
        points, theta_step_deg=10.0, thin_to=90, prior=None, refuse_twins=False
    )
    truth = pose_of(fixture["truth"])
    assert math.hypot(found.pose.x - truth.x, found.pose.y - truth.y) < 0.4, found.pose
    assert abs(wrap_angle(found.pose.theta - truth.theta)) < math.radians(35.0), found.pose
    assert confidence > 0.6


def test_the_carry_scan_really_is_a_twin(flat) -> None:  # type: ignore[no-untyped-def]
    """The reason the confirmation rule exists: the same scan scores nearly as well in the
    corner the board chose as at the base. If this ever stops being true the rule is free."""
    fixture = json.loads((FIXTURES / "scan_base_after_carry.json").read_text())
    points = points_of(fixture["scan"])
    matcher = CorrelativeMatcher(flat)
    at_base = matcher.inlier_fraction(pose_of(fixture["truth"]), points)
    at_corner = matcher.inlier_fraction(pose_of(fixture["impostor"]), points)
    assert at_base > 0.6 and at_corner > 0.5, (at_base, at_corner)
