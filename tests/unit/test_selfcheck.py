"""A source vouching for itself: does its next answer land where its last one predicted?

Every number here is about ONE sensor's repeatability against its own past, carried over the
odometry; the point of the check is that no sensor is ever judged by another's pose, and two of
these tests exist only to prove that path does not exist.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_localizer_sources import drive, tracker, whole

from pepin.fusion import PoseMeasurement
from pepin.localization import SWITCHES
from pepin.odometry import Pose2D
from pepin.selfcheck import MAX_GAP_S, MAX_INFLATION, SelfCheck
from pepin.sources import CAMERA, DEPTH, LIDAR, ScanObservation

CLAIM_XY_M = 0.02  # what every measurement below claims: 2 cm and a degree
CLAIM_YAW_DEG = 1.0
SCATTER = 4.0  # the jumping source is this many times worse than it claims


def measurement(
    pose: Pose2D,
    source: str = DEPTH,
    stamp: float = 0.0,
    xy_m: float = CLAIM_XY_M,
    yaw_deg: float = CLAIM_YAW_DEG,
) -> PoseMeasurement:
    """One source's word on the pose, claiming ``xy_m`` and ``yaw_deg`` of spread."""
    covariance = np.diag([xy_m**2, xy_m**2, math.radians(yaw_deg) ** 2])
    return PoseMeasurement(pose.x, pose.y, pose.theta, covariance, source, stamp, 0.6)


def scattered(n: int, seed: int = 0) -> list[Pose2D]:
    """``n`` poses of a cart standing still, seen by a source that scatters ``SCATTER`` times
    as far as it claims in all three directions."""
    rng = np.random.default_rng(seed)
    return [
        Pose2D(
            float(rng.normal(0.0, SCATTER * CLAIM_XY_M)),
            float(rng.normal(0.0, SCATTER * CLAIM_XY_M)),
            float(rng.normal(0.0, SCATTER * math.radians(CLAIM_YAW_DEG))),
        )
        for _ in range(n)
    ]


def test_a_source_that_scatters_four_times_its_claim_is_inflated_sixteenfold() -> None:
    """The failure of 2026-09-13 in miniature: a source claims 2 cm while its answers jump 8 cm
    at rest. Its own record says so within a window's worth of measurements, and its covariance
    is multiplied by the variance ratio it earned — about (8/2)^2."""
    check = SelfCheck()
    at_rest = Pose2D()
    out = [
        check.checked(measurement(pose, stamp=0.1 * k), at_rest)
        for k, pose in enumerate(scattered(21))
    ]
    assert 10.0 < check.ratio(DEPTH) < 25.0  # 16 in expectation, 20 samples of 3 dof
    assert check.inflation(DEPTH) == pytest.approx(check.ratio(DEPTH))
    widened = out[-1].covariance / np.diag([CLAIM_XY_M**2] * 2 + [math.radians(1.0) ** 2])
    assert np.allclose(np.diag(widened), check.inflation(DEPTH))


def test_the_inflation_arrives_within_a_few_measurements() -> None:
    """It must not take the whole window to notice: four measurements already carry it past 4x."""
    check = SelfCheck()
    for k, pose in enumerate(scattered(5)):
        check.checked(measurement(pose, stamp=0.1 * k), Pose2D())
    assert check.inflation(DEPTH) > 4.0


def test_a_consistent_source_is_untouched() -> None:
    """A source whose every answer lands exactly where the odometry carried the previous one is
    not touched at all — the same object comes back out, covariance and all."""
    check = SelfCheck()
    odom = [Pose2D(0.02 * k, 0.0, 0.05 * k) for k in range(10)]
    for k, pose in enumerate(odom):
        given = measurement(pose, stamp=0.1 * k)
        assert check.checked(given, pose) is given
    assert check.ratio(DEPTH) < 1e-20  # the prediction is the measurement, to the last bit
    assert check.inflation(DEPTH) == 1.0


def test_a_source_better_than_it_claims_keeps_its_covariance() -> None:
    """Half the claimed scatter is a ratio of 0.25 and still no change: the check takes an
    over-claim back, it never rewards a source for being better than its covariance."""
    check = SelfCheck()
    for k, pose in enumerate(scattered(21)):
        half = Pose2D(pose.x / 8.0, pose.y / 8.0, pose.theta / 8.0)
        out = check.checked(measurement(half, stamp=0.1 * k), Pose2D())
        assert out.covariance[0, 0] == CLAIM_XY_M**2
    assert check.ratio(DEPTH) < 1.0
    assert check.inflation(DEPTH) == 1.0


def test_the_inflation_is_capped() -> None:
    """A source that answers metres apart while claiming centimetres is capped, not silenced:
    the fusion's own disagreement gate is what refuses a wild pose."""
    check = SelfCheck()
    for k in range(6):
        check.checked(measurement(Pose2D(2.0 * (k % 2), 0.0, 0.0), stamp=0.1 * k), Pose2D())
    assert check.inflation(DEPTH) == MAX_INFLATION


def test_a_gap_starts_the_record_over() -> None:
    """A source that went away for longer than the odometry carry is honest over is not judged
    across the hole: the record restarts and the measurement passes untouched."""
    check = SelfCheck()
    for k, pose in enumerate(scattered(21)):
        check.checked(measurement(pose, stamp=0.1 * k), Pose2D())
    assert check.inflation(DEPTH) > 1.0
    late = measurement(Pose2D(), stamp=2.1 + MAX_GAP_S)
    assert check.checked(late, Pose2D()) is late
    assert check.ratio(DEPTH) == 1.0
    assert check.record(DEPTH).samples == 0


def test_a_slipping_wheel_is_not_the_sensor_s_fault() -> None:
    """With the odometry step untrusted the prediction is wrong for a reason that has nothing
    to do with the sensor, so nothing is recorded."""
    check = SelfCheck()
    for k, pose in enumerate(scattered(11)):
        check.checked(measurement(pose, stamp=0.1 * k), Pose2D(), trust_odometry=False)
    assert check.record(DEPTH).samples == 0
    assert check.inflation(DEPTH) == 1.0


def test_off_measures_but_does_not_widen() -> None:
    """The flag off is the old behaviour exactly — and the ratio is still measured, so the
    report line shows what turning it on would do before anybody turns it on."""
    check = SelfCheck(enabled=False)
    for k, pose in enumerate(scattered(21)):
        out = check.checked(measurement(pose, stamp=0.1 * k), Pose2D())
        assert out.covariance[0, 0] == CLAIM_XY_M**2
    assert check.record(DEPTH).ratio > 10.0  # measured...
    assert check.inflation(DEPTH) == 1.0  # ...and not applied
    assert check.text().startswith("self_check off")


def test_one_source_never_reads_another_s_pose() -> None:
    """The whole point: a jumping camera interleaved with a steady lidar changes nothing about
    the lidar's record, and a lidar-only run gives bit-for-bit the same numbers."""
    both, lidar_only, camera_only = SelfCheck(), SelfCheck(), SelfCheck()
    odom = [Pose2D(0.02 * k, 0.0, 0.0) for k in range(21)]
    for k, (pose, wild) in enumerate(zip(odom, scattered(21), strict=True)):
        both.checked(measurement(pose, source=LIDAR, stamp=0.1 * k), pose)
        both.checked(measurement(wild, source=CAMERA, stamp=0.1 * k), pose)
        lidar_only.checked(measurement(pose, source=LIDAR, stamp=0.1 * k), pose)
        camera_only.checked(measurement(wild, source=CAMERA, stamp=0.1 * k), pose)
    assert both.inflation(LIDAR) == 1.0
    assert both.inflation(CAMERA) > 10.0
    # Bit for bit what each source's own run gives: neither record saw the other's pose.
    assert both.ratio(LIDAR) == lidar_only.ratio(LIDAR)
    assert both.ratio(CAMERA) == camera_only.ratio(CAMERA)


def test_the_flag_is_the_tracker_s_and_reads_in_its_report_line() -> None:
    """``self_check`` is a live switch of the tracker (``ros2 param set``) and its state and
    every source's factor are in the line the node prints."""
    assert "self_check" in SWITCHES
    loc = tracker()
    assert "self_check on" in loc.settings()
    loc.switch("self_check", False)
    assert loc.self_check is False
    assert "self_check off" in loc.settings()


@pytest.mark.parametrize("with_camera", [False, True])
def test_the_tracker_s_lidar_is_never_judged_by_the_camera(with_camera: bool) -> None:
    """Through the tracker's own update path: a clean lidar drive keeps its covariance, and a
    camera measurement riding along — however wrong — leaves the lidar's ratio untouched."""
    truth, odom = drive(steps=8)
    loc = tracker()
    for t, o in zip(truth, odom, strict=True):
        extra = (
            [measurement(Pose2D(t.x + 0.5, t.y - 0.5, t.theta), source=CAMERA, stamp=0.0)]
            if with_camera
            else []
        )
        loc.update_from(o, [ScanObservation(LIDAR, whole(t))], measurements=extra)
    entry = loc.sources_report(0.0)["sources"][LIDAR]
    assert entry["self_check"][1] == 1.0  # the lidar vouched for itself, nothing widened
    assert loc.sources_report(0.0)["sources"][LIDAR]["self_check"][0] < 1.0
