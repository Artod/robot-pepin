"""ros/tools/odom_bench: the three-cornered hat, the windows and the scores on generated data
whose answers are known, and the frozen drive set's file."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ros" / "tools"))

from odom_bench import hat, metrics, rewrite, sources, truth  # noqa: E402
from odom_bench.drives import Drive, DriveSet, goal_span  # noqa: E402
from vio_score import Trajectory  # noqa: E402

from pepin.wheel_noise import WheelNoiseLaw  # noqa: E402


def triple(n: int, sigmas: tuple[float, float, float], seed: int = 3) -> np.ndarray:
    """Hat rows (V, sV, W, sW, X, sX, T): one signal seen by three sources whose errors are
    exactly uncorrelated with each other and with it (orthonormal columns), so the hat is exact."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(np.column_stack([np.ones(n), rng.normal(size=(n, 4))]))
    basis = q[:, 1:] * math.sqrt(n)  # zero mean (orthogonal to the constant column), unit power
    signal = 0.2 + 0.1 * basis[:, 0]
    rows = np.zeros((n, 7))
    for p, s in enumerate(sigmas):
        rows[:, 2 * p] = signal + s * basis[:, p + 1]
        rows[:, 2 * p + 1] = s
    rows[:, 6] = signal
    return rows


def test_signed_sqrt_keeps_the_sign_of_a_negative_mean_square() -> None:
    assert hat.sq(9.0) == 3.0
    assert hat.sq(-4.0) == -2.0


def test_the_hat_recovers_each_sources_own_variance_without_the_truth() -> None:
    sigmas = (0.01, 0.02, 0.04)
    rows = triple(500, sigmas)
    for p, s in enumerate(sigmas):
        assert np.mean(hat.own2(rows, p)) == pytest.approx(s**2, rel=1e-9)


def test_the_hat_ignores_what_the_three_share() -> None:
    rows = triple(100, (0.01, 0.02, 0.04))
    shifted = rows.copy()
    shifted[:, [0, 2, 4]] += 0.37  # a common error is nobody's own
    for p in range(3):
        assert np.allclose(hat.own2(shifted, p), hat.own2(rows, p), atol=1e-12)


def test_boot_mean_pools_the_rows_of_every_drive_and_skips_nan() -> None:
    q = {"0001": np.array([1.0, 2.0, np.nan]), "0002": np.array([3.0])}
    mean, lo, hi, draws = hat.boot_mean(q, ["0001", "0002"])
    assert mean == 2.0
    assert lo <= mean <= hi
    assert len(draws) == hat.NBOOT
    assert hat.boot_mean(q, ["0001", "0002"])[1:3] == (lo, hi)  # seeded


def test_boot_ratio_is_the_ratio_of_the_pooled_sums() -> None:
    qn = {"a": np.array([1.0, 3.0]), "b": np.array([2.0, np.nan])}
    qd = {"a": np.array([2.0, 2.0]), "b": np.array([4.0, 1.0])}
    assert hat.boot_ratio(qn, qd, ["a", "b"])[0] == pytest.approx(6.0 / 8.0)


def test_sokal_reads_the_correlation_time_of_white_and_ar1_errors() -> None:
    rng = np.random.default_rng(5)
    n, rho = 20000, 0.5
    ar = np.zeros(n)
    noise = rng.normal(0.0, 1.0, n)
    for i in range(1, n):
        ar[i] = rho * ar[i - 1] + noise[i]
    errors = np.column_stack([ar, rng.normal(0.0, 1.0, n), rng.normal(0.0, 1.0, n)])
    maxlag = 30
    pairs = ((0, 1), (0, 2), (1, 2))
    sums = np.zeros((1, 3, maxlag + 1))
    counts = np.zeros((1, 3, maxlag + 1))
    for j, (x, y) in enumerate(pairs):
        d = errors[:, x] - errors[:, y]
        for lag in range(maxlag + 1):
            sums[0, j, lag] = np.sum(d[: n - lag] * d[lag:])
            counts[0, j, lag] = n - lag
    out = hat.sokal(sums, counts, maxlag)
    tau = out[:, maxlag + 1]  # s, 0.1 s bins
    assert tau[0] == pytest.approx(0.1 * (1 + rho) / (1 - rho), rel=0.15)
    assert tau[1] == pytest.approx(0.1, abs=0.03)
    assert tau[2] == pytest.approx(0.1, abs=0.03)
    assert out[0, 1] == pytest.approx(rho, abs=0.05)  # rho at one bin


def test_floors_is_the_truths_own_error_from_the_four_source_hat() -> None:
    rng = np.random.default_rng(7)
    n = 4000
    move = rng.normal(0.5, 0.2, (n, 3))
    err = {"T": 0.03, "V": 0.01, "W": 0.02, "R": 0.015, "G": 0.005}
    seen = {k: (move + rng.normal(0.0, s, (n, 3))).tolist() for k, s in err.items()}
    windows = [{k: tuple(seen[k][i]) for k in err} for i in range(n)]
    rows = hat.floor_parts({"disp": {"s1": windows}}, "s1")
    fl = hat.floors({"0001": rows}, ["0001"], across=True)
    assert fl[0] == pytest.approx(0.03**2, rel=0.1)  # along: the VWR hat
    assert fl[1] == pytest.approx(0.03**2, rel=0.1)  # across, only when asked
    assert fl[2] == pytest.approx(0.03**2, rel=0.1)  # yaw: the GVW hat
    assert hat.floors({"0001": rows}, ["0001"])[1] == 0.0


def test_a_window_mean_is_overlap_weighted_and_its_sigma_assumes_independence() -> None:
    src = sources.Src(
        np.array([0.0, 1.0]),
        np.array([1.0, 2.0]),
        {"vx": np.array([1.0, 3.0])},
        {"vx": np.array([1.0, 1.0])},
    )
    mean, sd = src.over("vx", 0.5, 1.5, 0.9) or (math.nan, math.nan)
    assert mean == pytest.approx(2.0)
    assert sd == pytest.approx(math.sqrt(0.5**2 + 0.5**2))
    assert src.over("vx", 0.5, 1.5, 1.1) is None  # under the coverage asked
    assert src.pieces(0.5, 1.5) == [(0.5, 1.0), (1.0, 1.5)]


def test_integrating_a_constant_twist_follows_the_arc() -> None:
    a = np.arange(0.0, 2.0, 0.01)
    v, w = 0.3, 0.4
    src = sources.Src(
        a,
        a + 0.01,
        {"vx": np.full(len(a), v), "vy": np.zeros(len(a)), "wz": np.full(len(a), w)},
        {},
    )
    dx, dy, dth = src.integ(0.0, 1.0, 0.8) or (math.nan, math.nan, math.nan)
    assert dth == pytest.approx(w)
    assert dx == pytest.approx(v * math.sin(w) / w, abs=1e-5)
    assert dy == pytest.approx(v * (1 - math.cos(w)) / w, abs=1e-5)


def test_the_truths_fit_reads_a_straight_drives_speed() -> None:
    t = np.arange(0.0, 2.0, 0.1)
    yaw = 0.7
    tr = np.column_stack(
        [t, 0.3 * t * math.cos(yaw), 0.3 * t * math.sin(yaw), np.full(len(t), yaw)]
    )
    vx, wz = truth.fit_vw(tr, 1.0) or (math.nan, math.nan)
    assert vx == pytest.approx(0.3)
    assert wz == pytest.approx(0.0, abs=1e-12)
    assert truth.fit_vw(tr[:3], 0.1) is None  # under 4 scans


def test_pose_nees_divides_by_the_covariance_grown_over_the_window() -> None:
    ca = np.zeros(10)
    cb = np.array([0.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    assert metrics.pose_nees(np.array([2.0, 0.0, 0.0]), ca, cb, 0.0) == pytest.approx(1.0 / 3.0)
    # the same growth seen from a start turned by 90 deg: the error along y meets the 4
    assert metrics.pose_nees(np.array([0.0, 2.0, 0.0]), ca, cb, math.pi / 2) == pytest.approx(
        1.0 / 3.0
    )
    assert math.isnan(metrics.pose_nees(np.ones(3), cb, ca, 0.0))  # shrinking: not a growth


def test_the_end_error_splits_along_and_across_the_displacement() -> None:
    t = np.linspace(0.0, 10.0, 11)
    tr = Trajectory(t, t / 10, np.zeros(11), np.zeros(11))
    arm = Trajectory(t, t / 10, 0.005 * t, np.zeros(11))
    err, along, across, head, disp = metrics.end_parts(tr, arm)
    assert err == pytest.approx(5.0)
    assert along == pytest.approx(0.0, abs=1e-9)
    assert across == pytest.approx(5.0)  # + left
    assert head == pytest.approx(0.0)
    assert disp == pytest.approx(1.0)


def test_metrics_net_of_a_zero_floor_is_the_error_as_read() -> None:
    rows = np.array([[0.03, 0.04, 0.0, 0.10, 1.0, math.nan], [0.0, 0.05, 0.0, 0.10, 1.0, 2.0]])
    scores = {"0001": {"s1": rows, "m1": rows, "end": (5.0, 0.0, 5.0, 1.0, 1.0)}}
    floor_rows = {"0001": np.zeros((1, 12))}
    m = metrics.metrics(scores, floor_rows, ["0001"])
    assert m["s1_read"] == pytest.approx(0.05)
    assert m["s1_net"] == pytest.approx(0.05)
    assert m["s1_pct_net"] == pytest.approx(50.0)
    assert m["s1_pnees"] == pytest.approx(2.0)
    assert m["end_mean"] == 5.0
    assert m["head_abs"] == 1.0


def test_scaling_a_twist_covariance_scales_the_cross_terms_by_the_product() -> None:
    cov = np.ones(36)
    rewrite.scale_twist_cov(cov, 4.0, 9.0)
    assert cov[0] == cov[1] == cov[6] == cov[7] == 4.0
    assert cov[35] == 9.0
    assert cov[5] == cov[30] == cov[11] == cov[31] == 6.0
    assert cov[14] == 1.0
    held = np.ones(36)
    held[0] = 1e5  # a withheld twist: only its yaw rate counts
    rewrite.scale_twist_cov(held, 4.0, 9.0)
    assert held[35] == 9.0
    assert held[0] == 1e5 and held[7] == 1.0 and held[5] == 1.0


def test_the_wheel_law_is_fifty_samples_of_the_one_second_sigma() -> None:
    law = WheelNoiseLaw()
    sv, sw = law.sigmas(0.5)
    assert rewrite.law_variances(law, -0.5) == pytest.approx((50 * sv**2, 50 * sw**2))


def tiny_set(tmp_path: Path) -> DriveSet:
    """Three 20 fps drives: two references and one recorded with the sqrt(rate) rule."""
    drives = {
        n: Drive(n, f"{n}_20261007_000000Z_home", "20fps", n != "0003", n == "0003", "", "", None)
        for n in ("0001", "0002", "0003")
    }
    groups = {"20fps": 20}
    return DriveSet(
        tmp_path / "s.json", tmp_path / "m.yaml", 3.0, 87.5, groups, "20fps", "0002", drives
    )


def detected(lin: float, yaw: float) -> dict[str, object]:
    return {
        "matched": 10,
        "lin_p10": lin,
        "yaw_p50": yaw,
        "odom_moving": 5,
        "law_ratio_p50": 1.0,
        "law_ratio_p90": 0.0,
    }


def test_claims_recorded_under_the_rule_are_brought_back_to_the_base(tmp_path: Path) -> None:
    det = {"0001": detected(3.0, 6.25), "0002": detected(3.1, 6.25), "0003": detected(6.1, 12.5)}
    claims = rewrite.classify(det, tiny_set(tmp_path))
    assert claims["0001"].vio_lin_var == 1.0
    assert claims["0003"].vio_lin_var == claims["0003"].vio_yaw_var == 0.5
    assert claims["0003"].status.endswith("rule x1.41 removed")
    assert claims["0003"].law_live


def test_a_claim_off_every_known_configuration_stops_the_bench(tmp_path: Path) -> None:
    det = {"0001": detected(3.0, 6.25), "0002": detected(3.1, 6.25), "0003": detected(3.0, 6.25)}
    with pytest.raises(SystemExit, match="off every known configuration"):
        rewrite.classify(det, tiny_set(tmp_path))


def test_the_goal_span_is_the_tapes_first_executing_goal(tmp_path: Path) -> None:
    tape = tmp_path / "t.jsonl"
    rows = [
        {"topic": "nav", "action": "navigate_to_pose", "status": [4], "t": 1.0},
        {"topic": "nav", "action": "navigate_to_pose", "status": [2], "t": 2.0},
        {"topic": "nav", "action": "navigate_to_pose", "status": [2], "t": 3.0},
        {"topic": "nav", "action": "navigate_to_pose", "status": [4], "t": 9.5},
    ]
    tape.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n")
    assert goal_span(tape) == (2.0, 9.5)


def test_the_frozen_set_names_every_drive_with_its_hashes() -> None:
    s = DriveSet.load(REPO / "ros/tools/odom_bench/sets/0355-0411.json")
    assert len(s.drives) == 43
    assert s.live_group == "20fps" and s.old_group == "10fps"
    assert s.truth_map.exists()
    assert s.replay_check in s.drives
    for d in s.drives.values():
        assert d.bag.startswith(d.n + "_")
        for h in (d.mcap_sha256, d.tape_sha256, d.truth_sha256):
            assert h is not None and len(h) == 64
