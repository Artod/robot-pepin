"""ros/tools/vio_score.py: the pre-registered metrics read back what synthetic trajectories hold."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def _tool() -> Any:
    spec = importlib.util.spec_from_file_location("vio_score", REPO / "ros/tools/vio_score.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("vio_score", module)
    spec.loader.exec_module(module)
    return module


TOOL = _tool()


def _drive(length_m: float = 12.0, speed: float = 0.2, turn: float = 0.0) -> Any:
    """The truth: a cart at ``speed`` along an arc (``turn`` rad/m), sampled at 10 Hz."""
    t = np.arange(0.0, length_m / speed, 0.1)
    s = speed * t
    yaw = turn * s
    if turn:
        x, y = np.sin(yaw) / turn, (1 - np.cos(yaw)) / turn
    else:
        x, y = s, np.zeros_like(s)
    return TOOL.Trajectory(t, x, y, yaw)


def _scaled(truth: Any, scale: float, offset_yaw: float = 0.7) -> Any:
    """The same motion with a scale error, seen in another odometry frame (rotated, shifted)."""
    c, s = math.cos(offset_yaw), math.sin(offset_yaw)
    x, y = truth.x * scale, truth.y * scale
    return TOOL.Trajectory(
        truth.t, 3.0 + c * x - s * y, -1.0 + s * x + c * y, truth.yaw + offset_yaw
    )


def test_a_two_percent_scale_error_reads_two_percent_per_metre_whatever_the_frame() -> None:
    truth = _drive(turn=0.1)
    score = TOOL.score(truth, _scaled(truth, 1.02))
    assert len(score.rpe) == 11, "eleven whole metres of a 12 m path"
    assert np.median(score.rpe) == pytest.approx(0.02, rel=0.02)
    assert np.median(score.yaw_rpe) == pytest.approx(0.0, abs=1e-6)
    assert score.end_point_m == pytest.approx(0.02 * math.hypot(truth.x[-1], truth.y[-1]), rel=0.02)
    perfect = TOOL.score(truth, _scaled(truth, 1.0))
    assert np.max(perfect.rpe) == pytest.approx(0.0, abs=1e-9)


def test_an_arm_that_does_not_cover_a_segment_leaves_it_out() -> None:
    truth = _drive()
    half = TOOL.Trajectory(truth.t[:300], truth.x[:300], truth.y[:300], truth.yaw[:300])
    score = TOOL.score(truth, half)
    assert np.isnan(score.rpe[-1]) and np.isfinite(score.rpe[0])
    assert math.isnan(score.end_point_m)


def test_the_bootstrap_ci_excludes_zero_for_a_one_cm_per_metre_paired_difference() -> None:
    """Six drives of ten 1 m segments each (60), B at 3 %/m and E at 2 %/m with 0.5 %/m of
    per-segment noise: the CI of median(E) - median(B) lies below 0 and E wins every drive; the
    same arm against itself does not pass."""
    rng = np.random.default_rng(7)
    drives = {}
    for k in range(6):
        b = 0.03 + rng.normal(0, 0.005, 10)
        e = b - 0.01 + rng.normal(0, 0.002, 10)
        drives[f"06{k:02d}"] = {
            "B": TOOL.DriveScore(b, np.zeros(10), 0.1),
            "E": TOOL.DriveScore(e, np.zeros(10), 0.1),
        }
    result = TOOL.compare(drives, "E", "B", resamples=1000, seed=1)
    assert result.segments == 60 and result.wins == 6
    assert result.difference == pytest.approx(-0.01, abs=0.003)
    assert result.ci[1] < 0.0 and result.passes(min_wins=5)
    same = TOOL.compare(drives, "B", "B", resamples=200, seed=1)
    assert not same.passes(min_wins=5), "no difference is no pass"


def test_the_table_from_csv_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    truth = _drive()
    for name, arm in (("truth", truth), ("B", _scaled(truth, 1.03)), ("E", _scaled(truth, 1.01))):
        rows = ["stamp,x,y,yaw"] + [
            f"{t},{x},{y},{yaw}" for t, x, y, yaw in zip(arm.t, arm.x, arm.y, arm.yaw, strict=True)
        ]
        (tmp_path / f"{name}.csv").write_text("\n".join(rows) + "\n")
    assert (
        TOOL.main([str(tmp_path), "--arms", "B", "E", "--resamples", "100", "--min-wins", "1"]) == 0
    )
    out = capsys.readouterr().out
    assert "B: RPE 3.00 %/m" in out and "E: RPE 1.00 %/m" in out
    assert "E - B: -2.00 %/m" in out and "PASSES" in out


def test_the_offline_stereo_odometry_takes_the_live_launch_s_own_table() -> None:
    """ros/vio_replay.sh's arm B runs stereo_odometry with vslam.launch.py's VISUAL_ODOMETRY,
    read from the launch file (ros/tools/vo_params.py), plus the frame the launch adds."""
    import yaml

    spec = importlib.util.spec_from_file_location("vo_params", REPO / "ros/tools/vo_params.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = yaml.safe_load(module.as_yaml(module.table()))["stereo_odometry"]["ros__parameters"]
    assert values["frame_id"] == "base_link" and values["publish_tf"] is False
    assert values["guess_frame_id"] == "" and values["Odom/ResetCountdown"] == "1"


def test_the_replay_never_plays_the_drive_s_own_visual_odometry_into_the_arm() -> None:
    """A drive bag since 2026-10-06 carries the live relay's /vo_twist, OpenVINS's outputs and the
    keeper's seed: ros/vio_replay.sh must leave every one of them out of the play, or the replayed
    EKF and relay read them beside the arm's own (and arm A is not "no VIO")."""
    script = (REPO / "ros/vio_replay.sh").read_text()
    play = script[script.index("ros2 bag play") :].split("sleep", 1)[0].replace("\\\n", " ")
    excluded = set(play.split("--exclude-topics", 1)[1].split())
    live = {"/vo", "/vo_twist", "/odometry/filtered", "/vo/raw", "/ov_msckf/poseimu",
            "/ov_msckf/odomimu", "/ov_msckf/health", "/ov_msckf/points_msckf",
            "/ov_msckf/points_slam", "/vio/seed_twist"}  # fmt: skip
    assert live <= excluded


def test_the_calibration_dance_is_a_scan_the_gaze_arbiter_accepts() -> None:
    """ros/tools/neck_dance.py asks the arbiter (the neck's one owner) for one operator-band scan;
    pepin.gaze parses it as the arbiter would and finds every view within the neck's reach."""
    from pepin.gaze import OPERATOR, GazeSettings, Look, look_from_json
    from pepin.neck import NeckConfig

    spec = importlib.util.spec_from_file_location("neck_dance", REPO / "ros/tools/neck_dance.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = NeckConfig.from_json(REPO / "config/neck.json")
    look = look_from_json(module.request(), GazeSettings(), cfg, lambda frame, xyz: "unused")
    assert isinstance(look, Look), look
    assert look.band == OPERATOR and len(look.views) == len(module.views()) == 16
    pans = [math.degrees(v.pan_rad) for v in look.views]
    tilts = [math.degrees(v.tilt_rad) for v in look.views]
    assert (
        max(pans) == 45.0
        and min(pans) == -45.0
        and min(tilts) == pytest.approx(-18.0)
        and max(tilts) == pytest.approx(60.0)
    )
