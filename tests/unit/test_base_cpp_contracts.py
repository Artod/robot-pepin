"""The C++ base bridge's own contracts, compiled and run: the gyro's bias tracker, the IMU's probe
schedule, the zero-velocity update and the neck's model (ros/pepin_base_cpp/test/*_contract.cpp).

The package has no ament test target and the board image is not built on a laptop, so each
contract is one stand-alone main() over its header, no ROS and no gtest. This test compiles each
with the host's c++ and holds its verdict; without a compiler it is skipped, never passed.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from pepin.camera import quaternion_from_rpy
from pepin.neck import NeckConfig, NeckPivot, bridge_parameters, camera_pose, joint_angles

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "ros/pepin_base_cpp"
CONTRACTS = (
    "gyro_bias_contract",
    "zupt_contract",
    "head_imu_contract",
    "mast_contract",
    "imu_probe_contract",
)


@pytest.mark.slow
@pytest.mark.parametrize("contract", CONTRACTS)
def test_the_contract_holds_against_its_header(contract: str, tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("no c++ on this host")
    binary = tmp_path / contract
    built = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-O2",
            "-I",
            str(PACKAGE / "include"),
            str(PACKAGE / "test" / f"{contract}.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "the contract holds" in ran.stdout


def _stdin_model(cfg: NeckConfig) -> list[float]:
    """The contract's first line, taken from the parameters robot.launch.py hands the bridge
    (pepin.neck.bridge_parameters), so the launch's mapping is held too."""
    p = bridge_parameters(cfg)
    return [
        p["neck_reference_pan_ticks"],
        p["neck_reference_tilt_ticks"],
        p["neck_pan_sign"],
        p["neck_tilt_sign"],
        p["neck_mount_x_m"],
        p["neck_mount_y_m"],
        p["neck_mount_z_m"],
        math.radians(p["neck_mount_pitch_deg"]),  # the bridge converts, as here
        p["neck_tilt_from_pan_x_m"],
        p["neck_tilt_from_pan_z_m"],
        p["neck_camera_from_tilt_x_m"],
        p["neck_camera_from_tilt_z_m"],
    ]


def _neck_models() -> list[NeckConfig]:
    """The repo's config/neck.json, the same with its reference unread, and one with every
    lever arm and both signs exercised."""
    cfg = NeckConfig.from_json(REPO / "config/neck.json")
    unread = replace(cfg, reference=replace(cfg.reference, pan_ticks=None, tilt_ticks=None))
    levers = replace(
        cfg,
        reference=replace(cfg.reference, pan_sign=1, tilt_sign=-1, x_m=0.04, y_m=-0.01),
        pivot=NeckPivot(0.03, 0.05, 0.025, 0.086),
    )
    return [cfg, unread, levers]


@pytest.mark.slow
@pytest.mark.parametrize("model", range(3))
def test_the_neck_model_in_c_plus_plus_answers_what_pepin_neck_answers(
    model: int, tmp_path: Path
) -> None:
    """The base bridge publishes /neck/state and base_link -> camera_link from neck.hpp; the
    laptop's tools and the old node answered from pepin.neck. One geometry, two languages: every
    reading from the limits through the encoder's wrap gets the same angles and the same pose."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("no c++ on this host")
    binary = tmp_path / "neck_contract"
    built = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-O2",
            "-I",
            str(PACKAGE / "include"),
            str(PACKAGE / "test" / "neck_contract.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    cfg = _neck_models()[model]
    readings = [
        (pan, tilt)
        for pan in (0, 257, 1993, 2029, 2048, 3812, 4095)
        for tilt in (0, 1814, 2311, 2760, 4095)
    ]
    ran = subprocess.run(
        [str(binary)],
        input=" ".join(str(v) for v in _stdin_model(cfg))
        + "\n"
        + "\n".join(f"{p} {t}" for p, t in readings)
        + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr
    *rows, verdict = ran.stdout.strip().splitlines()
    assert verdict == "the contract holds" and len(rows) == len(readings)
    for (pan, tilt), row in zip(readings, rows, strict=True):
        angles = joint_angles(cfg, pan, tilt)
        x, y, z, roll, pitch, yaw = camera_pose(cfg, angles)
        expected = (
            angles.pan_rad,
            angles.pitch_rad,
            x,
            y,
            z,
            *quaternion_from_rpy(roll, pitch, yaw),
        )
        assert [float(v) for v in row.split()] == pytest.approx(expected, abs=1e-9), (pan, tilt)


def _compile(contract: str, tmp_path: Path) -> Path:
    """One contract compiled with the host's c++ (the test is skipped without one)."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("no c++ on this host")
    binary = tmp_path / contract
    built = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-O2",
            "-I",
            str(PACKAGE / "include"),
            str(PACKAGE / "test" / f"{contract}.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    return binary


@pytest.mark.slow
def test_the_head_imu_in_c_plus_plus_parses_what_pepin_head_imu_parses(tmp_path: Path) -> None:
    """The bridge publishes /head/imu from head_imu.hpp; pepin.head_imu is the same wire in
    Python (the fake head_server, the recorder, the tests). The same rows — a NaN, a gap, an
    out-of-order sample, a 1 kHz chip under the 200 Hz cap — give the same samples and verdicts."""
    from pepin.head_imu import HeadConfig, HeadDecimator, HeadRate, sample_from_row

    binary = _compile("head_imu_contract", tmp_path)
    config = HeadConfig(3, 1000.0, 0.0048)
    rows: list[list[float]] = []
    for i in range(30):
        t = 200.0 + i * 0.001 + (0.004 if i >= 20 else 0.0)
        rows.append([t, 1e6 + i * 1000, 0.001 * i, -0.002 * i, 0.003, 0.01 * i, -0.1, 9.8])
    rows[7][5] = math.nan
    rows[25][0] = rows[24][0] - 0.0005  # out of order
    text = f"config {config.cfg} {config.rate_hz} {config.filter_delay_s}\n"
    text += "\n".join("row " + " ".join(repr(float(v)) for v in row) for row in rows) + "\n"
    ran = subprocess.run(
        [str(binary), "--stdin"], input=text, capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stderr
    lines = ran.stdout.strip().splitlines()
    rate, cap = HeadRate(), HeadDecimator(200.0)
    expected: list[str] = []
    for row in rows:
        sample = sample_from_row(row, config.cfg)
        if sample is None:
            expected.append("refused")
            continue
        rate.add(sample.t_mono_s, config.rate_hz)
        due = cap.due(config.rate_hz)
        values = " ".join(f"{v:.9f}" for v in (*sample.gyro, *sample.accel))
        expected.append(f"sample {sample.t_mono_s:.9f} {sample.esp_us:.1f} {values} {int(due)}")
    expected.append(f"rate {rate.samples} {rate.gaps} {rate.out_of_order} {rate.longest_gap_s:.9f}")
    assert lines == expected


@pytest.mark.slow
def test_the_mast_filter_in_c_plus_plus_answers_what_pepin_mast_answers(tmp_path: Path) -> None:
    """The bridge filters the sway with mast.hpp; pepin.mast is its reference. A ring, a base
    turn, a hold, a NaN and a gap through both give the same theta and omega to 1e-9."""
    from pepin.mast import MastFilter, ring_rate

    binary = _compile("mast_contract", tmp_path)
    script: list[str] = []
    for i in range(300):
        t = 10.0 + i / 200.0
        rate = ring_rate(i / 200.0, math.radians(0.2), 5.3, 1.1)
        script.append(f"{t!r} {0.001 * i!r} {rate!r} {0.3 + 0.01 * math.sin(i)!r} 0.3")
        if i == 150:
            script.append("hold")
        if i == 200:
            script.append(f"{t + 0.001!r} nan 0.0 0.0 0.0")
    script.append(f"{12.0!r} 0.0 0.1 0.0 0.0")  # after a gap: re-armed
    ran = subprocess.run(
        [str(binary), "--stdin"],
        input="\n".join(script) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    mast = MastFilter()
    lines = ran.stdout.strip().splitlines()
    assert len(lines) == len(script)
    for command, line in zip(script, lines, strict=True):
        if command == "hold":
            mast.hold()
            assert line == "held"
            continue
        t, hx, hy, hz, yaw = (float(v) for v in command.split())
        out = mast.update(t, (hx, hy, hz), yaw)
        held, *values = line.split()
        assert int(held) == int(out.held), command
        got = [float(v) for v in values]
        want = [*out.theta, *out.omega]
        for g, w in zip(got, want, strict=True):
            assert (math.isnan(g) and math.isnan(w)) or g == pytest.approx(w, abs=1e-9), command
