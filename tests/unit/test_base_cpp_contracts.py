"""The C++ base bridge's own contracts, compiled and run: the gyro's bias tracker, the
zero-velocity update and the neck's model (ros/pepin_base_cpp/test/*_contract.cpp).

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
CONTRACTS = ("gyro_bias_contract", "zupt_contract")


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
