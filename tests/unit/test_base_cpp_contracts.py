"""The C++ base bridge's own contracts, compiled and run: the gyro's bias tracker and the
zero-velocity update (ros/pepin_base_cpp/test/*_contract.cpp).

The package has no ament test target and the board image is not built on a laptop, so each
contract is one stand-alone main() over its header, no ROS and no gtest. This test compiles each
with the host's c++ and holds its verdict; without a compiler it is skipped, never passed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

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
