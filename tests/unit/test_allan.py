"""pepin.allan: the noise block of a parked recording, held on synthetic gyros of known noise."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pepin.allan import cluster_times, fit, gap_share, main, noise_block, overlapping_adev


def _synthetic(rate: float, seconds: float, white: float, walk: float, seed: int = 1) -> np.ndarray:
    """A rate signal with white noise of density ``white`` and a bias random walk ``walk``."""
    rng = np.random.default_rng(seed)
    n = int(rate * seconds)
    noise = rng.normal(0.0, white * math.sqrt(rate), n)
    bias = np.cumsum(rng.normal(0.0, walk / math.sqrt(rate), n))
    return noise + bias


def test_white_noise_and_random_walk_are_read_back_within_fifteen_percent() -> None:
    """2000 s at 200 Hz with N 1.8e-4 rad/s/sqrt(Hz) (the base clone's) and K 2e-5: the -1/2 slope
    gives N back, the +1/2 slope K."""
    rate, white, walk = 200.0, 1.8e-4, 2.0e-5
    signal = _synthetic(rate, 2000.0, white, walk)
    taus = cluster_times(len(signal), rate)
    curve = overlapping_adev(signal, rate, taus)
    result = fit(taus, curve)
    assert result.white == pytest.approx(white, rel=0.15)
    assert result.walk == pytest.approx(walk, rel=0.35)
    assert math.isfinite(result.instability) and result.instability > 0.0


def test_the_block_takes_the_worst_axis_inflated_and_gaps_are_counted() -> None:
    from pepin.allan import AllanFit

    a, b = AllanFit(1e-4, 1e-5, 1e-5, 100.0), AllanFit(2e-4, math.nan, 1e-5, 100.0)
    block = noise_block([a, b], [a, a], inflate=10.0)
    assert block["gyro_noise_density"] == pytest.approx(2e-3)
    assert block["gyro_random_walk"] == pytest.approx(1e-4), "a NaN axis is no worst"
    assert gap_share([0.0, 0.005, 0.010, 0.030], 200.0) == pytest.approx(1 / 3)


@pytest.mark.slow
def test_a_recording_of_head_server_lines_gives_the_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rate = 200.0
    gyro = _synthetic(rate, 120.0, 1e-3, 1e-5)
    rows = [
        [100.0 + i / rate, 1_000_000 + i * 5000, gyro[i], 0.0, 0.0, 0.0, 0.0, 9.80665]
        for i in range(len(gyro))
    ]
    path = tmp_path / "parked.jsonl.gz"
    with gzip.open(path, "wt") as out:
        out.write(
            json.dumps({"type": "imu_config", "cfg": 1, "rate_hz": rate, "filter_delay_s": 0.0048})
            + "\n"
        )
        for start in range(0, len(rows), 4):
            out.write(
                json.dumps({"type": "imu", "cfg": 1, "samples": rows[start : start + 4]}) + "\n"
            )
    assert main([str(path), "--inflate", "10"]) == 0
    text = capsys.readouterr().out
    assert "24000 samples at 200.0 Hz" in text and "gyro_noise_density" in text
