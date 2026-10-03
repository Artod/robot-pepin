"""The head IMU's wire (head_server's TCP 3340, the face agent's schema), the bridge's parsing,
rate counter and publish cap in Python (pepin.head_imu, the twin of head_imu.hpp), its parameters
from config/head_imu.json and config/camera.json, and the fake head_server for the bench."""

from __future__ import annotations

import gzip
import json
import math
import re
import socket
import time
from pathlib import Path

import numpy as np
import pytest

from pepin.head_imu import (
    FakeChip,
    FakeHeadServer,
    HeadDecimator,
    HeadImuConfig,
    HeadRate,
    camera_from_imu,
    parse_config,
    parse_line,
    parse_status,
    read_recording,
    sample_age,
)

REPO = Path(__file__).resolve().parents[2]


def test_an_imu_line_is_si_gyro_first_with_both_clocks_and_its_config_dates_it() -> None:
    """vio.md's 3340 schema, head_server's lines: an imu_config first (rate, filter delay), then
    rows [t_mono_s, esp_us, gx, gy, gz, ax, ay, az] in rad/s and m/s^2. A broken row is refused
    and counted, not guessed; a config that cannot date samples is no config."""
    config = parse_config(
        {"type": "imu_config", "cfg": 2, "rate_hz": 200, "dlpf": 3, "gyro_fs_dps": 500,
         "accel_fs_g": 4, "filter_delay_s": 0.0048}
    )  # fmt: skip
    assert config is not None and (config.cfg, config.rate_hz, config.filter_delay_s) == (
        2,
        200.0,
        0.0048,
    )
    assert parse_config({"type": "imu_config", "cfg": 2, "rate_hz": 0, "filter_delay_s": 0}) is None
    assert parse_config({"type": "imu_config", "cfg": True, "rate_hz": 200}) is None
    batch = parse_line(
        {
            "type": "imu",
            "cfg": 2,
            "samples": [
                [100.000, 5_000_000, 0.01, -0.02, 0.03, 0.1, -0.2, 9.80665],
                [100.005, 5_005_000, 0.01, "x", 0.0, 0.0, 0.0, 9.8],
                [100.010, 5_010_000, 0.0, 0.0, 0.0, 0.0, 9.8],
            ],
        }
    )
    assert batch is not None and batch.cfg == 2 and len(batch.samples) == 1 and batch.refused == 2
    sample = batch.samples[0]
    assert sample.t_mono_s == 100.0 and sample.esp_us == 5_000_000 and sample.cfg == 2
    assert sample.gyro == (0.01, -0.02, 0.03) and sample.accel == (0.1, -0.2, 9.80665)
    assert parse_line({"type": "status"}) is None
    assert parse_line({"type": "imu", "cfg": 2, "s": []}) is None, "the old key is no line"


def test_the_status_line_carries_the_clock_map_the_minute_line_prints() -> None:
    clock = parse_status(
        {"type": "status", "clock": {"ready": True, "spread_ms": 0.21, "esp_fast_ppm": 4.5}}
    )
    assert clock is not None and clock.ready and clock.spread_ms == 0.21
    assert clock.esp_fast_ppm == 4.5 and clock.min_rtt_ms == -1.0
    assert parse_status({"type": "imu"}) is None
    unready = parse_status({"type": "status", "clock": {"ready": False}})
    assert unready is not None and not unready.ready


def test_the_rate_counts_gaps_and_the_cap_keeps_one_in_n() -> None:
    """The same numbers head_imu_contract.cpp checks: a 15 ms gap at 200 Hz, an out-of-order
    sample, 1 kHz under a 200 Hz cap is 200 samples a second, 500 Hz keeps 1 in 3."""
    rate = HeadRate()
    for i in range(200):
        rate.add(i * 0.005, 200.0)
    rate.add(200 * 0.005 + 0.010, 200.0)
    rate.add(200 * 0.005 + 0.005, 200.0)
    assert (rate.samples, rate.gaps, rate.out_of_order) == (202, 1, 1)
    assert rate.longest_gap_s == pytest.approx(0.015)
    capped = HeadDecimator(200.0)
    assert (capped.every(1000.0), capped.every(500.0), capped.every(200.0)) == (5, 3, 1)
    assert sum(capped.due(1000.0) for _ in range(1000)) == 200
    assert HeadDecimator(0.0).every(1000.0) == 1
    assert sample_age(10.0, 9.99, 0.5) == pytest.approx(0.01)
    assert sample_age(10.0, 10.01, 0.5) is None and sample_age(10.0, 9.0, 0.5) is None


def test_the_bridge_parameters_are_the_ones_the_cpp_declares() -> None:
    """config/head_imu.json through HeadImuConfig.bridge_parameters is what robot.launch.py hands
    the bridge: every key must be a parameter base_bridge.cpp declares, or rclcpp refuses the
    launch; the defaults in the C++ are the file's."""
    config = HeadImuConfig.load(REPO / "config/head_imu.json")
    params = config.bridge_parameters([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], enable=True)
    cpp = (REPO / "ros/pepin_base_cpp/src/base_bridge.cpp").read_text()
    declared = set(re.findall(r'declare_parameter<[^(]+>\(\s*"(\w+)"', cpp))
    assert set(params) <= declared, set(params) - declared
    assert params["head_imu_enable"] is True and params["head_imu_publish_hz"] == 200.0
    for name, value in params.items():
        if isinstance(value, float):
            match = re.search(rf'declare_parameter<double>\("{name}", ([-0-9.e]+)\)', cpp)
            assert match and float(match.group(1)) == pytest.approx(value), name
    assert config.noise["gyro_noise_density"] == 1e-3 and config.rate_hz == 200.0
    off = config.bridge_parameters(None, enable=False)
    assert off["head_imu_camera_rotation"] == [] and off["head_imu_enable"] is False


def _camera_config_with_imu(tmp_path: Path, t_cam_imu: list[list[float]] | None) -> Path:
    data = json.loads((REPO / "config/camera.json").read_text())
    if t_cam_imu is not None:
        data["stereo"]["head_imu"] = {"T_cam_imu": t_cam_imu, "time_offset_s": 0.004}
    (tmp_path / "camera.json").write_text(json.dumps(data))
    return tmp_path


def test_the_imu_s_rotation_into_camera_link_goes_through_the_optical_frame(
    tmp_path: Path,
) -> None:
    """Kalibr's T_cam_imu is relative to the rectified left eye's OPTICAL frame; the mast filter
    needs R(camera_link <- head_imu) = R(link <- optical) R(optical <- imu). An IMU whose axes
    ARE the optical axes rotates by the optical frame alone (z forward is camera_link's x)."""
    from pepin.mounts import load_camera_mounts

    assert camera_from_imu(_camera_config_with_imu(tmp_path, None)) is None
    identity = [[1.0, 0, 0, 0.01], [0, 1.0, 0, 0.02], [0, 0, 1.0, -0.03], [0, 0, 0, 1.0]]
    config_dir = _camera_config_with_imu(tmp_path, identity)
    rotation = camera_from_imu(config_dir)
    assert rotation is not None
    optical = load_camera_mounts(config_dir).optical.rotation()
    assert np.allclose(np.array(rotation).reshape(3, 3), optical, atol=1e-9)
    z_forward = np.array(rotation).reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
    assert z_forward[0] > 0.99, "the optical z looks along camera_link's x (the eye's few deg)"
    mounts = load_camera_mounts(config_dir)
    assert mounts.imu is not None and mounts.imu_frame == "head_imu"
    assert mounts.imu.transform()[:3] == pytest.approx((0.01, 0.02, -0.03))


def test_a_head_imu_block_that_is_not_a_rigid_transform_is_refused(tmp_path: Path) -> None:
    from pepin.camera import head_imu_transform

    assert head_imu_transform(None) == ()
    with pytest.raises(ValueError, match="4x4"):
        head_imu_transform({"T_cam_imu": [[1, 0, 0]]})
    with pytest.raises(ValueError, match="orthonormal"):
        head_imu_transform({"T_cam_imu": [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})
    with pytest.raises(ValueError, match="det"):
        head_imu_transform({"T_cam_imu": [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})
    with pytest.raises(ValueError, match="last row"):
        head_imu_transform({"T_cam_imu": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]})


@pytest.mark.slow
def test_the_fake_head_server_streams_to_a_subscriber_on_the_board_clock(tmp_path: Path) -> None:
    """The bench stand-in serves head_server's schema: nothing before the subscribe line, then a
    line every ~20 ms whose samples are dated on this machine's monotonic clock, and a status."""
    server = FakeHeadServer(port=0, chip=FakeChip(rate_hz=200.0, ring_deg=0.2)).start()
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=2.0) as conn:
            conn.settimeout(2.0)
            conn.sendall(b'{"cmd":"subscribe","imu":true}\n')
            received = b""
            end = time.monotonic() + 1.3
            while time.monotonic() < end:
                received += conn.recv(65536)
        lines = [json.loads(x) for x in received.decode().splitlines() if x.strip()]
        assert lines[0]["type"] == "imu_config", "the config before the first sample"
        config = parse_config(lines[0])
        assert config is not None and config.rate_hz == 200.0 and config.filter_delay_s == 0.0048
        batches = [parse_line(m) for m in lines if m.get("type") == "imu"]
        samples = [s for b in batches if b is not None for s in b.samples]
        assert len(samples) > 150, len(samples)
        assert all(abs(time.monotonic() - s.t_mono_s) < 2.0 for s in samples)
        assert any(m.get("type") == "status" for m in lines), "a status line within the second"
        gz = tmp_path / "rec.jsonl.gz"
        with gzip.open(gz, "wt") as out:
            for line in received.decode().splitlines():
                out.write(line + "\n")
        assert len(list(read_recording(gz))) == len(samples)
        assert max(abs(s.gyro[1]) for s in samples) > math.radians(3.0), "the ring is in gyro y"
    finally:
        server.stop()
