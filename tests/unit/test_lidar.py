"""LD19 parsing on a real capture, scan assembly, and mount geometry."""

import math
from pathlib import Path

import numpy as np
import pytest

from pepin.lidar import FrameParser, LaserScan, LidarFrame, LidarMount, ScanAssembler

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "ld19_sample.bin"


def test_real_capture_parses_with_perfect_crc() -> None:
    parser = FrameParser()
    frames = parser.feed(SAMPLE.read_bytes())
    assert parser.frames > 350 and parser.crc_failures == 0
    assert all(len(f.points) == 12 for f in frames)
    assert 3000 < frames[0].speed_dps < 4500  # ~10 rev/s


def test_parser_survives_split_and_junk_input() -> None:
    data = SAMPLE.read_bytes()[: 47 * 3 + 20]
    whole = FrameParser().feed(data)
    parser = FrameParser()
    pieces = [parser.feed(data[i : i + 7]) for i in range(0, len(data), 7)]
    assert [f for chunk in pieces for f in chunk] == whole
    assert FrameParser().feed(b"\x00\x54" + data)[: len(whole)] == whole


def test_mirror_and_yaw_offset_map_to_robot_frame() -> None:
    mount = LidarMount(mirror=True, yaw_offset_deg=90.0)
    # Sensor 270 deg -> mirrored 90 -> minus yaw offset -> robot forward (0 rad).
    assert mount.to_robot_angle_rad(270.0) == pytest.approx(0.0)
    # Sensor 180 -> mirrored 180 -> 90 deg -> robot left (+pi/2).
    assert mount.to_robot_angle_rad(180.0) == pytest.approx(math.pi / 2)


def test_masked_sectors_wrap_around_zero() -> None:
    mount = LidarMount(masked_sectors_deg=((350.0, 10.0), (100.0, 120.0)))
    assert mount.is_masked(5.0) and mount.is_masked(355.0) and mount.is_masked(110.0)
    assert not mount.is_masked(50.0)


def frame(angles: list[float], dist_mm: int = 1000) -> LidarFrame:
    return LidarFrame(3600.0, tuple((a, dist_mm, 200) for a in angles), 0)


def test_assembler_emits_one_scan_per_revolution_and_masks_points() -> None:
    mount = LidarMount(mirror=False, masked_sectors_deg=((100.0, 200.0),))
    asm = ScanAssembler(mount)
    assert asm.feed(frame([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])) is None
    scan = asm.feed(frame([1, 31, 61, 91, 121, 151, 181, 211, 241, 271, 301, 331]))
    assert isinstance(scan, LaserScan)
    assert len(scan.angles) == 12
    assert np.isnan(scan.ranges[[4, 5, 6]]).all()  # 120, 150, 180 fall in the mask
    assert scan.ranges[0] == pytest.approx(1.0)
    assert scan.speed_rps == pytest.approx(10.0)


def test_points_xy_applies_sensor_offset_and_drops_nans() -> None:
    mount = LidarMount(x_m=0.005)
    scan = LaserScan(
        0.0, np.array([0.0, math.pi / 2]), np.array([1.0, math.nan]), np.array([1, 1]), 10.0
    )
    xy = scan.points_xy(mount)
    assert xy.shape == (1, 2)
    assert xy[0] == pytest.approx([1.005, 0.0])


def test_revolution_boundary_inside_a_frame_is_detected() -> None:
    asm = ScanAssembler(LidarMount(mirror=False))
    asm.feed(frame([300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355]))
    scan = asm.feed(frame([356, 357, 358, 359, 0, 1, 2, 3, 4, 5, 6, 7]))
    assert scan is not None and len(scan.angles) == 16  # 12 + the four points before the wrap


def test_real_capture_yields_about_ten_revolutions_per_second() -> None:
    asm = ScanAssembler(LidarMount())
    scans = [s for f in FrameParser().feed(SAMPLE.read_bytes()) if (s := asm.feed(f)) is not None]
    assert 8 <= len(scans) <= 11
    # The first scan is partial by nature (the stream starts mid-revolution).
    assert all(400 < len(s.angles) < 520 for s in scans[1:])


def test_tcp_source_raises_when_the_bridge_closes(monkeypatch) -> None:
    import socket

    from pepin.lidar import TcpSource

    class Closed:
        def settimeout(self, value: float) -> None:
            pass

        def recv(self, n: int) -> bytes:
            return b""

        def close(self) -> None:
            pass

    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: Closed())
    source = TcpSource("host", 1)
    with pytest.raises(ConnectionError):
        source.read(4096)


# -- LidarClient over a replayed capture ----------------------------------------


class ReplaySource:
    """Serves the fixture in chunks, then dies like a closed bridge."""

    def __init__(self) -> None:
        self._data = SAMPLE.read_bytes()
        self._pos = 0
        self.closed = False

    def read(self, max_bytes: int) -> bytes:
        if self._pos >= len(self._data):
            raise ConnectionError("bridge closed")
        chunk = self._data[self._pos : self._pos + max_bytes]
        self._pos += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


def test_lidar_client_drains_revolutions_and_reconnects_after_a_drop() -> None:
    import time

    from pepin.lidar import LidarClient

    sources: list[ReplaySource] = []

    def factory() -> ReplaySource:
        sources.append(ReplaySource())
        return sources[-1]

    client = LidarClient("unused", LidarMount(), source_factory=factory, retry_s=0.01).start()
    scans = []
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and (len(scans) < 5 or client.reconnects < 1):
        scans.extend(client.drain())
        time.sleep(0.01)
    client.close()
    assert len(scans) >= 5
    assert client.reconnects >= 1 and len(sources) >= 2
    assert client.latest is not None and client.age_s(time.monotonic()) < 5.0
    assert sources[0].closed


def test_passive_lidar_client_steps_aside_when_kicked() -> None:
    import time

    from pepin.lidar import LidarClient

    sources: list[ReplaySource] = []

    def factory() -> ReplaySource:
        sources.append(ReplaySource())
        return sources[-1]

    client = LidarClient(
        "unused", LidarMount(), source_factory=factory, retry_s=0.01, reconnect=False
    )
    client.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and client._thread.is_alive():
        time.sleep(0.01)
    assert not client._thread.is_alive(), "a passive client must stop after the bridge closes"
    assert len(sources) == 1 and client.reconnects == 0
    assert client.latest is not None  # what it saw before being kicked is kept


def test_lidar_client_reconnects_when_the_bridge_is_open_but_silent() -> None:
    import time

    from pepin.lidar import LidarClient

    class SilentSource:
        def read(self, max_bytes: int) -> bytes:
            time.sleep(0.005)
            return b""

        def close(self) -> None:
            pass

    opened: list[SilentSource] = []

    def factory() -> SilentSource:
        opened.append(SilentSource())
        return opened[-1]

    client = LidarClient("unused", LidarMount(), source_factory=factory, retry_s=0.01)
    client.silence_s = 0.05
    client.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and client.reconnects < 2:
        time.sleep(0.01)
    client.close()
    assert client.reconnects >= 2 and len(opened) >= 2
