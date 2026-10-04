"""The head's serial frames and payloads (pepin.head_link), as protocol.h has them."""

from __future__ import annotations

import math

import pytest

from pepin.head_link import (
    EXPRESSION,
    IMU,
    INFO_TEXT_MAX,
    PING,
    FrameDecoder,
    HeadStatus,
    ImuConfig,
    ImuSample,
    crc8,
    decode_imu,
    decode_pong,
    decode_status,
    encode_config,
    encode_expression,
    encode_frame,
    encode_imu,
    encode_info,
    encode_mouth,
    encode_ping,
    encode_status,
    show_items,
)


def test_crc8_is_the_smbus_crc() -> None:
    """Poly 0x07, init 0: CRC-8/SMBUS, whose check value for "123456789" is 0xF4."""
    assert crc8(b"123456789") == 0xF4
    assert crc8(b"") == 0


def test_a_frame_is_sync_type_length_payload_crc() -> None:
    frame = encode_frame(PING, encode_ping(0x01020304))
    assert frame[:4] == bytes((0xA5, ord("Q"), 4, 0))
    assert frame[4:8] == bytes((4, 3, 2, 1))
    assert frame[-1] == crc8(frame[1:-1])
    with pytest.raises(ValueError, match="at most 1024"):
        encode_frame(IMU, bytes(1025))


def test_the_decoder_survives_noise_damage_and_split_reads() -> None:
    good = encode_frame(EXPRESSION, encode_expression(3, 1.0, 280))
    damaged = bytearray(encode_frame(EXPRESSION, encode_expression(5, 1.0, 280)))
    damaged[5] ^= 0xFF
    stream = b"ets Jun  8 2016 boot\r\n" + bytes(damaged) + good + b"\xa5" + good
    decoder = FrameDecoder()
    frames = []
    for i in range(0, len(stream), 7):  # the port hands over whatever arrived
        frames += decoder.feed(stream[i : i + 7])
    assert frames == [(EXPRESSION, good[4:-1]), (EXPRESSION, good[4:-1])]
    assert decoder.crc_errors >= 1
    assert decoder.skipped_bytes > 0


def test_an_impossible_length_costs_only_that_sync_byte() -> None:
    good = encode_frame(PING, encode_ping(9))
    decoder = FrameDecoder()
    assert decoder.feed(bytes((0xA5, ord("I"), 0xFF, 0xFF)) + good) == [(PING, good[4:-1])]
    assert decoder.length_errors == 1


def test_imu_samples_round_trip_17_bytes_each() -> None:
    samples = [
        ImuSample(4_294_967_000, (1, -2, 16384), (-32768, 32767, 0), 7),
        ImuSample(5, (0, 0, 0), (1, 1, 1), 7),
    ]
    payload = encode_imu(samples)
    assert len(payload) == 34
    assert decode_imu(payload) == samples
    with pytest.raises(ValueError, match="not n x 17"):
        decode_imu(payload[:-1])


def test_status_round_trips_36_bytes() -> None:
    status = HeadStatus(123, 49.5, 1000, 2, 3, 4, 5, 6, "asleep", "polled", 0x72, 9, 1, 150, 11)
    payload = encode_status(status)
    assert len(payload) == 36
    assert decode_status(payload) == status


def test_pong_expression_mouth_and_config_payloads() -> None:
    assert decode_pong(bytes((1, 0, 0, 0, 0x10, 0x27, 0, 0))) == (1, 10000)
    assert encode_expression(13, 0.6, 280) == bytes((13, 153, 24, 1))
    assert encode_expression(1, 2.0, -5) == bytes((1, 255, 0, 0))  # clamped, not refused
    assert encode_mouth(0.5) == bytes((128,))
    assert encode_mouth(-1) == bytes((0,))
    config = ImuConfig(id=2, rate_hz=500, dlpf=4, accel_fs=2, gyro_fs=3)
    assert encode_config(config, 300) == bytes((2, 0xF4, 0x01, 4, 2, 3, 255))


def test_imu_config_scales_delay_and_refusals() -> None:
    config = ImuConfig(id=1, accel_fs=1, gyro_fs=1, dlpf=3)
    assert config.acc_scale == pytest.approx(9.80665 / 8192)
    assert config.gyro_scale == pytest.approx(math.radians(1 / 65.5))
    assert config.delay_s == 0.0048
    for bad in ({"rate_hz": 3}, {"rate_hz": 300}, {"dlpf": 0}, {"gyro_fs": 4}, {"id": 256}):
        with pytest.raises(ValueError):
            ImuConfig(**bad)  # type: ignore[arg-type]


def test_an_info_screen_payload_and_utf8_cut_between_characters() -> None:
    payload = encode_info(
        [{"text": "Сервы"}, {"key": "left", "value": "41 C"}, {"bar": "pan", "frac": 0.5}], 8.0
    )
    assert payload[:3] == bytes((0x40, 0x1F, 3))  # 8000 ms, three items
    title = "Сервы".encode()
    assert payload[3:5] == bytes((0, len(title))) and payload[5 : 5 + len(title)] == title
    long = encode_info([{"text": "ж" * 40}], 1.0)
    assert long[4] == INFO_TEXT_MAX - 1  # 23 two-byte letters: 46 bytes, not half a 24th
    assert encode_info([{"text": str(i)} for i in range(12)], 1.0)[2] == 8


def test_show_markup_makes_bars_rows_and_lines() -> None:
    text = "Servo temperatures\nleft: 41/70 C\nright: 39/70 C\nbattery: 72%\nmode: idle"
    items = show_items(text)
    assert items == [
        {"text": "Servo temperatures"},
        {"bar": "left", "frac": pytest.approx(41 / 70), "value": "41/70 C"},
        {"bar": "right", "frac": pytest.approx(39 / 70), "value": "39/70 C"},
        {"bar": "battery", "frac": pytest.approx(0.72), "value": "72%"},
        {"key": "mode", "value": "idle"},
    ]
    assert show_items("a | b: 1 C") == [{"text": "a"}, {"key": "b", "value": "1 C"}]
