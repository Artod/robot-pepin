"""The head's serial frames and payloads (pepin.head_link), as protocol.h has them."""

from __future__ import annotations

import math
import struct
from dataclasses import replace
from typing import Any

import pytest

from pepin.face import eye_set_id
from pepin.head_link import (
    EXPRESSION,
    EYE_RELEASE,
    EYE_STANDING,
    EYE_TIMED,
    IMU,
    INFO_TEXT_MAX,
    PING,
    EyeGazeLink,
    Eyes,
    EyesFrame,
    FrameDecoder,
    HeadClient,
    HeadStatus,
    ImuConfig,
    ImuSample,
    cli_request,
    crc8,
    decode_eyes,
    decode_imu,
    decode_pong,
    decode_status,
    encode_config,
    encode_expression,
    encode_eyes,
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
    assert decode_status(payload).eyes_ma is None  # firmware 3: no eyes


def test_a_status_with_the_eyes_carries_five_bytes_more() -> None:
    """Firmware 4 appends the eyes: current mA u16, brightness u8, mode u8, layout id u8."""
    status = HeadStatus(123, 49.5, 200, 0, 0, 0, 5, 6, "face", "interrupt", 0x72, 1, 4, 150, 0,
                        eyes_ma=310, eyes_brightness=64, eyes_mode="standing",
                        eye_layout=2)  # fmt: skip
    payload = encode_status(status)
    assert len(payload) == 41 and payload[36:] == bytes((0x36, 0x01, 64, 1, 2))
    assert decode_status(payload) == status
    assert decode_status(payload).eye_set is None  # a firmware 4 built before the sets
    with pytest.raises(struct.error, match="36, 41 with the eyes or 42 with their set"):
        decode_status(payload[:39])


def test_a_status_with_the_eye_set_carries_one_byte_more() -> None:
    """A firmware 4 of 2026-10-15 on appends the look set in force (pepin.face.EYE_SETS: 0
    classic, 1 kawaii) to the 41 bytes; an id it does not know is kept as its number."""
    status = HeadStatus(123, 49.5, 200, 0, 0, 0, 5, 6, "face", "interrupt", 0x72, 1, 4, 150, 0,
                        eyes_ma=310, eyes_brightness=64, eyes_mode="standing", eye_layout=2,
                        eye_set="kawaii")  # fmt: skip
    payload = encode_status(status)
    assert len(payload) == 42 and payload[36:] == bytes((0x36, 0x01, 64, 1, 2, 1))
    assert decode_status(payload) == status
    assert decode_status(payload[:41]) == replace(status, eye_set=None)
    assert decode_status(payload[:41] + bytes((0,))).eye_set == "classic"
    assert decode_status(payload[:41] + bytes((7,))).eye_set == "7"
    with pytest.raises(struct.error, match="a status of 43 bytes"):
        decode_status(payload + bytes(1))


def test_an_eyes_frame_is_16_bytes_with_a_sentinel_for_every_field_left_alone() -> None:
    frame = EyesFrame(Eyes(brow_raise=-60, lid_top=120, iris=(255, 136, 0)), EYE_TIMED, 1500)
    payload = encode_eyes(frame)
    assert payload == bytes(
        (2, 0xDC, 0x05, 0x80, 0xC4, 0xFF, 120, 0xFF, 0x80, 0x80, 255, 136, 0, 0xFF, 0, 0)
    )
    assert decode_eyes(payload) == frame
    release = encode_eyes(EyesFrame(mode=EYE_RELEASE))
    assert release == bytes((0, 0, 0, 0x80, 0x80, 0xFF, 0xFF, 0xFF, 0x80, 0x80, 0, 0, 0, 0xFF, 0,
                             0))  # fmt: skip
    wink = encode_eyes(EyesFrame(Eyes(gaze_x=-100, gaze_y=40), blink=3, side=2))
    assert wink[8:10] == bytes((0x9C, 40)) and wink[-2:] == bytes((3, 2))


def test_an_eyes_frame_calling_an_effect_is_19_bytes_and_the_effect_no_look_field() -> None:
    """The 19-byte form appends the effect's id (pepin.face.EYE_EFFECTS) and its life in ms;
    a frame without an effect stays 16 bytes (the firmware keeps the running one)."""
    frame = EyesFrame(Eyes(lid_top=120, effect="spin", effect_ms=2000), EYE_STANDING)
    payload = encode_eyes(frame)
    assert len(payload) == 19 and payload[16:] == bytes((3, 0xD0, 0x07))
    assert decode_eyes(payload) == frame
    assert len(encode_eyes(EyesFrame(Eyes(lid_top=120)))) == 16
    assert encode_eyes(EyesFrame(Eyes(effect="none")))[16:] == bytes((0, 0, 0))  # the look's own
    assert decode_eyes(payload[:16] + bytes((0xFF, 0, 0))).eyes.effect is None  # 0xFF: keep
    assert decode_eyes(payload[:16] + bytes((9, 0, 0))).eyes.effect == "none"  # as the firmware
    tear = encode_eyes(EyesFrame(Eyes(effect="tear")))  # the kawaii sad's drop, id 7
    assert tear[16] == 7 and decode_eyes(tear).eyes.effect == "tear"
    # the effect is no look field: an effect alone overrides nothing, over() leaves it out
    assert Eyes(effect="pulse").empty and Eyes(effect="pulse").as_dict() == {}
    assert Eyes(lid_top=3, effect="pulse").over(Eyes()) == Eyes(lid_top=3)
    with pytest.raises(ValueError, match="no eye effect 'twirl'"):
        Eyes(effect="twirl")


def test_eyes_from_a_door_command_in_the_face_tables_units() -> None:
    """The door speaks config/face.json's units (-1..1, 0..1), the wire -100..100 and 0..254;
    out of range is clamped, an unknown field refused."""
    eyes = Eyes.of(
        {"brow_pos": 3, "lid_top": 1.5, "glow": -4, "gaze_x": 0.126, "lid_bottom": 0.5,
         "iris": "#ff8800"}
    )  # fmt: skip
    assert eyes == Eyes(brow_pos=100, lid_top=254, glow=0, gaze_x=13, lid_bottom=127,
                        iris=(255, 136, 0))  # fmt: skip
    assert eyes.as_dict() == {"brow_pos": 1.0, "lid_top": 1.0, "lid_bottom": 0.5, "gaze_x": 0.13,
                              "glow": 0.0, "iris": [255, 136, 0]}  # fmt: skip
    assert Eyes.of({"iris": [300, 0, 5], "lid_bottom": None}) == Eyes(iris=(255, 0, 5))
    with pytest.raises(ValueError, match="no eye field brow"):
        Eyes.of({"brow": 0.1})
    with pytest.raises(ValueError, match="#RRGGBB"):
        Eyes.of({"iris": "orange"})
    # an override laid over another: its own fields win, the other's show through
    under = Eyes(brow_raise=20, gaze_x=50)
    assert Eyes(brow_raise=-40).over(under) == Eyes(brow_raise=-40, gaze_x=50)
    assert Eyes().empty and not under.empty


def test_pong_expression_mouth_and_config_payloads() -> None:
    assert decode_pong(bytes((1, 0, 0, 0, 0x10, 0x27, 0, 0))) == (1, 10000)
    assert encode_expression(13, 0.6, 280) == bytes((13, 153, 24, 1))
    assert encode_expression(1, 2.0, -5) == bytes((1, 255, 0, 0))  # clamped, not refused
    assert encode_mouth(0.5) == bytes((128,))
    assert encode_mouth(-1) == bytes((0,))
    config = ImuConfig(id=2, rate_hz=500, dlpf=4, accel_fs=2, gyro_fs=3)
    assert encode_config(config, 300) == bytes((2, 0xF4, 0x01, 4, 2, 3, 255, 0))
    assert encode_config(config, 300, style=1)[-1] == 1  # the face's style, config/face.json
    # the ninth byte: the eyes' brightness (0 off); without it the firmware leaves it alone
    assert encode_config(config, 300, 1, eyes_brightness=90, eye_set=None) == bytes(
        (2, 0xF4, 0x01, 4, 2, 3, 255, 1, 90)
    )
    assert encode_config(config, 300, 1, eyes_brightness=999, eye_set=None)[-1] == 255
    # the tenth: the eyes' look set (0 follow the face style, the default; 1 classic, 2 kawaii),
    # only behind the brightness
    assert encode_config(config, 300, 1, eyes_brightness=90) == bytes(
        (2, 0xF4, 0x01, 4, 2, 3, 255, 1, 90, 0)
    )
    assert encode_config(config, 300, 1, 90, eye_set_id("kawaii"))[8:] == bytes((90, 2))
    assert encode_config(config, 300, 1, 90, eye_set_id("classic"))[-1] == 1
    assert len(encode_config(config, 300, 1, None, 2)) == 8
    with pytest.raises(ValueError, match="eye set 3: 0 follow, 1 classic, 2 kawaii"):
        encode_config(config, 300, 1, 90, 3)
    with pytest.raises(ValueError, match="no eye set 'cute': follow, classic, kawaii"):
        eye_set_id("cute")


def test_the_head_client_asks_for_an_eye_set_in_a_config() -> None:
    """The eye set is a config of the head's (standing, not a source's)."""
    sent: list[dict[str, Any]] = []
    client = HeadClient("127.0.0.1", source="llm")
    client.send = sent.append  # type: ignore[method-assign]
    client.eye_set("kawaii")
    assert sent == [{"cmd": "config", "eye_set": "kawaii"}]


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


class Sink:
    def __init__(self) -> None:
        self.sent: list[tuple[float, float, bool]] = []

    def gaze(self, x: float, y: float, *, tracking: bool = False) -> None:
        self.sent.append((x, y, tracking))


def test_the_eye_gaze_link_sends_changes_of_2_percent_at_10_hz_and_refreshes() -> None:
    """At the gaze node's 20 Hz: a head sweeping 1 % a step goes out every other step at most,
    a still head once, and again after 5 s; tracking flipping goes out at the next slot."""
    sink = Sink()
    link = EyeGazeLink(sink)
    t = 0.0
    for i in range(40):  # 2 s of a sweep, 0.01 a step
        link.update(i * 0.01, 0.0, False, t)
        t += 0.05
    assert len(sink.sent) == 20  # 10 Hz, never faster
    gaps = [b[0] - a[0] for a, b in zip(sink.sent, sink.sent[1:], strict=False)]
    assert min(gaps) >= 0.02 - 1e-9
    n = len(sink.sent)
    for _ in range(60):  # 3 s still
        link.update(0.39, 0.0, False, t)
        t += 0.05
    assert len(sink.sent) == n  # 0.39 is within 2 % of the 0.38 sent
    link.update(0.39, 0.0, True, t)
    assert sink.sent[-1] == (0.39, 0.0, True)  # tracking flipped
    n = len(sink.sent)
    for _ in range(101):  # 5 s still
        t += 0.05
        link.update(0.39, 0.0, True, t)
    assert len(sink.sent) == n + 1  # the refresh
    link.update(3.0, -2.0, True, t + 0.2)
    assert sink.sent[-1] == (1.0, -1.0, True)  # clamped


def test_the_command_line_makes_the_door_requests() -> None:
    assert cli_request("eyes", ["brow=-0.6", "lid=0.5", "ms=1500"]) == {
        "cmd": "eyes",
        "source": "cli",
        "brow_raise": -0.6,
        "lid_top": 0.5,
        "ms": 1500.0,
    }
    assert cli_request("eyes", ["side=left", "iris=#ff8800", "glow=0.3"]) == {
        "cmd": "eyes", "source": "cli", "side": "left", "iris": "#ff8800", "glow": 0.3
    }  # fmt: skip
    assert cli_request("eyes", ["release"]) == {"cmd": "eyes_release", "source": "cli"}
    assert cli_request("eyes", ["effect", "spin", "2000"]) == {
        "cmd": "eyes_effect", "source": "cli", "name": "spin", "ms": 2000.0,
    }  # fmt: skip
    assert cli_request("eyes", ["effect", "none"])["name"] is None
    assert cli_request("eyes", ["effect", "pulse"])["ms"] is None
    assert cli_request("eyes", ["brightness", "90"]) == {"cmd": "config", "eyes_brightness": 90}
    assert cli_request("eyes", ["set", "kawaii"]) == {"cmd": "config", "eye_set": "kawaii"}
    assert cli_request("eyes", ["effect", "tear"])["name"] == "tear"
    assert cli_request("blink", []) == {"cmd": "blink", "source": "cli"}
    assert cli_request("wink", ["right"]) == {"cmd": "wink", "source": "cli", "side": "right"}
    assert cli_request("gaze", ["0.4"]) == {"cmd": "gaze", "x": 0.4, "y": 0.0, "tracking": False}
    assert cli_request("style", ["anime"]) == {"cmd": "config", "style": "anime"}
    with pytest.raises(ValueError, match="no eye field 'nose'"):
        cli_request("eyes", ["nose=3"])
    with pytest.raises(ValueError, match="field=value"):
        cli_request("eyes", ["up"])
