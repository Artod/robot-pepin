"""The XVF3800's control answers decoded, with a scripted transport instead of USB."""

import struct

import pytest

from pepin.xvf3800 import (
    COMMANDS,
    READ_FLAG,
    STATUS_BUSY,
    Xvf3800,
    XvfBusyError,
    XvfError,
    firmware_warning,
    parse_doa,
    parse_response,
)


class ScriptedTransport:
    """Answers each read with the next scripted byte string and records what was asked."""

    def __init__(self, *answers: bytes) -> None:
        self.answers = list(answers)
        self.asked: list[tuple[int, int, int]] = []
        self.closed = False

    def read(self, value: int, index: int, length: int) -> bytes:
        self.asked.append((value, index, length))
        return self.answers.pop(0)

    def close(self) -> None:
        self.closed = True


def doa_answer(deg: int, speech: int, status: int = 0) -> bytes:
    return bytes([status]) + struct.pack("<HH", deg, speech)


def test_doa_value_is_two_little_endian_uint16_after_the_status_byte() -> None:
    # 300 does not fit in a byte: Seeed's respeaker_get_doa.py prints only the low byte (44).
    assert parse_response(COMMANDS["DOA_VALUE"], doa_answer(300, 1)) == (300, 1)
    assert parse_doa((300, 1)) == (300, True)
    assert COMMANDS["DOA_VALUE"].length == 5


def test_the_doa_read_asks_resource_20_command_18_with_the_read_flag() -> None:
    transport = ScriptedTransport(doa_answer(45, 0))
    assert Xvf3800(transport).doa() == (45, False)
    assert transport.asked == [(READ_FLAG | 18, 20, 5)]


def test_busy_answers_are_asked_again_and_then_given_up_on() -> None:
    busy = bytes([STATUS_BUSY, 0, 0, 0, 0])
    sleeps: list[float] = []
    transport = ScriptedTransport(busy, busy, doa_answer(90, 1))
    assert Xvf3800(transport, sleep=sleeps.append).doa() == (90, True)
    assert len(sleeps) == 2
    stuck = ScriptedTransport(*[busy] * 3)
    with pytest.raises(XvfError, match="still busy"):
        Xvf3800(stuck, tries=3, sleep=lambda _: None).doa()


def test_an_error_status_a_short_answer_or_an_impossible_angle_is_an_error() -> None:
    with pytest.raises(XvfBusyError):
        parse_response(COMMANDS["DOA_VALUE"], doa_answer(0, 0, status=STATUS_BUSY))
    with pytest.raises(XvfError, match="status 3"):
        parse_response(COMMANDS["DOA_VALUE"], doa_answer(0, 0, status=3))
    with pytest.raises(XvfError, match="expected 5"):
        parse_response(COMMANDS["DOA_VALUE"], doa_answer(0, 0)[:3])
    with pytest.raises(XvfError, match="empty"):
        parse_response(COMMANDS["DOA_VALUE"], b"")
    with pytest.raises(XvfError, match="implausible"):
        parse_doa((360, 0))
    with pytest.raises(XvfError, match="implausible"):
        parse_doa((10, 2))


def test_version_and_float_parameters_decode() -> None:
    transport = ScriptedTransport(
        bytes([0, 2, 1, 1]), bytes([0]) + struct.pack("<4f", 0.5, 0.0, 1.5708, 0.5)
    )
    array = Xvf3800(transport)
    assert array.version() == (2, 1, 1)
    beams = array.read("AEC_AZIMUTH_VALUES")
    assert beams[2] == pytest.approx(1.5708, abs=1e-4) and len(beams) == 4
    assert transport.asked[1] == (READ_FLAG | 75, 33, 17)
    array.close()
    assert transport.closed


def test_old_firmware_is_named_for_what_it_lacks() -> None:
    assert "DOA_VALUE" in str(firmware_warning((2, 0, 5)))
    assert "LED_EFFECT" in str(firmware_warning((2, 0, 9)))
    assert firmware_warning((2, 0, 10)) is None and firmware_warning((2, 1, 1)) is None
