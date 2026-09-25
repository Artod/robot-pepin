from collections.abc import Callable, Iterator

import pytest

from pepin.kinematics import Twist
from pepin.teleop import (
    FAST_ANGULAR_RAD_S,
    FAST_LINEAR_M_S,
    SLOW_ANGULAR_RAD_S,
    SLOW_LINEAR_M_S,
    DriveState,
    apply_key,
    read_key,
)

UP, DOWN, RIGHT, LEFT = "\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D"
S_UP, S_DOWN, S_RIGHT, S_LEFT = "\x1b[1;2A", "\x1b[1;2B", "\x1b[1;2C", "\x1b[1;2D"


def chars(text: str) -> Callable[[float], str | None]:
    """A fake terminal: hands out ``text`` one character per call, then None."""
    it: Iterator[str] = iter(text)
    return lambda _timeout_s: next(it, None)


@pytest.mark.parametrize(
    ("key", "twist"),
    [
        (UP, Twist(FAST_LINEAR_M_S, 0.0)),
        (DOWN, Twist(-FAST_LINEAR_M_S, 0.0)),
        (LEFT, Twist(0.0, FAST_ANGULAR_RAD_S)),
        (RIGHT, Twist(0.0, -FAST_ANGULAR_RAD_S)),
        (S_UP, Twist(SLOW_LINEAR_M_S, 0.0)),
        (S_DOWN, Twist(-SLOW_LINEAR_M_S, 0.0)),
        (S_LEFT, Twist(0.0, SLOW_ANGULAR_RAD_S)),
        (S_RIGHT, Twist(0.0, -SLOW_ANGULAR_RAD_S)),
    ],
)
def test_each_arrow_latches_its_twist(key: str, twist: Twist) -> None:
    assert apply_key(DriveState(), key).twist == twist


def test_speeds_are_the_max_and_the_parking_min() -> None:
    assert apply_key(DriveState(), UP).twist == Twist(0.15, 0.0)
    assert apply_key(DriveState(), LEFT).twist == Twist(0.0, 0.5)
    assert apply_key(DriveState(), S_UP).twist == Twist(0.04, 0.0)
    assert apply_key(DriveState(), S_LEFT).twist == Twist(0.0, 0.15)


def test_a_key_replaces_the_latched_command_and_repeats_do_not_accumulate() -> None:
    s = apply_key(apply_key(DriveState(), UP), UP)
    assert s.twist == Twist(FAST_LINEAR_M_S, 0.0)
    assert apply_key(s, LEFT).twist == Twist(0.0, FAST_ANGULAR_RAD_S)


def test_space_stops() -> None:
    assert apply_key(apply_key(DriveState(), UP), " ").twist == Twist(0.0, 0.0)


@pytest.mark.parametrize("key", ["x", "q", "w", "ц", "k", "\x1b", "\x1b[1;5A", "\n"])
def test_other_keys_change_nothing(key: str) -> None:
    s = apply_key(DriveState(), UP)
    assert apply_key(s, key) == s


@pytest.mark.parametrize("seq", [UP, DOWN, LEFT, RIGHT, S_UP, S_DOWN, S_LEFT, S_RIGHT])
def test_read_key_assembles_whole_arrow_sequences(seq: str) -> None:
    assert read_key(chars(seq)) == seq


def test_read_key_splits_consecutive_keys() -> None:
    source = chars(S_UP + " " + LEFT + "x")
    assert [read_key(source) for _ in range(5)] == [S_UP, " ", LEFT, "x", None]


def test_read_key_plain_lone_escape_and_nothing_pending() -> None:
    assert read_key(chars(" ")) == " "
    assert read_key(chars("\x1b")) == "\x1b"
    assert read_key(chars("")) is None


def test_read_key_bounds_a_runaway_sequence() -> None:
    assert read_key(chars("\x1b[" + "1" * 50)) == "\x1b[111111"


def test_key_reader_without_a_terminal_reads_nothing_and_raises_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import sys

    from pepin.teleop import KeyReader

    with open(os.devnull) as fake_stdin:
        monkeypatch.setattr(sys, "stdin", fake_stdin)
        with KeyReader() as keys:
            assert keys.read() is None
