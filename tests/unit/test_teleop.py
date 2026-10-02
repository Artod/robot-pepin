from collections.abc import Callable, Iterator

import pytest

from pepin.kinematics import STOP, Twist
from pepin.teleop import (
    FAST_ANGULAR_RAD_S,
    FAST_LINEAR_M_S,
    SLOW_ANGULAR_RAD_S,
    SLOW_LINEAR_M_S,
    CommandStream,
    DriveState,
    GameCommand,
    HeldKeys,
    apply_key,
    game_command,
    read_key,
    status_lines,
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
    assert apply_key(DriveState(), UP).twist == Twist(0.45, 0.0)
    assert apply_key(DriveState(), LEFT).twist == Twist(0.0, 1.0)
    assert apply_key(DriveState(), S_UP).twist == Twist(0.064, 0.0)
    assert apply_key(DriveState(), S_LEFT).twist == Twist(0.0, 0.24)


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


# -- game mode: keys act while held ------------------------------------------------------------


class FakeLink:
    """A base link that records what the stream sends, in order."""

    def __init__(self) -> None:
        self.sent: list[tuple[object, ...]] = []

    def set_twist(self, twist: Twist) -> None:
        self.sent.append(("twist", twist))

    def stop(self) -> None:
        self.sent.append(("stop",))

    def neck_jog(self, pan: int, tilt: int, *, slow: bool = False) -> None:
        self.sent.append(("jog", pan, tilt, slow))


@pytest.mark.parametrize(
    ("keys", "twist"),
    [
        (HeldKeys(up=True), Twist(FAST_LINEAR_M_S, 0.0)),
        (HeldKeys(down=True), Twist(-FAST_LINEAR_M_S, 0.0)),
        (HeldKeys(left=True), Twist(0.0, FAST_ANGULAR_RAD_S)),
        (HeldKeys(right=True), Twist(0.0, -FAST_ANGULAR_RAD_S)),
        (HeldKeys(up=True, shift=True), Twist(SLOW_LINEAR_M_S, 0.0)),
        (HeldKeys(down=True, shift=True), Twist(-SLOW_LINEAR_M_S, 0.0)),
        (HeldKeys(left=True, shift=True), Twist(0.0, SLOW_ANGULAR_RAD_S)),
        (HeldKeys(right=True, shift=True), Twist(0.0, -SLOW_ANGULAR_RAD_S)),
        (HeldKeys(up=True, left=True), Twist(FAST_LINEAR_M_S, FAST_ANGULAR_RAD_S)),
        (HeldKeys(up=True, down=True), STOP),
        (HeldKeys(left=True, right=True, up=True), Twist(FAST_LINEAR_M_S, 0.0)),
        (HeldKeys(), STOP),
    ],
)
def test_held_arrows_give_the_terminal_speeds_and_opposite_keys_cancel(
    keys: HeldKeys, twist: Twist
) -> None:
    command = game_command(keys)
    assert command.twist == twist and command.slow is keys.shift and not command.stop_all


def test_the_game_arrows_agree_with_the_latching_keys() -> None:
    """One mapping, two readers: the held Up is the latched Up, Shift included."""
    assert game_command(HeldKeys(up=True)).twist == apply_key(DriveState(), UP).twist
    assert game_command(HeldKeys(left=True)).twist == apply_key(DriveState(), LEFT).twist
    assert game_command(HeldKeys(up=True, shift=True)).twist == apply_key(DriveState(), S_UP).twist
    assert game_command(HeldKeys(up=True)).twist == Twist(0.45, 0.0), "config/base.json's max"


@pytest.mark.parametrize(
    ("keys", "pan", "tilt"),
    [
        (HeldKeys(pan_left=True), 1, 0),
        (HeldKeys(pan_right=True), -1, 0),
        (HeldKeys(tilt_down=True), 0, 1),
        (HeldKeys(tilt_up=True), 0, -1),
        (HeldKeys(pan_left=True, tilt_up=True), 1, -1),
        (HeldKeys(pan_left=True, pan_right=True, tilt_down=True), 0, 1),
        (HeldKeys(), 0, 0),
    ],
)
def test_wasd_jogs_the_head_in_the_neck_models_signs(keys: HeldKeys, pan: int, tilt: int) -> None:
    """A = pan +1 (left), D = -1; S = tilt +1 (down), W = -1: pepin.neck.NeckAngles' signs, so the
    board applies config/neck.json's servo signs and the client never sees a tick direction."""
    command = game_command(keys)
    assert (command.pan, command.tilt) == (pan, tilt) and command.twist == STOP


def test_space_overrides_everything_held_with_it() -> None:
    command = game_command(HeldKeys(up=True, pan_left=True, space=True, shift=True))
    assert command == GameCommand(stop_all=True, slow=True)


def test_an_unfocused_window_holds_nothing_whatever_the_keyboard_does() -> None:
    """The runaway hazard: pygame never sees the key-up of a key held while the focus left, so
    the loop reads HeldKeys() unfocused — and that is a full stop."""
    assert game_command(HeldKeys()) == GameCommand()


def test_the_stream_sends_a_twist_every_tick_held_and_one_stop_on_release() -> None:
    link = FakeLink()
    stream = CommandStream(link)
    forward = game_command(HeldKeys(up=True))
    for _ in range(3):
        stream.tick(forward)
    assert link.sent == [("twist", forward.twist)] * 3, "re-armed every tick, inside the deadman"
    stream.tick(game_command(HeldKeys()))
    stream.tick(game_command(HeldKeys()))
    assert link.sent[3:] == [("stop",)], "one stop the tick the key goes up, then silence"


def test_the_stream_jogs_every_tick_held_and_sends_one_zero_jog_on_release() -> None:
    link = FakeLink()
    stream = CommandStream(link)
    look_left = game_command(HeldKeys(pan_left=True, shift=True))
    stream.tick(look_left)
    stream.tick(look_left)
    assert link.sent == [("jog", 1, 0, True)] * 2
    stream.tick(game_command(HeldKeys(tilt_down=True)))  # a different key, no gap
    assert link.sent[2:] == [("jog", 0, 1, False)]
    stream.tick(game_command(HeldKeys()))
    stream.tick(game_command(HeldKeys()))
    assert link.sent[3:] == [("jog", 0, 0, False)], "the head stops the tick the key goes up"


def test_the_stream_keeps_wheels_and_head_apart_and_space_stops_both_every_tick() -> None:
    link = FakeLink()
    stream = CommandStream(link)
    stream.tick(game_command(HeldKeys(up=True, tilt_up=True)))
    assert link.sent == [("twist", Twist(FAST_LINEAR_M_S, 0.0)), ("jog", 0, -1, False)]
    stream.tick(game_command(HeldKeys(tilt_up=True)))  # the arrow released, W still held
    assert link.sent[2:] == [("stop",), ("jog", 0, -1, False)]
    stream.tick(game_command(HeldKeys(space=True)))
    stream.tick(game_command(HeldKeys(space=True)))
    assert link.sent[4:] == [("stop",), ("jog", 0, 0, False)] * 2
    stream.tick(game_command(HeldKeys()))
    assert len(link.sent) == 8, "nothing was moving: nothing to say"


def test_stop_all_sends_both_stops_whatever_the_previous_tick_was() -> None:
    link = FakeLink()
    CommandStream(link).stop_all()
    assert link.sent == [("stop",), ("jog", 0, 0, False)]
    link = FakeLink()
    CommandStream(link).stop_all(repeats=3, gap_s=0.0)
    assert link.sent == [("stop",), ("jog", 0, 0, False)] * 3, "the exit: one line may go astray"


def test_the_entry_point_names_the_missing_pygame_in_one_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    from pepin.teleop import main

    monkeypatch.setitem(sys.modules, "pygame", None)  # an install without the macos group
    with pytest.raises(SystemExit) as left:
        main(["--game"])
    assert left.value.code not in (0, None)
    assert "pygame is not installed: uv sync --group macos" in capsys.readouterr().err


def test_status_lines_show_the_twist_the_neck_and_the_focus() -> None:
    from pepin.base_link import decode_state
    from pepin.neck import NeckReading

    message = {
        "t": 1.0, "x": 0.0, "y": 0.0, "theta": 0.0, "dl": 0.0, "dr": 0.0, "v": 0.45, "w": 0.0,
        "moving": True, "armed": True, "deadman": False, "bus_ok": True,
    }  # fmt: skip
    state = decode_state(message, received_at=0.0)
    neck = NeckReading(pan_ticks=2029, tilt_ticks=2311, age_s=0.01, read_ms=1.4)
    command = game_command(HeldKeys(up=True, pan_left=True))
    text = "\n".join(status_lines(command, state, neck, None, focused=True))
    assert "v +0.45 m/s" in text and "FAST" in text and "moving, armed" in text
    assert "pan 2029 ticks" in text and "tilt 2311 ticks" in text and "jog pan +1 tilt +0" in text
    assert "UNFOCUSED" not in text and "refused" not in text
    text = "\n".join(
        status_lines(GameCommand(slow=True), None, None, "the wheels are moving", focused=False)
    )
    assert "SLOW" in text and "connecting" in text and "encoders unread" in text
    assert "refused: the wheels are moving" in text and "UNFOCUSED" in text


def test_the_entry_point_insists_on_game_mode() -> None:
    from pepin.teleop import main

    with pytest.raises(SystemExit):
        main([])  # the latching teleop is a ROS node, not this entry point


@pytest.mark.slow
def test_the_window_loop_runs_headless_and_stops_everything_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pygame plumbing end to end under SDL's dummy driver: the loop starts against a real
    base server on localhost, reads an unfocused window (nothing held, nothing sent), and the
    window closing stops the wheels and the head on the way out."""
    import threading
    import time

    pygame = pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from test_base_server import energised, make_neck_core

    from pepin.base_server import DRIVING_COMMANDS, serve
    from pepin.streams import JsonLinesServer
    from pepin.teleop import run_game

    core, bus = make_neck_core()
    server = JsonLinesServer(
        0, on_last_client_left={"cmd": "release"}, driving_commands=DRIVING_COMMANDS
    ).start()
    stop = threading.Event()
    worker = threading.Thread(target=serve, args=(core, server, 50.0, 20.0, stop), daemon=True)
    worker.start()
    seen_commands = []
    original = core.command

    def spy(message: dict[str, object], now: float) -> object:
        seen_commands.append(message.get("cmd"))
        return original(message, now)  # type: ignore[arg-type]

    monkeypatch.setattr(core, "command", spy)

    def close_window_soon() -> None:
        time.sleep(0.6)
        pygame.event.post(pygame.event.Event(pygame.QUIT))

    threading.Thread(target=close_window_soon, daemon=True).start()
    try:
        run_game("127.0.0.1", server.port, hz=20.0)
    finally:
        stop.set()
        worker.join(timeout=2.0)
    assert "neck" in seen_commands, "the window polled the encoders"
    # Besides the polls (and the server's own "release" once the client has left, which may or
    # may not have been ticked yet): only the exit's stops, three times — unfocused, nothing
    # was driven.
    driving = [cmd for cmd in seen_commands if cmd not in ("neck", "release")]
    assert driving == ["stop", "neck_jog"] * 3, "the exit stopped the wheels and the head, thrice"
    assert not core.moving and bus.torque == [], "the wheels were never armed"
    assert not energised(bus), "the head was never energised"


def test_the_keys_are_read_by_place_not_by_letter() -> None:
    """A Russian layout puts 'ц' on the W key: the window reads SDL scancodes (the key's place),
    so W/A/S/D and the arrows work under any layout. ``pg`` stands in for pygame's constants."""
    from types import SimpleNamespace

    from pepin.teleop import _held_keys

    pg = SimpleNamespace(
        KSCAN_UP=82, KSCAN_DOWN=81, KSCAN_LEFT=80, KSCAN_RIGHT=79, KSCAN_W=26, KSCAN_S=22,
        KSCAN_A=4, KSCAN_D=7, KSCAN_LSHIFT=225, KSCAN_RSHIFT=229, KSCAN_SPACE=44,
    )  # fmt: skip
    held = _held_keys({26, 4, 229}, pg)  # W, A and the right Shift, whatever their letters
    assert held.tilt_up and held.pan_left and held.shift
    assert not (held.up or held.down or held.tilt_down or held.pan_right or held.space)
    assert _held_keys(set(), pg) == HeldKeys()
