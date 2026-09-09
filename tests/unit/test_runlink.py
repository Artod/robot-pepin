"""One command opens a tape where the sensors are; one status names it."""

from pepin.runlink import (
    IDLE,
    RECORDING,
    RunLink,
    RunStatus,
    parse_command,
    start_command,
    stop_command,
)


def test_commands_round_trip_and_noise_is_refused() -> None:
    assert parse_command(start_command("printer")) == ("start", "printer")
    assert parse_command(stop_command()) == ("stop", None)
    assert parse_command('{"cmd": "start"}') is None  # a tape needs a name
    assert parse_command('{"cmd": "start", "name": ""}') is None
    assert parse_command("not json") is None
    assert parse_command('{"cmd": "dance"}') is None


def test_the_status_round_trips_and_a_broken_one_is_none() -> None:
    s = RunStatus(RECORDING, 42, "/maps/rec/0042_x_printer.jsonl", "printer")
    assert RunStatus.from_json(s.to_json()) == s
    assert RunStatus.from_json(RunStatus(IDLE, 41).to_json()) == RunStatus(IDLE, 41, None, None)
    assert RunStatus.from_json("{}") is None
    assert RunStatus.from_json("[]") is None


def test_the_link_knows_when_its_run_started_and_stopped() -> None:
    link = RunLink()
    assert link.stopped() and link.run == 0 and link.recording is None
    link.observe(RunStatus(RECORDING, 7, "/maps/rec/0007_home.jsonl", "home"))
    assert link.started("home") and not link.started("printer") and not link.stopped()
    assert link.run == 7 and link.recording == "/maps/rec/0007_home.jsonl"
    link.observe(RunStatus(IDLE, 7))
    assert link.stopped() and not link.started("home")
    assert link.run == 7  # the last run keeps its number for the "done" event
