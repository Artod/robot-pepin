"""A run's tape starts before the command did, and never outlives the run."""

import io
from pathlib import Path

from pepin.tape import RunTape, next_run_number


class Tape(io.StringIO):
    """A StringIO that keeps its text after close, so a test can read what was written."""

    def __init__(self) -> None:
        super().__init__()
        self.text = ""

    def close(self) -> None:
        self.text = self.getvalue()
        super().close()


class Fake:
    """A clock and a writer under the test's control."""

    def __init__(self) -> None:
        self.now = 0.0
        self.streams: list[Tape] = []

    def clock(self) -> float:
        return self.now

    def open(self, path: Path) -> Tape:
        stream = Tape()
        self.streams.append(stream)
        return stream


def tape(prelude_s: float = 15.0, max_run_s: float = 900.0) -> tuple[RunTape, Fake]:
    fake = Fake()
    return (
        RunTape(
            prelude_s=prelude_s,
            max_run_s=max_run_s,
            clock=fake.clock,
            opener=fake.open,
            sync=lambda _stream: None,
        ),
        fake,
    )


def records(stream: Tape) -> list[str]:
    text = stream.text if stream.closed else stream.getvalue()
    return [line for line in text.splitlines() if line]


def test_the_seconds_before_the_command_are_on_the_tape() -> None:
    """The first turn happens while the goal is still being accepted; it must be recorded."""
    run, fake = tape()
    for i in range(5):
        fake.now = float(i)
        run.add({"t": fake.now, "topic": "scan", "n": i})
    run.start(Path("/tmp/run.jsonl"))
    written = records(fake.streams[0])
    assert len(written) == 5, "the prelude was dropped"
    assert '"n":0' in written[0] and '"n":4' in written[4], "the prelude lost its order"


def test_the_prelude_forgets_what_is_older_than_its_window() -> None:
    run, fake = tape(prelude_s=3.0)
    for i in range(10):
        fake.now = float(i)
        run.add({"topic": "scan", "n": i})
    run.start(Path("/tmp/run.jsonl"))
    assert len(records(fake.streams[0])) == 4  # t in [6, 9]: the last three seconds and now


def test_records_go_to_the_file_while_a_run_is_open() -> None:
    run, fake = tape()
    run.start(Path("/tmp/run.jsonl"))
    assert run.recording
    for i in range(3):
        fake.now += 0.1
        run.add({"topic": "cmd", "n": i})
    run.stop()
    assert not run.recording
    assert len(records(fake.streams[0])) == 3
    assert fake.streams[0].closed


def test_a_stopped_tape_remembers_again_instead_of_writing() -> None:
    """After a run the buffer refills, so the next run also starts with its prelude."""
    run, fake = tape()
    run.start(Path("/tmp/one.jsonl"))
    run.add({"topic": "scan", "n": 1})
    run.stop()
    fake.now += 1.0
    run.add({"topic": "scan", "n": 2})
    run.start(Path("/tmp/two.jsonl"))
    assert len(records(fake.streams[1])) == 1, "the second run did not get its prelude"
    assert '"n":2' in records(fake.streams[1])[0]


def test_a_run_nobody_stops_closes_itself() -> None:
    """An orphan recorder ate a core for forty minutes once; the tape stops on its own limit."""
    run, fake = tape(max_run_s=10.0)
    run.start(Path("/tmp/run.jsonl"))
    fake.now = 5.0
    run.add({"topic": "scan", "n": 1})
    fake.now = 20.0
    run.add({"topic": "scan", "n": 2})
    assert not run.recording
    assert len(records(fake.streams[0])) == 1
    assert fake.streams[0].closed


def test_starting_twice_closes_the_first_tape() -> None:
    run, fake = tape()
    run.start(Path("/tmp/one.jsonl"))
    run.start(Path("/tmp/two.jsonl"))
    assert fake.streams[0].closed and not fake.streams[1].closed


def test_stopping_a_tape_that_never_started_is_harmless() -> None:
    run, _ = tape()
    run.stop()
    run.stop()
    assert not run.recording


def test_a_real_file_is_written_and_reread(tmp_path: Path) -> None:
    """The default writer must produce jsonl a replay can parse, not just satisfy the fakes."""
    run = RunTape()
    path = run.start(tmp_path / "rec" / "run.jsonl")
    run.add({"t": 1.0, "topic": "scan", "ranges": [0.1, 0.2]})
    run.stop()
    import json

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"t": 1.0, "topic": "scan", "ranges": [0.1, 0.2]}
    ]


def test_runs_are_numbered_so_they_can_be_named_aloud(tmp_path: Path) -> None:
    """Artem refers to a drive by its number; the number must survive a restart of the node."""
    assert next_run_number(tmp_path) == 1
    assert next_run_number(tmp_path) == 2
    assert (tmp_path / ".run_seq").read_text().strip() == "2"


def test_a_lost_counter_falls_back_to_the_files_on_disk(tmp_path: Path) -> None:
    for number in (1, 2, 3):
        (tmp_path / f"{number:04d}_20260908_120000_home.jsonl").write_text("")
    assert next_run_number(tmp_path) == 4
    (tmp_path / ".run_seq").write_text("nonsense")
    assert next_run_number(tmp_path) == 4


def test_the_counter_directory_is_made_when_it_is_missing(tmp_path: Path) -> None:
    assert next_run_number(tmp_path / "rec") == 1
    assert (tmp_path / "rec" / ".run_seq").exists()
