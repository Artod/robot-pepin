"""The cascade's transcriber: the engine the config names, the control file's choice, and the
swap between utterances (fake engines, no model)."""

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.control import ControlFile
from pepin.voice_cascade.stages import Transcriber
from pepin.voice_cascade.transcriber import (
    Choice,
    SwitchingTranscriber,
    build_transcriber,
    choice_of,
    configured,
    wanted_from,
)

CONFIG = Path(__file__).resolve().parents[2] / "config" / "voice_cascade.json"


class Named:
    """An engine that answers with its own name; remembers being closed."""

    def __init__(self, name: str, gate: threading.Event | None = None) -> None:
        self.name, self.gate, self.closed = name, gate, False
        self.started = threading.Event()

    def transcribe(self, pcm: bytes) -> str:
        self.started.set()
        if self.gate is not None:
            self.gate.wait(5)
        return self.name

    def close(self) -> None:
        self.closed = True


def now(work: Callable[[], None]) -> None:
    work()


def test_the_file_picks_gigaam_and_switches_to_the_neural_engine_for_one_run() -> None:
    config = CascadeConfig.load(CONFIG)
    assert configured(config) == Choice("gigaam", "gigaam-v3-e2e-rnnt")
    assert configured(config).label == "gigaam:gigaam-v3-e2e-rnnt"
    ane = CascadeConfig.load(CONFIG, ["transcriber.engine=whispercpp"])
    assert configured(ane) == Choice("whispercpp", "large-v3-turbo")
    assert (ane.whispercpp_units, ane.whispercpp_gpu) == ("ane", False)
    large = CascadeConfig.load(
        CONFIG, ["transcriber.engine=whispercpp", "transcriber.whispercpp.model=large-v3"]
    )
    assert configured(large).label == "whispercpp:large-v3"
    for bad in (["transcriber.engine=vosk"], ["transcriber.whispercpp.coreml_units=npu"]):
        with pytest.raises(ValueError, match="transcriber"):
            CascadeConfig.load(CONFIG, bad)


def test_a_control_value_names_an_engine_and_maybe_its_model() -> None:
    config = CascadeConfig.load(CONFIG)
    assert choice_of("whispercpp", config) == Choice("whispercpp", "large-v3-turbo")
    assert choice_of("whispercpp:large-v3", config) == Choice("whispercpp", "large-v3")
    assert choice_of({"engine": "mlx"}, config) == Choice(
        "mlx", "mlx-community/whisper-large-v3-turbo"
    )
    assert choice_of({"engine": "whispercpp", "model": "large-v3"}, config) == Choice(
        "whispercpp", "large-v3"
    )
    for bad in ("vosk", 3, None, {"model": "large-v3"}):
        assert choice_of(bad, config) is None


def test_the_control_file_counts_only_after_it_changes(tmp_path: Path) -> None:
    config = CascadeConfig.load(CONFIG)
    path = tmp_path / "control.json"
    path.write_text(json.dumps({"transcriber": "whispercpp", "voice": "say"}))
    control = ControlFile(tmp_path)  # the one reader; the loop polls it
    wanted = wanted_from(control, config)
    assert control.poll() is None and wanted() is None  # an earlier run's: the config wins
    path.write_text(json.dumps({"voice": "qwen3", "transcriber": "whispercpp:large-v3"}))
    os.utime(path, ns=(1, 2_000_000_000))
    request = control.poll()
    assert request is not None and request.transcriber == "whispercpp:large-v3"
    assert wanted() == Choice("whispercpp", "large-v3")
    path.write_text("{ half written")
    os.utime(path, ns=(1, 3_000_000_000))
    assert control.poll() is None
    assert wanted() == Choice("whispercpp", "large-v3")  # the last good choice
    path.write_text(json.dumps({"voice": "say"}))
    os.utime(path, ns=(1, 4_000_000_000))
    assert control.poll() is not None
    assert wanted() == Choice("whispercpp", "large-v3")  # no transcriber key: no change
    absent = ControlFile(tmp_path / "absent")
    assert absent.poll() is None and wanted_from(absent, config)() is None


def test_the_engine_is_swapped_between_utterances_and_the_old_one_closed() -> None:
    first, second = Named("first"), Named("second")
    wanted: list[Choice | None] = [None]
    lines: list[str] = []
    switch = SwitchingTranscriber(
        first, Choice("mlx", "a"), lambda choice: second, lambda: wanted[0],
        report=lines.append, spawn=now, settle_s=0,
    )  # fmt: skip
    switch.poll()
    assert switch.transcribe(b"") == "first" and switch.name == "mlx:a"
    wanted[0] = Choice("whispercpp", "b")
    switch.poll()
    assert switch.transcribe(b"") == "second" and switch.name == "whispercpp:b"
    assert first.closed and not second.closed
    assert lines and "mlx:a -> whispercpp:b" in lines[0]
    switch.close()
    assert second.closed


def test_a_swap_waits_for_the_call_in_flight() -> None:
    gate = threading.Event()
    first, second = Named("first", gate), Named("second")
    want = Choice("whispercpp", "b")
    switch = SwitchingTranscriber(
        first, Choice("mlx", "a"), lambda choice: second, lambda: want, report=lambda _: None,
        settle_s=0,
    )  # fmt: skip
    heard: list[str] = []
    reader = threading.Thread(target=lambda: heard.append(switch.transcribe(b"")))
    reader.start()
    assert first.started.wait(5)
    loader = threading.Thread(target=switch.poll)
    loader.start()
    loader.join(5)
    assert not first.closed  # the load waits on its thread while the call runs
    gate.set()
    reader.join(5)
    for _ in range(100):
        if first.closed:
            break
        threading.Event().wait(0.01)
    assert heard == ["first"] and first.closed and switch.transcribe(b"") == "second"


def test_an_engine_that_fails_to_load_is_reported_once_and_the_old_one_kept() -> None:
    first = Named("first")
    calls: list[Choice] = []
    lines: list[str] = []

    def broken(choice: Choice) -> Transcriber:
        calls.append(choice)
        raise FileNotFoundError("ggml-large-v3.bin")

    outcomes: list[tuple[Choice, BaseException | None]] = []
    switch = SwitchingTranscriber(
        first, Choice("mlx", "a"), broken, lambda: Choice("whispercpp", "large-v3"),
        report=lines.append, spawn=now, on_loaded=lambda c, e: outcomes.append((c, e)),
    )  # fmt: skip
    switch.poll()
    switch.poll()
    assert len(calls) == 1 and switch.transcribe(b"") == "first" and not first.closed
    assert "failed to load" in lines[0] and "keeping mlx:a" in lines[0]
    assert switch.failed == Choice("whispercpp", "large-v3")
    assert len(outcomes) == 1 and isinstance(outcomes[0][1], FileNotFoundError)


def test_a_switch_back_while_another_engine_loads_keeps_the_one_in_use() -> None:
    first, slow = Named("first"), Named("slow")
    loads: list[Callable[[], None]] = []  # the loads asked for, run by hand
    wanted: list[Choice | None] = [Choice("whispercpp", "b")]
    outcomes: list[tuple[str, Choice, BaseException | None]] = []
    switch = SwitchingTranscriber(
        first, Choice("mlx", "a"), lambda choice: slow, lambda: wanted[0],
        report=lambda _: None, spawn=loads.append, settle_s=0,
    )  # fmt: skip
    switch.on_loaded = lambda c, e: outcomes.append(("asked", c, e))
    switch.poll()  # whispercpp starts loading (34 s the first time: CoreML compiles)
    wanted[0] = Choice("mlx", "a")  # asked back meanwhile: the one in use
    switch.on_loaded = lambda c, e: outcomes.append(("asked back", c, e))
    switch.poll()
    assert len(loads) == 1  # nothing new to load
    loads[0]()  # the slow load finishes
    assert switch.transcribe(b"") == "first" and switch.name == "mlx:a" and not first.closed
    assert slow.closed  # built, never used
    ((who, choice, error),) = outcomes
    assert who == "asked" and choice == Choice("whispercpp", "b")  # answered to its own request
    assert isinstance(error, RuntimeError) and "superseded" in str(error)
    wanted[0] = Choice("whispercpp", "b")
    switch.poll()  # asked for again: loaded again
    loads[1]()
    assert switch.transcribe(b"") == "slow"


def test_a_whispercpp_build_without_its_files_names_the_missing_one(tmp_path: Path) -> None:
    config = CascadeConfig.load(CONFIG, ["transcriber.engine=whispercpp"])
    with pytest.raises(FileNotFoundError, match="bin/whisper-server"):
        build_transcriber(config, configured(config), tmp_path)
