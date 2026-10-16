"""The sound tool (pepin.tools.sound): a registry sound played by the voice cascade's player, its
answer (the cut, the style, now or after the speech), "stop", the refusals in words, and its
description naming the sounds whose file is here, read from the merged registry."""

from __future__ import annotations

from typing import Any

import pytest

import pepin.tools.sound as sound_tool
from pepin.sounds import (
    DEFAULT_BEAT,
    BeatGrid,
    FakeSoundPlayer,
    MusicKnobs,
    Playing,
    Registry,
    Sound,
    registry_from,
)
from pepin.tools import TOOLS, Result, Robot
from pepin.tools.fakes import FakeClock, fake_robot


def call(robot: Robot, /, **arguments: Any) -> Result:
    return TOOLS.call("sound", arguments, robot)


def drums() -> Registry:
    grid = BeatGrid(120.0, tuple(0.5 * k for k in range(8)), (1.0,) * 8)
    return Registry(
        {
            "drum": Sound("drum", "drum.wav", 0.0, 4.0, "a drum machine", "rock", grid),
            "chime": Sound("chime", None, 0.0, 2.0, "no file yet", "laugh"),
        },
        MusicKnobs(),
    )


def robot_with_player() -> tuple[Robot, FakeSoundPlayer]:
    clock = FakeClock()
    player = FakeSoundPlayer(drums(), clock)
    return fake_robot(clock=clock, sounds=player), player


def test_a_sound_plays_its_cut_now_and_stop_stops_it() -> None:
    robot, player = robot_with_player()
    played = call(robot, name="Drum")
    assert played == {"ok": True, "sound": "drum", "seconds": 4.0, "style": "rock",
                      "starts": "now"}  # fmt: skip
    assert player.calls == [("play", "drum")]
    assert call(robot, name="drum", seconds=1.5)["seconds"] == 1.5
    assert call(robot, name="drum", seconds="9")["seconds"] == 4.0  # never past the cut
    assert call(robot, name="stop") == {"ok": True, "stopped": "drum"}
    assert player.playing is None
    assert call(robot, name=" STOP ") == {"ok": True, "stopped": None}


def test_a_sound_asked_while_the_robot_speaks_says_it_starts_after_the_speech() -> None:
    class Waiting(FakeSoundPlayer):
        def play(self, name: str, seconds: float | None = None, style: str | None = None) -> Any:
            playing = super().play(name, seconds, style)
            return Playing(playing.sound, playing.id, playing.t0, playing.beats, waiting=True)

    clock = FakeClock()
    robot = fake_robot(clock=clock, sounds=Waiting(drums(), clock))
    assert call(robot, name="drum")["starts"] == "after your speech"


def test_the_refusals_come_back_in_words() -> None:
    robot, _ = robot_with_player()
    unknown = call(robot, name="fireworks")
    assert not unknown["ok"] and "there is no sound 'fireworks'" in unknown["why"]
    assert "the sounds:" in unknown["why"]
    absent = call(robot, name="chime")
    assert not absent["ok"] and "no file" in absent["why"]
    for bad in (0, -1.0, float("nan")):
        refused = call(robot, name="drum", seconds=bad)
        assert not refused["ok"] and "above 0" in refused["why"]
    nobody = call(fake_robot(sounds=None), name="drum")
    assert not nobody["ok"] and "voice cascade" in nobody["why"]


def test_the_description_names_the_sounds_whose_file_is_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registered description (the public registry: the default beat, made here on first
    use) and the names read again from a merged registry: a recording without its file is left
    out, the order is the registry's."""
    description = TOOLS["sound"].description
    assert f"The sounds: {DEFAULT_BEAT}" in description and "{names}" not in description
    assert len(description.split()) < 60, "short: the brain reads it every turn"
    merged = registry_from({"sounds": [
        {"name": "hava_nagila", "file": "nowhere_here.wav", "style": "folk"},
        {"name": "default_beat", "file": "default_beat.wav", "synthesized": True},
    ]})  # fmt: skip
    monkeypatch.setattr(sound_tool, "load_registry", lambda: merged)
    assert sound_tool.sound_names() == ("default_beat",)

    def broken() -> Registry:
        raise ValueError("sounds: 'x' twice")

    monkeypatch.setattr(sound_tool, "load_registry", broken)
    assert sound_tool.sound_names() == ()
