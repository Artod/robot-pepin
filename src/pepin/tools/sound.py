"""Sounds as a tool: a sound of the registry (config/sounds.json, a private tree's copy merged
over it) through the robot's speaker, the body dancing to its beats when the dance family has
its style (:class:`pepin.expression.listeners.MusicPulse`); ``stop`` stops it.

The player is the voice cascade's (:class:`pepin.sound_player.StreamSoundPlayer`): a sound asked
during a reply plays once the reply's speech is over, and speech that comes while it plays stops
it. The marks' door is ``{sound:NAME}`` (:mod:`pepin.sounds`). The description names the sounds
whose file is here, read from the merged registry when this module is imported: a private
entry shows once its tree is mounted, a file dropped in later after a restart.
"""

from __future__ import annotations

import logging
import math

from pepin.sound_player import STOP, playable
from pepin.sounds import load_registry
from pepin.tools.registry import Result, fail, ok, tool
from pepin.tools.robot import Robot

logger = logging.getLogger(__name__)


def sound_names() -> tuple[str, ...]:
    """The registry's sounds the player can play here, in its order; none when it is
    unreadable."""
    try:
        registry = load_registry()
    except (OSError, ValueError) as error:
        logger.warning("config/sounds.json unreadable (%s): the sound tool names none", error)
        return ()
    return tuple(s.name for s in registry.sounds.values() if playable(s))


def sound(robot: Robot, name: str, seconds: float | None = None) -> Result:
    """Play a sound or a tune through the speaker when a person asks for music, a song or a
    dance: it starts after what you are saying and the body dances to its beat; speaking while
    it plays stops it, "stop" stops it. The sounds: {names}.

    Args:
        name: a sound from the list, or "stop".
        seconds: how long to play it (left out: the whole sound, never longer).
    """
    player = robot.sounds
    if player is None:
        return fail("sounds play only in the voice cascade (its player owns the speaker)")
    word = name.strip().lower()
    if word == STOP:
        playing = player.current()
        if playing is None:
            return ok(stopped=None)
        player.stop(playing)
        return ok(stopped=playing.sound.name)
    if seconds is not None and not (math.isfinite(seconds) and seconds > 0.0):
        return fail("seconds is a number above 0, or left out for the whole sound")
    try:
        playing = player.play(word, seconds)
    except KeyError:
        return fail(f"there is no sound {name!r}; the sounds: {', '.join(sound_names())}")
    except RuntimeError as error:
        return fail(str(error))
    return ok(
        sound=playing.sound.name,
        seconds=round(playing.sound.seconds, 1),
        style=playing.sound.style,
        starts="after your speech" if playing.waiting else "now",
    )


# the names filled in from the merged registry, then registered (the docstring is the schema)
sound.__doc__ = (sound.__doc__ or "").format(names=", ".join(sound_names()) or "none here")
tool(sound)
