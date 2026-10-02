"""The robot as the tools see it: one client per owner of a truth, and the clock they wait by.

A tool's first parameter is a :class:`Robot`. It carries clients, not state: every answer a tool
gives was asked of its owner during the call. :meth:`Robot.connect` builds the real clients
from the sockets' hosts and ports (nothing else is configured, and nothing connects until a tool
asks); the tests build one from fakes (:func:`pepin.tools.fakes.fake_robot`).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pepin import goal_link
from pepin.tools.clients import (
    AUDIO_PORT,
    CAMERA_PORT,
    GAZE_URL,
    WORLD_URL,
    BoardSpeech,
    Camera,
    GazeNeck,
    GoalServer,
    GoalServerLink,
    Neck,
    Speech,
    UstreamerCamera,
    World,
    WorldHttp,
)
from pepin.tools.registry import ToolError

logger = logging.getLogger(__name__)

BOARD_HOST = "10.0.0.187"  # the board, as the ros/*.sh scripts default it
DRIVE_TIMEOUT_S = 240.0  # a drive that has not ended by then is cancelled: nobody queues motions


@dataclass(frozen=True)
class Endpoints:
    """Where the owners listen. The board serves the voice and the camera; the goal server and
    the gaze arbiter (the head's owner) are the laptop's, beside Nav2 (``ros/laptop.sh nav``),
    and ``world`` is the laptop's."""

    board: str = BOARD_HOST
    goal_host: str = "127.0.0.1"
    goal_port: int = goal_link.PORT
    gaze_url: str = GAZE_URL
    audio_port: int = AUDIO_PORT
    camera_port: int = CAMERA_PORT
    world_url: str = WORLD_URL

    @classmethod
    def from_env(cls) -> Endpoints:
        """The defaults, overridden by ``PEPIN_HOST`` (the board, as every ros/ script reads
        it), ``PEPIN_GOAL_HOST``, ``PEPIN_GAZE_URL`` and ``PEPIN_WORLD_URL``."""
        env = os.environ
        return cls(
            board=env.get("PEPIN_HOST") or BOARD_HOST,
            goal_host=env.get("PEPIN_GOAL_HOST") or "127.0.0.1",
            gaze_url=env.get("PEPIN_GAZE_URL") or GAZE_URL,
            world_url=env.get("PEPIN_WORLD_URL") or WORLD_URL,
        )


@dataclass
class Robot:
    """Everything a tool may touch. ``clock``/``sleep`` are what the tools wait by (a fake
    clock makes a two-second wait instant in a test); ``drive_timeout_s`` bounds a drive."""

    goals: GoalServer
    neck: Neck
    world: World
    camera: Camera
    speech: Speech
    clock: Callable[[], float] = field(default=time.monotonic)
    sleep: Callable[[float], None] = field(default=time.sleep)
    drive_timeout_s: float = DRIVE_TIMEOUT_S

    @classmethod
    def connect(cls, endpoints: Endpoints | None = None) -> Robot:
        """The real clients at ``endpoints`` (:meth:`Endpoints.from_env` when None)."""
        where = endpoints or Endpoints.from_env()
        return cls(
            goals=GoalServerLink(where.goal_host, where.goal_port),
            neck=GazeNeck(where.gaze_url),
            world=WorldHttp(where.world_url),
            camera=UstreamerCamera(where.board, where.camera_port),
            speech=BoardSpeech(where.board, where.audio_port),
        )

    def halt(self) -> str:
        """Cancel every drive, in words; never raises. Called when a caller gives up on a tool
        that set the robot in motion (Ctrl-C in a chat loop, a cancelled MCP call)."""
        try:
            answer = self.goals.cancel()
        except (ToolError, OSError) as error:
            why = error.why if isinstance(error, ToolError) else str(error)
            logger.error("halt: %s", why)
            return f"the cancel did NOT reach the robot: {why}"
        logger.warning("halt: %s", answer)
        return "cancel sent: every drive is stopping"
