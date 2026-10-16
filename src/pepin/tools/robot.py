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
from typing import TYPE_CHECKING

from pepin import goal_link
from pepin.expression.executors import Animator
from pepin.people.door import PeopleDoor
from pepin.sounds import SoundPlayer
from pepin.tools.clients import (
    AUDIO_PORT,
    BASE_PORT,
    CAMERA_PORT,
    GAZE_URL,
    HEAD_PORT,
    WORLD_URL,
    BaseBody,
    BoardSpeech,
    Body,
    Camera,
    ChannelSightings,
    Eyes,
    Face,
    Facing,
    GazeNeck,
    GoalServer,
    GoalServerLink,
    HeadFace,
    Neck,
    Seen,
    ServiceCamera,
    Sightings,
    Speech,
    SwitchedCamera,
    UstreamerCamera,
    World,
    WorldHttp,
)
from pepin.tools.registry import ToolError

if TYPE_CHECKING:
    from pepin.perception.client import Perception
    from pepin.perception.source import VisionSource

logger = logging.getLogger(__name__)

BOARD_HOST = "10.0.0.187"  # the board, as the ros/*.sh scripts default it
DEPTH_URL = "http://127.0.0.1:8790"  # the laptop's depth host (ros/depth_host.sh)
DRIVE_TIMEOUT_S = 240.0  # a drive that has not ended by then is cancelled: nobody queues motions
STILL_WAIT_S = 0.3  # see's wait for the first still frame (config/models.json perception)


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
    head_port: int = HEAD_PORT
    base_port: int = BASE_PORT
    depth_url: str = DEPTH_URL

    @classmethod
    def from_env(cls) -> Endpoints:
        """The defaults, overridden by ``PEPIN_HOST`` (the board, as every ros/ script reads
        it), ``PEPIN_GOAL_HOST``, ``PEPIN_GAZE_URL``, ``PEPIN_WORLD_URL`` and
        ``PEPIN_DEPTH_HOST_URL``."""
        env = os.environ
        return cls(
            board=env.get("PEPIN_HOST") or BOARD_HOST,
            goal_host=env.get("PEPIN_GOAL_HOST") or "127.0.0.1",
            gaze_url=env.get("PEPIN_GAZE_URL") or GAZE_URL,
            world_url=env.get("PEPIN_WORLD_URL") or WORLD_URL,
            depth_url=env.get("PEPIN_DEPTH_HOST_URL") or DEPTH_URL,
        )


@dataclass
class Robot:
    """Everything a tool may touch. ``clock``/``sleep`` are what the tools wait by (a fake
    clock makes a two-second wait instant in a test); ``drive_timeout_s`` bounds a drive;
    ``animator`` performs gestures (None where no gesture engine runs: the tool says so);
    ``sounds`` plays the registry's sounds (None where no voice cascade runs);
    ``people`` names and forgets people (None where recognition does not run); ``sightings``
    says who is in front of the robot and ``eyes`` where a point of the picture is on the map
    (the approach tool's).

    ``seen`` is the one thing a tool leaves for another: where the head pointed for the picture
    ``see`` last showed the model and that picture's pixel space, so a box the model draws on it
    is placed in the space it was drawn in, and refused once the head has turned (or the robot
    driven) since and the box would point at something else."""

    goals: GoalServer
    neck: Neck
    world: World
    camera: Camera
    speech: Speech
    face: Face
    body: Body
    eyes: Eyes
    sightings: Sightings
    clock: Callable[[], float] = field(default=time.monotonic)
    sleep: Callable[[float], None] = field(default=time.sleep)
    drive_timeout_s: float = DRIVE_TIMEOUT_S
    animator: Animator | None = None  # the gesture engine, where one runs (the voice cascade)
    people: PeopleDoor | None = None  # person recognition, where it runs (the voice cascade)
    gaze: Facing | None = None  # the speaker gaze, where it runs (the voice cascade)
    sounds: SoundPlayer | None = None  # the sound player, where it runs (the voice cascade)
    still_wait_s: float = STILL_WAIT_S  # see: the longest wait for the first still frame
    seen: Seen | None = None

    @classmethod
    def connect(
        cls,
        endpoints: Endpoints | None = None,
        *,
        vision: VisionSource | None = None,
        perception: Perception | None = None,
    ) -> Robot:
        """The real clients at ``endpoints`` (:meth:`Endpoints.from_env` when None). The
        picture is the perception service's (``perception``, else a client of config/
        models.json's) while ``vision`` (the live ``vision.source``; else its start value)
        says service, ustreamer's snapshot otherwise."""
        from pepin.perception.client import PerceptionClient
        from pepin.perception.config import PerceptionConfig
        from pepin.perception.source import VisionSource
        from pepin.tools.eyes import stereo_eyes

        where = endpoints or Endpoints.from_env()
        try:
            channel = PerceptionConfig.load()
        except (OSError, ValueError) as error:
            logger.warning("config/models.json perception: %s; its defaults", error)
            channel = PerceptionConfig()
        source = vision if vision is not None else VisionSource(channel.source)
        service = perception if perception is not None else PerceptionClient(channel.url)
        goals = GoalServerLink(where.goal_host, where.goal_port)
        neck = GazeNeck(where.gaze_url)
        return cls(
            goals=goals,
            neck=neck,
            world=WorldHttp(where.world_url),
            camera=SwitchedCamera(
                source, ServiceCamera(service), UstreamerCamera(where.board, where.camera_port)
            ),
            speech=BoardSpeech(where.board, where.audio_port),
            face=HeadFace(where.board, where.head_port),
            body=BaseBody(where.board, where.base_port),
            eyes=stereo_eyes(where.board, where.camera_port, where.depth_url, neck, goals),
            sightings=ChannelSightings(service, source, wait_s=channel.look_wait_s),
            still_wait_s=channel.see_wait_s,
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
