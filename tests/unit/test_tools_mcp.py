"""pepin.tools.mcp: the registry served as MCP tools — the registry's own schemas, results as
text with pictures beside them, a cancel served while a drive blocks, and an abandoned drive
halted. The server runs in process against a robot made of fakes."""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterator
from typing import Any

import anyio
from mcp import Client

from pepin.tools import TOOLS
from pepin.tools.clients import HeadMove
from pepin.tools.fakes import FakeCamera, FakeGoalServer, fake_robot
from pepin.tools.mcp import build_server, call, to_mcp
from pepin.tools.registry import Image, fail, ok


def test_the_mcp_tools_are_the_registry_s() -> None:
    async def listed() -> list[Any]:
        async with Client(build_server(fake_robot())) as client:
            return list((await client.list_tools()).tools)

    tools = anyio.run(listed)
    assert [t.name for t in tools] == [t.name for t in TOOLS]
    for served, registered in zip(tools, TOOLS, strict=True):
        assert served.description == registered.description
        assert served.input_schema == registered.input_schema()


def test_calls_come_back_as_text_pictures_and_the_error_flag() -> None:
    camera = FakeCamera()

    async def calls() -> list[Any]:
        async with Client(build_server(fake_robot(camera=camera))) as client:
            return [
                await client.call_tool("go_to", {"place": "printer"}),
                await client.call_tool("see", {}),
                await client.call_tool("go_to", {"place": "kitchen"}),
            ]

    drove, saw, refused = anyio.run(calls)
    assert not drove.is_error and json.loads(drove.content[0].text)["arrived"] is True
    assert [c.type for c in saw.content] == ["text", "image"]
    assert base64.b64decode(saw.content[1].data) == camera.jpeg
    assert refused.is_error and "Known places" in json.loads(refused.content[0].text)["why"]


def test_to_mcp_keeps_the_picture_out_of_the_text() -> None:
    result = to_mcp(ok(image=Image(b"png!", "image/png", 2, 2)))
    assert [c.type for c in result.content] == ["text", "image"]
    assert result.content[1].mime_type == "image/png" and not result.is_error
    assert to_mcp(fail("no")).is_error


class BlockingGoals(FakeGoalServer):
    """A goal server whose drive runs until someone cancels it."""

    def __init__(self, robot_clock: Any) -> None:
        super().__init__(robot_clock)
        self.driving = threading.Event()
        self.stopped = threading.Event()

    def go(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        self.asked.append(request)
        yield {"event": "accepted", "run": 1, "place": "printer", "x": -1.2, "y": 3.4, "yaw_deg": 0}
        self.driving.set()
        self.stopped.wait(5.0)
        yield {"event": "done", "run": 1, "status": 5, "seconds": 2.0}

    def cancel(self) -> dict[str, Any]:
        self.stopped.set()
        return super().cancel()


def test_a_cancel_is_served_while_a_drive_blocks() -> None:
    """go_to holds its call until the drive ends; the cancel must not wait behind it."""
    robot = fake_robot()
    goals = BlockingGoals(robot.clock)
    robot.goals = goals
    results: dict[str, Any] = {}

    async def scene() -> None:
        async with Client(build_server(robot)) as client:

            async def drive() -> None:
                results["go_to"] = await client.call_tool("go_to", {"place": "printer"})

            async with anyio.create_task_group() as group:
                group.start_soon(drive)
                await anyio.to_thread.run_sync(goals.driving.wait, 5.0)
                results["cancel"] = await client.call_tool("cancel", {})

    anyio.run(scene)
    assert not results["cancel"].is_error
    why = json.loads(results["go_to"].content[0].text)["why"]
    assert why == "the drive was cancelled from outside this call"


def test_an_abandoned_drive_is_halted() -> None:
    """The client stops waiting for go_to (Escape, a timeout): the robot must not drive on."""
    robot = fake_robot()
    goals = BlockingGoals(robot.clock)
    robot.goals = goals

    async def abandon() -> None:
        with anyio.move_on_after(5.0) as scope:
            async with anyio.create_task_group() as group:
                group.start_soon(call, TOOLS, TOOLS["go_to"], {"place": "printer"}, robot)
                await anyio.to_thread.run_sync(goals.driving.wait, 5.0)
                group.cancel_scope.cancel()
        assert not scope.cancelled_caught

    anyio.run(abandon)
    assert goals.cancelled == 1 and goals.stopped.is_set()


def test_an_abandoned_look_does_not_halt_anything() -> None:
    """Only a tool that moves the robot halts it when abandoned: a head turn is let finish."""
    robot = fake_robot()
    goals = robot.goals
    assert isinstance(goals, FakeGoalServer)
    turning, release = threading.Event(), threading.Event()
    turn = robot.neck.turn

    def slow_turn(pan_deg: float | None, tilt_deg: float | None) -> HeadMove:
        turning.set()
        release.wait(5.0)
        return turn(pan_deg, tilt_deg)

    robot.neck.turn = slow_turn  # type: ignore[method-assign]

    async def abandon() -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(call, TOOLS, TOOLS["look"], {"pan_deg": 10.0}, robot)
            await anyio.to_thread.run_sync(turning.wait, 5.0)
            group.cancel_scope.cancel()
            threading.Timer(0.05, release.set).start()

    anyio.run(abandon)
    assert goals.cancelled == 0 and robot.neck.pose().pan_deg == 10.0
