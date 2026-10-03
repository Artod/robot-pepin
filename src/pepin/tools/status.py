"""What is up: every owner asked once, in parallel, each answer in one line."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pepin.tools.registry import Result, ToolError, ok, tool
from pepin.tools.robot import Robot


@tool
def status(robot: Robot) -> Result:
    """What is up and what is not: the goal server (driving and where the robot is), the head,
    the memory (world), the voice, the camera and the face screen, each with a short detail.
    Call it when a tool says a service is not answering."""
    checks: dict[str, Callable[[], Any]] = {
        "goal_server": lambda: _goal_server(robot),
        "head": lambda: robot.neck.pose().as_dict(),
        "world": robot.world.health,
        "voice": robot.speech.health,
        "camera": robot.camera.health,
        "face": robot.face.health,
    }
    with ThreadPoolExecutor(len(checks)) as pool:
        futures = {name: pool.submit(probe, check) for name, check in checks.items()}
        services = {name: future.result() for name, future in futures.items()}
    return ok(
        up=[name for name, s in services.items() if s["up"]],
        down=[name for name, s in services.items() if not s["up"]],
        services=services,
    )


def probe(check: Callable[[], Any]) -> dict[str, Any]:
    """One service asked: up or not, its answer or why not, and how long it took."""
    started = time.perf_counter()
    try:
        answer, up = check(), True
    except ToolError as error:
        answer, up = error.why, False
    except Exception as error:  # a status must answer for every service, whatever broke
        answer, up = f"{type(error).__name__}: {error}", False
    return {"up": up, "detail": answer, "ms": round((time.perf_counter() - started) * 1000)}


def _goal_server(robot: Robot) -> str:
    """The goal server's ``where`` in one line."""
    where = robot.goals.where()
    pose = (
        f"pose {float(where.get('age_s', 0.0)):.2f} s old"
        if where.get("pose") == "tf"
        else "NO pose on the map"
    )
    return f"{pose}, planner {where.get('planner')}, lidar {where.get('lidar')}"
