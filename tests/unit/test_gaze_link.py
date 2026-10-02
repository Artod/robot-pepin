"""The gaze arbiter's HTTP door (pepin.gaze_link): routes, bodies, errors, on a real socket."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pepin.gaze_link import JsonDoor, ask

pytestmark = pytest.mark.slow  # a real socket on the loopback


@pytest.fixture
def door() -> Iterator[str]:
    seen: list[dict[str, Any]] = []

    def look(body: dict[str, Any]) -> dict[str, Any]:
        seen.append(body)
        return {"status": "done", "source": body.get("source")}

    server = JsonDoor(
        "127.0.0.1", 0, {("POST", "/look"): look, ("GET", "/state"): lambda _b: {"phase": "home"}}
    ).start()
    yield f"http://127.0.0.1:{server.port}"
    server.close()


def test_a_post_and_a_get_are_answered_by_their_routes(door: str) -> None:
    assert ask(door, "/look", {"source": "llm.look"}) == {"status": "done", "source": "llm.look"}
    assert ask(door, "/state") == {"phase": "home"}


def test_an_unknown_route_is_an_answer_too(door: str) -> None:
    assert "no POST /nothing" in ask(door, "/nothing", {})["error"]


def test_no_door_is_an_os_error() -> None:
    with pytest.raises(OSError):
        ask("http://127.0.0.1:9", "/state", timeout_s=0.5)
