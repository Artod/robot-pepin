"""The gaze arbiter's HTTP door (pepin.gaze_link): routes, bodies, errors, on a real socket."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pepin.gaze import IDLE
from pepin.gaze_link import IDLE_BAND, GazeClient, JsonDoor, ask

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


def test_a_source_s_client_names_itself_on_every_call() -> None:
    seen: list[tuple[str, dict[str, Any]]] = []

    def route(path: str) -> Any:
        def answer(body: dict[str, Any]) -> dict[str, Any]:
            seen.append((path, body))
            return {"status": "granted", "renewed": 1, "released": 2, "phase": "still"}

        return answer

    routes = {
        ("POST", "/look"): route("/look"),
        ("POST", "/renew"): route("/renew"),
        ("POST", "/release"): route("/release"),
        ("GET", "/state"): route("/state"),
    }
    server = JsonDoor("127.0.0.1", 0, routes).start()
    try:
        client = GazeClient("voice.gesture", f"http://127.0.0.1:{server.port}")
        key = {"at_s": 0.0, "pan_rad": 0.0, "tilt_rad": 0.05, "speed_deg_s": 90.0}
        assert client.clip([key], ttl_s=2.0)["status"] == "granted"
        assert client.renew() == 1 and client.release(fade_s=0.35) == 2
        assert client.state()["phase"] == "still"
    finally:
        server.close()
    look, renew, release, _state = seen
    assert look[1]["source"] == "voice.gesture" and look[1]["kind"] == "clip"
    assert look[1]["band"] == IDLE == IDLE_BAND
    assert look[1]["target"] == {"keys": [key], "loop": False, "idle": False}
    assert renew[1] == {"source": "voice.gesture"}
    assert release[1] == {"source": "voice.gesture", "fade_s": 0.35}
