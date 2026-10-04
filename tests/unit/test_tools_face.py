"""The face's tools (pepin.tools.face) and their clients, against fakes and a fake head server."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from typing import Any

import pytest

from pepin.face import load_face_table
from pepin.tools import TOOLS
from pepin.tools.clients import BaseBody, HeadFace, ServiceDownError
from pepin.tools.face import EMOTIONS
from pepin.tools.fakes import FakeBody, FakeFace, fake_robot
from pepin.tools.registry import ToolError
from pepin.tools.schemas import gemini_function_declarations

REPO = Path(__file__).resolve().parents[2]


def call(robot: Any, name: str, **arguments: Any) -> dict[str, Any]:
    return TOOLS.call(name, arguments, robot)


def test_the_emotions_are_the_face_tables_expressions() -> None:
    """One list of expressions: config/face.json's. The tool's enum is held equal to it."""
    assert load_face_table(REPO / "config" / "face.json").names == EMOTIONS
    declaration = next(d for d in gemini_function_declarations(TOOLS) if d["name"] == "express")
    assert declaration["parameters"]["properties"]["emotion"]["enum"] == list(EMOTIONS)


def test_express_shows_an_expression_for_a_while() -> None:
    face = FakeFace()
    robot = fake_robot(face=face)
    assert call(robot, "express", emotion="happy", seconds=4) == {
        "ok": True,
        "showing": "happy",
        "seconds": 4.0,
    }
    assert call(robot, "express", emotion="sad")["seconds"] == 5.0
    assert face.expressed == [("happy", 4.0), ("sad", 5.0)]
    assert "must be one of" in call(robot, "express", emotion="smug")["why"]
    assert "between 1 and 60" in call(robot, "express", emotion="sad", seconds=600)["why"]


def test_show_puts_lines_on_the_screen() -> None:
    face = FakeFace()
    robot = fake_robot(face=face)
    result = call(robot, "show", text="Servo temperatures\nleft wheel: 41/70 C")
    assert result == {"ok": True, "items": 2, "seconds": 8.0}
    assert face.shown == [("Servo temperatures\nleft wheel: 41/70 C", 8.0)]
    assert call(robot, "show", text="  \n ")["why"] == "nothing to show: the text is empty"
    many = call(robot, "show", text="\n".join(f"line {i}" for i in range(10)))
    assert many["note"] == "only the first 8 of 10 lines fit"


def test_servo_temperatures_then_show_them() -> None:
    """'Pepin, show the servo temperatures': read them, then put them on the face."""
    robot = fake_robot(body=FakeBody({"left": 41, "right": 39, "pan": 36, "tilt": 35}))
    result = call(robot, "servo_temperatures")
    assert result == {
        "ok": True,
        "temperatures_c": {"left": 41, "right": 39, "pan": 36, "tilt": 35},
        "limit_c": 70,
        "hottest_c": 41,
    }
    assert "base server" in call(fake_robot(body=FakeBody(down=True)), "servo_temperatures")["why"]


def test_a_face_that_is_down_is_named() -> None:
    robot = fake_robot(face=FakeFace(down=True))
    assert "head server" in call(robot, "express", emotion="happy")["why"]
    assert "face" in call(robot, "status")["down"]


class LineServer:
    """A TCP server that answers each JSON line with ``answer(line)`` and can push lines first."""

    def __init__(self, answer: Any, first: list[dict[str, Any]] | None = None) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(2)
        self.port = self.sock.getsockname()[1]
        self.answer, self.first = answer, first or []
        self.got: list[dict[str, Any]] = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                for line in self.first:
                    conn.sendall((json.dumps(line) + "\n").encode())
                buffer = b""
                while b"\n" not in buffer:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                if buffer:
                    request = json.loads(buffer.split(b"\n")[0])
                    self.got.append(request)
                    conn.sendall((json.dumps(self.answer(request)) + "\n").encode())

    def close(self) -> None:
        self.sock.close()


@pytest.mark.slow
def test_head_face_speaks_the_head_servers_door() -> None:
    def answer(request: dict[str, Any]) -> dict[str, Any]:
        if request["cmd"] == "status":
            return {"type": "status", "link": "up", "showing": "grin", "esp": {"fps": 49.0}}
        if request.get("name") == "smug":
            return {"type": "error", "error": "no expression 'smug'"}
        return {"type": "ack", "cmd": request["cmd"], "showing": request.get("name")}

    server = LineServer(answer, first=[{"type": "imu", "s": []}])  # a stream line first
    try:
        face = HeadFace("127.0.0.1", server.port)
        assert face.express("grin", 3.0)["showing"] == "grin"
        assert server.got[-1] == {"cmd": "express", "source": "llm", "name": "grin", "hold_s": 3.0}
        face.show("a\nb: 1 C", 5.0)
        assert server.got[-1] == {"cmd": "show", "text": "a\nb: 1 C", "seconds": 5.0}
        assert face.health()["face_fps"] == 49.0
        with pytest.raises(ToolError, match="refused it: no expression 'smug'"):
            face.express("smug", 1.0)
    finally:
        server.close()
    with pytest.raises(ServiceDownError, match="head server"):
        HeadFace("127.0.0.1", server.port, timeout_s=0.3).express("grin", 1.0)


@pytest.mark.slow
def test_base_body_reads_the_temperatures_off_a_state_line() -> None:
    lines = [
        {"type": "state", "temp_c": None},
        {"type": "state", "x": 0.0, "temp_c": {"left": 40, "right": 38}},
    ]
    server = LineServer(lambda request: {}, first=lines)
    try:
        assert BaseBody("127.0.0.1", server.port).temperatures() == {"left": 40, "right": 38}
    finally:
        server.close()
    silent = LineServer(lambda request: {}, first=[{"type": "state", "temp_c": None}])
    try:
        with pytest.raises(ToolError, match="not read the servos' temperatures yet"):
            BaseBody("127.0.0.1", silent.port, timeout_s=0.3).temperatures()
    finally:
        silent.close()
