"""pepin.tools.clients against fake owners on local sockets: the goal server's JSON lines, the
gaze arbiter's door, world's and the camera's HTTP, and the audio link through an injected
connection. Real sockets, hence ``slow``."""

from __future__ import annotations

import contextlib
import json
import math
import shutil
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import cv2
import numpy as np
import pytest

from pepin.gaze_link import JsonDoor
from pepin.neck import NeckConfig
from pepin.tools import clients
from pepin.tools.clients import (
    BoardSpeech,
    GazeNeck,
    GoalServerLink,
    MacSay,
    ServiceDownError,
    UstreamerCamera,
    WorldHttp,
    audio_client_link,
    one_eye,
)
from pepin.tools.registry import ToolError

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
NECK = NeckConfig.from_json(REPO / "config/neck.json")


def closed_port() -> int:
    """A local port nobody listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class LineServer:
    """A JSON-lines TCP server: each connection handled by ``serve(request_lines, send)``."""

    def __init__(self) -> None:
        self.asked: list[dict[str, Any]] = []
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self.port = int(self._listener.getsockname()[1])
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                connection, _ = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._session, args=(connection,), daemon=True).start()

    def _session(self, connection: socket.socket) -> None:
        with connection, connection.makefile("r") as lines, contextlib.suppress(OSError):
            self.serve(lines, lambda m: connection.sendall((json.dumps(m) + "\n").encode()))

    def serve(self, lines: Any, send: Any) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self._listener.close()


class GoalServerFake(LineServer):
    """One request per connection; ``go`` streams a short drive."""

    def serve(self, lines: Any, send: Any) -> None:
        request = json.loads(lines.readline())
        self.asked.append(request)
        cmd = request["cmd"]
        if cmd == "where":
            send({"event": "where", "pose": "tf", "x": 1.0, "y": 2.0, "yaw_deg": 90.0})
        elif cmd == "places":
            send({"event": "places", "places": {"home": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}}})
        elif cmd == "cancel":
            send({"event": "cancelled", "had_goal": False, "navigators": {}})
        elif cmd == "go":
            send({"event": "accepted", "x": 0.0, "y": 0.0, "yaw_deg": 0.0})
            send({"event": "done", "status": 4, "seconds": 1.0})


@pytest.fixture
def goal_server() -> Iterator[GoalServerFake]:
    server = GoalServerFake()
    yield server
    server.close()


def test_the_goal_server_link_speaks_the_four_commands(goal_server: GoalServerFake) -> None:
    link = GoalServerLink("127.0.0.1", goal_server.port)
    assert link.where()["y"] == 2.0
    assert link.places() == {"home": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}}
    assert [e["event"] for e in link.go({"cmd": "go", "place": "home"})] == ["accepted", "done"]
    assert link.cancel()["event"] == "cancelled"
    assert [r["cmd"] for r in goal_server.asked] == ["where", "places", "go", "cancel"]


def test_a_goal_server_nobody_runs_is_named() -> None:
    link = GoalServerLink("127.0.0.1", closed_port())
    with pytest.raises(
        ServiceDownError, match=r"the goal server \(127.0.0.1:\d+\) is not answering"
    ):
        link.where()
    with pytest.raises(ServiceDownError):
        list(link.go({"cmd": "go", "place": "home"}))


class GazeDoorFake:
    """The gaze arbiter's door (pepin.gaze_link): /state with the head at home, /look answered
    as arrived where it was asked (or denied, when ``deny`` says why), /renew counted."""

    def __init__(self) -> None:
        self.asked: list[dict[str, Any]] = []
        self.deny = ""
        self.head = (0.0, math.radians(23.8))
        self._door = JsonDoor(
            "127.0.0.1",
            0,
            {
                ("GET", "/state"): self._state,
                ("POST", "/look"): self._look,
                ("POST", "/renew"): self._renew,
            },
        ).start()
        self.url = f"http://127.0.0.1:{self._door.port}"

    def close(self) -> None:
        self._door.close()

    def _state(self, _body: dict[str, Any]) -> dict[str, Any]:
        return {"phase": "home", "pan_rad": self.head[0], "tilt_rad": self.head[1]}

    def _look(self, body: dict[str, Any]) -> dict[str, Any]:
        self.asked.append(body)
        if self.deny:
            return {"status": "denied", "reason": self.deny, "reached": False}
        target = body["target"]
        self.head = (target["pan_rad"], target["tilt_rad"])
        return {
            "status": "done",
            "reached": True,
            "pan_rad": self.head[0],
            "tilt_rad": self.head[1],
            "took_ms": 900.0,
        }

    def _renew(self, body: dict[str, Any]) -> dict[str, Any]:
        self.asked.append(body)
        return {"renewed": 1}


@pytest.fixture
def gaze() -> Iterator[GazeDoorFake]:
    door = GazeDoorFake()
    yield door
    door.close()


def test_the_neck_asks_the_arbiter_in_radians_and_reads_its_answers(gaze: GazeDoorFake) -> None:
    head = GazeNeck(gaze.url, NECK)
    rest = head.pose()
    assert rest.pan_deg == pytest.approx(0.0) and rest.tilt_deg == pytest.approx(23.8)
    move = head.turn(45.0, None)  # the tilt stays where the encoders say
    assert move.reached and move.pose is not None and move.ms == 900.0
    assert move.pose.pan_deg == pytest.approx(45.0)
    look = gaze.asked[-1]
    assert look == {
        "source": "llm.look",
        "kind": "angles",
        "band": 2,
        "target": {
            "pan_rad": pytest.approx(math.radians(45.0)),
            "tilt_rad": pytest.approx(rest.tilt_deg * math.pi / 180),
        },
        "frames": 0,
        "hold": True,
    }
    head.keep()
    assert gaze.asked[-1] == {"source": "llm.look"}


def test_the_neck_reports_a_denial_in_the_arbiter_s_words(gaze: GazeDoorFake) -> None:
    gaze.deny = "the head does not move during a drive"
    refused = GazeNeck(gaze.url, NECK).turn(30.0, 30.0)
    assert not refused.reached and refused.why == "the head does not move during a drive"


def test_the_neck_s_reach_and_rest_come_from_its_config() -> None:
    head = GazeNeck(f"http://127.0.0.1:{closed_port()}", NECK)
    reach = head.reach()
    assert reach.pan_left_deg == pytest.approx(155.7, abs=0.5)
    assert reach.refusal(0.0, 70.0) is not None and reach.refusal(-150.0, 60.0) is None
    assert head.rest().tilt_deg == 23.8
    with pytest.raises(ServiceDownError, match="gaze arbiter"):
        head.pose()


class WorldFake(BaseHTTPRequestHandler):
    """``world``'s HTTP: /latest in an envelope, /objects as a bare list, /remember echoed."""

    asked: ClassVar[list[tuple[str, str, Any]]] = []

    def do_GET(self) -> None:
        self.asked.append(("GET", self.path, None))
        if self.path.startswith("/latest"):
            self._send(200, {"sightings": [{"label": "cup", "range": 1.0}, "junk"]})
        elif self.path.startswith("/objects"):
            self._send(200, [{"id": "o1", "label": "cup"}])
        elif self.path == "/tree":
            self._send(200, {"text": "flat", "json": {}})
        else:
            self._send(404, {"error": "no such thing"})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.asked.append(("POST", self.path, body))
        self._send(200, {"stored": body})

    def _send(self, code: int, payload: Any) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def http_server() -> Iterator[ThreadingHTTPServer]:
    WorldFake.asked = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), WorldFake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def test_world_over_http(http_server: ThreadingHTTPServer) -> None:
    world = WorldHttp(f"http://127.0.0.1:{http_server.server_address[1]}")
    assert world.latest("cup", 1.5) == [{"label": "cup", "range": 1.0}]
    assert world.objects("cup") == [{"id": "o1", "label": "cup"}]
    assert world.tree()["text"] == "flat"
    assert (
        world.remember({"what": "zone", "name": "hall", "x": 1.0, "y": 2.0})["stored"]["name"]
        == "hall"
    )
    assert WorldFake.asked[0] == ("GET", "/latest?label=cup&within_s=1.5", None)
    assert WorldFake.asked[-1] == (
        "POST",
        "/remember",
        {"what": "zone", "name": "hall", "x": 1.0, "y": 2.0},
    )
    with pytest.raises(ToolError, match="HTTP 404"):
        world.health()


def test_a_world_nobody_runs_is_named() -> None:
    with pytest.raises(ServiceDownError, match=r"world .* is not answering"):
        WorldHttp(f"http://127.0.0.1:{closed_port()}").latest(None, 1.0)


def jpeg(frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return bytes(buffer.tobytes())


def test_one_eye_of_a_stereo_frame_upright() -> None:
    """The module is taped upside down: the left eye is the right half, turned over."""
    frame = np.zeros((60, 160, 3), dtype=np.uint8)
    frame[:, 80:] = 255  # the right half of the wire frame is white
    left = one_eye(jpeg(frame), upside_down=True)
    assert (left.width, left.height) == (80, 60)
    decoded = cv2.imdecode(np.frombuffer(left.data, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert decoded.mean() > 200
    straight = one_eye(jpeg(frame), upside_down=False)
    assert cv2.imdecode(np.frombuffer(straight.data, np.uint8), cv2.IMREAD_GRAYSCALE).mean() < 50
    mono = jpeg(frame)
    assert one_eye(mono, upside_down=None).data == mono
    with pytest.raises(ToolError, match="not a picture"):
        one_eye(b"garbage", None)


def test_the_camera_over_http(monkeypatch: pytest.MonkeyPatch) -> None:
    picture = jpeg(np.full((30, 40, 3), 128, dtype=np.uint8))

    class Ustreamer(WorldFake):
        def do_GET(self) -> None:
            if self.path == "/snapshot":
                self.send_response(200)
                self.send_header("Content-Length", str(len(picture)))
                self.end_headers()
                self.wfile.write(picture)
            else:
                self._send(200, {"result": {"source": {"online": True, "captured_fps": 15}}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Ustreamer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(clients, "_upside_down_stereo", lambda: None)
    try:
        camera = UstreamerCamera("127.0.0.1", server.server_address[1])
        shot = camera.snapshot()
        assert (shot.width, shot.height, shot.data) == (40, 30, picture)
        assert camera.health() == {"online": True, "fps": 15, "resolution": None}
    finally:
        server.shutdown()
        server.server_close()


class FakeLink:
    """One audio connection that keeps what it was given."""

    def __init__(self) -> None:
        self.spoken: list[bytes] = []
        self.closed = False

    @property
    def play_rate(self) -> int:
        return 16_000

    def speak(self, pcm: bytes) -> float:
        self.spoken.append(pcm)
        return len(pcm) / 2 / self.play_rate

    def status(self) -> dict[str, Any] | None:
        return {"type": "status", "playing": False}

    def close(self) -> None:
        self.closed = True


class SilentSynthesizer:
    def pcm(self, text: str, rate: int) -> bytes:
        return b"\x00\x00" * rate * len(text)  # a second a character


def test_speech_synthesises_at_the_board_s_rate_and_closes() -> None:
    link = FakeLink()
    speech = BoardSpeech("board", synthesizer=SilentSynthesizer(), link=lambda host, port: link)
    assert speech.say("hi") == pytest.approx(2.0)
    assert link.closed and len(link.spoken[0]) == 64_000
    assert speech.health()["type"] == "status"


def test_speech_to_a_board_that_does_not_answer() -> None:
    def refused(host: str, port: int) -> FakeLink:
        raise ConnectionRefusedError("refused")

    speech = BoardSpeech("board", synthesizer=SilentSynthesizer(), link=refused)
    with pytest.raises(ServiceDownError, match="audio server"):
        speech.say("hi")


def test_the_audio_link_is_the_mic_array_branch_s_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without pepin.audio_link the voice says it is not merged; with it, its client and its
    pacer carry the utterance."""

    def missing(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(clients.importlib, "import_module", missing)
    with pytest.raises(ToolError, match="branch mic-array"):
        audio_client_link("board", 3338)

    started: list[tuple[str, int, bool]] = []

    class Client:
        play_rate = 16_000

        def __init__(self, host: str, port: int) -> None:
            self.address = (host, port)

        def start(self, *, listen: bool) -> Client:
            started.append((*self.address, listen))
            return self

        def status(self) -> dict[str, Any]:
            return {"type": "status"}

        def close(self) -> None:
            pass

    def pace(speaker: Any, pcm: bytes, *, rate: int) -> float:
        return len(pcm) / 2 / rate

    module = SimpleNamespace(AudioClient=Client, play_paced=pace)
    monkeypatch.setattr(clients.importlib, "import_module", lambda name: module)
    link = audio_client_link("board", 3338)
    assert started == [("board", 3338, False)]
    assert link.play_rate == 16_000 and link.speak(b"\x00\x00" * 8000) == 0.5
    assert link.status() == {"type": "status"}


@pytest.mark.skipif(shutil.which("say") is None, reason="macOS `say` only")
def test_mac_say_renders_mono_pcm_at_the_rate_asked() -> None:
    pcm = MacSay().pcm("hi", 16_000)
    assert len(pcm) % 2 == 0 and 0.1 < len(pcm) / 2 / 16_000 < 3.0
    assert not math.isnan(float(np.frombuffer(pcm, np.int16).std()))
