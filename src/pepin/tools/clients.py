"""The owners' sockets as the tools see them: one small client per service, each behind a Protocol.

The tools hold no truth of their own (``.claude/research/brain/CONTRACTS.md``): where the robot
is and the places are the goal server's (TCP 3337, :mod:`pepin.goal_link`), the head is the gaze
arbiter's (HTTP 3339 on this Mac, :mod:`pepin.gaze_link`), what is where is ``world``'s (HTTP
8798), the voice is ``pepin.audio_server``'s (TCP 3338) and the picture is the camera's
(ustreamer, HTTP 8080).
Each client here speaks one of them and nothing else; none of them can send the wheels a twist.

Every Protocol is what a tool relies on, so a fake stands in for a service in the tests
(:mod:`pepin.tools.fakes`) and for one that does not exist yet: ``world`` and the audio server
are written against their contracts, and the tools work the day those services answer. A client
that cannot reach its service raises :class:`ServiceDownError`, whose words reach the model.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pepin import goal_link
from pepin.gaze_link import GAZE_PORT, ask
from pepin.neck import NeckConfig, angle_limits
from pepin.tools.registry import Image, ToolError

logger = logging.getLogger(__name__)

AUDIO_PORT = 3338  # pepin.audio_link.AUDIO_PORT (branch mic-array)
CAMERA_PORT = 8080  # ustreamer on the board: config/robot.json ports.camera_http
WORLD_URL = "http://127.0.0.1:8798"  # CONTRACTS.md: world, a host process on the Mac
GAZE_URL = f"http://127.0.0.1:{GAZE_PORT}"  # the gaze arbiter's door, beside Nav2 on this Mac
# A look is answered when the head has settled, or denied, within the person band's TTL (10 s);
# the answer is waited a little longer than that.
LOOK_ANSWER_S = 15.0
HTTP_TIMEOUT_S = 3.0


class ServiceDownError(ToolError):
    """A service did not answer: which one, where it was looked for, and what to check."""

    def __init__(self, service: str, where: str, detail: object, hint: str) -> None:
        """``hint``: what a person would check first."""
        super().__init__(f"the {service} ({where}) is not answering: {detail}. {hint}")
        self.service = service


# -- the goal server: driving, where the robot is, the places -----------------------------------


class GoalServer(Protocol):
    """The goal server's socket (pepin_bringup.goal_server): it owns every drive."""

    def where(self) -> dict[str, Any]:
        """The ``where`` answer: ``pose`` ("tf" or "none"), x, y, yaw_deg, age_s, planner."""
        ...

    def places(self) -> dict[str, dict[str, float]]:
        """The book of named places: ``{name: {"x", "y", "yaw_deg"}}`` in the map frame."""
        ...

    def go(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Send one ``go`` request and yield its events (accepted, feedback, lost, done, error)
        until the server closes the connection."""
        ...

    def cancel(self) -> dict[str, Any]:
        """Cancel every goal on the robot; the ``cancelled`` answer."""
        ...


class GoalServerLink:
    """:class:`GoalServer` over TCP, through :mod:`pepin.goal_link`: one connection per command."""

    HINT = "Is the navigation stack up (ros/laptop.sh start, or pepin-ros on the board)?"

    def __init__(self, host: str = "127.0.0.1", port: int = goal_link.PORT) -> None:
        """The goal server at ``host:port``; nothing connects until a command is sent."""
        self.host, self.port = host, port

    def where(self) -> dict[str, Any]:
        """The ``where`` answer."""
        return self._ask({"cmd": "where"})

    def places(self) -> dict[str, dict[str, float]]:
        """The ``places`` answer's book."""
        book = self._ask({"cmd": "places"}).get("places")
        return dict(book) if isinstance(book, dict) else {}

    def go(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """The drive's events; a lost connection mid-drive raises ``OSError`` to the caller."""
        try:
            yield from goal_link.events(
                request, self.host, self.port, goal_link.ASK_TIMEOUT_S, goal_link.GOAL_SILENCE_S
            )
        except goal_link.GoalServerUnreachableError as error:
            raise self._down(error) from error

    def cancel(self) -> dict[str, Any]:
        """The ``cancelled`` answer, waited for as long as the server may take to confirm it."""
        return self._ask({"cmd": "cancel"}, timeout_s=goal_link.CANCEL_CONFIRM_S + 3.0)

    def _ask(
        self, request: dict[str, Any], timeout_s: float = goal_link.ASK_TIMEOUT_S
    ) -> dict[str, Any]:
        try:
            return goal_link.ask(request, self.host, self.port, timeout_s)
        except goal_link.GoalServerUnreachableError as error:
            raise self._down(error) from error

    def _down(self, error: Exception) -> ServiceDownError:
        return ServiceDownError("goal server", f"{self.host}:{self.port}", error, self.HINT)


# -- the head: the base server's neck commands ---------------------------------------------------


@dataclass(frozen=True)
class HeadPose:
    """Where the head points: ``pan_deg`` left of the nose (+) or right (-), ``tilt_deg`` below
    level (+) or above it (-)."""

    pan_deg: float
    tilt_deg: float

    def as_dict(self) -> dict[str, float]:
        """The pose as a tool's payload, rounded to a tenth of a degree."""
        return {"pan_deg": round(self.pan_deg, 1), "tilt_deg": round(self.tilt_deg, 1)}


@dataclass(frozen=True)
class HeadReach:
    """How far the head turns: pan from ``pan_right_deg`` (negative) to ``pan_left_deg``, tilt
    from ``tilt_up_deg`` (negative, above level) to ``tilt_down_deg``."""

    pan_right_deg: float
    pan_left_deg: float
    tilt_up_deg: float
    tilt_down_deg: float

    def refusal(self, pan_deg: float | None, tilt_deg: float | None) -> str | None:
        """Why the head cannot point there, in words; None when it can."""
        if pan_deg is not None and not self.pan_right_deg <= pan_deg <= self.pan_left_deg:
            return (
                f"pan {pan_deg:+.0f} deg is past the neck's reach, which is"
                f" {self.pan_left_deg:.0f} deg left to {-self.pan_right_deg:.0f} deg right;"
                " to look further round, the robot itself has to turn"
            )
        if tilt_deg is not None and not self.tilt_up_deg <= tilt_deg <= self.tilt_down_deg:
            return (
                f"tilt {tilt_deg:+.0f} deg is past the neck's reach, which is"
                f" {-self.tilt_up_deg:.0f} deg up to {self.tilt_down_deg:.0f} deg down"
            )
        return None


@dataclass(frozen=True)
class HeadMove:
    """How a head move ended: whether it arrived, where the head is, and why not in the gaze
    arbiter's words (empty when it arrived)."""

    reached: bool
    pose: HeadPose | None
    why: str = ""
    ms: float = 0.0


class Neck(Protocol):
    """The head, in degrees, as the gaze arbiter lends it: a turn holds the head for a while
    (:meth:`keep` extends it), then the arbiter brings it home by itself."""

    def reach(self) -> HeadReach:
        """How far the head turns."""
        ...

    def rest(self) -> HeadPose:
        """The pose the camera mount was measured in: the head's working pose for driving."""
        ...

    def pose(self) -> HeadPose:
        """Where the head points now, from the encoders."""
        ...

    def turn(self, pan_deg: float | None, tilt_deg: float | None) -> HeadMove:
        """Point the head (None leaves that axis where it is) and wait until it arrives or the
        arbiter says why not."""
        ...

    def keep(self) -> None:
        """Keep the head where the last turn put it a while longer (``see``)."""
        ...


class GazeNeck:
    """:class:`Neck` through the gaze arbiter's door (:mod:`pepin.gaze_link`): every turn is a
    request of the person's band (``llm.look``) that the arbiter grants, holds behind a more
    urgent one (a stall look), or denies in words — during a drive, past the reach. A turn holds
    the head for the band's TTL (10 s) and :meth:`keep` restarts it; then the head goes home by
    itself, so no tool turns it back. It sends requests, nothing else: never a neck command and
    never a twist."""

    SOURCE = "llm.look"  # one source for every tool: a new turn replaces the last one
    HINT = "Is Nav2 up on this Mac (ros/laptop.sh nav)? The gaze arbiter runs beside it."

    def __init__(self, url: str = GAZE_URL, config: NeckConfig | None = None) -> None:
        """The arbiter's door at ``url``; ``config`` is config/neck.json (read when None)."""
        self.url = url
        self._cfg = config

    @property
    def cfg(self) -> NeckConfig:
        """config/neck.json, read on first use."""
        if self._cfg is None:
            from pepin.deployment import config_file

            self._cfg = NeckConfig.from_json(config_file("neck.json"))
        return self._cfg

    def reach(self) -> HeadReach:
        """The configured tick limits, in degrees."""
        try:
            (pan_lo, pan_hi), (tilt_lo, tilt_hi) = angle_limits(self.cfg)
        except ValueError as error:
            raise ToolError(f"the head cannot be pointed: {error}") from error
        return HeadReach(_deg(pan_lo), _deg(pan_hi), _deg(tilt_lo), _deg(tilt_hi))

    def rest(self) -> HeadPose:
        """Pan 0, the measured mount's pitch."""
        return HeadPose(0.0, self.cfg.reference.pitch_deg)

    def pose(self) -> HeadPose:
        """The encoders' newest reading, as the arbiter heard it, in degrees."""
        state = self._ask("/state", None, HTTP_TIMEOUT_S)
        pan, tilt = state.get("pan_rad"), state.get("tilt_rad")
        if pan is None or tilt is None:
            raise ToolError("the neck's encoders have not been heard (is the board's neck up?)")
        return HeadPose(_deg(float(pan)), _deg(float(tilt)))

    def turn(self, pan_deg: float | None, tilt_deg: float | None) -> HeadMove:
        """A look at those angles, held for the person band's TTL; an axis left None stays
        where the encoders say it is."""
        if pan_deg is None or tilt_deg is None:
            here = self.pose()
            pan_deg = here.pan_deg if pan_deg is None else pan_deg
            tilt_deg = here.tilt_deg if tilt_deg is None else tilt_deg
        request = {
            "source": self.SOURCE,
            "kind": "angles",
            "band": 2,  # the person's word (pepin.gaze.PERSON)
            "target": {"pan_rad": _rad(pan_deg), "tilt_rad": _rad(tilt_deg)},
            "frames": 0,  # a picture is ustreamer's, not the volume's: settled is enough
            "hold": True,
        }
        answer = self._ask("/look", request, LOOK_ANSWER_S)
        pan, tilt = answer.get("pan_rad"), answer.get("tilt_rad")
        pose = (
            HeadPose(_deg(float(pan)), _deg(float(tilt)))
            if pan is not None and tilt is not None
            else None
        )
        reached = answer.get("status") == "done" and bool(answer.get("reached"))
        why = "" if reached else str(answer.get("reason") or answer.get("status", ""))
        return HeadMove(reached, pose, why, float(answer.get("took_ms", 0.0)))

    def keep(self) -> None:
        """Restart the last turn's TTL (``/renew``)."""
        self._ask("/renew", {"source": self.SOURCE}, HTTP_TIMEOUT_S)

    def _ask(self, path: str, body: dict[str, Any] | None, wait_s: float) -> dict[str, Any]:
        try:
            return ask(self.url, path, body, wait_s)
        except (OSError, ValueError) as error:
            raise ServiceDownError("gaze arbiter", self.url, error, self.HINT) from error


_deg, _rad = math.degrees, math.radians


# -- world: the memory of what is where ----------------------------------------------------------


class World(Protocol):
    """The ``world`` service (CONTRACTS.md): sightings, merged objects, the places tree."""

    def latest(self, label: str | None, within_s: float) -> list[dict[str, Any]]:
        """Raw sightings of the last ``within_s`` seconds (what is in view NOW), of ``label``
        or of anything: label, score, x, y, z (map), range, bearing (rad, +left), size_m."""
        ...

    def objects(self, label: str | None = None) -> list[dict[str, Any]]:
        """Merged objects: label, x, y, z, size, zone, relations, first/last seen, count,
        confidence."""
        ...

    def tree(self) -> dict[str, Any]:
        """The hierarchy for the model: ``{"text": ..., "json": ...}``."""
        ...

    def remember(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Store a name a person gave: ``{"what": "zone"|"object", "name", "x","y" or "id"}``."""
        ...

    def health(self) -> dict[str, Any]:
        """The service's own health answer."""
        ...


class WorldHttp:
    """:class:`World` over HTTP (CONTRACTS.md, ``world``); proxies bypassed, JSON both ways."""

    HINT = "The memory service (world, branch world) is not running."

    def __init__(self, url: str = WORLD_URL, timeout_s: float = HTTP_TIMEOUT_S) -> None:
        """``world`` at ``url``."""
        self.url, self.timeout_s = url.rstrip("/"), timeout_s

    def latest(self, label: str | None, within_s: float) -> list[dict[str, Any]]:
        """``GET /latest``."""
        answer = self._call("GET", "/latest", {"label": label or "", "within_s": within_s})
        return _rows(answer, "sightings")

    def objects(self, label: str | None = None) -> list[dict[str, Any]]:
        """``GET /objects``."""
        return _rows(self._call("GET", "/objects", {"label": label or ""}), "objects")

    def tree(self) -> dict[str, Any]:
        """``GET /tree``."""
        answer = self._call("GET", "/tree")
        return answer if isinstance(answer, dict) else {"text": str(answer)}

    def remember(self, entry: dict[str, Any]) -> dict[str, Any]:
        """``POST /remember``."""
        answer = self._call("POST", "/remember", body=entry)
        return answer if isinstance(answer, dict) else {"answer": answer}

    def health(self) -> dict[str, Any]:
        """``GET /health``."""
        answer = self._call("GET", "/health")
        return answer if isinstance(answer, dict) else {"answer": answer}

    def _call(
        self,
        method: str,
        path: str,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        url = self.url + path + ("?" + urllib.parse.urlencode(query) if query else "")
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        return json.loads(_fetch(request, self.timeout_s, "world", self.url, self.HINT) or b"null")


def _rows(answer: Any, key: str) -> list[dict[str, Any]]:
    """A list answer, or the list under ``key`` of an object answer (the contract names the
    fields of a row, not the envelope)."""
    rows = answer.get(key, []) if isinstance(answer, dict) else answer
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # LAN and loopback only


def _fetch(
    request: urllib.request.Request, timeout_s: float, who: str, where: str, hint: str
) -> bytes:
    """The body of one HTTP exchange: :class:`ToolError` for a refusal (4xx/5xx, in the
    service's words), :class:`ServiceDownError` when nobody answered."""
    try:
        with _OPENER.open(request, timeout=timeout_s) as response:
            body: bytes = response.read()
            return body
    except urllib.error.HTTPError as error:
        said = error.read()[:300].decode(errors="replace")
        raise ToolError(
            f"the {who} refused {request.full_url}: HTTP {error.code} {said}"
        ) from error
    except (urllib.error.URLError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise ServiceDownError(who, where, reason, hint) from error


# -- the camera: one picture ---------------------------------------------------------------------


class Camera(Protocol):
    """The head camera: one picture on demand."""

    def snapshot(self) -> Image:
        """The newest frame as a JPEG: for a stereo head, the left eye, upright."""
        ...

    def health(self) -> dict[str, Any]:
        """Whether the camera streams, and at what rate."""
        ...


class UstreamerCamera:
    """:class:`Camera` from ustreamer on the board (``/snapshot``, ``/state``). A stereo head
    (config/camera.json's active rig) arrives as one side-by-side frame; the left eye is cut out
    by :class:`pepin.stereo.SideBySide`, as the laptop's camera node cuts it."""

    HINT = "Is the board up and pepin-camera (ustreamer) running on it?"

    def __init__(
        self, host: str, port: int = CAMERA_PORT, timeout_s: float = HTTP_TIMEOUT_S
    ) -> None:
        """ustreamer at ``host:port``."""
        self.base = f"http://{host}:{port}"
        self.timeout_s = timeout_s

    def snapshot(self) -> Image:
        """``GET /snapshot``, cut to one eye when the head is a stereo rig."""
        data = _fetch(
            urllib.request.Request(self.base + "/snapshot"),
            self.timeout_s,
            "camera",
            self.base,
            self.HINT,
        )
        return one_eye(data, _upside_down_stereo())

    def health(self) -> dict[str, Any]:
        """``GET /state``: ustreamer's own report."""
        body = _fetch(
            urllib.request.Request(self.base + "/state"),
            self.timeout_s,
            "camera",
            self.base,
            self.HINT,
        )
        state = json.loads(body or b"{}")
        source = state.get("result", {}).get("source", {}) if isinstance(state, dict) else {}
        return {
            "online": source.get("online"),
            "fps": source.get("captured_fps"),
            "resolution": source.get("resolution"),
        }


def _upside_down_stereo() -> bool | None:
    """None for a one-eyed head, else whether the stereo module is mounted upside down (the
    active rig of config/camera.json)."""
    from pepin.camera import active_camera
    from pepin.deployment import config_file

    data = json.loads(config_file("camera.json").read_text())
    rig = data[active_camera(data)].get("rig")
    return bool(rig.get("upside_down", False)) if isinstance(rig, dict) else None


def one_eye(jpeg: bytes, upside_down: bool | None) -> Image:
    """A frame as the :class:`Image` a model sees: as it came for a one-eyed head
    (``upside_down`` None), the left eye of a side-by-side stereo frame otherwise."""
    import cv2
    import numpy as np

    from pepin.stereo import SideBySide

    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ToolError(f"the camera sent {len(jpeg)} bytes that are not a picture")
    if upside_down is None:
        return Image(jpeg, "image/jpeg", int(frame.shape[1]), int(frame.shape[0]))
    left, _ = SideBySide(upside_down=upside_down).eyes(frame)
    encoded, buffer = cv2.imencode(".jpg", left, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not encoded:
        raise ToolError("the left eye could not be encoded as JPEG")
    return Image(buffer.tobytes(), "image/jpeg", int(left.shape[1]), int(left.shape[0]))


# -- the voice: text to speech through the board's audio server ----------------------------------


class Speech(Protocol):
    """The robot's voice."""

    def say(self, text: str) -> float:
        """Speak ``text`` through the robot's speaker; the seconds of speech, once played."""
        ...

    def health(self) -> dict[str, Any]:
        """The audio server's status answer."""
        ...


class Synthesizer(Protocol):
    """Text into speech: s16le mono PCM at the rate asked."""

    def pcm(self, text: str, rate: int) -> bytes:
        """``text`` spoken, as raw 16-bit mono samples at ``rate`` Hz."""
        ...


class AudioLink(Protocol):
    """One connection to the board's audio server (:class:`pepin.audio_link.AudioClient`)."""

    @property
    def play_rate(self) -> int:
        """The sample rate the board plays at."""
        ...

    def speak(self, pcm: bytes) -> float:
        """Play one utterance at real time; the seconds of audio sent."""
        ...

    def status(self) -> dict[str, Any] | None:
        """The server's status line, or None when it did not come."""
        ...

    def close(self) -> None:
        """Drop the connection."""
        ...


class MacSay:
    """A STAND-IN voice until the voice stack is chosen: macOS ``say`` rendered to PCM. Text
    with Cyrillic letters is read by the Russian voice (Milena) when it is installed."""

    RUSSIAN_VOICE = "Milena"

    def pcm(self, text: str, rate: int) -> bytes:
        """``say -o`` into a WAV at ``rate``, read back as raw samples."""
        if shutil.which("say") is None:
            raise ToolError("no speech synthesizer here: macOS `say` is missing")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "say.wav"
            command = ["say", "-o", str(path), "--file-format=WAVE", f"--data-format=LEI16@{rate}"]
            voice = ["-v", self.RUSSIAN_VOICE] if re.search("[\u0400-\u04ff]", text) else []
            done = subprocess.run([*command, *voice, "--", text], capture_output=True, timeout=60)
            if done.returncode != 0 and voice:  # the voice is not installed: the default one
                done = subprocess.run([*command, "--", text], capture_output=True, timeout=60)
            if done.returncode != 0:
                raise ToolError(f"`say` failed: {done.stderr.decode(errors='replace')[:200]}")
            with wave.open(str(path)) as audio:
                return bytes(audio.readframes(audio.getnframes()))


class BoardSpeech:
    """:class:`Speech` through ``pepin.audio_server`` on the board (branch mic-array): a
    connection per utterance, the text synthesised here at the board's play rate and sent at
    real time with a second's lead (``pepin.audio_link.play_paced``)."""

    HINT = "Is the board up and pepin-audio running on it?"

    def __init__(
        self,
        host: str,
        port: int = AUDIO_PORT,
        synthesizer: Synthesizer | None = None,
        link: Callable[[str, int], AudioLink] | None = None,
    ) -> None:
        """``link`` opens one connection (the audio link's client by default)."""
        self.host, self.port = host, port
        self._synthesizer = synthesizer or MacSay()
        self._link = link or audio_client_link

    def say(self, text: str) -> float:
        """Synthesise at the board's rate, play, close."""
        link = self._open()
        try:
            return link.speak(self._synthesizer.pcm(text, link.play_rate))
        finally:
            link.close()

    def health(self) -> dict[str, Any]:
        """One status line."""
        link = self._open()
        try:
            return link.status() or {"status": "no answer"}
        finally:
            link.close()

    def _open(self) -> AudioLink:
        try:
            return self._link(self.host, self.port)
        except OSError as error:
            raise ServiceDownError(
                "audio server", f"{self.host}:{self.port}", error, self.HINT
            ) from error


class _AudioClientLink:
    """:class:`AudioLink` over the audio link's own client and pacer (``Any``: the module is
    imported by name, since it lives on branch mic-array until that is merged)."""

    def __init__(self, client: Any, pacer: Callable[..., float]) -> None:
        self._client, self._pacer = client, pacer

    @property
    def play_rate(self) -> int:
        return int(self._client.play_rate)

    def speak(self, pcm: bytes) -> float:
        return float(self._pacer(self._client, pcm, rate=self.play_rate))

    def status(self) -> dict[str, Any] | None:
        answer = self._client.status()
        return dict(answer) if isinstance(answer, dict) else None

    def close(self) -> None:
        self._client.close()


def audio_client_link(host: str, port: int) -> AudioLink:
    """A started ``pepin.audio_link.AudioClient`` (not listening: the voice only speaks);
    :class:`ToolError` when this build has no audio link yet."""
    try:
        module = importlib.import_module("pepin.audio_link")
    except ImportError as error:
        raise ToolError(
            "the robot has no voice in this build: pepin.audio_link (branch mic-array) is not"
            " merged yet"
        ) from error
    client = module.AudioClient(host, port).start(listen=False)
    return _AudioClientLink(client, module.play_paced)


# -- the face: the head server's mouth and info screen ------------------------------------------

HEAD_PORT = 3340  # pepin.head_link.HEAD_PORT
BASE_PORT = 3336  # pepin.base_link.BASE_PORT


class Face(Protocol):
    """The mouth on the head's screen and the info screen beside it (the head server)."""

    def express(self, name: str, seconds: float) -> dict[str, Any]:
        """Show expression ``name`` for ``seconds``; the head server's ack (what shows now)."""
        ...

    def show(self, text: str, seconds: float) -> dict[str, Any]:
        """Show ``text`` (pepin.head_link.show_items markup) for ``seconds``; the ack."""
        ...

    def health(self) -> dict[str, Any]:
        """The head server's link, what it shows, the face's frame rate."""
        ...


class HeadFace:
    """:class:`Face` through the board's head server (``pepin.head_server``, TCP 3340): one
    connection per call, its answer awaited. The tools' expressions are the ``llm`` source's and
    always timed, so the face goes back by itself to whatever stood before (a drive's focus, the
    voice loop's listening)."""

    SOURCE = "llm"
    HINT = "Is the board up and pepin-head running on it, with the head's ESP32 plugged in?"

    def __init__(self, host: str, port: int = HEAD_PORT, timeout_s: float = HTTP_TIMEOUT_S) -> None:
        """The head server at ``host:port``."""
        self.host, self.port, self.timeout_s = host, port, timeout_s

    def express(self, name: str, seconds: float) -> dict[str, Any]:
        """An ``express`` of the ``llm`` source, held ``seconds``."""
        request = {"cmd": "express", "source": self.SOURCE, "name": name, "hold_s": seconds}
        return self._ask(request)

    def show(self, text: str, seconds: float) -> dict[str, Any]:
        """A ``show`` of ``text``."""
        return self._ask({"cmd": "show", "text": text, "seconds": seconds})

    def health(self) -> dict[str, Any]:
        """The status line, cut to what a person checks first."""
        status = self._ask({"cmd": "status"})
        esp = status.get("esp") or {}
        return {
            "link": status.get("link"),
            "showing": status.get("showing"),
            "face_fps": esp.get("fps"),
            "imu_hz": esp.get("imu_rate_hz"),
        }

    def _ask(self, request: dict[str, Any]) -> dict[str, Any]:
        from pepin.head_link import ask as ask_head

        where = f"{self.host}:{self.port}"
        try:
            answer = ask_head(request, self.host, self.port, self.timeout_s)
        except (OSError, ValueError) as error:
            raise ServiceDownError("head server", where, error, self.HINT) from error
        if answer.get("type") == "error":
            raise ToolError(f"the head server refused it: {answer.get('error')}")
        return answer


class Body(Protocol):
    """The base's own readings beyond where it is."""

    def temperatures(self) -> dict[str, int]:
        """Each servo's temperature, degrees C, by name."""
        ...


class BaseBody:
    """:class:`Body` from the base server's state lines (``pepin.base_server``, TCP 3336): it
    connects, reads until a state line carries the temperatures, and leaves. It sends nothing,
    so it is never a driver and its leaving releases nothing."""

    HINT = "Is the board up and pepin-base running on it?"

    def __init__(self, host: str, port: int = BASE_PORT, timeout_s: float = HTTP_TIMEOUT_S) -> None:
        """The base server at ``host:port``."""
        self.host, self.port, self.timeout_s = host, port, timeout_s

    def temperatures(self) -> dict[str, int]:
        """The newest state line's ``temp_c`` (the servos are read every 5 s)."""
        import socket
        import time

        where = f"{self.host}:{self.port}"
        heard = False  # a state line came: the server is up, the temperatures are not
        try:
            with socket.create_connection((self.host, self.port), self.timeout_s) as sock:
                deadline = time.monotonic() + self.timeout_s
                buffer = b""
                while (left := deadline - time.monotonic()) > 0:
                    sock.settimeout(left)
                    try:
                        chunk = sock.recv(65536)
                    except TimeoutError:
                        break
                    if not chunk:
                        break
                    buffer += chunk
                    *lines, buffer = buffer.split(b"\n")
                    for line in reversed(lines):
                        state = json.loads(line) if line.strip() else {}
                        heard = heard or state.get("type") == "state"
                        if state.get("type") == "state" and isinstance(state.get("temp_c"), dict):
                            return {str(k): int(v) for k, v in state["temp_c"].items()}
        except (OSError, ValueError) as error:
            raise ServiceDownError("base server", where, error, self.HINT) from error
        if not heard:
            raise ServiceDownError("base server", where, "no state line", self.HINT)
        raise ToolError(
            "the base server has not read the servos' temperatures yet (it does every 5 s, while"
            " the bus answers); ask again in a few seconds"
        )
