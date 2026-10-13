"""The perception service as its consumers see it: records, the ring's JPEGs, embeddings.

:class:`Perception` is what the speaker gaze, person recognition and the ``see`` tool rely on;
:class:`PerceptionClient` speaks it over HTTP to the service on this Mac, and
:class:`pepin.perception.fakes.FakePerception` stands in for it in the tests. Every call answers
``None`` for any failure and never raises (the caller decides what a missing frame means); a
connection the service refused leaves it alone for ``retry_s`` (calls answer ``None`` at once,
:attr:`PerceptionClient.down` says so), so a gaze looking ten times a second never pays a
connect per look while the service is down. Each thread keeps its own kept-alive connection, so
a long poll on one never holds another's call.

:class:`HealthWatch` reads ``/health`` every few seconds on its own thread, for the voice loop's
vision cell (:func:`vision_cell`): the loop's own threads never wait on the network for it.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pepin.face_sensor.models import Detection
from pepin.perception import URL
from pepin.perception.record import Record

Vector = NDArray[np.float32]
RETRY_S = 1.0
TIMEOUT_MARGIN_S = 1.0  # beyond a long poll's own wait, before the socket gives up
HEALTH_EVERY_S = 2.0
STALE_FRAMES_S = 2.0  # no frame for this long: the cell says so (camera_stream's feed is off)
DOWN_HINT = "ros/models.sh start perception"


@dataclass(frozen=True)
class Jpeg:
    """One frame of the ring as JPEG bytes: its number, the board's exposure stamp, its size,
    the optics' name, and whether it is the still frame asked for (exposed after the stamp
    given) or only the newest one there was."""

    data: bytes
    seq: int
    stamp: float
    width: int
    height: int
    optics: str
    still: bool = True


class Perception(Protocol):
    """The perception service (see the module)."""

    url: str

    @property
    def down(self) -> bool:
        """Whether calls are being skipped because the service did not answer lately."""
        ...

    def latest(
        self, since: int = 0, min_stamp: float | None = None, timeout_s: float = 0.0
    ) -> Record | None:
        """The newest record numbered after ``since`` and exposed at ``min_stamp`` (the board's
        clock) or later, waiting up to ``timeout_s``; None when none came or on any failure."""
        ...

    def frame(self, seq: int) -> Jpeg | None:
        """Frame ``seq`` of the ring; None when it has left the ring or on any failure."""
        ...

    def still(self, min_stamp: float | None = None, timeout_s: float = 0.0) -> Jpeg | None:
        """The newest frame exposed at ``min_stamp`` or later, waiting up to ``timeout_s`` for
        one, else the newest frame there is (``still`` False); None without any frame or on
        any failure."""
        ...

    def embed(self, seq: int, detection: Detection) -> Vector | None:
        """ArcFace's unit-length embedding of ``detection`` (box and five points) in frame
        ``seq``; None when the frame has left the ring or on any failure."""
        ...

    def health(self) -> dict[str, Any] | None:
        """``/health``: rates, milliseconds, the newest record's counts; None when down."""
        ...


class PerceptionClient:
    """:class:`Perception` over HTTP (see the module)."""

    def __init__(
        self,
        url: str = URL,
        *,
        timeout_s: float = 1.0,
        retry_s: float = RETRY_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """The service at ``url`` (``http://127.0.0.1:8792``)."""
        if not url.startswith("http://"):
            raise ValueError(f"the perception service URL starts with http://, not {url!r}")
        self.url = url.rstrip("/")
        host, _, port = self.url[len("http://") :].partition(":")
        self._host, self._port = host, int(port) if port else 80
        self.timeout_s, self.retry_s = float(timeout_s), float(retry_s)
        self._clock = clock
        self._local = threading.local()
        self._down_until = -float("inf")
        self.ok = 0
        self.failed = 0
        self.skipped = 0
        self.last_error = ""

    @property
    def down(self) -> bool:
        """Whether calls are being skipped (the service refused lately)."""
        return self._clock() < self._down_until

    def latest(
        self, since: int = 0, min_stamp: float | None = None, timeout_s: float = 0.0
    ) -> Record | None:
        """See :meth:`Perception.latest`."""
        query = {"since": str(int(since)), "timeout": f"{max(0.0, timeout_s):.3f}"}
        if min_stamp is not None:
            query["min_stamp"] = f"{min_stamp:.6f}"
        answer = self._get("/latest", query, timeout_s)
        if answer is None or answer[0] != 200:
            return None
        try:
            return Record.from_dict(json.loads(answer[2]))
        except (ValueError, KeyError, TypeError) as exc:
            self._note(f"a record that is not one: {exc}")
            return None

    def frame(self, seq: int) -> Jpeg | None:
        """See :meth:`Perception.frame`."""
        answer = self._get(f"/frame/{int(seq)}.jpg", {}, 0.0)
        return _jpeg(answer) if answer is not None and answer[0] == 200 else None

    def still(self, min_stamp: float | None = None, timeout_s: float = 0.0) -> Jpeg | None:
        """See :meth:`Perception.still`."""
        query = {"timeout": f"{max(0.0, timeout_s):.3f}"}
        if min_stamp is not None:
            query["min_stamp"] = f"{min_stamp:.6f}"
        answer = self._get("/frame/newest.jpg", query, timeout_s)
        return _jpeg(answer) if answer is not None and answer[0] == 200 else None

    def embed(self, seq: int, detection: Detection) -> Vector | None:
        """See :meth:`Perception.embed`."""
        body = json.dumps(
            {"seq": int(seq), "box": list(detection.box), "kps": [list(p) for p in detection.kps]}
        ).encode()
        answer = self._call("POST", "/embed", body, {"Content-Type": "application/json"}, 0.0)
        if answer is None or answer[0] != 200:
            return None
        try:
            values = json.loads(answer[2])["embedding"]
            return np.asarray(values, dtype=np.float32)
        except (ValueError, KeyError, TypeError) as exc:
            self._note(f"an embedding that is not one: {exc}")
            return None

    def health(self) -> dict[str, Any] | None:
        """See :meth:`Perception.health`."""
        answer = self._get("/health", {}, 0.0)
        if answer is None or answer[0] != 200:
            return None
        try:
            data = json.loads(answer[2])
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def _get(
        self, path: str, query: Mapping[str, str], wait_s: float
    ) -> tuple[int, dict[str, str], bytes] | None:
        if query:
            path += "?" + urllib.parse.urlencode(query)
        return self._call("GET", path, None, {}, wait_s)

    def _call(
        self, method: str, path: str, body: bytes | None, headers: Mapping[str, str], wait_s: float
    ) -> tuple[int, dict[str, str], bytes] | None:
        """(status, lower-cased headers, body), or None for a failure (counted)."""
        if self.down:
            self.skipped += 1
            return None
        timeout = self.timeout_s + max(0.0, wait_s)
        for attempt in (0, 1):
            conn = self._connection(timeout)
            reused = getattr(self._local, "used", False)
            try:
                conn.request(method, path, body=body, headers=dict(headers))
                response = conn.getresponse()
                data = response.read()
            except (ConnectionResetError, BrokenPipeError, http.client.BadStatusLine) as exc:
                self._drop()
                if reused and attempt == 0:
                    continue  # the service dropped the kept-alive link: once more, fresh
                self._away(exc)
                return None
            except (ConnectionRefusedError, ConnectionAbortedError) as exc:
                self._drop()
                self._away(exc)
                return None
            except (OSError, http.client.HTTPException) as exc:
                self._drop()
                self._note(f"{type(exc).__name__}: {exc}")
                return None
            self._local.used = True
            self.ok += 1
            return response.status, {k.lower(): v for k, v in response.getheaders()}, data
        return None

    def _connection(self, timeout: float) -> http.client.HTTPConnection:
        conn: http.client.HTTPConnection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = http.client.HTTPConnection(self._host, self._port, timeout=timeout)
            self._local.conn, self._local.used = conn, False
        else:
            conn.timeout = timeout
            if conn.sock is not None:
                conn.sock.settimeout(timeout)
        return conn

    def _drop(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
        self._local.conn, self._local.used = None, False

    def _away(self, exc: BaseException) -> None:
        self._note(f"{type(exc).__name__}: {exc}")
        self._down_until = self._clock() + self.retry_s

    def _note(self, why: str) -> None:
        self.failed += 1
        self.last_error = why[:160]


def _jpeg(answer: tuple[int, dict[str, str], bytes]) -> Jpeg | None:
    _status, headers, data = answer
    try:
        return Jpeg(
            data=data,
            seq=int(headers["x-seq"]),
            stamp=float(headers["x-stamp"]),
            width=int(headers.get("x-width", "0")),
            height=int(headers.get("x-height", "0")),
            optics=headers.get("x-optics", "").split(" ", 1)[0],
            still=headers.get("x-still", "1") == "1",
        )
    except (KeyError, ValueError):
        return None


def vision_cell(health: Mapping[str, Any] | None, url: str = URL) -> str:
    """The voice loop's vision cell from a ``/health`` answer (None: down)."""
    if health is None:
        return f"vision DOWN ({url.removeprefix('http://')}: {DOWN_HINT})"
    frames = health.get("frames", {})
    detect = health.get("detect", {})
    latest = health.get("latest") or {}
    idle = frames.get("idle_s")
    if idle is None:
        return "vision up, no frame yet (camera_stream's perception_feed?)"
    if float(idle) > STALE_FRAMES_S:
        return f"vision up, no frame for {float(idle):.0f} s (camera_stream's perception_feed?)"
    ms = detect.get("ms", {}).get("total", {})
    faces, persons = int(latest.get("faces", 0)), int(latest.get("persons", 0))
    return (
        f"vision {float(frames.get('per_s', 0.0)):.1f} in / {float(detect.get('per_s', 0.0)):.1f}"
        f" detected per s, detect {float(ms.get('median', 0.0)):.0f} ms,"
        f" {faces} face{'s' if faces != 1 else ''} {persons} person{'s' if persons != 1 else ''}"
    )


class HealthWatch:
    """``/health`` every ``every_s`` on a daemon thread; :meth:`cell` reads the last answer."""

    def __init__(self, service: Perception, every_s: float = HEALTH_EVERY_S) -> None:
        self.service, self.every_s = service, float(every_s)
        self._last: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="vision-health", daemon=True)

    def start(self) -> HealthWatch:
        """Read once now, then start the thread; returns self."""
        self._last = self.service.health()
        self._thread.start()
        return self

    @property
    def last(self) -> dict[str, Any] | None:
        """The last ``/health`` answer (None: down or not read yet)."""
        return self._last

    def cell(self) -> str:
        """The vision cell of the last answer."""
        return vision_cell(self._last, self.service.url)

    def close(self) -> None:
        """Stop the thread."""
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.every_s):
            self._last = self.service.health()
