"""The gaze arbiter's door for processes that are not ROS nodes: JSON over HTTP on the Mac.

The arbiter (``pepin_bringup.gaze``) runs in the navigation container and answers here, on
:data:`GAZE_PORT` (published on this Mac's loopback only, as the goal server's port is):

    POST /look     a request (pepin.gaze.look_from_json's JSON), answered when it ends; a looped
                   overlay (kind clip, loop) at once
    POST /renew    {"source": ...}: restart that source's TTL (``see`` keeps a look)
    POST /release  {"source": ..., "fade_s": ...}: that source lets the head go now, its overlay
                   back on the holder's aim over fade_s
    GET  /state    the published state, the live requests, the overlay, the driver in use

:class:`JsonDoor` is the server half (a thread per request, so a look that waits never holds up
a ``/state``), :func:`ask` the client half and :class:`GazeClient` one source's calls over it;
both speak one JSON object each way.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

GAZE_PORT = 3339
GAZE_URL = f"http://127.0.0.1:{GAZE_PORT}"
IDLE_BAND = 5  # pepin.gaze.IDLE: expression's band, below every other
Route = Callable[[dict[str, Any]], dict[str, Any]]


class JsonDoor:
    """A threaded HTTP server of JSON routes: ``routes[(method, path)](body) -> answer``."""

    def __init__(self, host: str, port: int, routes: dict[tuple[str, str], Route]) -> None:
        self._routes = routes
        door = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                door.serve(self, "GET")

            def do_POST(self) -> None:
                door.serve(self, "POST")

            def log_message(self, format: str, *args: Any) -> None:
                return  # the node's report line says what the door did

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="gaze-door")

    @property
    def port(self) -> int:
        """The port it listens on (the one asked for, or the one the system gave for 0)."""
        return int(self._server.server_address[1])

    def start(self) -> JsonDoor:
        """Serve on a daemon thread; returns self."""
        self._thread.daemon = True
        self._thread.start()
        return self

    def close(self) -> None:
        """Stop serving and free the port."""
        self._server.shutdown()
        self._server.server_close()

    def serve(self, request: BaseHTTPRequestHandler, method: str) -> None:
        """One request: its route's answer as JSON, 404 for no route, 400 for a bad body."""
        route = self._routes.get((method, request.path.split("?")[0]))
        if route is None:
            self._reply(request, 404, {"error": f"no {method} {request.path}"})
            return
        body: Any = {}
        length = int(request.headers.get("Content-Length") or 0)
        if length:
            try:
                body = json.loads(request.rfile.read(length))
            except ValueError as exc:
                self._reply(request, 400, {"error": f"not JSON: {exc}"})
                return
        if not isinstance(body, dict):
            self._reply(request, 400, {"error": "the body is one JSON object"})
            return
        self._reply(request, 200, route(body))

    @staticmethod
    def _reply(request: BaseHTTPRequestHandler, code: int, answer: dict[str, Any]) -> None:
        data = json.dumps(answer).encode()
        try:
            request.send_response(code)
            request.send_header("Content-Type", "application/json")
            request.send_header("Content-Length", str(len(data)))
            request.end_headers()
            request.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            return  # the asker stopped waiting (its own timeout): the request lives on regardless


def ask(
    url: str, path: str, body: dict[str, Any] | None = None, timeout_s: float = 3.0
) -> dict[str, Any]:
    """One request to the door at ``url`` (``http://127.0.0.1:3339``): POST ``body``, or GET
    when there is none; the answer, or ``OSError`` when the door is not there."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        answer = json.loads(exc.read() or b"{}")
    if not isinstance(answer, dict):
        raise OSError(f"{url}{path} answered {type(answer).__name__}, not an object")
    return answer


class GazeClient:
    """One source's calls on the arbiter's door: a look, a head overlay (``clip``), a renewal,
    a release and the state; each one HTTP exchange that may wait (call from a worker thread),
    raising ``OSError`` when the door is not there (``TimeoutError`` past ``timeout_s``)."""

    def __init__(self, source: str, url: str = GAZE_URL, timeout_s: float = 3.0) -> None:
        """Calls of ``source`` to the door at ``url``, waiting at most ``timeout_s`` each."""
        self.source = source
        self.url = url
        self.timeout_s = timeout_s

    def look(self, request: Mapping[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        """``POST /look`` of ``request`` as this source's, answered when it ends or refused."""
        body = {**request, "source": self.source}
        return ask(self.url, "/look", body, self.timeout_s if timeout_s is None else timeout_s)

    def clip(
        self,
        keys: Sequence[Mapping[str, float]],
        *,
        ttl_s: float,
        band: int = IDLE_BAND,
        loop: bool = False,
        idle: bool = False,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """A head overlay of ``keys`` ({at_s, pan_rad, tilt_rad, speed_deg_s}, offsets from the
        holder's aim): answered when it ends (a loop: at once), or refused in words."""
        target: dict[str, Any] = {"keys": [dict(key) for key in keys], "loop": loop, "idle": idle}
        request = {"kind": "clip", "band": band, "ttl_s": ttl_s, "target": target}
        return self.look(request, timeout_s)

    def renew(self) -> int:
        """Restart this source's TTL (its overlay's too); how many of its requests live."""
        answer = ask(self.url, "/renew", {"source": self.source}, self.timeout_s)
        return int(answer.get("renewed", 0))

    def release(self, fade_s: float = 0.0) -> int:
        """Let the head go now: every request of this source, its overlay back on the holder's
        aim over ``fade_s``; how many went."""
        body = {"source": self.source, "fade_s": fade_s}
        return int(ask(self.url, "/release", body, self.timeout_s).get("released", 0))

    def state(self) -> dict[str, Any]:
        """``GET /state``: the head, the holder, the blind interval, the overlay, the driver."""
        return ask(self.url, "/state", None, self.timeout_s)
