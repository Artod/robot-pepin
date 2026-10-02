"""The gaze arbiter's door for processes that are not ROS nodes: JSON over HTTP on the Mac.

The arbiter (``pepin_bringup.gaze``) runs in the navigation container and answers here, on
:data:`GAZE_PORT` (published on this Mac's loopback only, as the goal server's port is):

    POST /look    a request (pepin.gaze.look_from_json's JSON), answered when it ends
    POST /renew   {"source": ...}: restart that source's TTL (``see`` keeps a look)
    GET  /state   the published state, the live requests, the driver in use

:class:`JsonDoor` is the server half (a thread per request, so a look that waits never holds up
a ``/state``), :func:`ask` the client half; both speak one JSON object each way.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

GAZE_PORT = 3339
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
        request.send_response(code)
        request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(data)))
        request.end_headers()
        request.wfile.write(data)


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
