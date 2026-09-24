"""Learned models on the laptop's own GPU, served to the containers over the loopback.

Docker on macOS cannot see the GPU, so a network run inside a container runs on the Docker VM's
CPU: XFeat + LighterGlue took 0.67 s a registration inside RTAB-Map there, and a place descriptor
712 ms, against 27 ms (XFeat on MPS) and 42 ms (BoQ-DINOv2 on MPS) natively (2026-09-24,
scratch/vpr, scratch/xfeat). This module is the skeleton every such host process is built from;
:mod:`pepin.depth_service` is the first of them and keeps its own proven server, and
:mod:`pepin.localization_service` is built on this one. One process is one FAILURE DOMAIN: the
depth network and the localisation models live in two processes, so one crashing or hanging
never takes the other with it.

THE PIECES, each small and each replaceable:

* :class:`Model` — what a model is to the server: an endpoint name, a ``tag`` (the model's id and
  the first hex digits of its weights' hash, so a client can tell two sets of weights apart), the
  device it runs on, and three stages: :meth:`Model.decode` (bytes to inputs, off the lock),
  :meth:`Model.infer` (under the device's lock) and :meth:`Model.encode` (outputs to bytes, off the
  lock). :meth:`Model.cache_key` names requests whose answer may be reused (``None``: never).
* :class:`ModelServer` — ``POST /<name>`` per model and ``GET /health`` (every model's tag,
  device, counters and per-stage ms median/p95, and the process's uptime), on a stdlib
  ``ThreadingHTTPServer`` bound to 127.0.0.1 (the containers reach it as
  ``host.docker.internal``; nothing is open on the LAN). ONE LOCK PER DEVICE: two models on the
  GPU take turns — there is one GPU, and a queue on it only adds latency — while a CPU model and
  a GPU model run side by side. Answers named by a cache key are kept in a small LRU per model
  and served without touching the lock.
* the codecs, shared by both ends: images as JPEG or raw 8-bit (grey or RGB, the size in
  headers), and numeric arrays as a sequence of ``.npy`` blobs (self-describing: dtype and shape
  travel with the bytes) named in ``X-Arrays``.
* :class:`RemoteModel` — one endpoint as a client: a keep-alive connection, a timeout, and ``None``
  for ANY failure (never an exception); after a connection-level failure the service is left
  alone for ``retry_s`` seconds (the call answers ``None`` at once, counted as ``skipped``), so a
  caller does not pay a timeout on every call while the service is down.
* :class:`ServiceOrLocal` — the switch a caller wraps around a :class:`RemoteModel` and its own
  in-process implementation: ``service`` (the service or nothing), ``local`` (in-process only) or
  ``auto`` (the service, and the local answer when it fails), with counters for a report line.

HTTP/1.1 with keep-alive for the reasons :mod:`pepin.depth_service` measured: both ends are the
standard library, ``curl --data-binary`` debugs it, and the framing is nothing against the payload.
"""

from __future__ import annotations

import http.client
import http.server
import io
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from pepin.telemetry import LatencyTracker

log = logging.getLogger("pepin.model_service")

CONTENT_JPEG = "image/jpeg"
CONTENT_RAW8 = "application/x-pepin-raw8"  # uint8 rows, X-Height x X-Width x X-Channels
CONTENT_NPY = "application/x-pepin-npy"  # .npy blobs back to back, names in X-Arrays
CONTENT_JSON = "application/json"
# The stages a request passes through, timed separately: "wait" is the time spent queued for the
# device's lock, which is what two models sharing one GPU cost each other.
STAGES = ("decode", "wait", "infer", "encode", "total")
REPORT_S = 60.0

Image = npt.NDArray[np.uint8]
In = TypeVar("In")
Out = TypeVar("Out")


class BadRequestError(ValueError):
    """A request the model cannot read: a missing header, an unknown content type, a bad shape."""


# ---------------------------------------------------------------- the codecs
def lower_keys(headers: Mapping[str, str] | Any) -> dict[str, str]:
    """Header names lower-cased, whatever mapping carried them (http.client's, email's, a dict)."""
    return {str(k).lower(): str(v) for k, v in headers.items()}


def encode_image(
    image: npt.ArrayLike, encoding: str = "raw", quality: int = 90
) -> tuple[dict[str, str], bytes]:
    """An 8-bit picture (H x W grey, or H x W x 3 in RGB order) as request headers and body:
    ``raw`` (the bytes as they are: lossless, what a keypoint detector must see) or ``jpeg`` at
    ``quality`` (a tenth of the bytes, for a model that does not care about the last grey level)."""
    array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if array.ndim == 2:
        channels = 1
    elif array.ndim == 3 and array.shape[2] in (1, 3):
        channels = int(array.shape[2])
    else:
        raise ValueError(f"a picture is H x W or H x W x 3, not {array.shape}")
    h, w = int(array.shape[0]), int(array.shape[1])
    size = {"X-Height": str(h), "X-Width": str(w), "X-Channels": str(channels)}
    if encoding == "raw":
        return {"Content-Type": CONTENT_RAW8, **size}, array.tobytes()
    if encoding != "jpeg":
        raise ValueError(f"encoding must be 'raw' or 'jpeg', not {encoding!r}")
    import cv2

    pixels = array if channels == 1 else array[:, :, ::-1]  # OpenCV writes BGR
    ok, buf = cv2.imencode(".jpg", pixels, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return {"Content-Type": CONTENT_JPEG, **size}, buf.tobytes()


def decode_image(headers: Mapping[str, str], body: bytes) -> Image:
    """The picture a request carries — H x W for grey, H x W x 3 in RGB order for colour — from
    either encoding; :class:`BadRequestError` when it is not one."""
    h = lower_keys(headers)
    kind = h.get("content-type", "")
    if kind == CONTENT_RAW8:
        try:
            height, width = int(h["x-height"]), int(h["x-width"])
            channels = int(h.get("x-channels", "1"))
        except (KeyError, ValueError) as exc:
            raise BadRequestError("raw pictures need X-Height, X-Width and X-Channels") from exc
        if channels not in (1, 3) or len(body) != height * width * channels:
            raise BadRequestError(f"{len(body)} bytes is not {height}x{width}x{channels}")
        flat = np.frombuffer(body, dtype=np.uint8)
        out: Image = (
            flat.reshape(height, width) if channels == 1 else flat.reshape(height, width, 3)
        )
        return out
    if kind == CONTENT_JPEG:
        import cv2

        grey = h.get("x-channels", "3") == "1"
        mode = cv2.IMREAD_GRAYSCALE if grey else cv2.IMREAD_COLOR
        pixels = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), mode)
        if pixels is None:
            raise BadRequestError("the body is not a JPEG")
        decoded: Image = np.asarray(pixels if grey else pixels[:, :, ::-1], dtype=np.uint8)
        return np.ascontiguousarray(decoded)
    raise BadRequestError(f"unknown picture content type {kind!r}")


def encode_arrays(arrays: Mapping[str, npt.ArrayLike]) -> tuple[dict[str, str], bytes]:
    """Named numeric arrays as headers and body: their ``.npy`` blobs back to back (each carries
    its own dtype and shape) and the names, in the same order, in ``X-Arrays``."""
    names = list(arrays)
    for name in names:
        if not name or "," in name:
            raise ValueError(f"an array's name is a word without commas, not {name!r}")
    buf = io.BytesIO()
    for name in names:
        np.lib.format.write_array(  # type: ignore[no-untyped-call]
            buf, np.ascontiguousarray(arrays[name]), allow_pickle=False
        )
    return {"Content-Type": CONTENT_NPY, "X-Arrays": ",".join(names)}, buf.getvalue()


def decode_arrays(headers: Mapping[str, str], body: bytes) -> dict[str, npt.NDArray[Any]]:
    """The named arrays of a request or an answer; :class:`BadRequestError` when the body is not
    the blobs ``X-Arrays`` names (never unpickled: an object array is refused)."""
    h = lower_keys(headers)
    if h.get("content-type", "") != CONTENT_NPY:
        raise BadRequestError(f"not arrays: {h.get('content-type', '')!r}")
    names = [n for n in h.get("x-arrays", "").split(",") if n]
    buf = io.BytesIO(body)
    out: dict[str, npt.NDArray[Any]] = {}
    try:
        for name in names:
            out[name] = np.lib.format.read_array(buf, allow_pickle=False)  # type: ignore[no-untyped-call]
    except (ValueError, EOFError) as exc:
        raise BadRequestError(f"the body does not hold the arrays {names}: {exc}") from exc
    if buf.tell() != len(body):
        raise BadRequestError(f"{len(body) - buf.tell()} bytes left after the arrays {names}")
    return out


# ---------------------------------------------------------------- what a model is
@dataclass(frozen=True)
class Request:
    """One POST: the endpoint, its headers (lower-cased) and its body."""

    path: str
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class Response:
    """One answer: its content type, body and extra headers."""

    content_type: str
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


class Model(Protocol[In, Out]):
    """A model the server can serve: ``POST /<name>`` runs decode -> infer -> encode."""

    @property
    def name(self) -> str:
        """The endpoint: ``POST /<name>``."""
        ...

    @property
    def tag(self) -> str:
        """The model's id and its weights' hash prefix (``xfeat@3f2a9c1e``): what answered."""
        ...

    @property
    def device(self) -> str:
        """Where inference runs (``mps``, ``cpu``); models on one device share one lock."""
        ...

    @property
    def cache_size(self) -> int:
        """How many answers the server may keep for :meth:`cache_key` (0: none)."""
        ...

    def warm(self) -> None:
        """Build the network and run it once, so the first real request pays nothing."""
        ...

    def decode(self, request: Request) -> In:
        """The request's inputs; :class:`BadRequestError` when they cannot be read."""
        ...

    def cache_key(self, inputs: In) -> Hashable | None:
        """What identifies an answer that may be reused, or ``None`` for one that may not."""
        ...

    def infer(self, inputs: In) -> Out:
        """The network on the inputs, under the device's lock."""
        ...

    def encode(self, outputs: Out) -> Response:
        """The outputs as the answer's bytes."""
        ...


class ModelStats:
    """One model's counters and per-stage latencies, safe to update from the handler threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.cache_hits = 0
        self.window = 0  # requests since the last report line
        self._timing = {stage: LatencyTracker(stage) for stage in STAGES}

    def served(self, stages: Mapping[str, float], cached: bool) -> None:
        """Count one answered request and record its stage times (seconds)."""
        with self._lock:
            self.requests += 1
            self.window += 1
            self.cache_hits += int(cached)
            for stage, seconds in stages.items():
                self._timing[stage].add(seconds)

    def refused(self) -> None:
        """Count one request that was not answered (a bad request or a failed network)."""
        with self._lock:
            self.errors += 1

    def ms(self) -> dict[str, dict[str, float]]:
        """Median, p95 and max per stage, in milliseconds (zeros for an unused stage)."""
        with self._lock:
            summaries = {stage: t.summary() for stage, t in self._timing.items()}
        return {
            stage: {
                "median": round(s.median_ms, 1),
                "p95": round(s.p95_ms, 1),
                "max": round(s.max_ms, 1),
            }
            for stage, s in summaries.items()
        }

    def take_window(self) -> int:
        """Requests since the last call, and a new window."""
        with self._lock:
            count, self.window = self.window, 0
        return count


class AnswerCache:
    """The last ``size`` answers by key, least recently used out first; thread-safe."""

    def __init__(self, size: int) -> None:
        self._size = max(0, int(size))
        self._items: OrderedDict[Hashable, Response] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable | None) -> Response | None:
        """The kept answer for ``key``, or ``None``."""
        if key is None or self._size == 0:
            return None
        with self._lock:
            answer = self._items.get(key)
            if answer is not None:
                self._items.move_to_end(key)
            return answer

    def put(self, key: Hashable | None, answer: Response) -> None:
        """Keep ``answer`` under ``key``, dropping the least recently used past the size."""
        if key is None or self._size == 0:
            return
        with self._lock:
            self._items[key] = answer
            self._items.move_to_end(key)
            while len(self._items) > self._size:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------- the server
class ModelServer(http.server.ThreadingHTTPServer):
    """Serves ``POST /<name>`` for each model and ``GET /health``; one inference lock per device.

    Requests are answered in threads: decoding and encoding run side by side, inference takes
    its device's lock. A model that raises answers 500 (the connection stays up) and is counted
    against that model; a request it cannot read answers 400."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, address: tuple[str, int], models: Sequence[Model[Any, Any]], name: str = "models"
    ) -> None:
        super().__init__(address, ModelHandler)
        names = [m.name for m in models]
        if len(set(names)) != len(names):
            raise ValueError(f"two models answer one endpoint: {names}")
        self.service = name
        self.models: dict[str, Model[Any, Any]] = {m.name: m for m in models}
        self.stats = {m.name: ModelStats() for m in models}
        self.caches = {m.name: AnswerCache(m.cache_size) for m in models}
        self.locks = {m.device: threading.Lock() for m in models}
        self._since = time.monotonic()
        self._reporter = threading.Thread(target=self._report_loop, daemon=True)

    def answer(self, request: Request) -> Response:
        """One request through its model: decode, the cache, the device's lock, infer, encode.
        :class:`KeyError` for an endpoint no model answers."""
        model = self.models[request.path.strip("/")]
        stats = self.stats[model.name]
        t0 = time.perf_counter()
        inputs = model.decode(request)
        key = model.cache_key(inputs)
        cached = self.caches[model.name].get(key)
        t1 = time.perf_counter()
        if cached is not None:
            stats.served({"decode": t1 - t0, "total": t1 - t0}, cached=True)
            return Response(cached.content_type, cached.body, {**cached.headers, "X-Cache": "hit"})
        with self.locks[model.device]:
            t2 = time.perf_counter()
            outputs = model.infer(inputs)
            t3 = time.perf_counter()
        answer = model.encode(outputs)
        t4 = time.perf_counter()
        answer = Response(
            answer.content_type,
            answer.body,
            {**answer.headers, "X-Model": model.tag, "X-Infer-Ms": f"{(t3 - t2) * 1e3:.1f}"},
        )
        self.caches[model.name].put(key, answer)
        stats.served(
            {
                "decode": t1 - t0,
                "wait": t2 - t1,
                "infer": t3 - t2,
                "encode": t4 - t3,
                "total": t4 - t0,
            },
            cached=False,
        )
        return answer

    def health(self) -> dict[str, Any]:
        """What ``GET /health`` says: the service, its uptime, and per model its tag, device,
        counters and per-stage ms."""
        return {
            "service": self.service,
            "uptime_s": round(time.monotonic() - self._since, 1),
            "models": {
                name: {
                    "tag": model.tag,
                    "device": model.device,
                    "requests": self.stats[name].requests,
                    "errors": self.stats[name].errors,
                    "cache_hits": self.stats[name].cache_hits,
                    "cached": len(self.caches[name]),
                    "ms": self.stats[name].ms(),
                }
                for name, model in self.models.items()
            },
        }

    def report(self) -> str:
        """One line: per model the window's rate, the totals and the total ms median/p95."""
        parts = []
        for name, model in self.models.items():
            stats = self.stats[name]
            total = stats.ms()["total"]
            infer = stats.ms()["infer"]
            parts.append(
                f"{name} ({model.tag} on {model.device}) {stats.take_window() / REPORT_S:.2f}/s,"
                f" {stats.requests} served ({stats.cache_hits} cached), {stats.errors} refused,"
                f" total {total['median']:.0f}/{total['p95']:.0f} ms,"
                f" infer {infer['median']:.0f}/{infer['p95']:.0f} ms"
            )
        return f"{self.service}: " + "; ".join(parts)

    def serve(self) -> None:
        """Serve until interrupted, with a report line every minute in which a request came."""
        self._reporter.start()
        self.serve_forever()

    def _report_loop(self) -> None:
        while True:
            time.sleep(REPORT_S)
            if any(s.window for s in self.stats.values()):
                log.info(self.report())


class ModelHandler(http.server.BaseHTTPRequestHandler):
    """The HTTP side of :class:`ModelServer`: keep-alive, one request per POST."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass  # the server reports itself

    @property
    def model_server(self) -> ModelServer:
        server: ModelServer = self.server  # type: ignore[assignment]
        return server

    def do_GET(self) -> None:
        if self.path != "/health":
            self._reply(404, "text/plain", {}, self._endpoints().encode())
            return
        self._reply(200, CONTENT_JSON, {}, json.dumps(self.model_server.health()).encode())

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Content-Length is not a number")  # closes: the body is unknown
            return
        body = self.rfile.read(length) if length else b""
        server = self.model_server
        name = self.path.strip("/")
        if name not in server.models:
            self._reply(404, "text/plain", {}, self._endpoints().encode())
            return
        request = Request(self.path, lower_keys(self.headers), body)
        try:
            answer = server.answer(request)
        except BadRequestError as exc:
            server.stats[name].refused()
            self._reply(400, "text/plain", {}, f"{exc}\n".encode())
            return
        except Exception as exc:  # the network failed: answer, keep the connection
            server.stats[name].refused()
            log.exception("%s failed", name)
            self._reply(500, "text/plain", {}, f"{type(exc).__name__}: {exc}\n".encode())
            return
        self._reply(200, answer.content_type, answer.headers, answer.body)

    def _endpoints(self) -> str:
        posts = ", ".join(f"POST /{name}" for name in self.model_server.models)
        return f"GET /health, {posts}\n"

    def _reply(self, status: int, kind: str, extra: Mapping[str, str], body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------- the client
def split_url(url: str) -> tuple[str, int]:
    """``http://host:port`` as (host, port); :class:`ValueError` for anything else."""
    if not url.startswith("http://"):
        raise ValueError(f"a model service URL starts with http://, not {url!r}")
    host, _, port = url[len("http://") :].rstrip("/").partition(":")
    return host, int(port) if port else 80


class RemoteModel:
    """One endpoint of a model service as a client that never raises.

    :meth:`call` answers the response's headers and body, or ``None`` for ANY failure: no
    service, a timeout, an HTTP error, a broken connection. A REFUSED or broken connection means
    the service is not there, and the next ``retry_s`` seconds of calls answer ``None`` at once
    without touching the network (``skipped``); the first call after that tries again. A TIMEOUT
    means it is there and slow — measured 2026-09-24, a GPU model's first inference after two
    seconds idle takes 200-460 ms instead of 50 (scratch/models/place_parity.py's service) — so
    only ``timeouts`` of them in a row back off; one alone costs its own call. An HTTP error means
    the service refused this one request, and costs nothing more. Thread-safe: calls on one client
    take turns on its one keep-alive connection."""

    def __init__(
        self,
        url: str,
        endpoint: str,
        timeout_s: float = 1.0,
        retry_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        timeouts: int = 3,
    ) -> None:
        self.url = url.rstrip("/")
        self.endpoint = "/" + endpoint.strip("/")
        self._host, self._port = split_url(url)
        self.timeout_s = float(timeout_s)
        self.retry_s = float(retry_s)
        self._clock = clock
        self._conn: http.client.HTTPConnection | None = None
        self._lock = threading.Lock()
        self._down_until = -float("inf")
        self._timeouts_allowed = max(1, int(timeouts))
        self._timeouts = 0  # timeouts in a row
        self.ok = 0
        self.failed = 0
        self.skipped = 0
        self.last_error = ""
        self.last_model = ""  # the X-Model tag of the last answer
        self.round_trip = LatencyTracker(self.endpoint)

    @property
    def down(self) -> bool:
        """Whether calls are being skipped because the service did not answer lately."""
        return self._clock() < self._down_until

    def close(self) -> None:
        """Drop the connection; the next call opens a new one."""
        with self._lock:
            self._drop()

    def call(
        self, headers: Mapping[str, str], body: bytes, timeout_s: float | None = None
    ) -> tuple[dict[str, str], bytes] | None:
        """POST ``body`` to the endpoint: the answer's (lower-cased headers, body), or ``None``."""
        with self._lock:
            if self.down:
                self.skipped += 1
                return None
            timeout = self.timeout_s if timeout_s is None else float(timeout_s)
            t0 = time.perf_counter()
            try:
                conn = self._connection(timeout)
                conn.request("POST", self.endpoint, body=body, headers=dict(headers))
                response = conn.getresponse()
                data = response.read()
            except (OSError, http.client.HTTPException) as exc:
                self._drop()  # a late answer must not be read as the next call's
                slow = isinstance(exc, TimeoutError)
                self._timeouts = self._timeouts + 1 if slow else 0
                back_off = not slow or self._timeouts >= self._timeouts_allowed
                self._failed(f"{type(exc).__name__}: {exc}", back_off=back_off)
                return None
            self._timeouts = 0
            if response.status != 200:
                self._failed(f"{response.status}: {data[:160].decode(errors='replace').strip()}")
                return None
            self.round_trip.add(time.perf_counter() - t0)
            self.ok += 1
            reply = lower_keys(dict(response.getheaders()))
            self.last_model = reply.get("x-model", self.last_model)
            return reply, data

    def health(self, timeout_s: float = 2.0) -> dict[str, Any] | None:
        """The service's ``/health`` as a dict, or ``None``; on a connection of its own, so a
        report timer may ask while a call is in flight."""
        conn = http.client.HTTPConnection(self._host, self._port, timeout=timeout_s)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            data = response.read()
            if response.status != 200:
                return None
            result: dict[str, Any] = json.loads(data)
            return result
        except (OSError, http.client.HTTPException, ValueError):
            return None
        finally:
            conn.close()

    def status(self) -> str:
        """For a report line: ``ok 118, failed 2 (last: ...), skipped 5, rt 12/30 ms``."""
        rt = self.round_trip.summary()
        text = f"ok {self.ok}, failed {self.failed}, skipped {self.skipped}"
        if self.failed:
            text += f" (last: {self.last_error})"
        if self.down:
            text += f", down for {self._down_until - self._clock():.0f} s more"
        return text + f", round trip {rt.median_ms:.0f}/{rt.p95_ms:.0f} ms"

    def _connection(self, timeout: float) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self._host, self._port, timeout=timeout)
        else:
            self._conn.timeout = timeout
            if self._conn.sock is not None:
                self._conn.sock.settimeout(timeout)
        return self._conn

    def _drop(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def _failed(self, why: str, back_off: bool = False) -> None:
        self.failed += 1
        self.last_error = why[:160]
        if back_off:
            self._down_until = self._clock() + self.retry_s


# ---------------------------------------------------------------- the switch
SERVICE, LOCAL, AUTO = "service", "local", "auto"
MODES = (SERVICE, LOCAL, AUTO)


class ServiceOrLocal[X, Y]:
    """One computation with two implementations — a service and this process — and the switch
    between them, read on every call.

    ``service``: the service's answer, or ``empty`` when it has none (counted ``failed``).
    ``local``: this process's answer. ``auto``: the service's, and this process's when the
    service has none (counted ``fallback``). A local implementation that raises is caught and
    answered with ``empty`` (counted ``failed``): the caller is a C++ host that must get arrays,
    not an exception. The counters are cumulative; :meth:`line` renders them."""

    def __init__(
        self,
        remote: Callable[[X], Y | None],
        local: Callable[[X], Y],
        empty: Callable[[X], Y],
    ) -> None:
        self._remote, self._local, self._empty = remote, local, empty
        self.service = 0  # answered by the service
        self.local = 0  # answered here because the mode said so
        self.fallback = 0  # answered here because the service had no answer (auto)
        self.failed = 0  # answered empty
        self.last_error = ""

    def __call__(self, x: X, mode: str) -> Y:
        if mode not in MODES:
            mode = AUTO
        if mode != LOCAL:
            answer = self._remote(x)
            if answer is not None:
                self.service += 1
                return answer
            if mode == SERVICE:
                self.failed += 1
                return self._empty(x)
        try:
            answer = self._local(x)
        except Exception as exc:
            self.failed += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:160]
            return self._empty(x)
        if mode == LOCAL:
            self.local += 1
        else:
            self.fallback += 1
        return answer

    def line(self) -> str:
        """``service 118, local 0, fallback 2, failed 0``, with the last local error if any."""
        text = (
            f"service {self.service}, local {self.local}, fallback {self.fallback},"
            f" failed {self.failed}"
        )
        return text + (f" (last local error: {self.last_error})" if self.last_error else "")


class EveryMinute:
    """Whether a periodic line is due: true at most once per ``period_s`` of ``clock``."""

    def __init__(self, period_s: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._period_s = period_s
        self._clock = clock
        self._last = clock()

    def due(self) -> bool:
        """True once a period has passed since construction or since the last true answer."""
        now = self._clock()
        if now - self._last < self._period_s:
            return False
        self._last = now
        return True
