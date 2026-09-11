"""The depth network on the laptop's own GPU, as a service the container's depth node calls.

Docker on macOS cannot see the GPU, so the network in the container runs on the CPU (0.2-0.3 s a
frame). This module runs the same network on the host with torch's Metal backend and serves it
over HTTP on the loopback, which the container reaches as ``host.docker.internal`` (Docker
Desktop forwards it to the host's 127.0.0.1, so nothing is exposed on the LAN). One request is
one frame: a JPEG (or the raw RGB bytes) in, the depth image as float16 out, resized by the
server to the frame's size exactly as :class:`pepin_bringup.depth_stream.MonoDepth` does
(bilinear, ``align_corners=False``), so the node cannot tell the two apart. The model's native
output can be asked for instead (``X-Depth-Size: native``) and resized here.

Inference is strictly sequential under one lock, connections are served in threads: the node
keeps only the newest frame while the network is busy (it drops on its side), so a queue here
would only add latency; a second client (the bench, a probe) waits its turn at the lock and
``/health`` answers meanwhile. The node wraps the client in :class:`Fallback`: when the service
does not answer, the CPU model in the container takes over and the service is tried again every
so often — a live parameter picks remote, local or auto.

HTTP/1.1 with keep-alive was chosen over a socket protocol because both ends are the standard
library, one TCP connection is reused for every frame, ``curl --data-binary @frame.jpg`` debugs
it, and the framing overhead is nothing against the payload. Measured (2026-09-11,
scratch/depth_backend_bench.py, 640x360 frames, M5 Pro): the Small model answers in 20.6 ms on
MPS against 172 ms on the CPU inside the container, and the whole round trip from inside
pepin-vslam is 26.0 ms of which 6.1 ms is transport — so the hop costs a quarter of what the
GPU saves. Send JPEG, not raw: q90 costs 1.1 ms to encode and saves 1.8 ms of wire, 37.7 KB
through Docker's NAT instead of 675 KB.

The node's side (pepin_bringup.depth_stream): the choice flag ``depth_backend`` in {remote,
local, auto}, default from PEPIN_DEPTH_BACKEND and ``local`` without it, sets
:attr:`Fallback.mode` live; ``depth_url`` defaults to PEPIN_DEPTH_URL or :data:`DEFAULT_URL`.
The node builds ``Fallback(RemoteDepth(url), LazyDepth(lambda: MonoDepth(...)))`` and calls it
where it called the model — same argument, same return; :class:`LazyDepth` builds the CPU model
on its first frame, so a node on the service never loads it. A build that fails is not tried
again: every later frame raises :class:`DepthModelError` at once, which the node treats as
fatal in ``local`` mode (the process ends, the launch respawns it — as loud as a model that
failed in the constructor) and as a lost frame in ``auto`` (the service keeps being probed).
:attr:`Fallback.status` is the phrase for the report line.
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from pepin.depth import Array
from pepin.telemetry import LatencyTracker

log = logging.getLogger("pepin.depth_service")

Rgb = npt.NDArray[np.uint8]
Depth = npt.NDArray[np.float32]

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}
DEFAULT_PORT = 8790
DEFAULT_URL = f"http://host.docker.internal:{DEFAULT_PORT}"  # the host, seen from the container
CONTENT_JPEG = "image/jpeg"
CONTENT_RGB = "application/x-pepin-rgb8"
CONTENT_DEPTH = "application/x-pepin-depth16"
STAGES = ("decode", "infer", "pack", "total")
REPORT_S = 30.0


def model_id(name: str) -> str:
    """The Hugging Face id of ``small``, ``base`` or ``large``; any other name is one already."""
    return MODELS.get(name.lower(), name)


class DepthServiceError(RuntimeError):
    """The service did not answer, or answered something that is not a depth image."""


class DepthModelError(RuntimeError):
    """The CPU model could not be built — no cached weights and no hub, or no memory — and
    :class:`LazyDepth` will not try again: every later frame raises this at once."""


class BadRequestError(ValueError):
    """A request the service cannot read (no size, no image, an unknown content type)."""


# ---------------------------------------------------------------- codec: frames and depths
def lower_keys(headers: Mapping[str, str] | Any) -> dict[str, str]:
    """Header names lower-cased, whatever mapping carried them (http.client's, email's, a dict)."""
    return {str(k).lower(): str(v) for k, v in headers.items()}


def encode_frame(
    rgb: Rgb, encoding: str = "jpeg", quality: int = 90
) -> tuple[dict[str, str], bytes]:
    """An RGB frame as request headers and body: a JPEG at ``quality`` (60-100 KB for 640x360,
    a few ms to encode) or the raw bytes (``raw``: 3 bytes a pixel, nothing to encode)."""
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if encoding == "raw":
        body = np.ascontiguousarray(rgb).tobytes()
        return {"Content-Type": CONTENT_RGB, "X-Height": str(h), "X-Width": str(w)}, body
    if encoding != "jpeg":
        raise ValueError(f"encoding must be 'jpeg' or 'raw', not {encoding!r}")
    import cv2

    ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return {"Content-Type": CONTENT_JPEG, "X-Height": str(h), "X-Width": str(w)}, buf.tobytes()


def decode_frame(headers: Mapping[str, str], body: bytes) -> Rgb:
    """The RGB frame a request carries, from either encoding; :class:`BadRequestError` otherwise."""
    h = lower_keys(headers)
    kind = h.get("content-type", "")
    if kind == CONTENT_RGB:
        try:
            height, width = int(h["x-height"]), int(h["x-width"])
        except (KeyError, ValueError) as exc:
            raise BadRequestError("raw frames need X-Height and X-Width") from exc
        if len(body) != height * width * 3:
            raise BadRequestError(f"{len(body)} bytes is not {height}x{width}x3")
        rgb: Rgb = np.frombuffer(body, dtype=np.uint8).reshape(height, width, 3)
        return rgb
    if kind == CONTENT_JPEG:
        import cv2

        bgr = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise BadRequestError("the body is not a JPEG")
        out: Rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        return out
    raise BadRequestError(f"unknown content type {kind!r}")


def requested_size(headers: Mapping[str, str], frame: tuple[int, int]) -> tuple[int, int] | None:
    """The (height, width) a request wants its depth at: the frame's own size unless
    ``X-Depth-Size: native`` asks for the network's output as is (``None``)."""
    h = lower_keys(headers)
    if h.get("x-depth-size", "").lower() == "native":
        return None
    return frame


def pack_depth(depth: Depth) -> tuple[dict[str, str], bytes]:
    """A depth image as response headers and float16 body (metres; 1 mm at 2 m, 8 mm at 8 m —
    under the network's own noise), the size in ``X-Height`` and ``X-Width``."""
    h, w = int(depth.shape[0]), int(depth.shape[1])
    body = np.ascontiguousarray(depth, dtype=np.float16).tobytes()
    return {"Content-Type": CONTENT_DEPTH, "X-Height": str(h), "X-Width": str(w)}, body


def unpack_depth(headers: Mapping[str, str], body: bytes) -> Depth:
    """The float32 depth image of a response; :class:`DepthServiceError` when it is not one."""
    h = lower_keys(headers)
    if h.get("content-type", "") != CONTENT_DEPTH:
        raise DepthServiceError(f"not a depth image: {h.get('content-type', '')!r}")
    try:
        height, width = int(h["x-height"]), int(h["x-width"])
    except (KeyError, ValueError) as exc:
        raise DepthServiceError("a depth image without a size") from exc
    if len(body) != height * width * 2:
        raise DepthServiceError(f"{len(body)} bytes is not {height}x{width} float16")
    depth: Depth = np.frombuffer(body, dtype=np.float16).reshape(height, width).astype(np.float32)
    return depth


def resize_depth(depth: Depth, size: tuple[int, int]) -> Depth:
    """A depth image at another (height, width), bilinear with half-pixel centres — what
    torch's ``interpolate(mode="bilinear", align_corners=False)`` computes, so a native-size
    answer resized here matches the server's (2e-06 m worst pixel over 30 frames,
    scratch/depth_backend_bench.py)."""
    if tuple(depth.shape) == tuple(size):
        return depth
    import cv2

    resized = cv2.resize(depth, (int(size[1]), int(size[0])), interpolation=cv2.INTER_LINEAR)
    out: Depth = np.asarray(resized, dtype=np.float32)
    return out


# ---------------------------------------------------------------- the network on a device
class DepthNet:
    """Depth Anything V2 on one device (``mps`` on the laptop, ``cpu`` anywhere): an RGB frame in,
    a float32 depth image out at the asked size (``None``: the network's own output size, the
    longer or the shorter side at 518 px, whichever scale is nearer 1 — 294x518 for 640x360,
    518x924 for 1280x720)."""

    def __init__(self, model: str = "small", device: str = "mps", half: bool = False) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.name = model_id(model)
        if device == "mps" and not torch.backends.mps.is_available():
            log.warning("MPS is not available here: the network runs on the CPU")
            device = "cpu"
        self.device = device
        self.half = half and device != "cpu"
        self._torch = torch
        self._processor: Any = AutoImageProcessor.from_pretrained(self.name)  # type: ignore[no-untyped-call]
        net = AutoModelForDepthEstimation.from_pretrained(self.name).eval().to(device)
        self._model = net.half() if self.half else net

    def __call__(self, rgb: Rgb, size: tuple[int, int] | None = None) -> Depth:
        torch = self._torch
        with torch.no_grad():
            inputs = self._processor(images=rgb, return_tensors="pt")
            pixels = inputs["pixel_values"].to(self.device)
            if self.half:
                pixels = pixels.half()
            predicted = self._model(pixel_values=pixels).predicted_depth.unsqueeze(1).float()
            if size is not None:
                predicted = torch.nn.functional.interpolate(
                    predicted, size=tuple(size), mode="bilinear", align_corners=False
                )
            depth: Depth = predicted[0, 0].cpu().numpy().astype(np.float32)
        return depth


# ---------------------------------------------------------------- the server
class DepthServer(http.server.ThreadingHTTPServer):
    """Serves ``POST /depth`` (a frame in, its depth out) and ``GET /health`` (the model, the
    device, the counters and the per-stage latencies as JSON). Frames are inferred one at a
    time under :attr:`lock`; the timing of every stage is tracked and reported every 30 s."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], net: Callable[..., Depth], name: str = "") -> None:
        super().__init__(address, DepthHandler)
        self.net = net
        self.name = name or getattr(net, "name", "?")
        self.device = str(getattr(net, "device", "?"))
        self.lock = threading.Lock()
        self.timing = {stage: LatencyTracker(stage) for stage in STAGES}
        self.requests = 0
        self.errors = 0
        self._since = time.monotonic()
        self._window_requests = 0
        self._reporter = threading.Thread(target=self._report_loop, daemon=True)

    def answer(self, headers: Mapping[str, str], body: bytes) -> tuple[dict[str, str], bytes]:
        """One frame through the network: the response headers and the float16 body."""
        t0 = time.perf_counter()
        rgb = decode_frame(headers, body)
        t1 = time.perf_counter()
        with self.lock:
            depth = self.net(rgb, requested_size(headers, (rgb.shape[0], rgb.shape[1])))
        t2 = time.perf_counter()
        out_headers, out = pack_depth(depth)
        t3 = time.perf_counter()
        self.timing["decode"].add(t1 - t0)
        self.timing["infer"].add(t2 - t1)
        self.timing["pack"].add(t3 - t2)
        self.timing["total"].add(t3 - t0)
        self.requests += 1
        self._window_requests += 1
        out_headers["X-Infer-Ms"] = f"{(t2 - t1) * 1e3:.1f}"
        out_headers["X-Model"] = self.name
        log.debug(
            "frame %dx%d -> %dx%d: decode %.1f infer %.1f pack %.1f ms",
            rgb.shape[1],
            rgb.shape[0],
            depth.shape[1],
            depth.shape[0],
            (t1 - t0) * 1e3,
            (t2 - t1) * 1e3,
            (t3 - t2) * 1e3,
        )
        return out_headers, out

    def health(self) -> dict[str, Any]:
        """What ``/health`` says: model, device, counters, uptime and the latency summaries."""
        return {
            "model": self.name,
            "device": self.device,
            "requests": self.requests,
            "errors": self.errors,
            "uptime_s": round(time.monotonic() - self._since, 1),
            "ms": {
                stage: {
                    "median": round(t.summary().median_ms, 1),
                    "p95": round(t.summary().p95_ms, 1),
                    "max": round(t.summary().max_ms, 1),
                }
                for stage, t in self.timing.items()
            },
        }

    def report(self) -> str:
        """One line with the window's rate and the per-stage median/p95 in ms."""
        stages = " ".join(
            f"{stage} {t.summary().median_ms:.0f}/{t.summary().p95_ms:.0f}"
            for stage, t in self.timing.items()
        )
        return (
            f"depth service ({self.name.rsplit('/', 1)[-1]} on {self.device}):"
            f" {self._window_requests / REPORT_S:.1f} frames/s, {self.requests} served,"
            f" {self.errors} refused, ms median/p95: {stages}"
        )

    def serve(self) -> None:
        """Serve until interrupted, reporting every 30 s in which a frame arrived."""
        self._reporter.start()
        self.serve_forever()

    def _report_loop(self) -> None:
        while True:
            time.sleep(REPORT_S)
            if self._window_requests:
                log.info(self.report())
                self._window_requests = 0


class DepthHandler(http.server.BaseHTTPRequestHandler):
    """The HTTP side of :class:`DepthServer`: keep-alive, one frame per POST."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass  # the server reports itself; the per-request line is at debug level

    @property
    def depth_server(self) -> DepthServer:
        server: DepthServer = self.server  # type: ignore[assignment]
        return server

    def do_GET(self) -> None:
        if self.path != "/health":
            self._reply(404, "text/plain", {}, b"only /health and POST /depth\n")
            return
        body = json.dumps(self.depth_server.health()).encode()
        self._reply(200, "application/json", {}, body)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Content-Length is not a number")  # closes: the body is unknown
            return
        body = self.rfile.read(length) if length else b""
        if self.path != "/depth":
            self._reply(404, "text/plain", {}, b"only POST /depth\n")
            return
        try:
            headers, out = self.depth_server.answer(lower_keys(self.headers), body)
        except BadRequestError as exc:
            self.depth_server.errors += 1
            self._reply(400, "text/plain", {}, f"{exc}\n".encode())
            return
        except Exception as exc:  # the network itself failed: answer, do not drop the connection
            self.depth_server.errors += 1
            log.exception("the network failed on a frame")
            self._reply(500, "text/plain", {}, f"{type(exc).__name__}: {exc}\n".encode())
            return
        self._reply(200, headers.pop("Content-Type"), headers, out)

    def _reply(self, status: int, kind: str, extra: Mapping[str, str], body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------- the client and the switch
class DepthBackend(Protocol):
    """Whatever turns an RGB frame into a depth image of the same size (float32 metres)."""

    def __call__(self, rgb: Rgb) -> Array: ...


class RemoteDepth:
    """The service as a :class:`DepthBackend`: one keep-alive connection, a frame sent as a JPEG
    (or raw), the answer as float32 at the frame's size. Any failure — no service, a timeout, a
    wrong answer — is a :class:`DepthServiceError` and drops the connection so the next call
    reconnects. ``native=True`` asks for the network's own output size and resizes here.

    One caller sends frames (the node's worker thread); :meth:`health` is the exception and
    takes a connection of its own, so a report timer may ask while a frame is in flight."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        timeout_s: float = 2.0,
        encoding: str = "jpeg",
        quality: int = 90,
        native: bool = False,
    ) -> None:
        if not url.startswith("http://"):
            raise ValueError(f"the depth service URL must start with http://, not {url!r}")
        self.url = url
        host_port = url[len("http://") :].rstrip("/")
        self._host, _, port = host_port.partition(":")
        self._port = int(port) if port else 80
        self._timeout = timeout_s
        self._encoding = encoding
        self._quality = quality
        self._native = native
        self._conn: http.client.HTTPConnection | None = None
        self.timing = {"encode": LatencyTracker("encode"), "round_trip": LatencyTracker("rt")}
        self.last_infer_ms = 0.0  # the server's own inference time of the last frame

    def _connection(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        return self._conn

    def close(self) -> None:
        """Drop the connection; the next call opens a new one."""
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __call__(self, rgb: Rgb) -> Array:
        t0 = time.perf_counter()
        headers, body = encode_frame(rgb, self._encoding, self._quality)
        if self._native:
            headers["X-Depth-Size"] = "native"
        t1 = time.perf_counter()
        try:
            conn = self._connection()
            conn.request("POST", "/depth", body=body, headers=headers)
            response = conn.getresponse()
            data = response.read()
            if response.status != 200:
                raise DepthServiceError(f"{response.status}: {data[:200].decode(errors='replace')}")
            reply = lower_keys(dict(response.getheaders()))
            depth = unpack_depth(reply, data)
        except (OSError, http.client.HTTPException, DepthServiceError) as exc:
            self.close()
            if isinstance(exc, DepthServiceError):
                raise
            raise DepthServiceError(f"{type(exc).__name__}: {exc}") from exc
        t2 = time.perf_counter()
        self.last_infer_ms = float(reply.get("x-infer-ms", "0") or 0.0)
        self.timing["encode"].add(t1 - t0)
        self.timing["round_trip"].add(t2 - t1)
        out: Array = np.asarray(resize_depth(depth, (int(rgb.shape[0]), int(rgb.shape[1]))))
        return out

    def health(self) -> dict[str, Any]:
        """The service's ``/health`` as a dict; :class:`DepthServiceError` when it is not up.

        On a connection of its own, not the frames': a report timer may ask while the worker
        thread has a frame in flight, and two requests interleaved on one connection would
        scramble both."""
        conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            data = response.read()
            if response.status != 200:
                raise DepthServiceError(f"/health answered {response.status}")
        except (OSError, http.client.HTTPException) as exc:
            raise DepthServiceError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            conn.close()
        result: dict[str, Any] = json.loads(data)
        return result


class LazyDepth:
    """A :class:`DepthBackend` built on its first frame. The CPU model costs a gigabyte and
    seconds to load; behind :class:`Fallback` in remote or auto mode it is never asked while the
    service answers, and this is what keeps it from being paid for anyway. One build, under a
    lock: two frames racing the first call get one model. One attempt, too: a build that fails
    (no cached weights and no hub, no memory) is remembered and every later frame raises
    :class:`DepthModelError` at once — an uncached model re-entering the hub's timeouts on every
    frame would keep the worker inside ``from_pretrained`` for tens of seconds at a time instead
    of asking the service. What the failure means is the node's call; a restart is the retry."""

    def __init__(self, build: Callable[[], DepthBackend]) -> None:
        self._build = build
        self._backend: DepthBackend | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def built(self) -> bool:
        """Whether the backend exists yet."""
        return self._backend is not None

    @property
    def failed(self) -> str | None:
        """Why the build failed (``Type: message``, one line), or ``None`` while it has not."""
        return self._error

    def __call__(self, rgb: Rgb) -> Array:
        backend = self._backend
        if backend is None:
            with self._lock:
                backend = self._backend
                if backend is None:
                    backend = self._backend = self._build_once()
        return backend(rgb)

    def _build_once(self) -> DepthBackend:
        """The build, or the remembered failure raised again — with its cause only the first
        time, so the traceback is logged once and not per frame."""
        if self._error is not None:
            raise DepthModelError(f"the CPU model failed to build: {self._error}")
        t0 = time.perf_counter()
        log.info("building the local depth model")
        try:
            backend = self._build()
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:200]
            log.error("the local depth model failed to build: %s", self._error)
            raise DepthModelError(f"the CPU model failed to build: {self._error}") from exc
        log.info("local depth model ready in %.1f s", time.perf_counter() - t0)
        return backend


MODES = ("remote", "local", "auto")


class Fallback:
    """The switch between the service and the CPU model, as one :class:`DepthBackend`.

    ``auto``: the remote answers; a frame it fails on goes to the local model, and after
    ``failures`` consecutive failures the remote is left alone (no timeout paid per frame) and
    tried again on the first frame after every ``retry_s`` seconds — a success switches back.
    ``remote``: the service only, its errors raised to the caller. ``local``: the CPU model only.
    ``mode`` is settable live; :attr:`status` is the phrase for the node's report line.
    """

    def __init__(
        self,
        remote: DepthBackend,
        local: DepthBackend,
        mode: str = "auto",
        failures: int = 3,
        retry_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._remote, self._local = remote, local
        self._failures, self._retry_s, self._clock = failures, retry_s, clock
        self._mode = ""
        self.mode = mode
        self._consecutive = 0  # failures in a row while the remote is trusted
        self._down_since: float | None = None  # the moment the remote was given up on
        self._last_try = 0.0
        self.last_error = ""
        self.remote_frames = 0
        self.local_frames = 0
        self.failures = 0

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in MODES:
            raise ValueError(f"depth backend must be one of {MODES}, not {value!r}")
        if value != self._mode:
            self._mode = value
            self._consecutive, self._down_since = 0, None  # a new mode trusts the remote afresh

    @property
    def on_remote(self) -> bool:
        """Whether the next frame goes to the service first."""
        if self._mode == "remote":
            return True
        if self._mode == "local":
            return False
        if self._down_since is None:
            return True
        return self._clock() - self._last_try >= self._retry_s

    @property
    def status(self) -> str:
        """For the report line: ``remote``, ``local``, or ``auto`` with where the frames went."""
        if self._mode != "auto":
            return self._mode
        if self._down_since is None:
            return f"auto: remote ({self.local_frames} frames fell to local)"
        wait = max(0.0, self._retry_s - (self._clock() - self._last_try))
        return (
            f"auto: local since {self._clock() - self._down_since:.0f} s"
            f" ({self.last_error}; remote retried in {wait:.0f} s)"
        )

    def __call__(self, rgb: Rgb) -> Array:
        if self.on_remote:
            self._last_try = self._clock()
            try:
                depth = self._remote(rgb)
            except Exception as exc:
                if self._mode == "remote":
                    raise
                self._failed(exc)
            else:
                if self._down_since is not None:
                    log.info("depth service is back: frames go to the GPU again")
                self._consecutive, self._down_since = 0, None
                self.remote_frames += 1
                return depth
        self.local_frames += 1
        return self._local(rgb)

    def _failed(self, exc: Exception) -> None:
        self.failures += 1
        self.last_error = f"{exc}"[:120]
        if self._down_since is not None:
            return  # a probe that failed: the next one waits another retry_s
        self._consecutive += 1
        if self._consecutive >= self._failures:
            self._down_since = self._clock()
            log.warning(
                "depth service gave up after %d failures (%s): the CPU model answers, the"
                " service is retried every %.0f s",
                self._consecutive,
                self.last_error,
                self._retry_s,
            )


# ---------------------------------------------------------------- entry point and bench
def _bench_frames(frames_dir: str | None, count: int) -> list[Rgb]:
    """``count`` RGB frames from a directory of images, or synthetic 640x360 ones."""
    import cv2

    frames: list[Rgb] = []
    if frames_dir:
        from pathlib import Path

        for path in sorted(Path(frames_dir).iterdir()):
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is not None:
                frames.append(np.ascontiguousarray(bgr[:, :, ::-1]))
    if not frames:
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(count)]
    return (frames * (count // len(frames) + 1))[:count]


def bench(net: DepthNet, frames: list[Rgb], url: str | None) -> None:
    """Time the network here and, when ``url`` is given, the service's round trip on the same
    frames: fps and median/p95 ms per frame."""
    net(frames[0])  # the first call carries the device's warm-up
    local = LatencyTracker("local")
    for rgb in frames:
        with local.measure():
            net(rgb, (rgb.shape[0], rgb.shape[1]))
    s = local.summary()
    print(
        f"{net.name.rsplit('/', 1)[-1]} on {net.device}{' fp16' if net.half else ''},"
        f" {frames[0].shape[1]}x{frames[0].shape[0]}: {1e3 / s.median_ms:.1f} fps,"
        f" median {s.median_ms:.0f} ms, p95 {s.p95_ms:.0f} ms over {s.count} frames"
    )
    if not url:
        return
    for encoding in ("jpeg", "raw"):
        client = RemoteDepth(url, encoding=encoding)
        client(frames[0])
        tracker = LatencyTracker(encoding)
        for rgb in frames:
            with tracker.measure():
                client(rgb)
        rt = tracker.summary()
        print(
            f"service via {url} ({encoding}): {1e3 / rt.median_ms:.1f} fps, median"
            f" {rt.median_ms:.0f} ms, p95 {rt.p95_ms:.0f} ms, of which the server's inference"
            f" {client.last_infer_ms:.0f} ms"
        )
        client.close()


def main(argv: list[str] | None = None) -> None:
    """Serve the network (default), or ``--bench`` it and a running service on saved frames."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="small", help="small, base, large or a HF id")
    parser.add_argument("--device", default="mps", help="mps or cpu")
    parser.add_argument("--half", action="store_true", help="run the network in float16")
    parser.add_argument(
        "--host", default="127.0.0.1", help="bind address (loopback reaches the container)"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--bench", type=int, metavar="N", help="time N frames instead of serving")
    parser.add_argument("--frames", help="a directory of images for --bench (else synthetic)")
    parser.add_argument("--url", help="with --bench: also time this running service")
    args = parser.parse_args(argv)
    from pepin.log import setup_logging

    setup_logging("depth_service", log_dir=args.log_dir)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # the hub's HEAD checks, one per file
    t0 = time.perf_counter()
    net = DepthNet(args.model, args.device, args.half)
    net(np.zeros((360, 640, 3), dtype=np.uint8))  # the warm-up: the first frame compiles the graph
    log.info("%s loaded on %s in %.1f s", net.name, net.device, time.perf_counter() - t0)
    if args.bench:
        bench(net, _bench_frames(args.frames, args.bench), args.url)
        return
    server = DepthServer((args.host, args.port), net)
    log.info("depth service on http://%s:%d (POST /depth, GET /health)", args.host, args.port)
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        log.info("depth service stopped after %d frames", server.requests)


if __name__ == "__main__":
    main()
