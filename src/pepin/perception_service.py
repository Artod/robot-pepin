"""The perception service: the robot's one camera channel, fanned out on this Mac (:8792).

camera_stream (pepin-vslam) POSTs each rectified left eye it publishes (:mod:`pepin.perception.
feed`); this process keeps the last ``ring_s`` of them, finds faces and people on the newest one
at ``detect_hz``, and answers everyone on this Mac who wants the camera, so the board has one
reader and the voice process holds no pixels (:mod:`pepin.perception`):

==========================================  ==================================================
``POST /frame``                             a JPEG and its headers (X-Stamp, X-Frame, X-Run,
                                            X-Optics, X-Brightness); 204 at once
``GET /latest?since=&min_stamp=&timeout=``  the newest record past a number and an exposure
                                            stamp, long-polled; 204 when none came in time
``GET /frame/<n>.jpg``                      frame n of the ring as it came; 404 once it left
``GET /frame/newest.jpg?min_stamp=&timeout=``  the newest frame exposed after the stamp, waited
                                            for, else the newest there is (``X-Still: 0``)
``POST /embed``                             ArcFace of ``{"seq", "box", "kps"}``, on demand
``GET /health``                             rates, ms per stage, the newest record's counts,
                                            the clock envelope (arrival minus exposure stamp)
==========================================  ==================================================

THREADS. The receiver is the HTTP thread of camera_stream's kept-alive connection: it reads the
body and puts it into the ring, nothing else. One detector thread (:class:`pepin.perception.
detect.DetectLoop`) decodes and detects. Long polls wait on conditions, holding no lock. The
identity net loads at the first ``/embed`` and is let go after ``identity_unload_after_s`` idle.

COMPUTE. SCRFD on CoreML's CPU and Neural Engine (``faces: ane``), Apple Vision on the Neural
Engine, ArcFace on demand (``identity_device``): nothing here asks for the GPU the voice's TTS
uses. Started and stopped by ``ros/models.sh start|stop perception`` (a launchd job, bound to
127.0.0.1: the container reaches it as host.docker.internal).
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import statistics
import sys
import threading
import time
import urllib.parse
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

from pepin.face_sensor.models import Detection, Embedder
from pepin.perception.config import PerceptionConfig
from pepin.perception.detect import (
    DetectLoop,
    FaceDetector,
    PersonDetector,
    Rate,
    VisionPersons,
)
from pepin.perception.record import Optics
from pepin.perception.ring import Frame, Records, Ring
from pepin.telemetry import LatencyTracker

log = logging.getLogger("pepin.perception_service")

MAX_WAIT_S = 5.0  # the longest long poll a client may ask for
REPORT_S = 60.0
ENVELOPE_FRAMES = 400  # arrival-minus-stamp samples the clock envelope is taken over
IDLE_CHECK_S = 10.0


class Identity:
    """ArcFace on demand: loaded at the first embedding, let go after ``unload_after_s``
    without one (0: kept); thread-safe."""

    def __init__(
        self,
        load: Callable[[], Embedder],
        name: str,
        *,
        unload_after_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``load`` builds the net (0.1 s with a warm CoreML cache); ``name`` is its model's
        name, the key person recognition files its samples under."""
        self._load, self.name = load, name
        self.unload_after_s = float(unload_after_s)
        self._clock = clock
        self._net: Embedder | None = None
        self._lock = threading.Lock()
        self._last = clock()
        self.loads = 0
        self.requests = 0
        self.failed = 0
        self.last_error = ""
        self.ms = LatencyTracker("embed")

    @property
    def loaded(self) -> bool:
        """Whether the net is in memory."""
        return self._net is not None

    def embed(self, image: Any, face: Detection) -> Any:
        """The unit-length embedding of ``face`` in ``image`` (BGR)."""
        with self._lock:
            if self._net is None:
                t0 = time.perf_counter()
                self._net = self._load()
                self.loads += 1
                log.info("identity net %s loaded in %.2f s", self.name, time.perf_counter() - t0)
            t0 = time.perf_counter()
            vector = self._net.embed(image, face)
            self.ms.add(time.perf_counter() - t0)
            self._last = self._clock()
            self.requests += 1
            return vector

    def idle(self) -> bool:
        """Let the net go when it has been idle ``unload_after_s``; whether it was let go."""
        with self._lock:
            if self._net is None or self.unload_after_s <= 0:
                return False
            if self._clock() - self._last < self.unload_after_s:
                return False
            self._net = None
        log.info("identity net %s let go after %.0f s idle", self.name, self.unload_after_s)
        return True


class Perceiver:
    """What the HTTP handler calls: the ring, the newest record, the detector, the identity
    net, and the counters ``/health`` reports. ROS-free and socket-free (the tests drive it)."""

    def __init__(
        self,
        loop: DetectLoop,
        identity: Identity | None,
        *,
        faces: str = "",
        persons: str = "",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``faces`` / ``persons``: what runs, in words, for ``/health``."""
        self.loop, self.ring, self.records = loop, loop.ring, loop.records
        self.identity = identity
        self.faces, self.persons = faces, persons
        self._clock = clock
        self._since = clock()
        self.received = Rate(clock=clock)
        self.refused = 0
        self.last_refusal = ""
        self._last_arrival: float | None = None
        self._ages: deque[float] = deque(maxlen=ENVELOPE_FRAMES)
        self._optics: dict[str, Optics] = {}

    # ---- the receiver ------------------------------------------------------------------------
    def receive(self, headers: Mapping[str, str], body: bytes) -> Frame:
        """One POSTed frame into the ring; ``ValueError`` for a request that is not one."""
        h = {k.lower(): v for k, v in headers.items()}
        if not body:
            raise ValueError("an empty frame")
        try:
            stamp = float(h["x-stamp"])
        except (KeyError, ValueError) as exc:
            raise ValueError("X-Stamp (the board's exposure stamp, seconds) is required") from exc
        text = h.get("x-optics", "")
        optics = self._optics.get(text)
        if optics is None:
            optics = Optics.parse(text)
            self._optics[text] = optics
        frame = self.ring.put(
            body,
            stamp=stamp,
            optics=optics,
            frame=int(h.get("x-frame", "0") or 0),
            run=h.get("x-run", ""),
            brightness=float(h.get("x-brightness", "0") or 0.0),
        )
        self.received.add()
        self._last_arrival = self._clock()
        self._ages.append(frame.arrived - frame.stamp)
        return frame

    # ---- the readers -------------------------------------------------------------------------
    def embed(self, seq: int, box: Any, kps: Any) -> tuple[str, list[float], float]:
        """(model, embedding, ms) for a face of frame ``seq``; ``LookupError`` when the frame
        has left the ring, ``RuntimeError`` without an identity net, ``ValueError`` for a face
        without its five points."""
        if self.identity is None:
            raise RuntimeError("no identity net here (perception.identity_device, the weights)")
        frame = self.ring.get(seq)
        if frame is None:
            raise LookupError(f"frame {seq} is not in the ring (it keeps {self.ring.keep_s:g} s)")
        x0, y0, x1, y1 = (float(v) for v in box)
        points = tuple((float(x), float(y)) for x, y in kps)
        if len(points) < 5:
            raise ValueError("a face needs its five points to be aligned")
        image = self.loop.decode(frame.jpeg)
        if image is None:
            raise ValueError(f"frame {seq} is not a JPEG")
        t0 = time.perf_counter()
        try:
            vector = self.identity.embed(image, Detection((x0, y0, x1, y1), 1.0, points))
        except Exception as exc:
            self.identity.failed += 1
            self.identity.last_error = f"{type(exc).__name__}: {exc}"[:200]
            raise RuntimeError(self.identity.last_error) from exc
        ms = (time.perf_counter() - t0) * 1e3
        return self.identity.name, [float(v) for v in vector], ms

    def health(self) -> dict[str, Any]:
        """What ``GET /health`` says."""
        now = self._clock()
        loop = self.loop
        latest = self.records.latest()
        ages = sorted(self._ages)
        envelope = (
            {
                "min_ms": round(ages[0] * 1e3, 1),
                "median_ms": round(statistics.median(ages) * 1e3, 1),
            }
            if ages
            else None
        )
        identity = self.identity
        return {
            "service": "perception",
            "uptime_s": round(now - self._since, 1),
            "frames": {
                "received": self.ring.received,
                "per_s": round(self.received.per_s(), 2),
                "refused": self.refused,
                "last_refusal": self.last_refusal,
                "idle_s": None
                if self._last_arrival is None
                else round(now - self._last_arrival, 2),
                "ring": len(self.ring),
                "ring_s": round(self.ring.span_s(), 2),
                "arrival_minus_stamp": envelope,
            },
            "detect": {
                "hz": loop.hz,
                "per_s": round(loop.rate.per_s(), 2),
                "detected": loop.detected,
                "skipped": loop.skipped,
                "failed": loop.failed,
                "last_error": loop.last_error,
                "faces": self.faces,
                "persons": self.persons,
                "ms": {
                    stage: {
                        "median": round(s.median_ms, 2),
                        "p95": round(s.p95_ms, 2),
                        "max": round(s.max_ms, 2),
                    }
                    for stage, s in ((k, t.summary()) for k, t in loop.timing.items())
                },
            },
            "latest": None
            if latest is None
            else {
                "seq": latest.seq,
                "stamp": latest.stamp,
                "faces": len(latest.faces),
                "persons": len(latest.persons),
                "brightness": latest.brightness,
            },
            "embed": None
            if identity is None
            else {
                "model": identity.name,
                "loaded": identity.loaded,
                "loads": identity.loads,
                "requests": identity.requests,
                "failed": identity.failed,
                "last_error": identity.last_error,
                "ms": {
                    "median": round(identity.ms.summary().median_ms, 2),
                    "p95": round(identity.ms.summary().p95_ms, 2),
                },
            },
        }

    def report(self) -> str:
        """One line for the log: rates, the detector's ms, the newest record, the ages."""
        h = self.health()
        f, d = h["frames"], h["detect"]
        ms = d["ms"]
        latest = h["latest"] or {}
        env = f["arrival_minus_stamp"]
        ages = (
            f"arrival - stamp min/median {env['min_ms']:.0f}/{env['median_ms']:.0f} ms"
            if env
            else "no ages"
        )
        return (
            f"perception: {f['per_s']:.1f} frames/s in ({f['refused']} refused), ring"
            f" {f['ring']} frames {f['ring_s']:.1f} s; detect {d['per_s']:.1f}/s of {d['hz']:g}"
            f" ({d['skipped']} skipped, {d['failed']} failed), total"
            f" {ms['total']['median']:.1f}/{ms['total']['p95']:.1f} ms (decode"
            f" {ms['decode']['median']:.1f}, faces {ms['faces']['median']:.1f}, persons"
            f" {ms['persons']['median']:.1f}); latest {latest.get('faces', 0)} faces"
            f" {latest.get('persons', 0)} persons; {ages}"
        )


class PerceptionServer(http.server.ThreadingHTTPServer):
    """The HTTP side: one thread per connection (camera_stream's kept-alive POSTs are one)."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], perceiver: Perceiver) -> None:
        super().__init__(address, PerceptionHandler)
        self.perceiver = perceiver
        self.clients_gone = 0

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A client that left before its answer is counted, not traced."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            self.clients_gone += 1
            return
        log.exception("a request from %s failed", client_address)


class PerceptionHandler(http.server.BaseHTTPRequestHandler):
    """The routes of the module's table; HTTP/1.1 keep-alive."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass  # the service reports itself

    @property
    def perceiver(self) -> Perceiver:
        server: PerceptionServer = self.server  # type: ignore[assignment]
        return server.perceiver

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Content-Length is not a number")
            return
        body = self.rfile.read(length) if length else b""
        path = urllib.parse.urlsplit(self.path).path
        if path == "/frame":
            try:
                self.perceiver.receive(dict(self.headers.items()), body)
            except ValueError as exc:
                self.perceiver.refused += 1
                self.perceiver.last_refusal = str(exc)[:200]
                self._reply(400, "text/plain", f"{exc}\n".encode())
                return
            self._reply(204, "text/plain", b"")
        elif path == "/embed":
            self._embed(body)
        else:
            self._reply(404, "text/plain", ROUTES.encode())

    def do_GET(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(url.query)
        p = self.perceiver
        if url.path == "/health":
            self._json(200, p.health())
        elif url.path == "/latest":
            try:
                since = int(_one(query, "since", "0"))
                min_stamp = _stamp(query)
                wait = _wait(query)
            except ValueError as exc:
                self._reply(400, "text/plain", f"{exc}\n".encode())
                return
            record = p.records.wait(since, min_stamp, wait)
            if record is None:
                self._reply(204, "text/plain", b"")
            else:
                self._json(200, record.to_dict())
        elif url.path == "/frame/newest.jpg":
            try:
                min_stamp, wait = _stamp(query), _wait(query)
            except ValueError as exc:
                self._reply(400, "text/plain", f"{exc}\n".encode())
                return
            frame = p.ring.wait_newest(min_stamp, wait)
            still = frame is not None
            frame = frame or p.ring.newest()
            if frame is None:
                self._reply(404, "text/plain", b"no frame yet\n")
            else:
                self._frame(frame, still)
        elif url.path.startswith("/frame/") and url.path.endswith(".jpg"):
            try:
                seq = int(url.path[len("/frame/") : -len(".jpg")])
            except ValueError:
                self._reply(404, "text/plain", ROUTES.encode())
                return
            kept = p.ring.get(seq)
            if kept is None:
                self._reply(404, "text/plain", f"frame {seq} is not in the ring\n".encode())
            else:
                self._frame(kept, True)
        else:
            self._reply(404, "text/plain", ROUTES.encode())

    def _embed(self, body: bytes) -> None:
        try:
            data = json.loads(body)
            seq, box, kps = int(data["seq"]), data["box"], data["kps"]
        except (KeyError, ValueError, TypeError) as exc:
            self._reply(400, "text/plain", f"seq, box and kps are required: {exc}\n".encode())
            return
        try:
            model, vector, ms = self.perceiver.embed(seq, box, kps)
        except LookupError as exc:
            self._reply(404, "text/plain", f"{exc}\n".encode())
            return
        except (ValueError, TypeError) as exc:
            self._reply(400, "text/plain", f"{exc}\n".encode())
            return
        except RuntimeError as exc:
            self._reply(503, "text/plain", f"{exc}\n".encode())
            return
        self._json(200, {"model": model, "embedding": vector, "ms": round(ms, 2)})

    def _frame(self, frame: Frame, still: bool) -> None:
        self._reply(
            200,
            "image/jpeg",
            frame.jpeg,
            {
                "X-Seq": str(frame.seq),
                "X-Stamp": f"{frame.stamp:.6f}",
                "X-Arrived": f"{frame.arrived:.6f}",
                "X-Optics": frame.optics.header(),
                "X-Width": str(frame.optics.width),
                "X-Height": str(frame.optics.height),
                "X-Still": "1" if still else "0",
            },
        )

    def _json(self, status: int, data: Any) -> None:
        self._reply(status, "application/json", json.dumps(data).encode())

    def _reply(
        self, status: int, kind: str, body: bytes, extra: Mapping[str, str] | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if body:
            self.wfile.write(body)


ROUTES = (
    "POST /frame, GET /latest?since=&min_stamp=&timeout=, GET /frame/<n>.jpg,"
    " GET /frame/newest.jpg?min_stamp=&timeout=, POST /embed, GET /health\n"
)


def _one(query: Mapping[str, list[str]], key: str, default: str) -> str:
    values = query.get(key)
    return values[0] if values else default


def _stamp(query: Mapping[str, list[str]]) -> float | None:
    text = _one(query, "min_stamp", "")
    return float(text) if text else None


def _wait(query: Mapping[str, list[str]]) -> float:
    return min(MAX_WAIT_S, max(0.0, float(_one(query, "timeout", "0"))))


# ---- building it -------------------------------------------------------------------------------
def build_faces(config: PerceptionConfig) -> tuple[FaceDetector | None, str]:
    """SCRFD on ``config.faces`` (CoreML's units) and its description; None for ``none``."""
    if config.faces == "none":
        return None, "off"
    from pepin.face_sensor.backends import Scrfd
    from pepin.face_sensor.config import FaceSensorConfig

    cfg = FaceSensorConfig.load()
    m = cfg.models
    detector = Scrfd(
        cfg.path(m.detector_pack),
        device=config.faces,
        cache=cfg.path(m.coreml_cache),
        det_size=m.det_size,
        det_threshold=m.det_threshold,
    )
    return detector, f"{detector.name} on {detector.device} (threshold {m.det_threshold})"


def build_persons(config: PerceptionConfig) -> tuple[PersonDetector | None, str]:
    """Apple Vision's human rectangles, or None (``none``, or no pyobjc-framework-Vision)."""
    if config.persons == "none":
        return None, "off"
    vision = VisionPersons()
    try:
        vision.warm()
    except ImportError as exc:
        log.warning("no person detector (%s): faces only", exc)
        return None, f"off ({exc})"
    return vision, "Apple Vision human rectangles"


def build_identity(config: PerceptionConfig) -> Identity | None:
    """ArcFace (config/people.json's ``face.model``) on ``identity_device``, loaded on demand;
    None when the weights are missing."""
    from pepin.people.config import PeopleConfig

    people = PeopleConfig.load()
    model = people.path(people.face.model)
    if not model.exists():
        log.warning("no identity net at %s: /embed answers 503", model)
        return None
    cache = people.path("models/face/coreml_cache")

    def load() -> Embedder:
        from pepin.face_sensor.backends import ArcFace

        return ArcFace(model, device=config.identity_device, cache=cache)

    return Identity(load, model.stem, unload_after_s=config.identity_unload_after_s)


def warm(loop: DetectLoop, size: tuple[int, int] = (800, 600)) -> None:
    """One grey frame through both detectors, so the first real frame pays no compilation."""
    import numpy as np

    from pepin.perception.feed import encode_jpeg

    image = np.full((size[1], size[0], 3), 96, dtype=np.uint8)
    jpeg = encode_jpeg(image, 90)
    optics = Optics("warm", 450.0, 450.0, size[0] / 2, size[1] / 2, size[0], size[1])
    ring = Ring(1.0)
    loop.detect(ring.put(jpeg, stamp=0.0, optics=optics))
    loop.reset_timing()  # the warm-up is not a frame of the robot's


def main(argv: list[str] | None = None) -> None:
    """Serve the camera channel on the loopback (config/models.json's ``perception`` block)."""
    from pepin.model_service import port_taken

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", help="the models config (default: config/models.json)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, help="overrides the config's port")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args(argv)
    from pepin.log import setup_logging

    setup_logging("perception_service", log_dir=args.log_dir)
    config = PerceptionConfig.load(args.config)
    port = args.port or config.port
    taken = port_taken(args.host, port)
    if taken is not None:  # before any model loads: launchd restarts a failed job
        log.error("perception service not started: %s", taken)
        raise SystemExit(2)
    t0 = time.perf_counter()
    faces, faces_text = build_faces(config)
    persons, persons_text = build_persons(config)
    loop = DetectLoop(
        Ring(config.ring_s),
        Records(),
        faces,
        persons,
        hz=config.detect_hz,
        head_frac=config.body_head_frac,
        min_score=config.min_score,
    )
    warm(loop)
    perceiver = Perceiver(loop, build_identity(config), faces=faces_text, persons=persons_text)
    server = PerceptionServer((args.host, port), perceiver)
    stop = threading.Event()
    threading.Thread(target=loop.run, args=(stop,), name="detect", daemon=True).start()
    threading.Thread(
        target=_housekeeping, args=(perceiver, stop), name="report", daemon=True
    ).start()
    log.info(
        "perception service on http://%s:%d in %.1f s: faces %s, persons %s, detect %.0f Hz,"
        " ring %.0f s, identity %s (GET /health)",
        args.host,
        port,
        time.perf_counter() - t0,
        faces_text,
        persons_text,
        config.detect_hz,
        config.ring_s,
        "on demand" if perceiver.identity is not None else "none",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        log.info("perception service stopped")


def _housekeeping(perceiver: Perceiver, stop: threading.Event) -> None:
    """The report line every minute a frame came, and the identity net's idle check."""
    last_report = time.monotonic()
    received = perceiver.ring.received
    while not stop.wait(IDLE_CHECK_S):
        if perceiver.identity is not None:
            perceiver.identity.idle()
        if time.monotonic() - last_report >= REPORT_S:
            last_report = time.monotonic()
            if perceiver.ring.received != received:
                received = perceiver.ring.received
                log.info(perceiver.report())


if __name__ == "__main__":
    main()
