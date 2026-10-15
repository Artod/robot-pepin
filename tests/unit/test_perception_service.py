"""The perception service over real loopback sockets: the sender's POSTs, the long polls, the
ring's JPEGs, the embeddings, the client's back-off; and, with the weights on this Mac, the real
detectors on a stored picture."""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from pepin.expression.finder import ServiceFinder
from pepin.face_sensor.models import Detection
from pepin.perception.client import PerceptionClient
from pepin.perception.config import PerceptionConfig
from pepin.perception.detect import DetectLoop
from pepin.perception.fakes import unit_vector
from pepin.perception.feed import FrameFeed
from pepin.perception.record import Optics
from pepin.perception.ring import Records, Ring
from pepin.perception_service import Identity, Perceiver, PerceptionServer

pytestmark = pytest.mark.slow  # real sockets, threads and waits

REPO = Path(__file__).resolve().parents[2]
MAIN = REPO.parents[2] if REPO.parent.parent.name == ".claude" else REPO  # a worktree's main
OPTICS = Optics("rectified-left", 454.473, 454.473, 405.278, 265.687, 800, 600)
KPS = ((380.0, 280.0), (420.0, 280.0), (400.0, 300.0), (385.0, 315.0), (415.0, 315.0))


class Faces:
    def detect(self, image: Any) -> list[Detection]:
        return [Detection((370.0, 260.0, 430.0, 330.0), 0.9, KPS)]


class Net:
    name = "w600k_r50"

    def embed(self, image: Any, face: Detection) -> Any:
        return unit_vector(int(face.box[0]))


def picture() -> np.ndarray:
    image = np.full((600, 800, 3), 90, np.uint8)
    image[260:330, 370:430] = 200
    return image


@pytest.fixture
def served() -> Iterator[tuple[str, Perceiver]]:
    """A service with a scripted face detector and identity net on an ephemeral port."""
    loop = DetectLoop(Ring(), Records(), Faces(), None, hz=50.0)
    perceiver = Perceiver(loop, Identity(Net, "w600k_r50"), faces="fake", persons="off")
    server = PerceptionServer(("127.0.0.1", 0), perceiver)
    stop = threading.Event()
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True),
        threading.Thread(target=loop.run, args=(stop,), daemon=True),
    ]
    for t in threads:
        t.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", perceiver
    finally:
        stop.set()
        server.shutdown()
        server.server_close()


def test_frames_posted_by_the_feed_are_detected_long_polled_and_served_back(
    served: tuple[str, Perceiver],
) -> None:
    url, _perceiver = served
    client = PerceptionClient(url)
    assert client.latest(0, None, 0.0) is None, "nothing yet: 204"
    feed = FrameFeed(url, quality=90, hz=0.0)
    try:
        t0 = time.time()
        feed.offer(picture(), t0, OPTICS.header(), 90.0)
        record = client.latest(0, t0 - 1.0, 2.0)
        assert record is not None and record.stamp == pytest.approx(t0, abs=1e-5)
        assert record.optics == "rectified-left" and record.brightness == 90.0
        (face,) = record.faces
        assert face.width_px == 60.0 and face.kps == KPS and face.brightness > 150
        pan, tilt = OPTICS.bearing_deg(400.0, 295.0)
        assert (face.pan_deg, face.tilt_deg) == pytest.approx((pan, tilt), abs=1e-3)
        jpeg = client.frame(record.seq)
        assert jpeg is not None and jpeg.seq == record.seq and jpeg.width == 800 and jpeg.still
        decoded = cv2.imdecode(np.frombuffer(jpeg.data, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (600, 800, 3)
        assert client.frame(record.seq + 1000) is None, "never sent: 404"

        # the first still frame: none after the settle yet, then the next one sent
        settle = t0 + 0.5
        late = client.still(settle, 0.05)
        assert late is not None and not late.still and late.seq == record.seq
        later = threading.Timer(0.05, lambda: feed.offer(picture(), settle + 0.01, OPTICS.header()))
        later.start()
        still = client.still(settle, 2.0)
        later.join()
        assert still is not None and still.still and still.stamp >= settle

        # the gaze's finder over it: past its last record, exposed after the settle
        finder = ServiceFinder(client, wait_s=2.0)
        frame = finder.look(not_before=settle)
        assert frame is not None and frame.seq == still.seq and frame.image is None

        # an embedding of the record's face, and of one in a frame the ring never had
        det = Detection(face.box, face.score, face.kps)
        vector = client.embed(record.seq, det)
        assert vector is not None and np.allclose(vector, unit_vector(370))
        assert client.embed(record.seq + 1000, det) is None
        health = client.health()
        assert health is not None and health["frames"]["received"] == 2
        assert health["embed"]["requests"] == 1 and health["detect"]["detected"] >= 2
        env = health["frames"]["arrival_minus_stamp"]
        assert env is not None and env["min_ms"] < 0.0, "the second was stamped ahead of now"
        assert feed.counts["sent"] == 2 and feed.counts["failed"] == 0
    finally:
        feed.close()


def test_a_bad_frame_is_refused_and_counted(served: tuple[str, Perceiver]) -> None:
    url, perceiver = served
    import http.client

    host, port = url.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=2.0)
    conn.request("POST", "/frame", body=b"jpeg", headers={"X-Optics": OPTICS.header()})
    response = conn.getresponse()
    assert response.status == 400 and b"X-Stamp" in response.read()
    conn.request("GET", "/nowhere")
    response = conn.getresponse()
    assert response.status == 404 and b"POST /frame" in response.read()
    conn.close()
    assert perceiver.refused == 1 and "X-Stamp" in perceiver.last_refusal


def test_a_client_of_a_service_that_is_not_there_backs_off_and_answers_none_at_once() -> None:
    import socket

    with socket.socket() as probe:  # a port nobody listens on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    clock = [0.0]
    client = PerceptionClient(f"http://127.0.0.1:{port}", retry_s=1.0, clock=lambda: clock[0])
    assert client.latest(0, None, 0.0) is None and client.down and client.failed == 1
    assert client.still(None, 0.0) is None and client.skipped == 1, "not even tried"
    clock[0] = 1.5
    assert client.health() is None and client.failed == 2, "tried again after the back-off"


def weights() -> Path | None:
    for root in (REPO, MAIN):
        if (root / "models/face/buffalo_l/det_10g.onnx").exists():
            return root / "models/face"
    return None


def sample() -> Path | None:
    spec = importlib.util.find_spec("insightface")
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin).parent / "data/images/t1.jpg"  # the package's own group photo
    return path if path.exists() else None


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision and CoreML: macOS only")
@pytest.mark.skipif(weights() is None, reason="no models/face/buffalo_l (models/ is gitignored)")
@pytest.mark.skipif(sample() is None, reason="needs the face group: uv sync --group face")
def test_the_real_detectors_find_faces_and_people_and_embed_one() -> None:
    """SCRFD on the CPU and the Neural Engine and Apple Vision from the JPEG bytes, on
    InsightFace's own group photo (six people, faces 40-60 px), letterboxed into the eye's
    800x600: every face found, people boxed, an ArcFace embedding of unit length."""
    from pepin.face_sensor.backends import ArcFace, Scrfd
    from pepin.perception.detect import VisionPersons
    from pepin.perception.feed import encode_jpeg

    models, path = weights(), sample()
    assert models is not None and path is not None
    photo = cv2.imread(str(path))
    eye = np.zeros((600, 800, 3), np.uint8)
    h, w = photo.shape[:2]
    scale = min(800 / w, 600 / h)
    small = cv2.resize(photo, (round(w * scale), round(h * scale)))
    eye[: small.shape[0], : small.shape[1]] = small
    detector = Scrfd(models / "buffalo_l", device="ane", cache=models / "coreml_cache",
                     det_threshold=0.6)  # fmt: skip
    config = PerceptionConfig.load()
    loop = DetectLoop(Ring(), Records(), detector, VisionPersons(), hz=config.detect_hz)
    frame = loop.ring.put(encode_jpeg(eye, 90), stamp=1.0, optics=OPTICS)
    record = loop.detect(frame)
    assert len(record.faces) >= 5, f"{len(record.faces)} faces"
    assert record.persons, "Vision found no person"
    assert all(-42 < f.pan_deg < 42 and f.sharpness > 0 for f in record.faces)
    assert record.ms["total"] < 500.0
    arc = ArcFace(models / "buffalo_l/w600k_r50.onnx", device="ane", cache=models / "coreml_cache")
    identity = Identity(lambda: arc, "w600k_r50")
    perceiver = Perceiver(loop, identity)
    best = max(record.faces, key=lambda f: f.width_px)
    model, vector, _ms = perceiver.embed(frame.seq, best.box, best.kps)
    assert model == "w600k_r50" and np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-3)
