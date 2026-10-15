"""The one camera channel without sockets: the record and its geometry on the rectified optics,
the ring and its waits, the paced detector on a fake clock, the sender's mailbox, the service's
receiver and health, and every consumer over a scripted service (the gaze's finder, person
recognition's shots and nets, see's first still frame, the live vision.source knob)."""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from pepin.expression.finder import (
    PersonBox,
    ServiceFinder,
    SwitchedFinder,
    frame_of,
)
from pepin.expression.speaker import FaceFrame, FaceSighting
from pepin.face_sensor.models import Detection
from pepin.face_sensor.quality import Quality
from pepin.people.config import FaceKnobs
from pepin.people.face import RoutedIdentity, ServiceIdentity, Shot, gate, shots
from pepin.perception.client import Jpeg, vision_cell
from pepin.perception.config import PerceptionConfig
from pepin.perception.detect import DetectLoop, Pacer, faces_of, persons_of
from pepin.perception.fakes import FakePerception, unit_vector
from pepin.perception.feed import FrameFeed
from pepin.perception.record import Face, Optics, Person, Record
from pepin.perception.ring import Records, Ring
from pepin.perception.source import SNAPSHOT, VisionSource
from pepin.perception_service import Identity, Perceiver
from pepin.tools import TOOLS
from pepin.tools.clients import ServiceCamera, ServiceDownError, SwitchedCamera
from pepin.tools.fakes import FakeCamera, FakeClock, FakeNeck, fake_robot
from pepin.voice_cascade.control import Request, write_request

# The robot's rectified left eye (config/stereo_calibration.json, 2026-10-04).
RECTIFIED = Optics("rectified-left", 454.473, 454.473, 405.278, 265.687, 800, 600)
KPS = ((380.0, 280.0), (420.0, 280.0), (400.0, 300.0), (385.0, 315.0), (415.0, 315.0))


class Clock:
    """Seconds that pass when a test says so."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


def face(seq_box: tuple[float, float, float, float] = (370.0, 260.0, 430.0, 330.0)) -> Face:
    return Face(seq_box, 0.8, KPS, 1.5, -2.0, 3.0, 42.0, 120.0)


def record(seq: int, stamp: float, faces: tuple[Face, ...] = (), **kw: Any) -> Record:
    return Record(seq, stamp, stamp + 0.08, faces=faces, optics=RECTIFIED.name, **kw)


# ---- the record ------------------------------------------------------------------------------
def test_the_optics_header_names_the_pixel_space_and_parses_back() -> None:
    text = RECTIFIED.header()
    assert text == "rectified-left 454.473 454.473 405.278 265.687 800 600"
    assert Optics.parse(text) == RECTIFIED
    for bad in ("", "rectified 454 454 400 300 800", "x 0 1 2 3 800 600", "x a b c d e f"):
        with pytest.raises(ValueError):
            Optics.parse(bad)


def test_a_bearing_is_the_exact_atan_through_the_rectified_pinhole_in_the_neck_s_signs() -> None:
    """Pan positive left, tilt positive down, pan about the vertical first: the principal point
    is straight ahead, a pixel fx to the right is 45 deg right, one fy below the principal point
    on its column is 45 deg down, and the far corner's tilt is taken after its pan."""
    o = RECTIFIED
    assert o.bearing_deg(o.cx, o.cy) == pytest.approx((0.0, 0.0))
    assert o.bearing_deg(o.cx + o.fx, o.cy) == pytest.approx((-45.0, 0.0))
    assert o.bearing_deg(o.cx - o.fx, o.cy) == pytest.approx((45.0, 0.0))
    assert o.bearing_deg(o.cx, o.cy + o.fy) == pytest.approx((0.0, 45.0))
    pan, tilt = o.bearing_deg(o.cx + o.fx, o.cy + o.fy)
    assert pan == pytest.approx(-45.0)
    assert tilt == pytest.approx(math.degrees(math.atan2(1.0, math.sqrt(2.0))))
    left, _ = o.bearing_deg(0.0, o.cy)
    right, _ = o.bearing_deg(800.0, o.cy)
    assert left - right == pytest.approx(82.8, abs=0.2), "the rectified eye is 82.8 deg wide"


def test_a_record_round_trips_through_json() -> None:
    person = Person((10.0, 20.0, 110.0, 400.0), 0.6, 30.0, 5.0, 31.0, -10.0, 40.0, cut=True)
    r = Record(7, 1_750_000_000.123456, 1_750_000_000.2, 3, "run", "rectified-left", 47.4,
               (face(),), (person,), {"total": 16.7})  # fmt: skip
    back = Record.from_dict(r.to_dict())
    assert back.seq == 7 and back.stamp == pytest.approx(r.stamp, abs=1e-6)
    assert back.faces[0].box == face().box and back.faces[0].kps == KPS
    assert back.faces[0].width_px == pytest.approx(60.0) and back.faces[0].yaw_deg == 3.0
    assert back.persons[0] == person and back.ms == {"total": 16.7}
    with pytest.raises((ValueError, KeyError, TypeError)):
        Record.from_dict({"faces": []})


def test_faces_and_persons_get_their_angles_quality_and_the_gaze_s_head_rule() -> None:
    image = np.full((600, 800, 3), 128, np.uint8)
    image[260:330, 370:430] = np.indices((70, 60)).sum(axis=0)[..., None] % 2 * 255
    centre = Detection((RECTIFIED.cx - 30, RECTIFIED.cy - 35, RECTIFIED.cx + 30,
                        RECTIFIED.cy + 35), 0.9, KPS)  # fmt: skip
    weak = Detection((0.0, 0.0, 10.0, 10.0), 0.3)
    (f,) = faces_of([centre, weak], image, RECTIFIED, min_score=0.5)
    assert (f.pan_deg, f.tilt_deg) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert f.yaw_deg == pytest.approx(0.0) and f.score == 0.9
    assert f.brightness > 0 and f.sharpness > 0
    upright = PersonBox(RECTIFIED.cx - 50, 100.0, RECTIFIED.cx + 50, 500.0, 0.7)
    lying = PersonBox(100.0, 300.0, 700.0, 400.0, 0.6)
    cut = PersonBox(300.0, 0.0, 400.0, 300.0, 0.5)
    stand, lie, top = persons_of([upright, lying, cut], RECTIFIED, head_frac=0.12)
    assert stand.pan_deg == pytest.approx(0.0, abs=1e-9) and not stand.cut
    assert stand.head_tilt_deg == pytest.approx(
        RECTIFIED.bearing_deg(RECTIFIED.cx, 100.0 + 0.12 * 400.0)[1]
    )
    assert lie.head_tilt_deg == pytest.approx(RECTIFIED.bearing_deg(400.0, 350.0)[1])
    assert top.cut and top.head_tilt_deg == pytest.approx(RECTIFIED.bearing_deg(350.0, 0.0)[1])


# ---- the ring ----------------------------------------------------------------------------------
def test_the_ring_numbers_frames_and_keeps_ten_seconds_of_arrivals() -> None:
    clock = Clock()
    ring = Ring(10.0, clock=clock, first=0)
    for k in range(25):  # 2 frames a second for 12.5 s
        ring.put(b"jpeg%d" % k, stamp=1000.0 + k * 0.5, optics=RECTIFIED, frame=k + 1)
        clock.sleep(0.5)
    assert ring.received == 25 and len(ring) == 21 and ring.span_s() == pytest.approx(10.0)
    assert ring.get(4) is None and ring.get(5) is not None, "older than 10 s: gone"
    newest = ring.newest()
    assert newest is not None and newest.seq == 25 and newest.frame == 25
    assert newest.jpeg == b"jpeg24" and newest.optics == RECTIFIED
    capped = Ring(10.0, clock=clock, first=0, max_frames=3)
    for k in range(5):
        capped.put(b"x", stamp=float(k), optics=RECTIFIED)
    assert len(capped) == 3 and capped.get(2) is None, "a burst never grows past the cap"


def test_the_ring_s_numbers_grow_across_a_restart_of_the_service() -> None:
    before = Ring(first=None)
    before.put(b"x", stamp=1.0, optics=RECTIFIED)
    after = Ring(first=None)
    frame = after.put(b"y", stamp=2.0, optics=RECTIFIED)
    old = before.newest()
    assert old is not None and frame.seq >= old.seq, "a client's since never hides new frames"


def test_the_newest_still_frame_is_waited_for_and_none_comes_back_in_time() -> None:
    ring = Ring(first=0)
    ring.put(b"moving", stamp=10.00, optics=RECTIFIED)
    assert ring.wait_newest(10.05, 0.0) is None, "exposed before the settle: not still"
    found = ring.wait_newest(None, 0.0)
    assert found is not None and found.jpeg == b"moving"
    later = threading.Timer(0.02, lambda: ring.put(b"still", stamp=10.10, optics=RECTIFIED))
    later.start()
    still = ring.wait_newest(10.05, 2.0)
    later.join()
    assert still is not None and still.jpeg == b"still"
    assert ring.wait_after(still.seq, 0.0) is None and ring.wait_after(0, 0.0) is still


def test_the_newest_record_is_long_polled_past_a_number_and_an_exposure() -> None:
    records = Records()
    assert records.wait(0, None, 0.0) is None
    records.publish(record(5, 10.0))
    assert records.wait(0, None, 0.0) == records.latest()
    assert records.wait(5, None, 0.0) is None, "seen already"
    assert records.wait(0, 10.5, 0.0) is None, "exposed before the head settled"
    later = threading.Timer(0.02, lambda: records.publish(record(6, 10.6)))
    later.start()
    got = records.wait(5, 10.5, 2.0)
    later.join()
    assert got is not None and got.seq == 6


# ---- the detector -------------------------------------------------------------------------
class Faces:
    def __init__(self, found: list[Detection]) -> None:
        self.found = found
        self.calls = 0

    def detect(self, image: Any) -> list[Detection]:
        self.calls += 1
        return self.found


class Persons:
    def __init__(self, found: list[PersonBox]) -> None:
        self.found = found
        self.jpegs: list[bytes] = []

    def detect_jpeg(self, jpeg: bytes, width: int, height: int) -> list[PersonBox]:
        self.jpegs.append(jpeg)
        return self.found


def test_the_detector_takes_the_newest_frame_at_ten_hertz_while_frames_come_at_twenty() -> None:
    clock = Clock()
    ring = Ring(clock=clock, first=0)
    records = Records()
    persons = Persons([PersonBox(300.0, 100.0, 500.0, 500.0, 0.7)])
    loop = DetectLoop(ring, records, Faces([Detection((380, 260, 440, 330), 0.9, KPS)]),
                      persons, hz=10.0, clock=clock,
                      decode=lambda jpeg: np.zeros((600, 800, 3), np.uint8))  # fmt: skip
    taken = []
    for k in range(40):  # two seconds of frames at 20 Hz
        ring.put(b"jpeg%d" % k, stamp=500.0 + k * 0.05, optics=RECTIFIED, brightness=40.0)
        got = loop.tick()
        if got is not None:
            taken.append(got.seq)
        assert loop.tick() is None, "the same frame is never taken twice"
        clock.sleep(0.05)
    assert len(taken) == 20 and taken[:3] == [1, 3, 5], "every other frame: 10 a second"
    assert loop.detected == 20 and loop.skipped == 19
    latest = records.latest()
    assert latest is not None and latest.seq == 39 and latest.stamp == pytest.approx(501.9)
    assert latest.optics == "rectified-left" and latest.brightness == 40.0
    assert len(latest.faces) == 1 and len(latest.persons) == 1
    assert persons.jpegs[-1] == b"jpeg38", "the person detector reads the JPEG bytes as sent"
    assert set(latest.ms) == {"decode", "faces", "persons", "total"}
    assert loop.rate.per_s() == pytest.approx(10.0, rel=0.06)


def test_the_pacer_spaces_starts_by_one_period() -> None:
    clock = Clock()
    pacer = Pacer(10.0, clock)
    assert pacer.wait_s() == 0.0
    pacer.started()
    clock.sleep(0.04)
    assert pacer.wait_s() == pytest.approx(0.06)
    clock.sleep(0.06)
    assert pacer.wait_s() == 0.0


def test_the_detector_thread_survives_a_detector_that_raises_and_stops_on_its_event() -> None:
    class Broken:
        def detect(self, image: Any) -> list[Detection]:
            raise RuntimeError("CoreML went away")

    ring, records = Ring(first=0), Records()
    loop = DetectLoop(ring, records, Broken(), None, hz=100.0,
                      decode=lambda jpeg: np.zeros((6, 8, 3), np.uint8))  # fmt: skip
    stop = threading.Event()
    thread = threading.Thread(target=loop.run, args=(stop,))
    thread.start()
    ring.put(b"a", stamp=1.0, optics=RECTIFIED)
    for _ in range(200):
        if loop.failed:
            break
        threading.Event().wait(0.005)
    stop.set()
    thread.join(2.0)
    assert not thread.is_alive() and loop.failed == 1 and "CoreML" in loop.last_error
    assert records.latest() is None


# ---- the sender -------------------------------------------------------------------------------
class Post:
    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.sent: list[tuple[bytes, dict[str, str]]] = []

    def __call__(self, body: bytes, headers: dict[str, str]) -> int:
        self.sent.append((body, headers))
        answer = self.answers.pop(0) if self.answers else 204
        if isinstance(answer, BaseException):
            raise answer
        return int(answer)


def feed(post: Post, clock: Clock, **kw: Any) -> FrameFeed:
    return FrameFeed("http://h:1", clock=clock, post=post, start=False,
                     encode=lambda image, q: b"jpeg-q%d" % q, **kw)  # fmt: skip


def test_the_sender_posts_the_picture_with_its_stamp_optics_and_brightness() -> None:
    clock, post = Clock(), Post()
    sender = feed(post, clock, quality=85)
    sender.offer(np.zeros((2, 2, 3), np.uint8), 1_750_000_000.25, RECTIFIED.header(), 47.4)
    assert sender.send_one()
    ((body, headers),) = post.sent
    assert body == b"jpeg-q85"
    assert headers["X-Stamp"] == "1750000000.250000" and headers["X-Frame"] == "1"
    assert headers["X-Optics"] == RECTIFIED.header() and headers["X-Brightness"] == "47.4"
    assert headers["X-Run"] == sender.run and sender.counts["sent"] == 1
    assert not sender.send_one(), "nothing waiting"


def test_the_mailbox_holds_one_frame_and_the_rate_cap_passes_frames_over() -> None:
    clock, post = Clock(), Post()
    sender = feed(post, clock, hz=20.0)
    image = np.zeros((2, 2, 3), np.uint8)
    sender.offer(image, 1.0, "o")
    clock.sleep(0.05)
    sender.offer(image, 2.0, "o")  # the first was never sent: replaced
    clock.sleep(0.01)
    sender.offer(image, 3.0, "o")  # 10 ms after the last taken: over the cap
    assert sender.send_one() and not sender.send_one()
    assert post.sent[0][1]["X-Stamp"] == "2.000000"
    assert sender.counts["dropped"] == 1 and sender.counts["paced"] == 1
    sender.hz = 10.0
    clock.sleep(0.05)
    sender.offer(image, 4.0, "o")  # 60 ms: under three quarters of 100
    clock.sleep(0.03)
    sender.offer(image, 5.0, "o")  # 90 ms: taken
    assert sender.counts["paced"] == 2 and sender.send_one()


def test_a_refused_service_is_left_alone_for_the_back_off_and_a_slow_one_costs_a_frame() -> None:
    clock = Clock()
    post = Post(ConnectionRefusedError(61, "refused"), TimeoutError("timed out"), 500)
    sender = feed(post, clock, backoff_s=2.0)
    image = np.zeros((2, 2, 3), np.uint8)
    sender.offer(image, 1.0, "o")
    sender.send_one()
    clock.sleep(0.5)
    sender.offer(image, 2.0, "o")  # inside the back-off: not even queued
    assert not sender.send_one() and sender.counts["down"] == 1
    assert "service down" in sender.report()
    clock.sleep(2.0)
    for k in range(2):
        sender.offer(image, 3.0 + k, "o")
        sender.send_one()
        clock.sleep(0.1)
    assert sender.counts == {"offered": 4, "sent": 0, "dropped": 0, "paced": 0, "refused": 1,
                             "failed": 2, "down": 1}  # fmt: skip
    line = sender.report()
    assert "0 dropped, 1 refused, 2 failed" in line and "HTTP 500" in line


def test_a_name_that_does_not_resolve_backs_off_like_a_refusal() -> None:
    import socket

    clock = Clock()
    sender = feed(Post(socket.gaierror(8, "nodename nor servname provided")), clock)
    sender.offer(np.zeros((2, 2, 3), np.uint8), 1.0, "o")
    sender.send_one()
    assert sender.counts["refused"] == 1 and "gaierror" in sender.report()


# ---- the service's receiver and health ---------------------------------------------------------
def perceiver(identity: Identity | None = None, clock: Clock | None = None) -> Perceiver:
    clock = clock or Clock()
    loop = DetectLoop(Ring(first=0), Records(), Faces([]), None, clock=clock,
                      decode=lambda jpeg: np.zeros((600, 800, 3), np.uint8))  # fmt: skip
    return Perceiver(loop, identity, faces="fake", persons="off", clock=clock)


def test_the_receiver_keeps_the_frame_and_refuses_a_request_that_is_not_one() -> None:
    p = perceiver()
    frame = p.receive({"X-Stamp": "12.5", "X-Optics": RECTIFIED.header(), "X-Frame": "9",
                       "X-Run": "r", "X-Brightness": "40"}, b"jpeg")  # fmt: skip
    assert frame.seq == 1 and frame.stamp == 12.5 and frame.frame == 9 and frame.run == "r"
    assert frame.optics == RECTIFIED and frame.brightness == 40.0
    for headers, body in (
        ({"X-Optics": RECTIFIED.header()}, b"jpeg"),
        ({"X-Stamp": "1", "X-Optics": "nonsense"}, b"jpeg"),
        ({"X-Stamp": "1", "X-Optics": RECTIFIED.header()}, b""),
    ):
        with pytest.raises(ValueError):
            p.receive(headers, body)


def test_health_carries_the_rates_the_stages_the_newest_record_and_the_clock_envelope() -> None:
    clock = Clock()
    p = perceiver(clock=clock)
    for _ in range(4):
        p.receive({"X-Stamp": f"{clock.now - 0.07:.6f}", "X-Optics": RECTIFIED.header()}, b"j")
        p.loop.tick()
        clock.sleep(0.1)
    h = p.health()
    assert h["frames"]["received"] == 4 and h["frames"]["ring"] == 4
    assert h["detect"]["detected"] == 4 and h["detect"]["hz"] == 10.0
    assert h["latest"]["faces"] == 0 and set(h["detect"]["ms"]) >= {"decode", "faces", "total"}
    assert h["embed"] is None
    assert vision_cell(h).startswith("vision ") and "0 faces 0 persons" in vision_cell(h)
    assert "frames/s in" in p.report()


def test_an_embedding_is_taken_of_a_face_in_a_frame_still_in_the_ring() -> None:
    class Net:
        name = "w600k_r50"

        def embed(self, image: Any, face: Detection) -> Any:
            assert image.shape == (600, 800, 3) and face.kps == KPS
            return unit_vector(1)

    loads: list[int] = []
    clock = Clock()

    def load() -> Any:
        loads.append(1)
        return Net()

    identity = Identity(load, "w600k_r50", unload_after_s=300.0, clock=clock)
    p = perceiver(identity, clock)
    seq = p.receive({"X-Stamp": "1", "X-Optics": RECTIFIED.header()}, b"jpeg").seq
    model, vector, _ms = p.embed(seq, [370, 260, 430, 330], [list(k) for k in KPS])
    assert model == "w600k_r50" and len(vector) == 512 and loads == [1]
    with pytest.raises(LookupError):
        p.embed(seq + 1, [0, 0, 1, 1], [list(k) for k in KPS])
    with pytest.raises(ValueError):
        p.embed(seq, [0, 0, 1, 1], [[0, 0]])
    assert not identity.idle() and identity.loaded
    clock.sleep(301.0)
    assert identity.idle() and not identity.loaded, "let go after the idle time"
    with pytest.raises(RuntimeError):
        perceiver(None).embed(1, [0, 0, 1, 1], [list(k) for k in KPS])


# ---- the gaze's finder -------------------------------------------------------------------------
def test_the_service_finder_asks_past_its_last_record_and_the_settle_and_holds_no_pixels() -> None:
    clock = FakeClock()
    person = Person((300.0, 100.0, 500.0, 500.0), 0.7, 1.0, 12.0, 1.0, -20.0, 45.0)
    service = FakePerception(
        [record(3, 10.0, (face(),), persons=(person,)), record(4, 10.2, (face(),))], clock=clock
    )
    finder = ServiceFinder(service, wait_s=0.4)
    found = finder.look(not_before=10.1)
    assert found is not None and found.seq == 4 and found.stamp == 10.2 and found.image is None
    assert service.calls[-1] == ("latest", {"since": 0, "min_stamp": 10.1, "timeout_s": 0.4})
    (f,) = found.faces
    assert (f.pan_deg, f.tilt_deg, f.width_px, f.yaw_deg) == (1.5, -2.0, 60.0, 3.0)
    assert f.detection is not None and f.detection.kps == KPS
    assert (f.sharpness, f.brightness) == (42.0, 120.0)
    assert finder.look() is None and clock.now == pytest.approx(0.4), "nothing newer: waited"
    assert service.calls[-1][1]["since"] == 4 and finder.missed == 1
    whole = frame_of(service.records[0])
    assert whole.bodies[0].head_tilt_deg == -20.0 and whole.bodies[0].half_width_deg == 12.0
    service.down = True
    assert finder.look() is None


def test_the_switched_finder_follows_vision_source_and_builds_the_snapshot_one_once() -> None:
    source = VisionSource()
    service = FakePerception([record(1, 1.0, (face(),))])
    built: list[int] = []

    class Snapshot:
        def look(self, not_before: float | None = None) -> FaceFrame | None:
            return FaceFrame(2.0, (), image=np.zeros((1, 1, 3), np.uint8))

    def build() -> Snapshot:
        built.append(1)
        return Snapshot()

    finder = SwitchedFinder(source, ServiceFinder(service), build)
    first = finder.look()
    assert first is not None and first.seq == 1 and built == []
    assert source.set(SNAPSHOT) is None
    assert finder.look() is not None and finder.look() is not None and built == [1]
    assert source.set("nope") is not None and source.value == SNAPSHOT


def test_the_gaze_asks_the_finder_for_a_frame_exposed_after_the_arbiter_s_blind_interval() -> None:
    from test_speaker_gaze import rig, view

    r = rig()
    r.door.view = view(blind_until=1234.5)
    r.person(10.0)
    read = r.wake(10.0)
    r.machine.tick(read)
    assert r.scene.asked and set(r.scene.asked) == {1234.5}


# ---- person recognition -------------------------------------------------------------------------
def test_a_service_frame_gives_shots_by_number_gated_on_the_service_s_measures() -> None:
    det = Detection((370.0, 260.0, 430.0, 330.0), 0.9, KPS)
    sighting = FaceSighting(0.0, 0.0, 60.0, 0.9, yaw_deg=0.0, detection=det, sharpness=42.0,
                            brightness=120.0)  # fmt: skip
    (shot,) = shots(FaceFrame(1.0, (sighting,), seq=77), None, 1.4)
    assert shot.image is None and shot.seq == 77 and shot.speaker
    assert shot.quality == Quality(42.0, 120.0) and gate(shot, FaceKnobs()) == "ok"
    assert gate(replace(shot, quality=Quality(3.0, 120.0)), FaceKnobs()) == "blurry"
    assert gate(replace(shot, quality=None), FaceKnobs()) == "unmeasured"
    assert shots(FaceFrame(1.0, (sighting,)), None, 1.4) == [], "no picture and no number"


def test_a_face_by_number_is_embedded_by_the_service_and_a_picture_here() -> None:
    service = FakePerception([record(5, 1.0)])
    remote = ServiceIdentity(service, "w600k_r50")
    det = Detection((0.0, 0.0, 50.0, 50.0), 0.9, KPS)
    vector = remote.embed(Shot(None, det, True, seq=5))
    assert np.allclose(vector, unit_vector(5)) and service.calls[-1][0] == "embed"
    with pytest.raises(RuntimeError, match="no embedding"):
        remote.embed(Shot(None, det, True, seq=6))  # left the ring

    class Local:
        loaded, warmed = False, 0

        def warm(self) -> None:
            self.warmed += 1

        def embed(self, shot: Shot) -> Any:
            return unit_vector(9)

        def unload(self) -> bool:
            return False

    source, local = VisionSource(), Local()
    routed = RoutedIdentity(local, remote, source, "w600k_r50")  # type: ignore[arg-type]
    routed.warm()
    assert local.warmed == 0, "with the service, no net loads in the voice process"
    assert np.allclose(routed.embed(Shot(None, det, True, seq=5)), unit_vector(5))
    picture = np.zeros((10, 10, 3), np.uint8)
    assert np.allclose(routed.embed(Shot(picture, det, True)), unit_vector(9))
    source.set(SNAPSHOT)
    routed.warm()
    assert local.warmed == 1 and routed.name == "w600k_r50"


# ---- see --------------------------------------------------------------------------------------
def jpeg(seq: int, stamp: float, still: bool = True) -> Jpeg:
    return Jpeg(b"\xff\xd8jpeg%d\xff\xd9" % seq, seq, stamp, 800, 600, "rectified-left", still)


def test_see_waits_for_the_head_then_asks_for_the_first_frame_after_it_settled() -> None:
    clock = FakeClock()
    neck = FakeNeck(clock, moving_until=0.12, still_from=55.5)
    camera = FakeCamera()
    robot = fake_robot(clock=clock, neck=neck, camera=camera)
    result = TOOLS.call("see", {}, robot)
    assert result["ok"] and neck.poses == 4, "polled every 50 ms while the head moved"
    ((not_before, wait),) = camera.asked
    assert not_before == 55.5 and wait == pytest.approx(0.3 - 0.15)
    robot.neck = FakeNeck(clock, moving_until=clock() + 10.0, still_from=None)
    camera.asked.clear()
    TOOLS.call("see", {}, robot)
    ((not_before, wait),) = camera.asked
    assert not_before is None and wait == 0.0, "never still within 0.3 s: the newest frame"


def test_see_hands_the_ring_s_bytes_with_its_frame_and_names_the_service_when_down() -> None:
    clock = FakeClock()
    service = FakePerception(frames=[jpeg(41, 55.4), jpeg(42, 55.6)], clock=clock)
    robot = fake_robot(clock=clock, camera=ServiceCamera(service))
    robot.neck.still_from = 55.5  # type: ignore[attr-defined]
    result = TOOLS.call("see", {}, robot)
    assert result["ok"] and result["frame"] == {"seq": 42, "still": True,
                                                "optics": "rectified-left"}  # fmt: skip
    robot.neck.still_from = 60.0  # type: ignore[attr-defined]
    late = TOOLS.call("see", {}, robot)
    assert late["ok"] and late["frame"]["still"] is False and clock() == pytest.approx(0.3)
    service.down = True
    down = TOOLS.call("see", {}, robot)
    assert not down["ok"] and "perception service" in down["why"]
    assert "ros/models.sh start perception" in down["why"]
    with pytest.raises(ServiceDownError, match=r"vision\.source=snapshot"):
        ServiceCamera(service).snapshot()
    service.down, service.frames = False, []
    with pytest.raises(ServiceDownError, match="no frame"):
        ServiceCamera(service).snapshot()


def test_the_switched_camera_follows_vision_source() -> None:
    source = VisionSource()
    service, snapshot = FakeCamera(jpeg=b"service"), FakeCamera(jpeg=b"snapshot")
    camera = SwitchedCamera(source, service, snapshot)
    assert camera.snapshot(1.0, 0.3).data == b"service" and service.asked == [(1.0, 0.3)]
    source.set(SNAPSHOT)
    assert camera.snapshot().data == b"snapshot"


# ---- the knob and the config -----------------------------------------------------------------
def test_a_knob_travels_in_the_control_file_and_the_config_names_the_start_value(
    tmp_path: Any,
) -> None:
    request = write_request(tmp_path, None, None, knobs={"vision.source": "snapshot"})
    data = (tmp_path / "control.json").read_text()
    back = Request.from_dict(__import__("json").loads(data))
    assert back.knobs == {"vision.source": "snapshot"} and back.id == request.id
    assert Request.from_dict({"knobs": {"a": 1, "b": " "}}).knobs == {}
    config = PerceptionConfig.load()
    # detect_hz 20 since 2026-10-14 (the camera lock on a person while the body turns)
    assert config.port == 8792 and config.source == "service" and config.detect_hz == 20.0
    assert VisionSource.load().value == config.source
    with pytest.raises(ValueError):
        PerceptionConfig.from_dict({"faces": "gpu"})
    with pytest.raises(ValueError):
        VisionSource("gpu")


def test_the_vision_cell_says_down_idle_or_the_rates() -> None:
    assert vision_cell(None).startswith("vision DOWN (127.0.0.1:8792: ros/models.sh start")
    assert "no frame yet" in vision_cell({"frames": {"idle_s": None}})
    assert "no frame for 12 s" in vision_cell({"frames": {"idle_s": 12.0}})
    cell = vision_cell({
        "frames": {"idle_s": 0.05, "per_s": 20.0},
        "detect": {"per_s": 10.0, "ms": {"total": {"median": 16.7}}},
        "latest": {"faces": 1, "persons": 2},
    })  # fmt: skip
    assert cell == "vision 20.0 in / 10.0 detected per s, detect 17 ms, 1 face 2 persons"
