"""The speaker gaze's two outside seams: the arbiter's door (pepin.expression.door) over a fake
client, and the finder (pepin.expression.finder) over a fake camera and detectors: faces, and
people's boxes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pepin.expression.door import SpeakerDoor, answer_of, view_of
from pepin.expression.finder import (
    PersonBox,
    Picture,
    SnapshotFinder,
    VisionBodies,
    body_sighting,
    head_yaw_deg,
    snapshot_url,
)
from pepin.expression.speaker import Ease, Point
from pepin.face_sensor.geometry import Lens
from pepin.face_sensor.models import Detection
from pepin.gaze import Aim


class Client:
    """The door's client: answers as scripted, or raises ``error``."""

    def __init__(self, answer: dict[str, Any] | None = None, error: Exception | None = None):
        self.answer = answer or {}
        self.error = error
        self.calls: list[tuple[str, Any]] = []

    def _go(self, name: str, arg: Any) -> dict[str, Any]:
        self.calls.append((name, arg))
        if self.error is not None:
            raise self.error
        return self.answer

    def look(self, request: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        return self._go("look", (request, timeout_s))

    def renew(self) -> int:
        return int(self._go("renew", None).get("renewed", 0))

    def release(self, fade_s: float = 0.0) -> int:
        return int(self._go("release", fade_s).get("released", 0))

    def state(self) -> dict[str, Any]:
        return self._go("state", None)


def door(client: Client) -> SpeakerDoor:
    return SpeakerDoor(client=client, aim_wait_s=2.0)  # type: ignore[arg-type]


def test_an_aim_is_a_held_point_look_in_the_map_that_follows_of_the_idle_band() -> None:
    """The person is a point in the map: the arbiter resolves it from TF's pose of now and keeps
    the head on it as the cart turns (``follow``), held for its TTL, no depth frames; the
    answer carries where the head settled."""
    client = Client({"status": "done", "pan_rad": 0.5, "tilt_rad": -0.1})
    answer = door(client).aim(Point(1.8, 0.6, 1.1), speed="slow", ttl_s=6.0)
    assert answer.status == "done" and answer.head == Aim(0.5, -0.1)
    ((name, (request, wait)),) = client.calls
    assert name == "look" and wait == 2.0
    assert request == {
        "kind": "point",
        "band": 5,
        "target": {"frame": "map", "x": 1.8, "y": 0.6, "z": 1.1},
        "follow": True,
        "frames": 0,
        "hold": True,
        "ttl_s": 6.0,
        "speed": "slow",
    }


def test_a_door_that_keeps_an_aim_waiting_or_is_away_answers_in_words() -> None:
    here = Point(1.0, 0.0, 1.0)
    waiting = door(Client(error=TimeoutError("timed out"))).aim(here, speed="saccade", ttl_s=6)
    assert waiting.status == "queued" and "held behind" in waiting.reason
    away = door(Client(error=ConnectionRefusedError(61, "refused")))
    assert away.aim(here, speed="saccade", ttl_s=6).status == "denied"
    assert away.renew() == 0 and away.state() is None
    away.release()
    assert away.calls == 4


def test_the_state_as_the_speaker_gaze_reads_it() -> None:
    seen = view_of(
        {
            "pan_rad": 0.3,
            "tilt_rad": 0.1,
            "source": "voice.speaker",
            "band": 5,
            "blind": True,
            "blind_until": None,
            "driving": False,
            "overlay": {"source": "voice.gesture", "loop": False},
        }
    )
    assert seen.head == Aim(0.3, 0.1) and seen.holder == "voice.speaker" and seen.blind
    assert seen.overlay == "voice.gesture" and seen.blind_until is None
    bare = view_of({"pan_rad": None, "tilt_rad": float("nan"), "blind_until": 12.5})
    assert bare.head is None and bare.holder == "" and bare.band == 6 and bare.overlay == ""
    assert bare.blind_until == 12.5
    assert answer_of({"status": "denied", "reason": "band"}).head is None
    assert door(Client({"renewed": 2})).renew() == 2


class Camera:
    def __init__(self, picture: Picture | None) -> None:
        self.picture = picture

    def grab(self) -> Picture | None:
        return self.picture


class Faces:
    def __init__(self, found: list[Detection]) -> None:
        self.found = found
        self.images: list[Any] = []

    def detect(self, image: Any) -> list[Detection]:
        self.images.append(image)
        return self.found


def test_the_finder_bears_each_face_through_the_lens_and_loads_its_detector_once() -> None:
    lens = Lens.nominal(800, 600, 90.0)  # fx 400: a face 400 px right of centre is 45 deg right
    centred = Detection((380.0, 280.0, 420.0, 320.0), 0.9, ((390, 290), (410, 290), (400, 300)))
    right = Detection((780.0, 280.0, 820.0, 320.0), 0.7)
    loads: list[int] = []

    def load() -> Faces:
        loads.append(1)
        return Faces([centred, right])

    picture = Picture(np.zeros((600, 800, 3), dtype=np.uint8), 1234.5)
    finder = SnapshotFinder(Camera(picture), load, lens)
    finder.warm()
    found = finder.look()
    assert found is not None and found.stamp == 1234.5 and len(found.faces) == 2
    near, far = found.faces
    assert near.pan_deg == pytest.approx(0.0) and near.tilt_deg == pytest.approx(0.0)
    assert far.pan_deg == pytest.approx(-45.0), "pan + left: a face at the right edge is -45"
    assert near.yaw_deg == pytest.approx(0.0) and far.yaw_deg is None
    half = Picture(np.zeros((300, 400, 3), dtype=np.uint8), 0.0)  # the lens scales with it
    halved = SnapshotFinder(Camera(half), lambda: Faces([right]), lens).look()
    assert halved is not None and halved.faces[0].pan_deg == pytest.approx(
        -math.degrees(math.atan(600 / 200))
    )
    assert near.width_px == 40.0 and finder.timing.looks == 1
    finder.look()
    assert loads == [1]
    assert SnapshotFinder(Camera(None), load, lens).look() is None
    picky = SnapshotFinder(Camera(picture), load, lens, min_score=0.8).look()
    assert picky is not None and len(picky.faces) == 1


def test_the_head_s_yaw_from_the_five_points_and_the_snapshot_beside_the_stream() -> None:
    eyes = ((100.0, 100.0), (140.0, 100.0))
    assert head_yaw_deg(Detection((0, 0, 1, 1), 0.9, (*eyes, (130.0, 120.0)))) == pytest.approx(
        30.0
    )
    assert head_yaw_deg(Detection((0, 0, 1, 1), 0.9, (*eyes, (200.0, 120.0)))) == 90.0
    assert snapshot_url("http://10.0.0.187:8080/stream?extra_headers=1") == (
        "http://10.0.0.187:8080/snapshot?extra_headers=1"
    )


def test_an_eased_aim_carries_its_speed_and_ramp_to_the_arbiter() -> None:
    client = Client({"status": "done", "pan_rad": 0.5, "tilt_rad": 0.05})
    ease = Ease(200.0, 650.0, "turn")
    door(client).aim(Point(1.0, 0.5, 1.0), speed="saccade", ttl_s=6.0, ease=ease)
    ((_, (request, _)),) = client.calls
    assert (request["speed_deg_s"], request["acc_deg_s2"]) == (200.0, 650.0)
    assert request["speed"] == "saccade", "an arbiter that predates the ease moves at this"


class Bodies:
    def __init__(self, found: list[PersonBox]) -> None:
        self.found = found

    def detect(self, image: Any) -> list[PersonBox]:
        return self.found


def test_a_person_s_box_gives_the_head_near_its_top_its_middle_lying_and_the_edge_cut() -> None:
    """Through a 90 deg pinhole (fx 400 at 800x600): a standing box 120 px wide and 400 tall,
    centred: its head 12 % of the height below the top; a lying one (wider than tall): its
    middle; one whose top the frame cuts: the edge."""
    lens = Lens.nominal(800, 600, 90.0)
    standing = body_sighting(PersonBox(340, 100, 460, 500, 0.7), lens, head_frac=0.12)
    assert standing.pan_deg == pytest.approx(0.0) and standing.head_pan_deg == pytest.approx(0.0)
    assert standing.head_tilt_deg == pytest.approx(math.degrees(math.atan((148 - 300) / 400)))
    assert standing.half_width_deg == pytest.approx(math.degrees(math.atan(60 / 400)), abs=0.3)
    assert not standing.cut
    lying = body_sighting(PersonBox(100, 380, 500, 520, 0.6), lens, head_frac=0.12)
    middle = math.degrees(math.atan2(150 / 400, math.hypot(1.0, 100 / 400)))  # (300, 450) px
    assert lying.head_tilt_deg == pytest.approx(middle)
    cut = body_sighting(PersonBox(300, 0, 500, 600, 0.8), lens, head_frac=0.12)
    assert cut.cut and cut.head_tilt_deg == pytest.approx(-math.degrees(math.atan(300 / 400)))
    picture = Picture(np.zeros((600, 800, 3), dtype=np.uint8), 0.0)
    finder = SnapshotFinder(Camera(picture), lambda: Faces([]), lens,
                            bodies=Bodies([PersonBox(340, 100, 460, 500, 0.7)]))  # fmt: skip
    found = finder.look()
    assert found is not None and found.faces == () and found.bodies == (standing,)


def test_the_vision_body_detector_runs_where_pyobjc_vision_is_installed() -> None:
    """Apple Vision's human rectangles on an empty picture: no person, no error (skipped
    without pyobjc-framework-Vision: uv add --group macos pyobjc-framework-Vision)."""
    pytest.importorskip("Vision")
    detector = VisionBodies()
    detector.warm()
    assert detector.detect(np.zeros((600, 800, 3), dtype=np.uint8)) == []


def test_a_finder_whose_body_detector_cannot_load_runs_faces_only() -> None:
    class Missing:
        def warm(self) -> None:
            raise ImportError("No module named 'Vision'")

        def detect(self, image: Any) -> list[PersonBox]:
            raise AssertionError("never asked")

    picture = Picture(np.zeros((600, 800, 3), dtype=np.uint8), 0.0)
    finder = SnapshotFinder(Camera(picture), lambda: Faces([]), Lens.nominal(800, 600, 90.0),
                            bodies=Missing())  # fmt: skip
    finder.warm()
    found = finder.look()
    assert finder.bodies is None and found is not None and found.bodies == ()
