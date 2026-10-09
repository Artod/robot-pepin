"""The face sensor's parts: config, crops and bearing, quality gate, smoothing, identity vote,
publishing."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from face_sensor_fakes import direction, expression, flat, near, noise

from pepin.face_sensor import latest
from pepin.face_sensor.affect import AttentionWindow, ExpressionWindow, facing
from pepin.face_sensor.config import (
    AttentionConfig,
    ExpressionConfig,
    FaceSensorConfig,
    GateConfig,
    IdentityConfig,
)
from pepin.face_sensor.geometry import Lens, crop_box
from pepin.face_sensor.identity import (
    UNKNOWN,
    Gallery,
    IdentityVote,
    Match,
    decide,
    forget_person,
    list_people,
    load_person,
    save_person,
    valid_name,
)
from pepin.face_sensor.models import Mesh
from pepin.face_sensor.publish import Publisher, report_line
from pepin.face_sensor.quality import Quality, gate, measure

REPO = Path(__file__).resolve().parents[2]


# ---- config ----------------------------------------------------------------------------------
def test_the_tracked_config_loads_with_its_measured_defaults() -> None:
    cfg = FaceSensorConfig.load(REPO / "config" / "face_sensor.json")
    assert cfg.rate_hz == 3.0
    assert cfg.gate.min_expression_px == 48
    assert cfg.gate.min_identity_px == 18
    assert cfg.identity.vote_frames == 5
    assert cfg.models.expression_device == "cpu"
    assert cfg.path(cfg.faces_dir) == REPO / "data" / "faces"


def test_config_sections_cast_their_values_and_ignore_notes() -> None:
    cfg = FaceSensorConfig.from_dict(
        {"rate_hz": 5, "gate": {"min_expression_px": 60, "note": "x"}, "paths": {"out": "/tmp/f"}}
    )
    assert cfg.rate_hz == 5.0
    assert cfg.gate.min_expression_px == 60
    assert cfg.path(cfg.out_dir) == Path("/tmp/f")


@pytest.mark.parametrize(
    "data",
    [
        {"rate_hz": 0},
        {"device": "cuda"},
        {"models": {"expression_device": "tpu"}},
        {"source": {"eye": "middle"}},
        {"identity": {"accept_cos": 0.2, "unknown_cos": 0.3}},
        {"identity": {"min_votes": 6, "vote_frames": 5}},
        {"gate": {"min_identity_px": 60, "min_expression_px": 48}},
    ],
)
def test_config_refuses_values_no_run_could_use(data: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FaceSensorConfig.from_dict(data)


# ---- crops and bearing -----------------------------------------------------------------------
def test_crop_box_grows_squares_and_clips_to_the_image() -> None:
    image = noise(100, 200)
    assert crop_box(image, (10, 20, 30, 60)).shape[:2] == (40, 20)
    assert crop_box(image, (10, 20, 30, 60), margin=0.5, square=True).shape[:2] == (60, 50)
    edge = crop_box(image, (190, 90, 260, 140))
    assert edge.shape[:2] == (10, 10)
    assert crop_box(image, (500, 500, 600, 600)).size > 0


def test_bearing_is_zero_on_the_axis_left_and_down_positive() -> None:
    lens = Lens.nominal(800, 600, 90.0)
    assert lens.bearing_deg(400, 300) == pytest.approx((0.0, 0.0))
    pan, tilt = lens.bearing_deg(0, 300)
    assert pan == pytest.approx(45.0)
    assert tilt == pytest.approx(0.0)
    pan, tilt = lens.bearing_deg(400, 600)
    assert pan == pytest.approx(0.0)
    assert tilt == pytest.approx(math.degrees(math.atan(300 / 400)))


def test_bearing_undistorts_the_raw_stereo_eye() -> None:
    lens = Lens.stereo_eye(REPO / "config" / "stereo_calibration.json", "left")
    pinhole = replace(lens, dist=())
    pan_raw, _ = pinhole.bearing_deg(50, lens.cy)
    pan, tilt = lens.bearing_deg(50, lens.cy)
    # Barrel distortion pulls the edge inwards: the true ray is further out than the pinhole says.
    assert pan > pan_raw + 3.0
    assert abs(tilt) < 1.0
    assert lens.bearing_deg(lens.cx, lens.cy) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_a_lens_scales_to_the_frame_it_is_used_on() -> None:
    lens = Lens.nominal(800, 600, 90.0)
    half = lens.scaled(400, 300)
    assert half.fx == pytest.approx(lens.fx / 2)
    assert half.bearing_deg(0, 150) == pytest.approx(lens.bearing_deg(0, 300))
    assert lens.scaled(800, 600) is lens


# ---- quality gate ----------------------------------------------------------------------------
def test_quality_reads_texture_as_sharp_and_flat_as_blurry() -> None:
    sharp = measure(noise(120, 120))
    blank = measure(flat(120, 120))
    assert sharp.sharpness > 100
    assert blank.sharpness == pytest.approx(0.0)
    assert blank.brightness == pytest.approx(128.0)


def test_gate_keys_each_stage_on_the_face_width() -> None:
    cfg = GateConfig()
    good = Quality(sharpness=200.0, brightness=120.0)
    assert gate(60, good, cfg) == gate(60, good, cfg)
    big = gate(60, good, cfg)
    assert (big.identity, big.expression, big.attention, big.reason) == (True, True, True, "ok")
    mid = gate(40, good, cfg)
    assert (mid.identity, mid.expression, mid.attention, mid.reason) == (True, False, True, "small")
    tiny = gate(20, good, cfg)
    assert (tiny.identity, tiny.expression, tiny.attention) == (True, False, False)
    assert not gate(10, good, cfg).identity


@pytest.mark.parametrize(
    ("quality", "reason"),
    [
        (Quality(200.0, 10.0), "dark"),
        (Quality(200.0, 250.0), "bright"),
        (Quality(1.0, 120.0), "blurry"),
    ],
)
def test_gate_refuses_every_stage_on_a_bad_crop(quality: Quality, reason: str) -> None:
    g = gate(100, quality, GateConfig())
    assert (g.identity, g.expression, g.attention, g.reason) == (False, False, False, reason)


# ---- expression and attention windows --------------------------------------------------------
def test_expression_is_the_mean_of_the_window() -> None:
    window = ExpressionWindow(ExpressionConfig(window_s=3.0, min_samples=2))
    window.add(0.0, expression("happiness", 0.9, valence=0.8, arousal=0.4))
    assert window.mean(0.0) is None
    window.add(1.0, expression("neutral", 0.6, valence=0.0, arousal=0.0))
    window.add(2.0, expression("neutral", 0.6, valence=0.1, arousal=-0.4))
    smooth = window.mean(2.0)
    assert smooth is not None
    assert smooth.top == "neutral"
    assert smooth.valence == pytest.approx(0.3)
    assert smooth.samples == 3
    later = window.mean(4.5)  # the first two have left the window
    assert later is None
    assert smooth.to_json() == {"top": "neutral", "p_3s": 0.4, "valence_3s": 0.3, "arousal_3s": 0.0}


def test_looking_is_the_head_inside_the_cone_by_majority() -> None:
    cfg = AttentionConfig(max_yaw_deg=20, max_pitch_deg=20, look_window_s=1.0)
    assert facing(Mesh(10, -15, 0, 0), cfg)
    assert not facing(Mesh(35, 0, 0, 0), cfg)
    window = AttentionWindow(cfg)
    assert window.state(0.0) is None
    window.add(0.0, Mesh(40, 0, 0, 0.1))
    window.add(0.4, Mesh(5, 0, 0, 0.1))
    window.add(0.8, Mesh(3, 2, 0, 0.1))
    state = window.state(0.8)
    assert state is not None
    assert state.looking is True
    assert state.talking is None  # three jaw readings, four needed


def test_talking_is_the_jaw_moving() -> None:
    cfg = AttentionConfig(talk_window_s=2.0, talk_jaw_std=0.04, min_talk_samples=4)
    still, talking = AttentionWindow(cfg), AttentionWindow(cfg)
    for i, jaw in enumerate((0.05, 0.06, 0.05, 0.06, 0.05, 0.06)):
        still.add(i / 3, Mesh(0, 0, 0, jaw))
    for i, jaw in enumerate((0.02, 0.25, 0.05, 0.3, 0.1, 0.2)):
        talking.add(i / 3, Mesh(0, 0, 0, jaw))
    s, t = still.state(5 / 3), talking.state(5 / 3)
    assert s is not None and t is not None
    assert s.talking is False
    assert t.talking is True
    assert t.to_json() == {"looking": True, "talking": True}


# ---- identity --------------------------------------------------------------------------------
def test_a_frame_votes_by_threshold_and_margin() -> None:
    cfg = IdentityConfig(accept_cos=0.45, unknown_cos=0.30, margin=0.08)
    assert decide(Match("artem", 0.6, 0.1), cfg) == "artem"
    assert decide(Match("artem", 0.6, 0.55), cfg) is None  # too close to the runner-up
    assert decide(Match("artem", 0.4, 0.1), cfg) is None  # between the two thresholds
    assert decide(Match("artem", 0.2, 0.1), cfg) == UNKNOWN
    assert decide(Match(None, -1.0, -1.0), cfg) == UNKNOWN


def test_the_vote_needs_min_votes_of_the_window() -> None:
    artem = direction(1)
    vote = IdentityVote(IdentityConfig(min_votes=3, vote_frames=5), Gallery({"artem": artem}))
    vote.add(0.0, near(artem, 0.7), 60)
    vote.add(0.3, near(artem, 0.7, seed=2), 60)
    who = vote.current(0.3)
    assert who is not None and who.id is None and who.votes == 2
    vote.add(0.6, near(artem, 0.7, seed=3), 60)
    who = vote.current(0.6)
    assert who is not None
    assert who.id == "artem"
    assert who.cos == pytest.approx(0.7, abs=1e-3)
    assert who.to_json() == {"id": "artem", "cos": 0.7, "votes": "3/3"}
    for i in range(4):
        vote.add(1.0 + i / 3, near(artem, 0.7, seed=10 + i), 60)
    assert vote.current(2.0).to_json()["votes"] == "5/5"
    assert vote.current(10.0) is None  # forgotten after forget_after_s without a face


def test_unknown_faces_become_one_session_guest_each() -> None:
    artem = direction(1)
    cfg = IdentityConfig(min_votes=3, vote_frames=5, guest_cos=0.45, enrol_min_px=40)
    vote = IdentityVote(cfg, Gallery({"artem": artem}))
    misha, olga = direction(7), direction(8)
    for i in range(3):
        vote.add(i / 3, near(misha, 0.8, seed=20 + i), 60)
    who = vote.current(2 / 3)
    assert who is not None and who.id == "guest-1"
    assert "guest-1" in vote.guests.centroids
    later = vote.add(5.0, near(misha, 0.8, seed=30), 60)  # a new window: matched straight away
    assert later.label == "guest-1"
    for i in range(3):
        vote.add(10.0 + i / 3, near(olga, 0.8, seed=40 + i), 60)
    assert vote.current(10.7).id == "guest-2"


def test_small_unknown_faces_never_become_guests() -> None:
    vote = IdentityVote(IdentityConfig(enrol_min_px=40), Gallery({"artem": direction(1)}))
    stranger = direction(5)
    for i in range(5):
        vote.add(i / 3, near(stranger, 0.8, seed=i), 25)
    who = vote.current(4 / 3)
    assert who is not None and who.id == UNKNOWN
    assert not vote.guests.centroids


def test_enrolment_files_round_trip_and_merge(tmp_path: Path) -> None:
    a, b = near(direction(1), 0.9, seed=1), near(direction(1), 0.9, seed=2)
    person = save_person(tmp_path, "artem", [a], "buffalo_l")
    assert person.count == 1
    merged = save_person(tmp_path, "artem", [b], "buffalo_l")
    assert merged.count == 2
    loaded = load_person(tmp_path / "artem.npz")
    assert loaded.count == 2 and loaded.model == "buffalo_l"
    assert np.linalg.norm(loaded.centroid) == pytest.approx(1.0)
    low, mean = loaded.spread()
    assert low <= mean <= 1.0
    with pytest.raises(ValueError, match="buffalo_l"):
        save_person(tmp_path, "artem", [a], "buffalo_s")
    assert save_person(tmp_path, "artem", [a], "buffalo_s", replace=True).count == 1
    save_person(tmp_path, "olga", [b], "buffalo_l")
    assert [p.name for p in list_people(tmp_path)] == ["artem", "olga"]
    gallery = Gallery.load(tmp_path, "buffalo_l")
    assert list(gallery.centroids) == ["olga"]  # artem is buffalo_s now
    assert forget_person(tmp_path, "olga")
    assert not forget_person(tmp_path, "olga")


@pytest.mark.parametrize("name", ["Artem", "guest-1", "unknown", "", "1x", "a b"])
def test_names_that_cannot_be_enrolled(name: str, tmp_path: Path) -> None:
    assert not valid_name(name)
    with pytest.raises(ValueError):
        save_person(tmp_path, name, [direction(1)], "p")


def test_enrolment_needs_a_sample(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no usable face"):
        save_person(tmp_path, "artem", [], "p")


# ---- publishing ------------------------------------------------------------------------------
def test_latest_reads_a_fresh_line_and_refuses_a_stale_one(tmp_path: Path) -> None:
    publisher = Publisher(tmp_path)
    record = {"stamp": 1000.0, "faces": 0, "fps": 3.0}
    publisher.publish(record, {"total": 12.0})
    assert latest(tmp_path / "latest.json", now=1002.0) == record
    assert latest(tmp_path / "latest.json", now=1003.5) is None
    assert latest(tmp_path / "missing.json", now=1000.0) is None
    (tmp_path / "latest.json").write_text("{half")
    assert latest(tmp_path / "latest.json", now=1000.0) is None
    (tmp_path / "latest.json").write_text('{"faces": 1}')
    assert latest(tmp_path / "latest.json", now=1000.0) is None


def test_the_day_log_keeps_every_line_with_its_stage_times(tmp_path: Path) -> None:
    publisher = Publisher(tmp_path)
    publisher.publish({"stamp": 1000.0, "faces": 0}, {"detect": 6.24})
    publisher.publish({"stamp": 1001.0, "faces": 1}, None)
    logs = list(tmp_path.glob("*.jsonl"))
    assert len(logs) == 1
    lines = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert lines[0]["ms"] == {"detect": 6.2}
    assert "ms" not in lines[1]
    assert json.loads((tmp_path / "latest.json").read_text()) == {"stamp": 1001.0, "faces": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_report_line_says_who_where_and_how() -> None:
    record = {
        "stamp": 1000.0,
        "faces": 1,
        "who": {"id": "artem", "cos": 0.84, "votes": "5/5"},
        "face_px": 62,
        "bearing_deg": {"pan": 4.1, "tilt": -2.0},
        "expr": {"top": "neutral", "p_3s": 0.66, "valence_3s": 0.12, "arousal_3s": -0.25},
        "attention": {"looking": True, "talking": None},
        "quality": {"sharpness": 120, "brightness": 90, "gate": "ok", "det": 0.88},
        "fps": 3.0,
    }
    line = report_line(record, {"detect": 6.4, "total": 20.1})
    assert "artem cos 0.84 5/5" in line
    assert "62 px pan +4.1 tilt -2.0" in line
    assert "neutral 0.66" in line
    assert "look yes talk -" in line
    assert "detect 6 total 20" in line
    empty = report_line({"stamp": 1000.0, "faces": 0, "fps": 2.7, "face_px": None}, {})
    assert "no face" in empty
