"""The face sensor end to end on fakes: frame sources, the per-frame pipeline, the published line,
the loops, enrolment and the bench."""

from __future__ import annotations

import io
import json
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import IO

import cv2
import numpy as np
import pytest
from face_sensor_fakes import (
    FakeDetector,
    FakeEmbedder,
    FakeExpression,
    FakeLandmarker,
    box,
    direction,
    flat,
    near,
    noise,
)

from pepin.face_sensor.bench import BenchInputs, run_bench
from pepin.face_sensor.config import FaceSensorConfig, ModelsConfig
from pepin.face_sensor.enrol import collect
from pepin.face_sensor.frames import (
    BoardStream,
    FakeFrames,
    Frame,
    MjpegFile,
    Newest,
    Stills,
    cut_eye,
    open_source,
    sample,
)
from pepin.face_sensor.geometry import Lens
from pepin.face_sensor.identity import Gallery, IdentityVote
from pepin.face_sensor.models import Mesh
from pepin.face_sensor.pipeline import FacePipeline, FaceState, best_face, percentiles
from pepin.face_sensor.publish import Publisher, latest
from pepin.face_sensor.runner import Stats, run_live, run_recording

CFG = FaceSensorConfig()


def jpeg(image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".jpg", image)
    assert ok
    return bytes(data)


def multipart(bodies: list[bytes], stamps: list[float]) -> bytes:
    """ustreamer's stream as curl records it: a boundary, headers with stamps, the JPEG."""
    out = b""
    for body, stamp in zip(bodies, stamps, strict=True):
        out += (
            b"--boundarydonotcross\r\nContent-Type: image/jpeg\r\n"
            + f"Content-Length: {len(body)}\r\nX-Timestamp: {stamp:.6f}\r\n\r\n".encode()
            + body
            + b"\r\n"
        )
    return out


def pipeline(**kwargs: object) -> FacePipeline:
    models: dict[str, object] = {
        "embedder": FakeEmbedder([direction(1)]),
        "expression": FakeExpression(),
        "landmarker": FakeLandmarker(),
        "lens": Lens.nominal(800, 600, 90.0),
    }
    models.update(kwargs)
    detector = models.pop("detector", FakeDetector([box(400, 300, 80)]))
    return FacePipeline(CFG, detector, **models)


# ---- frames ----------------------------------------------------------------------------------
def test_cut_eye_keeps_one_upright_eye_of_a_side_by_side_frame() -> None:
    frame = np.zeros((600, 1600, 3), dtype=np.uint8)
    frame[:, :800] = 10
    frame[:, 800:] = 200
    upright_left = cut_eye(frame, "left", upside_down=False)
    assert upright_left.shape == (600, 800, 3) and upright_left[0, 0, 0] == 10
    # Upside down, each half turns 180 degrees and the halves trade places.
    turned_left = cut_eye(frame, "left", upside_down=True)
    assert turned_left[0, 0, 0] == 200
    assert cut_eye(frame, "full", upside_down=True).shape == (600, 1600, 3)
    mono = np.zeros((600, 800, 3), dtype=np.uint8)
    assert cut_eye(mono, "left", upside_down=True) is mono


def test_an_mjpeg_clip_gives_stamped_frames_decoded_on_demand(tmp_path: Path) -> None:
    clip = tmp_path / "0001_cam.mjpeg"
    side = np.zeros((60, 160, 3), dtype=np.uint8)
    side[:, 80:] = 255
    clip.write_bytes(multipart([jpeg(side), jpeg(side)], [100.0, 100.05]))
    frames = list(MjpegFile(clip, eye="left", stamp="send").frames())
    assert [f.stamp for f in frames] == [100.0, 100.05]
    assert frames[1].index == 1
    image = frames[0].image()
    assert image.shape == (60, 80, 3)
    assert image.mean() < 30


def test_sampling_a_recording_keeps_one_frame_a_period() -> None:
    frames = [Frame(i * 0.05, i, lambda: noise(4, 4)) for i in range(40)]  # 2 s at 20 fps
    kept = list(sample(iter(frames), 3.0))
    assert [f.index for f in kept] == [0, 7, 14, 21, 28, 35]


def test_stills_are_offered_one_period_apart(tmp_path: Path) -> None:
    for name in ("b.jpg", "a.png", "notes.txt"):
        path = tmp_path / name
        if name.endswith("txt"):
            path.write_text("x")
        else:
            cv2.imwrite(str(path), noise(20, 30))
    stills = Stills([tmp_path], fps=2.0, repeat=2, start=10.0)
    frames = list(stills.frames())
    assert [f.stamp for f in frames] == [10.0, 10.5, 11.0, 11.5]
    assert [p.name for p in stills.paths] == ["a.png", "b.jpg"]
    assert frames[0].image().shape == (20, 30, 3)
    not_an_image = next(iter(Stills([tmp_path / "notes.txt"]).frames()))
    with pytest.raises(ValueError, match="not an image"):
        not_an_image.image()


def test_open_source_picks_the_reader_by_name(tmp_path: Path) -> None:
    clip = tmp_path / "x.mjpeg"
    clip.write_bytes(b"")
    image = tmp_path / "x.jpg"
    cv2.imwrite(str(image), noise(10, 10))
    kw = {"eye": "left", "upside_down": False, "stamp": "grab", "still_fps": 3.0}
    assert isinstance(open_source(str(clip), **kw), MjpegFile)
    assert isinstance(open_source(str(image), **kw), Stills)
    board = open_source("board", stream_url="http://b:8080/stream", **kw)
    assert isinstance(board, BoardStream) and board.live
    with pytest.raises(ValueError):
        open_source("board", **kw)
    with pytest.raises(FileNotFoundError):
        open_source(str(tmp_path / "nope.mjpeg"), **kw)


def test_the_board_stream_reconnects_and_stamps_from_its_headers() -> None:
    body = jpeg(noise(20, 40))
    opened: list[str] = []

    def opener(url: str, timeout: float) -> IO[bytes]:
        opened.append(url)
        if len(opened) == 1:
            raise OSError("No route to host")
        return io.BytesIO(multipart([body, body], [5.0, 5.1]))

    stream = BoardStream("http://b/stream", stamp="send", opener=opener, retry_s=0.0)
    frames = []
    for frame in stream.frames():
        frames.append(frame)
        if len(frames) == 3:
            stream.close()
    assert [f.stamp for f in frames] == [5.0, 5.1, 5.0]
    assert [f.index for f in frames] == [0, 1, 2]
    assert len(opened) == 3
    assert frames[0].image().shape == (20, 20, 3)  # side by side: the left eye


def test_newest_hands_out_the_latest_frame_once_and_counts_the_skipped() -> None:
    gate = threading.Event()

    class Slow:
        live = True

        def frames(self) -> Iterator[Frame]:
            for i in range(5):
                yield Frame(float(i), i, lambda: noise(2, 2))
            gate.wait(2.0)

        def close(self) -> None:
            gate.set()

    newest = Newest(Slow())
    deadline = time.monotonic() + 2.0
    while newest.skipped < 4 and time.monotonic() < deadline:
        time.sleep(0.001)
    frame = newest.take(timeout=1.0)
    assert frame is not None and frame.index == 4
    assert newest.skipped == 4
    assert newest.take(timeout=0.01) is None
    newest.close()
    assert newest.ended


# ---- the pipeline ----------------------------------------------------------------------------
def test_the_best_face_is_the_widest() -> None:
    near_person, far_person = box(200, 300, 90), box(600, 300, 40)
    assert best_face([far_person, near_person]) is near_person
    assert best_face([]) is None


def test_a_big_face_goes_through_every_stage() -> None:
    p = pipeline(detector=FakeDetector([box(600, 300, 80), box(200, 300, 30)]))
    result = p.process(Frame(1.0, 0, lambda: noise()))
    assert result.faces == 2
    obs = result.best
    assert obs is not None
    assert obs.face_px == pytest.approx(80)
    assert obs.gate.reason == "ok"
    assert obs.embedding is not None and obs.expression is not None and obs.mesh is not None
    assert obs.bearing is not None and obs.bearing[0] < 0  # right of centre: pan negative
    assert set(result.ms) == {"grab", "detect", "mesh", "expr", "embed", "total"}
    assert p.landmarker.shapes[0][:2] == (128, 128)


def test_a_small_face_gets_identity_only() -> None:
    p = pipeline(detector=FakeDetector([box(400, 300, 24)]))
    result = p.process(Frame(1.0, 0, lambda: noise()))
    obs = result.best
    assert obs is not None
    assert obs.embedding is not None
    assert obs.expression is None and obs.mesh is None
    assert "expr" not in result.ms and "mesh" not in result.ms


def test_a_face_turned_away_gets_no_expression() -> None:
    away = FakeLandmarker([Mesh(yaw=45, pitch=0, roll=0, jaw_open=0.1)])
    expr = FakeExpression()
    p = pipeline(landmarker=away, expression=expr)
    obs = p.process(Frame(1.0, 0, lambda: noise())).best
    assert obs is not None and obs.mesh is not None and obs.expression is None
    assert expr.calls == 0


def test_a_face_the_landmarker_cannot_read_gets_no_expression() -> None:
    expr = FakeExpression()
    p = pipeline(landmarker=FakeLandmarker([None]), expression=expr)
    obs = p.process(Frame(1.0, 0, lambda: noise())).best
    assert obs is not None and obs.mesh_read and obs.mesh is None
    assert obs.expression is None and expr.calls == 0
    assert obs.embedding is not None


def test_a_profile_abstains_in_the_published_vote() -> None:
    p = pipeline(
        embedder=FakeEmbedder([direction(5)]),
        landmarker=FakeLandmarker([Mesh(yaw=60, pitch=0, roll=0, jaw_open=0.0)]),
    )
    state = FaceState(CFG, IdentityVote(CFG.identity, Gallery({"artem": direction(1)})))
    for i in range(5):
        state.update(p.process(Frame(i / 3, i, lambda: noise())))
    record = state.record(3.0)
    assert record is not None
    assert record["who"] == {"id": None, "cos": record["who"]["cos"], "votes": "0/5"}
    assert record["attention"]["looking"] is False and record["expr"] is None


def test_a_blurry_face_feeds_no_stage() -> None:
    embedder = FakeEmbedder([direction(1)])
    p = pipeline(embedder=embedder)
    obs = p.process(Frame(1.0, 0, lambda: flat())).best
    assert obs is not None and obs.gate.reason == "blurry"
    assert obs.embedding is None and obs.mesh is None and obs.expression is None
    assert embedder.calls == 0


def test_switched_off_stages_never_run() -> None:
    p = pipeline(embedder=None, expression=None, landmarker=None, lens=None)
    result = p.process(Frame(1.0, 0, lambda: noise()))
    assert set(result.ms) == {"grab", "detect", "total"}
    assert result.best is not None and result.best.bearing is None


def test_no_face_is_a_result_too() -> None:
    result = pipeline(detector=FakeDetector([])).process(Frame(1.0, 0, lambda: noise()))
    assert result.faces == 0 and result.best is None


# ---- the published line ----------------------------------------------------------------------
def state_after(n: int, *, embedder: FakeEmbedder | None = None, dt: float = 1 / 3) -> FaceState:
    artem = direction(1)
    p = pipeline(embedder=embedder or FakeEmbedder([near(artem, 0.8)]))
    state = FaceState(CFG, IdentityVote(CFG.identity, Gallery({"artem": artem})))
    for i in range(n):
        state.update(p.process(Frame(1000.0 + i * dt, i, lambda: noise())))
    return state


def test_the_line_says_who_where_and_how_after_a_few_frames() -> None:
    state = state_after(6)
    record = state.record(3.0)
    assert record is not None
    assert record["stamp"] == pytest.approx(1000.0 + 5 / 3, abs=1e-3)
    assert record["who"] == {"id": "artem", "cos": 0.8, "votes": "5/5"}
    assert record["face_px"] == 80
    assert record["bearing_deg"] == {"pan": 0.0, "tilt": 0.0}
    assert record["expr"] == {"top": "neutral", "p_3s": 0.6, "valence_3s": 0.1, "arousal_3s": -0.2}
    assert record["attention"] == {"looking": True, "talking": False}
    assert record["quality"]["gate"] == "ok"
    assert record["faces"] == 1 and record["fps"] == 3.0
    json.dumps(record)


def test_face_fields_are_null_without_a_face_and_held_for_a_moment() -> None:
    state = state_after(3)
    gone = FacePipeline(CFG, FakeDetector([]))
    state.update(gone.process(Frame(1000.0 + 3 / 3, 3, lambda: noise())))
    held = state.record(3.0)
    assert held is not None and held["face_px"] == 80 and held["faces"] == 0
    state.update(gone.process(Frame(1000.0 + 8 / 3, 4, lambda: noise())))
    empty = state.record(3.0)
    assert empty is not None
    assert empty["faces"] == 0
    assert all(empty[k] is None for k in ("who", "face_px", "bearing_deg", "expr", "attention"))
    assert FaceState(CFG, None).record(3.0) is None


def test_expression_is_null_for_a_face_turned_away() -> None:
    p = pipeline(landmarker=FakeLandmarker([Mesh(0, 0, 0, 0.05)] * 3 + [Mesh(50, 0, 0, 0.05)] * 6))
    state = FaceState(CFG, None)
    for i in range(9):
        state.update(p.process(Frame(i / 3, i, lambda: noise())))
    record = state.record(3.0)
    assert record is not None
    assert record["attention"]["looking"] is False
    assert record["expr"] is None
    assert record["who"] is None


# ---- the loops -------------------------------------------------------------------------------
def test_a_recording_publishes_a_line_per_second_of_its_own_time(tmp_path: Path) -> None:
    pictures = [noise(seed=seed) for seed in range(3)]
    images = [pictures[i % 3] for i in range(60)]
    source = FakeFrames(images, [100.0 + i * 0.05 for i in range(60)])  # 3 s at 20 fps
    state = FaceState(CFG, IdentityVote(CFG.identity, Gallery({"artem": direction(1)})))
    publisher = Publisher(tmp_path)
    lines: list[str] = []
    stats = run_recording(source, pipeline(), state, CFG, publisher=publisher, report=lines.append)
    assert stats.frames == 9  # every 7th of 60
    assert source.decoded == 9  # the skipped frames were never decoded
    assert publisher.lines == 2 and len(lines) == 2
    assert latest(tmp_path / "latest.json", now=103.0) is not None
    summary = stats.summary()
    assert summary["frames_with_face"] == 9
    assert summary["ms"]["detect"]["n"] == 9
    assert summary["rss_peak_mb"] > 0


def test_the_live_loop_takes_the_newest_frame_at_its_own_rate(tmp_path: Path) -> None:
    now = time.time()
    source = FakeFrames([noise(60, 80)] * 4, [now + i * 0.01 for i in range(4)], live=True)
    cfg = replace(CFG, rate_hz=50.0, publish_hz=50.0)
    stop = threading.Event()
    timer = threading.Timer(1.0, stop.set)
    timer.start()
    publisher = Publisher(tmp_path)
    stats = run_live(
        source,
        pipeline(detector=FakeDetector([box(40, 30, 20)])),
        FaceState(cfg, None),
        cfg,
        publisher=publisher,
        stop=stop,
        report=None,
    )
    timer.cancel()
    assert 1 <= stats.frames <= 4
    assert stats.frames + stats.skipped == 4
    assert publisher.lines >= 1


def test_stats_give_percentiles_and_the_rate_by_stamps() -> None:
    assert percentiles([]) == {"p50": 0.0, "p90": 0.0, "mean": 0.0, "max": 0.0, "n": 0}
    p = percentiles([float(v) for v in range(1, 11)])
    assert p["p50"] == pytest.approx(5.5) and p["p90"] == pytest.approx(9.1) and p["n"] == 10
    stats = Stats()
    result = pipeline().process(Frame(10.0, 0, lambda: noise()))
    for i in range(6):
        stats.add(replace(result, stamp=10.0 + i / 3))
    assert stats.fps(10.0 + 5 / 3) == pytest.approx(3.0)
    assert set(stats.recent_ms()) >= {"detect", "total"}
    assert stats.recent_ms() == {}


# ---- enrolment and bench ---------------------------------------------------------------------
def test_enrolment_takes_only_clear_single_faces() -> None:
    cfg = FaceSensorConfig()
    frames = [
        Frame(0.0, 0, lambda: noise()),
        Frame(1.0, 1, lambda: noise()),
        Frame(2.0, 2, lambda: noise()),
        Frame(3.0, 3, lambda: noise()),
        Frame(4.0, 4, lambda: flat()),
    ]
    answers = [
        [box(400, 300, 100)],
        [],
        [box(200, 300, 100), box(600, 300, 90)],
        [box(400, 300, 30)],
        [box(400, 300, 100)],
    ]
    calls = iter(answers)
    detector = FakeDetector(by_image=lambda image: next(calls))
    samples = collect(frames, detector, FakeEmbedder([direction(1)]), cfg)
    assert [s.reason for s in samples] == ["ok", "no face", "2 faces", "small", "blurry"]
    assert [s.embedding is not None for s in samples] == [True, False, False, False, False]


def test_the_bench_reports_stills_and_clips(tmp_path: Path) -> None:
    still = tmp_path / "face.jpg"
    cv2.imwrite(str(still), noise())
    clip = tmp_path / "0001_cam.mjpeg"
    clip.write_bytes(
        multipart([jpeg(noise(seed=i)) for i in range(10)], [i * 0.1 for i in range(10)])
    )
    cfg = replace(CFG, models=ModelsConfig())
    artem = direction(1)
    inputs = BenchInputs(
        cfg, pipeline(embedder=FakeEmbedder([near(artem, 0.8)])), Gallery({"artem": artem})
    )
    lines: list[str] = []
    result = run_bench(
        [still, clip], inputs, repeat=6, crops_dir=tmp_path / "crops", report=lines.append
    )
    face, recorded = result["files"]
    assert face["face_px"] == 80 and face["identity"]["verdict"] == "artem"
    assert face["published"]["who"]["id"] == "artem"
    assert recorded["frames"] == 3 and recorded["frames_with_face"] == 3  # 0, 0.4, 0.8 s
    assert len(list((tmp_path / "crops").glob("*.jpg"))) == 3
    assert result["summary"]["frames"] == 4 + 3  # 6 offers of the still less 2 warm-up
    assert any(line.startswith("p50 / p90") for line in lines)
