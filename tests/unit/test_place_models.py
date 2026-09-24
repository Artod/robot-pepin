"""The place models' preprocessing and offline loading, and the stamper's worker thread: the
parts that decide whether the service's vectors are the VPR bench's, and whether a snapshot ever
leaves without its descriptor or out of order."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pepin.global_descriptor import PlaceDescriptor
from pepin.place_models import (
    MEAN,
    PLACE_MODELS,
    STD,
    cached_repo,
    place_spec,
    preprocess,
)
from pepin.place_stamp import (
    ERROR,
    FAILED,
    LATE,
    NO_PICTURE,
    NOT_UNIT,
    OFF,
    PlaceStamper,
    rgb_of,
)


def test_the_two_models_and_their_measured_inputs() -> None:
    assert place_spec("boq_dinov2").size == (322, 322) and place_spec("boq_dinov2").dim == 12288
    assert place_spec("boq_r50").size == (384, 384) and place_spec("boq_r50").dim == 16384
    assert set(PLACE_MODELS) == {"boq_dinov2", "boq_r50"}
    with pytest.raises(ValueError, match="boq_dinov2"):
        place_spec("netvlad")


def test_preprocessing_is_pil_bilinear_then_imagenet_normalisation() -> None:
    """A flat grey picture must come out as (0.5 - mean) / std per channel at the model's size."""
    rgb = np.full((600, 800, 3), 127, dtype=np.uint8)
    x = preprocess(rgb, (322, 322))
    assert x.shape == (3, 322, 322) and x.dtype == np.float32
    for c in range(3):
        assert np.allclose(x[c], (127 / 255.0 - MEAN[c]) / STD[c], atol=1e-5)


def test_a_hub_repository_is_found_in_the_cache_or_named_when_missing(tmp_path: Path) -> None:
    (tmp_path / "amaralibey_bag-of-queries_main").mkdir()
    assert cached_repo("amaralibey/bag-of-queries", tmp_path).name.endswith("_main")
    (tmp_path / "owner_repo_dev").mkdir()
    assert cached_repo("owner/repo:dev", tmp_path).name == "owner_repo_dev"
    with pytest.raises(FileNotFoundError, match="facebookresearch/dinov2"):
        cached_repo("facebookresearch/dinov2", tmp_path)


def test_an_image_message_becomes_rgb_whatever_its_encoding() -> None:
    class Image:
        def __init__(self, encoding: str, pixels: np.ndarray, pad: int = 0) -> None:
            h, w = pixels.shape[:2]
            n = 1 if pixels.ndim == 2 else pixels.shape[2]
            rows = np.zeros((h, w * n + pad), dtype=np.uint8)
            rows[:, : w * n] = pixels.reshape(h, w * n)
            self.encoding, self.height, self.width = encoding, h, w
            self.step, self.data = w * n + pad, rows.tobytes()

    bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    bgr[..., 0] = 200  # blue
    rgb = rgb_of(Image("bgr8", bgr, pad=4))  # a row step longer than the pixels
    assert rgb is not None and rgb.shape == (2, 3, 3) and rgb[0, 0].tolist() == [0, 0, 200]
    assert rgb_of(Image("rgb8", bgr))[0, 0].tolist() == [200, 0, 0]  # type: ignore[index]
    grey = rgb_of(Image("mono8", np.full((2, 3), 9, dtype=np.uint8)))
    assert grey is not None and grey.shape == (2, 3, 3) and grey[1, 2].tolist() == [9, 9, 9]
    assert rgb_of(Image("32FC1", np.zeros((2, 3), dtype=np.uint8))) is None


def test_the_worker_publishes_in_packing_order_and_drains_on_close() -> None:
    out: list[tuple[int, str]] = []
    slow = threading.Event()

    def describe(_rgb: Any, _t: float) -> tuple[np.ndarray, str]:
        slow.wait(0.2)
        v = np.zeros(4, dtype=np.float32)
        v[0] = 1.0
        return v, "m@1"

    def attach(msg: dict[str, Any], d: PlaceDescriptor) -> None:
        msg["d"] = d

    stamper = PlaceStamper(4, describe, attach, lambda m: out.append((m["i"], m["d"].tag)))
    picture = np.zeros((2, 2, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    for i in range(6):
        stamper.submit({"i": i}, None if i % 3 == 2 else (lambda: picture), i != 4, 5.0)
    assert time.perf_counter() - t0 < 0.05, "submit never waits for the service"
    slow.set()
    stamper.close(5.0)
    assert [i for i, _ in out] == list(range(6)), "published in the order packed"
    assert [tag for _, tag in out] == ["m@1", "m@1", "null", "m@1", "null", "null"]
    assert stamper.nulls[NO_PICTURE] == 2 and stamper.nulls[OFF] == 1 and stamper.described == 3


def test_a_snapshot_that_waited_its_budget_is_not_asked_for() -> None:
    """Queued at 0 s, handled at 1 s, on a 0.5 s budget: the null descriptor, no call."""
    asked: list[float] = []
    moments = iter([0.0, 1.0])  # submit's stamp, then the worker's look at the clock

    def describe(_rgb: Any, timeout: float) -> tuple[np.ndarray, str]:
        asked.append(timeout)
        return np.ones(3, np.float32), "m"

    stamper = PlaceStamper(
        3,
        describe,
        lambda msg, d: msg.update(d=d),
        lambda msg: None,
        threaded=False,
        clock=lambda: next(moments),
    )
    msg: dict[str, Any] = {}
    stamper.submit(msg, lambda: np.zeros((1, 1, 3), dtype=np.uint8), True, 0.5)
    assert msg["d"].is_null and stamper.nulls[LATE] == 1 and asked == []


def _unit(n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[0] = 1.0
    return v


def test_nothing_one_snapshot_raises_stops_the_worker() -> None:
    """One exception on the worker used to kill it: every later snapshot then waited in the queue
    for ever and RTAB-Map received nothing. Now that snapshot goes out with the null descriptor,
    counted, and the next ones as before — on the threaded path the node runs."""
    out: list[tuple[int, str]] = []
    errors: list[str] = []
    calls = {"n": 0}

    def describe(_rgb: Any, _t: float) -> tuple[np.ndarray, str]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("a picture that does not decode")
        return _unit(4), "m@1"

    def picture(i: int) -> Any:
        def build() -> np.ndarray:
            if i == 3:
                raise ValueError("a buffer shorter than its header says")
            return np.zeros((2, 2, 3), dtype=np.uint8)

        return build

    def publish(msg: dict[str, Any]) -> None:
        if msg["i"] == 4:
            raise RuntimeError("the context is shutting down")
        out.append((msg["i"], msg["d"].tag))

    stamper = PlaceStamper(
        4, describe, lambda m, d: m.update(d=d), publish, threaded=True, on_error=errors.append
    )
    for i in range(7):
        stamper.submit({"i": i}, picture(i), True, 5.0)
    stamper.close(5.0)
    assert stamper._thread is not None and not stamper._thread.is_alive(), "drained and ended"
    assert out == [(0, "m@1"), (1, "null"), (2, "m@1"), (3, "null"), (5, "m@1"), (6, "m@1")]
    assert stamper.nulls[ERROR] == 2 and stamper.unpublished == 1 and len(errors) == 3
    assert "NOT PUBLISHED" in stamper.line() and "shutting down" in stamper.line()


def test_an_answer_that_is_not_a_finite_unit_vector_goes_out_null() -> None:
    """One NaN aborts RTAB-Map (UASSERT_MSG(dotProd >= 0), Signature.cpp:263)."""
    answers = [np.full(4, np.nan, np.float32), np.full(4, 2.0, np.float32), _unit(4)]
    msgs: list[dict[str, Any]] = [{} for _ in answers]
    stamper = PlaceStamper(
        4,
        lambda _rgb, _t: (answers.pop(0), "m@1"),
        lambda m, d: m.update(d=d),
        lambda m: None,
        threaded=False,
    )
    for msg in msgs:
        stamper.submit(msg, lambda: np.zeros((1, 1, 3), dtype=np.uint8), True, 1.0)
    assert [m["d"].is_null for m in msgs] == [True, True, False]
    assert stamper.nulls[NOT_UNIT] == 2 and stamper.described == 1


def test_the_recent_window_counts_only_pictures_the_service_was_asked_about() -> None:
    """rtabmap_frame takes the descriptor likelihood away while too many of the recent camera
    snapshots were not described; a lidar-only snapshot or the service switched off says nothing
    about the service, and the window holds the last ten."""
    down = {"now": True}
    stamper = PlaceStamper(
        4,
        lambda _rgb, _t: None if down["now"] else (_unit(4), "m@1"),
        lambda m, d: None,
        lambda m: None,
        threaded=False,
    )
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    for _ in range(3):
        stamper.submit({}, lambda: rgb, True, 1.0)  # the service down: no answer
    stamper.submit({}, None, True, 1.0)  # lidar-only
    stamper.submit({}, lambda: rgb, False, 1.0)  # place_descriptor off
    assert stamper.recent() == (3, 3) and stamper.nulls[FAILED] == 3
    down["now"] = False
    for _ in range(9):
        stamper.submit({}, lambda: rgb, True, 1.0)
    assert stamper.recent() == (10, 1), "the last ten: one failure left in the window"
