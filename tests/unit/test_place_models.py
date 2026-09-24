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
from pepin.place_stamp import LATE, NO_PICTURE, OFF, PlaceStamper, rgb_of


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
