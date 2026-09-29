"""The localisation service without its networks: the config it reads, the three endpoints'
layouts through a real server with fake models, the client that refuses an answer of the wrong
layout, and a model that fails to build. The real models run in the slow test at the end, when
the XFeat checkout and the hub's caches are on this machine."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Hashable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pepin.localization_service import (
    DEFAULT_URL,
    MATCH,
    PLACE,
    XFEAT,
    BrokenModel,
    LocalizationClient,
    LocalizationConfig,
    MatchModel,
    PlaceModel,
    XFeatModel,
    load_config,
    models_url,
)
from pepin.model_service import ModelServer, Request, Response, encode_arrays

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- the config
def test_the_shipped_config_names_every_model_on_a_measured_device() -> None:
    config = load_config(REPO / "config/models.json")
    assert config.port == 8791
    assert (config.xfeat.device, config.match.device, config.place.device) == ("cpu", "cpu", "mps")


def test_the_config_ignores_comments_defaults_the_rest_and_refuses_typos() -> None:
    config = LocalizationConfig.from_json({"_note": "x", "xfeat": {"_note": "y", "cache": 8}})
    assert config.xfeat.cache == 8 and config.xfeat.device == "cpu" and config.port == 8791
    with pytest.raises(ValueError, match=r"models\.json"):
        LocalizationConfig.from_json({"xfeat": {"devise": "cpu"}})


def test_the_url_comes_from_the_environment_or_is_the_host_seen_from_a_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PEPIN_MODELS_URL", raising=False)
    assert models_url() == DEFAULT_URL == "http://host.docker.internal:8791"
    monkeypatch.setenv("PEPIN_MODELS_URL", "http://127.0.0.1:18791")
    assert models_url() == "http://127.0.0.1:18791"


# ---------------------------------------------------------------- the endpoints, faked
class Fake:
    """A stand-in for one of the three models: the real decode (the service's own, bound from the
    real class), and an infer that answers in the real layout without a network."""

    cache_size = 4

    def __init__(self, name: str, answer: Any) -> None:
        self.name, self.device, self.tag = name, "cpu", f"{name}@feedbeef"
        self._answer = answer
        self._real = {XFEAT: XFeatModel, MATCH: MatchModel, PLACE: PlaceModel}[name]
        self._top_k, self._min_conf = 2048, 0.1
        self.seen: list[Any] = []

    def warm(self) -> None:
        pass

    def decode(self, request: Request) -> Any:
        return self._real.decode(self, request)  # type: ignore[arg-type]

    def cache_key(self, inputs: Any) -> Hashable | None:
        return self._real.cache_key(self, inputs) if self.name == XFEAT else None  # type: ignore[arg-type]

    def infer(self, inputs: Any) -> Any:
        self.seen.append(inputs)
        return self._answer(inputs)

    def encode(self, outputs: Any) -> Response:
        return self._real.encode(self, outputs)  # type: ignore[arg-type]


def _features(job: Any) -> tuple[np.ndarray, np.ndarray]:
    n = min(job.top_k, 5)
    k = np.zeros((n, 3), dtype=np.float32)
    k[:, 0] = np.arange(n)
    return k, np.ones((n, 64), dtype=np.float32) / 8.0


def _pairs(job: Any) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(job.dq), len(job.dt))
    return np.stack([np.arange(n), np.arange(n)[::-1]], axis=1).astype(np.int32), np.full(
        n, job.min_conf, dtype=np.float32
    )


def _vector(picture: np.ndarray) -> np.ndarray:
    v = np.zeros(12, dtype=np.float32)
    v[int(picture.shape[2] == 3)] = 1.0
    return v


@pytest.fixture
def service() -> Iterator[tuple[LocalizationClient, dict[str, Fake]]]:
    fakes = {XFEAT: Fake(XFEAT, _features), MATCH: Fake(MATCH, _pairs), PLACE: Fake(PLACE, _vector)}
    server = ModelServer(("127.0.0.1", 0), list(fakes.values()), name="localization service")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield LocalizationClient(f"http://127.0.0.1:{server.server_address[1]}", timeout_s=2.0), fakes
    server.shutdown()
    server.server_close()


@pytest.mark.slow
def test_xfeat_takes_a_grey_picture_and_answers_rtabmap_s_layout(service: Any) -> None:
    client, fakes = service
    grey = (np.arange(48 * 64) % 251).astype(np.uint8).reshape(48, 64)
    answer = client.xfeat(grey, top_k=3)
    assert answer is not None
    keypoints, descriptors = answer
    assert keypoints.shape == (3, 3) and descriptors.shape == (3, 64)
    assert keypoints.dtype == descriptors.dtype == np.float32
    assert keypoints.flags.c_contiguous and descriptors.flags.c_contiguous
    job = fakes[XFEAT].seen[0]
    assert np.array_equal(job.picture, grey) and job.top_k == 3  # raw: bit for bit
    assert client.xfeat(grey, top_k=3) is not None
    assert len(fakes[XFEAT].seen) == 1, "the same picture again is answered from the LRU"
    assert client.xfeat(grey, top_k=4) is not None and len(fakes[XFEAT].seen) == 2


@pytest.mark.slow
def test_match_sends_both_sides_and_the_sizes_and_answers_int32_pairs(service: Any) -> None:
    client, fakes = service
    kq, kt = np.zeros((4, 2)), np.zeros((6, 2))
    dq, dt = np.ones((4, 64)), np.ones((6, 64))
    answer = client.match(kq, kt, dq, dt, (800, 600), (640, 480), 0.2)
    assert answer is not None
    pairs, scores = answer
    assert pairs.dtype == np.int32 and pairs.shape == (4, 2) and scores.shape == (4,)
    job = fakes[MATCH].seen[0]
    assert job.size_query == (800, 600) and job.size_train == (640, 480)
    assert job.min_conf == pytest.approx(0.2) and job.kq.dtype == np.float32


@pytest.mark.slow
def test_place_answers_the_vector_and_the_tag_of_the_weights(service: Any) -> None:
    client, _fakes = service
    rgb = np.zeros((30, 40, 3), dtype=np.uint8)
    answer = client.place(rgb)
    assert answer is not None
    vector, tag = answer
    assert tag == "place@feedbeef" and vector.dtype == np.float32 and vector[1] == 1.0


@pytest.mark.slow
def test_an_answer_of_the_wrong_layout_is_none_and_counted(service: Any) -> None:
    client, fakes = service
    fakes[XFEAT]._answer = lambda _job: (
        np.zeros((2, 2), np.float32),
        np.zeros((2, 64), np.float32),
    )
    assert client.xfeat(np.zeros((8, 8), np.uint8), 2) is None
    assert "wrong layout" in client.remotes[XFEAT].last_error
    assert client.remotes[XFEAT].failed == 1 and not client.remotes[XFEAT].down


@pytest.mark.slow
def test_a_request_the_model_cannot_read_is_refused_not_crashed(service: Any) -> None:
    client, _fakes = service
    headers, body = encode_arrays({"kq": np.zeros((2, 2))})
    assert client.remotes[MATCH].call(headers, body) is None
    assert client.remotes[MATCH].last_error.startswith("400")
    health = client.health()
    assert health is not None and health["models"][MATCH]["errors"] == 1


@pytest.mark.slow
def test_a_model_that_failed_to_build_answers_500_and_names_itself() -> None:
    broken = BrokenModel(PLACE, "mps", "FileNotFoundError: no hub cache")
    server = ModelServer(("127.0.0.1", 0), [broken], name="localization service")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = LocalizationClient(f"http://127.0.0.1:{server.server_address[1]}")
        assert client.place(np.zeros((8, 8, 3), np.uint8)) is None
        assert client.remotes[PLACE].last_error.startswith("500")
        health = client.health()
        assert health is not None
        assert health["models"][PLACE]["tag"].startswith("failed: FileNotFoundError")
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------- the real models
XFEAT_CHECKOUT = os.environ.get(
    "PEPIN_XFEAT_DIR", "/Users/artem/robots/pepin/scratch/xfeat/data/accelerated_features"
)


@pytest.mark.slow
@pytest.mark.skipif(
    not Path(XFEAT_CHECKOUT, "weights/xfeat.pt").is_file(), reason="no XFeat checkout here"
)
def test_the_real_models_answer_through_the_service_on_the_cpu() -> None:
    """XFeat and LighterGlue built as the service builds them (on the CPU so the test runs
    anywhere), a textured picture against itself shifted by 12 px: the pairs must be the shift."""
    pytest.importorskip("kornia")
    from pepin.localization_service import MatchConfig, XFeatConfig

    xfeat = XFeatModel(XFeatConfig(device="cpu"), XFEAT_CHECKOUT)
    match = MatchModel(MatchConfig(device="cpu"), XFEAT_CHECKOUT)
    server = ModelServer(("127.0.0.1", 0), [xfeat, match], name="localization service")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = LocalizationClient(f"http://127.0.0.1:{server.server_address[1]}", timeout_s=30)
        rng = np.random.default_rng(0)
        big = (np.kron(rng.random((62, 82)), np.ones((10, 10))) * 255).astype(np.uint8)
        first, second = big[:600, :800], big[:600, 12:812]
        fa, fb = client.xfeat(first, 2048), client.xfeat(second, 2048)
        assert fa is not None and fb is not None
        found = client.match(fb[0][:, :2], fa[0][:, :2], fb[1], fa[1], (800, 600), (800, 600), 0.1)
        assert found is not None
        pairs, _scores = found
        shift = fa[0][pairs[:, 1], 0] - fb[0][pairs[:, 0], 0]
        right = int(np.sum(np.abs(shift - 12.0) < 2.0))
        assert right > 50 and right > 0.6 * len(pairs), (right, len(pairs))
        assert xfeat.tag.startswith("xfeat@") and match.tag.startswith("lighterglue@")
        health = json.loads(json.dumps(client.health()))
        assert health["models"][XFEAT]["requests"] == 2
    finally:
        server.shutdown()
        server.server_close()
