"""RTAB-Map's two Python adapters as clients of the localisation service: loaded by path as RTAB-Map
loads them, pointed at a real server on the loopback with fake models, and switched live through
the settings file rtabmap_frame writes — service, local and auto, and a service that goes away.

The local models are fakes too (pepin.xfeat_models' loaders are replaced), so the test pins the
switching, the layouts and the counters, and never needs torch."""

from __future__ import annotations

import importlib.util
import json
import threading
from collections.abc import Hashable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import pepin.xfeat_models as xm
from pepin.live_settings import (
    LiveFile,
    RegistrationSettings,
    RegistrationSwitch,
    StatusBoard,
)
from pepin.model_service import ModelServer, Request, Response, decode_arrays, decode_image
from pepin.model_service import encode_arrays as pack

ADAPTERS = Path(__file__).resolve().parents[2] / "ros" / "xfeat"


class ServiceXFeat:
    """/xfeat that answers one keypoint whose x is 1.0 (so a test can tell the service's answer)."""

    name, device, tag, cache_size = "xfeat", "cpu", "xfeat@feedbeef", 0

    def warm(self) -> None:
        pass

    def decode(self, request: Request) -> Any:
        return decode_image(request.headers, request.body), int(request.headers["x-top-k"])

    def cache_key(self, inputs: Any) -> Hashable | None:
        return None

    def infer(self, inputs: Any) -> Any:
        return inputs

    def encode(self, outputs: Any) -> Response:
        _picture, top_k = outputs
        k = np.array([[1.0, float(top_k), 0.5]], dtype=np.float32)
        headers, body = pack({"keypoints": k, "descriptors": np.ones((1, 64), np.float32)})
        return Response(headers.pop("Content-Type"), body, headers)


class ServiceMatch:
    """/match that answers the pair (0, 0) and echoes the confidence floor as its score."""

    name, device, tag, cache_size = "match", "cpu", "lighterglue@feedbeef", 0

    def warm(self) -> None:
        pass

    def decode(self, request: Request) -> Any:
        return decode_arrays(request.headers, request.body), float(request.headers["x-min-conf"])

    def cache_key(self, inputs: Any) -> Hashable | None:
        return None

    def infer(self, inputs: Any) -> Any:
        return inputs

    def encode(self, outputs: Any) -> Response:
        _arrays, min_conf = outputs
        headers, body = pack(
            {"matches": np.array([[0, 0]], np.int32), "scores": np.array([min_conf], np.float32)}
        )
        return Response(headers.pop("Content-Type"), body, headers)


class LocalXFeat:
    """The local XFeat: one keypoint whose x is 2.0."""

    def detectAndCompute(self, image: Any, top_k: int) -> Any:  # noqa: N802 (XFeat's own name)
        import types

        tensor = types.SimpleNamespace
        return [
            {
                "keypoints": tensor(cpu=lambda: tensor(numpy=lambda: np.array([[2.0, 0.0]]))),
                "scores": tensor(cpu=lambda: tensor(numpy=lambda: np.array([0.5]))),
                "descriptors": tensor(cpu=lambda: tensor(numpy=lambda: np.ones((1, 64)))),
            }
        ]


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """A service with fake models, the settings and status files in tmp, the local models faked,
    and a loader for the two adapters (fresh module objects, as RTAB-Map's process has)."""
    server = ModelServer(("127.0.0.1", 0), [ServiceXFeat(), ServiceMatch()], name="test")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("PEPIN_MODELS_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("PEPIN_REGISTRATION_FILE", str(tmp_path / "registration.json"))
    monkeypatch.setenv("PEPIN_REGISTRATION_STATUS", str(tmp_path / "status.json"))
    built = {"xfeat": 0, "glue": 0}

    def load_xfeat(*_a: Any, **_k: Any) -> Any:
        built["xfeat"] += 1
        return LocalXFeat()

    def load_lighterglue(*_a: Any, **_k: Any) -> Any:
        built["glue"] += 1
        return "local glue"

    def local_match(matcher: Any, kq: Any, kt: Any, dq: Any, dt: Any, *rest: Any) -> Any:
        return xm.as_pairs([[1, 1]]), np.zeros(1, np.float32)

    monkeypatch.setattr(xm, "load_xfeat", load_xfeat)
    monkeypatch.setattr(xm, "load_lighterglue", load_lighterglue)
    monkeypatch.setattr(xm, "match", local_match)

    def adapter(name: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, ADAPTERS / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    settings = LiveFile(tmp_path / "registration.json")
    yield {"server": server, "adapter": adapter, "settings": settings, "built": built}
    server.shutdown()
    server.server_close()


def _set(world: dict[str, Any], **values: Any) -> None:
    world["settings"].write(RegistrationSettings(**values).to_json())


def test_auto_asks_the_service_and_never_builds_the_local_model(world: dict[str, Any]) -> None:
    detector = world["adapter"]("rtabmap_xfeat")
    detector.init(0)
    points, descriptors = detector.detect(np.zeros((6, 8), np.uint8))
    assert points.dtype == np.float32 and points.shape == (1, 3) and descriptors.shape == (1, 64)
    assert points[0, 0] == 1.0 and points[0, 1] == 2048.0, "the service's, at the default top_k"
    assert world["built"]["xfeat"] == 0, "torch is never imported while the service answers"


def test_the_switch_is_read_on_every_call_and_moves_live(world: dict[str, Any]) -> None:
    detector = world["adapter"]("rtabmap_xfeat")
    detector.init(0)
    _set(world, backend="local")
    assert detector.detect(np.zeros((6, 8), np.uint8))[0][0, 0] == 2.0
    _set(world, backend="service", top_k=512)
    assert detector.detect(np.zeros((6, 8), np.uint8))[0][0, 1] == 512.0
    assert world["built"]["xfeat"] == 1, "the local model is built once, on its first use"


def test_a_service_that_goes_away_falls_back_in_auto_and_is_nothing_in_service(
    world: dict[str, Any],
) -> None:
    detector = world["adapter"]("rtabmap_xfeat")
    detector.init(0)
    world["server"].shutdown()
    world["server"].server_close()
    _set(world, backend="auto", timeout_s=0.2)
    assert detector.detect(np.zeros((6, 8), np.uint8))[0][0, 0] == 2.0, "the local fallback"
    _set(world, backend="service", timeout_s=0.2)
    points, descriptors = detector.detect(np.zeros((6, 8), np.uint8))
    assert points.shape == (0, 3) and descriptors.shape == (0, 64), "the service or nothing"
    assert detector._client.remotes["xfeat"].down, "and the dead service is left alone"
    assert detector._switch.line().startswith("service 0, local 0, fallback 1, failed 1")


def test_the_matcher_sends_the_confidence_floor_and_answers_int32_pairs(
    world: dict[str, Any],
) -> None:
    glue = world["adapter"]("rtabmap_lighterglue")
    glue.init(64, 0.25, 0, 0, "")
    k, d = np.zeros((3, 2), np.float32), np.ones((3, 64), np.float32)
    found = glue.match(k, k, np.zeros(3), np.zeros(3), d, d, 800, 600)
    assert found.dtype == np.int32 and found.tolist() == [[0, 0]], "the service's pair"
    _set(world, backend="local")
    assert glue.match(k, k, np.zeros(3), np.zeros(3), d, d, 800, 600).tolist() == [[1, 1]]
    none = glue.match(np.zeros((0, 2)), k, np.zeros(0), np.zeros(3), np.zeros((0, 64)), d, 8, 6)
    assert none.shape == (0, 2), "a side with no keypoints asks nobody"
    assert world["built"]["glue"] == 1


def test_the_counters_reach_the_status_file_once_a_minute(world: dict[str, Any]) -> None:
    detector = world["adapter"]("rtabmap_xfeat")
    detector.init(0)
    detector._minute._last -= 61.0
    detector.detect(np.zeros((6, 8), np.uint8))
    block = StatusBoard().read()["xfeat"]
    assert block["backend"] == "auto" and block["service"] == 1 and block["fallback"] == 0
    detector.detect(np.zeros((6, 8), np.uint8))
    assert StatusBoard().read()["xfeat"]["service"] == 1, "not before the next minute"


# ---------------------------------------------------------------- the settings file
def test_a_missing_or_broken_file_is_the_environment_s_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PEPIN_REGISTRATION_BACKEND", "local")
    monkeypatch.setenv("PEPIN_XFEAT_TOP_K", "1024")
    switch = RegistrationSwitch(tmp_path / "none.json")
    assert switch.settings() == RegistrationSettings("local", 1.0, 1024)
    (tmp_path / "bad.json").write_text("{not json")
    assert RegistrationSwitch(tmp_path / "bad.json").settings().backend == "local"
    (tmp_path / "odd.json").write_text(json.dumps({"backend": "gpu", "timeout_s": -1, "top_k": 9}))
    odd = RegistrationSwitch(tmp_path / "odd.json").settings()
    assert odd == RegistrationSettings("local", 1.0, 9), "only the values it can use"


def test_the_file_is_written_whole_and_only_when_it_changes(tmp_path: Path) -> None:
    live = LiveFile(tmp_path / "d" / "registration.json")
    assert live.write_if_changed({"backend": "auto"}) and not live.write_if_changed(
        {"backend": "auto"}
    )
    reader = LiveFile(tmp_path / "d" / "registration.json")
    assert reader.read() == {"backend": "auto"}
    assert live.write_if_changed({"backend": "local"})
    assert reader.read() == {"backend": "local"}, "a rename is seen even within one mtime tick"
    assert [p.name for p in (tmp_path / "d").iterdir()] == ["registration.json"], "no temp left"
