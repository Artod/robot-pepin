"""The localisation models on the laptop's own GPU, as one host process the containers call.

THREE ENDPOINTS, one process, one failure domain (the depth network is the other,
:mod:`pepin.depth_service`, so either may crash or hang without the other):

* ``POST /xfeat`` — a grey picture (raw 8-bit: lossless, what a keypoint detector must see) in,
  XFeat's keypoints (N x 3 float32: x, y, score) and descriptors (N x 64 float32) out, the best
  ``X-Top-K`` of them. RTAB-Map re-extracts the SAME stored node pictures on every registration
  (``RGBD/LoopClosureReextractFeatures``), so answers are kept in a small LRU keyed by the
  picture's bytes and top_k, and a node seen before costs no inference.
* ``POST /match`` — two keypoint and descriptor sets and their pictures' sizes in, LighterGlue's
  pairs (M x 2 int32: query index, train index) and their confidences (M float32) out, at the
  confidence floor ``X-Min-Conf`` (0.1).
* ``POST /place`` — a colour picture (JPEG is fine) in, the place model's L2-normalised float32
  vector out, its length in ``X-Dim`` and the weights that said it in ``X-Model``.

WHERE EACH RUNS is config/models.json (``device`` per model), because it is a measurement, not a
belief (scratch/models/endpoint_bench.py, through the service, 800x600, 2026-09-24): XFeat 46 ms
median, 54 ms p90 on the CPU against 41-81 ms median and up to 241 ms p90 on MPS, which the depth
host's RAFT-Stereo keeps 80 % busy — so the CPU; LighterGlue 109-129 ms on the CPU against 518 ms
on MPS; BoQ-DINOv2 57 ms on MPS against 167 ms on the CPU. Models on one device take turns under
that device's lock (:class:`pepin.model_service.ModelServer`).

A MODEL THAT FAILS TO BUILD does not take the process down: its endpoint answers 500 with the
reason and ``/health`` names it (the tag reads ``failed: ...``), while the other two serve. A
launchd job (ros/models.sh) keeps the process itself alive.

THE CLIENT (:class:`LocalizationClient`) is what the containers import: stdlib and numpy only,
``None`` for any failure, and a back-off while the service is down
(:class:`pepin.model_service.RemoteModel`). Its URL is ``PEPIN_MODELS_URL``, else
:data:`DEFAULT_URL` (the host as a container sees it).

Run: ``uv run --group localization python -m pepin.localization_service`` (ros/models.sh does).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.global_descriptor import unit_or_none
from pepin.model_service import (
    BadRequestError,
    ModelServer,
    RemoteModel,
    Request,
    Response,
    decode_arrays,
    decode_image,
    encode_arrays,
    encode_image,
    lower_keys,
    port_taken,
)

log = logging.getLogger("pepin.localization_service")

DEFAULT_PORT = 8791
DEFAULT_URL = f"http://host.docker.internal:{DEFAULT_PORT}"  # the host, seen from a container
URL_ENV = "PEPIN_MODELS_URL"
CONFIG_NAME = "models.json"
XFEAT, MATCH, PLACE = "xfeat", "match", "place"

Float = npt.NDArray[np.float32]
Pairs = npt.NDArray[np.int32]


def models_url() -> str:
    """The service's URL: ``PEPIN_MODELS_URL``, else :data:`DEFAULT_URL`."""
    return os.environ.get(URL_ENV) or DEFAULT_URL


# ---------------------------------------------------------------- the configuration
@dataclass(frozen=True)
class XFeatConfig:
    """XFeat's device, its default keypoint cap and how many answers the LRU keeps."""

    device: str = "cpu"
    top_k: int = 2048
    cache: int = 64


@dataclass(frozen=True)
class MatchConfig:
    """LighterGlue's device and its default confidence floor."""

    device: str = "cpu"
    min_conf: float = 0.1


@dataclass(frozen=True)
class PlaceConfig:
    """The place model's name (pepin.place_models.PLACE_MODELS) and device."""

    model: str = "boq_dinov2"
    device: str = "mps"


@dataclass(frozen=True)
class LocalizationConfig:
    """config/models.json's ``localization`` block: the port, torch's CPU threads, the models."""

    port: int = DEFAULT_PORT
    cpu_threads: int = 4
    xfeat: XFeatConfig = field(default_factory=XFeatConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    place: PlaceConfig = field(default_factory=PlaceConfig)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> LocalizationConfig:
        """The block as parsed JSON; keys starting with ``_`` are comments, unknown keys refused."""

        def fields(block: Mapping[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in block.items() if not k.startswith("_")}

        top = fields(data)
        try:
            return cls(
                port=int(top.get("port", DEFAULT_PORT)),
                cpu_threads=int(top.get("cpu_threads", 4)),
                xfeat=XFeatConfig(**fields(top.get(XFEAT, {}))),
                match=MatchConfig(**fields(top.get(MATCH, {}))),
                place=PlaceConfig(**fields(top.get(PLACE, {}))),
            )
        except TypeError as exc:
            raise ValueError(f"config/{CONFIG_NAME}: {exc}") from exc


def load_config(path: str | Path | None = None) -> LocalizationConfig:
    """config/models.json's ``localization`` block (pepin.deployment.config_file finds the file
    on the laptop and in a container); the defaults when the file has no such block."""
    from pepin.deployment import config_file

    raw = json.loads(Path(path or config_file(CONFIG_NAME)).read_text())
    return LocalizationConfig.from_json(raw.get("localization", {}))


def place_dim(config: LocalizationConfig | None = None) -> int:
    """The length of the configured place model's vector: every node's descriptor has it."""
    from pepin.place_models import place_spec

    return place_spec((config or load_config()).place.model).dim


# ---------------------------------------------------------------- the models
def _size(headers: Mapping[str, str], name: str) -> tuple[int, int]:
    """A ``(width, height)`` header written ``800,600``."""
    try:
        width, height = (int(v) for v in headers[name].split(","))
    except (KeyError, ValueError) as exc:
        raise BadRequestError(f"{name} must be 'width,height'") from exc
    return width, height


def _grey(picture: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    if picture.ndim == 2:
        return picture
    import cv2

    return np.asarray(cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY), dtype=np.uint8)


@dataclass(frozen=True)
class XFeatJob:
    """One /xfeat request: the grey picture, its bytes' digest and the keypoint cap."""

    picture: npt.NDArray[np.uint8]
    digest: bytes
    top_k: int


class XFeatModel:
    """``POST /xfeat``: XFeat's keypoints and descriptors of a grey picture."""

    name = XFEAT

    def __init__(self, config: XFeatConfig, xfeat_dir: str) -> None:
        from pepin.xfeat_models import XFEAT_WEIGHTS, load_xfeat, weights_tag

        self.device = config.device
        self.cache_size = config.cache
        self._top_k = config.top_k
        self.tag = weights_tag(XFEAT, os.path.join(xfeat_dir, XFEAT_WEIGHTS))
        self._model = load_xfeat(xfeat_dir, top_k=config.top_k, device=config.device)

    def warm(self) -> None:
        rng = np.random.default_rng(0)
        picture = (np.kron(rng.random((60, 80)), np.ones((10, 10))) * 255).astype(np.uint8)
        self.infer(XFeatJob(picture, b"", self._top_k))

    def decode(self, request: Request) -> XFeatJob:
        picture = _grey(decode_image(request.headers, request.body))
        top_k = int(request.headers.get("x-top-k", self._top_k))
        if not 1 <= top_k <= 16384:
            raise BadRequestError(f"X-Top-K {top_k} is not 1..16384")
        return XFeatJob(picture, hashlib.blake2b(request.body, digest_size=16).digest(), top_k)

    def cache_key(self, inputs: XFeatJob) -> Hashable | None:
        return (inputs.digest, inputs.picture.shape, inputs.top_k)

    def infer(self, inputs: XFeatJob) -> tuple[Float, Float]:
        from pepin.xfeat_models import features

        return features(self._model, inputs.picture, inputs.top_k)

    def encode(self, outputs: tuple[Float, Float]) -> Response:
        headers, body = encode_arrays({"keypoints": outputs[0], "descriptors": outputs[1]})
        return Response(headers.pop("Content-Type"), body, headers)


@dataclass(frozen=True)
class MatchJob:
    """One /match request: both sides' keypoints and descriptors, sizes, the floor."""

    kq: Float
    kt: Float
    dq: Float
    dt: Float
    size_query: tuple[int, int]
    size_train: tuple[int, int]
    min_conf: float


class MatchModel:
    """``POST /match``: LighterGlue's pairs between two XFeat feature sets."""

    name = MATCH
    cache_size = 0

    def __init__(self, config: MatchConfig, xfeat_dir: str) -> None:
        from pepin.xfeat_models import LIGHTERGLUE_WEIGHTS, load_lighterglue, weights_tag

        self.device = config.device
        self._min_conf = config.min_conf
        self.tag = weights_tag("lighterglue", os.path.join(xfeat_dir, LIGHTERGLUE_WEIGHTS))
        self._model = load_lighterglue(xfeat_dir, device=config.device)

    def warm(self) -> None:
        rng = np.random.default_rng(0)
        k = (rng.random((200, 2)) * 400).astype(np.float32)
        d = rng.standard_normal((200, 64)).astype(np.float32)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.infer(MatchJob(k, k, d, d, (400, 400), (400, 400), self._min_conf))

    def decode(self, request: Request) -> MatchJob:
        arrays = decode_arrays(request.headers, request.body)
        try:
            kq, kt = arrays["kq"].astype(np.float32), arrays["kt"].astype(np.float32)
            dq, dt = arrays["dq"].astype(np.float32), arrays["dt"].astype(np.float32)
        except KeyError as exc:
            raise BadRequestError(f"/match needs kq, kt, dq and dt, not {sorted(arrays)}") from exc
        if kq.reshape(-1, 2).shape[0] != len(dq) or kt.reshape(-1, 2).shape[0] != len(dt):
            raise BadRequestError("each side needs one descriptor per keypoint")
        min_conf = float(request.headers.get("x-min-conf", self._min_conf))
        return MatchJob(
            kq,
            kt,
            dq,
            dt,
            _size(request.headers, "x-size-query"),
            _size(request.headers, "x-size-train"),
            min_conf,
        )

    def cache_key(self, inputs: MatchJob) -> Hashable | None:
        return None

    def infer(self, inputs: MatchJob) -> tuple[Pairs, Float]:
        from pepin.xfeat_models import match

        return match(
            self._model,
            inputs.kq,
            inputs.kt,
            inputs.dq,
            inputs.dt,
            inputs.size_query,
            inputs.size_train,
            inputs.min_conf,
        )

    def encode(self, outputs: tuple[Pairs, Float]) -> Response:
        headers, body = encode_arrays({"matches": outputs[0], "scores": outputs[1]})
        return Response(headers.pop("Content-Type"), body, headers)


class PlaceModel:
    """``POST /place``: the place model's L2-normalised vector of a colour picture."""

    name = PLACE
    cache_size = 0

    def __init__(self, config: PlaceConfig) -> None:
        from pepin.place_models import load_place_model, place_spec, place_tag

        self.device = config.device
        self._spec = place_spec(config.model)
        self.dim = self._spec.dim
        self.tag = place_tag(config.model)
        self._model = load_place_model(config.model, config.device)

    def warm(self) -> None:
        self.infer(np.full((120, 160, 3), 128, dtype=np.uint8))

    def decode(self, request: Request) -> npt.NDArray[np.uint8]:
        picture = decode_image(request.headers, request.body)
        if picture.ndim != 3:
            picture = np.repeat(picture[:, :, None], 3, axis=2)
        return picture

    def cache_key(self, inputs: npt.NDArray[np.uint8]) -> Hashable | None:
        return None

    def infer(self, inputs: npt.NDArray[np.uint8]) -> Float:
        from pepin.place_models import describe

        vector = unit_or_none(describe(self._model, inputs, self._spec.size, self.device))
        if vector is None:  # a NaN would abort RTAB-Map at its next comparison: answer 500
            raise ValueError(f"{self.tag} gave no finite unit vector for this picture")
        return vector

    def encode(self, outputs: Float) -> Response:
        headers, body = encode_arrays({"vector": outputs})
        return Response(headers.pop("Content-Type"), body, {**headers, "X-Dim": str(outputs.size)})


class BrokenModel:
    """A model that failed to build: its endpoint answers 500 with the reason, ``/health``
    names it, and the process's other models keep serving."""

    cache_size = 0

    def __init__(self, name: str, device: str, why: str) -> None:
        self.name = name
        self.device = device
        self.tag = f"failed: {why}"[:200]

    def warm(self) -> None:
        pass

    def decode(self, request: Request) -> None:
        return None

    def cache_key(self, inputs: None) -> Hashable | None:
        return None

    def infer(self, inputs: None) -> None:
        raise RuntimeError(f"{self.name} is not serving: {self.tag}")

    def encode(self, outputs: None) -> Response:
        raise RuntimeError(f"{self.name} is not serving")


def build_models(config: LocalizationConfig, xfeat_dir: str, warm: bool = True) -> list[Any]:
    """The three models on their configured devices, each warmed; a model that fails to build is
    a :class:`BrokenModel` and the others are built anyway."""
    import torch

    if config.cpu_threads > 0:
        torch.set_num_threads(config.cpu_threads)
    builders: list[tuple[str, str, Any]] = [
        (XFEAT, config.xfeat.device, lambda: XFeatModel(config.xfeat, xfeat_dir)),
        (MATCH, config.match.device, lambda: MatchModel(config.match, xfeat_dir)),
        (PLACE, config.place.device, lambda: PlaceModel(config.place)),
    ]
    models: list[Any] = []
    for name, device, build in builders:
        t0 = time.perf_counter()
        try:
            model = build()
            if warm:
                model.warm()
        except Exception as exc:
            why = f"{type(exc).__name__}: {' '.join(str(exc).split())}"
            log.exception("%s failed to build on %s", name, device)
            models.append(BrokenModel(name, device, why))
            continue
        log.info("%s (%s) on %s in %.1f s", name, model.tag, device, time.perf_counter() - t0)
        models.append(model)
    return models


# ---------------------------------------------------------------- the client
class LocalizationClient:
    """The three endpoints as one client that never raises: every method answers ``None`` when
    the service does not (no service, a timeout, a refusal, an answer of the wrong layout).
    One :class:`pepin.model_service.RemoteModel` per endpoint, so one endpoint's back-off does
    not silence the others."""

    def __init__(
        self, url: str | None = None, timeout_s: float = 1.0, retry_s: float = 10.0
    ) -> None:
        self.url = url or models_url()
        self.remotes = {
            name: RemoteModel(self.url, name, timeout_s=timeout_s, retry_s=retry_s)
            for name in (XFEAT, MATCH, PLACE)
        }

    def xfeat(
        self, grey: npt.ArrayLike, top_k: int, timeout_s: float | None = None
    ) -> tuple[Float, Float] | None:
        """(N x 3 keypoints, N x 64 descriptors) of a grey picture, or ``None``."""
        headers, body = encode_image(np.asarray(grey, dtype=np.uint8), "raw")
        answer = self.remotes[XFEAT].call({**headers, "X-Top-K": str(int(top_k))}, body, timeout_s)
        arrays = self._arrays(XFEAT, answer)
        if arrays is None:
            return None
        k, d = arrays.get("keypoints"), arrays.get("descriptors")
        if k is None or d is None or k.ndim != 2 or k.shape[1] != 3 or d.shape != (len(k), 64):
            self._malformed(XFEAT, arrays)
            return None
        return (
            np.ascontiguousarray(k, dtype=np.float32),
            np.ascontiguousarray(d, dtype=np.float32),
        )

    def match(
        self,
        kq: npt.ArrayLike,
        kt: npt.ArrayLike,
        dq: npt.ArrayLike,
        dt: npt.ArrayLike,
        size_query: tuple[int, int],
        size_train: tuple[int, int],
        min_conf: float,
        timeout_s: float | None = None,
    ) -> tuple[Pairs, Float] | None:
        """(M x 2 int32 pairs, M confidences) between two feature sets, or ``None``."""
        headers, body = encode_arrays(
            {
                "kq": np.asarray(kq, dtype=np.float32),
                "kt": np.asarray(kt, dtype=np.float32),
                "dq": np.asarray(dq, dtype=np.float32),
                "dt": np.asarray(dt, dtype=np.float32),
            }
        )
        headers.update(
            {
                "X-Size-Query": f"{int(size_query[0])},{int(size_query[1])}",
                "X-Size-Train": f"{int(size_train[0])},{int(size_train[1])}",
                "X-Min-Conf": f"{float(min_conf):g}",
            }
        )
        arrays = self._arrays(MATCH, self.remotes[MATCH].call(headers, body, timeout_s))
        if arrays is None:
            return None
        pairs, scores = arrays.get("matches"), arrays.get("scores")
        if pairs is None or scores is None or pairs.reshape(-1, 2).shape[0] != scores.size:
            self._malformed(MATCH, arrays)
            return None
        return (
            np.ascontiguousarray(pairs.reshape(-1, 2), dtype=np.int32),
            np.ascontiguousarray(scores.reshape(-1), dtype=np.float32),
        )

    def place(
        self, rgb: npt.ArrayLike, encoding: str = "jpeg", timeout_s: float | None = None
    ) -> tuple[Float, str] | None:
        """(the unit-length place vector, the tag of the weights that said it), or ``None`` —
        also for an answer that is not a finite unit vector."""
        headers, body = encode_image(np.asarray(rgb, dtype=np.uint8), encoding, quality=90)
        remote = self.remotes[PLACE]
        arrays = self._arrays(PLACE, remote.call(headers, body, timeout_s))
        if arrays is None:
            return None
        vector = arrays.get("vector")
        unit = None if vector is None or vector.ndim != 1 else unit_or_none(vector)
        if unit is None or not remote.last_model:  # never a NaN or a non-unit vector onward
            self._malformed(PLACE, arrays)
            return None
        return unit, remote.last_model

    def health(self) -> dict[str, Any] | None:
        """The service's ``/health``, or ``None``."""
        return self.remotes[PLACE].health()

    def _arrays(
        self, name: str, answer: tuple[dict[str, str], bytes] | None
    ) -> dict[str, npt.NDArray[Any]] | None:
        if answer is None:
            return None
        try:
            return decode_arrays(lower_keys(answer[0]), answer[1])
        except BadRequestError as exc:
            self.remotes[name].last_error = f"undecodable answer: {exc}"[:160]
            self.remotes[name].failed += 1
            return None

    def _malformed(self, name: str, arrays: Mapping[str, npt.NDArray[Any]]) -> None:
        shapes = {k: v.shape for k, v in arrays.items()}
        self.remotes[name].last_error = f"an answer of the wrong layout: {shapes}"[:160]
        self.remotes[name].failed += 1


# ---------------------------------------------------------------- entry point
def main(argv: list[str] | None = None) -> None:
    """Serve the three models on the loopback (config/models.json's ``localization`` block)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", help="the models config (default: config/models.json)")
    parser.add_argument(
        "--host", default="127.0.0.1", help="bind address (loopback reaches the container)"
    )
    parser.add_argument("--port", type=int, help="overrides the config's port")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--xfeat-dir", default=None, help="the XFeat checkout (PEPIN_XFEAT_DIR)")
    args = parser.parse_args(argv)
    from pepin.log import setup_logging

    setup_logging("localization_service", log_dir=args.log_dir)
    config = load_config(args.config)
    port = args.port or config.port
    # The port first, the models second: under launchd a job whose port is taken would otherwise
    # load ~2 GB of models before failing to bind, once per restart.
    taken = port_taken(args.host, port)
    if taken is not None:
        log.error("localization service not started: %s", taken)
        raise SystemExit(2)
    xfeat_dir = (
        args.xfeat_dir
        or os.environ.get("PEPIN_XFEAT_DIR")
        or str(Path(__file__).resolve().parents[2] / "scratch/xfeat/data/accelerated_features")
    )
    t0 = time.perf_counter()
    models = build_models(config, xfeat_dir)
    server = ModelServer((args.host, port), models, name="localization service")
    log.info(
        "localization service on http://%s:%d in %.1f s: %s (GET /health)",
        args.host,
        port,
        time.perf_counter() - t0,
        ", ".join(f"POST /{m.name} ({m.tag} on {m.device})" for m in models),
    )
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        log.info("localization service stopped")


if __name__ == "__main__":
    main()
