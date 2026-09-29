"""Which place is this picture: a learned global descriptor, one vector per picture.

RTAB-Map's own answer to "which database node is this" is the bag of ORB words, and against the
daylight database under the evening lamps its hypotheses read 0.05-0.07 — nothing to recognise
on. A learned descriptor was benchmarked on the same fixed set (scratch/vpr/vpr_bench.py,
2026-09-24: 147 daylight nodes against 219 evening frames of runs 0455-0465 with the lidar-held
truth, a retrieval right within 1.5 m and 45 deg): BoQ over DINOv2 recognises R@1 0.99 in 42 ms
on the laptop's GPU (712 ms on the Docker VM's CPU), BoQ over ResNet-50 R@1 0.98 in 24 ms. Both
are Bag-of-Queries (amaralibey/bag-of-queries, MIT) with the authors' weights from torch.hub.

THE PREPROCESSING IS THE BENCHMARK'S, exactly, because that is what the numbers above were measured
with: the RGB picture resized to the model's square input by PIL's bilinear filter (which
antialiases when it shrinks — OpenCV's does not, and the vectors would differ), scaled to [0, 1],
normalised by ImageNet's mean and deviation; the model's first output flattened and L2-normalised.

OFFLINE BY CONSTRUCTION. torch.hub asks GitHub for a repository's default branch whenever a
``owner/repo`` without a branch is loaded — BoQ's own DINOv2 backbone loads
``facebookresearch/dinov2`` that way — so a start without internet would wait on the network
first. :func:`load_place_model` loads every hub repository from its cached checkout
(``~/.cache/torch/hub/<owner>_<repo>_<branch>``) with ``source="local"`` and fails loudly, naming
the missing directory, when it is not cached; the weights come from the hub's checkpoint cache.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.global_descriptor import normalised
from pepin.xfeat_models import weights_tag

MEAN = (0.485, 0.456, 0.406)  # ImageNet's, the benchmark's normalisation
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PlaceSpec:
    """One place model: its hub repository and entry point, the entry's arguments, the square
    input it was measured at (height, width), its vector's length and its checkpoint's file name
    in the hub's cache (what its tag hashes)."""

    repo: str
    entry: str
    kwargs: tuple[tuple[str, Any], ...]
    size: tuple[int, int]
    dim: int
    checkpoint: str


PLACE_MODELS = {
    "boq_dinov2": PlaceSpec(
        repo="amaralibey/bag-of-queries",
        entry="get_trained_boq",
        kwargs=(("backbone_name", "dinov2"), ("output_dim", 12288)),
        size=(322, 322),
        dim=12288,
        checkpoint="dinov2_12288.pth",
    ),
    "boq_r50": PlaceSpec(
        repo="amaralibey/bag-of-queries",
        entry="get_trained_boq",
        kwargs=(("backbone_name", "resnet50"), ("output_dim", 16384)),
        size=(384, 384),
        dim=16384,
        checkpoint="resnet50_16384.pth",
    ),
}
DEFAULT_PLACE_MODEL = "boq_dinov2"


def place_spec(name: str) -> PlaceSpec:
    """The spec of ``name``; :class:`ValueError` naming the known ones otherwise."""
    if name not in PLACE_MODELS:
        raise ValueError(f"unknown place model {name!r}; models: {sorted(PLACE_MODELS)}")
    return PLACE_MODELS[name]


def hub_dir() -> Path:
    """torch.hub's directory (``TORCH_HOME``/hub, ``~/.cache/torch/hub`` by default)."""
    torch_home = os.environ.get("TORCH_HOME", os.path.join(Path.home(), ".cache", "torch"))
    return Path(torch_home) / "hub"


def cached_repo(repo: str, root: Path | None = None) -> Path:
    """The cached checkout of ``owner/repo[:branch]`` (branch ``main`` unless named);
    :class:`FileNotFoundError` naming it when it is not there."""
    name, _, branch = repo.partition(":")
    owner, _, project = name.partition("/")
    path = (root or hub_dir()) / f"{owner}_{project}_{branch or 'main'}"
    if not path.is_dir():
        raise FileNotFoundError(
            f"{repo} is not in torch.hub's cache ({path}): load it once with network"
        )
    return path


@contextlib.contextmanager
def hub_from_cache() -> Iterator[None]:
    """Within the block, every ``torch.hub.load("owner/repo", ...)`` loads the cached checkout
    with ``source="local"`` — no GitHub request, whoever calls it (BoQ's DINOv2 backbone does)."""
    import torch

    original = torch.hub.load

    def load(repo_or_dir: str, model: str, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("source", "github") == "github" and not os.path.isdir(repo_or_dir):
            kwargs["source"] = "local"
            kwargs.pop("trust_repo", None)
            kwargs.pop("force_reload", None)
            kwargs.pop("skip_validation", None)
            repo_or_dir = str(cached_repo(repo_or_dir))
        return original(repo_or_dir, model, *args, **kwargs)  # type: ignore[no-untyped-call]

    torch.hub.load = load
    try:
        yield
    finally:
        torch.hub.load = original


def load_place_model(name: str, device: str = "mps") -> Any:
    """The place model ``name`` in eval mode on ``device``, from the hub's caches only."""
    import torch

    spec = place_spec(name)
    with hub_from_cache():
        model = torch.hub.load(spec.repo, spec.entry, **dict(spec.kwargs))  # type: ignore[no-untyped-call]
    return model.eval().to(device)


def place_tag(name: str) -> str:
    """``name@<checkpoint hash prefix>``: which weights described a place."""
    spec = place_spec(name)
    return weights_tag(name, hub_dir() / "checkpoints" / spec.checkpoint)


def preprocess(rgb: npt.NDArray[np.uint8], size: tuple[int, int]) -> npt.NDArray[np.float32]:
    """An RGB picture (H x W x 3, uint8) as the model's normalised 3 x H x W input at
    ``size`` (height, width), resized by PIL's bilinear filter exactly as the benchmark did."""
    from PIL import Image

    picture = Image.fromarray(np.ascontiguousarray(rgb))
    resized = picture.resize((int(size[1]), int(size[0])), Image.Resampling.BILINEAR)
    x = np.asarray(resized, dtype=np.float32) / 255.0
    x = (x - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def describe(model: Any, rgb: npt.NDArray[np.uint8], size: tuple[int, int], device: str) -> Any:
    """One picture's L2-normalised float32 place vector."""
    import torch

    x = torch.from_numpy(preprocess(rgb, size))[None, ...].to(device)
    with torch.inference_mode():
        out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return normalised(out.reshape(-1).float().cpu().numpy())
