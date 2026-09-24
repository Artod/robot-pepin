"""XFeat keypoints and LighterGlue matches: the one implementation both RTAB-Map's adapters and
the localisation service run.

XFeat is verlab/accelerated_features (Apache-2.0) at the commit ros/Dockerfile.xfeat pins; its
weights ship in that checkout (``PEPIN_XFEAT_DIR``: /opt/xfeat/accelerated_features in the image,
scratch/xfeat/data/accelerated_features on the laptop). LighterGlue is that checkout's
``modules/lighterglue.py`` over kornia's LightGlue (Apache-2.0, kornia pinned to 0.8.3 on both
sides). Nothing here imports torch at module import: the arrays' layout (:func:`rtabmap_arrays`,
:func:`as_pairs`) is testable without it, and a caller that only talks to the service never
loads it.

THE LAYOUT IS RTAB-MAP'S, and it is the contract: RTAB-Map's C++ reads the detector's two buffers
and the matcher's one row by row with no strides (rtabmap/corelib/src/python/PyDetector.cpp,
PyMatcher.cpp), so the keypoints are a contiguous float32 N x 3 of (x, y, score), the descriptors
a contiguous float32 N x 64 and the matches a contiguous int32 M x 2 of (query index, train
index). A transposed or non-float32 array would be misread in silence.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# Where the vendored checkout lives (the image sets PEPIN_XFEAT_DIR; the laptop's clone of the same
# commit is scratch/xfeat/data/accelerated_features).
XFEAT_DIR = os.environ.get("PEPIN_XFEAT_DIR", "/opt/xfeat/accelerated_features")
# Keypoints kept a picture, best score first: the cap the offline benchmark measured with
# (scratch/xfeat/xfeat_bench.py); XFeat's own examples use 2048-4096. RTAB-Map applies no cap of
# its own to a Python detector (Features2d.cpp skips limitKeypoints for it), so this is the cap.
TOP_K = 2048
DESCRIPTOR_DIM = 64
# LighterGlue's match confidence floor: 0.1 in XFeat's own match_lighterglue and in the benchmark
# (RTAB-Map's PyMatcher/Threshold default 0.2 is SuperGlue's).
MIN_CONF = 0.1
XFEAT_WEIGHTS = "weights/xfeat.pt"
LIGHTERGLUE_WEIGHTS = "weights/xfeat-lighterglue.pt"

Float = npt.NDArray[np.float32]
Pairs = npt.NDArray[np.int32]


def import_torch() -> Any:
    """torch, loaded so that ITS OWN BLAS answers its matrix products inside RTAB-Map.

    RTAB-Map's process has already loaded the system's reference libblas.so.3 (through its own
    dependencies) when the embedded interpreter first imports torch, and the dynamic linker binds
    libtorch_cpu's sgemm_ and friends to that first definition in the global scope: measured in
    pepin-laptop:xfeat, LighterGlue took 1.5 s a pair inside RTAB-Map against 0.4 s standalone,
    XFeat's convolutions (no BLAS) the same in both. RTLD_DEEPBIND puts torch's own dependency
    chain ahead of the global scope for the libraries this import opens. Elsewhere (a plain
    interpreter, macOS, which has no such flag) it changes nothing.
    """
    if "torch" in sys.modules:
        return sys.modules["torch"]
    flags = sys.getdlopenflags()
    sys.setdlopenflags(flags | getattr(os, "RTLD_DEEPBIND", 0))
    try:
        import torch
    finally:
        sys.setdlopenflags(flags)
    return torch


def weights_tag(name: str, path: str | Path, digits: int = 8) -> str:
    """``name@<the first digits of the weights file's sha256>``: which weights answered."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return f"{name}@{digest.hexdigest()[:digits]}"


def _on_path(xfeat_dir: str) -> None:
    if xfeat_dir not in sys.path:
        sys.path.insert(0, xfeat_dir)


def load_xfeat(
    xfeat_dir: str = XFEAT_DIR, top_k: int = TOP_K, threads: int = 0, device: str = "cpu"
) -> Any:
    """XFeat with the checkout's weights, on ``device`` (``cpu`` or ``mps``); ``threads`` > 0 sets
    torch's intra-op threads (inside RTAB-Map's container, where the cores are shared)."""
    torch = import_torch()
    _on_path(xfeat_dir)
    from modules.xfeat import XFeat

    torch.set_grad_enabled(False)
    if threads > 0:
        torch.set_num_threads(threads)
    model = XFeat(weights=os.path.join(xfeat_dir, XFEAT_WEIGHTS), top_k=top_k)
    if device != "cpu":
        # XFeat picks its device in the constructor (cuda or cpu) and moves every input there
        # (preprocess_tensor), so both the network and that choice move.
        model.dev = torch.device(device)
        model.net = model.net.to(device)
    return model


def load_lighterglue(xfeat_dir: str = XFEAT_DIR, device: str = "cpu") -> Any:
    """LighterGlue with the checkout's weights, on ``device``."""
    torch = import_torch()
    _on_path(xfeat_dir)
    from modules.lighterglue import LighterGlue

    torch.set_grad_enabled(False)
    matcher = LighterGlue(weights=os.path.join(xfeat_dir, LIGHTERGLUE_WEIGHTS)).eval()
    if device != "cpu":
        matcher.dev = torch.device(device)
        matcher.net = matcher.net.to(device)
    return matcher


def rtabmap_arrays(
    keypoints: npt.ArrayLike, scores: npt.ArrayLike, descriptors: npt.ArrayLike
) -> tuple[Float, Float]:
    """XFeat's output as RTAB-Map reads it: contiguous float32 N x 3 (x, y, score) and N x 64;
    an empty result is (0, 3) and (0, 64)."""
    xy = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    score = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
    desc = np.asarray(descriptors, dtype=np.float32).reshape(-1, DESCRIPTOR_DIM)
    if not (len(xy) == len(score) == len(desc)):
        raise ValueError(f"{len(xy)} keypoints, {len(score)} scores, {len(desc)} descriptors")
    return np.ascontiguousarray(np.hstack([xy, score])), np.ascontiguousarray(desc)


def no_features() -> tuple[Float, Float]:
    """The empty answer in RTAB-Map's layout: (0, 3) keypoints and (0, 64) descriptors."""
    return rtabmap_arrays(np.zeros((0, 2)), np.zeros(0), np.zeros((0, DESCRIPTOR_DIM)))


def features(model: Any, image: npt.ArrayLike, top_k: int = TOP_K) -> tuple[Float, Float]:
    """Keypoints and descriptors of one grey picture (H x W, uint8), as :func:`rtabmap_arrays`."""
    out = model.detectAndCompute(np.asarray(image), top_k=top_k)[0]
    return rtabmap_arrays(
        out["keypoints"].cpu().numpy(),
        out["scores"].cpu().numpy(),
        out["descriptors"].cpu().numpy(),
    )


def as_pairs(matches: npt.ArrayLike) -> Pairs:
    """Matches as RTAB-Map reads them: a contiguous int32 M x 2 array, (0, 2) when empty."""
    return np.ascontiguousarray(np.asarray(matches, dtype=np.int32).reshape(-1, 2))


def match(
    matcher: Any,
    keypoints_query: npt.ArrayLike,
    keypoints_train: npt.ArrayLike,
    descriptors_query: npt.ArrayLike,
    descriptors_train: npt.ArrayLike,
    size_query: tuple[int, int],
    size_train: tuple[int, int],
    min_conf: float = MIN_CONF,
) -> tuple[Pairs, Float]:
    """(query index, train index) of every pair LighterGlue keeps at ``min_conf`` and each pair's
    confidence, for pictures of (width, height) ``size_query`` and ``size_train``; the input is
    built as XFeat's ``match_lighterglue`` builds it. A side with no keypoints is no pairs,
    answered before the model is touched."""
    kq = np.asarray(keypoints_query, dtype=np.float32).reshape(-1, 2)
    kt = np.asarray(keypoints_train, dtype=np.float32).reshape(-1, 2)
    if len(kq) == 0 or len(kt) == 0:
        return as_pairs(np.zeros((0, 2))), np.zeros(0, dtype=np.float32)
    torch = import_torch()
    device = getattr(matcher, "dev", "cpu")
    dq = np.asarray(descriptors_query, dtype=np.float32).reshape(len(kq), -1)
    dt = np.asarray(descriptors_train, dtype=np.float32).reshape(len(kt), -1)

    def tensor(array: npt.NDArray[np.float32]) -> Any:
        return torch.from_numpy(np.ascontiguousarray(array))[None, ...].to(device)

    data = {
        "keypoints0": tensor(kq),
        "keypoints1": tensor(kt),
        "descriptors0": tensor(dq),
        "descriptors1": tensor(dt),
        "image_size0": torch.tensor(tuple(int(v) for v in size_query))[None, ...].to(device),
        "image_size1": torch.tensor(tuple(int(v) for v in size_train))[None, ...].to(device),
    }
    out = matcher(data, min_conf=min_conf)
    pairs = as_pairs(out["matches"][0].cpu().numpy())
    scores = np.ascontiguousarray(out["scores"][0].cpu().numpy(), dtype=np.float32).reshape(-1)
    return pairs, scores
