"""RTAB-Map's Python matcher: LighterGlue, the small LightGlue trained on XFeat's descriptors.

RTAB-Map loads this file by path (``Vis/CorNNType 6``, ``PyMatcher/Path``) into its embedded
interpreter, calls :func:`init` once and then :func:`match` for every pair it registers
(rtabmap/corelib/src/python/PyMatcher.cpp). The arguments come in RTAB-Map's order — the QUERY
side first (the picture being localised), then the TRAIN side (the database node) — and the
answer is an M x 2 integer array of (query index, train index) pairs, which RTAB-Map turns into
its matches as they are. It only gets here with float descriptors of equal width on both sides
and one camera (RegistrationVis.cpp), which is what :mod:`rtabmap_xfeat` produces.

LighterGlue is verlab/accelerated_features' ``modules/lighterglue.py`` (Apache-2.0) over kornia's
LightGlue (Apache-2.0), at the commit the image pins; its weights ship in that checkout.
``PyMatcher/Threshold`` is its match confidence floor (``min_conf``, 0.1 in XFeat's own
``match_lighterglue`` and in the offline benchmark).
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import numpy.typing as npt

XFEAT_DIR = os.environ.get("PEPIN_XFEAT_DIR", "/opt/xfeat/accelerated_features")
DEFAULT_MIN_CONF = 0.1

_matcher: Any = None
_min_conf = DEFAULT_MIN_CONF


def import_torch() -> Any:
    """torch, loaded so that ITS OWN BLAS answers its matrix products inside RTAB-Map.

    RTAB-Map's process has already loaded the system's reference libblas.so.3 (through its own
    dependencies) when the embedded interpreter first imports torch, and the dynamic linker binds
    libtorch_cpu's sgemm_ and friends to that first definition in the global scope: measured in
    pepin-laptop:xfeat, LighterGlue took 1.5 s a pair inside RTAB-Map against 0.4 s standalone,
    XFeat's convolutions (no BLAS) the same in both. RTLD_DEEPBIND puts torch's own dependency
    chain ahead of the global scope for the libraries this import opens. Elsewhere (a plain
    interpreter, macOS without the flag) it changes nothing. Kept in both adapters because RTAB-Map
    loads each by its own path and either may come first.
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


def load_lighterglue(xfeat_dir: str = XFEAT_DIR) -> Any:
    """LighterGlue on the CPU with the weights of the checkout at ``xfeat_dir``; the model."""
    torch = import_torch()
    if xfeat_dir not in sys.path:
        sys.path.insert(0, xfeat_dir)
    from modules.lighterglue import LighterGlue

    torch.set_grad_enabled(False)
    return LighterGlue(weights=os.path.join(xfeat_dir, "weights", "xfeat-lighterglue.pt")).eval()


def as_pairs(matches: npt.ArrayLike) -> npt.NDArray[np.int32]:
    """Matches as RTAB-Map reads them: a contiguous int32 M x 2 array, (0, 2) when empty."""
    return np.ascontiguousarray(np.asarray(matches, dtype=np.int32).reshape(-1, 2))


def pairs(
    matcher: Any,
    keypoints_query: npt.ArrayLike,
    keypoints_train: npt.ArrayLike,
    descriptors_query: npt.ArrayLike,
    descriptors_train: npt.ArrayLike,
    width: int,
    height: int,
    min_conf: float = DEFAULT_MIN_CONF,
) -> npt.NDArray[np.int32]:
    """(query index, train index) of every pair LighterGlue keeps at ``min_conf``, for two
    pictures of ``width`` x ``height``, the input built as XFeat's ``match_lighterglue`` does.
    A side with no keypoints is no pairs, answered before the model is touched."""
    kq = np.asarray(keypoints_query, dtype=np.float32).reshape(-1, 2)
    kt = np.asarray(keypoints_train, dtype=np.float32).reshape(-1, 2)
    if len(kq) == 0 or len(kt) == 0:
        return as_pairs(np.zeros((0, 2)))
    import torch

    dq = np.asarray(descriptors_query, dtype=np.float32).reshape(len(kq), -1)
    dt = np.asarray(descriptors_train, dtype=np.float32).reshape(len(kt), -1)
    size = torch.tensor((int(width), int(height)))[None, ...]
    data = {
        "keypoints0": torch.from_numpy(np.ascontiguousarray(kq))[None, ...],
        "keypoints1": torch.from_numpy(np.ascontiguousarray(kt))[None, ...],
        "descriptors0": torch.from_numpy(np.ascontiguousarray(dq))[None, ...],
        "descriptors1": torch.from_numpy(np.ascontiguousarray(dt))[None, ...],
        "image_size0": size,
        "image_size1": size,
    }
    out = matcher(data, min_conf=min_conf)
    return as_pairs(out["matches"][0].cpu().numpy())


def init(
    descriptor_dim: int, match_threshold: float, iterations: int, cuda: int, model: str
) -> None:
    """RTAB-Map's first call: load the model. ``match_threshold`` is ``min_conf`` (a value of 0
    or less keeps LighterGlue's own 0.1); the iteration count, CUDA and model name are
    SuperGlue's and mean nothing here."""
    global _matcher, _min_conf
    _matcher = load_lighterglue()
    _min_conf = float(match_threshold) if match_threshold > 0 else DEFAULT_MIN_CONF


def match(
    keypoints_query: npt.ArrayLike,
    keypoints_train: npt.ArrayLike,
    scores_query: npt.ArrayLike,
    scores_train: npt.ArrayLike,
    descriptors_query: npt.ArrayLike,
    descriptors_train: npt.ArrayLike,
    image_width: int,
    image_height: int,
) -> npt.NDArray[np.int32]:
    """RTAB-Map's call per pair: M x 2 (query index, train index), int32. Scores are unused:
    LighterGlue takes positions and descriptors only."""
    if _matcher is None:
        init(len(np.asarray(descriptors_query).reshape(-1)), DEFAULT_MIN_CONF, 0, 0, "")
    return pairs(
        _matcher,
        keypoints_query,
        keypoints_train,
        descriptors_query,
        descriptors_train,
        image_width,
        image_height,
        _min_conf,
    )


if __name__ == "__main__":
    # The image build's check: a textured picture and itself shifted by 12 px must match on the
    # shift. Needs rtabmap_xfeat beside this file, as it is in the image.
    import rtabmap_xfeat

    rng = np.random.default_rng(0)
    big = (np.kron(rng.random((62, 82)), np.ones((10, 10))) * 255).astype(np.uint8)
    first, second = big[:600, :800], big[:600, 12:812]
    rtabmap_xfeat.init(0)
    points_a, desc_a = rtabmap_xfeat.detect(first)
    points_b, desc_b = rtabmap_xfeat.detect(second)
    init(64, DEFAULT_MIN_CONF, 0, 0, "")
    found = match(points_b[:, :2], points_a[:, :2], points_b[:, 2], points_a[:, 2], desc_b, desc_a,
                  800, 600)  # fmt: skip
    shift = points_a[found[:, 1], 0] - points_b[found[:, 0], 0]
    right = int(np.sum(np.abs(shift - 12.0) < 2.0))
    assert found.dtype == np.int32 and found.shape[1] == 2, found.shape
    assert right > 50 and right > 0.6 * len(found), (right, len(found))
    print(f"rtabmap_lighterglue: {len(found)} matches, {right} on the 12 px shift")
