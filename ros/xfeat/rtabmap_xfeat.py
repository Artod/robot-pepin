"""RTAB-Map's Python feature detector: XFeat keypoints with their 64-float descriptors.

RTAB-Map loads this file by path (``Vis/FeatureType 15``, ``PyDetector/Path``) into the Python
interpreter it embeds, calls :func:`init` once and then :func:`detect` with a grey 8-bit picture,
and takes back two float32 arrays: the keypoints as N x 3 rows of (x, y, score) and the
descriptors as N x 64 (rtabmap/corelib/src/python/PyDetector.cpp). The score becomes the
keypoint's response. RTAB-Map applies no feature cap of its own to a Python detector
(Features2d.cpp skips ``limitKeypoints`` for it), so :data:`TOP_K` here is the cap.

XFeat is verlab/accelerated_features (Apache-2.0), vendored into the image at a pinned commit
(ros/Dockerfile.xfeat); its weights ship in that checkout. Torch is imported in :func:`init`, not
at import time, so the conversion below is testable without it.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import numpy.typing as npt

# Where the vendored checkout lives (the image sets PEPIN_XFEAT_DIR; the offline benchmark points
# it at its own clone of the same commit).
XFEAT_DIR = os.environ.get("PEPIN_XFEAT_DIR", "/opt/xfeat/accelerated_features")
# Keypoints kept a picture, best score first: the cap the offline benchmark measured with
# (scratch/xfeat/xfeat_bench.py), XFeat's own examples use 2048-4096.
TOP_K = int(os.environ.get("PEPIN_XFEAT_TOP_K", "2048"))
# Torch's intra-op threads in RTAB-Map's process. The laptop container shares its cores with the
# depth network and the rest of the stack; the benchmark's timing in the image is at this value.
THREADS = int(os.environ.get("PEPIN_XFEAT_THREADS", "4"))
DESCRIPTOR_DIM = 64

_model: Any = None


def load_xfeat(xfeat_dir: str = XFEAT_DIR, top_k: int = TOP_K, threads: int = THREADS) -> Any:
    """XFeat on the CPU with the weights of the checkout at ``xfeat_dir``; returns the model."""
    import torch

    if xfeat_dir not in sys.path:
        sys.path.insert(0, xfeat_dir)
    from modules.xfeat import XFeat

    torch.set_grad_enabled(False)
    if threads > 0:
        torch.set_num_threads(threads)
    return XFeat(weights=os.path.join(xfeat_dir, "weights", "xfeat.pt"), top_k=top_k)


def rtabmap_arrays(
    keypoints: npt.ArrayLike, scores: npt.ArrayLike, descriptors: npt.ArrayLike
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """XFeat's output as RTAB-Map reads it: contiguous float32 N x 3 (x, y, score) and N x 64.

    RTAB-Map reads both buffers row by row with no strides, so a transposed or non-float32 array
    would be misread silently; an empty result is (0, 3) and (0, 64).
    """
    xy = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    score = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
    desc = np.asarray(descriptors, dtype=np.float32).reshape(-1, DESCRIPTOR_DIM)
    if not (len(xy) == len(score) == len(desc)):
        raise ValueError(f"{len(xy)} keypoints, {len(score)} scores, {len(desc)} descriptors")
    return np.ascontiguousarray(np.hstack([xy, score])), np.ascontiguousarray(desc)


def features(
    model: Any, image: npt.ArrayLike, top_k: int = TOP_K
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Keypoints and descriptors of one grey picture (H x W, uint8), as :func:`rtabmap_arrays`."""
    out = model.detectAndCompute(np.asarray(image), top_k=top_k)[0]
    return rtabmap_arrays(
        out["keypoints"].cpu().numpy(),
        out["scores"].cpu().numpy(),
        out["descriptors"].cpu().numpy(),
    )


def init(cuda: int) -> None:
    """RTAB-Map's first call: load the model. ``cuda`` is ignored — the container has no GPU."""
    global _model
    _model = load_xfeat()


def detect(image: npt.ArrayLike) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """RTAB-Map's call per picture: (N x 3 keypoints, N x 64 descriptors), float32."""
    if _model is None:
        init(0)
    return features(_model, image)


if __name__ == "__main__":
    # The image build's check: the model loads and answers on a picture of random tiles, whose
    # corners are what a detector finds.
    rng = np.random.default_rng(0)
    picture = (np.kron(rng.random((60, 80)), np.ones((10, 10))) * 255).astype(np.uint8)
    init(0)
    points, descriptors = detect(picture)
    assert points.dtype == np.float32 and points.shape[1] == 3, points.shape
    assert descriptors.shape == (len(points), DESCRIPTOR_DIM), descriptors.shape
    assert len(points) > 100, len(points)
    print(f"rtabmap_xfeat: {len(points)} keypoints, top_k {TOP_K}, threads {THREADS}")
