"""RTAB-Map's Python feature detector: XFeat keypoints with their 64-float descriptors, computed by
the localisation service on the laptop's GPU, or here in RTAB-Map's own process when it cannot.

RTAB-Map loads this file by path (``Vis/FeatureType 15``, ``PyDetector/Path``) into the Python
interpreter it embeds, calls :func:`init` once and then :func:`detect` with a grey 8-bit picture,
and takes back two float32 arrays: the keypoints as N x 3 rows of (x, y, score) and the
descriptors as N x 64 (rtabmap/corelib/src/python/PyDetector.cpp). The score becomes the
keypoint's response. RTAB-Map applies no feature cap of its own to a Python detector
(Features2d.cpp skips ``limitKeypoints`` for it), so the ``top_k`` in force here is the cap.

WHERE IT IS COMPUTED is a live switch (pepin.live_settings, owned by rtabmap_frame's
``registration_backend`` flag and read on every call with one ``stat``): ``service`` — the
localisation service (``POST /xfeat``, pepin.localization_service; ``PEPIN_MODELS_URL``) or no
keypoints at all; ``local`` — XFeat here, on the Docker VM's CPU, as before 2026-09-24; ``auto``
(the default) — the service, and XFeat here for a call it does not answer. The service answers
the same arrays (pepin.xfeat_models is the one implementation both sides run), in 21-40 ms on
the GPU where this process took about 70 ms on the VM's CPU, and a node's stored picture it has
seen before costs it nothing (its LRU). The local model is built on the FIRST call that needs it
— torch is never imported while the service answers — and a service that does not answer is left
alone for 10 s at a time, so a dead service costs one timeout per 10 s, not one per call.

The counters (service / local / fallback / failed) go to RTAB-Map's log and to
pepin.live_settings' status file at most once a minute; rtabmap_frame prints them.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import numpy.typing as npt

try:  # the laptop image puts the library on PYTHONPATH; RTAB-Map's interpreter may not see it
    import pepin  # noqa: F401
except ImportError:  # pragma: no cover - depends on the interpreter RTAB-Map embeds
    sys.path.insert(0, os.environ.get("PEPIN_SRC", "/ws/pepin_src"))

from pepin import xfeat_models
from pepin.live_settings import RegistrationSettings, RegistrationSwitch, StatusBoard
from pepin.localization_service import LocalizationClient
from pepin.model_service import EveryMinute, ServiceOrLocal
from pepin.xfeat_models import DESCRIPTOR_DIM, TOP_K, XFEAT_DIR, rtabmap_arrays

__all__ = ["DESCRIPTOR_DIM", "TOP_K", "XFEAT_DIR", "detect", "init", "rtabmap_arrays"]

# Torch's intra-op threads for the LOCAL model in RTAB-Map's process: the container shares its
# cores with the rest of the stack; the benchmark's timing in the image is at this value.
THREADS = int(os.environ.get("PEPIN_XFEAT_THREADS", "4"))
NAME = "rtabmap_xfeat"

Features = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
Job = tuple[npt.NDArray[np.uint8], RegistrationSettings]

_local: Any = None
_client: LocalizationClient | None = None
_switch: ServiceOrLocal[Job, Features] | None = None
_settings = RegistrationSwitch()
_status = StatusBoard()
_minute = EveryMinute()


def _remote(job: Job) -> Features | None:
    assert _client is not None
    image, settings = job
    return _client.xfeat(image, settings.top_k, settings.timeout_s)


def _here(job: Job) -> Features:
    global _local
    image, settings = job
    if _local is None:
        _local = xfeat_models.load_xfeat(XFEAT_DIR, top_k=settings.top_k, threads=THREADS)
    return xfeat_models.features(_local, image, settings.top_k)


def init(cuda: int) -> None:
    """RTAB-Map's first call: the service's client and the switch. No model is loaded here — the
    local one is built by the first call that needs it. ``cuda`` is ignored (no GPU in Docker)."""
    global _client, _switch
    _client = LocalizationClient()
    _switch = ServiceOrLocal[Job, Features](_remote, _here, lambda _job: xfeat_models.no_features())


def detect(image: npt.ArrayLike) -> Features:
    """RTAB-Map's call per picture: (N x 3 keypoints, N x 64 descriptors), float32."""
    if _switch is None:
        init(0)
    assert _switch is not None and _client is not None
    settings = _settings.settings()
    picture = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    answer = _switch((picture, settings), settings.backend)
    if _minute.due():
        _report(settings)
    return answer


def _report(settings: RegistrationSettings) -> None:
    assert _switch is not None and _client is not None
    remote = _client.remotes["xfeat"]
    print(
        f"{NAME}: backend {settings.backend}, top_k {settings.top_k}; {_switch.line()};"
        f" service {_client.url}: {remote.status()}",
        file=sys.stderr,
        flush=True,
    )
    _status.publish(
        "xfeat",
        {
            "backend": settings.backend,
            "service": _switch.service,
            "local": _switch.local,
            "fallback": _switch.fallback,
            "failed": _switch.failed,
            "skipped": remote.skipped,
            "last_error": remote.last_error or _switch.last_error,
            "round_trip_ms": round(remote.round_trip.summary().median_ms, 1),
        },
    )


if __name__ == "__main__":
    # The image build's check: the model loads and answers on a picture of random tiles, whose
    # corners are what a detector finds — here, whatever the service does (local only).
    rng = np.random.default_rng(0)
    tiles = (np.kron(rng.random((60, 80)), np.ones((10, 10))) * 255).astype(np.uint8)
    points, descriptors = _here((tiles, RegistrationSettings.defaults()))
    assert points.dtype == np.float32 and points.shape[1] == 3, points.shape
    assert descriptors.shape == (len(points), DESCRIPTOR_DIM), descriptors.shape
    assert len(points) > 100, len(points)
    print(f"{NAME}: {len(points)} keypoints, top_k {TOP_K}, threads {THREADS}")
