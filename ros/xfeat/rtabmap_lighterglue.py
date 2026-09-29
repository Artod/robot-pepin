"""RTAB-Map's Python matcher: LighterGlue, the small LightGlue trained on XFeat's descriptors,
computed by the localisation service on the laptop, or here in RTAB-Map's process when it cannot.

RTAB-Map loads this file by path (``Vis/CorNNType 6``, ``PyMatcher/Path``) into its embedded
interpreter, calls :func:`init` once and then :func:`match` for every pair it registers
(rtabmap/corelib/src/python/PyMatcher.cpp). The arguments come in RTAB-Map's order — the QUERY
side first (the picture being localised), then the TRAIN side (the database node) — and the
answer is an M x 2 integer array of (query index, train index) pairs, which RTAB-Map turns into
its matches as they are. It only gets here with float descriptors of equal width on both sides
and one camera (RegistrationVis.cpp), which is what :mod:`rtabmap_xfeat` produces.

LighterGlue is verlab/accelerated_features' ``modules/lighterglue.py`` (Apache-2.0) over kornia's
LightGlue (Apache-2.0), at the commit the image pins; pepin.xfeat_models is the one implementation
the service and this process run. ``PyMatcher/Threshold`` is its match confidence floor
(``min_conf``, 0.1 in XFeat's own ``match_lighterglue`` and in the offline benchmark).

WHERE IT IS COMPUTED is the same live switch as the detector's (rtabmap_xfeat's docstring):
``service`` (``POST /match``, or no pairs), ``local`` (here, as before 2026-09-24) or ``auto`` (the
service, and here for a pair it does not answer); the local model is built by the first call
that needs it.
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
from pepin.xfeat_models import MIN_CONF, XFEAT_DIR, as_pairs

__all__ = ["DEFAULT_MIN_CONF", "XFEAT_DIR", "as_pairs", "init", "match", "pairs"]

DEFAULT_MIN_CONF = MIN_CONF
NAME = "rtabmap_lighterglue"


class Pair:
    """One pair to match: both sides' keypoints (N x 2) and descriptors, the pictures' size, the
    confidence floor and the settings of this call."""

    def __init__(
        self,
        kq: npt.NDArray[np.float32],
        kt: npt.NDArray[np.float32],
        dq: npt.NDArray[np.float32],
        dt: npt.NDArray[np.float32],
        size: tuple[int, int],
        min_conf: float,
        settings: RegistrationSettings,
    ) -> None:
        self.kq, self.kt, self.dq, self.dt = kq, kt, dq, dt
        self.size, self.min_conf, self.settings = size, min_conf, settings


Pairs = npt.NDArray[np.int32]

_local: Any = None
_min_conf = DEFAULT_MIN_CONF
_client: LocalizationClient | None = None
_switch: ServiceOrLocal[Pair, Pairs] | None = None
_settings = RegistrationSwitch()
_status = StatusBoard()
_minute = EveryMinute()


def pairs(
    matcher: Any,
    keypoints_query: npt.ArrayLike,
    keypoints_train: npt.ArrayLike,
    descriptors_query: npt.ArrayLike,
    descriptors_train: npt.ArrayLike,
    width: int,
    height: int,
    min_conf: float = DEFAULT_MIN_CONF,
) -> Pairs:
    """(query index, train index) of every pair ``matcher`` keeps at ``min_conf`` for two
    pictures of ``width`` x ``height`` (pepin.xfeat_models.match, the pairs alone)."""
    found, _scores = xfeat_models.match(
        matcher,
        keypoints_query,
        keypoints_train,
        descriptors_query,
        descriptors_train,
        (width, height),
        (width, height),
        min_conf,
    )
    return found


def _remote(pair: Pair) -> Pairs | None:
    assert _client is not None
    answer = _client.match(
        pair.kq, pair.kt, pair.dq, pair.dt, pair.size, pair.size, pair.min_conf,
        pair.settings.timeout_s,
    )  # fmt: skip
    return None if answer is None else answer[0]


def _here(pair: Pair) -> Pairs:
    global _local
    if _local is None:
        _local = xfeat_models.load_lighterglue(XFEAT_DIR)
    return pairs(_local, pair.kq, pair.kt, pair.dq, pair.dt, *pair.size, pair.min_conf)


def init(
    descriptor_dim: int, match_threshold: float, iterations: int, cuda: int, model: str
) -> None:
    """RTAB-Map's first call: the service's client and the switch; no model is loaded here.
    ``match_threshold`` is ``min_conf`` (a value of 0 or less keeps LighterGlue's own 0.1); the
    iteration count, CUDA and model name are SuperGlue's and mean nothing here."""
    global _client, _switch, _min_conf
    _min_conf = float(match_threshold) if match_threshold > 0 else DEFAULT_MIN_CONF
    _client = LocalizationClient()
    _switch = ServiceOrLocal[Pair, Pairs](_remote, _here, lambda _pair: as_pairs(np.zeros((0, 2))))


def match(
    keypoints_query: npt.ArrayLike,
    keypoints_train: npt.ArrayLike,
    scores_query: npt.ArrayLike,
    scores_train: npt.ArrayLike,
    descriptors_query: npt.ArrayLike,
    descriptors_train: npt.ArrayLike,
    image_width: int,
    image_height: int,
) -> Pairs:
    """RTAB-Map's call per pair: M x 2 (query index, train index), int32. Scores are unused:
    LighterGlue takes positions and descriptors only. A side with no keypoints is no pairs,
    answered before anything is asked."""
    if _switch is None:
        init(64, DEFAULT_MIN_CONF, 0, 0, "")
    assert _switch is not None
    kq = np.asarray(keypoints_query, dtype=np.float32).reshape(-1, 2)
    kt = np.asarray(keypoints_train, dtype=np.float32).reshape(-1, 2)
    if len(kq) == 0 or len(kt) == 0:
        return as_pairs(np.zeros((0, 2)))
    dq = np.asarray(descriptors_query, dtype=np.float32).reshape(len(kq), -1)
    dt = np.asarray(descriptors_train, dtype=np.float32).reshape(len(kt), -1)
    settings = _settings.settings()
    pair = Pair(kq, kt, dq, dt, (int(image_width), int(image_height)), _min_conf, settings)
    answer = _switch(pair, settings.backend)
    if _minute.due():
        _report(settings)
    return answer


def _report(settings: RegistrationSettings) -> None:
    assert _switch is not None and _client is not None
    remote = _client.remotes["match"]
    print(
        f"{NAME}: backend {settings.backend}, min_conf {_min_conf:g}; {_switch.line()};"
        f" service {_client.url}: {remote.status()}",
        file=sys.stderr,
        flush=True,
    )
    _status.publish(
        "match",
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
    # The image build's check: a textured picture and itself shifted by 12 px must match on the
    # shift, computed here (local) whatever the service does.
    rng = np.random.default_rng(0)
    big = (np.kron(rng.random((62, 82)), np.ones((10, 10))) * 255).astype(np.uint8)
    first, second = big[:600, :800], big[:600, 12:812]
    xfeat = xfeat_models.load_xfeat(XFEAT_DIR)
    points_a, desc_a = xfeat_models.features(xfeat, first)
    points_b, desc_b = xfeat_models.features(xfeat, second)
    glue = xfeat_models.load_lighterglue(XFEAT_DIR)
    found = pairs(glue, points_b[:, :2], points_a[:, :2], desc_b, desc_a, 800, 600)
    shift = points_a[found[:, 1], 0] - points_b[found[:, 0], 0]
    right = int(np.sum(np.abs(shift - 12.0) < 2.0))
    assert found.dtype == np.int32 and found.shape[1] == 2, found.shape
    assert right > 50 and right > 0.6 * len(found), (right, len(found))
    print(f"{NAME}: {len(found)} matches, {right} on the 12 px shift")
