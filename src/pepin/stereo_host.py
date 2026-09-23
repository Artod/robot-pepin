"""RAFT-Stereo as the second model of the laptop's GPU host: a rectified pair in, a disparity out.

WHY A HOST AT ALL. Docker on macOS cannot see Metal, and the depth node runs in the laptop's
Linux container. The same forward pass is 89 ms on this Mac's MPS and roughly 3 fps on the
container's CPU at the full size — and the full size is the only size RAFT is worth having (see
:class:`RaftSettings`). So the network lives where the GPU is, in the NATIVE process
:mod:`pepin.depth_service` already runs there for the mono network, and the node talks to it over
the same loopback HTTP the mono depth uses (``host.docker.internal``, nothing on the LAN).

ONE PROCESS, TWO MODELS. This is a second endpoint of the existing depth service, not a second
service: ``POST /disparity`` beside ``POST /depth``, one port, one inference lock. Two processes
would each hold a Metal context and race for the one GPU with nothing serialising them, while the
node only ever runs ONE depth source — ``depth_source: network`` never asks for a disparity and
``depth_source: stereo`` never asks for a depth — so the model that is not in use costs nothing:
each is built on its first request. ``ros/depth_host.sh stereo`` builds RAFT at start instead, to
keep the first pair of a drive off the 1-2 s load.

THE WIRE. One request is one pair. The rectifier stays in the node — it owns the calibration and
the remap tables — so what crosses is two RECTIFIED grey eyes, raw: ``X-Height`` and ``X-Width``
are one eye's, the body is the left eye's bytes followed by the right eye's. Raw and not JPEG, in
spite of what the mono path measured: a disparity is read to a fraction of a pixel, and q90 ringing
on the texture either eye is matched by is exactly the error a subpixel fit cannot tell from a
real shift. 800x600 grey is 480 KB an eye, and it costs nothing measurable: 89.2 ms a pair end to
end against the host's own 88-90 ms forward pass (host_smoke.py, warmed host). The
answer is float16 (``application/x-pepin-disparity16``): at 128 px of disparity — the near end of
this rig, 0.18 m — float16's step is 0.06 px, an eighth of the half pixel
:data:`pepin.stereo_depth.DISPARITY_SIGMA_PX` the reach is derived from, and 0.09 mm of depth.

THE NODE'S SIDE is :class:`pepin.stereo_depth.RaftMatcher`: the live ``stereo_matcher`` flag in
{sgbm, raft} picks which engine answers the next pair, and a pair the host cannot answer falls to
SGBM and is counted, the way ``depth_backend: auto`` falls to the CPU model.

MEASURED 2026-09-22 on the 8 rectified pairs of scratch/stereo_net/frames/pairs.npz, grabbed off
the parked robot — glossy herringbone parquet with the ceiling lamp's reflection in view — against
``StereoMatcher()`` with the node's own defaults. ``sgbm_vs_raft.py`` scores both clouds against
ONE floor plane per frame, fitted from the SGBM cloud; ``phantom_where.py`` counts what a costmap
is actually told; ``host_smoke.py`` runs the whole shipped path, node side and host:

===================================  ========  ==========  ======
                                     SGBM      RAFT rt     source
===================================  ========  ==========  ======
airborne phantom pixels (0.5-1.5 m
up, 0.15-2.5 m ahead, |y| < 0.6 m)       6651        5265  sgbm_vs_raft.py
blobs those form, whole picture           450          94  phantom_where.py
of them, 50 px or more                    192          37  phantom_where.py
phantom pixels in the robot's own
path (|y| < 0.35 m)                      4486        2566  phantom_where.py
points within 5 cm of the floor        204459      958432  sgbm_vs_raft.py
points answered for at all             633851     2719782  sgbm_vs_raft.py
ms a pair, 800x600, end to end           16.8        89.2  host_smoke.py
===================================  ========  ==========  ======

THE PIXEL COUNT IS NOT THE ARGUMENT — it falls by a fifth, and RAFT is the worse of the two on
three of the eight pairs. What a costmap receives is: SGBM's airborne mass is 450 separate specks
sprayed over the open parquet where the lamp reflects, RAFT's is 94 blobs against the bright
doorway, and in the robot's own path the count halves. The floor is the other half of it: RAFT
answers for 4.7x as many points within 5 cm of the fitted plane, so the floor comes back whole
instead of in patches.

WHERE THE MEASUREMENT CAME FROM, and a warning. scratch/stereo_net/matcher_raft.patch, the note
this work was built from, gives the same comparison as 31314 -> 5338 phantoms, 426 -> 40 blobs and
274039 -> 1009276 floor points. THE RAFT HALF REPRODUCES (5264 in process, 5265 through the host,
958458 floor points); THE SGBM HALF DOES NOT — running that note's own script unchanged today
prints 6651 and 204459, not 31314 and 274039. Whatever SGBM configuration produced those, it is
not ``StereoMatcher()`` on these pairs, so none of them are quoted anywhere in this stack. The
numbers above are the ones the scripts print now, and the scripts are named beside each.

HALF SIZE IS DISQUALIFIED, not merely cheap: the note measured 248103 in-path phantoms at 1/2
size, twelve times SGBM's, because the network's disparity is scaled back up and its error with
it. If RAFT cannot be afforded at full size it cannot be afforded.
"""

from __future__ import annotations

import http.client
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from pepin.stereo import Array
from pepin.telemetry import LatencyTracker

log = logging.getLogger("pepin.stereo_host")

CONTENT_PAIR = "application/x-pepin-pair8"  # two rectified grey eyes, left then right
CONTENT_DISPARITY = "application/x-pepin-disparity16"
DISPARITY_PATH = "/disparity"
DEFAULT_WEIGHTS = "models/raftstereo-realtime.pth"  # relative to the repo root; see .gitignore
DEFAULT_EYE = (600, 800)  # (height, width) of one eye of the stereo head, for the warm-up
STAGES = ("decode", "infer", "pack", "total")


class StereoHostError(RuntimeError):
    """The stereo host did not answer, or answered something that is not a disparity image."""


class PairError(ValueError):
    """A request the host cannot read as a rectified pair (no size, wrong length, wrong type)."""


def repo_root() -> Path:
    """The checkout this module lives in: what a relative weights path is resolved against."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RaftSettings:
    """Every number the network takes, frozen, with the measurement behind each default.

    Only ``weights`` has no default worth having: an empty one means this head has no checkpoint
    and the ``stereo_matcher`` flag cannot be turned on for it. The rest are measured — see the
    module docstring's table.
    """

    # The checkpoint. "realtime" is not a compromise here: it MEASURED better than "middlebury"
    # on these frames (5264 airborne phantoms against the note's 42138) as well as four times
    # faster (89 ms a pair against 437).
    # Relative paths are resolved against the repo root, so config/camera.json carries one path
    # that means the same thing from any working directory.
    weights: str = ""
    # 7 is the realtime architecture's own trained iteration count. 12 and 32 are the middlebury
    # recipe's and cost 3.8x for a worse answer on this rig (scratch/stereo_net/bench_fps.py).
    iters: int = 7
    device: str = "mps"  # mps, cpu or cuda; a device torch cannot see falls back to the CPU
    # The node's own texture gate (pepin.stereo_depth's Sobel gate) over the network's dense
    # output. Off: the network is not confidently wrong on blank wall the way SGBM is, and the
    # floor RAFT wins is exactly what a texture gate throws away. Kept as a knob, unmeasured
    # through this path — the note that proposed it measured 4 % of the phantoms against 8 % of
    # the floor, and that note's SGBM numbers did not reproduce, so neither is quoted as fact.
    gate: bool = False
    # RAFT's correlation pyramid. Structural, but not derivable from a checkpoint on their own
    # (only their product is), so they are stated: these are upstream's for every released
    # checkpoint, and a wrong pair fails loudly at load rather than quietly at inference.
    corr_levels: int = 4
    corr_radius: int = 4

    def path(self) -> Path:
        """Where the checkpoint is, absolute; :class:`StereoHostError` when none is configured."""
        if not self.weights:
            raise StereoHostError(
                "no RAFT checkpoint configured: config/camera.json, the active head's 'net' block"
            )
        weights = Path(self.weights)
        return weights if weights.is_absolute() else repo_root() / weights

    def describe(self) -> str:
        """One phrase for a report line: the checkpoint's name, the iterations, the device."""
        name = Path(self.weights).name.removesuffix(".pth") or "no weights"
        return f"{name}/{self.iters} on {self.device}"


def architecture(state: Mapping[str, Any], settings: RaftSettings) -> SimpleNamespace:
    """The model arguments this CHECKPOINT was trained with, read off its own tensors rather than
    guessed from its filename — a file renamed by hand would otherwise build the wrong network
    and load into it with a plausible-looking error.

    ``shared_backbone``: the realtime recipe drops the separate feature encoder, so a checkpoint
    with no ``fnet.*`` shares its backbone. ``n_gru_layers``: one ``context_zqr_convs.N`` per
    layer. ``n_downsample``: the upsampling mask predicts ``(2^n)^2 * 9`` channels. ``hidden_dims``:
    the three GRUs' own widths. ``context_norm``: a batch norm carries running statistics, the
    others carry none. ``slow_fast_gru`` is the only one that is not in the file — it is a
    schedule, not a shape — and it travels with the shared backbone, which is upstream's realtime
    recipe (``--shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru``).
    """
    keys = set(state)
    shared = not any(k.startswith("fnet.") for k in keys)
    layers = sum(1 for k in keys if k.startswith("context_zqr_convs.") and k.endswith(".weight"))
    mask_out = int(state["update_block.mask.2.weight"].shape[0])
    factor = round((mask_out / 9) ** 0.5)
    downsample = max(factor.bit_length() - 1, 0)
    if factor * factor * 9 != mask_out or 1 << downsample != factor:
        raise StereoHostError(f"the upsampling mask predicts {mask_out} channels: not (2^n)^2 * 9")
    widths = [
        int(state[f"update_block.gru{level}.convz.weight"].shape[0]) for level in ("32", "16", "08")
    ]
    return SimpleNamespace(
        hidden_dims=widths,
        shared_backbone=shared,
        n_gru_layers=layers,
        n_downsample=downsample,
        slow_fast_gru=shared,
        context_norm="batch" if "cnet.norm1.running_mean" in keys else "instance",
        corr_implementation="reg",  # pure PyTorch: upstream's CUDA sampler does not build on MPS
        corr_levels=settings.corr_levels,
        corr_radius=settings.corr_radius,
        mixed_precision=False,
    )


class RaftNet:
    """RAFT-Stereo on one device: two rectified grey eyes in, a float32 disparity in pixels out.

    Built once, called from one thread at a time (the server holds its lock). Everything torch is
    imported here and not at module scope: the node's own process reads this module for the codec
    and must not pull in torch to do it.
    """

    def __init__(self, settings: RaftSettings | None = None) -> None:
        import torch

        self.settings = settings or RaftSettings()
        self._torch = torch
        path = self.settings.path()
        if not path.exists():
            raise StereoHostError(f"no RAFT checkpoint at {path}")
        device = self.settings.device
        if device == "mps" and not torch.backends.mps.is_available():
            log.warning("MPS is not available here: RAFT runs on the CPU, at about 3 fps")
            device = "cpu"
        self.device = device
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Upstream saves a DataParallel wrapper's state, so every key is prefixed.
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.args = architecture(state, self.settings)
        from pepin.vendor.raft_stereo import RAFTStereo

        model = RAFTStereo(self.args)
        model.load_state_dict(state)  # strict: a shape this checkpoint does not fit is loud
        self.model = model.to(device).eval()
        self.name = path.name
        log.info(
            "RAFT-Stereo %s on %s: %d GRU layers, 1/%d resolution, %s backbone, %d iterations",
            self.name,
            device,
            self.args.n_gru_layers,
            1 << self.args.n_downsample,
            "shared" if self.args.shared_backbone else "separate",
            self.settings.iters,
        )

    def __call__(self, left: Array, right: Array) -> Array:
        """The disparity of one rectified pair, float32 pixels at the pictures' own size. Dense:
        the network has no "I do not know" of its own, and the optional texture gate is the only
        thing that writes a NaN here."""
        torch = self._torch
        from pepin.vendor.raft_stereo import InputPadder

        if left.shape != right.shape:
            raise PairError(f"the eyes are {left.shape} and {right.shape}: not one rig")
        tensors = [
            torch.from_numpy(np.repeat(np.ascontiguousarray(eye)[:, :, None], 3, axis=2))
            .permute(2, 0, 1)
            .float()[None]
            .to(self.device)
            for eye in (left, right)
        ]
        padder = InputPadder(tensors[0].shape, divis_by=32)
        padded = padder.pad(*tensors)
        with torch.no_grad():
            _, flow = self.model(*padded, iters=self.settings.iters, test_mode=True)
        if self.device == "mps":
            torch.mps.synchronize()  # the queue is drained here, so the timing above it is real
        # RAFT answers a leftward flow; a disparity is its magnitude.
        disparity = -padder.unpad(flow).squeeze().cpu().numpy().astype(np.float32)
        if self.settings.gate:
            from pepin.stereo_depth import MatcherSettings, _textureless

            settings = MatcherSettings()
            blank = _textureless(left, settings.texture_threshold, settings.texture_window)
            if blank is not None:
                disparity[blank] = np.nan
        out: Array = disparity
        return out


# ---------------------------------------------------------------- codec: pairs and disparities
def encode_pair(left: Array, right: Array) -> tuple[dict[str, str], bytes]:
    """One rectified grey pair as request headers and body: the size is ONE eye's, the body is
    the left eye's bytes followed by the right eye's."""
    a, b = np.ascontiguousarray(left, dtype=np.uint8), np.ascontiguousarray(right, dtype=np.uint8)
    if a.shape != b.shape or a.ndim != 2:
        raise PairError(f"a pair is two grey pictures of one size, not {a.shape} and {b.shape}")
    h, w = int(a.shape[0]), int(a.shape[1])
    headers = {"Content-Type": CONTENT_PAIR, "X-Height": str(h), "X-Width": str(w)}
    return headers, a.tobytes() + b.tobytes()


def decode_pair(headers: Mapping[str, str], body: bytes) -> tuple[Array, Array]:
    """The two grey eyes a request carries; :class:`PairError` when it carries something else."""
    h = {str(k).lower(): str(v) for k, v in headers.items()}
    if h.get("content-type", "") != CONTENT_PAIR:
        raise PairError(f"unknown content type {h.get('content-type', '')!r}")
    try:
        height, width = int(h["x-height"]), int(h["x-width"])
    except (KeyError, ValueError) as exc:
        raise PairError("a pair needs X-Height and X-Width") from exc
    if height <= 0 or width <= 0 or len(body) != 2 * height * width:
        raise PairError(f"{len(body)} bytes is not two {height}x{width} grey eyes")
    both = np.frombuffer(body, dtype=np.uint8).reshape(2, height, width)
    return np.ascontiguousarray(both[0]), np.ascontiguousarray(both[1])


def pack_disparity(disparity: Array) -> tuple[dict[str, str], bytes]:
    """A disparity image as response headers and float16 body, its size in X-Height/X-Width."""
    h, w = int(disparity.shape[0]), int(disparity.shape[1])
    body = np.ascontiguousarray(disparity, dtype=np.float16).tobytes()
    return {"Content-Type": CONTENT_DISPARITY, "X-Height": str(h), "X-Width": str(w)}, body


def unpack_disparity(headers: Mapping[str, str], body: bytes) -> Array:
    """The float32 disparity of a response; :class:`StereoHostError` when it is not one."""
    h = {str(k).lower(): str(v) for k, v in headers.items()}
    if h.get("content-type", "") != CONTENT_DISPARITY:
        raise StereoHostError(f"not a disparity image: {h.get('content-type', '')!r}")
    try:
        height, width = int(h["x-height"]), int(h["x-width"])
    except (KeyError, ValueError) as exc:
        raise StereoHostError("a disparity image without a size") from exc
    if len(body) != height * width * 2:
        raise StereoHostError(f"{len(body)} bytes is not {height}x{width} float16")
    out: Array = np.frombuffer(body, dtype=np.float16).reshape(height, width).astype(np.float32)
    return out


# ---------------------------------------------------------------- the model behind the endpoint
class StereoModel:
    """The ``POST /disparity`` half of the GPU host: the codec, the network, and the counters the
    service's ``/health`` and report line print.

    The network is built on the FIRST PAIR, not here: ``depth_source: network`` never asks for a
    disparity, and a host started for the mono model should not pay 40 MB and a second of load for
    a model nobody is going to call. ``warm()`` is what ``ros/depth_host.sh stereo`` calls to build
    it at start instead.
    """

    def __init__(
        self,
        settings: RaftSettings | None = None,
        net: Callable[[Array, Array], Array] | None = None,
        warm_size: tuple[int, int] = DEFAULT_EYE,
    ) -> None:
        self.settings = settings or RaftSettings()
        self.warm_size = warm_size
        self.timing = {stage: LatencyTracker(stage) for stage in STAGES}
        self.requests = 0
        self.errors = 0
        # A network handed in is used as it stands — a test's fake, or one built elsewhere — and
        # nothing here ever imports torch then.
        self._net = net
        self._error: str | None = None

    @property
    def device(self) -> str:
        """Which device the network is on, or the one it will be built on."""
        return str(getattr(self._net, "device", self.settings.device))

    def warm(self) -> Callable[[Array, Array], Array]:
        """Build the network now (and keep it); a build that failed once is not retried, so a
        host whose checkpoint is missing says so on every pair instead of stalling on each."""
        if self._net is not None:
            return self._net
        if self._error is not None:
            raise StereoHostError(f"RAFT could not be built: {self._error}")
        t0 = time.perf_counter()
        try:
            self._net = RaftNet(self.settings)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:200]
            log.error("RAFT could not be built: %s", self._error)
            raise StereoHostError(f"RAFT could not be built: {self._error}") from exc
        # The FIRST forward pass compiles the Metal kernels and costs 2.2 s against the 91 ms
        # every later pair costs, and it compiles them PER SHAPE — so the warm-up runs at the eye
        # the active head actually serves, not at a token size. Paying it here is the whole point
        # of building at start: a pair that hit an unwarmed model would sit inside the host's lock
        # for two seconds, and the node's 2 s timeout would drop it and count a fall for nothing.
        blank = np.zeros(self.warm_size, dtype=np.uint8)
        self._net(blank, blank)
        log.info("RAFT-Stereo ready in %.1f s", time.perf_counter() - t0)
        return self._net

    def answer(self, headers: Mapping[str, str], body: bytes) -> tuple[dict[str, str], bytes]:
        """One pair through the network: the response headers and the float16 body."""
        t0 = time.perf_counter()
        left, right = decode_pair(headers, body)
        t1 = time.perf_counter()
        disparity = self.warm()(left, right)
        t2 = time.perf_counter()
        out_headers, out = pack_disparity(disparity)
        t3 = time.perf_counter()
        for stage, seconds in (
            ("decode", t1 - t0),
            ("infer", t2 - t1),
            ("pack", t3 - t2),
            ("total", t3 - t0),
        ):
            self.timing[stage].add(seconds)
        self.requests += 1
        out_headers["X-Infer-Ms"] = f"{(t2 - t1) * 1e3:.1f}"
        out_headers["X-Model"] = self.settings.describe()
        return out_headers, out

    def health(self) -> dict[str, Any]:
        """What ``/health``'s ``stereo`` block says: the checkpoint, the device, the counters and
        the per-stage latencies; ``built`` false while no pair has been asked for yet."""
        name = Path(self.settings.weights).name.removesuffix(".pth") or "no weights"
        return {
            "model": f"{name}/{self.settings.iters}",  # the device is its own key beside this
            "device": self.device,
            "iters": self.settings.iters,
            "built": self._net is not None,
            "error": self._error,
            "requests": self.requests,
            "errors": self.errors,
            "ms": {
                stage: {
                    "median": round(t.summary().median_ms, 1),
                    "p95": round(t.summary().p95_ms, 1),
                    "max": round(t.summary().max_ms, 1),
                }
                for stage, t in self.timing.items()
            },
        }

    def report(self) -> str:
        """One clause for the service's report line: the model, what it served, its stages."""
        stages = " ".join(
            f"{stage} {t.summary().median_ms:.0f}/{t.summary().p95_ms:.0f}"
            for stage, t in self.timing.items()
        )
        state = "ready" if self._net is not None else (self._error or "not built yet")
        return (
            f"stereo ({self.settings.describe()}, {state}): {self.requests} pairs,"
            f" {self.errors} refused, ms median/p95: {stages}"
        )


# ---------------------------------------------------------------- the client
class RemoteDisparity:
    """The host's ``/disparity`` endpoint behind one call: two rectified grey eyes in, a float32
    disparity in pixels out, on one keep-alive connection.

    Any failure — no host, a timeout, a wrong answer — is a :class:`StereoHostError` and drops the
    connection, so the next call reconnects. One caller sends pairs (the node's depth worker);
    :meth:`health` takes a connection of its own so a report timer may ask mid-pair.
    """

    def __init__(self, url: str, timeout_s: float = 2.0) -> None:
        if not url.startswith("http://"):
            raise ValueError(f"the stereo host URL must start with http://, not {url!r}")
        self.url = url
        host_port = url[len("http://") :].rstrip("/")
        self._host, _, port = host_port.partition(":")
        self._port = int(port) if port else 80
        self._timeout = timeout_s
        self._conn: http.client.HTTPConnection | None = None
        self.timing = {"encode": LatencyTracker("encode"), "round_trip": LatencyTracker("rt")}
        self.last_infer_ms = 0.0  # the host's own forward pass on the last pair
        self.last_model = ""  # and which checkpoint answered it

    def close(self) -> None:
        """Drop the connection; the next call opens a new one."""
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __call__(self, left: Array, right: Array) -> Array:
        t0 = time.perf_counter()
        headers, body = encode_pair(left, right)
        t1 = time.perf_counter()
        try:
            if self._conn is None:
                self._conn = http.client.HTTPConnection(self._host, self._port, self._timeout)
            self._conn.request("POST", DISPARITY_PATH, body=body, headers=headers)
            response = self._conn.getresponse()
            data = response.read()
            if response.status != 200:
                raise StereoHostError(f"{response.status}: {data[:200].decode(errors='replace')}")
            reply = {k.lower(): v for k, v in response.getheaders()}
            disparity = unpack_disparity(reply, data)
        except (OSError, http.client.HTTPException) as exc:
            self.close()
            raise StereoHostError(f"{type(exc).__name__}: {exc}") from exc
        except StereoHostError:
            self.close()
            raise
        t2 = time.perf_counter()
        self.last_infer_ms = float(reply.get("x-infer-ms", "0") or 0.0)
        self.last_model = reply.get("x-model", "")
        self.timing["encode"].add(t1 - t0)
        self.timing["round_trip"].add(t2 - t1)
        return disparity

    def health(self) -> dict[str, Any]:
        """The host's ``/health`` as a dict; :class:`StereoHostError` when it is not up."""
        conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            data = response.read()
            if response.status != 200:
                raise StereoHostError(f"/health answered {response.status}")
        except (OSError, http.client.HTTPException) as exc:
            raise StereoHostError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            conn.close()
        result: dict[str, Any] = json.loads(data)
        return result


def settings_from_config(config: str | Path, board: str = "127.0.0.1") -> RaftSettings:
    """The active head's ``net`` block of ``config/camera.json`` as :class:`RaftSettings` — the
    checkpoint, the iterations and the device belong to the camera, not to this module."""
    from pepin.camera import CameraConfig

    cfg = CameraConfig.load(config, board=board)
    return RaftSettings(
        weights=cfg.net_weights,
        iters=cfg.net_iters,
        device=cfg.net_device,
        gate=cfg.net_gate,
    )


def eye_from_config(config: str | Path, board: str = "127.0.0.1") -> tuple[int, int]:
    """The active head's ONE EYE as (height, width): the shape the warm-up must compile for,
    which is the shape the node will send, and it is the camera that says what it is."""
    from pepin.camera import CameraConfig

    cfg = CameraConfig.load(config, board=board)
    return int(cfg.height), int(cfg.width)
