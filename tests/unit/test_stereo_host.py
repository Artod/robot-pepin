"""pepin.stereo_host and pepin.stereo_depth.RaftMatcher: the pair-and-disparity wire, the
architecture read off a checkpoint, and the node's side of the GPU host.

NO TORCH IS IMPORTED HERE and no network is ever run: the host's model is a fake handed to
:class:`pepin.stereo_host.StereoModel`, served by a real :class:`pepin.depth_service.DepthServer`
on the loopback, so what is exercised is the codec, the routing, the counters and the fallback —
everything that can be wrong without a GPU. What RAFT itself measures is in scratch/stereo_net/
and in the modules' docstrings.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from pepin.depth_service import DepthServer
from pepin.stereo_depth import (
    MATCHERS,
    Baseline,
    MatcherSettings,
    RaftMatcher,
    StereoDepth,
    StereoMatcher,
    StereoUnavailableError,
    build_matcher,
)
from pepin.stereo_host import (
    CONTENT_DISPARITY,
    CONTENT_PAIR,
    PairError,
    RaftSettings,
    RemoteDisparity,
    StereoHostError,
    StereoModel,
    architecture,
    decode_pair,
    encode_pair,
    pack_disparity,
    repo_root,
    settings_from_config,
    unpack_disparity,
)

CONFIG = repo_root() / "config/camera.json"


def _eye(h: int = 12, w: int = 20, shift: int = 0) -> np.ndarray:
    """A grey picture with a readable gradient, so a round trip that transposed it would show."""
    rows = np.arange(h)[:, None] * 5
    cols = np.arange(w)[None, :] * 3
    return np.ascontiguousarray(((rows + cols + shift) % 256).astype(np.uint8))


# ---------------------------------------------------------------- codec
def test_a_pair_and_a_disparity_survive_the_round_trip() -> None:
    """Both eyes come back as they went, in order, and a disparity to well under the half pixel
    the reach is derived from — float16 is the format and this is what it costs."""
    left, right = _eye(), _eye(shift=7)
    headers, body = encode_pair(left, right)
    assert headers["Content-Type"] == CONTENT_PAIR and len(body) == 2 * 12 * 20
    back_left, back_right = decode_pair(headers, body)
    assert np.array_equal(back_left, left) and np.array_equal(back_right, right)

    disparity = np.linspace(0.5, 128.0, 12 * 20, dtype=np.float32).reshape(12, 20)
    headers, body = pack_disparity(disparity)
    assert headers["Content-Type"] == CONTENT_DISPARITY and len(body) == 12 * 20 * 2
    back = unpack_disparity(headers, body)
    assert back.dtype == np.float32 and back.shape == (12, 20)
    assert float(np.max(np.abs(back - disparity))) < 0.07  # 0.06 px at the near end, 128 px


def test_a_nan_disparity_stays_a_nan() -> None:
    """The texture gate's "no answer" is a NaN, and a costmap marked where it was lost would be
    marked from nothing: float16 must carry it."""
    disparity = np.full((2, 3), np.nan, dtype=np.float32)
    disparity[0, 0] = 10.0
    back = unpack_disparity(*pack_disparity(disparity))
    assert back[0, 0] == pytest.approx(10.0) and np.isnan(back[0, 1:]).all()


def test_what_the_host_refuses_and_what_the_client_refuses() -> None:
    headers, body = encode_pair(_eye(4, 4), _eye(4, 4))
    with pytest.raises(PairError, match="not two 4x4 grey eyes"):
        decode_pair(headers, body[:-1])
    with pytest.raises(PairError, match="X-Height"):
        decode_pair({"content-type": CONTENT_PAIR}, body)
    with pytest.raises(PairError, match="unknown content type"):
        decode_pair({"content-type": "image/jpeg"}, body)
    with pytest.raises(PairError, match="two grey pictures of one size"):
        encode_pair(_eye(4, 4), _eye(4, 5))
    with pytest.raises(StereoHostError, match="not a disparity"):
        unpack_disparity({"content-type": "text/plain"}, b"no")
    with pytest.raises(StereoHostError, match="float16"):
        unpack_disparity({"content-type": CONTENT_DISPARITY, "x-height": "2", "x-width": "2"}, b"")


# ---------------------------------------------------------------- the settings and the checkpoint
def test_the_checkpoint_path_is_resolved_against_the_repo_and_an_empty_one_is_refused() -> None:
    """config/camera.json carries one path that means the same thing from any working directory,
    and a head with no checkpoint says so instead of loading something else's."""
    assert RaftSettings(weights="models/x.pth").path() == repo_root() / "models/x.pth"
    assert RaftSettings(weights="/tmp/x.pth").path() == __import__("pathlib").Path("/tmp/x.pth")
    with pytest.raises(StereoHostError, match="no RAFT checkpoint configured"):
        RaftSettings().path()
    assert RaftSettings(weights="a/raftstereo-realtime.pth").describe() == (
        "raftstereo-realtime/7 on mps"
    )


def _state(shared: bool, gru_layers: int, downsample: int, width: int = 128) -> dict[str, Any]:
    """The keys and shapes :func:`architecture` reads, as a checkpoint carries them."""
    factor = 1 << downsample
    state: dict[str, Any] = {
        "update_block.mask.2.weight": np.zeros((factor * factor * 9, 1)),
        "cnet.norm1.running_mean": np.zeros(64),
    }
    for level in ("32", "16", "08"):
        state[f"update_block.gru{level}.convz.weight"] = np.zeros((width, 1))
    for i in range(gru_layers):
        state[f"context_zqr_convs.{i}.weight"] = np.zeros((width * 3, 1))
    if not shared:
        state["fnet.conv1.weight"] = np.zeros((64, 3))
    return state


def test_the_architecture_is_read_off_the_checkpoint_not_off_its_filename() -> None:
    """A renamed file would otherwise build the wrong network. The two released recipes: the
    realtime one shares its backbone, has two GRU layers and works at 1/8; the middlebury one
    keeps a separate feature encoder, has three and works at 1/4."""
    realtime = architecture(_state(True, 2, 3), RaftSettings())
    assert realtime.shared_backbone and realtime.n_gru_layers == 2
    assert realtime.n_downsample == 3 and realtime.slow_fast_gru
    assert realtime.hidden_dims == [128, 128, 128] and realtime.context_norm == "batch"
    assert realtime.corr_implementation == "reg", "upstream's CUDA sampler does not build on MPS"

    middlebury = architecture(_state(False, 3, 2), RaftSettings())
    assert not middlebury.shared_backbone and middlebury.n_gru_layers == 3
    assert middlebury.n_downsample == 2 and not middlebury.slow_fast_gru

    broken = _state(True, 2, 3)
    broken["update_block.mask.2.weight"] = np.zeros((100, 1))
    with pytest.raises(StereoHostError, match=r"not \(2\^n\)\^2 \* 9"):
        architecture(broken, RaftSettings())


def test_the_heads_net_block_comes_from_the_camera_config() -> None:
    """The numbers belong to the sensor, not to the module that loads the network: the stereo
    head names a checkpoint, and the mono webcam names none, so the flag cannot be turned on."""
    import os

    stereo = settings_from_config(CONFIG)
    assert stereo.weights.endswith(".pth") and stereo.iters == 7
    assert stereo.device == "mps" and not stereo.gate
    os.environ["PEPIN_CAMERA"] = "overview"
    try:
        mono = settings_from_config(CONFIG)
    finally:
        del os.environ["PEPIN_CAMERA"]
    assert mono.weights == "", "a head with no net block cannot have the flag turned on"


# ---------------------------------------------------------------- the host and the node's client
class FakeRaft:
    """A network whose disparity is the left eye's column index: cheap, and a transposed or
    swapped pair would read wrong rather than merely differently."""

    device = "test"

    def __init__(self) -> None:
        self.pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        self.pairs.append((left.shape, right.shape))
        h, w = left.shape
        return np.repeat(np.arange(w, dtype=np.float32)[None, :], h, axis=0)


@pytest.fixture
def host() -> Iterator[tuple[str, FakeRaft, DepthServer]]:
    """A real depth service on the loopback carrying a fake stereo model and no mono network."""
    net = FakeRaft()
    model = StereoModel(RaftSettings(weights="models/fake.pth"), net=net)

    def no_mono(*_args: Any, **_kwargs: Any) -> np.ndarray:
        raise RuntimeError("this host serves no mono network")

    server = DepthServer(("127.0.0.1", 0), no_mono, name="fake/Net", stereo=model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", net, server
    finally:
        server.shutdown()
        server.server_close()


def test_a_pair_goes_through_the_host_and_comes_back_as_a_disparity(
    host: tuple[str, FakeRaft, DepthServer],
) -> None:
    url, net, server = host
    client = RemoteDisparity(url, timeout_s=2.0)
    disparity = client(_eye(6, 9), _eye(6, 9, shift=3))
    assert disparity.shape == (6, 9) and disparity.dtype == np.float32
    assert disparity[0, 8] == pytest.approx(8.0)
    assert net.pairs == [((6, 9), (6, 9))]
    client(_eye(6, 9), _eye(6, 9))  # the second pair reuses the connection
    assert client.timing["round_trip"].count == 2
    assert client.last_infer_ms >= 0.0 and client.last_model == "fake/7 on mps"
    health = client.health()
    assert health["stereo"]["requests"] == 2 and health["stereo"]["built"]
    assert set(health["stereo"]["ms"]) == {"decode", "infer", "pack", "total"}
    assert "stereo (fake/7 on mps, ready): 2 pairs" in server.report()
    client.close()


def test_a_host_with_no_stereo_model_refuses_the_endpoint_and_says_which_verb_starts_it() -> None:
    """``ros/depth_host.sh start`` and ``stereo`` are the same process; a build that somehow has
    no matcher must say so rather than time out."""
    server = DepthServer(("127.0.0.1", 0), lambda *_a, **_k: np.zeros((2, 2), dtype=np.float32))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = RemoteDisparity(f"http://127.0.0.1:{server.server_address[1]}")
        with pytest.raises(StereoHostError, match=r"ros/depth_host\.sh stereo"):
            client(_eye(4, 4), _eye(4, 4))
    finally:
        server.shutdown()
        server.server_close()


def test_a_refused_pair_is_counted_against_the_stereo_model(
    host: tuple[str, FakeRaft, DepthServer],
) -> None:
    """/health must blame the right model: a bad pair is the matcher's refusal, not the mono
    network's."""
    import http.client

    _url, _net, server = host
    headers, body = encode_pair(_eye(4, 4), _eye(4, 4))
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=2.0)
    conn.request("POST", "/disparity", body=body[:-1], headers=headers)
    response = conn.getresponse()
    assert response.status == 400 and b"not two 4x4 grey eyes" in response.read()
    conn.close()
    assert server.stereo is not None and server.stereo.errors == 1 and server.errors == 0


# ---------------------------------------------------------------- the matcher the node holds
def test_the_matcher_sends_grey_eyes_and_answers_what_the_host_says(
    host: tuple[str, FakeRaft, DepthServer],
) -> None:
    """The node's engine: colour in (the left eye is colour off the wire and the right is not),
    one grey pair on the wire — a network fed one of each is being asked a question nobody
    trained — and the host's disparity out, with the fallback untouched."""
    url, net, _server = host
    sgbm = StereoMatcher()
    matcher = RaftMatcher(url, sgbm)
    colour = np.repeat(_eye(6, 9)[:, :, None], 3, axis=2)
    disparity = matcher(colour, _eye(6, 9))
    assert disparity.shape == (6, 9) and disparity[3, 4] == pytest.approx(4.0)
    assert net.pairs == [((6, 9), (6, 9))], "grey, and the right eye is the right eye"
    assert matcher.pairs == 1 and matcher.fell_back == 0
    assert matcher.search_px == 9, "the network refines over the whole row"
    assert matcher.timing["total"].summary().median_ms > 0.0
    words = matcher.describe()
    assert "raft fake/7 on mps" in words and "ms a pair" in words and "host " in words
    with pytest.raises(StereoUnavailableError, match="not one rig"):
        matcher(_eye(6, 9), _eye(6, 8))


def test_a_host_that_does_not_answer_falls_to_sgbm_counted_and_is_given_up_on() -> None:
    """The shape of depth_backend: auto. A pair the host loses is answered by the very SGBM the
    flag would otherwise have used; after three in a row the host is left alone rather than
    costing a timeout per pair, and it is probed again after retry_s."""
    clock = [0.0]
    sgbm = StereoMatcher(MatcherSettings(num_disparities=16))
    matcher = RaftMatcher(
        "http://127.0.0.1:1",  # nothing listens there
        sgbm,
        timeout_s=0.05,
        failures=3,
        retry_s=30.0,
        clock=lambda: clock[0],
    )
    left, right = _eye(64, 64), _eye(64, 64, shift=2)
    for _ in range(3):
        assert matcher(left, right).shape == (64, 64)
    assert matcher.pairs == 0 and matcher.fell_back == 3 and matcher.last_error
    assert not matcher.on_host, "three failures in a row: the host is given up on"
    matcher(left, right)
    assert matcher.fell_back == 4, "and no round trip is paid while it is"
    assert matcher.search_px == 64, "the width the fallback saw is still the width"
    clock[0] = 31.0
    assert matcher.on_host, "probed again after retry_s"
    assert "4 fell to sgbm" in matcher.describe() and "down 31 s" in matcher.describe()


def test_the_flag_names_the_engine_and_an_unknown_name_is_refused() -> None:
    """A costmap marked from a matcher nobody chose is the bug build_matcher prevents."""
    assert MATCHERS == ("sgbm", "raft")
    sgbm = build_matcher("sgbm", MatcherSettings(num_disparities=16))
    assert isinstance(sgbm, StereoMatcher) and sgbm.search_px == 16
    raft = build_matcher("raft", url="http://127.0.0.1:1")
    assert isinstance(raft, RaftMatcher) and isinstance(raft.fallback, StereoMatcher)
    with pytest.raises(ValueError, match="must be one of"):
        build_matcher("elas")
    with pytest.raises(ValueError, match="needs the stereo host's URL"):
        build_matcher("raft")


def test_the_source_reads_its_near_end_from_whichever_engine_is_in_front(
    host: tuple[str, FakeRaft, DepthServer],
) -> None:
    """``near`` is a property of the SEARCH, and the two engines search differently: SGBM over a
    fixed window, the network over the whole row. The report line must not keep SGBM's number
    while RAFT is answering."""
    url, _net, _server = host
    rig = Baseline(fx=373.0, baseline_m=0.063)
    source = StereoDepth(rig, StereoMatcher(MatcherSettings(num_disparities=128)))
    assert source.near == pytest.approx(373.0 * 0.063 / 128, rel=1e-6)
    source.matcher = RaftMatcher(url, StereoMatcher())
    assert source.near == 0.0, "nothing searched yet"
    source(_eye(6, 9), _eye(6, 9))
    assert source.near == pytest.approx(373.0 * 0.063 / 9, rel=1e-6)
    assert "raft fake/7 on mps" in source.describe() and "% valid" in source.describe()
