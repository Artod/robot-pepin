"""The depth service: the codec both ends share, the round trip through a real server on the
loopback with a fake network, the client's failures, and the switch that falls back to the CPU
model and tries the service again."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest

from pepin.depth_service import (
    CONTENT_DEPTH,
    CONTENT_RGB,
    BadRequestError,
    DepthServer,
    DepthServiceError,
    Fallback,
    LazyDepth,
    RemoteDepth,
    decode_frame,
    encode_frame,
    model_id,
    pack_depth,
    requested_size,
    unpack_depth,
)


def _frame(h: int = 36, w: int = 64) -> np.ndarray:
    rows = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    cols = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    return np.stack([rows + 0 * cols, 0 * rows + cols, (rows + cols) / 2], axis=2).astype(np.uint8)


# ---------------------------------------------------------------- codec
def test_raw_frames_and_float16_depths_survive_the_round_trip() -> None:
    rgb = _frame()
    headers, body = encode_frame(rgb, "raw")
    assert headers["Content-Type"] == CONTENT_RGB and len(body) == 36 * 64 * 3
    assert np.array_equal(decode_frame(headers, body), rgb)
    depth = np.linspace(0.3, 9.0, 36 * 64, dtype=np.float32).reshape(36, 64)
    headers, body = pack_depth(depth)
    assert headers["Content-Type"] == CONTENT_DEPTH and len(body) == 36 * 64 * 2
    back = unpack_depth(headers, body)
    assert back.dtype == np.float32 and back.shape == (36, 64)
    assert np.max(np.abs(back - depth) / depth) < 1e-3  # float16: a millimetre at two metres


def test_a_jpeg_frame_comes_back_as_the_same_picture() -> None:
    rgb = _frame(72, 128)
    headers, body = encode_frame(rgb, "jpeg", quality=90)
    assert headers["Content-Type"] == "image/jpeg" and len(body) < rgb.nbytes / 4
    back = decode_frame(headers, body)
    assert back.shape == rgb.shape and back.dtype == np.uint8
    assert np.mean(np.abs(back.astype(int) - rgb.astype(int))) < 3.0


def test_what_the_server_refuses_and_what_the_client_refuses() -> None:
    with pytest.raises(BadRequestError):
        decode_frame({"content-type": CONTENT_RGB, "x-height": "2", "x-width": "2"}, b"\0" * 5)
    with pytest.raises(BadRequestError):
        decode_frame({"content-type": CONTENT_RGB}, b"")
    with pytest.raises(BadRequestError):
        decode_frame({"content-type": "text/plain"}, b"hello")
    with pytest.raises(DepthServiceError):
        unpack_depth({"content-type": "text/plain"}, b"no")
    with pytest.raises(DepthServiceError):
        unpack_depth({"content-type": CONTENT_DEPTH, "x-height": "2", "x-width": "2"}, b"\0" * 7)
    with pytest.raises(ValueError):
        encode_frame(_frame(), "png")


def test_the_size_asked_for_and_the_model_names() -> None:
    assert requested_size({"content-type": CONTENT_RGB}, (36, 64)) == (36, 64)
    assert requested_size({"X-Depth-Size": "native"}, (36, 64)) is None
    assert model_id("Large").endswith("Metric-Indoor-Large-hf")
    assert model_id("org/some-model") == "org/some-model"


# ---------------------------------------------------------------- the server and the client
class FakeNet:
    """A network whose depth is the frame's row index in metres, at the size asked for (or at
    a fixed 9x16 'native' size), after an optional delay."""

    name = "fake/Net"
    device = "test"

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.calls: list[tuple[int, int] | None] = []

    def __call__(self, rgb: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
        self.calls.append(size)
        if self.delay_s:
            time.sleep(self.delay_s)
        h, w = size if size is not None else (9, 16)
        return np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1) + 0.5


@pytest.fixture
def service() -> Iterator[tuple[str, FakeNet, DepthServer]]:
    net = FakeNet()
    server = DepthServer(("127.0.0.1", 0), net)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", net, server
    finally:
        server.shutdown()
        server.server_close()


def test_a_frame_goes_through_the_service_and_comes_back_at_its_own_size(
    service: tuple[str, FakeNet, DepthServer],
) -> None:
    url, net, server = service
    rgb = _frame(36, 64)
    for encoding in ("raw", "jpeg"):
        client = RemoteDepth(url, timeout_s=2.0, encoding=encoding)
        depth = client(rgb)
        assert depth.shape == (36, 64) and depth.dtype == np.float32
        assert depth[0, 0] == pytest.approx(0.5) and depth[35, 63] == pytest.approx(35.5)
        assert client.last_infer_ms >= 0.0
        client(rgb)  # the second frame reuses the connection
        assert client.timing["round_trip"].count == 2
        client.close()
    assert net.calls == [(36, 64)] * 4
    health = RemoteDepth(url).health()
    assert health["model"] == "fake/Net" and health["requests"] == 4 and health["errors"] == 0
    assert set(health["ms"]) == {"decode", "infer", "pack", "total"}
    assert "fake/Net" in server.report() or "Net on test" in server.report()


def test_the_native_size_is_resized_by_the_client(
    service: tuple[str, FakeNet, DepthServer],
) -> None:
    url, net, _ = service
    depth = RemoteDepth(url, native=True, encoding="raw")(_frame(18, 32))
    assert net.calls == [None] and depth.shape == (18, 32)
    # bilinear with half-pixel centres: the first row of a 9-row ramp upsampled twice starts at
    # the ramp's start, the last ends at its end
    assert depth[0, 0] == pytest.approx(0.5) and depth[17, 0] == pytest.approx(8.5)
    assert depth[1, 0] == pytest.approx(0.75) and depth[2, 0] == pytest.approx(1.25)


def test_a_refused_request_is_an_error_the_client_recovers_from(
    service: tuple[str, FakeNet, DepthServer],
) -> None:
    url, _, server = service
    client = RemoteDepth(url, encoding="raw")
    frame = _frame(4, 4)
    headers, body = encode_frame(frame, "raw")
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=2.0)
    conn.request("POST", "/depth", body=body[:-1], headers=headers)  # a byte short
    response = conn.getresponse()
    assert response.status == 400 and b"not 4x4x3" in response.read()
    conn.request("GET", "/nowhere")  # the same connection: a refusal does not close it
    assert conn.getresponse().status == 404
    conn.close()
    assert server.errors == 1
    assert client(frame).shape == (4, 4)  # the client's own connection is unaffected


def test_a_network_that_raises_is_a_500_and_the_connection_survives() -> None:
    """A frame the network itself chokes on (out of memory, a Metal fault) must come back as an
    answer the client can fall back on, not as a dropped connection with a traceback."""

    def angry(rgb: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
        raise RuntimeError("MPS backend out of memory")

    server = DepthServer(("127.0.0.1", 0), angry, name="angry/Net")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = RemoteDepth(f"http://127.0.0.1:{server.server_address[1]}", encoding="raw")
        with pytest.raises(DepthServiceError, match="out of memory"):
            client(_frame(4, 4))
        assert server.errors == 1
        assert client.health()["errors"] == 1  # the same connection still answers
    finally:
        server.shutdown()
        server.server_close()


def test_no_service_and_a_slow_service_are_errors_within_the_timeout() -> None:
    with pytest.raises(DepthServiceError):
        RemoteDepth("http://127.0.0.1:9", timeout_s=0.5, encoding="raw")(_frame(4, 4))
    with pytest.raises(ValueError):
        RemoteDepth("https://elsewhere")
    net = FakeNet(delay_s=0.4)
    server = DepthServer(("127.0.0.1", 0), net)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = RemoteDepth(
            f"http://127.0.0.1:{server.server_address[1]}", timeout_s=0.05, encoding="raw"
        )
        t0 = time.perf_counter()
        with pytest.raises(DepthServiceError, match="timed out"):
            client(_frame(4, 4))
        assert time.perf_counter() - t0 < 0.3
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------- the switch
class FakeBackend:
    def __init__(self, name: str, fail: bool = False) -> None:
        self.name, self.fail, self.calls = name, fail, 0

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.fail:
            raise DepthServiceError(f"{self.name} is down")
        return np.full(rgb.shape[:2], 1.0 if self.name == "remote" else 2.0, dtype=np.float32)


def test_auto_falls_back_per_frame_then_gives_up_and_retries_later() -> None:
    now = [100.0]
    remote, local = FakeBackend("remote", fail=True), FakeBackend("local")
    switch = Fallback(remote, local, mode="auto", failures=3, retry_s=30.0, clock=lambda: now[0])
    frame = _frame(4, 4)
    for _ in range(3):  # every frame is answered — by the local model — while the remote fails
        assert switch(frame)[0, 0] == 2.0
    assert remote.calls == 3 and local.calls == 3 and not switch.on_remote
    assert switch.status.startswith(
        "auto: local since 0 s (remote is down; remote retried in 30 s)"
    )
    now[0] += 10.0
    switch(frame)
    assert remote.calls == 3  # no timeout paid within the retry period
    now[0] += 25.0
    switch(frame)
    assert remote.calls == 4 and local.calls == 5  # one probe, still down, the frame from local
    now[0] += 30.0
    remote.fail = False
    assert switch(frame)[0, 0] == 1.0  # the probe succeeded: back on the GPU at once
    assert switch.on_remote and switch.status == "auto: remote (5 frames fell to local)"
    assert switch.remote_frames == 1 and switch.failures == 4


def test_remote_raises_and_local_never_asks() -> None:
    remote, local = FakeBackend("remote", fail=True), FakeBackend("local")
    frame = _frame(4, 4)
    strict = Fallback(remote, local, mode="remote")
    with pytest.raises(DepthServiceError):
        strict(frame)
    assert local.calls == 0 and strict.status == "remote"
    only_local = Fallback(remote, local, mode="local")
    assert only_local(frame)[0, 0] == 2.0 and remote.calls == 1
    only_local.mode = "auto"  # the switch is live: the remote is trusted afresh
    assert only_local.on_remote
    with pytest.raises(ValueError):
        only_local.mode = "gpu"


def test_a_single_hiccup_does_not_switch_the_backend() -> None:
    remote, local = FakeBackend("remote", fail=True), FakeBackend("local")
    switch = Fallback(remote, local, failures=3, clock=lambda: 0.0)
    frame = _frame(4, 4)
    switch(frame)
    remote.fail = False
    assert switch(frame)[0, 0] == 1.0 and switch.on_remote
    assert switch.local_frames == 1 and switch.remote_frames == 1


def test_the_local_model_is_built_on_its_first_frame_and_once() -> None:
    """The CPU model costs a gigabyte and seconds: the node hands the switch a LazyDepth, which
    builds it on the first frame that actually goes local — never while the service answers."""
    built: list[FakeBackend] = []

    def build() -> FakeBackend:
        built.append(FakeBackend("local"))
        return built[-1]

    lazy = LazyDepth(build)
    assert not lazy.built and built == []
    frame = _frame(4, 4)
    assert lazy(frame)[0, 0] == 2.0 and lazy.built
    lazy(frame)
    assert len(built) == 1 and built[0].calls == 2
    remote = FakeBackend("remote")
    untouched = LazyDepth(build)
    switch = Fallback(remote, untouched, mode="auto", clock=lambda: 0.0)
    for _ in range(5):
        assert switch(frame)[0, 0] == 1.0
    assert not untouched.built and len(built) == 1, "the service answered: no model loaded"
    Fallback(remote, untouched, mode="remote")(frame)
    assert not untouched.built
    remote.fail = True
    assert switch(frame)[0, 0] == 2.0 and untouched.built, "the first fallback builds it"
