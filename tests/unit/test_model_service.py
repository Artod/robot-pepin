"""The model-service skeleton: the codecs both ends share, a real server on the loopback with
fake models (the per-device lock, the answer cache, what it refuses and how it says so), and the
client that never raises — its timeout, its back-off while the service is down, and the switch
between the service and an in-process implementation."""

from __future__ import annotations

import threading
import time
from collections.abc import Hashable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from pepin.model_service import (
    AUTO,
    CONTENT_JPEG,
    CONTENT_NPY,
    CONTENT_RAW8,
    LOCAL,
    SERVICE,
    BadRequestError,
    EveryMinute,
    ModelServer,
    RemoteModel,
    Request,
    Response,
    ServiceOrLocal,
    decode_arrays,
    decode_image,
    encode_arrays,
    encode_image,
    split_url,
)


def _picture(h: int = 36, w: int = 64) -> np.ndarray:
    rows = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    cols = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    return np.stack([rows + 0 * cols, 0 * rows + cols, (rows + cols) / 2], axis=2).astype(np.uint8)


# ---------------------------------------------------------------- codecs
def test_raw_pictures_come_back_bit_for_bit_grey_and_colour() -> None:
    rgb = _picture()
    headers, body = encode_image(rgb, "raw")
    assert headers["Content-Type"] == CONTENT_RAW8 and headers["X-Channels"] == "3"
    assert np.array_equal(decode_image(headers, body), rgb)
    grey = rgb[:, :, 0].copy()
    headers, body = encode_image(grey, "raw")
    assert headers["X-Channels"] == "1" and len(body) == 36 * 64
    back = decode_image(headers, body)
    assert back.shape == (36, 64) and np.array_equal(back, grey)


def test_a_jpeg_picture_comes_back_close_and_in_rgb_order() -> None:
    rgb = _picture(72, 128)
    rgb[:, :, 2] = 250  # a strong blue: a swapped channel order would read as red
    headers, body = encode_image(rgb, "jpeg", quality=95)
    assert headers["Content-Type"] == CONTENT_JPEG and len(body) < rgb.nbytes / 3
    back = decode_image(headers, body)
    assert back.shape == rgb.shape
    assert np.mean(np.abs(back.astype(int) - rgb.astype(int))) < 3.0
    grey_headers, grey_body = encode_image(rgb[:, :, 1], "jpeg")
    assert decode_image(grey_headers, grey_body).shape == (72, 128)


def test_named_arrays_keep_their_dtype_shape_and_order_even_empty() -> None:
    arrays = {
        "keypoints": np.arange(12, dtype=np.float32).reshape(4, 3),
        "descriptors": np.zeros((0, 64), dtype=np.float32),
        "matches": np.array([[0, 1], [2, 3]], dtype=np.int32),
    }
    headers, body = encode_arrays(arrays)
    assert headers["Content-Type"] == CONTENT_NPY
    assert headers["X-Arrays"] == "keypoints,descriptors,matches"
    back = decode_arrays(headers, body)
    assert list(back) == list(arrays)
    for name, array in arrays.items():
        assert back[name].dtype == array.dtype and back[name].shape == array.shape
        assert np.array_equal(back[name], array)


def test_what_the_codecs_refuse() -> None:
    headers, body = encode_image(_picture(), "raw")
    with pytest.raises(BadRequestError):
        decode_image(headers, body[:-1])
    with pytest.raises(BadRequestError):
        decode_image({"Content-Type": "text/plain"}, b"")
    with pytest.raises(BadRequestError):
        decode_image({"Content-Type": CONTENT_JPEG}, b"not a jpeg")
    headers, body = encode_arrays({"a": np.ones(3)})
    with pytest.raises(BadRequestError):
        decode_arrays(headers, body + b"x")  # bytes nobody named
    with pytest.raises(BadRequestError):
        decode_arrays({**headers, "X-Arrays": "a,b"}, body)  # a name with no bytes
    with pytest.raises(ValueError):
        encode_arrays({"a,b": np.ones(1)})
    with pytest.raises(ValueError):
        encode_image(np.zeros((2, 2, 2), dtype=np.uint8))
    assert split_url("http://host.docker.internal:8791/") == ("host.docker.internal", 8791)
    with pytest.raises(ValueError):
        split_url("https://x")


# ---------------------------------------------------------------- a real server, fake models
@dataclass
class Doubler:
    """A fake model: arrays in, twice them out; ``sleep_s`` inside the lock; counts overlaps."""

    name: str = "double"
    device: str = "mps"
    cache_size: int = 0
    sleep_s: float = 0.0
    fail: bool = False

    def __post_init__(self) -> None:
        self.tag = f"{self.name}@0123abcd"
        self.inside = 0
        self.overlap = 0
        self.calls = 0
        self._count = threading.Lock()

    def warm(self) -> None:
        pass

    def decode(self, request: Request) -> np.ndarray:
        return decode_arrays(request.headers, request.body)["x"]

    def cache_key(self, inputs: np.ndarray) -> Hashable | None:
        return inputs.tobytes() if self.cache_size else None

    def infer(self, inputs: np.ndarray) -> np.ndarray:
        with self._count:
            self.inside += 1
            self.calls += 1
            self.overlap = max(self.overlap, self.inside)
        try:
            time.sleep(self.sleep_s)
            if self.fail:
                raise RuntimeError("the network fell over")
            return inputs * 2
        finally:
            with self._count:
                self.inside -= 1

    def encode(self, outputs: np.ndarray) -> Response:
        headers, body = encode_arrays({"y": outputs})
        return Response(headers["Content-Type"], body, {"X-Arrays": headers["X-Arrays"]})


@pytest.fixture
def served() -> Iterator[Any]:
    """A factory: serve these models on a free loopback port, closed after the test."""
    servers: list[ModelServer] = []

    def serve(*models: Any) -> tuple[ModelServer, str]:
        server = ModelServer(("127.0.0.1", 0), list(models), name="test models")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server, f"http://127.0.0.1:{server.server_address[1]}"

    yield serve
    for server in servers:
        server.shutdown()
        server.server_close()


def _ask(client: RemoteModel, x: np.ndarray) -> np.ndarray | None:
    headers, body = encode_arrays({"x": x})
    answer = client.call(headers, body)
    return None if answer is None else decode_arrays(*answer)["y"]


def test_a_request_goes_through_and_health_names_every_model(served: Any) -> None:
    server, url = served(Doubler(), Doubler(name="other", device="cpu"))
    client = RemoteModel(url, "double")
    assert np.array_equal(_ask(client, np.arange(4.0)), np.arange(4.0) * 2)
    assert client.last_model == "double@0123abcd" and client.ok == 1
    health = client.health()
    assert health is not None and health["service"] == "test models"
    assert set(health["models"]) == {"double", "other"}
    double = health["models"]["double"]
    assert double["tag"] == "double@0123abcd" and double["device"] == "mps"
    assert double["requests"] == 1 and double["errors"] == 0
    assert set(double["ms"]) == {"decode", "wait", "infer", "encode", "total"}
    assert "double (double@0123abcd on mps)" in server.report()


def test_the_gpu_lock_serialises_two_models_on_one_device(served: Any) -> None:
    """Measured by wall time: four 40 ms inferences on one device cannot finish in under 160 ms,
    while the same four split over two devices overlap and finish sooner."""
    one = [Doubler(name=n, device="mps", sleep_s=0.04) for n in ("a", "b")]
    two = [
        Doubler(name="c", device="mps", sleep_s=0.04),
        Doubler(name="d", device="cpu", sleep_s=0.04),
    ]

    def run(models: list[Doubler]) -> float:
        _server, url = served(*models)
        threads = [
            threading.Thread(target=_ask, args=(RemoteModel(url, m.name), np.ones(1)))
            for m in models * 2
        ]
        t0 = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return time.perf_counter() - t0

    serial = run(one)
    assert serial >= 0.16, "the lock's own guarantee: never two inferences on one device at once"
    assert run(two) < serial - 0.03, "two devices overlap (loose: a loaded machine slows both)"


def test_an_answer_named_by_a_cache_key_is_served_again_without_the_network(served: Any) -> None:
    model = Doubler(cache_size=2)
    server, url = served(model)
    client = RemoteModel(url, "double")
    for x in (np.ones(3), np.ones(3), np.zeros(3), np.full(3, 5.0), np.ones(3)):
        _ask(client, x)
    # ones, ones (hit), zeros, fives: the cache holds zeros and fives, so the last ones misses
    assert model.calls == 4
    assert server.health()["models"]["double"]["cache_hits"] == 1


def test_a_bad_request_is_400_a_failed_network_500_and_both_are_counted(served: Any) -> None:
    server, url = served(Doubler(), Doubler(name="broken", fail=True))
    client = RemoteModel(url, "double")
    assert client.call({"Content-Type": "text/plain"}, b"junk") is None
    assert client.last_error.startswith("400") and not client.down  # the service is up
    broken = RemoteModel(url, "broken")
    assert _ask(broken, np.ones(2)) is None and broken.last_error.startswith("500")
    assert _ask(client, np.ones(2)) is not None  # the connection survived both
    assert RemoteModel(url, "nowhere").call({}, b"") is None
    health = server.health()["models"]
    assert health["double"]["errors"] == 1 and health["broken"]["errors"] == 1


class Clock:
    """A settable monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_slow_service_costs_its_timeouts_and_three_in_a_row_back_off(served: Any) -> None:
    model = Doubler(sleep_s=0.3)
    _server, url = served(model)
    clock = Clock()
    client = RemoteModel(url, "double", timeout_s=0.05, retry_s=10.0, clock=clock, timeouts=3)
    for n in range(1, 4):
        t0 = time.perf_counter()
        assert _ask(client, np.ones(1)) is None
        assert time.perf_counter() - t0 < 0.25
        assert client.down == (n == 3), "one slow answer is not a dead service; three are"
    t0 = time.perf_counter()
    assert _ask(client, np.ones(1)) is None  # skipped: no timeout paid
    assert time.perf_counter() - t0 < 0.01 and client.skipped == 1
    model.sleep_s = 0.0
    time.sleep(0.95)  # the three late inferences finish on the server
    clock.now += 10.0
    assert np.array_equal(_ask(client, np.ones(1)), np.full(1, 2.0))  # tried again, and it works
    assert not client.down and "ok 1, failed 3, skipped 1" in client.status()


def test_a_timeout_between_answers_does_not_count_towards_the_back_off(served: Any) -> None:
    model = Doubler()
    _server, url = served(model)
    client = RemoteModel(url, "double", timeout_s=0.05, timeouts=2)
    for _ in range(3):
        model.sleep_s = 0.2
        assert _ask(client, np.ones(1)) is None
        model.sleep_s = 0.0
        time.sleep(0.25)
        assert _ask(client, np.ones(1)) is not None, "an answer resets the count"
    assert not client.down


def test_no_service_at_all_is_none_quickly_and_backs_off() -> None:
    clock = Clock()
    client = RemoteModel("http://127.0.0.1:9", "double", timeout_s=0.2, clock=clock)
    assert client.call({}, b"") is None and client.down and client.failed == 1
    assert client.call({}, b"") is None and client.skipped == 1
    assert client.health(timeout_s=0.2) is None


# ---------------------------------------------------------------- the switch
def test_the_switch_between_service_and_local() -> None:
    answers: dict[str, int | None] = {"remote": 1}

    def local(x: int) -> int:
        if x < 0:
            raise RuntimeError("no torch here")
        return 2

    switch = ServiceOrLocal[int, int](lambda _x: answers["remote"], local, lambda _x: 0)
    assert switch(1, AUTO) == 1 and switch(1, SERVICE) == 1 and switch(1, LOCAL) == 2
    answers["remote"] = None
    assert switch(1, AUTO) == 2  # the fallback
    assert switch(1, SERVICE) == 0  # the service or nothing
    assert switch(-1, AUTO) == 0  # a local that raises is an empty answer, not an exception
    assert switch(1, "garbage") == 2  # an unknown mode reads as auto
    assert switch.line().startswith("service 2, local 1, fallback 2, failed 2")
    assert "no torch here" in switch.line()


def test_a_periodic_line_is_due_once_a_period() -> None:
    clock = Clock()
    minute = EveryMinute(60.0, clock)
    assert not minute.due()
    clock.now += 61
    assert minute.due() and not minute.due()
    clock.now += 59
    assert not minute.due()
    clock.now += 2
    assert minute.due()


# ---------------------------------------------------------------- a service that went away and back
class OneAnswerPerConnection:
    """A raw HTTP/1.1 server that answers ONE request per connection and then closes it — what a
    kept-alive client sees of a service process that was restarted between two of its calls."""

    def __init__(self) -> None:
        import socket

        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        self.connections = 0
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1
            with conn:
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                head, _, body = data.partition(b"\r\n\r\n")
                length = next(
                    (
                        int(line.split(b":")[1])
                        for line in head.split(b"\r\n")
                        if line.lower().startswith(b"content-length")
                    ),
                    0,
                )
                while len(body) < length:
                    body += conn.recv(65536)
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )

    def close(self) -> None:
        self.sock.close()


def test_a_dropped_kept_alive_connection_is_tried_once_more_fresh_and_nothing_backs_off() -> None:
    """After a launchd restart the client's pooled connection is dead: its next call used to fail
    at once and back the endpoint off for 10 s (null descriptors, local fallbacks) although the
    service was already back. One retry on a fresh connection, and the call succeeds."""
    server = OneAnswerPerConnection()
    try:
        client = RemoteModel(server.url, "echo", timeout_s=1.0)
        for n in range(1, 4):
            answer = client.call({}, f"call {n}".encode())
            assert answer is not None and answer[1] == f"call {n}".encode(), client.status()
        assert client.failed == 0 and not client.down
        assert client.retried == 2 and server.connections == 3
        assert "2 retried on a fresh connection" in client.status()
    finally:
        server.close()


def test_a_client_that_left_is_counted_not_printed(
    served: Any, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """A client that timed out leaves the server writing into a closed socket: socketserver
    printed a BrokenPipe traceback into the job's output for each one. Now one count in /health;
    anything else is still logged whole."""
    server, _url = served(Doubler())
    for gone in (BrokenPipeError(), ConnectionResetError()):
        try:
            raise gone
        except OSError:
            server.handle_error(None, ("127.0.0.1", 1))
    assert server.health()["clients_gone"] == 2
    assert capsys.readouterr().err == "", "no traceback on the job's output"
    try:
        raise RuntimeError("a handler bug")
    except RuntimeError:
        server.handle_error(None, ("127.0.0.1", 1))
    assert "a handler bug" in caplog.text and server.health()["clients_gone"] == 2


def test_a_taken_port_is_named_before_any_model_loads() -> None:
    import socket

    from pepin.model_service import port_taken

    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        assert f"127.0.0.1:{port} is taken" in str(port_taken("127.0.0.1", port))
    finally:
        holder.close()
    assert port_taken("127.0.0.1", port) is None
