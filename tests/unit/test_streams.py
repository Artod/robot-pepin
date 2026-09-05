"""The shared JSON-lines client: framing across chunks, freshness, reconnects, sending when down."""

import time
from typing import Any

from pepin.streams import JsonLinesClient


class ScriptedSocket:
    """Delivers the given chunks one recv at a time, then closes (b"")."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self.sent: list[bytes] = []
        self.closed = False

    def settimeout(self, value: float) -> None:
        pass

    def recv(self, size: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        time.sleep(0.005)
        return b""

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def shutdown(self, how: int) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class Collector(JsonLinesClient):
    def __init__(self, connector) -> None:  # type: ignore[no-untyped-def]
        super().__init__("unused", 1, name="test", connector=connector, retry_s=0.01)
        self.messages: list[dict[str, Any]] = []

    def _ingest(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


def test_messages_are_reassembled_across_chunks_and_the_client_reconnects() -> None:
    sockets: list[ScriptedSocket] = []

    def connector(address):  # type: ignore[no-untyped-def]
        sockets.append(ScriptedSocket([b'{"a": 1}\n{"b"', b": 2}\n", b'{"c": 3}\n']))
        return sockets[-1]

    client = Collector(connector).start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and len(sockets) < 2:
        time.sleep(0.01)
    client.close()
    assert client.messages[:3] == [{"a": 1}, {"b": 2}, {"c": 3}]
    assert len(sockets) >= 2  # the closed stream was reopened
    assert client.age_s() < 5.0


def test_send_while_down_is_dropped_not_raised() -> None:
    client = Collector(lambda address: ScriptedSocket([]))
    client.send({"cmd": "twist"})  # never started: no socket
    assert client.age_s() == float("inf")
    assert not client.connected
