"""The Foxglove probe's pure halves: the websocket handshake, the frames, the channel book.

No socket is opened here — every function under test takes bytes and returns data, which is the
point of splitting them out of ``collect``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]


def _probe() -> ModuleType:
    """Import ros/tools/foxglove_probe.py by path (ros/tools is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "foxglove_probe", REPO / "ros/tools/foxglove_probe.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses looks the module up while the class is built
    spec.loader.exec_module(module)
    return module


def _frame(payload: bytes, opcode: int = 0x1) -> bytes:
    """A server-to-client websocket frame (unmasked, FIN), as the bridge writes them."""
    head = bytes([0x80 | opcode])
    if len(payload) < 126:
        return head + bytes([len(payload)]) + payload
    if len(payload) < 1 << 16:
        return head + b"\x7e" + len(payload).to_bytes(2, "big") + payload
    return head + b"\x7f" + len(payload).to_bytes(8, "big") + payload


def test_the_request_asks_for_both_subprotocols() -> None:
    """foxglove_bridge 3.4.1 answers HTTP 400 to a client that asks only for the old
    foxglove.websocket.v1: it speaks foxglove.sdk.v1. The request names both, as Studio does."""
    probe = _probe()
    request = probe.handshake_request("localhost", 8765, "/", "dGhlIHNhbXBsZSBub25jZQ==").decode()
    assert request.startswith("GET / HTTP/1.1\r\n")
    assert "Sec-WebSocket-Version: 13\r\n" in request
    offered = next(
        line.split(":", 1)[1].strip()
        for line in request.split("\r\n")
        if line.lower().startswith("sec-websocket-protocol")
    )
    assert "foxglove.sdk.v1" in offered and "foxglove.websocket.v1" in offered
    assert request.endswith("\r\n\r\n")


def test_the_accept_key_is_the_one_rfc_6455_prints() -> None:
    probe = _probe()
    assert probe.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_a_half_read_upgrade_is_not_an_answer_yet() -> None:
    """The response arrives in pieces; only a complete header block is parsed."""
    probe = _probe()
    assert probe.parse_handshake(b"HTTP/1.1 101 Switching Protocols\r\nupgrade: websoc") is None
    done = probe.parse_handshake(
        b"HTTP/1.1 101 Switching Protocols\r\n"
        b"Upgrade: websocket\r\n"
        b"Sec-WebSocket-Protocol: foxglove.sdk.v1\r\n\r\n\x81\x02hi"
    )
    assert done is not None
    assert done.status == 101
    assert done.subprotocol == "foxglove.sdk.v1"
    assert done.rest == b"\x81\x02hi"


def test_a_refusal_is_read_as_a_refusal() -> None:
    probe = _probe()
    answer = probe.parse_handshake(b"HTTP/1.1 400 Bad Request\r\n\r\nMissing expected header")
    assert answer is not None and answer.status == 400 and answer.subprotocol == ""


def test_frames_are_split_and_a_torn_tail_is_kept() -> None:
    probe = _probe()
    stream = _frame(b"one") + _frame(b"x" * 300) + _frame(b"", 0x8) + b"\x81\x05par"
    frames, rest = probe.decode_frames(stream)
    assert [(op, len(payload)) for op, payload in frames] == [(1, 3), (1, 300), (8, 0)]
    assert frames[0][1] == b"one"
    assert rest == b"\x81\x05par"


def test_the_channel_book_follows_advertise_and_unadvertise() -> None:
    """A channel withdrawn when its publisher goes is gone from the book — that withdrawal is
    exactly what empties a panel while the socket stays up."""
    probe = _probe()
    payloads = [
        json.dumps(
            {
                "op": "advertise",
                "channels": [
                    {"id": 1, "topic": "/scan", "schemaName": "sensor_msgs/msg/LaserScan"},
                    {"id": 2, "topic": "/amcl_path", "schemaName": "nav_msgs/msg/Path"},
                ],
            }
        ).encode(),
        b"not json at all",
        json.dumps({"op": "unadvertise", "channelIds": [2]}).encode(),
    ]
    channels = probe.read_channels(payloads)
    assert channels == {1: ("/scan", "sensor_msgs/msg/LaserScan")}


def test_server_info_is_found_among_the_noise() -> None:
    probe = _probe()
    payloads = [
        b"\x00\x01binary",
        json.dumps({"op": "serverInfo", "name": "b", "capabilities": ["assets"]}).encode(),
    ]
    info = probe.server_info(payloads)
    assert info is not None and info["name"] == "b"
    assert probe.server_info([b"{}"]) is None


def test_an_image_topic_counts_raw_or_compressed() -> None:
    probe = _probe()
    required = ["/scan", "/camera/image|/camera/image/compressed", "/plan"]
    assert probe.missing_topics(["/scan", "/camera/image/compressed"], required) == ["/plan"]
    assert probe.missing_topics(["/scan", "/camera/image", "/plan"], required) == []


def test_a_dead_bridge_fails_every_item_instead_of_raising() -> None:
    """Port 1 refuses; the probe must print FAIL lines and exit 1, never a traceback."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "ros/tools/foxglove_probe.py"),
            "--port",
            "1",
            "--seconds",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.startswith("FAIL")
    assert "Traceback" not in result.stderr
    assert result.stdout.count("FAIL") == 3 + len(_probe().DEFAULT_REQUIRED)


def test_the_scripts_parse_and_say_what_they_do() -> None:
    """foxglove.sh, and the two scripts that now call it, are valid bash; the wiring is there."""
    for script in ("ros/foxglove.sh", "ros/restart.sh", "ros/laptop.sh"):
        parsed = subprocess.run(
            ["bash", "-n", str(REPO / script)], capture_output=True, text=True, timeout=20
        )
        assert parsed.returncode == 0, parsed.stderr

    foxglove = (REPO / "ros/foxglove.sh").read_text()
    # The desktop app is "Foxglove" since v2; the old "Foxglove Studio" process name matches
    # nothing, and a reopen that never fires is worse than no reopen.
    assert 'APP_PROCESS="${PEPIN_FOXGLOVE_APP:-Foxglove}"' in foxglove
    assert "foxglove://open?ds=foxglove-websocket&ds.url=" in foxglove
    assert "wait_for_port" in foxglove  # the link must not be fired at a port that is not up

    restart = (REPO / "ros/restart.sh").read_text()
    assert "check_foxglove" in restart and "PEPIN_FOXGLOVE_PREFIX=2.9" in restart
    assert restart.index("check_foxglove") < restart.index('step "verdict"')
    assert '"$HERE/foxglove.sh" reopen' in restart

    laptop = (REPO / "ros/laptop.sh").read_text()
    assert '"$HERE/foxglove.sh" reopen' in laptop
    assert "PEPIN_FOXGLOVE_REOPEN" in laptop
