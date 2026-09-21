"""Read-only probe of a foxglove_bridge websocket: does it answer, and what does it advertise?

The operator's whole view of the robot hangs on one websocket, and when it is dead or half-dead
the symptom is an empty panel — not an error. This asks the bridge the same three questions
Foxglove Studio asks on connect, without a browser and without ROS:

1. the HTTP upgrade succeeds and the server picks one of the Foxglove subprotocols;
2. the ``serverInfo`` message arrives (name and capabilities);
3. the ``advertise`` messages arrive, and the topics the layout draws are among them.

Nothing is subscribed and nothing is published: the socket is opened, read for a couple of
seconds and closed, so it is safe to run while the robot drives. Stdlib only (no ``websockets``
package), because it runs on the laptop's bare ``python3``.

Prints one ``PASS``/``FAIL`` line per item, numbered ``<prefix>.<n>``; exits 1 if any failed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# Studio offers both and takes what the server picks: foxglove_bridge 3.4.1 (the Foxglove
# SDK rewrite) speaks "foxglove.sdk.v1" and answers HTTP 400 to a client that asks only for
# the old "foxglove.websocket.v1" — measured on the laptop's bridge, 2026-09-15.
SUBPROTOCOLS = ("foxglove.sdk.v1", "foxglove.websocket.v1")
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"  # RFC 6455
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8

# The topics the pepin_slam layout draws. An entry with "|" is satisfied by any one of its
# alternatives (an image topic may be raw or compressed, and the layout takes either).
DEFAULT_REQUIRED = (
    "/scan",
    "/tf",
    # The one map, at both ends of it: RTAB-Map's live grid as the laptop publishes it, and the
    # grid the board's tracker accepted and drives on (both costmaps' static layers read that one).
    "/map",
    "/map_tracked",
    "/tracker_pose",
    "/fusion/surface",
    # The camera's two words to the costmap: the frame that clears and the volume's slice that
    # marks (pepin_bringup.depth_fusion, 2026-09-21).
    "/depth_scan",
    "/depth_marks",
    "/camera/image|/camera/image/compressed",
    "/rtabmap/mapGraph",
    "/plan",
    "/local_costmap/costmap",
)


@dataclass(frozen=True)
class Handshake:
    """The server's answer to the HTTP upgrade: status line, headers, and the bytes after them."""

    status: int
    headers: dict[str, str]
    rest: bytes

    @property
    def subprotocol(self) -> str:
        """The subprotocol the server picked, lowercase, or "" if it named none."""
        return self.headers.get("sec-websocket-protocol", "")


def accept_key(key: str) -> str:
    """The ``Sec-WebSocket-Accept`` value a server must return for this client key (RFC 6455)."""
    from hashlib import sha1

    return base64.b64encode(sha1((key + WS_GUID).encode()).digest()).decode()


def handshake_request(host: str, port: int, path: str, key: str) -> bytes:
    """The HTTP upgrade request for a foxglove_bridge websocket, ready to put on the socket."""
    return (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"Sec-WebSocket-Protocol: {', '.join(SUBPROTOCOLS)}\r\n"
        "\r\n"
    ).encode()


def parse_handshake(raw: bytes) -> Handshake | None:
    """Parse an upgrade response; None while the header block is still incomplete."""
    end = raw.find(b"\r\n\r\n")
    if end < 0:
        return None
    head, rest = raw[:end].decode("latin-1"), raw[end + 4 :]
    lines = head.split("\r\n")
    parts = lines[0].split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"not an HTTP response: {lines[0]!r}")
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return Handshake(int(parts[1]), headers, rest)


def decode_frames(buf: bytes) -> tuple[list[tuple[int, bytes]], bytes]:
    """Split a server-to-client byte stream into (opcode, payload) frames plus the unread tail.

    Server frames are never masked; the bridge sends its JSON unfragmented, so a continuation
    frame is returned with its own opcode (0) and left to the caller.
    """
    frames: list[tuple[int, bytes]] = []
    i = 0
    while len(buf) - i >= 2:
        first, second = buf[i], buf[i + 1]
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        head = i + 2
        if length == 126:
            if len(buf) - head < 2:
                break
            length = int.from_bytes(buf[head : head + 2], "big")
            head += 2
        elif length == 127:
            if len(buf) - head < 8:
                break
            length = int.from_bytes(buf[head : head + 8], "big")
            head += 8
        mask = b""
        if masked:
            if len(buf) - head < 4:
                break
            mask, head = buf[head : head + 4], head + 4
        if len(buf) - head < length:
            break
        payload = buf[head : head + length]
        if mask:
            payload = bytes(b ^ mask[n % 4] for n, b in enumerate(payload))
        frames.append((opcode, payload))
        i = head + length
    return frames, buf[i:]


def read_channels(payloads: Iterable[bytes]) -> dict[int, tuple[str, str]]:
    """Fold the bridge's ``advertise``/``unadvertise`` messages into {channel id: (topic, schema)}.

    Channel ids are the bridge's own numbering and change on every restart of it — which is why
    a panel bound to yesterday's id draws nothing until the client reconnects.
    """
    channels: dict[int, tuple[str, str]] = {}
    for payload in payloads:
        try:
            message = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(message, dict):
            continue
        if message.get("op") == "advertise":
            for channel in message.get("channels", []):
                channels[int(channel["id"])] = (
                    str(channel.get("topic", "")),
                    str(channel.get("schemaName", "")),
                )
        elif message.get("op") == "unadvertise":
            for channel_id in message.get("channelIds", []):
                channels.pop(int(channel_id), None)
    return channels


def server_info(payloads: Iterable[bytes]) -> dict[str, object] | None:
    """The bridge's ``serverInfo`` message (name, capabilities), or None if it never came."""
    for payload in payloads:
        try:
            message = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(message, dict) and message.get("op") == "serverInfo":
            return message
    return None


def missing_topics(advertised: Iterable[str], required: Sequence[str]) -> list[str]:
    """Which required entries no advertised topic satisfies; "a|b" is met by either name."""
    have = set(advertised)
    return [entry for entry in required if not (set(entry.split("|")) & have)]


def collect(host: str, port: int, seconds: float) -> tuple[Handshake, list[bytes]]:
    """Open the websocket, read text frames for ``seconds``, close. Returns the upgrade and them.

    Raises OSError if the socket or the upgrade fails — the caller turns that into a FAIL line.
    """
    key = base64.b64encode(os.urandom(16)).decode()
    deadline = time.monotonic() + seconds
    with socket.create_connection((host, port), timeout=seconds) as sock:
        sock.sendall(handshake_request(host, port, "/", key))
        buf = b""
        upgrade: Handshake | None = None
        payloads: list[bytes] = []
        while time.monotonic() < deadline:
            sock.settimeout(max(0.05, deadline - time.monotonic()))
            try:
                chunk = sock.recv(65536)
            except TimeoutError:
                break
            if not chunk:
                break
            buf += chunk
            if upgrade is None:
                upgrade = parse_handshake(buf)
                if upgrade is None:
                    continue
                if upgrade.status != 101:
                    raise OSError(f"the server answered HTTP {upgrade.status}, not 101")
                buf = upgrade.rest
            frames, buf = decode_frames(buf)
            for opcode, payload in frames:
                if opcode == OP_CLOSE:
                    deadline = 0.0
                elif opcode in (OP_TEXT, OP_BINARY):
                    payloads.append(payload)
        if upgrade is None:
            raise OSError("no HTTP response within the timeout")
        return upgrade, payloads


def main(argv: Sequence[str] | None = None) -> int:
    """Run the probe and print its PASS/FAIL lines; 0 if every item passed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seconds", type=float, default=4.0, help="how long to read (default 4)")
    parser.add_argument("--prefix", default="fg", help="number the lines <prefix>.1, .2, ...")
    parser.add_argument(
        "--number-from", type=int, default=0, help="continue a caller's numbering (default 0)"
    )
    parser.add_argument(
        "--require",
        default=",".join(DEFAULT_REQUIRED),
        help="comma-separated topics that must be advertised; A|B is met by either",
    )
    args = parser.parse_args(argv)
    required = [entry for entry in args.require.split(",") if entry]

    failed = 0
    n = args.number_from

    def line(ok: bool, text: str) -> None:
        nonlocal failed, n
        n += 1
        failed += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} {args.prefix}.{n:<3} {text}")

    try:
        upgrade, payloads = collect(args.host, args.port, args.seconds)
    except OSError as error:
        line(False, f"websocket ws://{args.host}:{args.port}: {error}")
        line(False, "serverInfo: no handshake, nothing to read")
        line(False, "channels: no handshake, nothing to read")
        for entry in required:
            line(False, f"topic {entry}: no handshake")
        return 1

    line(
        upgrade.subprotocol in SUBPROTOCOLS,
        f"websocket ws://{args.host}:{args.port}: HTTP {upgrade.status}, "
        f"subprotocol {upgrade.subprotocol or 'none'}",
    )
    info = server_info(payloads)
    if info is None:
        line(
            False, f"serverInfo: none in {args.seconds:g} s (the bridge answered but said nothing)"
        )
    else:
        offered = info.get("capabilities")
        caps = ",".join(str(c) for c in offered) if isinstance(offered, list) else ""
        name = str(info.get("name") or "unnamed")
        line(True, f"serverInfo: {name}, capabilities {caps or 'none'}")

    channels = read_channels(payloads)
    topics = {topic for topic, _ in channels.values()}
    line(bool(channels), f"channels: {len(channels)} advertised")

    schema_of = {topic: schema for topic, schema in channels.values()}
    for entry in required:
        present = [name for name in entry.split("|") if name in topics]
        if present:
            line(True, f"topic {present[0]}: {schema_of[present[0]] or 'no schema'}")
        else:
            line(False, f"topic {entry}: not advertised (no publisher, or the route is dead)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
