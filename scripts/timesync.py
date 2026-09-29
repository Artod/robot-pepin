"""One clock for the robot: how far the board's clock is from the laptop's, measured over NTP.

Every ROS stamp on the laptop is the Docker VM's clock (all its containers share one kernel);
every stamp on the board is the board's. A scan, a transform and a map cross between the two, so
the robot needs ONE time base: the board's chrony syncs to a time server that serves the VM's
clock from a container on the laptop (``ros/chrony``, ``board/chrony.sh``), with the internet
pool as its fallback when the laptop is away. This module is the measurement that says whether
that worked: a minimal SNTP client (RFC 4330) run ON THE BOARD against the laptop's server,
which asks N times and keeps the exchange with the shortest round trip — the one whose
asymmetry can hide the least.

Standard library only, and no import from this package: ``ros/time.sh offset`` sends this file
to the board's system python3 over ssh (``python3 - <laptop>``), so nothing has to be installed
there for the robot's restart check (``ros/restart.sh``, check 1.15) to ask the question.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

NTP_PORT = 123
PACKET_BYTES = 48
# Seconds from the NTP era's origin (1900-01-01) to the Unix epoch (1970-01-01).
NTP_UNIX_OFFSET_S = 2_208_988_800
_FRACTION = 2**32

# WHERE THE BOARD TAKES ITS TIME FROM (CLAUDE.md rule 19: the old way stays one word away).
# "laptop": the laptop's time server first, the internet pool as the fallback — one time base for
# both machines' stamps. "pool": the internet pool alone, the references the board had under
# systemd-timesyncd (the daemon itself comes back with `ros/time.sh uninstall`). "laptop" is the
# default since the deploy was measured on the robot (2026-09-24: board minus laptop -0.2 ms, round
# trip 6 ms, against +21 ms under systemd-timesyncd); it is flipped here and in ros/lib.sh
# together. What DECIDES is the shell: ros/lib.sh (does the laptop start its server;
# pepin_time_source_check refuses any other value) and the board's /etc/default/pepin-ros
# (board/chrony.sh writes the board's sources from it, and its `install` writes "laptop" there
# explicitly). These constants are the contract the tests hold those scripts to.
TIME_SOURCES = ("laptop", "pool")
TIME_SOURCE_ENV = "PEPIN_TIME_SOURCE"
DEFAULT_TIME_SOURCE = "laptop"

# How far apart the two clocks may be before the restart check says so. A tenth of a second is
# the order of the stack's transform tolerances (0.1-0.5 s) and about twice what one NTP exchange
# over this radio can resolve (the best of eight round trips is 60-100 ms, so the asymmetry it can
# hide is +-30-50 ms). Information only: a restart is never failed on it and no drive is gated.
OFFSET_WARN_S = 0.1
SAMPLES = 8
TIMEOUT_S = 1.0

# Exit codes of the command line, which ros/time.sh and ros/restart.sh read.
EXIT_WITHIN = 0
EXIT_OVER = 1
EXIT_UNMEASURED = 2


def to_ntp(unix_s: float) -> int:
    """A Unix time in seconds as a 64-bit NTP timestamp (32.32 fixed point since 1900)."""
    return round((unix_s + NTP_UNIX_OFFSET_S) * _FRACTION) & 0xFFFF_FFFF_FFFF_FFFF


def from_ntp(value: int) -> float:
    """A 64-bit NTP timestamp as a Unix time in seconds."""
    return value / _FRACTION - NTP_UNIX_OFFSET_S


def request(transmit_unix_s: float) -> bytes:
    """A 48-byte SNTP client request (version 4, mode 3) carrying our transmit time, which the
    server echoes back as its originate timestamp — the proof that a reply answers THIS query."""
    first = (0 << 6) | (4 << 3) | 3  # leap 0, version 4, mode 3 (client)
    return struct.pack("!B39xQ", first, to_ntp(transmit_unix_s))


@dataclass(frozen=True)
class Reply:
    """What a server's answer says: its leap indicator, stratum and three timestamps (Unix s)."""

    leap: int
    stratum: int
    originate_s: float  # our transmit time, echoed
    receive_s: float  # when the server got the query (t2)
    transmit_s: float  # when the server sent the answer (t3)


def parse(reply: bytes, sent: bytes) -> Reply:
    """Parse a server's answer to ``sent``; raises ``ValueError`` on anything that is not one."""
    if len(reply) < PACKET_BYTES:
        raise ValueError(f"a {len(reply)}-byte reply is not an NTP packet")
    first, stratum = reply[0], reply[1]
    if first & 0x7 != 4:
        raise ValueError(f"mode {first & 0x7} is not a server's reply")
    originate, receive, transmit = struct.unpack("!QQQ", reply[24:48])
    if originate != struct.unpack("!Q", sent[40:48])[0]:
        raise ValueError("the reply answers another query (its originate is not our transmit)")
    return Reply(first >> 6, stratum, from_ntp(originate), from_ntp(receive), from_ntp(transmit))


@dataclass(frozen=True)
class Sample:
    """One exchange: the server's clock minus ours, and the round trip it was measured over."""

    offset_s: float
    delay_s: float
    stratum: int
    leap: int


def sample(sent_s: float, reply: Reply, received_s: float) -> Sample:
    """The standard NTP arithmetic over t1 (``sent_s``), t2, t3 and t4 (``received_s``): the
    offset assumes the two legs took equally long, which is why the delay bounds its error."""
    offset = ((reply.receive_s - sent_s) + (reply.transmit_s - received_s)) / 2.0
    delay = (received_s - sent_s) - (reply.transmit_s - reply.receive_s)
    return Sample(offset, delay, reply.stratum, reply.leap)


def best(samples: Sequence[Sample]) -> Sample | None:
    """The sample with the shortest round trip (the least room for asymmetry), or ``None``."""
    return min(samples, key=lambda s: s.delay_s, default=None)


def unsynchronised(s: Sample) -> bool:
    """Whether the server says its own time is not to be trusted (leap alarm or stratum 0/16)."""
    return s.leap == 3 or s.stratum in (0, 16)


def verdict(s: Sample, warn_s: float = OFFSET_WARN_S) -> tuple[int, str]:
    """The exit code and the one line a person reads: the BOARD's clock minus the laptop's.

    The client is the board, so ``offset_s`` is laptop minus board and the line flips its sign:
    positive means the board is ahead."""
    board_minus_laptop_ms = -s.offset_s * 1000.0
    ahead = "the board is ahead" if board_minus_laptop_ms >= 0 else "the board is behind"
    line = (
        f"board - laptop {board_minus_laptop_ms:+.1f} ms ({ahead}), "
        f"round trip {s.delay_s * 1000.0:.0f} ms, server stratum {s.stratum}"
    )
    if unsynchronised(s):
        return EXIT_UNMEASURED, f"{line}; the server calls itself unsynchronised: no measurement"
    if abs(s.offset_s) > warn_s:
        return EXIT_OVER, f"{line}; over {warn_s * 1000.0:.0f} ms"
    return EXIT_WITHIN, line


# One exchange with a server: send the packet, return (reply bytes, local receive time), or None
# on a timeout. A protocol-free callable, so a test drives measure() without a socket.
Exchange = Callable[[bytes], tuple[bytes, float] | None]


def udp_exchange(host: str, port: int = NTP_PORT, timeout_s: float = TIMEOUT_S) -> Exchange:
    """A real :data:`Exchange` over one UDP socket per call (closed again at once)."""

    def exchange(packet: bytes) -> tuple[bytes, float] | None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout_s)
            try:
                sock.sendto(packet, (host, port))
                data, _ = sock.recvfrom(1024)
            except OSError:  # a timeout is an OSError too
                return None
            return data, time.time()

    return exchange


def measure(
    exchange: Exchange, count: int = SAMPLES, clock: Callable[[], float] = time.time
) -> list[Sample]:
    """Up to ``count`` samples from ``exchange``; unanswered and malformed replies are skipped."""
    out: list[Sample] = []
    for _ in range(count):
        sent_s = clock()
        packet = request(sent_s)
        answer = exchange(packet)
        if answer is None:
            continue
        data, received_s = answer
        try:
            out.append(sample(sent_s, parse(data, packet), received_s))
        except ValueError:
            continue
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """``python3 timesync.py HOST [--port N] [--samples N] [--warn-ms N]``: one line, and the exit
    code :data:`EXIT_WITHIN`, :data:`EXIT_OVER` or :data:`EXIT_UNMEASURED`."""
    parser = argparse.ArgumentParser(description="The board's clock minus the laptop's, over NTP.")
    parser.add_argument("host", help="the laptop's address as the board reaches it")
    parser.add_argument("--port", type=int, default=NTP_PORT)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--warn-ms", type=float, default=OFFSET_WARN_S * 1000.0)
    args = parser.parse_args(argv)
    samples = measure(udp_exchange(args.host, args.port), args.samples)
    chosen = best(samples)
    if chosen is None:
        print(f"no answer from the time server at {args.host}:{args.port} in {args.samples} tries")
        return EXIT_UNMEASURED
    code, line = verdict(chosen, args.warn_ms / 1000.0)
    where = f"NTP from the board to {args.host}:{args.port}"
    print(f"{line} (best of {len(samples)}/{args.samples}, {where})")
    return code


if __name__ == "__main__":
    sys.exit(main())
