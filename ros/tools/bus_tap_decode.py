"""Read a servo bus tap log (:mod:`pepin.bus_tap`) as packets: who was told what, what answered,
and where the bytes were not clean (a failed checksum, noise between packets).

    uv run python ros/tools/bus_tap_decode.py data/bus_tap/20261013.log --from 19:51:20 \\
        --to 19:51:30
    uv run python ros/tools/bus_tap_decode.py data/bus_tap/20261013.log --ids 9,10 --writes
    uv run python ros/tools/bus_tap_decode.py data/bus_tap/20261013.log --summary

Times are the laptop's local clock (the board's clock is chrony-locked to it). ``--writes``
keeps only the server's writes (and the replies to acknowledged ones), ``--ids`` only packets
naming those servos (a sync write to several is kept when one of them matches).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from pepin.bus_tap import FROM_BUS, TO_BUS, Packet, Stream, describe, parse_line
from pepin.feetech import INST_SYNC_READ, INST_SYNC_WRITE, INST_WRITE


def hms(epoch: float) -> str:
    """Local HH:MM:SS.ff."""
    return datetime.fromtimestamp(epoch).strftime("%H:%M:%S.%f")[:-4]


def names_ids(packet: Packet, ids: set[int]) -> bool:
    """Whether the packet concerns one of ``ids``."""
    if packet.motor_id in ids:
        return True
    p = packet.params
    if packet.instruction == INST_SYNC_WRITE and len(p) >= 2:
        size = p[1]
        return any(p[k] in ids for k in range(2, len(p), 1 + size))
    if packet.instruction == INST_SYNC_READ and len(p) >= 2:
        return any(b in ids for b in p[2:])
    return False


def main(argv: list[str] | None = None) -> int:
    """Print the log's packets, filtered; a summary at the end."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("log", type=Path)
    parser.add_argument("--from", dest="start", default=None, help="local HH:MM:SS")
    parser.add_argument("--to", dest="end", default=None, help="local HH:MM:SS")
    parser.add_argument("--ids", default=None, help="only these servo ids, e.g. 9,10")
    parser.add_argument("--writes", action="store_true", help="the server's writes only")
    parser.add_argument("--summary", action="store_true", help="counts only")
    args = parser.parse_args(argv)
    ids = {int(x) for x in args.ids.split(",")} if args.ids else None
    streams = {TO_BUS: Stream(TO_BUS), FROM_BUS: Stream(FROM_BUS)}
    counts: Counter[str] = Counter()
    bad: list[str] = []
    shown = 0
    with open(args.log, encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            epoch, _mono, direction, data = parsed
            when = hms(epoch)
            if args.start and when < args.start:
                streams[direction].feed(epoch, data)  # keep the framer in step
                continue
            if args.end and when > args.end:
                break
            for stamp, packet in streams[direction].feed(epoch, data):
                kind = (
                    "reply"
                    if direction == FROM_BUS
                    else describe(packet, direction).split()[2 if packet.motor_id != 0xFE else 1]
                )
                counts[f"{direction} {kind}"] += 1
                if not packet.ok:
                    counts["CHK FAIL"] += 1
                if packet.noise:
                    counts["noise bytes"] += packet.noise
                text = f"{hms(stamp)} {direction} {describe(packet, direction)}"
                if not packet.ok or packet.noise:
                    bad.append(text)
                if args.summary:
                    continue
                if ids is not None and not names_ids(packet, ids):
                    continue
                if (
                    args.writes
                    and direction == TO_BUS
                    and packet.instruction
                    not in (
                        INST_WRITE,
                        INST_SYNC_WRITE,
                    )
                ):
                    continue
                if args.writes and direction == FROM_BUS:
                    continue
                print(text)
                shown += 1
    print(f"--- {shown} packets shown; counts: {dict(counts)}")
    if bad:
        print(f"--- {len(bad)} packets not clean:")
        for text in bad[:200]:
            print("  " + text)
    for direction, stream in streams.items():
        if stream.carry:
            print(f"--- {direction} tail not framed: {stream.carry.hex()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
