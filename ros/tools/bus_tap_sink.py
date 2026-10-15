"""The laptop's copy of the servo bus tap (:mod:`pepin.bus_tap`): a UDP listener that appends
every line the board's tap sends to ``data/bus_tap/<day>.log`` (the laptop's local date), so
the record of the bus outlives the robot's power switch. A ring of :data:`KEEP_DAYS` days:
older files are deleted when a new day's file opens (Artem, 2026-10-13: nothing of it goes to
the bucket; a case worth keeping is copied out by hand).

    uv run python ros/tools/bus_tap_sink.py serve              # listen on 3341 until killed
    uv run python ros/tools/bus_tap_sink.py install            # launchd: at login, kept alive
    uv run python ros/tools/bus_tap_sink.py status             # today's file, its last line
    uv run python ros/tools/bus_tap_sink.py prune              # the ring by hand: three days kept
    uv run python ros/tools/bus_tap_decode.py data/bus_tap/<day>.log --from 19:51:20 --to 19:51:30

Reading the record: ``ros/tools/bus_tap_decode.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIRECTORY = REPO / "data" / "bus_tap"
PORT = 3341
LABEL = "com.pepin.bustap"
FSYNC_EVERY_S = 1.0
KEEP_DAYS = 3  # the ring: today's file and the two before it


def prune(directory: Path, keep_days: int = KEEP_DAYS) -> list[Path]:
    """Delete the day files beyond the newest ``keep_days``; the files deleted."""
    days = sorted(directory.glob("[0-9]*.log"))
    gone = days[:-keep_days] if keep_days > 0 else days
    for path in gone:
        path.unlink()
    return gone


def serve(port: int, directory: Path) -> None:
    """Append every datagram to the day's file, flushed each line, fsynced every second."""
    directory.mkdir(parents=True, exist_ok=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    day = ""
    out = None
    synced = 0.0
    print(f"bus tap sink on udp {port} -> {directory}", flush=True)
    while True:
        data, _peer = sock.recvfrom(65535)
        today = datetime.now().strftime("%Y%m%d")
        if today != day or out is None:
            if out is not None:
                out.close()
            day = today
            out = open(directory / f"{day}.log", "ab")  # noqa: SIM115 (the day's file, kept open)
            for old_file in prune(directory):
                print(f"ring: {old_file.name} deleted", flush=True)
        out.write(data if data.endswith(b"\n") else data + b"\n")
        out.flush()
        now = time.monotonic()
        if now - synced >= FSYNC_EVERY_S:
            os.fsync(out.fileno())
            synced = now


def install(port: int, directory: Path) -> None:
    """A launchd agent: the sink at login, restarted whenever it dies."""
    agents = Path.home() / "Library" / "LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    python = REPO / ".venv" / "bin" / "python"
    plist = agents / f"{LABEL}.plist"
    log = Path.home() / "Library" / "Logs" / "pepin-bus-tap-sink.out"
    plist.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key><array>
    <string>{python}</string><string>{Path(__file__).resolve()}</string>
    <string>serve</string><string>--port</string><string>{port}</string>
    <string>--dir</string><string>{directory}</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict></plist>
"""
    )
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/{LABEL}"], capture_output=True)
    subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(plist)], check=True)
    print(f"installed {LABEL}: {plist}")


def status(directory: Path) -> None:
    """Today's file: size, lines, the last line's age."""
    today = directory / f"{datetime.now().strftime('%Y%m%d')}.log"
    if not today.exists():
        print(f"no file yet: {today}")
        return
    size = today.stat().st_size
    with open(today, "rb") as f:
        f.seek(max(0, size - 4096))
        tail = f.read().splitlines()
    last = tail[-1].decode(errors="replace") if tail else ""
    age = ""
    with contextlib.suppress(ValueError, IndexError):
        age = f", {time.time() - float(last.split()[0]):.1f} s ago"
    print(f"{today}: {size / 1e6:.1f} MB; last line: {last[:60]}{age}")


def main(argv: list[str] | None = None) -> int:
    """serve | install | status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("serve", "install", "status", "prune"))
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--dir", type=Path, default=DIRECTORY)
    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(args.port, args.dir)
    elif args.command == "install":
        install(args.port, args.dir)
    elif args.command == "prune":
        for path in prune(args.dir):
            print(f"deleted {path}")
    else:
        status(args.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
