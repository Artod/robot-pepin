#!/bin/bash
# The neck's two servos through the base server on the board (:3336) — where the head is, and
# where it should be. One JSON line in, one line out; no ROS, no container.
#   ros/neck.sh                 the same as read
#   ros/neck.sh read            what the encoders read now, in ticks and in degrees
#   ros/neck.sh home            back to the reference pose of config/neck.json, then torque off
#   ros/neck.sh goto PAN TILT   move to those encoder ticks, then torque off (push it by hand again)
#   ros/neck.sh hold PAN TILT   the same, but leave the servos energised so the head holds its pose
#   ros/neck.sh motion [KEY VALUE ...]
#                               the head's top speed, ramp and lease as the server holds them, and
#                               config/neck.json's; KEY is max_speed_deg_s, max_acc_deg_s2,
#                               tilt_max_acc_deg_s2 (the tilt's ramp under the other) or
#                               lease_s, set live until pepin-base restarts (every start reads the
#                               file's motion block)
#   ros/neck.sh registers SERVO ADDRESS [SIZE]
#                               raw bytes of a roster servo's control table (neck, head, left, ...),
#                               read by the server that owns the bus; e.g. "head 85" is the
#                               acceleration ceiling the server lifts to 254, "neck 41 1" the ramp
# Ticks, not degrees, on purpose: the encoders are what the server speaks, and config/neck.json's
# limits are in ticks (pan 257..3812, tilt 1814..2760). A target outside them is refused by the
# server, never quietly clamped; a move runs at the motion's top speed, wheels turning or not, and
# gives up with "NOT reached" after half again its travel time plus a second (3 s at least).
# PEPIN_HOST picks the board, PEPIN_BASE_PORT the port.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
PORT="${PEPIN_BASE_PORT:-3336}"
PYTHONPATH="$HERE/../src" exec python3 - "$BOARD" "$PORT" "$HERE/../config/neck.json" "$@" <<'PY'
"""One request to the base server's JSON-lines port, and its answer in ticks and degrees."""

import json
import math
import socket
import sys
import time
from typing import Any

from pepin.neck import NeckConfig, joint_angles

USAGE = (
    "usage: ros/neck.sh read | home | goto PAN TILT | hold PAN TILT | motion [KEY VALUE ...]"
    " | registers SERVO ADDRESS [SIZE]"
)


def request(argv: list[str]) -> tuple[dict[str, Any], str, float]:
    """The command line as (message, the reply type to wait for, how long to wait)."""
    action = argv[0] if argv else "read"
    if action == "read":
        return {"cmd": "neck"}, "neck", 3.0
    if action == "home":
        return {"cmd": "neck_home"}, "neck_goto", 20.0
    if action in ("goto", "hold") and len(argv) == 3:
        goal = {"cmd": "neck_goto", "pan_ticks": int(argv[1]), "tilt_ticks": int(argv[2])}
        return {**goal, "hold": action == "hold"}, "neck_goto", 20.0
    if action == "registers" and len(argv) in (3, 4):
        size = int(argv[3]) if len(argv) == 4 else 1
        read = {"cmd": "registers", "servo": argv[1], "address": int(argv[2]), "size": size}
        return read, "registers", 3.0
    if action == "motion" and len(argv) % 2 == 1:
        pairs = zip(argv[1::2], argv[2::2], strict=True)
        return {"cmd": "neck_motion", **{k: float(v) for k, v in pairs}}, "neck_motion", 3.0
    raise SystemExit(USAGE)


def report(cfg: NeckConfig, message: dict[str, Any]) -> None:
    """Print one reply as ticks, degrees and how the move went; exit 1 on an error in it."""
    if message.get("type") == "neck_motion":
        keys = ("max_speed_deg_s", "max_acc_deg_s2", "tilt_max_acc_deg_s2", "lease_s")
        was = message.get("was", {})
        print("   ".join(
            f"{k} {message.get(k)}" + (f" (was {was[k]})" if k in was else "")
            + f" [file {message.get('config', {}).get(k)}]"
            for k in keys
        ))
        if message.get("error"):
            print(f"error: {message['error']}", file=sys.stderr)
            raise SystemExit(1)
        return
    if message.get("type") == "registers":
        if message.get("busy"):
            message["error"] = "refused while the wheels turn"
        if "values" in message:
            print(f"{message['servo']} {message['address']}: {' '.join(map(str, message['values']))}")
        if message.get("error"):
            print(f"error: {message['error']}", file=sys.stderr)
            raise SystemExit(1)
        return
    pan, tilt = message.get("pan_ticks"), message.get("tilt_ticks")
    parts = []
    if pan is not None and tilt is not None:
        angles = joint_angles(cfg, int(pan), int(tilt))
        parts.append(f"pan {pan} ticks ({math.degrees(angles.pan_rad):+.1f} deg left)")
        parts.append(f"tilt {tilt} ticks ({math.degrees(angles.pitch_rad):+.1f} deg down)")
    if "reached" in message:
        arrival = "reached" if message["reached"] else "NOT reached"
        parts.append(f"{arrival} in {float(message.get('ms', 0.0)):.0f} ms")
    if message.get("hold"):
        parts.append("torque held")
    if "age_s" in message:
        parts.append(f"read {float(message['age_s']) * 1000:.0f} ms ago")
    print("   ".join(parts) if parts else json.dumps(message))
    if message.get("error"):
        print(f"error: {message['error']}", file=sys.stderr)
    if message.get("error") or message.get("reached") is False:
        raise SystemExit(1)  # a refused or unfinished move must fail the shell that asked


def main() -> None:
    """Send, then read lines until the answer we asked for shows up among the state lines."""
    host, port, config_path, *argv = sys.argv[1:]
    message, want, wait_s = request(argv)
    cfg = NeckConfig.from_json(config_path)
    sock = socket.create_connection((host, int(port)), timeout=3.0)
    sock.sendall((json.dumps(message) + "\n").encode())
    sock.settimeout(1.0)
    deadline, buffer = time.monotonic() + wait_s, b""
    while time.monotonic() < deadline:
        try:
            chunk = sock.recv(4096)
        except TimeoutError:
            continue
        if not chunk:
            raise SystemExit("the base server closed the connection")
        buffer += chunk
        *lines, buffer = buffer.split(b"\n")
        for line in lines:
            if not line.strip():
                continue
            reply = json.loads(line)
            # The port broadcasts a state line 50 times a second: ours is the one we asked for.
            if reply.get("type") == want:
                sock.close()
                report(cfg, reply)
                return
    raise SystemExit(f"no {want} answer from {host}:{port} in {wait_s:.0f} s")


main()
PY
