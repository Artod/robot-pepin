"""Restart this half of the laptop when the board's bridge is a new one, or gone.

Subscriptions made against one bridge do not follow it through a restart: after the board
rebooted, the laptop's costmap kept an old transform listener that never saw the new bridge's
/tf, no plan was ever produced and the tree spun the cart for 139 s (run 0148). This process
polls the board bridge's REST admin; when its zenoh id changes, or the admin has answered
nothing for a minute, or it answers for the first time to a watch that started without it
(pepin.deployment.BridgeIdentity), it waits for the board's routes to settle, then exits with
a distinct code. The launch is told to shut down on that exit and the container's restart
policy brings the whole half back with fresh subscriptions.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request

from pepin.deployment import BridgeIdentity, bridge_zid, routes_settled

BRIDGE_CHANGED_EXIT = 3
POLL_S = 5.0
SETTLE_S = 15.0  # routes stable this long after the change: the board's nodes are all declared
SETTLE_PATIENCE_S = 120.0
SILENCE_S = 60.0  # an admin silent this long is a wedged or absent bridge: restart the half


def fetch(url: str, timeout_s: float = 3.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as reply:
            return str(reply.read().decode("utf-8", "replace"))
    except Exception:
        return None


def route_count(board: str) -> int:
    text = fetch(f"http://{board}:8000/@/*/ros2/route/**", 6.0) or ""
    return text.count('"key"')


def wait_for_routes(board: str, expected: int | None) -> int:
    """Poll the new bridge's route count until it settles (pepin.deployment.routes_settled)
    or the patience runs out; the count reached."""
    last, since, stable_since = -1, time.monotonic(), time.monotonic()
    while time.monotonic() - since < SETTLE_PATIENCE_S:
        count = route_count(board)
        if count != last:
            last, stable_since = count, time.monotonic()
        elif routes_settled(count, expected, time.monotonic() - stable_since, SETTLE_S):
            break
        time.sleep(POLL_S)
    return last


def main() -> None:
    board = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PEPIN_HOST", "10.0.0.187")
    identity = BridgeIdentity(silence_s=SILENCE_S)
    expected: int | None = None  # the healthy bridge's route count, measured at first contact
    print(f"bridge watch: {board}:8000", flush=True)
    while True:
        zid = bridge_zid(fetch(f"http://{board}:8000/@/local/router") or "")
        if zid is not None and expected is None:
            expected = route_count(board) or None
            print(f"bridge watch: {zid} has {expected or 0} routes", flush=True)
        if identity.observe(zid, time.monotonic()):
            why = f"is a new one ({zid})" if zid else f"answered nothing for {SILENCE_S:.0f} s"
            print(f"bridge watch: the board's bridge {why}; waiting for its routes", flush=True)
            count = wait_for_routes(board, expected)
            print(f"bridge watch: routes settled at {count}; restarting this half", flush=True)
            os._exit(BRIDGE_CHANGED_EXIT)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
