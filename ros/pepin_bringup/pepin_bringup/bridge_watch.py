"""Restart this half of the laptop when the board's bridge is a new one.

Subscriptions made against one bridge do not follow it through a restart: after the board
rebooted, the laptop's costmap kept an old transform listener that never saw the new bridge's
/tf, no plan was ever produced and the tree spun the cart for 139 s (run 0148). This process
polls the board bridge's REST admin; when its zenoh id changes it waits for the board's routes
to settle, then exits with a distinct code. The launch is told to shut down on that exit and the
container's restart policy brings the whole half back with fresh subscriptions.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request

from pepin.deployment import BridgeIdentity, bridge_zid

BRIDGE_CHANGED_EXIT = 3
POLL_S = 5.0
SETTLE_S = 15.0  # routes stable this long after the change: the board's nodes are all declared


def fetch(url: str, timeout_s: float = 3.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as reply:
            return str(reply.read().decode("utf-8", "replace"))
    except Exception:
        return None


def route_count(board: str) -> int:
    text = fetch(f"http://{board}:8000/@/*/ros2/route/**", 6.0) or ""
    return text.count('"key"')


def main() -> None:
    board = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PEPIN_HOST", "10.0.0.187")
    identity = BridgeIdentity()
    print(f"bridge watch: {board}:8000", flush=True)
    while True:
        zid = bridge_zid(fetch(f"http://{board}:8000/@/local/router") or "")
        if identity.observe(zid):
            print(
                f"bridge watch: the board's bridge is a new one ({zid}); waiting for its routes",
                flush=True,
            )
            last, since = -1, time.monotonic()
            while time.monotonic() - since < 120.0:
                count = route_count(board)
                if count != last:
                    last, stable_since = count, time.monotonic()
                elif count >= 20 and time.monotonic() - stable_since >= SETTLE_S:
                    break
                time.sleep(POLL_S)
            print(f"bridge watch: routes settled at {last}; restarting this half", flush=True)
            os._exit(BRIDGE_CHANGED_EXIT)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
