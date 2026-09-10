"""Wait until the laptop's bridge has forgotten this launch's previous incarnation.

zenoh-bridge-ros2dds keeps a route's local nodes by name. A container replaced within the DDS
lease of its killed predecessor (ten seconds) shows the bridge two /rtabmap/rtabmap; when the
ghost expires the bridge drops the name from every route, and the live node is left with a deaf
/scan route and no map route out (RTAB-Map iterated for seven seconds at a time, only while some
other subscriber happened to exist, 2026-09-10). This process polls the bridge's REST admin
until none of the given node names are listed, then exits; the launch starts its nodes on that
exit. An unreachable bridge is not waited for: a bridge that starts later discovers cleanly.

Usage: python3 -m pepin_bringup.ghost_wait <bridge admin url> <node name>...
"""

from __future__ import annotations

import sys
import time
import urllib.request

from pepin.deployment import lingering_nodes

POLL_S = 0.5
PATIENCE_S = 30.0  # three DDS leases: a name still listed after that is a twin, not a ghost


def fetch(url: str, timeout_s: float = 3.0) -> str | None:
    """The body of ``url``, or ``None`` when it does not answer."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as reply:
            return str(reply.read().decode("utf-8", "replace"))
    except Exception:
        return None


def main() -> None:
    admin, names = sys.argv[1].rstrip("/"), tuple(sys.argv[2:])
    since = time.monotonic()
    ghosts: set[str] = set()
    while time.monotonic() - since < PATIENCE_S:
        reply = fetch(f"{admin}/@/local/ros2/node/**")
        if reply is None:
            print(f"ghost wait: no bridge admin at {admin}; not waiting", flush=True)
            return
        lingering = lingering_nodes(reply, names)
        if not lingering:
            waited = time.monotonic() - since
            what = f"{sorted(ghosts)} gone after {waited:.1f} s" if ghosts else "nothing lingering"
            print(f"ghost wait: {what}", flush=True)
            return
        ghosts |= lingering
        time.sleep(POLL_S)
    print(
        f"ghost wait: {sorted(ghosts)} still listed after {PATIENCE_S:.0f} s; starting anyway",
        flush=True,
    )


if __name__ == "__main__":
    main()
