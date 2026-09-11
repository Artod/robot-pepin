"""Wait until the bridge has forgotten a previous incarnation of the given node names.

zenoh-bridge-ros2dds keeps a route's local nodes by name. A container replaced within the DDS
lease of its killed predecessor (ten seconds) shows the bridge two /rtabmap/rtabmap; when the
ghost expires the bridge drops the name from every route, and the live node is left with a deaf
/scan route and no map route out (RTAB-Map iterated for seven seconds at a time, only while some
other subscriber happened to exist, 2026-09-10). A crashed node meets the same ghost: a crash
disposes nothing, the launch respawns the node two seconds later, and the bridge drops the
name's routes when the ghost expires (2026-09-11 03:07, the Nav2 container: /map and
/navigate_to_pose gone until a stack restart). This process polls the bridge's REST admin until
none of the given node names are listed, then exits — or, given a command after ``--``, becomes
it (``execvp``: same pid, so the launch's signals reach the node directly and a kick's ``pkill``
sees one process). An unreachable bridge is not waited for: a bridge that starts later
discovers cleanly.

Usage: python3 -m pepin_bringup.ghost_wait <bridge admin url> <node name>... [-- <command>...]
"""

from __future__ import annotations

import os
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


def split_args(argv: list[str]) -> tuple[str, tuple[str, ...], list[str]]:
    """``(admin, names, command)`` from the arguments after the module name: the names run up
    to ``--``, the command (possibly none) is everything after it."""
    admin = argv[0].rstrip("/")
    rest = argv[1:]
    if "--" in rest:
        at = rest.index("--")
        return admin, tuple(rest[:at]), rest[at + 1 :]
    return admin, tuple(rest), []


def wait(admin: str, names: tuple[str, ...]) -> None:
    """Block until the admin at ``admin`` lists none of ``names``, the patience runs out, or
    the admin does not answer; say which."""
    who = names[0] if len(names) == 1 else f"{len(names)} names"
    since = time.monotonic()
    ghosts: set[str] = set()
    while time.monotonic() - since < PATIENCE_S:
        reply = fetch(f"{admin}/@/local/ros2/node/**")
        if reply is None:
            print(f"ghost wait [{who}]: no bridge admin at {admin}; not waiting", flush=True)
            return
        lingering = lingering_nodes(reply, names)
        if not lingering:
            waited = time.monotonic() - since
            what = f"{sorted(ghosts)} gone after {waited:.1f} s" if ghosts else "nothing lingering"
            print(f"ghost wait [{who}]: {what}", flush=True)
            return
        ghosts |= lingering
        time.sleep(POLL_S)
    print(
        f"ghost wait [{who}]: {sorted(ghosts)} still listed after {PATIENCE_S:.0f} s; "
        "starting anyway",
        flush=True,
    )


def main() -> None:
    admin, names, command = split_args(sys.argv[1:])
    try:
        wait(admin, names)
    except KeyboardInterrupt:  # the launch is shutting down, or a kick: nothing to start
        sys.exit(130)
    if command:
        sys.stdout.flush()
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
