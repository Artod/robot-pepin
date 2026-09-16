"""The board's ear for one sentence from the laptop: "restart your zenoh bridge, mine is newer".

Of two zenoh bridges the one that starts LAST gets working routes. A route's DDS endpoint is
built when the route is created and only while the far bridge is already announcing, so after a
transport is dropped and re-established — a wireless stall, a bridge restarted on one side — the
side that has been up longer keeps routes with an empty endpoint and carries nothing, while
every count in the admin looks right (:func:`pepin.deployment.far_dead_routes`). That is why
ros/laptop.sh restarts the board's bridge over ssh (``settle_bridge``) the moment the laptop's
own bridge is up, and it is what cures the fault by hand.

The laptop's watch (``pepin_bringup.bridge_watch``) cannot do that: it lives in a container with
no ssh key and must not have one. So it publishes one ``std_msgs/String`` on
:data:`pepin.deployment.BRIDGE_KICK_TOPIC` and this handler — inside a node that already runs on
the board — touches :data:`pepin.deployment.BRIDGE_KICK_FLAG` in the ROS container's /run/pepin,
which is the board's own /run/pepin (ros/run.sh mounts it). A systemd path unit there
(board/pepin-bridge-kick.path) sees the file appear and runs ``systemctl restart pepin-bridge``
with the board's privileges, not with ours. Nothing in this process runs as the board's root and
nothing here can restart anything else.

The cost on the board (CLAUDE.md rule 20): one subscription to a topic that carries a message
only when a repair is already happening — no messages at all on a healthy link — and a 60-byte
write to tmpfs when it does. With the laptop gone there is nothing to kick and nothing runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from std_msgs.msg import String

from pepin.deployment import BRIDGE_KICK_FLAG, BRIDGE_KICK_TOPIC
from pepin_bringup.node_kit import bridged_qos_profile

# Two kicks closer together than this are one: the board's bridge takes ~25 s to come back (its
# unit waits for the stack's last node and 20 s more), and the laptop's watch is allowed to ask
# again long before that. The board's own script (board/bridge_kick.sh) keeps the same guard —
# this one only saves the file write.
KICK_COOLDOWN_S = 120.0
REASON_CHARS = 200  # a log line, not a message: whatever the laptop said, cut to one line
MOUNTINFO = Path("/proc/self/mountinfo")


def mounted(path: Path, mountinfo: Path = MOUNTINFO) -> bool | None:
    """Whether ``path`` is a mount point in this container — ``None`` when the question cannot
    be asked (no /proc: not Linux, so not the board).

    Asked once, at start, because the failure is otherwise silent: without the bind mount
    (ros/run.sh) the flag file is written into the container's own tmpfs, the log says it was
    written, and the board's systemd never sees anything.
    """
    try:
        lines = mountinfo.read_text().splitlines()
    except OSError:
        return None
    return any(f" {path} " in line for line in lines)


class BridgeKick:
    """Turns the laptop's kick message into a flag file the board's systemd watches.

    ``node`` is the board node that hosts it (the run recorder), ``flag`` the file to touch,
    ``enabled`` the node's live switch (CLAUDE.md rule 19: off, the request is logged and no
    file is written), ``cooldown_s`` how long a kick silences the next one, ``clock`` the
    monotonic clock a test drives.
    """

    def __init__(
        self,
        node: Any,
        flag: Path | str = BRIDGE_KICK_FLAG,
        enabled: Callable[[], bool] = lambda: True,
        cooldown_s: float = KICK_COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._node = node
        self._flag = Path(flag)
        self._enabled = enabled
        self._cooldown_s = cooldown_s
        self._now = (
            clock  # not _clock: an rclpy Node owns that name (tests/unit/test_nav_contract.py)
        )
        self._last: float | None = None
        self.kicks = 0
        node.create_subscription(
            String, f"/{BRIDGE_KICK_TOPIC}", self._on_kick, bridged_qos_profile(BRIDGE_KICK_TOPIC)
        )
        if mounted(self._flag.parent) is False:
            node.get_logger().warning(
                f"bridge kick: {self._flag.parent} is not mounted from the board, so a kick"
                " would be written into this container and seen by nobody — redeploy the board"
                " (ros/sync.sh, which brings ros/run.sh with the mount, then a stack restart)"
            )

    def _on_kick(self, msg: String) -> None:
        """One request from the laptop: write the flag, or say why not."""
        reason = str(getattr(msg, "data", ""))[:REASON_CHARS].replace("\n", " ")
        log = self._node.get_logger()
        if not self._enabled():
            log.warning(f"bridge kick asked for ({reason}) and bridge_kick is off: ignored")
            return
        now = self._now()
        if self._last is not None and now - self._last < self._cooldown_s:
            log.warning(
                f"bridge kick asked for ({reason}) {now - self._last:.0f} s after the last one:"
                f" ignored, the board's bridge needs {self._cooldown_s:.0f} s between restarts"
            )
            return
        self._last = now
        try:
            self._flag.parent.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._flag.write_text(f"{stamp} {reason}\n")
        except OSError as exc:
            log.error(f"bridge kick: {self._flag} could not be written ({exc}): the bridge stays")
            return
        self.kicks += 1
        log.error(
            f"bridge kick: {reason}; wrote {self._flag} — the board's systemd restarts"
            " pepin-bridge (journalctl -u pepin-bridge-kick)"
        )
