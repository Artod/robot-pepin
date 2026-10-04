"""OpenVINS's restart button: ``/vio/restart`` (std_srvs/Trigger), in its own container.

OpenVINS has no reset service and never re-initialises by itself once it diverges (2026-10-04:
the first fast head pan threw it metres off for good). pepin_bringup.visual_odometry's guard asks
for a restart once its samples have been implausible for a while with the wheels at rest
(``vio_restart_rejects``); this node, started by vio.launch.py in ``pepin-vio`` beside OpenVINS,
answers by sending OpenVINS's process the launch's own SIGINT, as ``ros/laptop.sh vio kick``
does by hand. The launch respawns it 2 s later, and it initialises again at rest on the next
motion (a head pan).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence

from rclpy.node import Node
from std_srvs.srv import Trigger

from pepin_bringup.node_kit import spin_main

RESTART_SERVICE = "/vio/restart"
PROCESS = "run_subscribe_msckf"  # OpenVINS's executable, matched on the command line
SIGNAL = ("pkill", "-INT", "-f", PROCESS)


def _run(command: Sequence[str]) -> int:
    """Run ``command`` and answer its exit code (pkill: 0 signalled, 1 nothing matched)."""
    return subprocess.run(list(command), check=False, timeout=5).returncode


class VioKeeper(Node):
    """Serves ``/vio/restart``: SIGINT to OpenVINS, which vio.launch.py respawns."""

    def __init__(self, run: Callable[[Sequence[str]], int] = _run) -> None:
        super().__init__("vio_keeper")
        self._run = run
        self._restarts = 0
        self.create_service(Trigger, RESTART_SERVICE, self._on_restart)
        self.get_logger().info(
            f"vio keeper up: {RESTART_SERVICE} sends {PROCESS} SIGINT, the launch respawns it"
            " in 2 s"
        )

    def _on_restart(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        code = self._run(SIGNAL)
        response.success = code == 0
        if response.success:
            self._restarts += 1
            response.message = (
                f"{PROCESS} signalled ({self._restarts} since the start): respawned in 2 s, it"
                " initialises at rest on the next motion"
            )
        else:
            response.message = f"no {PROCESS} process to signal (pkill exit {code})"
        self.get_logger().info(f"restart asked: {response.message}")
        return response


def main() -> None:
    spin_main(VioKeeper)


if __name__ == "__main__":
    main()
