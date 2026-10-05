"""OpenVINS's restart button in pepin-vio: /vio/restart signals the process the launch respawns."""

from __future__ import annotations

from collections.abc import Sequence

import ros_stubs

ros_stubs.install()

from pepin_bringup.vio_keeper import RESTART_SERVICE, VioKeeper  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402


def test_a_restart_sends_openvins_the_launchs_sigint_and_says_when_nothing_ran() -> None:
    ran: list[list[str]] = []
    codes = [0, 1]

    def run(command: Sequence[str]) -> int:
        ran.append(list(command))
        return codes.pop(0)

    face: list[str] = []

    class Face:
        def event(self, name: str, *, end: bool = False) -> None:
            face.append(name)

        def clear(self) -> None: ...
        def lease(self, seconds: float) -> None: ...
        def close(self) -> None: ...

    node = VioKeeper(run=run, face=Face())
    _, serve = node.services[RESTART_SERVICE]
    ok = serve(Trigger.Request(), Trigger.Response())
    assert ok.success and "signalled (1 since the start)" in ok.message
    assert ran == [["pkill", "-INT", "-f", "run_subscribe_msckf"]], "SIGINT, never -9"
    none = serve(Trigger.Request(), Trigger.Response())
    assert not none.success and "no run_subscribe_msckf process" in none.message
    assert any("vio keeper up" in t for t in node.get_logger().texts("info"))
    assert face == ["vio_restart"]  # the restart, not the miss, on the head's face
