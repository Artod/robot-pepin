"""The cart's one speed: config/base.json's ``max_wheel_speed_m_s``, and every place that holds it.

The base server slows any twist that would run a wheel faster (:func:`pepin.base.wheel_ceiling`),
and Nav2 is given the same number for each parameter of :data:`NAV2_SPEED` at launch
(ros/pepin_bringup/launch/nav.launch.py), so no controller plans a speed the wheels refuse.
``ros/speed.sh`` prints every place and sets them all live; every start reads the file again.

Stdlib only: the board's base server imports :func:`check_speed`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# What ros/speed.sh and the base server's ``max_wheel_speed`` command accept, m/s: (low, high].
# The top is the C++ bridge's own clamp (pepin.deployment.BASE_MAX_LINEAR_M_S), which nothing
# passes; the bottom is a cart that still overcomes its casters.
SPEED_RANGE_M_S = (0.05, 0.45)


def check_speed(value: object) -> float:
    """``value`` as a speed inside :data:`SPEED_RANGE_M_S`; ``ValueError`` saying why not."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"not a speed: {value!r}")
    try:
        speed = float(value)
    except ValueError:
        raise ValueError(f"not a number: {value!r}") from None
    low, high = SPEED_RANGE_M_S
    if not low < speed <= high:  # NaN fails here too
        raise ValueError(f"{speed} m/s is outside ({low}, {high}]")
    return speed


@dataclass(frozen=True)
class SpeedParam:
    """One Nav2 parameter that is the cart's speed: ``sign`` -1 for a reverse limit, ``index``
    the linear element of an [x, y, theta] array (None: a scalar)."""

    node: str
    name: str
    sign: float = 1.0
    index: int | None = None

    @property
    def label(self) -> str:
        """``node name`` as ros2 param names it, an array's element appended."""
        return f"{self.node} {self.name}" + ("" if self.index is None else f"[{self.index}]")

    def value(self, speed: float, current: Any = None) -> Any:
        """What the parameter becomes at ``speed``; an array keeps its other elements from
        ``current``."""
        if self.index is None:
            return self.sign * speed
        out = [float(x) for x in current]
        out[self.index] = self.sign * speed
        return out

    def held(self, current: Any) -> float:
        """The signed number ``current`` holds for this parameter (an array's linear element)."""
        return float(current if self.index is None else current[self.index])


# Every controller's linear limit and the velocity smoother's: a cap left out of this table is
# a second speed that wins silently (every tape until 2026-09-09 sat at 0.20 on such a number).
# The RPPs have no reverse limit of their own (reversing runs at desired_linear_vel).
NAV2_SPEED: tuple[SpeedParam, ...] = (
    SpeedParam("controller_server", "FollowPathMPPI.vx_max"),
    SpeedParam("controller_server", "FollowPathMPPI.vx_min", -1.0),
    SpeedParam("controller_server", "FollowPath.desired_linear_vel"),
    SpeedParam("controller_server", "FollowPathRS.desired_linear_vel"),
    SpeedParam("controller_server", "FollowPathShim.desired_linear_vel"),
    SpeedParam("controller_server", "FollowPathGraceful.v_linear_max"),
    SpeedParam("controller_server", "FollowPathDWB.max_vel_x"),
    SpeedParam("controller_server", "FollowPathDWB.min_vel_x", -1.0),
    SpeedParam("controller_server", "FollowPathDWB.max_speed_xy"),
    SpeedParam("velocity_smoother", "max_velocity", 1.0, 0),
    SpeedParam("velocity_smoother", "min_velocity", -1.0, 0),
)


def file_value(params: dict[str, Any], param: SpeedParam) -> Any:
    """``param``'s value in a parsed Nav2 params file; ``KeyError`` when the file lacks it."""
    node = params[param.node]["ros__parameters"]
    for key in param.name.split("."):
        node = node[key]
    return node


def nav2_overrides(speed: float, params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per node, the parameters that make Nav2 drive at ``speed``; ``params`` is the parsed Nav2
    params file, which an array's other elements are kept from."""
    out: dict[str, dict[str, Any]] = {}
    for param in NAV2_SPEED:
        current = None if param.index is None else file_value(params, param)
        out.setdefault(param.node, {})[param.name] = param.value(speed, current)
    return out


def _line(label: str, speed: float) -> str:
    return f"{label:<52} {speed:6.2f}"


def main(argv: Sequence[str] | None = None) -> int:
    """ros/speed.sh's laptop half: the range check, the file, and the base server's ceiling."""
    parser = argparse.ArgumentParser(description="The cart's one speed (config/base.json).")
    sub = parser.add_subparsers(dest="what", required=True)
    check = sub.add_parser("check", help="print X if it is a speed ros/speed.sh accepts")
    check.add_argument("speed")
    file = sub.add_parser("file", help="the speed config/base.json says")
    file.add_argument("--config", type=Path, default=Path(__file__).parents[2] / "config/base.json")
    base = sub.add_parser("base", help="the base server's wheel ceiling, set first with --set")
    base.add_argument("--host", required=True)
    base.add_argument("--port", type=int, required=True)
    base.add_argument("--set", dest="speed", type=float)
    args = parser.parse_args(argv)
    if args.what == "check":
        try:
            print(check_speed(args.speed))
        except ValueError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.what == "file":
        from pepin.geometry import BaseConfig

        speed = BaseConfig.from_json(args.config).max_wheel_speed_m_s
        print(_line("config/base.json max_wheel_speed_m_s (this checkout)", speed))
        return 0
    from pepin.base_link import ask

    message: dict[str, Any] = {"cmd": "max_wheel_speed"}
    if args.speed is not None:
        message["m_s"] = args.speed
    try:
        reply = ask(args.host, message, "max_wheel_speed", wait_s=3.0, port=args.port)
    except OSError as exc:
        print(f"base server {args.host}:{args.port}: {exc}")
        return 1
    if reply is None or "m_s" not in reply:
        print(f"base server {args.host}:{args.port}: no answer (an older pepin.base_server?)")
        return 1
    if "error" in reply:
        print(f"base server refused: {reply['error']}")
    was = f", was {reply['was_m_s']:.2f}" if "was_m_s" in reply else ""
    label = f"base server wheel ceiling (board file {reply['config_m_s']:.2f}{was})"
    print(_line(label, reply["m_s"]))
    return 1 if "error" in reply else 0


if __name__ == "__main__":
    sys.exit(main())
