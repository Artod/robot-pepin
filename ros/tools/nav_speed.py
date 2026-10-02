#!/usr/bin/env python3
"""Nav2's half of ros/speed.sh: every parameter of pepin.speed.NAV2_SPEED as Nav2 holds it, set
first with ``--set X``. One rclpy node, two services per Nav2 node; run inside pepin-macnav:

    docker exec pepin-macnav /pepin_entrypoint.sh python3 /tools/nav_speed.py [--set X]

One line per parameter, the signed number it holds last; exit 1 when a node did not answer or
refused a value (the reason is printed in place of the number).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from pepin.speed import NAV2_SPEED, SpeedParam, check_speed

WAIT_S = 5.0  # per service: discovery over rmw_zenoh is well under a second here


def _call(node: Any, srv_type: Any, name: str, request: Any) -> Any:
    """One service call, or None when the service is not there or does not answer in time."""
    import rclpy

    client = node.create_client(srv_type, name)
    try:
        if not client.wait_for_service(timeout_sec=WAIT_S):
            return None
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=WAIT_S)
        return future.result() if future.done() else None
    finally:
        node.destroy_client(client)


def _get(node: Any, target: str, names: list[str]) -> list[Any] | None:
    """The values of ``names`` on ``target`` as Python values, None when it did not answer."""
    from rcl_interfaces.srv import GetParameters
    from rclpy.parameter import parameter_value_to_python

    request = GetParameters.Request(names=names)
    reply = _call(node, GetParameters, f"/{target}/get_parameters", request)
    if reply is None:
        return None
    return [parameter_value_to_python(v) for v in reply.values]


def _set(node: Any, target: str, values: dict[str, Any]) -> dict[str, str] | None:
    """Set ``values`` on ``target``; the refusal reason per refused name, None when it did not
    answer."""
    from rcl_interfaces.srv import SetParameters
    from rclpy.parameter import Parameter

    messages = [Parameter(name, value=value).to_parameter_msg() for name, value in values.items()]
    request = SetParameters.Request(parameters=messages)
    reply = _call(node, SetParameters, f"/{target}/set_parameters", request)
    if reply is None:
        return None
    return {
        name: result.reason or "refused"
        for name, result in zip(values, reply.results, strict=True)
        if not result.successful
    }


def main() -> int:
    """Print (after setting, with ``--set``) every Nav2 speed parameter."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", dest="speed", type=float)
    args = parser.parse_args()
    speed = None if args.speed is None else check_speed(args.speed)

    import rclpy

    rclpy.init()
    node = rclpy.create_node("nav_speed")
    ok = True
    try:
        by_node: dict[str, list[SpeedParam]] = {}
        for param in NAV2_SPEED:
            by_node.setdefault(param.node, []).append(param)
        for target, params in by_node.items():
            names = [p.name for p in params]
            refused: dict[str, str] = {}
            if speed is not None:
                current = _get(node, target, names)  # the arrays' other elements are kept
                answer = None
                if current is not None:
                    pairs = zip(params, current, strict=True)
                    answer = _set(node, target, {p.name: p.value(speed, now) for p, now in pairs})
                if answer is None:
                    print(f"{target}: did not answer the set (is Nav2 up? ros/laptop.sh nav)")
                    ok = False
                    continue
                refused = answer
            held = _get(node, target, names)
            if held is None:
                print(f"{target}: did not answer (is Nav2 up? ros/laptop.sh nav)")
                ok = False
                continue
            for param, value in zip(params, held, strict=True):
                if param.name in refused:
                    print(f"{param.label:<52} REFUSED: {refused[param.name]}")
                    ok = False
                elif value is None:
                    print(f"{param.label:<52} NOT DECLARED")
                    ok = False
                else:
                    print(f"{param.label:<52} {param.held(value):6.2f}")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
