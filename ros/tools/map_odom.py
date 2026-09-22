#!/usr/bin/env python3
"""Is anybody correcting the pose: read ``map -> odom`` out of TF and print it on one line.

Under PEPIN_LOCALIZER=rtabmap that edge is the whole contract between the two halves — the
laptop's RTAB-Map broadcasts it, every consumer composes it with the board's own
``odom -> base_link`` — and two things about it are worth asking after a restart, neither of
which any topic answers:

* IS IT FRESH. The transform is re-broadcast at 20 Hz (vslam.launch.py's ``tf_delay``), so a
  stamp older than a second or two means the publisher is gone, not that the graph is quiet.
* IS IT STILL THE IDENTITY. A localiser that has recognised nothing yet publishes
  ``map == odom``, which composes into a pose that is simply the odometry's — the shape of
  start-up lie the board's own tracker was fixed for on 2026-09-21 (identity for 9.5 s, then a
  2.9 m jump). The identity is correct for the first seconds of a start and a fault after them.

    python3 /tools/map_odom.py [seconds=5] [--identity-m 0.01] [--identity-deg 0.5]

Prints one line — the translation, the heading, the stamp's age and ``identity``/``corrected`` —
and exits 0 when the edge was read and is not the identity, 1 when it is the identity or too
old to trust, 2 when nothing published it at all. Run it where the publisher is (the laptop's
container): reading TF costs a /tf subscription, which is ~100 messages a second on the board,
and CLAUDE.md rule 20 keeps that off it.
"""

from __future__ import annotations

import math
import sys
import time

import rclpy
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener

MAP_FRAME = "map"
ODOM_FRAME = "odom"
FRESH_S = 2.0  # the acceptance bar: 20 Hz means two seconds is forty missed broadcasts


def _yaw_deg(rotation: object) -> float:
    """The heading of a quaternion, in degrees, in the plane the whole stack lives in."""
    x = float(getattr(rotation, "x", 0.0))
    y = float(getattr(rotation, "y", 0.0))
    z = float(getattr(rotation, "z", 0.0))
    w = float(getattr(rotation, "w", 1.0))
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def main() -> int:
    """Wait up to ``seconds`` for the edge, then print it and say what it is worth."""
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    options = {a.split("=")[0]: a.split("=")[-1] for a in sys.argv[1:] if a.startswith("--")}
    seconds = float(argv[0]) if argv else 5.0
    identity_m = float(options.get("--identity-m", 0.01))
    identity_deg = float(options.get("--identity-deg", 0.5))
    rclpy.init()
    node = rclpy.create_node("pepin_map_odom")
    buffer = Buffer()
    TransformListener(buffer, node)
    try:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
            if buffer.can_transform(MAP_FRAME, ODOM_FRAME, rclpy.time.Time(), Duration(seconds=0)):
                break
        else:
            print(
                f"{MAP_FRAME} -> {ODOM_FRAME}: nobody publishes it"
                f" (nothing in TF in {seconds:.0f} s)"
            )
            return 2
        transform = buffer.lookup_transform(MAP_FRAME, ODOM_FRAME, rclpy.time.Time())
        t = transform.transform.translation
        stamp = transform.header.stamp
        age = node.get_clock().now().nanoseconds * 1e-9 - (stamp.sec + stamp.nanosec * 1e-9)
        yaw = _yaw_deg(transform.transform.rotation)
        shift = math.hypot(t.x, t.y)
        identity = shift <= identity_m and abs(yaw) <= identity_deg
        what = "identity (nothing has corrected the pose yet)" if identity else "corrected"
        print(
            f"{MAP_FRAME} -> {ODOM_FRAME}: ({t.x:+.3f}, {t.y:+.3f}) m, {yaw:+.1f} deg,"
            f" |shift| {shift:.3f} m, stamped {age:.2f} s ago: {what}"
        )
        return 1 if identity or age > FRESH_S else 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
