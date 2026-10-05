#!/usr/bin/env python3
"""Do the three ToF sensors reach the laptop? One line, OK or FAIL first.

Listens to /tof/front, /tof/left and /tof/right (sensor_msgs/Range from the board's tof_bridge,
~15 Hz each) for a few seconds and judges each sensor. SILENT: under ``MIN_HZ`` arrived (the
bridge, the board or the link is down). UNKNOWN: every reading was the bridge's "I do not know"
(range below zero): the sensor did not answer (i2c-2 locked, the sensor off the bus) or
something sits within 12 cm of its window. A locked bus keeps the topics flowing (the server
sends status 255, the bridge -1 m), so the rate alone would pass a dead sensor.

    python3 /tools/tof_check.py [seconds=2]

Prints "OK front 15.0 Hz, left 15.0 Hz, right 15.0 Hz" or, e.g., "FAIL right silent (0 in 2 s);
left unknown 30 of 30 (not answering, or < 12 cm)"; exit 1 on FAIL.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

NAMES = ("front", "left", "right")
MIN_HZ = 5.0  # of the bridge's ~15
FIRST_WAIT_S = 3.0  # for the first message of any sensor (discovery over the router)


@dataclass
class Tally:
    """What one sensor's topic delivered in the window."""

    count: int = 0
    unknown: int = 0  # readings below zero: the bridge's "I do not know"


def verdict(tallies: dict[str, Tally], seconds: float) -> tuple[bool, str]:
    """(passed, the line after OK/FAIL) for the three sensors' tallies over ``seconds``."""
    fails, oks = [], []
    for name in NAMES:
        t = tallies.get(name, Tally())
        hz = t.count / seconds
        if hz < MIN_HZ:
            fails.append(f"{name} silent ({t.count} in {seconds:.0f} s)")
        elif t.unknown == t.count:
            fails.append(f"{name} unknown {t.unknown} of {t.count} (not answering, or < 12 cm)")
        elif t.unknown:
            oks.append(f"{name} {hz:.1f} Hz ({100 * t.unknown // t.count}% unknown)")
        else:
            oks.append(f"{name} {hz:.1f} Hz")
    return (not fails, "; ".join(fails) if fails else ", ".join(oks))


def main() -> None:
    import rclpy
    from sensor_msgs.msg import Range

    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    rclpy.init()
    node = rclpy.create_node("pepin_tof_check")
    tallies = {name: Tally() for name in NAMES}
    counting = [False]

    def on_range(name: str, msg: Range) -> None:
        counting[0] = True
        tally = tallies[name]
        tally.count += 1
        if msg.range < 0.0:
            tally.unknown += 1

    try:
        for name in NAMES:
            node.create_subscription(Range, f"/tof/{name}", lambda m, n=name: on_range(n, m), 50)
        deadline = time.monotonic() + FIRST_WAIT_S
        while time.monotonic() < deadline and not counting[0]:
            rclpy.spin_once(node, timeout_sec=0.1)
        for tally in tallies.values():  # the window starts now
            tally.count = tally.unknown = 0
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        passed, line = verdict(tallies, seconds)
        print(("OK " if passed else "FAIL ") + line)
        if not passed:
            sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
