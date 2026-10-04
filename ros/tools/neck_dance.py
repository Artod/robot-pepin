#!/usr/bin/env python3
"""The head's calibration dance (docs/head_imu_calibration.md, procedure D): pans and tilts that
excite both of the IMU's rotation axes in front of the AprilGrid, each view held for a second of
still frames, asked of the gaze arbiter (the neck's one owner) as one operator-band scan.

    uv run python ros/tools/neck_dance.py              # prints the request; nothing moves
    uv run python ros/tools/neck_dance.py --move       # the arbiter moves the head; wheels never

The arbiter (pepin_bringup.gaze, its door on 127.0.0.1:3339, pepin.gaze_link) keeps its own speed
and acceleration caps (the tilt's 600 deg/s^2: the mast rings above it), refuses a view outside the
neck's reach, and sends the head home when the scan ends. Tilt is the pitch below level (home 23.8
deg down), pan positive left.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any

DOOR = "http://127.0.0.1:3339"
HOME_PAN_DEG, HOME_TILT_DEG = 0.0, 23.8
FRAMES_PER_VIEW = 10  # still frames after each settle: a second at the camera's 10 Hz
OPERATOR_BAND = 0  # pepin.gaze.OPERATOR: nothing else takes the head while it dances


def views() -> list[tuple[float, float]]:
    """(pan, tilt) in degrees: the pan's +-45, the tilt from 18 up to 60 down, the diagonals,
    home between groups: two rotation axes excited separately and together."""
    home = (HOME_PAN_DEG, HOME_TILT_DEG)
    pans = [(pan, HOME_TILT_DEG) for pan in (45.0, -45.0, 20.0, -20.0)]
    tilts = [(HOME_PAN_DEG, tilt) for tilt in (-18.0, 60.0, 0.0, 45.0)]  # reach: 20 up, 63 down
    diagonals = [(30.0, 0.0), (-30.0, 45.0), (30.0, 45.0), (-30.0, 0.0)]
    return [home, *pans, home, *tilts, home, *diagonals, home]


def request() -> dict[str, Any]:
    """The arbiter's look: one scan of every view, held FRAMES_PER_VIEW frames each."""
    return {
        "kind": "scan",
        "source": "calibration",
        "band": OPERATOR_BAND,
        "frames": FRAMES_PER_VIEW,
        "speed": "saccade",
        "ttl_s": 180.0,
        "target": {
            "views": [
                {"pan_rad": math.radians(pan), "tilt_rad": math.radians(tilt)}
                for pan, tilt in views()
            ]
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Print the request; with --move, ask the arbiter and wait for the scan to end."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--move", action="store_true", help="ask the arbiter (default: print)")
    parser.add_argument("--door", default=DOOR)
    args = parser.parse_args(argv)
    for pan, tilt in views():
        print(f"pan {pan:+6.1f} deg, tilt {tilt:5.1f} deg down, {FRAMES_PER_VIEW} still frames")
    if not args.move:
        print("nothing moved (--move asks the gaze arbiter)")
        return 0
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
    from pepin.gaze_link import ask

    state = ask(args.door, "/state")
    print(f"arbiter: {state.get('phase', state)}")
    answer = ask(args.door, "/look", request(), timeout_s=200.0)
    print(f"scan: {answer}")
    return 0 if answer.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
