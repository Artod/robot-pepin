#!/usr/bin/env python
"""Calibrate the neck and head servos (IDs 9 and 10).

The arm has a lerobot calibration preset; the neck does not, so we run the
same two-phase procedure ourselves:

1. Set the head by hand to its neutral pose (looking straight ahead, level).
   The script writes a homing offset into each servo so that this pose reads
   as the encoder center (2048). The offset persists inside the servo.
2. Move neck and head gently through their full SAFE range while the script
   records min/max encoder values. Press Enter to finish. Results are saved
   to config/neck.json for the future head driver.

Torque stays disabled the whole time — the head is moved by hand only.

Usage:
    uv run python scripts/calibrate_neck.py --port /dev/tty.usbmodemXXXX
"""

import argparse
import json
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

MODEL = "sts3215"
MOTORS = {"neck": 9, "head": 10}
OUTPUT = Path(__file__).resolve().parent.parent / "config" / "neck.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Record neck/head homing and ranges.")
    parser.add_argument("--port", required=True, help="serial port of the servo bus board")
    args = parser.parse_args()

    bus = FeetechMotorsBus(
        port=args.port,
        motors={
            name: Motor(id=id_, model=MODEL, norm_mode=MotorNormMode.RANGE_M100_100)
            for name, id_ in MOTORS.items()
        },
    )
    bus.connect()
    bus.disable_torque()

    input(
        "Set the head to its NEUTRAL pose by hand (looking straight ahead, level), "
        "then press Enter..."
    )
    homings = bus.set_half_turn_homings()
    print(f"Homing offsets written: {homings}")

    print(
        "Now move neck and head gently through their full SAFE range in all "
        "directions. Do not strain against hard stops. Press Enter when done."
    )
    mins, maxes = bus.record_ranges_of_motion()

    data = {
        name: {"id": id_, "center": 2048, "min": int(mins[name]), "max": int(maxes[name])}
        for name, id_ in MOTORS.items()
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Saved to {OUTPUT}")
    bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
