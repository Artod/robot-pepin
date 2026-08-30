#!/usr/bin/env python
"""Assign a bus ID to a single Feetech STS3215 servo.

Connect exactly ONE servo to the bus board (nothing else on the bus), then:

    uv run python scripts/setup_motor_id.py --port /dev/tty.usbmodemXXXX --id 7

The script first scans every supported baud rate and refuses to write unless
exactly one motor answers, so it cannot silently reflash a servo of an
assembled arm. The connected servo then gets the requested ID and the bus
default baud rate (1 Mbps), whatever its current settings are.
"""

import argparse
import sys

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

MODEL = "sts3215"


def main() -> None:
    parser = argparse.ArgumentParser(description="Set the ID of the single connected servo.")
    parser.add_argument("--port", required=True, help="serial port of the servo bus board")
    parser.add_argument("--id", type=int, required=True, help="target motor ID (1-253)")
    args = parser.parse_args()

    # Safety guard: lerobot's setup_motor() writes to the first motor that
    # answers, so an extra motor on the bus could get reflashed by accident.
    # Require exactly one responder before writing anything.
    found = FeetechMotorsBus.scan_port(args.port)
    answers = [(baud, id_) for baud, ids in found.items() for id_ in ids]
    if len(answers) != 1:
        listing = ", ".join(f"ID {i} @ {b} baud" for b, i in answers) or "nothing"
        sys.exit(f"Expected exactly one motor on the bus, found: {listing}. Aborting.")

    initial_baudrate, initial_id = answers[0]
    print(f"Found one motor: ID {initial_id} @ {initial_baudrate} baud.")

    bus = FeetechMotorsBus(
        port=args.port,
        motors={"target": Motor(id=args.id, model=MODEL, norm_mode=MotorNormMode.RANGE_M100_100)},
    )
    bus.setup_motor("target", initial_baudrate=initial_baudrate, initial_id=initial_id)
    print(f"Motor is now ID {args.id} @ {bus.default_baudrate} baud.")

    verify = bus.broadcast_ping()
    if verify and args.id in verify:
        print("Verification ping: OK")
    else:
        print(f"Verification ping FAILED, bus answered: {verify}")
    bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
