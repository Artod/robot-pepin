#!/usr/bin/env python
"""List every motor answering on a Feetech servo bus.

Probes the port at every supported baud rate and prints the IDs that answer.
Use this to verify what is actually on a bus before touching anything.

Usage:
    uv run python scripts/scan_bus.py --port /dev/tty.usbmodemXXXX
"""

import argparse

from lerobot.motors.feetech import FeetechMotorsBus


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a Feetech bus for motors.")
    parser.add_argument("--port", required=True, help="serial port of the servo bus board")
    args = parser.parse_args()

    found = FeetechMotorsBus.scan_port(args.port)
    if not found:
        print("No motors found. Check 12V power to the board and the servo cable.")
        return
    for baudrate, ids in found.items():
        print(f"baud {baudrate}: IDs {sorted(ids)}")


if __name__ == "__main__":
    main()
