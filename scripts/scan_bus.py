#!/usr/bin/env python
"""List every motor answering on a Feetech servo bus.

Probes the port at every supported baud rate and prints the IDs that answer.
Use this to verify what is actually on a bus before touching anything.

Usage:
    uv run python scripts/scan_bus.py --port /dev/tty.usbmodemXXXX
"""

import argparse
import logging

from lerobot.motors.feetech import FeetechMotorsBus

from pepin.log import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a Feetech bus for motors.")
    parser.add_argument("--port", required=True, help="serial port of the servo bus board")
    args = parser.parse_args()
    setup_logging("scan_bus")

    found = FeetechMotorsBus.scan_port(args.port)
    if not found:
        logger.info("No motors found. Check 12V power to the board and the servo cable.")
        return
    for baudrate, ids in found.items():
        logger.info("baud %s: IDs %s", baudrate, sorted(ids))


if __name__ == "__main__":
    main()
