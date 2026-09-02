#!/usr/bin/env python
"""First motion of the base over the bridge: roll forward, then back, with odometry.

Drives at 0.1 m/s for the given time in each direction while integrating
wheel-encoder odometry, and prints how far the encoders think the cart went.
Clear about 0.5 m in front of the cart before running.

Usage:
    uv run python scripts/base_smoke.py --seconds 2
"""

import argparse
import logging
import time

from pepin.base import DiffDriveBase
from pepin.bus import verify_motors
from pepin.feetech import FeetechTcpClient
from pepin.geometry import BaseConfig
from pepin.kinematics import Twist
from pepin.log import setup_logging
from pepin.odometry import DiffDriveOdometry
from pepin.transport import DEFAULT_HOST, SERVO_BUS_PORT

logger = logging.getLogger(__name__)

CONFIG = "config/base.json"
SPEED = 0.1
LOOP_HZ = 20


def roll(base: DiffDriveBase, odom: DiffDriveOdometry, speed: float, seconds: float) -> None:
    base.set_twist(Twist(linear=speed, angular=0.0))
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        odom.update(*base.read_wheel_travel())
        time.sleep(1.0 / LOOP_HZ)
    base.stop()
    time.sleep(0.3)
    odom.update(*base.read_wheel_travel())
    p = odom.pose
    logger.info(
        "after %+.2f m/s for %ss: x=%+.3f m  y=%+.3f m  theta=%+.3f rad",
        speed,
        seconds,
        p.x,
        p.y,
        p.theta,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll the base forward and back.")
    parser.add_argument("--seconds", type=float, default=2.0, help="duration of each leg")
    args = parser.parse_args()
    setup_logging("base_smoke")

    cfg = BaseConfig.from_json(CONFIG)
    motors = DiffDriveBase.motor_ids(cfg)
    with FeetechTcpClient(DEFAULT_HOST, SERVO_BUS_PORT, motors) as bus:
        verify_motors(bus, list(motors))
        odom = DiffDriveOdometry(cfg.geometry)
        with DiffDriveBase(bus, cfg) as base:
            base.read_wheel_travel()  # prime the encoder unwrappers
            roll(base, odom, SPEED, args.seconds)
            roll(base, odom, -SPEED, args.seconds)
        logger.info("%s", bus.latency.summary())


if __name__ == "__main__":
    main()
