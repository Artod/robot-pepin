#!/usr/bin/env python
"""Gentle single-motor jog for hardware bring-up.

Wheel test — switches the servo to velocity mode (persists in EEPROM, which is
what we want for wheels), spins it, stops:

    uv run python scripts/jog.py wheel --port /dev/tty.usbmodemXXXX --id 7

Neck test — position mode, a small move away and back:

    uv run python scripts/jog.py neck --port /dev/tty.usbmodemXXXX --id 9

Values are deliberately small and capped. The motor is always stopped and its
torque released on exit, including on Ctrl+C.
"""

import argparse
import sys
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

MODEL = "sts3215"
STEPS_PER_DEG = 4096.0 / 360.0
MAX_WHEEL_DEGPS = 360.0
MAX_NECK_DELTA = 300  # raw steps, ~26 degrees
NECK_PROFILE_SPEED = 400  # raw steps/s, ~35 deg/s


def jog_wheel(bus: FeetechMotorsBus, degps: float, seconds: float) -> None:
    raw = round(degps * STEPS_PER_DEG)
    bus.disable_torque()
    bus.write("Operating_Mode", "target", OperatingMode.VELOCITY.value)
    bus.enable_torque()
    try:
        print(f"Spinning at {degps} deg/s ({raw} raw) for {seconds} s...")
        bus.write("Goal_Velocity", "target", raw, normalize=False)
        time.sleep(seconds)
    finally:
        bus.write("Goal_Velocity", "target", 0, normalize=False)
        bus.disable_torque()
        print("Stopped, torque released. Servo stays in velocity mode.")


def jog_neck(bus: FeetechMotorsBus, delta: int) -> None:
    bus.disable_torque()
    bus.write("Operating_Mode", "target", OperatingMode.POSITION.value)
    bus.enable_torque()
    start = bus.read("Present_Position", "target", normalize=False)
    goal = start + delta
    print(f"Present position {start}, moving to {goal} and back...")
    try:
        # In position mode Goal_Velocity acts as the profile (max) speed.
        bus.write("Goal_Velocity", "target", NECK_PROFILE_SPEED, normalize=False)
        bus.write("Goal_Position", "target", goal, normalize=False)
        time.sleep(1.5)
        bus.write("Goal_Position", "target", start, normalize=False)
        time.sleep(1.5)
        end = bus.read("Present_Position", "target", normalize=False)
        print(f"Back at {end} (started at {start}).")
    finally:
        bus.disable_torque()
        print("Torque released.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gentle single-motor jog.")
    parser.add_argument("command", choices=["wheel", "neck"])
    parser.add_argument("--port", required=True, help="serial port of the servo bus board")
    parser.add_argument("--id", type=int, required=True, help="motor ID to jog")
    parser.add_argument(
        "--degps", type=float, default=90.0, help="wheel: speed in deg/s, sign sets direction"
    )
    parser.add_argument("--seconds", type=float, default=2.0, help="wheel: spin duration")
    parser.add_argument(
        "--delta", type=int, default=100, help="neck: move size in raw steps (4096 per turn)"
    )
    args = parser.parse_args()

    if abs(args.degps) > MAX_WHEEL_DEGPS:
        sys.exit(f"Refusing wheel speed above {MAX_WHEEL_DEGPS} deg/s.")
    if abs(args.delta) > MAX_NECK_DELTA:
        sys.exit(f"Refusing neck delta above {MAX_NECK_DELTA} steps.")

    bus = FeetechMotorsBus(
        port=args.port,
        motors={"target": Motor(id=args.id, model=MODEL, norm_mode=MotorNormMode.RANGE_M100_100)},
    )
    bus.connect()
    try:
        if args.command == "wheel":
            jog_wheel(bus, args.degps, args.seconds)
        else:
            jog_neck(bus, args.delta)
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
