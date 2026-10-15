"""Write config/neck.json's tick limits into the neck servos themselves, as lerobot's calibration
does for the arm: Min_Position_Limit / Max_Position_Limit in each servo's EEPROM, so the servo
refuses to drive past them whatever it is sent. The order is lerobot's (FeetechMotorsBus):
torque off, Lock 0, the limits written and acknowledged one by one, Lock 1; then read back.

The bus has one owner, pepin-base, and ser2net hands the port to one client: stop it first.

    ssh root@pepin.local systemctl stop pepin-base
    uv run python scripts/neck_limits.py --host 10.0.0.187          # read: what the servos hold
    uv run python scripts/neck_limits.py --host 10.0.0.187 --write  # write the file's limits
    ssh root@pepin.local systemctl start pepin-base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pepin.feetech import FeetechTcpClient
from pepin.neck import NeckConfig

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config/neck.json"
LIMITS = ("Min_Position_Limit", "Max_Position_Limit")


def read_limits(bus: FeetechTcpClient, names: list[str]) -> dict[str, tuple[int, int]]:
    """Each servo's (min, max) as its EEPROM holds them."""
    lows = bus.sync_read(LIMITS[0], names, normalize=False)
    highs = bus.sync_read(LIMITS[1], names, normalize=False)
    return {name: (lows[name], highs[name]) for name in names}


def main(argv: list[str] | None = None) -> int:
    """Read, or write and read back; 0 when the servos hold the file's limits afterwards."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--host", default="10.0.0.187")
    parser.add_argument("--port", type=int, default=3333)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--write", action="store_true", help="write the file's limits")
    args = parser.parse_args(argv)
    cfg = NeckConfig.from_json(args.config)
    joints = {cfg.pan.name: cfg.pan, cfg.tilt.name: cfg.tilt}
    wanted = {name: (j.min_ticks, j.max_ticks) for name, j in joints.items()}
    bus = FeetechTcpClient(args.host, args.port, {name: j.motor_id for name, j in joints.items()})
    bus.connect()
    try:
        names = list(joints)
        before = read_limits(bus, names)
        for name in names:
            print(f"{name} (id {joints[name].motor_id}): servo {before[name]}  file {wanted[name]}")
        if not args.write:
            return 0 if before == wanted else 1
        for name in names:
            bus.write("Torque_Enable", name, 0)
            bus.write("Lock", name, 0)
            bus.write(LIMITS[0], name, wanted[name][0])
            bus.write(LIMITS[1], name, wanted[name][1])
            bus.write("Lock", name, 1)
        after = read_limits(bus, names)
        for name in names:
            print(f"{name}: written {wanted[name]}, reads back {after[name]}")
        return 0 if after == wanted else 1
    finally:
        bus.close()


if __name__ == "__main__":
    sys.exit(main())
