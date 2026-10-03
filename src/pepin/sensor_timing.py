"""When a board sensor's measurement happened, against the moment its driver dates it.

config/imu.json's ``timing`` block: the MPU6050's output rate and the group delay of its low-pass,
both handed to the C++ base bridge (robot.launch.py), which subtracts the delay from every
/imu/data_raw stamp. config/tof.json's ``timing`` block: the offset from the ToF server's line time
``t`` to the middle of the ranging windows that line reports, which the ToF bridge adds when it
carries ``t`` onto the ROS clock (:func:`measurement_lag_s`). Each block's note says how its
numbers were measured.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The MPU-6050 register map (rev 4.2), CONFIG 0x1A: the gyro's group delay by DLPF_CFG, seconds.
# mpu6050.hpp writes kDlpfConfig and carries the same table; config/imu.json's filter_delay_s is
# this table's entry for it (tests/unit/test_sensor_timing.py holds the three equal).
MPU6050_GYRO_DELAY_S: dict[int, float] = {
    0: 0.00098,
    1: 0.0019,
    2: 0.0028,
    3: 0.0048,
    4: 0.0083,
    5: 0.0134,
    6: 0.0186,
}

# A board line older than this when it arrives is dated on arrival: on the board the ToF line is
# ~25 ms old, and an older one comes from a stalled reader or another machine's monotonic clock.
# The base bridge's neck_stamp_max_age_s, the same number for the same reason.
LINE_MAX_AGE_S = 0.5


@dataclass(frozen=True)
class ImuTiming:
    """The MPU6050's output rate (how fresh the sample read is) and its filter's group delay
    (how long before the read the motion it describes happened)."""

    output_rate_hz: float
    filter_delay_s: float

    def bridge_parameters(self) -> dict[str, float]:
        """The C++ base bridge's parameters for these two numbers."""
        return {
            "imu_output_rate_hz": self.output_rate_hz,
            "imu_filter_delay_s": self.filter_delay_s,
        }


def _block(name: str, path: str | Path | None) -> dict[str, Any]:
    """The ``timing`` block of config/<name> (or of ``path``); ``KeyError`` when there is none."""
    from pepin.deployment import config_file  # lazy: the helpers below stay import-light

    source = Path(path) if path is not None else config_file(name)
    block = json.loads(source.read_text())["timing"]
    if not isinstance(block, dict):
        raise ValueError(f"{source}: timing is a mapping")
    return block


def imu_timing(path: str | Path | None = None) -> ImuTiming:
    """config/imu.json's ``timing`` block; ``KeyError``/``ValueError`` when it is missing or
    holds a rate that is not positive or a delay that is negative."""
    block = _block("imu.json", path)
    timing = ImuTiming(float(block["output_rate_hz"]), float(block["filter_delay_s"]))
    if not timing.output_rate_hz > 0.0 or not timing.filter_delay_s >= 0.0:
        raise ValueError(f"imu timing {timing}: a positive rate and a non-negative delay")
    return timing


def tof_timing_offset_s(path: str | Path | None = None) -> float:
    """config/tof.json's ``timing.offset_s``: added to the ToF line's ``t`` to date its readings
    (negative: the ranging windows lie before ``t``)."""
    offset = float(_block("tof.json", path)["offset_s"])
    if not math.isfinite(offset):
        raise ValueError(f"tof timing offset {offset}")
    return offset


def measurement_lag_s(
    now_mono_s: float, t: object, offset_s: float, max_age_s: float = LINE_MAX_AGE_S
) -> float | None:
    """How long before now a board line's measurement happened: the line's age on the board's
    monotonic clock (``now_mono_s - t``) less ``offset_s``, to be subtracted from the ROS clock
    read at the same moment. ``None`` — date it on arrival — when ``t`` is not a number or the
    age is outside ``0..max_age_s``."""
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
        return None
    age = now_mono_s - float(t)
    if not 0.0 <= age <= max_age_s:
        return None
    return age - offset_s
