"""Does a source agree with ITS OWN previous word? A sensor's repeatability, measured live.

Every source arrives with a covariance somebody computed — off a score surface, or out of a
formula over the fit. On 2026-09-13 the camera's measurements claimed 25 cm by such a formula
while nobody had ever measured how far apart two of its answers fall a tenth of a second apart;
fused on that claim they took 20-45 % of the weight and pulled the board's pose 0.8-1.5 cm off
the lidar's. The claim was never checked against anything the source itself does.

This is that check, and it is deliberately blind to the other sensors. For one source: the
previous measurement carried over the odometry (wheels + gyro, the same trail a measurement
rides to an update) is a PREDICTION of the current one; the normalised squared distance
between the two under the sum of their covariances (:func:`pepin.fusion.disagreement`) is a
chi-square of 3 degrees of freedom whose mean is 1 when the covariance tells the truth, 16 when
the source scatters four times as far as it claims. The running mean over the last
:data:`WINDOW` measurements is the ratio ``r``; at ``r > 1`` the covariance is multiplied by
``r`` (capped at :data:`MAX_INFLATION`) before it is fused, which makes the fused weight the
one the source's OWN repeatability earns. At ``r <= 1`` nothing is touched: a source that is
better than it claims is not rewarded, only the over-claim is taken back.

Blind on purpose: no lidar pose ever enters the camera's ratio and no camera pose the lidar's,
so this can never become "the camera is judged by the sensor it was supposed to check". The
only outside input is the odometry, which both share anyway and which is millimetres over the
tenths of a second between two measurements of one source.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pepin.fusion import PoseMeasurement, carried, disagreement
from pepin.odometry import Pose2D
from pepin.scanmatch import relative_motion

__all__ = ["DOF", "MAX_GAP_S", "MAX_INFLATION", "WINDOW", "SelfCheck", "SourceRecord"]

# How many of a source's own measurements the ratio is averaged over. 20 is two seconds of a
# 10 Hz lidar and four of a 5 Hz camera: long enough that one bad match does not inflate a
# healthy source (a single chi-square of 3 dof sits anywhere between 0.1 and 3), short enough
# that a source going bad is answered within one leg of a drive.
WINDOW = 20
# The most a covariance is ever widened. A sigma five times the claimed one is already "this
# source measures nothing here"; past that the fusion's own disagreement gate is the right
# tool, and an unbounded factor would let one wild sample switch a sensor off for 20 updates.
MAX_INFLATION = 25.0
# A gap longer than this between two of a source's measurements is not a carry any more: the
# odometry trail is 5 s deep, a camera measurement is 0.1-0.3 s old, and a source that went
# away for two seconds starts its record over instead of being judged across the hole.
MAX_GAP_S = 2.0
DOF = 3.0  # x, y, yaw: the degrees of freedom the squared distance is normalised by


@dataclass
class SourceRecord:
    """One source's own record: how far its last measurements fell from what its previous one
    predicted (each normalised, so 1.0 is an honest covariance), the running ratio over them,
    the factor that ratio asks for, and the measurement the next one is predicted from with
    the odometry pose it was measured at."""

    ratio: float = 1.0
    inflation: float = 1.0
    samples: int = 0
    previous: PoseMeasurement | None = None
    odom: Pose2D | None = None
    history: deque[float] = field(default_factory=lambda: deque(maxlen=WINDOW))

    def text(self) -> str:
        """``lidar r 0.42 x1.00`` — what the report line shows for this source."""
        return f"r {self.ratio:.2f} x{self.inflation:.2f}"


class SelfCheck:
    """Every source's own repeatability, kept per source and never across sources.

    :meth:`checked` takes one measurement and the odometry pose it was measured at and returns
    it with its covariance widened by what that source's own last :data:`WINDOW` measurements
    say about it. ``enabled`` is the ``self_check`` flag: off, the ratio is still measured and
    reported (so the operator sees what the switch would do) and no covariance is touched.

    Pure: no clock, no ROS, no map. A fusion calls it, a report line reads it.
    """

    def __init__(
        self,
        window: int = WINDOW,
        max_inflation: float = MAX_INFLATION,
        max_gap_s: float = MAX_GAP_S,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.max_inflation = max_inflation
        self.max_gap_s = max_gap_s
        self._window = window
        self._records: dict[str, SourceRecord] = {}

    def checked(
        self, measurement: PoseMeasurement, odom: Pose2D, trust_odometry: bool = True
    ) -> PoseMeasurement:
        """``measurement`` with its covariance multiplied by its own source's inflation factor.

        ``odom`` is where the wheels say the cart was at the moment this measurement was
        measured (the odom frame; only differences between two of them are ever used), and it
        is the ONLY thing this brings in from outside the source. A sample that is not finite —
        a covariance that arrived NaN over the link (``NaN`` is a literal Python's json reads) —
        is dropped instead of recorded: averaged in, one of them would multiply that source's
        covariance by NaN for a whole window of updates, and the fused pose with it.
        ``trust_odometry`` False —
        a slipping wheel, an untrusted step — records no sample: the prediction would be wrong
        for a reason that has nothing to do with the sensor. The measurement is returned
        unchanged while the flag is off, for the first measurement of a source, after a gap
        longer than :attr:`max_gap_s`, and whenever the ratio is at or below 1.
        """
        record = self._records.setdefault(
            measurement.source, SourceRecord(history=deque(maxlen=self._window))
        )
        previous, previous_odom = record.previous, record.odom
        record.previous, record.odom = measurement, odom
        if previous is None or previous_odom is None or not trust_odometry:
            return measurement
        gap = measurement.stamp - previous.stamp
        if not 0.0 <= gap <= self.max_gap_s:
            record.history.clear()  # judged across a hole is not judged
            record.ratio = record.inflation = 1.0
            record.samples = 0
            return measurement
        predicted = carried(previous, relative_motion(previous_odom, odom), measurement.stamp)
        sample = disagreement(predicted, measurement) / DOF
        if not np.isfinite(sample):
            return measurement  # a NaN covariance is not evidence; recorded, it is 20 of them
        record.history.append(sample)
        record.samples = len(record.history)
        record.ratio = sum(record.history) / len(record.history)
        record.inflation = min(max(record.ratio, 1.0), self.max_inflation)
        if not self.enabled or record.inflation <= 1.0:
            return measurement
        return PoseMeasurement(
            measurement.x,
            measurement.y,
            measurement.yaw,
            np.asarray(measurement.covariance, dtype=float) * record.inflation,
            measurement.source,
            measurement.stamp,
            measurement.fit,
            rejected=measurement.rejected,
            edge=measurement.edge,
        )

    def record(self, source: str) -> SourceRecord:
        """That source's record — an empty one (ratio 1, nothing seen) for a source never
        measured, so a report never has to ask whether it exists."""
        return self._records.get(source, SourceRecord())

    def ratio(self, source: str) -> float:
        """The running ``mean(d^2)/dof`` of that source: 1.0 when its covariance is honest."""
        return self.record(source).ratio

    def inflation(self, source: str) -> float:
        """The factor that source's covariance is multiplied by (1.0 when it is honest, and
        1.0 for everything while the flag is off)."""
        return self.record(source).inflation if self.enabled else 1.0

    def status(self) -> dict[str, list[float]]:
        """Every source seen, as ``{name: [ratio, inflation]}``, for a JSON report."""
        return {
            name: [round(r.ratio, 3), round(r.inflation if self.enabled else 1.0, 3)]
            for name, r in self._records.items()
        }

    def text(self) -> str:
        """``self_check on: lidar r 0.42 x1.00, depth r 16.20 x16.20`` for a report line; the
        factors are the ones the ratios ask for, applied only while the flag is on."""
        body = ", ".join(f"{name} {r.text()}" for name, r in sorted(self._records.items()))
        return f"self_check {'on' if self.enabled else 'off'}: {body or 'nothing seen'}"

    def forget(self) -> None:
        """Drop every record: what a new map means for measurements made on the old one."""
        self._records.clear()
