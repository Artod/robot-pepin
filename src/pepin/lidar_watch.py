"""The lidar's driver brought back when it says nothing: onto its port, or to idle without one.

The LD19's driver opens its serial port once. A lidar unplugged and plugged back, or plugged in
after the stack started, leaves a driver that never reads again; nothing in its parameters asks
it to retry. Worse, a driver that LOST its port while running spins at 100 % of a core, while one
STARTED without a port idles at 3 % (both measured on the board). So two rules, one per case:

* the port is PRESENT and no scan came for SILENT_S: end the driver so it respawns on the port
  that is there now, paced by GRACE_S (then PATIENT_S for a lidar that is plugged in and dead);
* the port is ABSENT and no scan came for SILENT_S: end the driver ONCE per absence, so it
  respawns without a port and idles; never again until the port is seen or a scan arrives.

This is the decision only: given whether the port exists and when the last scan arrived, say when
the driver's process should be ended so that the launch respawns it. Pure (no ROS, no clock of its
own), so it is tested in microseconds; the node that owns it supplies the time, the port check and
the ending of the process.
"""

from __future__ import annotations

from dataclasses import dataclass

SILENT_S = 5.0  # a scan every 0.1 s: this long without one is a driver that does not read
GRACE_S = 15.0  # a respawned driver needs this long to open the port and send its first scan
PATIENT_AFTER = 3  # kicks in a row that brought no scan before the pace drops
PATIENT_S = 60.0  # ... to one try a minute (a lidar that is plugged in and dead)


@dataclass
class LidarWatch:
    """Decides, once a tick, whether the silent lidar's driver is to be restarted."""

    started_at: float
    last_scan_at: float | None = None
    last_kick_at: float | None = None
    kicks_in_a_row: int = 0
    kicks: int = 0
    idled: bool = False  # this absence of the port has had its one kick (driver respawned idle)

    def scan(self, now: float) -> None:
        """A scan arrived: the lidar is alive and the streak of fruitless kicks is over."""
        self.last_scan_at = now
        self.kicks_in_a_row = 0
        self.idled = False

    def silent_s(self, now: float) -> float:
        """Seconds since the last scan, or since the start where none ever came."""
        return now - (self.started_at if self.last_scan_at is None else self.last_scan_at)

    def due(self, now: float, port_present: bool) -> bool:
        """True when the driver is to be ended now, after SILENT_S without a scan. Port present:
        the last kick was given its time (GRACE_S, or PATIENT_S after PATIENT_AFTER in a row).
        Port absent: once per absence, so the respawned driver idles instead of spinning. Counts
        the kick."""
        if port_present:
            self.idled = False
        if self.silent_s(now) < SILENT_S:
            return False
        if not port_present:
            if self.idled:
                return False
            self.idled = True
            self.kicks += 1
            return True
        hold = PATIENT_S if self.kicks_in_a_row >= PATIENT_AFTER else GRACE_S
        if self.last_kick_at is not None and now - self.last_kick_at < hold:
            return False
        self.last_kick_at = now
        self.kicks_in_a_row += 1
        self.kicks += 1
        return True

    def status(self, now: float, port_present: bool) -> str:
        """One phrase for the operator: alive, or silent for how long and what is being done."""
        silent = self.silent_s(now)
        if silent < SILENT_S:
            return "ok"
        if not port_present:
            done = ", driver restarted to idle" if self.idled else ""
            return f"silent {silent:.0f} s, no port (unplugged){done}"
        pace = (
            "one try a minute" if self.kicks_in_a_row >= PATIENT_AFTER else "restarting its driver"
        )
        return (
            f"silent {silent:.0f} s with its port present, {pace} ({self.kicks_in_a_row} in a row)"
        )
