"""The live switches of the board's base bridge, ``/base_bridge``.

The node itself is C++ (ros/pepin_base_cpp, composed into robot.launch.py's base container):
/odom, odom -> base_link while no EKF owns it, the /cmd_vel sink, the MPU6050 as
/imu/data_raw and the rest /zupt. Its two sensor mutes are declared there by these names and
defaults; this table is their one description, read by ros/tools/flags_doc.py and
ros/flags.sh (``ros/sensor.sh mute imu|odom``), and a contract test holds the C++ to it. The
Python bridge that ran before 2026-09-06 is in git history.
"""

from __future__ import annotations

from pepin.flags import Flag, FlagSet

# The live switches (CLAUDE.md rule 19), printed in the bridge's link-up line. They mute a sensor
# where it is published, so a consumer sees what a dead sensor looks like — silence — without a
# restart and without losing any other live flag (ros/sensor.sh mute imu | mute odom).
FLAGS = FlagSet(
    Flag(
        "imu_publish",
        True,
        description="the MPU6050's readings leave the bridge as /imu/data_raw, where the EKF"
        " fuses index 11 (the yaw rate) and nothing else; off, the chip is still read and its"
        " bias still estimated, but no message is published",
        why="on, because the gyro is the heading: the wheels over-report a turn in place by"
        " 10-25 % on carpet, and odom0's vyaw — the only other yaw-rate source, live since"
        " 2026-09-15 — carries about 4 % of the weight beside it (ros/params/ekf.yaml)",
        on_when="always, unless the point of the run is what the stack does without a gyro",
        off_when="for one test of the heading on the wheels alone, or to see an EKF meet its"
        " sensor_timeout on a source that is simply gone; unmute and the rate is back within"
        " one IMU period (100 Hz)",
    ),
    Flag(
        "odom_publish",
        True,
        description="the base server's state line leaves the bridge as /odom and, while"
        " publish_tf is on, as the odom -> base_link transform; off, the wheels are still read"
        " and still commanded, and both go silent together — a transform still broadcast from a"
        " silent /odom is a state no sensor failure produces",
        why="on, because /odom is the only source of speed this filter has: odom0 fuses vx and"
        " vy at 0.001 (m/s)^2 and, since 2026-09-15, vyaw; ax and ay are off (a mount bias of"
        " -0.229 to +0.066 m/s^2 that no covariance can answer), so with /odom silent past the"
        " EKF's sensor_timeout of 0.5 s the filter has no velocity measurement left at all",
        on_when="always, unless the run is about what the stack does with dead wheel odometry",
        off_when="to watch a consumer meet a silent odometry — the EKF's sensor_timeout, Nav2's"
        " TF lookups, the tracker's dead reckoning — without stopping the base server; unmute"
        " and /odom is back on the next state line (50 Hz)",
    ),
)
