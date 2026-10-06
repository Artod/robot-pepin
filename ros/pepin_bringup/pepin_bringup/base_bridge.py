"""The live switches of the board's base bridge, ``/base_bridge``.

The node itself is C++ (ros/pepin_base_cpp, composed into robot.launch.py's base container):
/odom, odom -> base_link while no EKF owns it, the /cmd_vel sink, the MPU6050 as
/imu/data_raw and the rest /zupt, and (head_imu:=true) the head IMU from head_server as
/head/imu with the mast-sway filter. Its three sensor mutes, /odom's stamp and the sway's
composition into the camera edge are declared there by these names and defaults; this table is
their one description, read by ros/tools/flags_doc.py and ros/flags.sh (``ros/sensor.sh mute
imu|odom``), and contract tests hold the C++ to it. The Python bridge that ran before 2026-09-06
is in git history.
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
    Flag(
        "odom_stamp",
        "encoder",
        choices=("encoder", "arrival"),
        description="what /odom and odom -> base_link are dated by: the state line's encoder read"
        " carried onto the ROS clock (`encoder`, the stamp /neck/state of the same line carries)"
        " or the moment the line reached the bridge (`arrival`); a line older than 0.5 s is dated"
        " on arrival either way",
        why="encoder, because arrival is not when the wheels were read: for the same state line"
        " /odom's arrival stamp ran p50 6.4 ms, p90 7.0, p99 8.9, max 26.4 ms after the encoder"
        " read on a parked cart (scratch/gaze/odom_vs_neck_stamp.py, 2026-10-02), so the EKF"
        " placed every wheel pose that late against the gyro and the head's transform",
        on_when="`encoder` always: it is the measurement's own time",
        off_when="`arrival` to compare against the old stamps, or if the EKF or a TF consumer"
        " reports extrapolation into the past after the switch; live, the next state line",
    ),
    Flag(
        "odom_covariance",
        "law",
        choices=("law", "constant"),
        description="what /odom's twist covariance says about vx and vyaw: `law` sizes both from"
        " the wheels' MEASURED twist while they move (pepin.wheel_noise, config/base.json"
        " odometry_noise: over 1 s sigma_v = 0.034 |w| + 0.026 m/s, sigma_w = 0.20 |w| + 0.038"
        " rad/s, 50 sigma^2 per 50 Hz sample) and keeps 0.001 / 0.01 at rest; `constant` is"
        " 0.001 / 0.01 on every sample, as before 2026-10-06",
        why="law, because the constant was 3.7x (vx) and 2.7x (vyaw) too sure of a moving"
        " second and its error is not constant: against the lidar truth of drives 0329-0347"
        " (479 moving seconds, scratch/wheel_law/law2.py) the law's likelihood ratio over one"
        " constant is 25 (vx) and 169 (vyaw), and its per-bin mean z^2 stays 0.73-1.25 / 0.80-1.19"
        " where the best constant's runs 0.71-1.66 / 0.24-2.93; at rest the wheels' error is"
        " under the truth's own noise, so rest keeps the constant. Not yet replayed through the"
        " EKF: while moving, the VIO's twist now outweighs the wheels' vx ~40:1 instead of ~1:1",
        on_when="always, once a replay or a drive shows the EKF's distance no worse with it",
        off_when="the moment the EKF's speed follows a wrong VIO (a reset, a dark scene) more"
        " than it did, or Nav2's speed tracking gets worse; live, the next state line (50 Hz)",
    ),
    Flag(
        "head_imu_publish",
        True,
        description="the head IMU's samples (head_server's TCP 3340 stream, under the launch's"
        " head_imu:=true) leave the bridge as /head/imu in the chip's axes, dated by the sample's"
        " own moment on the board's clock; off, the link, the counters and the mast filter keep"
        " running and nothing is published",
        why="on by design, unmeasured: nothing exists to publish until the head is mounted and"
        " head_imu:=true; the VIO (pepin-vio) is its only consumer and the mute is how its"
        " behaviour without the IMU is seen without restarting the board",
        on_when="always once the head IMU is calibrated into config/camera.json",
        off_when="to watch OpenVINS meet a silent IMU (it drops images newer than its last IMU),"
        " or if /head/imu's bandwidth is ever in the way of the board's WiFi; unmute and the"
        " next batch (20 ms) is published",
    ),
    Flag(
        "mast_sway",
        False,
        description="the mast's sway, from the head gyro minus the base gyro's yaw and the"
        " neck's joints (mast.hpp), is composed INTO base_link -> camera_link as a rotation about"
        " the mast's hinge, so every consumer of that edge gets the corrected camera; off, the"
        " edge is the neck's alone, exactly as before, and /mast/state publishes either way",
        why="off until measured (vio.md S6): at the shipped tilt cap (600 deg/s^2) the ring is"
        " <= 0.08 deg (2026-10-02), under the correction's own pass line; wheel jerks are"
        " unmeasured, and a wrong sign DOUBLES the error, so the sign check (the image's ring by"
        " phase correlation against /mast/state's pitch, scratch/gaze/tilt_sway.py) comes first",
        on_when="after the sign check passes and the S3 drives' bags show more than ~0.15 deg"
        " p-p of sway; then the tilt_sway pass line: >= 70 % of a 0.3-0.4 deg ring removed,"
        " lag <= 10 ms, theta at rest <= 0.02 deg rms",
        off_when="the moment the depth law or RTAB-Map's registrations get worse with it on, or"
        " /mast/state reads more than a degree at rest; live, the next neck line (50 Hz)",
    ),
)
