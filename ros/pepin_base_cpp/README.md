# pepin_base_cpp

The base bridge in C++: `/cmd_vel` down to the board's base server, its state stream up as
`/odom` and `odom -> base_link`, the MPU6050 as `/imu/data_raw`, the rest `/zupt`, and the neck's
encoders as `/neck/state` and `base_link -> camera_link`. It is the
board's only base bridge: the Python one it was ported from (same node name, parameters and wire
protocol) cost ~190 MB RSS and ~10% of a core, this one ~25 MB and ~1%, and RAM runs out first;
that node is in git history before 2026-10-02. Its live switches are described in
`pepin_bringup/base_bridge.py` (the `FLAGS` table ros/flags.sh reads).

- `include/pepin_base_cpp/protocol.hpp` — the wire format, no ROS and no sockets in it.
- `include/pepin_base_cpp/link.hpp` — reconnecting JSON-lines TCP client, one reader thread.
- `include/pepin_base_cpp/mpu6050.hpp` — the IMU on an i2c-dev bus, SI units, no ROS.
- `include/pepin_base_cpp/twist_from_pose.hpp` — the twist the wheels measured, off two poses.
- `include/pepin_base_cpp/gyro_bias.hpp` — the gyro's zero, and the wheels' word on rest.
- `include/pepin_base_cpp/zupt.hpp` — when the cart is certainly still, for the EKF's /zupt.
- `include/pepin_base_cpp/neck.hpp` — the neck's ticks as joint angles and the camera's pose.
- `src/base_bridge.cpp` — the node: /odom, TF, the /cmd_vel sink, the 5 Hz resend, the neck.

## Neck

The base server reads the two neck servos in the same sync_read as the wheels and puts
`pan_ticks`/`tilt_ticks` in the state line under the same stamp `t` (the board's monotonic clock,
the middle of that encoder read). Every line that carries them becomes `/neck/state`
(`sensor_msgs/JointState`, `neck_pan` positive left and `head_tilt` the pitch below level, in
radians) and `base_link -> camera_link`, both stamped with that read carried onto the ROS clock
(`now() - (monotonic now - t)`; a line older than `neck_stamp_max_age_s`, 0.5, is stamped on
arrival and counted), at up to `neck_publish_hz` (50: every line). `/odom` and `odom -> base_link`
of the same line carry the same stamp under `odom_stamp` "encoder" (the default since 2026-10-02;
live, `ros2 param set /base_bridge odom_stamp arrival` is the old stamp on arrival, which ran p50
6.4 ms, p99 8.9, max 26.4 ms after the read on a parked cart). `neck.hpp` is the twin of
`pepin.neck` (`test/neck_contract.cpp` holds it to the Python model). A line without the ticks (a
silent neck) publishes nothing: no edge is held or republished.

Every number is a parameter, handed over by robot.launch.py from `config/neck.json`
(`pepin.neck.bridge_parameters`: `neck_reference_pan_ticks`/`_tilt_ticks`, -1 unread,
`neck_pan_sign`/`neck_tilt_sign`, `neck_mount_x_m`/`_y_m`/`_z_m`/`neck_mount_pitch_deg`, the four
lever arms `neck_tilt_from_pan_*`/`neck_camera_from_tilt_*`) and `config/camera.json`
(`neck_camera_frame`, the active camera's link frame; empty publishes no transform), with
`neck_parent_frame` (base_link) and `neck_joint_names`: a re-measured mount is `ros/sync.sh` and a
restart of the base container, never a rebuild. No switch: a rig without neck servos publishes
nothing here and runs the laptop with `ros/laptop.sh vslam --fixed-head`. The report line, once a
minute: how many lines carried the ticks, how many went out, where the head points.

Board cost, AN ESTIMATE until a census: one JointState and one transform per state line, about
what the /odom beside them costs (+2-3 % of a core); it retired `pepin_bringup.neck_state`, a
Python process at 10 % and 66 MB.

## IMU

`imu_enable` (false), `imu_device` (/dev/i2c-2), `imu_address` (0x68), `imu_rate_hz` (50),
`imu_frame` (base_link), `imu_bias_s` (2.0), `imu_bias_tracking` (true): a thread samples an
MPU6050 and publishes `imu/data_raw` without orientation; a missing chip is one warning and the
wheels carry on. Registers PWR_MGMT_1 0x01, SMPLRT_DIV 1000/rate-1, CONFIG 0x03 (DLPF ~44 Hz),
GYRO_CONFIG 0x08 (+-500 dps), ACCEL_CONFIG 0x08 (+-4 g), WHO_AM_I 0x68, 14 bytes from
ACCEL_XOUT_H 0x3B. The EKF that fuses the yaw rate with /odom is robot_localization, configured
outside this package.

robot.launch.py passes `imu_rate_hz` 100 (`pepin.deployment.IMU_RATE_HZ`, since 2026-10-01): the
DLPF's ~44 Hz is under that rate's Nyquist limit, so the samples are the same noise as at 50 Hz,
twice as many and unaliased. 200 Hz would want DLPF_CFG 2 (~94 Hz, 1.5x the per-sample noise, and
the zero-velocity update's gyro threshold re-measured), which `configure()` does not write yet. A
read is one 14-byte burst on the bus the three VL53L1X share; at the default 100 kHz (~1.6 ms a
read) their transactions held the IMU at 53 Hz, and since 2026-10-02 the bus runs at 400 kHz
(`board/i2c3-400k.dts`): 100.2 Hz with the ToF running.

### The gyro's zero is re-measured, not taken once

The chip's bias moves with temperature, so the boot calibration this node shipped with was wrong
by the hour: parked with the wheels blocked, /odom read 0.00 deg/min of yaw while the EKF — whose
only yaw-rate source is this gyro — read +0.19, -0.54 and +0.67 deg/min in one night, and
RTAB-Map's map, built on that odometry, turned +27 deg in 40 minutes under a cart that never moved.

While the cart stands still the gyro's reading IS its bias, and the wheels know when it stands
still. `imu_bias_tracking` (live, default true) keeps a REST BLOCK going for the life of the node:
`imu_bias_s` of rest, starting `imu_bias_s` after the last motion so the chassis has settled, and
a finished block replaces the bias with its mean. Rest is the wheels' word — an exactly zero
measured twist, no twist being applied, an unbroken state stream — so a cart pushed or turned by
hand breaks a block, and a cart nobody is watching (link down, /odom muted) is never called still.
Nothing is published until one block has finished, which also fixes a node started while the cart
was rolling: it now waits for rest instead of subtracting the roll forever.
`imu_bias_tracking:=false` is the old boot-only bias, for an A/B without a restart.

Measured on a 30 s at-rest tape: per-sample noise
0.036 deg/s, and a 2.0 s block mean scatters by 0.30 deg/min — a one-way creep becomes a
zero-mean walk (~0.35 deg over 40 min against the +27 measured). A longer `imu_bias_s` shrinks the
error in force at any one moment, which is what a drive inherits (0.21 deg/min at 5 s, 0.02 at 10);
the parked walk stays ~0.35 deg either way, the sqrt trade.

## Zero-velocity update

While the cart is certainly standing still the node publishes `zupt`, a nav_msgs/Odometry with a
twist of exactly zero (frame `odom`, child `base_link`) — the EKF's `odom2`, which fuses its vx, vy
and vyaw (`ros/params/ekf.yaml`) — and nothing at all otherwise. Parked on its charger on
2026-09-24 the EKF's heading crept ~5 deg an hour, pulled by rf2o's +1.5 deg/min at rest while the
bias-tracked gyro read -0.001; that input had been silent since the lidar tracker, its only
publisher, stopped starting.

Certainly still is three witnesses at once (`ZuptGate`): the wheels have
witnessed rest for `zupt_settle_s` (default `imu_bias_s`, the rest the gyro's bias tracker
trusts), no non-zero `/cmd_vel` is younger than `zupt_cmd_hold_s` (default `cmd_timeout_s`), and
the bias-corrected yaw rate has stayed under `zupt_gyro_quiet_rad_s` for that same window, so a
cart turned by hand on still wheels is not frozen. No gyro reading (IMU off, no bias block yet)
means no update. Each witness is read fresh every tick, so the first moving sample stops it
within one period.

Every tunable is a live parameter — `zupt_publish`, `zupt_rate_hz`, `zupt_var_linear`,
`zupt_var_yaw`, `zupt_settle_s`, `zupt_cmd_hold_s`, `zupt_gyro_quiet_rad_s` — set with
`ros2 param set /base_bridge ...` and in force at the next tick; a new rate re-times the timer at
once. An on-set callback refuses a value outside its range (`zupt.hpp`'s `kZuptRanges`, logged
and returned to the caller) and a post-set callback stores the accepted one in an atomic, so the
timer and the IMU loop look nothing up per tick. The two windows borrow their defaults without
moving the parameters they come from. The reasons for every default are in `zupt.hpp`; a value
outside its range, or not a number, is refused with the reason and the value in force stays
(`50` arriving as an integer and `1e-4` as a string are taken as the numbers they are). The
report line names `zupt_publish=on|off`, and the zupt state ends the link-up and minute lines
with every setting in force: `zupt publishing for N s, M sent [rate 10 Hz, var 1e-06 xy 1e-06
yaw, settle 2 s, cmd hold 0.5 s, gyro quiet 0.005 rad/s]`.

| name | default | range | what it does |
| --- | --- | --- | --- |
| `zupt_publish` | true | bool | the update at all; false is the bridge before 2026-09-24: nothing on `/zupt` |
| `zupt_rate_hz` | 10 | 1..100 Hz | how often the update is published while the cart is at rest |
| `zupt_var_linear` | 1e-6 | 1e-9..1 (m/s)^2 | the variance claimed on vx and vy |
| `zupt_var_yaw` | 1e-6 | 1e-9..1 (rad/s)^2 | the variance claimed on vyaw (the gyro's is 4e-4, rf2o's 2.5e-3) |
| `zupt_settle_s` | `imu_bias_s` (2.0) | 0..60 s | witnessed rest before the update, and the hold after a gyro turn |
| `zupt_cmd_hold_s` | `cmd_timeout_s` (0.5) | 0..10 s | how long a non-zero `/cmd_vel` holds the update off |
| `zupt_gyro_quiet_rad_s` | 0.005 | 1e-4..0.5 rad/s | a bias-corrected yaw rate at or above this is a turn (7.9 sigma of the parked chip's noise) |

A parked A/B, one variable at a time, reading the heading creep between each:

```bash
ros2 param set /base_bridge zupt_publish false        # the filter as it was: the baseline creep
ros2 param set /base_bridge zupt_publish true
ros2 param set /base_bridge zupt_rate_hz 50           # every other gyro sample's worth
ros2 param set /base_bridge zupt_var_yaw 1.0e-4       # a hundred times looser
ros2 param get /base_bridge zupt_rate_hz              # what is in force
```

Board cost: one Odometry message per tick while parked (10 Hz by default, 100 at most), a timer
at that rate and two atomic stores per IMU sample; no new process, no new thread.

## Build and switch

`ros/Dockerfile` apt-installs `nlohmann-json3-dev` and colcon-builds this package next to
`pepin_bringup` into `/ws/install`. A change here needs an image rebuild (`ros/build-image.sh --ship`),
not `ros/sync.sh`: only the Python package is mounted from the host.

    ros2 launch pepin_bringup robot.launch.py  # composed into base_container

## test/

No ament test target: the board image is not built on a laptop, so the contracts are stand-alone
`main()`s, no ROS and no gtest, compiled and run by `tests/unit/test_base_cpp_contracts.py` (slow,
skipped without a `c++`).

- `test/gyro_bias_contract.cpp` — the gyro bias tracker's table, replayed against `gyro_bias.hpp`;
  its own header comment has the `c++` line.
- `test/zupt_contract.cpp` — the zero-velocity update's table, replayed against `zupt.hpp` the same
  way, verdict words and defaults included.
- `test/neck_contract.cpp` — the neck's rate cap replayed as a table, then the model fed from
  stdin: `tests/unit/test_base_cpp_contracts.py` writes `config/neck.json` (and variants) in and
  holds every answer to `pepin.neck`'s within 1e-9.
- `test/protocol_samples.json` — wire lines recorded from the Python bridge this one was ported
  from: the twists and stops `protocol.hpp` must encode byte for byte, and the state lines it must
  parse.

`pepin.odometry.TwistFromPose` is the Python twin of the odometry header; the gyro and
zero-velocity headers carry their own contracts above.
