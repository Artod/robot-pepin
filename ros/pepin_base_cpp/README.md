# pepin_base_cpp

The base bridge in C++: `/cmd_vel` down to the board's base server, its state stream up as
`/odom` and `odom -> base_link`. Same node name, parameters and wire protocol as the Python
`base_bridge` in `pepin_bringup`, which stays. Why: on the board each rclpy process costs
~190 MB RSS and this node ~10% of a core; here it is ~25 MB and ~1%, and RAM runs out first.

- `include/pepin_base_cpp/protocol.hpp` — the wire format, no ROS and no sockets in it.
- `include/pepin_base_cpp/link.hpp` — reconnecting JSON-lines TCP client, one reader thread.
- `include/pepin_base_cpp/mpu6050.hpp` — the IMU on an i2c-dev bus, SI units, no ROS.
- `include/pepin_base_cpp/twist_from_pose.hpp` — the twist the wheels measured, off two poses.
- `include/pepin_base_cpp/gyro_bias.hpp` — the gyro's zero, and the wheels' word on rest.
- `src/base_bridge.cpp` — the node: /odom, TF, the /cmd_vel sink, the 5 Hz resend, `main`.

## IMU

`imu_enable` (false), `imu_device` (/dev/i2c-2), `imu_address` (0x68), `imu_rate_hz` (50),
`imu_frame` (imu_link), `imu_bias_s` (2.0), `imu_bias_tracking` (true): a thread samples an
MPU6050 and publishes `imu/data_raw` without orientation; a missing chip is one warning and the
wheels carry on. Registers PWR_MGMT_1 0x01, SMPLRT_DIV 1000/rate-1, CONFIG 0x03 (DLPF ~44 Hz),
GYRO_CONFIG 0x08 (+-500 dps), ACCEL_CONFIG 0x08 (+-4 g), WHO_AM_I 0x68, 14 bytes from
ACCEL_XOUT_H 0x3B. The EKF that fuses the yaw rate with /odom is robot_localization, configured
outside this package.

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

Measured on the 30 s at-rest tape (`scratch/gyro_block_mean_noise.py`): per-sample noise
0.036 deg/s, and a 2.0 s block mean scatters by 0.30 deg/min — a one-way creep becomes a
zero-mean walk (~0.35 deg over 40 min against the +27 measured). A longer `imu_bias_s` shrinks the
error in force at any one moment, which is what a drive inherits (0.21 deg/min at 5 s, 0.02 at 10);
the parked walk stays ~0.35 deg either way, the sqrt trade.

## Build and switch

`ros/Dockerfile` apt-installs `nlohmann-json3-dev` and colcon-builds this package next to
`pepin_bringup` into `/ws/install`. A change here needs an image rebuild (`ros/build.sh`),
not `ros/sync.sh`: only the Python package is mounted from the host.

    ros2 launch pepin_bringup robot.launch.py base_bridge_cpp:=true  # this node
    ros2 launch pepin_bringup robot.launch.py                        # the Python one (default)

## test/

No ament test target: the board image is not built on a laptop, so these run by hand.

- `test/gyro_bias_contract.cpp` — the table of
  `tests/unit/test_gyro.py::test_gyro_bias_contract_the_cpp_bridge_mirrors`, replayed against
  `gyro_bias.hpp`. One stand-alone `main()`, no ROS and no gtest; its own header comment has the
  `c++` line, and `scratch/syntax_check_base_bridge.sh` runs it beside a type-check of the node.
- `test/protocol_samples.json` — lines from the running Python stack
  (`scratch/gen_protocol_samples.py`). A gtest would assert: `encode_twist`/`encode_stop` reproduce
  every `requests[].line` byte for byte, `parse_state` yields `states[].parsed` and nothing for
  `not_states[]`, `LineReader` the `reader[].objects`.

The Python twins are the reference for both header classes with a contract test
(`pepin.odometry.TwistFromPose`, `pepin.gyro.GyroBiasTracker`, `pepin.gyro.RestWitness`): the maths
changes in `src/pepin/` and `tests/unit/` first, and the header follows.
