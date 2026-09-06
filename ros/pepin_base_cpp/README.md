# pepin_base_cpp

The base bridge in C++: `/cmd_vel` down to the board's base server, its state stream up as
`/odom` and `odom -> base_link`. Same node name, parameters and wire protocol as the Python
`base_bridge` in `pepin_bringup`, which stays. Why: on the board each rclpy process costs
~190 MB RSS and this node ~10% of a core; here it is ~25 MB and ~1%, and RAM runs out first.

- `include/pepin_base_cpp/protocol.hpp` — the wire format, no ROS and no sockets in it.
- `include/pepin_base_cpp/link.hpp` — reconnecting JSON-lines TCP client, one reader thread.
- `include/pepin_base_cpp/mpu6050.hpp` — the IMU on an i2c-dev bus, SI units, no ROS.
- `src/base_bridge.cpp` — the node: /odom, TF, the /cmd_vel sink, the 5 Hz resend, `main`.

## IMU

`imu_enable` (false), `imu_device` (/dev/i2c-2), `imu_address` (0x68), `imu_rate_hz` (50),
`imu_frame` (imu_link), `imu_bias_s` (2.0): a thread samples an MPU6050 and publishes
`imu/data_raw` without orientation; a missing chip is one warning and the wheels carry on.
Registers PWR_MGMT_1 0x01, SMPLRT_DIV 1000/rate-1, CONFIG 0x03 (DLPF ~44 Hz), GYRO_CONFIG 0x08
(+-500 dps), ACCEL_CONFIG 0x08 (+-4 g), WHO_AM_I 0x68, 14 bytes from ACCEL_XOUT_H 0x3B. The first
`imu_bias_s` of standing still is the gyro bias, logged then subtracted; the EKF that fuses it
with /odom is robot_localization, configured outside this package.

## Build and switch

`ros/Dockerfile` apt-installs `nlohmann-json3-dev` and colcon-builds this package next to
`pepin_bringup` into `/ws/install`. A change here needs an image rebuild (`ros/build.sh`),
not `ros/sync.sh`: only the Python package is mounted from the host.

    ros2 launch pepin_bringup robot.launch.py base_bridge_cpp:=true  # this node
    ros2 launch pepin_bringup robot.launch.py                        # the Python one (default)

## test/protocol_samples.json

Lines from the running Python stack (`scratch/gen_protocol_samples.py`). A gtest would assert:
`encode_twist`/`encode_stop` reproduce every `requests[].line` byte for byte, `parse_state`
yields `states[].parsed` and nothing for `not_states[]`, `LineReader` the `reader[].objects`.
