#!/bin/bash
# Persistent switches of the board's sensor stack (kept across restarts and reboots):
#   ros/feature.sh imu on|off    read the MPU6050 and fuse it with the wheels (on by default)
#   ros/feature.sh ekf on|off    fuse the live odometry sources and own odom -> base_link
#                                (on by default). The IMU is a source of this
#                                filter, never its switch: `imu off` leaves the EKF running on
#                                the wheels. Off, the bridge publishes odom -> base_link itself
#                                and nothing publishes /odometry/filtered, which every
#                                recorded drive reads
#   ros/feature.sh laser_odom on|off
#                                the lidar's own scan-to-scan odometry (rf2o, ~100 MB) as a twist
#                                for the EKF (odom3). On by default. A source of the filter, never
#                                its switch: off, the wheels, the gyro and the camera are what
#                                they were. It publishes no transform — the EKF owns odom -> base_link
#   ros/feature.sh tof on|off    the three ToF sensors into the local costmap (a Python bridge,
#                                ~150 MB; on by default)
#   ros/feature.sh board_bag on|off
#                                the board's raw sensors recorded on the board, always, in minute
#                                MCAP files under maps/board_rec (pepin.board_bag): 20 GB cap,
#                                10 GB of the card kept free; off by default
#   ros/feature.sh head_imu on|off
#                                the head IMU (head_server's 3340 stream) into the base bridge:
#                                /head/imu and the mast filter (/mast/state); off by default
# Each change restarts the one launch process (about 60 s); the robot does not move.
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
FEATURE="${1:?imu | ekf | laser_odom | tof | board_bag | head_imu}"; STATE="${2:?on | off}"
case "$FEATURE" in imu) VAR=PEPIN_IMU ;; ekf) VAR=PEPIN_EKF ;; laser_odom) VAR=PEPIN_LASER_ODOM ;; tof) VAR=PEPIN_TOF ;; board_bag) VAR=PEPIN_BOARD_BAG ;; head_imu) VAR=PEPIN_HEAD_IMU ;; *) echo "unknown feature $FEATURE"; exit 2 ;; esac
case "$STATE" in
    on) VAL=true ;;
    off) VAL=false ;;
    *) echo "on or off"; exit 2 ;;
esac
ssh "root@$BOARD" "grep -v '^$VAR=' /etc/default/pepin-ros > /etc/default/pepin-ros.new; echo '$VAR=$VAL' >> /etc/default/pepin-ros.new; mv /etc/default/pepin-ros.new /etc/default/pepin-ros; systemctl restart pepin-ros"
T0=$(date +%s); echo -n "$FEATURE $STATE; restarting the stack..."
for i in $(seq 1 60); do ssh "root@$BOARD" "docker logs pepin-ros 2>&1 | grep -q 'lifecycle_manager_sensors.*Managed nodes are active'" 2>/dev/null && break; sleep 2; done
echo " up in $(( $(date +%s) - T0 )) s"
ssh "root@$BOARD" "cat /etc/default/pepin-ros | tr '\\n' ' '; echo; free -m | sed -n 2p"
