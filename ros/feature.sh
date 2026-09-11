#!/bin/bash
# Persistent switches of the robot's stack (kept across ros/mode.sh calls and reboots):
#   ros/feature.sh cpp on|off    the C++ base bridge instead of the Python one (~25 MB vs ~190 MB)
#   ros/feature.sh imu on|off    read the MPU6050 and fuse it with the wheels (needs cpp on)
#   ros/feature.sh tof on|off    the three ToF sensors into the local costmap (a Python bridge, ~150 MB)
#   ros/feature.sh neck on|off   the neck's encoders as /neck/state and the live base_link -> camera_link
#                                (a Python node, ~150 MB); the laptop's SLAM must then run with
#                                `ros/laptop.sh vslam --neck`, or two nodes publish that one edge
# Each change restarts the one launch process (about 60 s); the robot does not move.
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
FEATURE="${1:?cpp | imu | tof | neck}"; STATE="${2:?on | off}"
case "$FEATURE" in cpp) VAR=PEPIN_CPP_BRIDGE ;; imu) VAR=PEPIN_IMU ;; tof) VAR=PEPIN_TOF ;; neck) VAR=PEPIN_NECK ;; *) echo "unknown feature $FEATURE"; exit 2 ;; esac
case "$STATE" in on) VAL=true ;; off) VAL=false ;; *) echo "on or off"; exit 2 ;; esac
if [ "$FEATURE" = imu ] && [ "$VAL" = true ]; then
    ssh "root@$BOARD" "grep -q 'PEPIN_CPP_BRIDGE=true' /etc/default/pepin-ros" || { echo "imu needs the C++ bridge: ros/feature.sh cpp on first"; exit 1; }
fi
ssh "root@$BOARD" "grep -v '^$VAR=' /etc/default/pepin-ros > /etc/default/pepin-ros.new; echo '$VAR=$VAL' >> /etc/default/pepin-ros.new; mv /etc/default/pepin-ros.new /etc/default/pepin-ros; systemctl restart pepin-ros"
T0=$(date +%s); echo -n "$FEATURE $STATE; restarting the stack..."
for i in $(seq 1 60); do ssh "root@$BOARD" "docker logs pepin-ros 2>&1 | grep -q 'lifecycle_manager_sensors.*Managed nodes are active'" 2>/dev/null && break; sleep 2; done
echo " up in $(( $(date +%s) - T0 )) s"
ssh "root@$BOARD" "cat /etc/default/pepin-ros | tr '\\n' ' '; echo; free -m | sed -n 2p"
