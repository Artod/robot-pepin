#!/usr/bin/env bash
# The offline A/B stand of the visual odometry (vio.md section 6): one recorded drive, one arm,
# the board's EKF (ros/params/ekf.yaml) replayed on the drive's own inputs plus the arm's /vo, and
# the arm's /odometry/filtered out as the scorer's CSV (ros/tools/vio_score.py).
#
#   ros/vio_replay.sh 0601 --arm E        # OpenVINS (/maps/vio) + the relay (vo_input vio)
#   ros/vio_replay.sh 0601 --arm E0       # E without OpenVINS's ZUPT, the first design (/maps/vio_nozupt)
#   ros/vio_replay.sh 0601 --arm B        # stereo_odometry + the relay (vo_input stereo)
#   ros/vio_replay.sh 0601 --arm A        # the EKF with no /vo at all
#   ros/vio_replay.sh --help
#
# Inputs in ros/maps/rec (PEPIN_REPLAY_REC moves them): the drive's bag 0601_*/ (it carries /odom,
# /imu/data_raw, /zupt, /odom_laser, /head/imu, /tf with the neck chain) and its camera bag
# 0601_*_cam.bag (ros/clip_to_bag.sh 0601 --require-grab). Output beside them:
# 0601_*_arm_<ARM>.bag (/odometry/filtered, /vo, /ov_msckf/poseimu) and 0601_*_arm_<ARM>.csv.
# The arm's config files are generated first (uv run python ros/tools/vio_config.py [--no-zupt
# --out ros/maps/vio_nozupt]). Real time (OpenVINS has no ROS 2 serial reader): a drive's length.
# A throwaway container of pepin-laptop:vio with no network at all; sim time from the bags' clock.
set -euo pipefail

ROS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$ROS_DIR")"
IMAGE="${PEPIN_REPLAY_VIO_IMAGE:-pepin-laptop:vio}"
REC="${PEPIN_REPLAY_REC:-$ROS_DIR/maps/rec}"
MAPS="${PEPIN_REPLAY_MAPS:-$ROS_DIR/maps}"

case "${1:-}" in
    "" | -h | --help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac
RUN="$1"; shift
ARM=""
while [ $# -gt 0 ]; do
    case "$1" in
        --arm) ARM="${2:-}"; shift 2 ;;
        *) echo "usage: ros/vio_replay.sh RUN --arm A|B|E|E0"; exit 2 ;;
    esac
done
case "$ARM" in A | B | E | E0) ;; *) echo "--arm A|B|E|E0"; exit 2 ;; esac
shopt -s nullglob
drives=("$REC"/"$RUN"_*/)
cams=("$REC"/"$RUN"_*_cam.bag)
[ "${#drives[@]}" -ge 1 ] || { echo "no drive bag $REC/${RUN}_*/"; exit 2; }
DRIVE="$(basename "${drives[0]}")"
CAM=""
if [ "$ARM" != A ]; then
    [ "${#cams[@]}" -eq 1 ] || { echo "no camera bag $REC/${RUN}_*_cam.bag: ros/clip_to_bag.sh $RUN --require-grab"; exit 2; }
    CAM="$(basename "${cams[0]}")"
fi
OUT="${DRIVE%/}_arm_${ARM}"
[ ! -e "$REC/$OUT.bag" ] || { echo "$REC/$OUT.bag exists: remove it first"; exit 2; }
CONFIG=/maps/vio/estimator_config.yaml
[ "$ARM" = E0 ] && CONFIG=/maps/vio_nozupt/estimator_config.yaml

docker run --rm --network none \
    -v "$REPO:/repo:ro" -v "$REC:/rec" -v "$MAPS:/maps:ro" -v "$ROS_DIR/params:/params:ro" \
    -v "$REPO/config:/ws/config:ro" -v "$REPO/src/pepin:/ws/pepin_src/pepin:ro" \
    -v "$ROS_DIR/pepin_bringup/pepin_bringup:/ws/install/pepin_bringup/lib/python3.12/site-packages/pepin_bringup:ro" \
    -v "$ROS_DIR/pepin_bringup/launch:/ws/install/pepin_bringup/share/pepin_bringup/launch:ro" \
    -e PYTHONDONTWRITEBYTECODE=1 -e ROS_DOMAIN_ID=78 -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e ARM="$ARM" -e DRIVE="/rec/$DRIVE" -e CAM="${CAM:+/rec/$CAM}" -e OUT="/rec/$OUT" -e CONFIG="$CONFIG" \
    --entrypoint /bin/bash "$IMAGE" -c '
set -eo pipefail
set -m  # job control: a background job of a non-interactive shell would ignore the SIGINT below
source /opt/ros/jazzy/setup.bash; source /ws/install/setup.bash
[ -f /ws_vio/install/setup.bash ] && source /ws_vio/install/setup.bash
SIM=(--ros-args -p use_sim_time:=true)
pids=()
python3 /repo/ros/tools/head_static_tf.py "${SIM[@]}" & pids+=($!)
ros2 run robot_localization ekf_node "${SIM[@]}" --params-file /params/ekf.yaml \
    -p publish_tf:=false -r __node:=ekf_filter_node & pids+=($!)
case "$ARM" in
    E | E0)
        ros2 launch pepin_bringup vio.launch.py config:="$CONFIG" & pids+=($!)
        python3 -m pepin_bringup.visual_odometry "${SIM[@]}" -p vo_input:=vio & pids+=($!) ;;
    B)
        python3 /repo/ros/tools/vo_params.py > /tmp/vo.yaml
        ros2 run rtabmap_odom stereo_odometry "${SIM[@]}" --params-file /tmp/vo.yaml \
            -r left/image_rect:=/camera/image -r left/camera_info:=/camera/camera_info \
            -r right/image_rect:=/camera/right/image -r right/camera_info:=/camera/right/camera_info \
            -r odom:=/vo/raw & pids+=($!)
        python3 -m pepin_bringup.visual_odometry "${SIM[@]}" -p vo_input:=stereo & pids+=($!) ;;
esac
ros2 bag record --storage mcap -o "$OUT.bag" /odometry/filtered /vo /ov_msckf/poseimu \
    --use-sim-time & REC_PID=$!
sleep 6  # every node subscribed before the first message
INPUTS=(-i "$DRIVE")
[ -n "$CAM" ] && INPUTS+=(-i "$CAM")
# The drive bag'"'"'s own visual odometry and fused odometry are what the arm replaces: /vo, and
# since 2026-10-06 also the live relay'"'"'s /vo_twist (the EKF'"'"'s twist0) and OpenVINS'"'"'s own
# outputs and the keeper'"'"'s seed, which would otherwise reach the replayed EKF and relay beside
# the arm'"'"'s (arm A would not be "no VIO"); its /tf keeps the neck chain (odom -> base_link in it
# is harmless: the replayed EKF publishes no transform).
ros2 bag play "${INPUTS[@]}" --clock 100 --exclude-topics /vo /vo_twist /odometry/filtered \
    /vo/raw /ov_msckf/poseimu /ov_msckf/odomimu /ov_msckf/health /ov_msckf/points_msckf \
    /ov_msckf/points_slam /vio/seed_twist
sleep 3
kill -INT "$REC_PID"; wait "$REC_PID" || true
# Each job is its own process group under job control: the wrappers (ros2 run, ros2 launch) and
# the nodes under them take the SIGINT together; whatever is still up 10 s later gets a SIGTERM.
for p in "${pids[@]}"; do kill -INT -- "-$p" 2>/dev/null || true; done
for _ in $(seq 1 20); do jobs -r | grep -q . || break; sleep 0.5; done
for p in "${pids[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; wait || true
python3 /repo/ros/tools/bag_poses.py "$OUT.bag" /odometry/filtered > "$OUT.csv"
echo "wrote $OUT.bag and $OUT.csv"
'
