#!/bin/bash
# Kalibr on a procedure D recording of ros/calib_record.sh (docs/head_imu_calibration.md):
#
#   ros/calib_run.sh BAG TAG_M                       one run: convert, configure, Kalibr, verdict
#   ros/calib_run.sh BAG TAG_M --against BAG2        and its agreement with the other run
#   ros/calib_run.sh BAG TAG_M --against BAG2 --apply    then into config/camera.json
#
# TAG_M: the printed tag's black square measured with a ruler, in metres (the A3 print: ~0.034).
# 1. the ROS 2 bag (MCAP) to ROS 1 with rosbags (uv): BAG/kalibr/calib.bag
# 2. Kalibr's camchain.yaml, imu.yaml, april.yaml from the repo's numbers into BAG/kalibr/:
#    ros/tools/vio_config.py --kalibr-only --tag-size TAG_M, run in pepin-laptop:latest because the
#    rectified focal is OpenCV's stereoRectify's and must be camera_stream's (4.6: 494.22 px, not
#    the 495.08 of uv's 4.13)
# 3. kalibr_calibrate_imu_camera (time calibration on) in pepin-kalibr: BAG/kalibr/calib-
#    camchain-imucam.yaml, -results-imucam.txt, -report-imucam.pdf, and kalibr.log
# 4. ros/tools/head_calib.py report: each eye's reprojection error, T_cam_imu, timeshift_cam_imu
#    (and the time offset at camera_stamp_lag_s's default), the runs' agreement (0.5 deg / 5 mm /
#    2 ms) and ACCEPTED or not
# --apply: head_calib.py apply writes the runs' mean into config/camera.json's stereo.head_imu
# (refused when a check fails; --force writes it anyway). A step whose output exists is not run
# again (--rerun runs Kalibr again). KALIBR_ARGS adds Kalibr options (--timeoffset-padding 0.05).
# pepin-kalibr is prehensile/kalibr:arm64, pinned below, retagged; pulled when missing (1.2 GB).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
KALIBR="pepin-kalibr"
KALIBR_PIN="prehensile/kalibr:arm64@sha256:68d089b2c5514d2f143ce95dd015981bce22e73cfd97edf1ee3fd7ebe41840ef"
CONFIG_IMAGE="${PEPIN_CALIB_CONFIG_IMAGE:-pepin-laptop:latest}"
ROSBAGS="rosbags==0.11.5"
TOPICS=(/camera/image /camera/right/image /head/imu)
usage() { echo "usage: ros/calib_run.sh BAG TAG_M [--against BAG2] [--apply] [--force] [--rerun]"; exit 2; }
case "${1:-}" in
    "" | -h | --help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac
[ $# -ge 2 ] || usage
BAG="$(cd "$1" && pwd)" || exit 2
TAG_M="$2"
shift 2
AGAINST="" APPLY=0 FORCE="" RERUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --against) AGAINST="$(cd "${2:?}" && pwd)"; shift ;;
        --apply) APPLY=1 ;;
        --force) FORCE="--force" ;;
        --rerun) RERUN=1 ;;
        *) usage ;;
    esac
    shift
done
[ -f "$BAG/metadata.yaml" ] || { echo "$BAG is not a ROS 2 bag (no metadata.yaml)"; exit 2; }
# A tag size in millimetres, or a whole-grid size, is the likely slip: refuse it before Kalibr runs.
awk -v t="$TAG_M" 'BEGIN { exit !(t + 0 >= 0.015 && t + 0 <= 0.06) }' \
    || { echo "TAG_M $TAG_M: metres, one tag's black square (0.015-0.06; the A3 print ~0.034)"; exit 2; }
WORK="$BAG/kalibr"
mkdir -p "$WORK"

# 1. ROS 2 -> ROS 1
if [ ! -f "$WORK/calib.bag" ]; then
    echo "== converting $(basename "$BAG") to ROS 1 (rosbags)"
    rm -f "$WORK/calib.tmp.bag"
    (cd "$REPO" && uv run -q --with "$ROSBAGS" rosbags-convert --src "$BAG" \
        --dst "$WORK/calib.tmp.bag" --include-topic "${TOPICS[@]}")
    mv "$WORK/calib.tmp.bag" "$WORK/calib.bag"
fi

# 2. Kalibr's inputs, in camera_stream's OpenCV
docker image inspect "$CONFIG_IMAGE" >/dev/null 2>&1 \
    || { echo "no $CONFIG_IMAGE here (ros/laptop-build.sh)"; exit 2; }
docker run --rm --network none \
    -v "$HERE/tools:/tools:ro" -v "$REPO/src/pepin:/ws/pepin_src/pepin:ro" \
    -v "$REPO/config:/ws/config:ro" -v "$WORK:/out" \
    --entrypoint python3 "$CONFIG_IMAGE" /tools/vio_config.py --kalibr-only --tag-size "$TAG_M" --out /out \
    | sed 's#/out/#'"$WORK"'/#'

# 3. Kalibr
if ! docker image inspect "$KALIBR" >/dev/null 2>&1; then
    echo "== pulling $KALIBR_PIN as $KALIBR"
    docker pull "$KALIBR_PIN" && docker tag "$KALIBR_PIN" "$KALIBR"
fi
if [ ! -f "$WORK/calib-camchain-imucam.yaml" ] || [ "$RERUN" = 1 ]; then
    rm -f "$WORK"/calib-camchain-imucam.yaml "$WORK"/calib-results-imucam.txt
    echo "== kalibr_calibrate_imu_camera on $(basename "$BAG") (minutes; output in $WORK/kalibr.log)"
    # The image's own entrypoint passes the command's words to catkin's setup and dies on them:
    # bash sources the workspace itself. MPLBACKEND: no display for the report's figures.
    # shellcheck disable=SC2086
    docker run --rm --network none -v "$WORK:/k" -e MPLBACKEND=Agg --entrypoint bash "$KALIBR" -c \
        'source /catkin_ws/devel/setup.bash && cd /k && rosrun kalibr kalibr_calibrate_imu_camera \
         --bag /k/calib.bag --cams /k/camchain.yaml --imu /k/imu.yaml --target /k/april.yaml \
         --dont-show-report '"${KALIBR_ARGS:-}" 2>&1 | tee "$WORK/kalibr.log" | tr '\r' '\n' \
        | grep -E --line-buffered "Extracted corners|Optimization|\[px\]|\[rad/s\]|\[m/s\^2\]|Error|Exception|Traceback" || true
    [ -f "$WORK/calib-camchain-imucam.yaml" ] \
        || { echo "Kalibr wrote no result: the end of $WORK/kalibr.log:"; tail -25 "$WORK/kalibr.log"; exit 1; }
fi

# 4. the verdict, then the block
BAGS=("$BAG")
[ -z "$AGAINST" ] || BAGS+=("$AGAINST")
echo "== the result"
VERDICT=0
(cd "$REPO" && uv run -q python ros/tools/head_calib.py report "${BAGS[@]}") || VERDICT=$?
echo "report: $WORK/calib-report-imucam.pdf (the gyro and accelerometer errors must look white)"
if [ "$APPLY" = 1 ]; then
    (cd "$REPO" && uv run -q python ros/tools/head_calib.py apply "${BAGS[@]}" ${FORCE:+"$FORCE"})
fi
exit "$VERDICT"
