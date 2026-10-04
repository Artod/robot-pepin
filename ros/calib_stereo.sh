#!/bin/bash
# The stereo head's camera calibration with Kalibr, on the RAW eyes (docs/stereo_calibration.md):
#
#   ros/calib_stereo.sh record [--seconds 90] [--tag 0.025]   the board's MJPEG stream copied
#                              as it arrives while the AprilGrid is waved by hand (guidance and
#                              a tag count per eye printed): ros/maps/rec/stereo_<UTC>Z/
#   ros/calib_stereo.sh run REC TAG_M [--hz 4] [--rerun] [--apply] [--force]
#
# run: 1. the sharpest pair of every 1/hz window cut into the upright eyes, mono8, into a ROS 1
# bag (REC/kalibr/stereo.bag; ros/tools/stereo_kalibr.py bag, uv with rosbags); 2. Kalibr's
# target (REC/kalibr/april.yaml: 6x6, TAG_M the black square measured with a ruler, spacing
# 0.3); 3. kalibr_calibrate_cameras --models pinhole-radtan pinhole-radtan in pepin-kalibr
# (REC/kalibr/stereo-camchain.yaml, -results-cam.txt, -report-cam.pdf, kalibr.log); a focal
# initialisation that fails (it needs whole-grid views) takes the current file's fx from stdin
# (KALIBR_MANUAL_FOCAL_LENGTH_INIT); 4. the report against config/stereo_calibration.json:
# per-eye reprojection, intrinsics, distortion, the bar, the rectified eye's turn and the range
# bias the current file has if Kalibr is right; ACCEPTED when each eye's RMS reprojection is
# under 0.3 px and the baseline within 2 mm of the rig's nominal 63 mm (exit 1 otherwise).
# --apply: the result into config/stereo_calibration.json (the previous file kept once as
# config/stereo_calibration.pre-kalibr-<day>.json) and camera.json's eye + head_imu carried
# through the rectified eye's turn; refused unless ACCEPTED (--force writes anyway). A step whose
# output exists is not run again (--rerun runs Kalibr again). KALIBR_ARGS adds Kalibr options.
# Offline after the recording: nothing here touches the robot or a running node; the record
# step is one more reader of ustreamer (PEPIN_HOST, default 10.0.0.187).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
BOARD="${PEPIN_HOST:-10.0.0.187}"
KALIBR="pepin-kalibr"
KALIBR_PIN="prehensile/kalibr:arm64@sha256:68d089b2c5514d2f143ce95dd015981bce22e73cfd97edf1ee3fd7ebe41840ef"
ROSBAGS="rosbags==0.11.5"
usage() {
    echo "usage: ros/calib_stereo.sh record [--seconds S] [--tag M]"
    echo "       ros/calib_stereo.sh run REC TAG_M [--hz HZ] [--rerun] [--apply] [--force]"
    exit 2
}
tool() { (cd "$REPO" && uv run -q python ros/tools/stereo_kalibr.py "$@"); }
CMD="${1:-}"
case "$CMD" in
    "" | -h | --help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    record | run) shift ;;
    *) usage ;;
esac

if [ "$CMD" = record ]; then
    SECONDS_ARG=90 TAG=0.025
    while [ $# -gt 0 ]; do
        case "$1" in
            --seconds) SECONDS_ARG="${2:?}"; shift ;;
            --tag) TAG="${2:?}"; shift ;;
            *) usage ;;
        esac
        shift
    done
    NAME="stereo_$(date -u +%Y%m%d_%H%M%S)Z"
    REC_DIR="${PEPIN_CALIB_REC:-$HERE/maps/rec}"
    tool record --out "$REC_DIR/$NAME" --seconds "$SECONDS_ARG" --host "$BOARD" --tag "$TAG"
    echo "next: ros/calib_stereo.sh run $REC_DIR/$NAME $TAG   (TAG_M: the black square, ruler)"
    exit 0
fi

[ $# -ge 2 ] || usage
REC="$(cd "$1" && pwd)" || exit 2
TAG_M="$2"
shift 2
HZ=4 RERUN=0 APPLY=0 FORCE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --hz) HZ="${2:?}"; shift ;;
        --rerun) RERUN=1 ;;
        --apply) APPLY=1 ;;
        --force) FORCE="--force" ;;
        *) usage ;;
    esac
    shift
done
[ -f "$REC/stereo.mjpeg" ] && [ -f "$REC/meta.json" ] \
    || { echo "$REC is not a ros/calib_stereo.sh recording (stereo.mjpeg + meta.json)"; exit 2; }
awk -v t="$TAG_M" 'BEGIN { exit !(t + 0 >= 0.015 && t + 0 <= 0.06) }' \
    || { echo "TAG_M $TAG_M: metres, one tag's black square (0.015-0.06; the A4 print 0.025)"; exit 2; }
WORK="$REC/kalibr"
mkdir -p "$WORK"

# 1. the bag
if [ ! -f "$WORK/stereo.bag" ]; then
    echo "== the sharpest pair per 1/$HZ s into $WORK/stereo.bag"
    (cd "$REPO" && uv run -q --with "$ROSBAGS" python ros/tools/stereo_kalibr.py bag "$REC" --hz "$HZ")
fi

# 2. the target, rewritten every run (the tag size is this run's argument)
TARGET="$(printf '%s\n' "# ros/calib_stereo.sh: tagSize is the printed black square, measured." \
    "target_type: aprilgrid" "tagCols: 6" "tagRows: 6" "tagSize: $TAG_M" "tagSpacing: 0.3")"
if [ ! -f "$WORK/april.yaml" ] || [ "$(cat "$WORK/april.yaml")" != "$TARGET" ]; then
    printf '%s\n' "$TARGET" >"$WORK/april.yaml"
    RERUN=1
fi

# 3. Kalibr
if ! docker image inspect "$KALIBR" >/dev/null 2>&1; then
    echo "== pulling $KALIBR_PIN as $KALIBR"
    docker pull "$KALIBR_PIN" && docker tag "$KALIBR_PIN" "$KALIBR"
fi
if [ ! -f "$WORK/stereo-camchain.yaml" ] || [ "$RERUN" = 1 ]; then
    rm -f "$WORK"/stereo-camchain.yaml "$WORK"/stereo-results-cam.txt
    FOCAL="$(cd "$REPO" && uv run -q python -c 'import json; print(round(json.load(open("config/stereo_calibration.json"))["k_left"][0][0], 1))')"
    echo "== kalibr_calibrate_cameras on $(basename "$REC") (minutes; $WORK/kalibr.log; focal fallback $FOCAL px)"
    # The image's entrypoint dies on the command's words: bash sources the workspace itself.
    # MPLBACKEND: no display for the report. stdin: the focal guess for each initialisation
    # that fails (read only then).
    # shellcheck disable=SC2086
    yes "$FOCAL" | head -20 | docker run --rm -i --network none -v "$WORK:/k" -e MPLBACKEND=Agg \
        -e KALIBR_MANUAL_FOCAL_LENGTH_INIT=1 --entrypoint bash "$KALIBR" -c \
        'source /catkin_ws/devel/setup.bash && cd /k && rosrun kalibr kalibr_calibrate_cameras \
         --bag /k/stereo.bag --topics /cam0/image_raw /cam1/image_raw \
         --models pinhole-radtan pinhole-radtan --target /k/april.yaml \
         --dont-show-report '"${KALIBR_ARGS:-}" 2>&1 | tee "$WORK/kalibr.log" | tr '\r' '\n' \
        | grep -E --line-buffered "Extracted corners|initialized to|focal|reprojection error|baseline|Processed|Error|Exception|Traceback|diverged" || true
    [ -f "$WORK/stereo-camchain.yaml" ] \
        || { echo "Kalibr wrote no result: the end of $WORK/kalibr.log:"; tail -25 "$WORK/kalibr.log"; exit 1; }
fi

# 4. the verdict, then the file
VERDICT=0
if [ "$APPLY" = 1 ]; then
    tool apply "$REC" ${FORCE:+"$FORCE"} || VERDICT=$?
else
    tool report "$REC" || VERDICT=$?
fi
echo "report: $WORK/stereo-report-cam.pdf (residuals by image position, the corners especially)"
exit "$VERDICT"
