#!/bin/bash
# Procedure D's recording for the head's camera-IMU calibration (docs/head_imu_calibration.md): a
# ROS 2 bag of the two rectified eyes and the head IMU while the gaze arbiter walks the head through
# ros/tools/neck_dance.py's poses. Head only: the wheels are never asked for (the cart's shuttles
# and pivots of procedure D are a separate step, by hand).
#
#   ros/calib_record.sh --check          one snapshot through Kalibr's detector: tags found, their
#                                        side in px, and the --centre/--distance to record with
#   ros/calib_record.sh --dry-run --centre 0.1 -3.9 --distance 0.28   the checks and the plan
#   ros/calib_record.sh --centre 0.1 -3.9 --distance 0.28             record and dance (60-90 s)
#
# The grid: --centre PAN TILT (deg) is the head pose that centres it in the picture, --distance
# the lens-to-grid metres (both printed by --check); without --centre it is upright on a wall
# --distance from the lens with the head level, its centre --height above the floor (0.28 and
# 1.2). --tag is the printed square (0.025: the A4 print, measured). The dance stays within
# +-20 deg of pan and +-15 of tilt of the centre with the whole grid in the picture.
# Refused unless camera_stream's camera_stamp is grab (ustreamer's send stamp jitters 1-68 ms
# and makes Kalibr's time offset garbage); a WARNING when the exposure is not capped
# (ros/exposure.sh capped 8 first, ros/exposure.sh auto after). The bag: ros/maps/rec/calib_<UTC>Z/
# (MCAP, recorded in pepin-vslam beside camera_stream, reliable QoS from ros/params/calib_qos.yaml),
# with record.log, dance.jsonl (every look's answer) and calib_meta.json (the stamp mode and live
# lag, the exposure, the arbiter's speed, the grid, the git commit) inside it. Then:
# ros/calib_run.sh. PEPIN_CALIB_REC is this side's path of the container's /maps/rec (a stub
# test points it at a scratch directory; on the robot it is ros/maps/rec).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$HERE/lib.sh"  # PEPIN_VSLAM_CONTAINER; multiplexed ssh for ros/exposure.sh
CONTAINER="$PEPIN_VSLAM_CONTAINER"
TOPICS=(/camera/image /camera/right/image /head/imu)
ROSBAGS="rosbags==0.11.5"
DRY=0 CHECK=0
TAG=0.025
GRID=()  # neck_dance.py's placement options, as given
usage() {
    echo "usage: ros/calib_record.sh [--check | --dry-run] [--centre PAN TILT] [--distance M] [--height M] [--tag M]"
    exit 2
}
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1 ;;
        --check) CHECK=1 ;;
        --centre) GRID+=(--centre "${2:?}" "${3:?}"); shift 2 ;;
        --distance | --height) GRID+=("$1" "${2:?}"); shift ;;
        --tag) TAG="${2:?}"; shift ;;
        -h | --help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) usage ;;
    esac
    shift
done
flag() { "$HERE/flags.sh" get "$1" "$2" 2>/dev/null | awk '/value is/ {print $NF}' || true; }
tool() { (cd "$REPO" && uv run -q python "$@"); }

if [ "$CHECK" = 1 ]; then  # positioning only: no flag, exposure or recorder is touched
    tool ros/tools/neck_dance.py --check --tag "$TAG"
    exit $?
fi
docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" \
    || { echo "refused: $CONTAINER is not running (ros/laptop.sh vslam)"; exit 2; }
STAMP="$(flag camera_stream camera_stamp)"
if [ "$STAMP" != grab ]; then
    echo "refused: camera_stream's camera_stamp is '${STAMP:-unreadable}', not grab:"
    echo "  ros/flags.sh set camera_stream camera_stamp grab"
    exit 2
fi
LAG="$(flag camera_stream camera_stamp_lag_s)"
[ -n "$LAG" ] || { echo "refused: camera_stream's camera_stamp_lag_s did not read"; exit 2; }
SLOW="$(flag gaze slow_deg_s)"
TIMEOUT="$(flag gaze move_timeout_s)"
if [ -z "$SLOW" ] || [ -z "$TIMEOUT" ]; then
    echo "refused: the gaze arbiter's slow_deg_s / move_timeout_s did not read (ros/laptop.sh nav)"
    exit 2
fi
EXPOSURE="$(mktemp -t pepin_exposure)"
"$HERE/exposure.sh" show >"$EXPOSURE" 2>&1 || echo "!! ros/exposure.sh show failed: $(tail -1 "$EXPOSURE")"
if ! tool ros/tools/head_calib.py exposure <"$EXPOSURE"; then
    echo "!! WARNING: the exposure is not capped: frames smear in the moves and the time offset"
    echo "!!          takes half an unknown exposure (ros/exposure.sh capped 8 first)"
fi
echo "camera_stamp grab, camera_stamp_lag_s $LAG; gaze slow_deg_s $SLOW, move_timeout_s $TIMEOUT"
DANCE=(ros/tools/neck_dance.py ${GRID[@]+"${GRID[@]}"} --tag "$TAG"
       --slow-deg-s "$SLOW" --move-timeout-s "$TIMEOUT")
tool "${DANCE[@]}" || { echo "refused: no dance for this grid (above)"; exit 2; }
if [ "$DRY" = 1 ]; then
    echo "dry run: nothing recorded, nothing moved"
    exit 0
fi

NAME="calib_$(date -u +%Y%m%d_%H%M%S)Z"
REC_DIR="${PEPIN_CALIB_REC:-$HERE/maps/rec}"  # = /maps/rec in the container
BAG="$REC_DIR/$NAME"
LOG="$REC_DIR/$NAME.record.log"
DANCE_LOG="$REC_DIR/$NAME.dance.jsonl"
REC=""
stop_bag() {  # SIGINT to ros2 bag record inside the container (it closes the MCAP), then wait
    [ -n "$REC" ] || return 0
    docker exec "$CONTAINER" pkill -INT -f "ros2 bag record.*$NAME" >/dev/null 2>&1 || true
    for _ in $(seq 1 60); do  # the recorder itself, not the docker client (a ^C ends that one)
        docker exec "$CONTAINER" pgrep -f "ros2 bag record.*$NAME" >/dev/null 2>&1 || break
        sleep 0.5
    done
    wait "$REC" 2>/dev/null || true
    REC=""
}
trap stop_bag EXIT
trap 'stop_bag; exit 130' INT TERM
docker exec "$CONTAINER" /pepin_entrypoint.sh ros2 bag record --storage mcap --output "/maps/rec/$NAME" \
    --qos-profile-overrides-path /params/calib_qos.yaml --topics "${TOPICS[@]}" >"$LOG" 2>&1 &
REC=$!
SUBSCRIBED=0
for _ in $(seq 1 60); do
    SUBSCRIBED="$(grep -c "Subscribed to topic" "$LOG" 2>/dev/null || true)"
    SUBSCRIBED="${SUBSCRIBED:-0}"
    [ "$SUBSCRIBED" -ge "${#TOPICS[@]}" ] && break
    kill -0 "$REC" 2>/dev/null || break
    sleep 0.5
done
if [ "$SUBSCRIBED" -lt "${#TOPICS[@]}" ]; then
    echo "refused: the recorder subscribed to $SUBSCRIBED of ${#TOPICS[@]} topics in 30 s:"
    tail -5 "$LOG"
    exit 1
fi
echo "recording $NAME; the dance starts"
set +e
tool ros/tools/neck_dance.py --move "${DANCE[@]:1}" --log "$DANCE_LOG"
DANCE_EXIT=$?
set -e
sleep 1  # the last still frames and IMU samples past the home look's answer
stop_bag
trap - EXIT INT TERM
[ -d "$BAG" ] || { echo "no bag at $BAG (the recorder's log: $LOG)"; exit 1; }
mv "$LOG" "$BAG/record.log"
[ -f "$DANCE_LOG" ] && mv "$DANCE_LOG" "$BAG/dance.jsonl"
tool ros/tools/head_calib.py meta "$BAG" --camera-stamp "$STAMP" --stamp-lag "$LAG" \
    --exposure-file "$EXPOSURE" --slow-deg-s "$SLOW" --move-timeout-s "$TIMEOUT" \
    ${GRID[@]+"${GRID[@]}"} --tag "$TAG" --dance-exit "$DANCE_EXIT"
rm -f "$EXPOSURE"
(cd "$REPO" && uv run -q --with "$ROSBAGS" python ros/tools/head_calib.py summary "$BAG") || true
if [ "$DANCE_EXIT" != 0 ]; then
    echo "!! the dance did not finish (exit $DANCE_EXIT): the bag is kept, but record again"
    exit 1
fi
echo "next: ros/calib_run.sh $BAG $TAG   (TAG_M: the black square as measured with a ruler)"
