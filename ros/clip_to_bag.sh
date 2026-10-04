#!/usr/bin/env bash
# A drive's board-side camera clip as a camera bag (ros/tools/clip_to_bag.py): the four stereo
# topics, dated by the V4L2 capture (grab) stamps the clip's headers carry, beside the drive's bag.
#
#   ros/clip_to_bag.sh 0512                  # ros/maps/rec/0512_*_cam.mjpeg -> 0512_*_cam.bag
#   ros/clip_to_bag.sh 0512 --require-grab   # refuse a clip recorded without the capture stamps
#   ros/clip_to_bag.sh 0512 --start 10 --end 70
#   ros/clip_to_bag.sh --help
#
# Then: ros2 bag play --clock <drive>.bag <run>_cam.bag (ros/README.md, "Replay").
# Runs in a throwaway container of the laptop image with no network at all (the ros/replay.sh
# pattern): nothing it does can reach the robot or the live containers. PEPIN_REPLAY_REC moves
# the recordings (a worktree has no ros/maps/rec of its own), PEPIN_CLIP_IMAGE the image.
set -euo pipefail

ROS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$ROS_DIR")"
IMAGE="${PEPIN_CLIP_IMAGE:-pepin-laptop:xfeat}"
REC="${PEPIN_REPLAY_REC:-$ROS_DIR/maps/rec}"

case "${1:-}" in
    "" | -h | --help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac
RUN="$1"; shift
shopt -s nullglob
clips=("$REC"/"$RUN"_*_cam.mjpeg)
if [ "${#clips[@]}" -ne 1 ]; then
    echo "expected one clip $REC/${RUN}_*_cam.mjpeg, found ${#clips[@]}" >&2
    exit 2
fi
clip="$(basename "${clips[0]}")"

docker run --rm --network none \
    -v "$REPO:/repo:ro" -v "$REC:/rec" \
    -v "$REPO/src/pepin:/ws/pepin_src/pepin:ro" \
    -v "$ROS_DIR/pepin_bringup/pepin_bringup:/ws/install/pepin_bringup/lib/python3.12/site-packages/pepin_bringup:ro" \
    -e PYTHONDONTWRITEBYTECODE=1 -e ROS_DOMAIN_ID=79 -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    --entrypoint /bin/bash "$IMAGE" -c \
    'source /opt/ros/jazzy/setup.bash && [ -f /ws/install/setup.bash ] && source /ws/install/setup.bash; exec python3 /repo/ros/tools/clip_to_bag.py "$@"' \
    clip_to_bag "/rec/$clip" --config /repo/config/camera.json "$@"
