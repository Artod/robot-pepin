#!/usr/bin/env bash
# The obstacle-pipeline replay stand (ros/README.md, "Replay"): the recorded drives through Nav2's
# own costmaps with candidate parameters, faster than real time, one fixed score per drive.
#
#   ros/replay.sh 483-498                                  # the current parameters, a table
#   ros/replay.sh 483-498 --set both.camera_layer.enabled=false \
#       --against ros/replay/baselines/0483-0498.json      # the delta per drive
#   ros/replay.sh 483-498 --params scratch/candidate.yaml  # a parameters file layered last
#   ros/replay.sh 483-498 --save ros/replay/baselines/0483-0498.json
#   ros/replay.sh --help
#
# Runs in a throwaway container of the board's image (the same nav2_costmap_2d build) with no
# network at all: nothing it does can reach the robot, the board or the live containers. The
# engine is built into the cache on first use (~10 s) and whenever its source changes.
# PEPIN_REPLAY_REC / PEPIN_REPLAY_MAPS / PEPIN_REPLAY_CACHE move the bags, the database and the
# cache (a worktree has no ros/maps/rec of its own); PEPIN_REPLAY_IMAGE the image.
set -euo pipefail

ROS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$ROS_DIR")"
IMAGE="${PEPIN_REPLAY_IMAGE:-pepin-ros:latest}"
REC="${PEPIN_REPLAY_REC:-$ROS_DIR/maps/rec}"
MAPS="${PEPIN_REPLAY_MAPS:-$ROS_DIR/maps}"
CACHE="${PEPIN_REPLAY_CACHE:-$ROS_DIR/replay/.cache}"

save=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--save" ]]; then
        save="${args[$((i + 1))]:-}"
    fi
done

mkdir -p "$CACHE"
rm -f "$CACHE/last.json" "$CACHE/last.txt"
git_sha="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no -- ros/params ros/replay 2>/dev/null)" ]]; then
    git_sha="$git_sha+dirty"
fi

docker run --rm --network none \
    -v "$REPO:/repo:ro" -v "$REC:/rec:ro" -v "$MAPS:/maps:ro" -v "$CACHE:/cache" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PEPIN_REPLAY_HOST_REPO="$REPO" -e PEPIN_REPLAY_HOST_CWD="$PWD" -e PEPIN_REPLAY_GIT="$git_sha" \
    -e ROS_DOMAIN_ID=231 -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    --entrypoint /pepin_entrypoint.sh "$IMAGE" python3 /repo/ros/replay/replay.py "$@"

if [[ -n "$save" ]]; then
    cp "$CACHE/last.json" "$save"
    cp "$CACHE/last.txt" "${save%.json}.txt"
    echo "saved $save and ${save%.json}.txt"
fi
