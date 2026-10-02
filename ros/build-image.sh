#!/bin/bash
# Build the board's sensor image on this Mac (linux/arm64: native on Apple Silicon, minutes instead
# of the board's half hour) and ship it to the board as a tarball over ssh.
#
# Usage:
#   ros/build-image.sh                 build only: pepin-ros:sensors here, the board untouched
#   ros/build-image.sh --ship          build, then load it on the board
#   ros/build-image.sh --ship-only     ship the image already built here
# A ship refuses a running stack (--force overrides), a board whose /etc/default/pepin-ros still
# says PEPIN_NAV=true or PEPIN_SLAM_TOOLBOX=true, and under 2 GB free on docker's root; it tags
# the rollback first, loads, and installs board/pepin-ros.service (daemon-reload, no restart).
#   PEPIN_BUILD_CPUS=4 ros/build-image.sh   build on 4 of the Docker VM's CPUs at low priority
#                                           (a builder of its own; the live containers keep theirs)
#
# Tags. Here the image is pepin-ros:sensors and nothing else: this Mac's pepin-ros:latest and
# :zenoh are the older image with Nav2 in it, which ros/Dockerfile no longer builds. On the board
# the load tags it pepin-ros:latest and pepin-ros:zenoh, the names ros/run.sh and
# pepin-zrouter.service ask for. The board's image from before the first sensors-only load keeps
# the tag pepin-ros:pre-sensors-2026-10-01 (set once, never moved by a later ship).
#
# Rollback to that image (restarts the router and the stack, the router first; nothing restarts
# when a tag fails):
#   ssh root@10.0.0.187 'docker tag pepin-ros:pre-sensors-2026-10-01 pepin-ros:zenoh && docker tag pepin-ros:pre-sensors-2026-10-01 pepin-ros:latest && systemctl stop pepin-ros && systemctl restart pepin-zrouter && systemctl start pepin-ros'
#
# Shipping is a separate step on purpose: `docker load` on the board while the robot drives is
# refused (--force overrides). Neither step restarts anything: the restart is printed at the end.
set -euo pipefail

BOARD="${PEPIN_HOST:-10.0.0.187}"
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE=pepin-ros
LOCAL="$IMAGE:sensors"
BOARD_TAGS=("$IMAGE:latest" "$IMAGE:zenoh")
ROLLBACK="$IMAGE:pre-sensors-2026-10-01"
# The multiplexed master of ros/lib.sh is deliberately NOT used here: this script runs while the
# robot is being driven from another session, and a shared master is a shared failure.
SSH=(ssh -o ControlPath=none -o ConnectTimeout=6)

DO_BUILD=1
DO_SHIP=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --ship) DO_SHIP=1 ;;
        --ship-only) DO_SHIP=1; DO_BUILD=0 ;;
        --force) FORCE=1 ;;
        *) echo "usage: $0 [--ship|--ship-only] [--force]" >&2; exit 2 ;;
    esac
done

size_of() { docker image inspect "$1" --format '{{.Size}}' | awk '{printf "%.2f GB", $1/1e9}'; }

if [ "$DO_BUILD" = 1 ]; then
    # The build context is ros/ with the library beside it (the Dockerfile's COPY pepin_src).
    mkdir -p "$HERE/pepin_src/pepin"
    rsync -a --delete --exclude '__pycache__' "$HERE/../src/pepin/" "$HERE/pepin_src/pepin/"

    BUILDX=(docker buildx build)
    if [ -n "${PEPIN_BUILD_CPUS:-}" ]; then
        # A docker-container builder pinned to the VM's LAST N CPUs with a tenth of the default
        # CPU weight: under contention the live stack wins. It keeps its own layer cache.
        JOBS="$PEPIN_BUILD_CPUS"
        TOTAL=$(docker info --format '{{.NCPU}}')
        BUILDER="pepin-board-${JOBS}cpu"
        docker buildx inspect "$BUILDER" >/dev/null 2>&1 \
            || docker buildx create --name "$BUILDER" --driver docker-container \
                --driver-opt "cpuset-cpus=$((TOTAL - JOBS))-$((TOTAL - 1))" \
                --driver-opt cpu-shares=102 >/dev/null
        BUILDX+=(--builder "$BUILDER")
    else
        JOBS="${PEPIN_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"
    fi

    echo "building $LOCAL for linux/arm64 with RF2O_JOBS=$JOBS"
    T0=$(date +%s)
    "${BUILDX[@]}" --platform linux/arm64 --load \
        --build-arg "RF2O_JOBS=$JOBS" \
        -t "$LOCAL" -f "$HERE/Dockerfile" "$HERE"
    echo "build took $(( $(date +%s) - T0 )) s"

    # A wrong architecture is silent until the board refuses to start the container.
    ARCH=$(docker image inspect "$LOCAL" --format '{{.Os}}/{{.Architecture}}')
    [ "$ARCH" = linux/arm64 ] || { echo "built $ARCH, not linux/arm64" >&2; exit 1; }
    echo "$LOCAL: $ARCH, $(size_of "$LOCAL")"
fi

if [ "$DO_SHIP" = 1 ]; then
    # Loading an image under a driving robot is how a drive is lost; the stack must be down.
    if "${SSH[@]}" "root@$BOARD" 'docker ps --format "{{.Names}}" | grep -qx pepin-ros'; then
        [ "$FORCE" = 1 ] || { echo "pepin-ros is running on the board: stop it first (systemctl stop pepin-ros), or --force" >&2; exit 1; }
    fi
    # A board still told to run navigation is refused: the sensors-only unit and launch have no
    # such switch, and the board would come up without what its file asks for.
    NAV_LINES=$("${SSH[@]}" "root@$BOARD" "grep -E '^PEPIN_(NAV|SLAM_TOOLBOX)=' /etc/default/pepin-ros || true")
    echo "the board's /etc/default/pepin-ros: ${NAV_LINES:-no PEPIN_NAV or PEPIN_SLAM_TOOLBOX line}" | tr '\n' ' '; echo
    if printf '%s\n' "$NAV_LINES" | grep -qE '=true$'; then
        echo "refused: the board is set to run navigation; set those lines to false first" >&2
        exit 1
    fi
    # Room for the layers on docker's own filesystem, refused below 2 GB.
    FREE_KB=$("${SSH[@]}" "root@$BOARD" 'df -Pk "$(docker info --format "{{.DockerRootDir}}")" | awk "NR == 2 {print \$4}"')
    case "$FREE_KB" in ''|*[!0-9]*) echo "refused: could not read the board's free disk ('$FREE_KB')" >&2; exit 1 ;; esac
    echo "free on the board's docker root: $((FREE_KB / 1024)) MB"
    if [ "$FREE_KB" -lt $((2 * 1024 * 1024)) ]; then
        echo "refused: $((FREE_KB / 1024)) MB free on the board's docker root, under 2048 MB" >&2
        exit 1
    fi
    # The rollback tag, once, and a hard step: a second ship must not move it onto a sensors-only
    # image, and a tag that failed stops the ship before anything is loaded.
    "${SSH[@]}" "root@$BOARD" "set -e
        if ! docker image inspect $ROLLBACK >/dev/null 2>&1; then docker tag ${BOARD_TAGS[1]} $ROLLBACK; fi
        ID=\$(docker image inspect --format '{{.Id}}' $ROLLBACK)
        echo \"rollback: $ROLLBACK = \$ID\""
    # The tarball goes over the wire AS IT IS: Docker Desktop's containerd image store saves
    # layers already compressed (zstd -3 took 0.7 % off it, 2026-09-22). PEPIN_SHIP_COMPRESS=zstd
    # or gzip is for a daemon whose store writes plain tars; the board must have the same tool.
    PACK=(cat); UNPACK='docker load'
    case "${PEPIN_SHIP_COMPRESS:-}" in
        zstd) PACK=(zstd -3 -T0 -c); UNPACK='zstd -d -c | docker load' ;;
        gzip) PACK=(gzip -1 -c);     UNPACK='gzip -d -c | docker load' ;;
    esac
    echo "shipping $LOCAL to $BOARD as ${BOARD_TAGS[*]} ($(size_of "$LOCAL"), pipe: ${PACK[0]})"
    T0=$(date +%s)
    docker save "$LOCAL" | "${PACK[@]}" \
        | "${SSH[@]}" "root@$BOARD" "$UNPACK && docker tag $LOCAL ${BOARD_TAGS[0]} && docker tag $LOCAL ${BOARD_TAGS[1]}"
    echo "ship took $(( $(date +%s) - T0 )) s"
    "${SSH[@]}" "root@$BOARD" "docker images $IMAGE"
    # The sensors-only unit goes with the image (installed, not restarted).
    scp -q -o ControlPath=none -o ConnectTimeout=6 "$HERE/../board/pepin-ros.service" \
        "root@$BOARD:/etc/systemd/system/pepin-ros.service"
    "${SSH[@]}" "root@$BOARD" "systemctl daemon-reload"
    echo "the image and board/pepin-ros.service are installed, nothing was restarted. To run them"
    echo "(the router uses the image too):"
    echo "  ssh root@$BOARD 'systemctl stop pepin-ros && systemctl restart pepin-zrouter && systemctl start pepin-ros'"
    echo "Code, params and config still travel separately (ros/sync.sh): the container mounts them."
fi
