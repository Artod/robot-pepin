#!/bin/bash
# Every thread's stack of a node that is "quietly stuck" (alive, no output, little CPU), taken
# BEFORE anyone kicks it: a kick destroys exactly the state that says why.
#
#   ros/tools/stack.sh NODE [CONTAINER]   e.g. ros/tools/stack.sh gaze
#                                              ros/tools/stack.sh run_subscribe_msckf pepin-vio
#
# NODE is a pepin_bringup module (gaze, visual_odometry, ...) or an executable's name (rtabmap,
# run_subscribe_msckf, component_container_isolated, ...); CONTAINER defaults to the node's home
# (pepin.deployment.node_host: a laptop container, else the board's pepin-ros). Out:
# ros/maps/rec/stacks/<node>-<UTC>.txt, its path printed last.
#   Python node: SIGUSR2, which node_kit's faulthandler answers with every thread's Python stack on
#     the node's stderr (the container's log; the node keeps running), copied from the log; on the
#     laptop also py-spy dump --native (Python and C frames) from a sidecar.
#   C++ process: gdb -batch "thread apply all bt" from a sidecar sharing the container's PID
#     namespace (laptop only: the images carry no gdb, and the board has no sidecar image).
# The sidecar is pepin-stack:latest (pepin-ros:sensors + gdb + py-spy), built here on first use
# (needs the network once). Nothing is restarted; the process is stopped only while gdb or
# py-spy read it (a second or two).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # ros/
NODE="${1:?usage: ros/tools/stack.sh NODE [CONTAINER]}"
BOARD="${PEPIN_HOST:-10.0.0.187}"
if [ -n "${2:-}" ]; then
    SIDE=laptop; CONTAINER="$2"
else
    read -r SIDE CONTAINER < <(cd "$HERE/.." && uv run -q python -c \
        "import sys; from pepin.deployment import node_host; print(*node_host(sys.argv[1]))" "$NODE")
fi
OUTDIR="$HERE/maps/rec/stacks"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/$NODE-$(date -u +%Y%m%dT%H%M%SZ).txt"
in_container() {  # command...: inside CONTAINER, on the laptop or the board
    if [ "$SIDE" = laptop ]; then
        docker exec "$CONTAINER" "$@"
    else
        ssh "root@$BOARD" "docker exec $CONTAINER $(printf '%q ' "$@")"
    fi
}
container_log_since() {  # RFC3339 time: the container's log from then on
    if [ "$SIDE" = laptop ]; then
        docker logs --since "$1" "$CONTAINER" 2>&1
    else
        ssh "root@$BOARD" "docker logs --since $1 $CONTAINER 2>&1"
    fi
}
# The pid: a pepin_bringup module's command line, else the executable by name.
PIDS="$(in_container sh -c 'pgrep -f "pepin_bringup[./]$1( |$)" || pgrep -x "$(echo "$1" | cut -c1-15)" || true' sh "$NODE")"
PID="$(echo "$PIDS" | head -1)"
[ -n "$PID" ] || { echo "no process for $NODE in $CONTAINER ($SIDE)"; exit 3; }
CMD="$(in_container sh -c 'tr "\0" " " < /proc/$1/cmdline' sh "$PID")"
{
    echo "# $NODE in $CONTAINER ($SIDE), pid $PID (all: $(echo "$PIDS" | tr '\n' ' '))"
    echo "# $CMD"
    echo "# taken $(date -u +%FT%TZ)"
} > "$OUT"
sidecar() {  # command...: in pepin-stack:latest sharing CONTAINER's PID namespace
    if ! docker image inspect pepin-stack:latest >/dev/null 2>&1; then
        printf 'FROM pepin-ros:sensors\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends gdb python3-pip >/dev/null && pip3 install -q --break-system-packages py-spy && rm -rf /var/lib/apt/lists/*\n' \
            | docker build -q -t pepin-stack:latest - >/dev/null
    fi
    docker run --rm --pid "container:$CONTAINER" --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
        --network none --entrypoint "$1" pepin-stack:latest "${@:2}"
}
case "$CMD" in
    *python*)
        SINCE="$(date -u +%FT%TZ)"
        in_container kill -USR2 "$PID"
        sleep 2
        { echo "## faulthandler (SIGUSR2) from the container's log"; container_log_since "$SINCE"; } >> "$OUT"
        if [ "$SIDE" = laptop ]; then
            { echo "## py-spy dump --native"; sidecar py-spy dump --native --pid "$PID" 2>&1 || true; } >> "$OUT"
        fi ;;
    *)
        if [ "$SIDE" != laptop ]; then
            echo "## a C++ process on the board: no gdb there (take it on the board by hand)" >> "$OUT"
        else
            { echo "## gdb thread apply all bt"
              sidecar gdb -iex "set sysroot /proc/$PID/root" -p "$PID" -batch -ex "set pagination off" \
                  -ex "thread apply all bt" 2>&1 || true; } >> "$OUT"
        fi ;;
esac
echo "$(grep -cE '^(Thread |Current thread)' "$OUT" || true) thread stacks; $OUT"
