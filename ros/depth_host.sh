#!/bin/bash
# The depth network on the laptop's own GPU, beside the containers: pepin.depth_service served
# from the repo's uv environment (dependency group "depth"), which the depth node in pepin-vslam
# reaches as http://host.docker.internal:8790 (Docker Desktop forwards that name to the host's
# loopback, so the service is bound to 127.0.0.1 and nothing is open on the LAN). Usage:
#   ros/depth_host.sh start [MODEL]   start (or restart) the service; MODEL small (default), base,
#                                     large, or a Hugging Face id; PEPIN_DEPTH_MODEL sets the default
#   ros/depth_host.sh stop
#   ros/depth_host.sh status          the service's /health: model, device, frames, per-stage ms
#   ros/depth_host.sh bench [N]       time N frames (default 30) through the network here and, when
#                                     the service is up, through it (JPEG and raw); the frames come
#                                     from scratch/_depth_bench/cam when that directory exists
# Off by default: ros/laptop.sh vslam starts it only with PEPIN_DEPTH_HOST=1, and then tells the
# node to use it. The weights come from the Hugging Face cache (~/.cache/huggingface/hub, the same
# files the laptop image carries); a model already cached is loaded offline, a new one is fetched.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PORT="${PEPIN_DEPTH_PORT:-8790}"
URL="http://127.0.0.1:$PORT"
PIDFILE="${TMPDIR:-/tmp}/pepin-depth-host.pid"
LOGDIR="$ROOT/logs"
HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"

health() { curl -s -m 3 "$URL/health"; }
running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
# The /health JSON as one line. %-formatting, not f-strings: the script is single-quoted for the
# shell, so the quotes an f-string needs around its dict keys cannot be escaped inside it.
summary() {
    python3 -c '
import json, sys
h = json.load(sys.stdin)
ms = " ".join("%s %.0f/%.0f" % (k, v["median"], v["p95"]) for k, v in h["ms"].items())
print("%s on %s: %d frames served, %d refused, up %.0f s, ms median/p95: %s"
      % (h["model"].rsplit("/", 1)[-1], h["device"], h["requests"], h["errors"],
         h["uptime_s"], ms))'
}
cached() {  # is MODEL ($1) in the hub cache? (short names map to the metric-indoor ids)
    case "$1" in
        small) id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf" ;;
        base) id="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf" ;;
        large) id="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf" ;;
        */*) id="$1" ;;
        *) return 1 ;;
    esac
    [ -d "$HUB/models--${id//\//--}" ]  # the hub's directory name for any org/model
}
stop_quiet() {
    if running; then
        kill "$(cat "$PIDFILE")" 2>/dev/null || true
        for _ in $(seq 1 50); do running || break; sleep 0.2; done
    fi
    rm -f "$PIDFILE"
    # The pidfile holds `uv run`; if it exited without passing the signal on, the python child
    # would keep the port. Named by its port, so a bench or another port is never touched.
    pkill -f "pepin\.depth_service .*--port $PORT" 2>/dev/null || true
}

case "${1:-status}" in
    start)
        MODEL="${2:-${PEPIN_DEPTH_MODEL:-small}}"
        stop_quiet
        mkdir -p "$LOGDIR"
        # A cached model never touches the network at start (the hub's HEAD checks would hang a
        # start without internet); an uncached one is allowed to download itself once.
        if cached "$MODEL"; then export HF_HUB_OFFLINE=1; else echo "depth host: $MODEL is not cached yet, fetching"; fi
        cd "$ROOT"
        nohup uv run --group depth python -m pepin.depth_service --model "$MODEL" --port "$PORT" \
            --log-dir "$LOGDIR" >"$LOGDIR/depth_host.out" 2>&1 &
        echo $! > "$PIDFILE"
        for _ in $(seq 1 180); do  # Small loads in 2 s from cache, Large in 3 s, a download longer
            if health >/dev/null 2>&1; then
                echo "depth host up on $URL: $(health | summary)"
                exit 0
            fi
            running || { echo "depth host died at start: $LOGDIR/depth_host.out"; tail -5 "$LOGDIR/depth_host.out"; exit 1; }
            sleep 1
        done
        echo "depth host did not answer on $URL within 3 min: $LOGDIR/depth_host.out"; exit 1 ;;
    stop)
        if running; then stop_quiet; echo "depth host stopped"; else rm -f "$PIDFILE"; echo "depth host was not running"; fi ;;
    status)
        if H="$(health)"; then echo "$H" | summary; else echo "depth host is not answering on $URL (ros/depth_host.sh start)"; exit 1; fi ;;
    bench)
        ARGS=(--bench "${2:-30}")
        [ -d "$ROOT/scratch/_depth_bench/cam" ] && ARGS+=(--frames "$ROOT/scratch/_depth_bench/cam")
        health >/dev/null 2>&1 && ARGS+=(--url "$URL")
        MODEL="${PEPIN_DEPTH_MODEL:-small}"
        cached "$MODEL" && export HF_HUB_OFFLINE=1
        cd "$ROOT"
        exec uv run --group depth python -m pepin.depth_service --model "$MODEL" --log-dir "$LOGDIR" "${ARGS[@]}" ;;
    *)
        echo "usage: ros/depth_host.sh [start [MODEL] | stop | status | bench [N]]"; exit 2 ;;
esac
