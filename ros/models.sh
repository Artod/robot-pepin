#!/bin/bash
# The learned models on the laptop, beside the containers, as three launchd jobs — three
# processes, three failure domains, so one crashing or hanging never takes another with it:
#   depth         pepin.depth_service on :8790 (the depth network and RAFT-Stereo; the same
#                 command ros/depth_host.sh start runs)
#   localization  pepin.localization_service on :8791 (XFeat, LighterGlue and the BoQ place
#                 model, devices in config/models.json)
#   perception    pepin.perception_service on :8792 (the one camera channel: camera_stream's
#                 rectified left eye in a ring, SCRFD and Apple Vision on the CPU and the Neural
#                 Engine, ArcFace on demand; config/models.json's perception block)
# Docker on macOS has no GPU and no Neural Engine, so these run on the host from the repo's uv
# environment and the containers reach them as http://host.docker.internal:<port> (bound to
# 127.0.0.1: nothing is open on the LAN).
#
# ON DEMAND, NOT AT LOGIN. A job is a launchd job while it is LOADED: launchd starts it, restarts
# it when it dies (KeepAlive, at most once every 30 s) and stops it on `stop`. Its plist lives in
# $PEPIN_LAUNCHD_DIR (~/Library/Application Support/pepin/launchd), NOT in ~/Library/LaunchAgents,
# so nothing loads it at login: the jobs hold ~2.5 GB of models, and they are resident only
# between ros/laptop.sh vslam (which starts them, as it started the depth host before) and
# ros/laptop.sh stop (which stops them), or an explicit start / stop here. JOB is depth,
# localization, perception or all. Usage:
#   ros/models.sh install   [JOB]  write the job's plist and start it
#   ros/models.sh uninstall [JOB]  stop it and remove its plist
#   ros/models.sh start     [JOB]  load an installed job, wait for /health
#   ros/models.sh stop      [JOB]  unload it: the process ends, nothing restarts it, and it stays
#                                  down until the next start
#   ros/models.sh restart   [JOB]  kill it and let launchd start it again
#   ros/models.sh status    [JOB]  launchd's word and each /health, one line
#   ros/models.sh installed JOB    exit 0 when the job is installed (not all)
#   ros/models.sh logs      [JOB]  follow the job's output (not all)
#   ros/models.sh plist     [JOB]  print the plist install would write (not all)
#   ros/models.sh fetch-xfeat      clone XFeat at the commit the image pins into
#                                  models/accelerated_features (network)
# The default target is all (logs: localization). ros/depth_host.sh keeps working for a depth host
# NOT installed here, and hands an installed one to this script. PEPIN_DEPTH_MODEL picks the depth
# model (small), PEPIN_XFEAT_DIR the XFeat checkout (else models/accelerated_features, else
# scratch/xfeat/data/accelerated_features); a checkout is taken only at the commit
# ros/Dockerfile.xfeat pins for RTAB-Map's local fallback (ARG XFEAT_SHA: its git HEAD, or the
# COMMIT file of a copy), because auto registration mixes the two — PEPIN_XFEAT_UNPINNED=1 takes
# another anyway. Nothing here touches a container or the board.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
LOGDIR="$ROOT/logs"
JOBS_DIR="${PEPIN_LAUNCHD_DIR:-$HOME/Library/Application Support/pepin/launchd}"
DOMAIN="gui/$(id -u)"
PREFIX="com.pepin.models"
DEPTH_PORT="${PEPIN_DEPTH_PORT:-8790}"
MODELS_PORT="${PEPIN_MODELS_PORT:-8791}"
PERCEPTION_PORT="${PEPIN_PERCEPTION_PORT:-8792}"
HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
LOG_MAX_BYTES=$((10 * 1024 * 1024))  # launchd never rotates a job's output; start does, past this

usage() {
    echo "usage: ros/models.sh install|uninstall|start|stop|restart|status|logs|plist [depth|localization|perception|all] | installed depth|localization|perception | fetch-xfeat"
    exit 2
}
targets() {  # the jobs a word names
    case "${1:-all}" in
        depth) echo depth ;;
        localization) echo localization ;;
        perception) echo perception ;;
        all) echo "depth localization perception" ;;
        *) usage ;;
    esac
}
label() { echo "$PREFIX.$1"; }
plist() { echo "$JOBS_DIR/$(label "$1").plist"; }
port() {
    case "$1" in
        depth) echo "$DEPTH_PORT" ;;
        perception) echo "$PERCEPTION_PORT" ;;
        *) echo "$MODELS_PORT" ;;
    esac
}
loaded() { launchctl print "$DOMAIN/$(label "$1")" >/dev/null 2>&1; }
pid_of() { launchctl print "$DOMAIN/$(label "$1")" 2>/dev/null | sed -n 's/^[[:space:]]*pid = \([0-9]*\).*/\1/p' | head -1; }
health() { curl -s -m 3 "http://127.0.0.1:$(port "$1")/health"; }

pinned_xfeat() { sed -n 's/^ARG XFEAT_SHA=\([0-9a-f]*\)$/\1/p' "$HERE/Dockerfile.xfeat"; }
xfeat_commit() {  # DIR: the commit an XFeat checkout is at (its git HEAD, or a copy's COMMIT file)
    if [ -e "$1/.git" ]; then git -C "$1" rev-parse HEAD 2>/dev/null || true
    elif [ -s "$1/COMMIT" ]; then head -1 "$1/COMMIT"; fi
}
xfeat_dir() {  # the XFeat checkout the localization job loads its weights from, at the pinned commit
    local d sha got
    sha="$(pinned_xfeat)"
    for d in "${PEPIN_XFEAT_DIR:-}" "$ROOT/models/accelerated_features" \
             "$ROOT/scratch/xfeat/data/accelerated_features"; do
        [ -n "$d" ] && [ -s "$d/weights/xfeat.pt" ] && [ -s "$d/weights/xfeat-lighterglue.pt" ] || continue
        got="$(xfeat_commit "$d")"
        if [ -n "$sha" ] && [ "$got" = "$sha" ]; then echo "$d"; return 0; fi
        if [ "${PEPIN_XFEAT_UNPINNED:-0}" = 1 ]; then
            echo "models: $d is XFeat ${got:-of an unknown commit}, not the pinned ${sha:-?}; taken (PEPIN_XFEAT_UNPINNED=1)" >&2
            echo "$d"; return 0
        fi
        echo "models: $d is XFeat ${got:-of an unknown commit}, the image pins ${sha:-?}: skipped" >&2
    done
    return 1
}
depth_cached() {  # is the depth model in the hub cache? (then the job starts offline)
    local id
    case "${PEPIN_DEPTH_MODEL:-small}" in
        small) id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf" ;;
        base) id="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf" ;;
        large) id="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf" ;;
        */*) id="$PEPIN_DEPTH_MODEL" ;;
        *) return 1 ;;
    esac
    [ -d "$HUB/models--${id//\//--}" ]
}
rotate_log() {  # JOB: its output moved to .out.1 once past LOG_MAX_BYTES, before launchd reopens it
    local f="$LOGDIR/models_$1.out"
    if [ -f "$f" ] && [ "$(stat -f %z "$f")" -gt "$LOG_MAX_BYTES" ]; then mv -f "$f" "$f.1"; fi
}

# The /health JSON as one line per job. %-formatting: the script is single-quoted for the shell.
summary() {
    python3 -c '
import json, sys
h = json.load(sys.stdin)
if "models" in h:
    parts = []
    for name, m in h["models"].items():
        t = m["ms"]["total"]
        parts.append("%s %s on %s: %d served (%d cached), %d refused, %.0f/%.0f ms"
                     % (name, m["tag"], m["device"], m["requests"], m["cache_hits"],
                        m["errors"], t["median"], t["p95"]))
    print("up %.0f s; %s" % (h["uptime_s"], "; ".join(parts)))
elif h.get("service") == "perception":
    f, d, latest = h["frames"], h["detect"], h.get("latest") or {}
    t = d["ms"]["total"]
    idle = "no frame yet" if f["idle_s"] is None else "last frame %.1f s ago" % f["idle_s"]
    print("up %.0f s; %.1f frames/s in (%d in all, %s), detect %.1f/s of %g, %.0f/%.0f ms;"
          " faces %s; persons %s; latest %d faces %d persons"
          % (h["uptime_s"], f["per_s"], f["received"], idle, d["per_s"], d["hz"], t["median"],
             t["p95"], d["faces"], d["persons"], latest.get("faces", 0),
             latest.get("persons", 0)))
else:
    ms = " ".join("%s %.0f/%.0f" % (k, v["median"], v["p95"]) for k, v in h["ms"].items())
    s = h.get("stereo") or {}
    print("up %.0f s; %s on %s: %d frames, %d refused, ms %s; stereo %s: %d pairs"
          % (h["uptime_s"], h["model"].rsplit("/", 1)[-1], h["device"], h["requests"],
             h["errors"], ms, s.get("model", "none"), s.get("requests", 0)))'
}

# The job's plist: the command, its environment, where it logs. uv's path is resolved here,
# because launchd's PATH has no ~/.local/bin or /opt/homebrew/bin. RunAtLoad and KeepAlive apply
# while the job is loaded (start .. stop); ThrottleInterval 30 keeps a job that cannot start (its
# port taken: the service exits at once, before loading a model) from reloading torch every 10 s.
# ProcessType Standard: the scheduler's ordinary priority, as the depth host's shell start had.
write_plist() {  # JOB [PATH]: the plist written to PATH (default: the job's plist)
    local job="$1" dest="${2:-$(plist "$1")}" uv args env_xml="" key
    uv="$(command -v uv)" || { echo "models: no uv on PATH"; return 1; }
    local -a envs=("PATH=$(dirname "$uv"):/usr/bin:/bin:/usr/sbin:/sbin" "HOME=$HOME" "PYTHONUNBUFFERED=1")
    if [ "$job" = depth ]; then
        args=("$uv" run --group depth python -m pepin.depth_service --model "${PEPIN_DEPTH_MODEL:-small}"
              --port "$DEPTH_PORT" --stereo --log-dir "$LOGDIR")
        depth_cached && envs+=("HF_HUB_OFFLINE=1")
    elif [ "$job" = perception ]; then
        args=("$uv" run --group face python -m pepin.perception_service
              --port "$PERCEPTION_PORT" --log-dir "$LOGDIR")
    else
        local dir
        dir="$(xfeat_dir)" || { echo "models: no XFeat checkout at the pinned commit with its weights (ros/models.sh fetch-xfeat, or PEPIN_XFEAT_DIR)"; return 1; }
        args=("$uv" run --group localization python -m pepin.localization_service
              --port "$MODELS_PORT" --log-dir "$LOGDIR")
        envs+=("PEPIN_XFEAT_DIR=$dir")
    fi
    for key in "${envs[@]}"; do
        env_xml+="    <key>${key%%=*}</key><string>${key#*=}</string>"$'\n'
    done
    local args_xml="" a
    for a in "${args[@]}"; do args_xml+="    <string>$a</string>"$'\n'; done
    mkdir -p "$(dirname "$dest")" "$LOGDIR"
    cat > "$dest" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$(label "$job")</string>
  <key>ProgramArguments</key>
  <array>
${args_xml}  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>EnvironmentVariables</key>
  <dict>
${env_xml}  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>ProcessType</key><string>Standard</string>
  <key>StandardOutPath</key><string>$LOGDIR/models_$job.out</string>
  <key>StandardErrorPath</key><string>$LOGDIR/models_$job.out</string>
</dict>
</plist>
EOF
    plutil -lint "$dest" >/dev/null
}

wait_health() {  # JOB: up to 3 min for /health (a first start may fetch a model)
    local job="$1"
    for _ in $(seq 1 180); do
        if H="$(health "$job")" && [ -n "$H" ]; then
            echo "$job: $(summary <<<"$H")"; return 0
        fi
        loaded "$job" || { echo "$job: unloaded while starting: ros/models.sh logs $job"; return 1; }
        sleep 1
    done
    echo "$job did not answer on :$(port "$job") within 3 min: ros/models.sh logs $job"; return 1
}

do_start() {  # JOB: load it if it is not, then wait for /health
    local job="$1"
    [ -f "$(plist "$job")" ] || { echo "$job: not installed (ros/models.sh install $job)"; return 1; }
    if ! loaded "$job"; then
        rotate_log "$job"
        launchctl bootstrap "$DOMAIN" "$(plist "$job")"
    fi
    wait_health "$job"
}

do_stop() {  # JOB: unload it — the process ends and nothing starts it again until a start
    local job="$1"
    if loaded "$job"; then launchctl bootout "$DOMAIN/$(label "$job")"; echo "$job: stopped"
    else echo "$job: was not running under launchd"; fi
}

do_install() {
    local job="$1"
    if [ "$job" = depth ]; then
        "$HERE/depth_host.sh" stop >/dev/null 2>&1 || true  # the pidfile host would hold the port
    elif [ "$job" = perception ]; then
        (cd "$ROOT" && uv run --group face python -c "import insightface, onnxruntime, Vision" ) \
            || { echo "perception: the face group (insightface, onnxruntime) or pyobjc-framework-Vision does not import"; return 1; }
        [ -s "$ROOT/models/face/buffalo_l/det_10g.onnx" ] \
            || echo "perception: no models/face/buffalo_l/det_10g.onnx: it will not start (config/face_sensor.json)"
    else
        (cd "$ROOT" && uv run --group localization python -c "import kornia" ) \
            || { echo "localization: the uv group does not install (network once, then cached)"; return 1; }
    fi
    loaded "$job" && launchctl bootout "$DOMAIN/$(label "$job")" 2>/dev/null || true
    write_plist "$job"
    echo "$job: installed $(plist "$job") (started on demand: ros/laptop.sh vslam, or start here)"
    do_start "$job"
}

do_status() {
    local job="$1" H state
    if ! [ -f "$(plist "$job")" ]; then state="not installed"
    elif loaded "$job"; then state="running under launchd, pid $(pid_of "$job")"
    else state="installed, stopped"; fi
    if H="$(health "$job")" && [ -n "$H" ]; then
        echo "$job ($state): $(summary <<<"$H")"
    else
        echo "$job ($state): not answering on :$(port "$job")"; return 1
    fi
}

ACTION="${1:-status}"
case "$ACTION" in
    install)
        for job in $(targets "${2:-all}"); do do_install "$job"; done ;;
    uninstall)
        for job in $(targets "${2:-all}"); do
            loaded "$job" && launchctl bootout "$DOMAIN/$(label "$job")" || true
            rm -f "$(plist "$job")"; echo "$job: uninstalled"
        done ;;
    start)
        STATUS=0
        for job in $(targets "${2:-all}"); do do_start "$job" || STATUS=1; done
        exit "$STATUS" ;;
    stop)
        for job in $(targets "${2:-all}"); do do_stop "$job"; done ;;
    restart)
        for job in $(targets "${2:-all}"); do
            loaded "$job" || { echo "$job: not running (ros/models.sh start $job)"; exit 1; }
            rotate_log "$job"
            launchctl kickstart -k "$DOMAIN/$(label "$job")"
            sleep 2
            wait_health "$job"
        done ;;
    status)
        STATUS=0
        for job in $(targets "${2:-all}"); do do_status "$job" || STATUS=1; done
        exit "$STATUS" ;;
    installed)
        job="${2:-}"; case "$job" in depth | localization | perception) ;; *) usage ;; esac
        [ -f "$(plist "$job")" ] ;;
    logs)
        job="${2:-localization}"; [ "$job" = all ] && usage; targets "$job" >/dev/null
        exec tail -n 50 -F "$LOGDIR/models_$job.out" ;;
    plist)
        job="${2:-localization}"; [ "$job" = all ] && usage; targets "$job" >/dev/null
        TMP="$(mktemp -d)"; write_plist "$job" "$TMP/$(label "$job").plist"
        cat "$TMP/$(label "$job").plist"; rm -rf "$TMP" ;;
    fetch-xfeat)
        SHA="$(pinned_xfeat)"
        [ -n "$SHA" ] || { echo "no ARG XFEAT_SHA in ros/Dockerfile.xfeat"; exit 1; }
        DEST="$ROOT/models/accelerated_features"
        if [ -s "$DEST/weights/xfeat.pt" ]; then
            [ "$(xfeat_commit "$DEST")" = "$SHA" ] && { echo "already there: $DEST at $SHA"; exit 0; }
            git -C "$DEST" fetch -q origin && git -C "$DEST" checkout -q "$SHA"
            echo "XFeat moved to $SHA in $DEST"; exit 0
        fi
        mkdir -p "$ROOT/models"
        git clone -q https://github.com/verlab/accelerated_features.git "$DEST"
        git -C "$DEST" checkout -q "$SHA"
        echo "XFeat $SHA in $DEST" ;;
    *) usage ;;
esac
