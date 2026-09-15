#!/bin/bash
# The operator's window on the robot: the laptop's foxglove_bridge and the desktop app that draws
# it. Usage:
#
#   ros/foxglove.sh check    one PASS/FAIL line per item: the container that carries the bridge,
#                            the listening port, the websocket handshake, the serverInfo and the
#                            channel count, every topic the layout draws, and how many times the
#                            bridge has died since the container started
#   ros/foxglove.sh reopen   tell the running desktop app to (re)connect to ws://localhost:8765
#                            through the foxglove:// deep link; if the app is not running, print
#                            the link instead of starting it
#   ros/foxglove.sh url      print the deep link and exit
#
# Why `reopen` exists. A Foxglove client is bound to ONE bridge process: channel ids are that
# process's own numbering and start again at 1 when it restarts, and the laptop's bridge restarts
# with every `ros/laptop.sh vslam` (the container is recreated) and with every restart of the
# launch. The app keeps the dead socket's panels on screen, empty, until it reconnects — which is
# the "panels load, then vanish" everyone has been asking about. The deep link is the reconnect,
# done by the script instead of by hand.
#
# Exit status: 1 if any check failed (or, for `reopen`, if the link could not be opened).
# PEPIN_FOXGLOVE_HOST / _PORT move the bridge; PEPIN_FOXGLOVE_LAYOUT names a layout file.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="${PEPIN_FOXGLOVE_CONTAINER:-pepin-vslam}"
HOST="${PEPIN_FOXGLOVE_HOST:-localhost}"
PORT="${PEPIN_FOXGLOVE_PORT:-8765}"
SECONDS_TO_READ="${PEPIN_FOXGLOVE_READ_S:-4}"
# The desktop app: /Applications/Foxglove.app since v2 ("Foxglove Studio" was the old name and the
# old process name — `pgrep -x "Foxglove Studio"` matches nothing on this laptop).
APP_PROCESS="${PEPIN_FOXGLOVE_APP:-Foxglove}"
PREFIX="${PEPIN_FOXGLOVE_PREFIX:-fg}"   # how the PASS/FAIL lines are numbered

usage() {
    echo "usage: ros/foxglove.sh check|reopen|url"
    exit 2
}

ws_url() { printf 'ws://%s:%s\n' "$HOST" "$PORT"; }

# The app's deep link. The scheme (`foxglove://`) and the `open` host are declared in
# /Applications/Foxglove.app/Contents/Info.plist (CFBundleURLSchemes) and app.asar
# (`foxglove://open`, `ds`, `ds.url`, `layoutId`); the websocket URL is percent-encoded because it
# is a query VALUE, colons and slashes included.
deep_link() {
    local encoded
    encoded="$(printf 'ws://%s:%s' "$HOST" "$PORT" | sed 's|:|%3A|g; s|/|%2F|g')"
    printf 'foxglove://open?ds=foxglove-websocket&ds.url=%s\n' "$encoded"
}

FAILED=0; N=0
line() {  # OK TEXT -> one numbered PASS/FAIL line
    N=$((N + 1))
    if [ "$1" = true ]; then
        printf 'PASS %-5s %s\n' "$PREFIX.$N" "$2"
    else
        FAILED=$((FAILED + 1))
        printf 'FAIL %-5s %s\n' "$PREFIX.$N" "$2"
    fi
}

check() {
    local started log starts restarts

    if started="$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER" 2>/dev/null)" &&
        [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = true ]; then
        line true "$CONTAINER is up since $started (it carries the bridge)"
    else
        line false "$CONTAINER is not running — nothing serves Foxglove (ros/laptop.sh vslam --neck)"
        started=""
    fi

    # bash's own /dev/tcp: no nc, no timeout, no coreutils assumption on macOS.
    if (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null; then
        line true "port $PORT accepts connections on $HOST"
    else
        line false "port $PORT refuses connections on $HOST (is -p $PORT:$PORT still on the container?)"
    fi

    # The handshake, the serverInfo, the channels and every topic of the layout: one PASS/FAIL
    # line each, printed by the probe with our numbering continued.
    local probe_out probe_status=0
    probe_out="$(python3 "$HERE/tools/foxglove_probe.py" \
        --host "$HOST" --port "$PORT" --seconds "$SECONDS_TO_READ" --prefix "$PREFIX" \
        --number-from "$N" 2>&1)" || probe_status=$?
    printf '%s\n' "$probe_out"
    N=$((N + $(grep -c '^\(PASS\|FAIL\)' <<<"$probe_out" || true)))
    [ "$probe_status" -eq 0 ] || FAILED=$((FAILED + $(grep -c '^FAIL' <<<"$probe_out" || true)))

    # How stable the socket has been. Every one of these lines is one dropped connection in the
    # app, with every channel id renumbered behind it: a client that was connected before it is
    # showing empty panels now.
    if [ -n "$started" ]; then
        log="$(docker logs --since "$started" "$CONTAINER" 2>&1 || true)"
        starts="$(grep -ac 'Starting foxglove_bridge' <<<"$log" || true)"
        restarts="$(docker inspect -f '{{.RestartCount}}' "$CONTAINER" 2>/dev/null || echo '?')"
        if [ "${starts:-0}" -le 1 ] 2>/dev/null; then
            line true "the bridge has started ${starts:-0}x since the container came up (docker restarts $restarts)"
        else
            line false "the bridge has started ${starts}x since the container came up (docker restarts $restarts) — each death drops the app's connection and renumbers every channel: $(grep -a 'bridge watch:.*restarting\|was required: shutting down' <<<"$log" | tail -1 | cut -c1-120)"
        fi
    fi
}

wait_for_port() {  # up to PEPIN_FOXGLOVE_WAIT_S for the bridge to listen; 0 = do not wait
    local limit="${PEPIN_FOXGLOVE_WAIT_S:-30}" waited=0
    while ! (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null; do
        [ "$waited" -lt "$limit" ] || return 1
        sleep 1; waited=$((waited + 1))
    done
    [ "$waited" -eq 0 ] || echo "foxglove: the bridge started listening after ${waited} s"
}

reopen() {
    local link; link="$(deep_link)"
    # A link fired at a port that is not up yet is a connection refused in the app, and the app
    # does not try again by itself: the container is ~10 s old when vslam prints its line.
    if ! wait_for_port; then
        echo "foxglove: nothing listens on $(ws_url) — not asking the app to connect to it"
        return 1
    fi
    if ! pgrep -x "$APP_PROCESS" >/dev/null 2>&1; then
        echo "foxglove: the app is not running; open it and connect to $(ws_url)"
        echo "  or: open '$link'"
        return 0
    fi
    # -g: the app reconnects without stealing the screen from whatever is being read right now.
    if open -g "$link" 2>/dev/null; then
        echo "foxglove: the app was told to reconnect to $(ws_url) (its old socket died with the bridge)"
    else
        echo "foxglove: could not open $link — connect to $(ws_url) by hand"
        return 1
    fi
}

case "${1:-}" in
    check)
        check
        if [ "$FAILED" -eq 0 ]; then
            echo "foxglove: $N checks, none failed"
        else
            echo "foxglove: $FAILED of $N checks failed"
            exit 1
        fi
        ;;
    reopen) reopen ;;
    url) deep_link ;;
    *) usage ;;
esac
