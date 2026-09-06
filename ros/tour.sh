#!/bin/bash
# The comparison lap: the robot drives a fixed route on the named map, autonomously, while the
# laptop records its head camera and the board logs scans, odometry and AMCL poses. Run it on the
# old map and on the new one from the same base spot and compare the two recordings.
#   ros/tour.sh lap3      old map (2026-09-03)      ros/tour.sh flat3     new map (2026-09-05)
# Ctrl-C at any moment cancels the Nav2 task and ends the lap; ros/stop.sh is the red button.
set -uo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
HERE="$(cd "$(dirname "$0")" && pwd)"
MAP="${1:?usage: ros/tour.sh lap3|flat3}"
STAMP=$(date +%Y%m%d_%H%M%S)
CAM="$HERE/maps/rec/${STAMP}_tour_${MAP}_cam.mkv"
mkdir -p "$HERE/maps/rec"
# the same route on both maps: west along the flat in three hops, then back to the base
STOPS="-2.92 2.28  -4.43 2.35  -6.11 2.17  0.0 0.0"

echo "[1/4] switching to nav on the $MAP map..."
YAML=$(ssh -o ConnectTimeout=5 "root@$BOARD" "ls /root/pepin-ros/maps/*${MAP}*.yaml 2>/dev/null | head -1")
[ -n "$YAML" ] || { echo "no *${MAP}*.yaml in the board's maps dir"; exit 1; }
"$HERE/mode.sh" nav "/maps/$(basename "$YAML")" >/dev/null || exit 1
echo "[2/4] recorders first (a refused lap still leaves its scans): head camera (laptop) + scans/odom/AMCL (board)"
ffmpeg -loglevel error -y -f mjpeg -use_wallclock_as_timestamps 1 -i "http://$BOARD:8080/stream" -c copy "$CAM" &
FFPID=$!
for i in $(seq 1 15); do ssh "root@$BOARD" "docker ps --format {{.Names}} | grep -q ^pepin-ros\$" && break; sleep 1; done
ssh "root@$BOARD" "docker exec -d pepin-ros /pepin_entrypoint.sh python3 /tools/session_logger.py /maps/rec/${STAMP}_tour_${MAP}.jsonl"
sleep 3
kill -0 $FFPID 2>/dev/null || { echo "camera recording died — not driving"; exit 1; }
finish() {
    echo; echo "[4/4] STOP: cancelling the navigation task on the board..."
    "$HERE/stop.sh"   # cancel the goal; a single zero twist is useless against a controller at 5 Hz
    watch_stop
    echo "stopping recorders, fetching..."
    ssh "root@$BOARD" "docker exec pepin-ros pkill -INT -f session_logger.py" || true
    kill -INT $FFPID 2>/dev/null; wait $FFPID 2>/dev/null
    ssh "root@$BOARD" "docker logs pepin-ros 2>&1" > "$HERE/maps/rec/${STAMP}_tour_${MAP}_board.log" 2>/dev/null || true
    rsync -a "root@$BOARD:/root/pepin-ros/maps/rec/" "$HERE/maps/rec/" || true
    echo "camera: $CAM"; ls -la "$HERE/maps/rec/" | grep "$STAMP" || true
    echo "DONE — Claude compares the recordings."
}
trap finish EXIT

echo -n "    stack starting (measured 15.5 s to Nav2 active)..."
for i in $(seq 1 45); do
    ssh "root@$BOARD" "docker logs pepin-ros 2>&1 | grep -q 'lifecycle_manager_navigation.*Managed nodes are active'" && break
    sleep 2
done
echo " up"
where() {  # the relocalizer's verdict; prints the raw answer so a failure is never a bare "?"
    ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh timeout 45 python3 /tools/call.py /where_am_i" 2>&1
}
fit_of() { echo "$1" | grep -oE "fit [0-9.]+" | head -1 | awk '{print $2}'; }
good_fit() { awk -v f="${1:-0}" 'BEGIN{exit !(f>=0.45)}'; }
T0=$(date +%s)
ANSWER=$(where); FIT=$(fit_of "$ANSWER")
echo "    where am I: ${ANSWER:-no answer} ($(( $(date +%s) - T0 )) s)"
if ! good_fit "$FIT"; then
    echo "    weak fit, whole-map relocalization (about 10 s)..."
    T0=$(date +%s)
    RELOC=$(ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh timeout 60 python3 /tools/call.py /relocalize 50" 2>&1)
    echo "    $RELOC ($(( $(date +%s) - T0 )) s)"
    # "relocalised ... fit 0.28 -> 0.61": the number after the arrow is the new fit
    FIT=$(echo "$RELOC" | grep -oE -- "-> [0-9.]+" | awk '{print $2}')
    if [ -z "$FIT" ]; then sleep 4; ANSWER=$(where); FIT=$(fit_of "$ANSWER"); echo "    where am I: ${ANSWER:-no answer}"; fi
    good_fit "$FIT" || { echo "!! not localized on $MAP (fit ${FIT:-?}) — not driving. Carry it a metre away and run ros/goto.sh relocalize, or tell Claude."; exit 1; }
fi


echo "[3/4] the lap on $MAP: 4 stops, ~14 m — watch the robot; Ctrl-C stops it"
echo "    (indented lines below are what Nav2 itself says: goals, controller verdicts, recoveries, AMCL)"
watch_start
set -- $STOPS
while [ $# -ge 2 ]; do
    echo ">>> goto $1 $2"
    # -t + -it: Ctrl-C reaches goto_ros on the board, which cancels the task itself; live feedback
    ssh -t "root@$BOARD" "docker exec -it -e PYTHONUNBUFFERED=1 pepin-ros /pepin_entrypoint.sh python3 /tools/goto_ros.py $1 $2" || { echo "goto interrupted/failed — ending the lap"; exit 1; }
    shift 2
done
echo "lap complete"
