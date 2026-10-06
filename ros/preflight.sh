#!/bin/bash
# Is the cart ready for a goal? Nine short lines, one fact each, OK or FAIL first: the pose and
# the lidar from the goal server's `where` (127.0.0.1:3337, Nav2 on this Mac), the planner it
# picked, the last snapshot and recognition reports of the laptop's mapping, the three ToF sensors
# reaching the laptop (ros/tools/tof_check.py: a rate and a reading each), the Foxglove bridge,
# the board's clock against the laptop's (its chrony) with the stamp -> receipt p50 of the key
# streams here (ros/tools/stream_latency.py, 2 s in pepin-macnav), and one plan from the planner
# (ros/tools/planner_check.py: a plan, never motion). Nothing here commands motion.
#   ros/preflight.sh [--no-plan]   --no-plan during a drive: the planner is not asked for a plan
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NOPLAN="${1:-}"
say() { printf '%-4s %-9s %s\n' "$1" "$2" "$3"; }
W="$(PYTHONPATH="$HERE/../src" python3 -m pepin.goal_link where 2>/dev/null)"
pose="$(sed -nE 's/.*"pose": "([a-z]+)".*"x": ([-0-9.]+)[0-9]*, "y": ([-0-9.]+)[0-9]*, "yaw_deg": ([-0-9.]+).*/\1 \2 \3 \4/p' <<<"$W")"
if [ -n "$pose" ]; then
    read -r _ x y yaw <<<"$pose"
    say OK pose "$(printf '(%.2f, %.2f, %.0f deg)' "$x" "$y" "$yaw")"
elif [ -n "$W" ]; then
    say FAIL pose "none: nothing publishes map -> base_link (ros/laptop.sh vslam)"
else
    say FAIL pose "no goal server on 127.0.0.1:3337 (ros/laptop.sh nav)"
fi
lidar="$(sed -nE 's/.*"lidar": "([^"]*)".*/\1/p' <<<"$W")"
if [ "$lidar" = ok ]; then say OK lidar ok; else say FAIL lidar "${lidar:-no answer}"; fi
planner="$(sed -nE 's/.*"planner": "([a-z0-9]+)".*/\1/p' <<<"$W")"
if [ -n "$planner" ]; then say OK planner "$planner"; else say FAIL planner "no answer"; fi
LOG="$(docker logs --since 70s pepin-vslam 2>&1)"
pack="$(grep '\[sensor_pack\]: sensor pack:' <<<"$LOG" | tail -1)"
snap="$(sed -nE 's/.*; last: ([a-z-]+).*/\1/p' <<<"$pack")"
cam="$(grep -oE 'camera (fresh [0-9.]+ Hz|silent [0-9.]+ s)' <<<"$pack" | head -1)"
if [ "$snap" = full ]; then say OK snapshots "full, $cam"; else say FAIL snapshots "${snap:-none}, $cam"; fi
tof="$(docker exec pepin-vslam /pepin_entrypoint.sh timeout -s KILL 8 python3 /tools/tof_check.py 2 2>&1 | grep -E '^(OK|FAIL) ' | tail -1)"
if [ "${tof%% *}" = OK ]; then say OK tof "${tof#OK }"
elif [ -n "$tof" ]; then say FAIL tof "${tof#FAIL }"
else say FAIL tof "no answer from ros/tools/tof_check.py in pepin-vslam"; fi
rec="$(grep '\[rtabmap_frame\]: rtabmap frame:' <<<"$LOG" | tail -2 \
    | sed -E 's/.*rtabmap frame: ([0-9]+) updates, ([0-9]+) recognised.*/\1 \2/' \
    | awk 'NR==1{u=$1;r=$2} END{if(NR>1) printf "%d of %d in the last 30 s", $2-r, $1-u; else print "one report only"}')"
if grep -qE '^[1-9]' <<<"$rec"; then say OK recognise "$rec"; else say FAIL recognise "$rec"; fi
fg="$("$HERE/foxglove.sh" check 2>&1 | grep -cE '^FAIL fg\.[345] ')"
if [ "$fg" = 0 ]; then say OK foxglove "bridge answers"; else say FAIL foxglove "no handshake"; fi
BOARD="${BOARD:-${PEPIN_HOST:-10.0.0.187}}"
chrony="$(ssh -o ConnectTimeout=3 -o BatchMode=yes "root@$BOARD" chronyc -c tracking 2>/dev/null | head -1)"
myip="$(ipconfig getifaddr "$(route -n get "$BOARD" 2>/dev/null | awk '/interface:/ {print $2}')" 2>/dev/null)"
lat="$(docker exec pepin-macnav /pepin_entrypoint.sh timeout -s KILL 10 python3 /tools/stream_latency.py 2 --chrony "$chrony" --laptop-ip "$myip" 2>&1 | grep -E '^(OK|FAIL) ' | tail -1)"
if [ "${lat%% *}" = OK ]; then say OK time "${lat#OK }"
elif [ -n "$lat" ]; then say FAIL time "${lat#FAIL }"
else say FAIL time "no answer from ros/tools/stream_latency.py in pepin-macnav"; fi
if [ "$NOPLAN" != --no-plan ]; then
    plan="$(docker exec pepin-vslam /pepin_entrypoint.sh timeout -s KILL 60 python3 /tools/planner_check.py 2>&1 | tail -1)"
    if grep -q 'planner: OK' <<<"$plan"; then
        say OK plan "$(sed -E 's/planner: OK — (path of [0-9]+ poses)[^(]*\(([A-Za-z]+)\).*/\1, \2/' <<<"$plan")"
    else
        say FAIL plan "$(sed -E 's/planner: ([A-Z]+).*/\1/' <<<"$plan")"
    fi
fi
