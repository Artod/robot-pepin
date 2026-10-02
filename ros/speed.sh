#!/bin/bash
# The cart's one speed — config/base.json's max_wheel_speed_m_s — in every place that holds it
# (pepin.speed): the base server's wheel ceiling and each Nav2 parameter that follows it.
#   ros/speed.sh      print it as the file, the base server and each Nav2 parameter hold it now
#   ros/speed.sh X    set the base server and Nav2 to X m/s (0.05 < X <= 0.45) until they restart;
#                     every start reads the file (the pepin-base service, ros/laptop.sh nav)
# The last line says whether every live place holds one number. PEPIN_HOST picks the board,
# PEPIN_BASE_PORT the base server's port.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
PORT="${PEPIN_BASE_PORT:-3336}"  # = pepin.base_link.BASE_PORT
NAV=pepin-macnav  # = pepin.deployment.NAV_CONTAINER
py() { PYTHONPATH="$HERE/../src" python3 -m pepin.speed "$@"; }
[ $# -le 1 ] || { echo "usage: ros/speed.sh [X]"; exit 2; }
SET=()
if [ $# -eq 1 ]; then
    X="$(py check "$1")" || exit 2  # refused before any place is touched
    SET=(--set "$X")
fi
py file || exit 2
rc=0
LIVE="$(py base --host "$BOARD" --port "$PORT" ${SET[@]+"${SET[@]}"})" || rc=1
LIVE="$LIVE
$(docker exec "$NAV" /pepin_entrypoint.sh python3 /tools/nav_speed.py ${SET[@]+"${SET[@]}"} 2>&1 \
    | grep -av 'zenoh::')" || rc=1  # rmw_zenoh's own warnings are not a place
echo "$LIVE"
# One number everywhere, the sign of a reverse limit aside; a line that does not end in a number
# (an error) is a place that could not say.
echo "$LIVE" | awk -v rc="$rc" '
    $NF ~ /^-?[0-9]+\.[0-9]+$/ { v = $NF < 0 ? -$NF : $NF; seen[sprintf("%.2f", v)]++; n++; next }
    NF { bad++ }
    END {
        k = 0; for (s in seen) { k++; list = list " " s }
        if (k == 1 && !bad && rc == 0) print "one speed:" list " m/s in all " n " live places"
        else print "NOT one speed:" list " m/s over " n " places, " bad + 0 " could not say"
    }'
exit "$rc"
