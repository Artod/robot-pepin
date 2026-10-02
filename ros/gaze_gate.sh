#!/bin/bash
# The gaze gate (pepin.gaze_gate) in every node that carries it, at once: depth_stream (no depth
# for a saccade frame, so depth_fusion, contact_scan and the costmap never see one), sensor_pack
# (no RTAB-Map snapshot from one) and visual_odometry (no /vo step across one).
#   ros/gaze_gate.sh                  the flag and the three knobs as each node holds them now
#   ros/gaze_gate.sh on | off         the gaze_gate flag everywhere: off is every frame passing, as
#                                     before the gate existed
#   ros/gaze_gate.sh KNOB VALUE       gate_exposure_s, gate_settle_s or gate_yaw_dps everywhere
# Live until the nodes restart; a default moves in config/knobs.json (the three blocks agree,
# tests/unit/test_gaze_gate.py) or in the flag's table (pepin.gaze_gate.GAZE_GATE).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NODES="depth_stream sensor_pack visual_odometry"
NAMES="gaze_gate gate_exposure_s gate_settle_s gate_yaw_dps"
usage() { echo "usage: ros/gaze_gate.sh [on | off | gate_exposure_s|gate_settle_s|gate_yaw_dps VALUE]"; exit 2; }
rc=0
case "$#:${1:-}" in
    0:)
        for node in $NODES; do
            for name in $NAMES; do
                echo "$node/$name $("$HERE/flags.sh" get "$node" "$name" 2>&1 | tail -1)"
            done
        done ;;
    1:on | 1:off)
        for node in $NODES; do "$HERE/flags.sh" set "$node" gaze_gate "$1" || rc=1; done ;;
    2:gate_exposure_s | 2:gate_settle_s | 2:gate_yaw_dps)
        for node in $NODES; do "$HERE/flags.sh" set "$node" "$1" "$2" || rc=1; done ;;
    *) usage ;;
esac
exit "$rc"
