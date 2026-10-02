#!/bin/bash
# Drive by keyboard through ROS. Usage:
#   ros/teleop.sh            just drive (current mode stays)
#   ros/teleop.sh NAME       RECORDED mapping run: starts TWO recorders (our jsonl + a rosbag)
#                            on the board's SD card,
#                            REFUSES to hand you the keyboard until both files are visibly growing,
#                            and on Ctrl-C stops the wheels, closes both recordings and copies
#                            everything to the laptop (ros/maps/rec/). The map itself is built
#                            OFFLINE afterwards — no other command for you to run.
# Keys: arrows drive at full speed (0.45 m/s, 1.0 rad/s), Shift+arrows slow (0.064 m/s, 0.24 rad/s);
# a key latches until the next one; space STOPS; Ctrl-C stops the wheels and ends the run. The
# keyboard node runs on the LAPTOP in pepin-vslam (a new process on the board stalls its link for
# 3-4 s). PEPIN_TELEOP=twist_keyboard: the old teleop_twist_keyboard on the board (i/,/j/l, k stops).
# The base's deadman stops the wheels within 0.5 s of the last command either way.
# Game mode, without ROS: `uv run python -m pepin.teleop --game [--host 10.0.0.187]` opens a small
# window where keys act only while HELD and the window is focused — arrows the wheels (same speeds),
# W/S tilt and A/D pan the head, Shift slow, Space stops all, Esc quits. It talks to the base server
# directly (:3336), so nothing here is recorded; this script stays the recorded way to drive.
set -uo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
HERE="$(cd "$(dirname "$0")" && pwd)"
MAPNAME="${1:-}"
TELEOP="${PEPIN_TELEOP:-keys}"
TWIST_KEYBOARD="docker exec -it pepin-ros /pepin_entrypoint.sh ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.12 -p turn:=0.5 -p repeat_rate:=5.0"

drive_by_keys() {
    if [ "$TELEOP" = "twist_keyboard" ]; then
        ssh -t "root@$BOARD" "$TWIST_KEYBOARD" || true
    else
        docker exec -it pepin-vslam /pepin_entrypoint.sh python3 -m pepin_bringup.teleop_keys || true
    fi
}

if [ -z "$MAPNAME" ]; then
    drive_by_keys
    if [ "$TELEOP" = "twist_keyboard" ]; then  # teleop_keys sends its own zero twist on Ctrl-C
        ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{}' >/dev/null 2>&1" || true
    fi
    exit 0
fi

STAMP=$(date +%Y%m%d_%H%M%S)
REC="rec/${STAMP}_${MAPNAME}"
ssh "root@$BOARD" "mkdir -p /root/pepin-ros/maps/rec"

echo "[1/3] starting both recorders..."
ssh "root@$BOARD" "docker exec -d pepin-ros /pepin_entrypoint.sh python3 /tools/session_logger.py /maps/${REC}.jsonl"
ssh "root@$BOARD" "docker exec -d pepin-ros /pepin_entrypoint.sh ros2 bag record -o /maps/${REC}_bag /ldlidar_node/scan /odom /tf /tf_static /cmd_vel"
sleep 8
echo "[2/3] checking that both recordings actually grow..."
CHECK='J=$(stat -c%s /root/pepin-ros/maps/REC.jsonl 2>/dev/null || echo 0); B=$(du -sb /root/pepin-ros/maps/REC_bag 2>/dev/null | cut -f1 || echo 0); sleep 4; J2=$(stat -c%s /root/pepin-ros/maps/REC.jsonl 2>/dev/null || echo 0); B2=$(du -sb /root/pepin-ros/maps/REC_bag 2>/dev/null | cut -f1 || echo 0); echo "$J $J2 $B $B2"'
SIZES=$(ssh "root@$BOARD" "${CHECK//REC/$REC}")
read -r J1 J2 B1 B2 <<<"$SIZES"
if [ "${J2:-0}" -le "${J1:-0}" ] || [ "${J2:-0}" -lt 20000 ]; then
    echo "!! jsonl recording is NOT growing ($J1 -> $J2 bytes) — NOT letting you drive. Tell Claude."
    exit 1
fi
if [ "${B2:-0}" -le "${B1:-0}" ]; then
    echo "   (rosbag not growing: $B1 -> $B2 — the jsonl IS growing, so you may drive; bag is the backup)"
else
    echo "   ok: jsonl $J1 -> $J2 bytes, bag $B1 -> $B2 bytes"
fi

finish() {
    echo; echo "[3/3] stopping recorders, copying everything to the laptop..."
    ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{}' >/dev/null 2>&1" || true
    ssh "root@$BOARD" "docker exec pepin-ros pkill -INT -f session_logger.py; docker exec pepin-ros pkill -INT -f 'ros2 bag record'" || true
    sleep 4
    mkdir -p "$HERE/maps/rec"
    rsync -a "root@$BOARD:/root/pepin-ros/maps/rec/" "$HERE/maps/rec/" || scp -r "root@$BOARD:/root/pepin-ros/maps/rec" "$HERE/maps/"
    echo
    echo "recorded on the board AND copied to the laptop:"
    ls -la "$HERE/maps/rec/" | grep "$STAMP" || true
    LINES=$(grep -c '"topic":"scan"' "$HERE/maps/rec/${STAMP}_${MAPNAME}.jsonl" 2>/dev/null || echo "?")
    echo "scans recorded: $LINES  (a lap of the flat is ~1500+)"
    echo "DONE. The map is built offline from these files — nothing else for you to run."
}
trap finish EXIT

echo
echo ">>> RECORDING. Drive a smooth lap of the flat, come back to the start spot facing the"
echo ">>> same way, press space to stop, then Ctrl-C. Everything is being written twice on the"
echo ">>> board's SD card and fetched to the laptop when you finish."
echo
drive_by_keys
