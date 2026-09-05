#!/bin/bash
# Send the robot to map coordinates X Y (meters, map frame) through Nav2 on the board.
# Usage: ros/goto.sh -2.0 0.5 [yaw_deg]     Cancel: ros/goto.sh stop
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
if [ "${1:-}" = "stop" ]; then
    # A zero twist on the wheel topic stops the cart at once; Nav2's task keeps running until the goal is cancelled.
    ssh "root@$BOARD" 'docker exec pepin-ros /pepin_entrypoint.sh ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"'
    exit 0
fi
X="$1"; Y="$2"; YAW_DEG="${3:-0}"
QZ=$(python3 -c "import math; print(math.sin(math.radians($YAW_DEG)/2))")
QW=$(python3 -c "import math; print(math.cos(math.radians($YAW_DEG)/2))")
ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \"{header: {frame_id: map}, pose: {position: {x: $X, y: $Y, z: 0.0}, orientation: {z: $QZ, w: $QW}}}\""
echo "goal sent: ($X, $Y) yaw $YAW_DEG deg — watch /plan in Foxglove"
