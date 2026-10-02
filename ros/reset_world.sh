#!/bin/bash
# Empty what the robot remembers about obstacles, without a restart: the voxel volume first (or
# it repaints the costmaps from its old voxels), then both of Nav2's costmaps. One line per call;
# the map (rtabmap.db) and the pose are not touched, and nothing commands motion.
#   ros/reset_world.sh
# Three service calls from the laptop's mapping container (/fusion/reset, std_srvs/Trigger, then
# Nav2's clear_entirely_* on the local and the global costmap).
set -uo pipefail
docker exec pepin-vslam bash -c '
source /opt/ros/jazzy/setup.bash
call() { printf "%-48s " "$1"; timeout 20 ros2 service call "$1" "$2" "{}" 2>&1 | grep -v "^$" | tail -1 | cut -c1-160; }
call /fusion/reset std_srvs/srv/Trigger
call /local_costmap/clear_entirely_local_costmap nav2_msgs/srv/ClearEntireCostmap
call /global_costmap/clear_entirely_global_costmap nav2_msgs/srv/ClearEntireCostmap
'
