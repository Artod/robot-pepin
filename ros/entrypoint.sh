#!/bin/bash
# Source ROS 2 and the workspace, then run whatever was asked.
set -e
source /opt/ros/jazzy/setup.bash
if [ -f /ws/install/setup.bash ]; then
    source /ws/install/setup.bash
    # The LD19 SDK library (package ldlidar) installs without an ament environment hook,
    # so the component container cannot find libldlidar.so without this.
    for lib in /ws/install/*/lib; do export LD_LIBRARY_PATH="$lib:${LD_LIBRARY_PATH:-}"; done
fi
exec "$@"
