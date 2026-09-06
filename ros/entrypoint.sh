#!/bin/bash
# Source ROS 2 and the workspace, then run whatever was asked.
# Sourcing the ROS setup files costs ~4.5 s on this board (measured 2026-09-06), so the resulting
# environment is cached once per container (in /tmp, gone with the container; the launch command
# is the first caller) and every later `docker exec` pays ~0.05 s instead.
set -e
CACHE=/tmp/pepin_env.sh
if [ -f "$CACHE" ]; then
    set -a; . "$CACHE"; set +a
else
    source /opt/ros/jazzy/setup.bash
    if [ -f /ws/install/setup.bash ]; then
        source /ws/install/setup.bash
        # The LD19 SDK library (package ldlidar) installs without an ament environment hook,
        # so the component container cannot find libldlidar.so without this.
        for lib in /ws/install/*/lib; do export LD_LIBRARY_PATH="$lib:${LD_LIBRARY_PATH:-}"; done
    fi
    export -p | grep -v -E '^declare -x (PWD|OLDPWD|SHLVL|_|CACHE)=' | sed 's/^declare -x /export /' > "$CACHE.$$" \
        && mv "$CACHE.$$" "$CACHE"
fi
exec "$@"
