#!/bin/bash
# Ready to drive after the cart was put on its base by hand, in one command: the pose seeded at the
# base (ros/goto.sh seed), the voxels and both costmaps emptied (ros/reset_world.sh), the planner
# proven (one plan, never motion), the pose and the Foxglove bridge read back.
#   ros/ready.sh [X Y YAW_DEG]   the base's spot in the map by default
# The default is where the cart's base stands in the flat's RTAB-Map frame; another room or a
# moved base is three numbers on the command line.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ $# -eq 3 ]; then SEED=("$@"); elif [ $# -eq 0 ]; then SEED=(-0.387 2.922 96.1); else
    echo "usage: ros/ready.sh [X Y YAW_DEG]"; exit 2
fi
"$HERE/goto.sh" seed "${SEED[@]}" 2>&1 | grep -E "^seeded|placed" | cut -c1-160
"$HERE/reset_world.sh" 2>&1 | cut -c1-120
docker exec pepin-vslam /pepin_entrypoint.sh timeout -s KILL 60 python3 /tools/planner_check.py 2>&1 | tail -1 | cut -c1-200
PYTHONPATH="$HERE/../src" python3 -m pepin.goal_link where 2>&1 | cut -c1-240
"$HERE/foxglove.sh" check 2>&1 | tail -1
