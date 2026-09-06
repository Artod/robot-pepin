#!/bin/bash
# Nav2 (AMCL + relocalizer) on a saved map. Usage: ros/nav.sh [MAP.yaml]   (same as ros/mode.sh nav MAP)
exec "$(dirname "$0")/mode.sh" nav "${1:-/maps/20260903_182653_lap3_loop.yaml}"
