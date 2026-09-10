#!/bin/bash
# Build the laptop image (pepin-laptop): the board's image plus RTAB-Map and the image pipeline.
# Usage: ros/laptop-build.sh        (a few minutes; needs pepin-ros:latest built by ros/build.sh)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
docker build -f "$HERE/Dockerfile.laptop" -t pepin-laptop:latest "$HERE"
echo "pepin-laptop:latest built"
