#!/bin/sh
# Inside an image build (ros/Dockerfile, Dockerfile.laptop, Dockerfile.xfeat, Dockerfile.vio): put the
# lost-wake-up librmw_zenoh_cpp.so (ros/tools/build_rmw_zenoh_fix.sh, ros/patches/rmw_zenoh-lost-
# wakeup.patch) over the apt one while the image's rmw_zenoh_cpp is the 0.2.10 it was built from;
# from 0.2.11 on the fix is upstream and the apt .so stays. Refuses a 0.2.10 image without the .so,
# so no image is built without the fix by accident.
#   sh install_rmw_zenoh_fix.sh DIR     (DIR holds the build's librmw_zenoh_cpp.so and BUILD.txt)
# ROS_LIB and FIX_INFO override /opt/ros/jazzy/lib and /opt/rmw_zenoh_fix (the unit test's).
set -eu
DIR="$1"
LIB="${ROS_LIB:-/opt/ros/jazzy/lib}"
INFO="${FIX_INFO:-/opt/rmw_zenoh_fix}"
have="$(dpkg-query -W -f='${Version}' ros-jazzy-rmw-zenoh-cpp)"
case "$have" in
    0.2.10-*)
        if [ ! -f "$DIR/librmw_zenoh_cpp.so" ]; then
            echo "rmw_zenoh_cpp $have needs the lost-wake-up .so: run ros/tools/build_rmw_zenoh_fix.sh first" >&2
            exit 1
        fi
        cp "$LIB/librmw_zenoh_cpp.so" "$LIB/librmw_zenoh_cpp.so.apt"
        cp "$DIR/librmw_zenoh_cpp.so" "$LIB/librmw_zenoh_cpp.so"
        mkdir -p "$INFO" && cp "$DIR/BUILD.txt" "$INFO/BUILD.txt"
        echo "rmw_zenoh_cpp $have: the lost-wake-up .so installed (the apt one kept as .so.apt)" ;;
    *)
        echo "rmw_zenoh_cpp $have: the lost-wake-up fix is upstream (0.2.11+), the apt .so stays" ;;
esac
