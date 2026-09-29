#!/bin/bash
# RTAB-Map 0.22.1 rebuilt from its own sources with Python support, installed OVER the apt build
# of the same version in /opt/ros/jazzy (ros/Dockerfile.xfeat runs this; nothing else should).
#   build_rtabmap.sh core   the core library and tools (rtabmap, WITH_PYTHON)
#   build_rtabmap.sh ros    the rtabmap_ros packages that link it, against the new headers
# Two phases so the image caches the long one and a failure in the second does not redo it.
#
# Why a rebuild and not a plugin: the apt ros-jazzy-rtabmap is built without WITH_PYTHON, and
# PyDetector / PyMatcher (Vis/FeatureType 15, Vis/CorNNType 6) are compiled out of it — its own
# strings say "RTAB-Map is not built with Python3 support". Python support also changes the
# LAYOUT of rtabmap::Rtabmap (a PythonInterface pointer at its end, Rtabmap.h under
# RTABMAP_PYTHON) and rtabmap_slam's CoreWrapper holds that class BY VALUE, so the ROS wrapper
# packages that link the core are rebuilt against the new headers too. rtabmap_msgs does not link
# the core and stays the apt one; rtabmap_viz and rtabmap_rviz_plugins need the Qt GUI library,
# which this build leaves out, and are not run on this robot.
#
# MEMORY, measured the hard way (2026-09-24 03:10Z): rtabmap_sync's CommonDataSubscriber*.cpp
# take about 2.5 GB each to compile, and nine of them in parallel filled the Docker VM's 16 GB.
# The build runs at oom_score_adj -500, so the kernel killed the live stack's processes instead
# (RTAB-Map among them). Hence the job counts below: CORE_JOBS for the core, whose objects are
# smaller, and ROS_JOBS for the wrappers — and ros/laptop-build.sh xfeat watches the VM's free
# memory and cancels the build before it can come to that again.
#
# Every source is pinned by commit, so the image is the same image tomorrow. The core is the
# upstream 0.22.1 tag and not 0.22.1-jazzy, the branch the apt package was cut from: the two differ
# in packaging and in one line that matters here — jazzy-devel dropped RTABMAP_CORE_EXPORT from
# PythonInterface (corelib/include/rtabmap/core/PythonInterface.h), and rtabmap_odom's nodes hold a
# PythonInterface in main() under RTABMAP_PYTHON, so the class has to be exported. (The link error
# that sent us there, "undefined reference to PythonInterface::PythonInterface()", turned out to
# be the library directory below; the export is kept because the odometry nodes need it.)
set -euo pipefail
PHASE="${1:?usage: build_rtabmap.sh core|ros}"
RTABMAP_REF="${RTABMAP_REF:-0.22.1}"
RTABMAP_SHA="${RTABMAP_SHA:-df6300e0ba3e90058f90b09c4d646279366d3516}"
RTABMAP_ROS_REF="${RTABMAP_ROS_REF:-0.22.1-jazzy}"
RTABMAP_ROS_SHA="${RTABMAP_ROS_SHA:-e73e7690cbb56af23a894bf6c4ee266e55772fc2}"
CORE_JOBS="${CORE_JOBS:-2}"
ROS_JOBS="${ROS_JOBS:-2}"
SRC=/opt/src
PREFIX=/opt/ros/jazzy

features_kept() {  # APT NEW: every optional library the apt core was built with, the new one has
    # Two files of `#define RTABMAP_*` lines (Version.h). Whatever the launch table does not name
    # takes the core's compile-time default, and some defaults follow what CMake FOUND —
    # Optimizer/Strategy 2 needs GTSAM, Icp/Strategy 1 libpointmatcher — so a library this build
    # missed would change the lidar's registration and the graph's optimiser without a word. Every
    # feature apt's header defines must be defined by the new one (it adds RTABMAP_PYTHON), and
    # the four the lidar path and the grid lean on are asked for by name whatever apt had.
    local missing feature
    missing="$(comm -23 <(sort -u "$1") <(sort -u "$2"))"
    if [ -n "$missing" ]; then
        echo "the rebuilt core lacks what the apt build had: $missing" >&2
        return 1
    fi
    for feature in GTSAM G2O POINTMATCHER OCTOMAP PYTHON; do
        if ! grep -qx "#define RTABMAP_$feature" "$2"; then
            echo "the rebuilt core has no RTABMAP_$feature" >&2
            return 1
        fi
    done
}

fetch() {  # repository ref sha dir: a shallow clone of one tag, refused unless it is that commit
    git clone -q --depth 1 --branch "$2" "$1" "$4"
    local got
    got="$(git -C "$4" rev-parse HEAD)"
    if [ "$got" != "$3" ]; then
        echo "$1 $2 is $got, pinned $3: refusing to build an unpinned source" >&2
        exit 1
    fi
}

set +u
# shellcheck disable=SC1091
source "$PREFIX/setup.bash"
set -u
mkdir -p "$SRC"
case "$PHASE" in
    core)
        # The apt build's own layout, the same optional libraries it found (g2o, GTSAM,
        # libpointmatcher, octomap: all in the image already; OpenNI 1 off as there), plus
        # Python. No Qt: the GUI library and the rtabmap app are not what this image runs.
        # THE LIBRARY DIRECTORY IS SAID OUT LOUD. MULTI_ARCH alone leaves it to GNUInstallDirs,
        # which picks the multi-arch directory only under /usr, so the first build installed its
        # core into /opt/ros/jazzy/lib beside apt's in lib/aarch64-linux-gnu: new headers
        # (RTABMAP_PYTHON), old library, and rtabmap_odom failed to link PythonInterface.
        LIBDIR="lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo aarch64-linux-gnu)"
        fetch https://github.com/introlab/rtabmap.git "$RTABMAP_REF" "$RTABMAP_SHA" "$SRC/rtabmap"
        cmake -S "$SRC/rtabmap" -B "$SRC/rtabmap/build" \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DMULTI_ARCH=ON \
            -DCMAKE_INSTALL_LIBDIR="$LIBDIR" \
            -DWITH_PYTHON=ON -DWITH_QT=OFF -DWITH_OPENNI=OFF \
            -DBUILD_APP=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=ON
        nice -n 19 cmake --build "$SRC/rtabmap/build" -j "$CORE_JOBS"
        # What the apt core was built with, read before the install overwrites its header.
        VERSION_H="$PREFIX/include/rtabmap-0.22/rtabmap/core/Version.h"
        grep '^#define RTABMAP_' "$VERSION_H" > /tmp/apt_defines
        # apt's CMake package files go first: an install only overwrites, and apt's
        # RTABMap_guiTargets.cmake left beside the new RTABMapConfig.cmake (built without Qt)
        # makes every find_package(RTABMap) fail on its "if( EQUAL 6)" (measured).
        rm -rf "${PREFIX:?}/$LIBDIR/rtabmap-0.22"
        cmake --install "$SRC/rtabmap/build"
        # Fail fast, here and not an hour later in the wrappers: the headers say Python and every
        # optional library apt's did (features_kept), the ONE core library carries Python, and no
        # second core sits anywhere else under /opt/ros/jazzy.
        grep '^#define RTABMAP_' "$VERSION_H" > /tmp/new_defines
        features_kept /tmp/apt_defines /tmp/new_defines
        # (Each check reads its command's whole output from a file: grep -q under pipefail stops
        # reading at the first match and the writer dies of SIGPIPE — exit 141 on a success.)
        readelf -Ws --dyn-syms "$PREFIX/$LIBDIR/librtabmap_core.so.0.22.1" > /tmp/core_symbols
        grep -q PythonInterface /tmp/core_symbols
        CORES="$(find "$PREFIX" -name 'librtabmap_core.so*' -not -path "$PREFIX/$LIBDIR/*")"
        if [ -n "$CORES" ]; then
            echo "a second RTAB-Map core under $PREFIX: $CORES" >&2
            exit 1
        fi
        rm -rf "$SRC/rtabmap"
        echo "rtabmap $RTABMAP_REF ($RTABMAP_SHA) built with Python into $PREFIX"
        ;;
    ros)
        # The wrapper packages that link the core, in dependency order, each installed where apt
        # put it (plain CMake per package: colcon would write its own setup files into /opt/ros).
        fetch https://github.com/introlab/rtabmap_ros.git "$RTABMAP_ROS_REF" "$RTABMAP_ROS_SHA" \
            "$SRC/rtabmap_ros"
        for pkg in rtabmap_conversions rtabmap_sync rtabmap_util rtabmap_odom rtabmap_slam; do
            cmake -S "$SRC/rtabmap_ros/$pkg" -B "$SRC/build/$pkg" \
                -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_TESTING=OFF
            nice -n 19 cmake --build "$SRC/build/$pkg" -j "$ROS_JOBS"
            cmake --install "$SRC/build/$pkg"
        done
        rm -rf "$SRC/rtabmap_ros" "$SRC/build"
        # The node that runs is linked against the core that was built: resolved by the loader,
        # not assumed from the install paths.
        ldd "$PREFIX/lib/rtabmap_slam/rtabmap" > /tmp/rtabmap_ldd
        grep -q "$PREFIX/lib/.*-linux-gnu/librtabmap_core.so" /tmp/rtabmap_ldd
        echo "rtabmap_ros $RTABMAP_ROS_REF ($RTABMAP_ROS_SHA) built against it"
        ;;
    *)
        echo "unknown phase $PHASE: core or ros" >&2
        exit 2
        ;;
esac
