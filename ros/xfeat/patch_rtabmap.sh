#!/bin/bash
# Our patches to RTAB-Map's core (ros/patches/rtabmap-*.patch, copied to /opt/xfeat/patches by
# ros/Dockerfile.xfeat), applied to the pinned 0.22.1 source and built as the core LIBRARY alone,
# swapped in place over the one build_rtabmap.sh installed. A script and a layer of its own, AFTER
# the core and wrapper builds, so a patch change rebuilds 6 minutes (measured 2026-09-24 at 2 jobs,
# scratch/models/build_patched_core.sh) and not the hour those two take: the patch may change
# function bodies only — same soname, same class layout — because the rtabmap_ros packages were
# built against the unpatched headers. A patch that changes a header belongs in build_rtabmap.sh.
#
# Today one: RTAB-Map 0.22.1 drops a node's global descriptor when it reloads the node's data for a
# registration (Memory::computeTransform replaces the signature's sensor data with the Data table's,
# which has none), and the next descriptor comparison with that node aborts it
# (Signature.cpp:252) — measured in this image, scratch/models/replay_place.py. Each applied patch
# leaves /opt/rtabmap_patches/<name>, which is how the nodes know at run time what this RTAB-Map
# carries (pepin.global_descriptor.KEEPS_DESCRIPTORS_MARKER).
set -euo pipefail
RTABMAP_REF="${RTABMAP_REF:-0.22.1}"
RTABMAP_SHA="${RTABMAP_SHA:-df6300e0ba3e90058f90b09c4d646279366d3516}"
CORE_JOBS="${CORE_JOBS:-2}"
SRC=/opt/src
PREFIX=/opt/ros/jazzy
set +u
# shellcheck disable=SC1091
source "$PREFIX/setup.bash"
set -u
LIBDIR="lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo aarch64-linux-gnu)"
mkdir -p "$SRC" /opt/rtabmap_patches
git clone -q --depth 1 --branch "$RTABMAP_REF" https://github.com/introlab/rtabmap.git "$SRC/rtabmap"
GOT="$(git -C "$SRC/rtabmap" rev-parse HEAD)"
if [ "$GOT" != "$RTABMAP_SHA" ]; then
    echo "rtabmap $RTABMAP_REF is $GOT, pinned $RTABMAP_SHA: refusing to patch an unpinned source" >&2
    exit 1
fi
for patch in /opt/xfeat/patches/rtabmap-*.patch; do
    git -C "$SRC/rtabmap" apply --verbose "$patch"
done
# The core phase's configuration exactly (build_rtabmap.sh core), so the library is that one plus
# the patches.
cmake -S "$SRC/rtabmap" -B "$SRC/rtabmap/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DMULTI_ARCH=ON \
    -DCMAKE_INSTALL_LIBDIR="$LIBDIR" \
    -DWITH_PYTHON=ON -DWITH_QT=OFF -DWITH_OPENNI=OFF \
    -DBUILD_APP=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=ON
nice -n 19 cmake --build "$SRC/rtabmap/build" --target rtabmap_core -j "$CORE_JOBS"
BUILT="$(find "$SRC/rtabmap/build" -name 'librtabmap_core.so.0.22.1' | head -1)"
cp "$BUILT" "$PREFIX/$LIBDIR/librtabmap_core.so.0.22.1"
# Still the Python core (grep -q reads a file: under pipefail it would kill the writer, exit 141).
readelf -Ws --dyn-syms "$PREFIX/$LIBDIR/librtabmap_core.so.0.22.1" > /tmp/core_symbols
grep -q PythonInterface /tmp/core_symbols
for patch in /opt/xfeat/patches/rtabmap-*.patch; do
    name="$(basename "$patch" .patch)"
    touch "/opt/rtabmap_patches/${name#rtabmap-}"
done
rm -rf "$SRC/rtabmap"
echo "rtabmap core rebuilt with: $(ls /opt/rtabmap_patches | tr '\n' ' ')"
