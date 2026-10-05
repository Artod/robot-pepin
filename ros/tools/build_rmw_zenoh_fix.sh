#!/bin/bash
# rmw_zenoh_cpp 0.2.10 with the lost wake-up fixed (ros/patches/rmw_zenoh-lost-wakeup.patch, upstream
# ros2/rmw_zenoh #1036/#1040): builds ONLY librmw_zenoh_cpp.so, in a throwaway container of the
# image every one of ours shares the ROS packages with (pepin-laptop:zenoh: the board's pepin-ros and
# all laptop images carry the same ros-jazzy-rmw-zenoh-cpp 0.2.10-1noble.20260902.013751 and
# zenoh-cpp-vendor 0.2.10-1noble.20260722.215603, arm64 like the board), against those installed
# packages, with no network in the container.
#
#   ros/tools/build_rmw_zenoh_fix.sh     -> ros/build/rmw_zenoh_fix/librmw_zenoh_cpp.so (stripped,
#                                           what the images COPY over the apt one) and BUILD.txt
#                                           (sources, sums, the interface check); the same .so with
#                                           symbols (for gdb) and colcon's log in
#                                           ros/build/rmw_zenoh_fix_debug/; the image tag
#                                           pepin-rmw-zenoh-fix:0.2.10 (the .so alone, at /)
#
# It refuses an image whose rmw_zenoh_cpp is not 0.2.10 (the patch is 0.2.10's; from 0.2.11 on the fix
# is upstream and this script goes), and a build whose exported symbols differ from the apt .so's.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # ros/
BASE="${PEPIN_ZFIX_BASE:-pepin-laptop:zenoh}"
VERSION=0.2.10
TAG="pepin-rmw-zenoh-fix:$VERSION"
OUT="$HERE/build/rmw_zenoh_fix"
DBG="$HERE/build/rmw_zenoh_fix_debug"
SRC_URL="https://github.com/ros2/rmw_zenoh/archive/refs/tags/$VERSION.tar.gz"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT" "$DBG"
curl -sfL "$SRC_URL" | tar xz -C "$WORK"
SRC="$WORK/rmw_zenoh-$VERSION"
(cd "$SRC" && git init -q && git apply "$HERE/patches/rmw_zenoh-lost-wakeup.patch")
docker run --rm --network none -v "$SRC:/src:ro" -v "$OUT:/out" -v "$DBG:/dbg" --entrypoint bash "$BASE" -c "
set -euo pipefail
have=\$(dpkg-query -W -f='\${Version}' ros-jazzy-rmw-zenoh-cpp)
case \"\$have\" in $VERSION-*) ;; *) echo \"rmw_zenoh_cpp \$have in $BASE, the patch is $VERSION's\" >&2; exit 2 ;; esac
set +u; source /opt/ros/jazzy/setup.bash; set -u
mkdir -p /ws/src && cp -r /src/rmw_zenoh_cpp /ws/src/
cd /ws && colcon build --packages-select rmw_zenoh_cpp --event-handlers console_direct- \
    --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF > /dbg/colcon.log 2>&1 \
    || { tail -30 /dbg/colcon.log >&2; exit 3; }
built=/ws/install/rmw_zenoh_cpp/lib/librmw_zenoh_cpp.so
apt=/opt/ros/jazzy/lib/librmw_zenoh_cpp.so
cp \$built /dbg/librmw_zenoh_cpp.so
strip --strip-debug -o /out/librmw_zenoh_cpp.so \$built
# The interface is the strong exports (the rmw_* C API rmw_implementation dlopens) and the libraries
# it needs; the weak ones are std:: template instantiations the apt build (noble's dpkg-buildflags,
# LTO) does not export, and are only counted.
nm -D --defined-only \$apt | awk '\$2 == \"T\" {print \$3}' | sort > /tmp/apt.sym
nm -D --defined-only /out/librmw_zenoh_cpp.so | awk '\$2 == \"T\" {print \$3}' | sort > /tmp/new.sym
if ! cmp -s /tmp/apt.sym /tmp/new.sym; then diff /tmp/apt.sym /tmp/new.sym | head >&2; echo 'strong exports differ' >&2; exit 4; fi
readelf -d \$apt | awk '/NEEDED/ {print \$NF}' > /tmp/apt.need
readelf -d /out/librmw_zenoh_cpp.so | awk '/NEEDED/ {print \$NF}' > /tmp/new.need
if ! cmp -s /tmp/apt.need /tmp/new.need; then echo 'needed libraries differ' >&2; exit 4; fi
weak=\$(nm -D --defined-only /out/librmw_zenoh_cpp.so | awk '\$2 == \"W\"' | wc -l)
{
  echo \"rmw_zenoh_cpp $VERSION (upstream tag) + ros/patches/rmw_zenoh-lost-wakeup.patch, RelWithDebInfo, stripped\"
  echo \"built in $BASE against: \$(dpkg-query -W -f='\${Package} \${Version}, ' ros-jazzy-rmw-zenoh-cpp ros-jazzy-zenoh-cpp-vendor ros-jazzy-rmw ros-jazzy-rcutils ros-jazzy-fastcdr)\"
  echo \"strong exports: \$(wc -l < /tmp/new.sym), identical to the apt .so's (\$(grep -c '^rmw_' /tmp/new.sym) rmw_* C functions); \$weak weak template instantiations more\"
  echo \"needed, identical to the apt .so's: \$(tr -d '[]' < /tmp/new.need | tr '\n' ' ')\"
  echo \"sha256 \$(sha256sum /out/librmw_zenoh_cpp.so | cut -d' ' -f1) librmw_zenoh_cpp.so\"
  echo \"sha256 \$(sha256sum \$apt | cut -d' ' -f1) the apt one\"
} > /out/BUILD.txt
"
printf 'FROM scratch\nCOPY librmw_zenoh_cpp.so BUILD.txt /\n' | docker build -q -t "$TAG" -f - "$OUT" >/dev/null
cat "$OUT/BUILD.txt"
echo "built: $OUT/librmw_zenoh_cpp.so and the image $TAG"
