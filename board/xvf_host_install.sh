#!/bin/bash
# Seeed's host-control tools for the reSpeaker XVF3800 array, installed on the board into
# /opt/xvf_host with two wrappers in /usr/local/bin:
#   xvf_host      the C tool for 64-bit Raspberry Pi OS (aarch64 ELF: glibc 2.34+, libstdc++,
#                 libusb-1.0; Armbian trixie has all three). Knows VERSION, AEC_AZIMUTH_VALUES,
#                 REBOOT, SAVE_CONFIGURATION and the tuning parameters, but predates DOA_VALUE.
#   xvf_host.py   Seeed's Python tool (pyusb + libusb-package in /opt/pepin), which knows every
#                 command, DOA_VALUE and the AIC3104 output levels included.
# Both come from github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY at one pinned commit and
# are checked by SHA-256. That repository has no licence file, so nothing of it is copied into
# this one: it is fetched from Seeed here, at install time. There is no source to build; these
# are Seeed's prebuilt binaries. pepin.audio_server does not need either: it reads the direction
# itself (pepin.xvf3800). They are for the firmware check, tuning and the day-one tests.
#
# Run on the board as root:   bash xvf_host_install.sh
# Then:                       xvf_host VERSION    xvf_host.py DOA_VALUE
set -euo pipefail

COMMIT=4b49bfd19977c63cf6e90dfe9e5827e74e20bf6d  # 2026-09-29
BASE="https://raw.githubusercontent.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/$COMMIT"
DEST=/opt/xvf_host
PYTHON=/opt/pepin/bin/python

FILES=(
    "host_control/rpi_64bit/xvf_host 63f89c6672c0d89bc82d8182cb36013ac3619288780f315a2e0373fb3ed771f2"
    "host_control/rpi_64bit/libcommand_map.so c1b424313e48cfe97c5cfce0530ac05fe47f818cc0fba15a9954198ef105282c"
    "host_control/rpi_64bit/libdevice_usb.so 5b52ee35ef17aa287555abb112ebb1dc8497e11d43acc1bd223170d02e28eddd"
    "host_control/rpi_64bit/libdevice_i2c.so 7acb02a6ae14c34e3291fb55ec91d681d2902b79916df9fef53b5501fc2d1b3b"
    "host_control/rpi_64bit/transport_config.yaml 071f1fb87cfbdeffd3ba624713fa0745f27debfbde0544e8ac1af3863c29034d"
    "host_control/rpi_64bit/dfu_cmds.yaml 67f6a982567b8d23da85c5806c40344094d21071c631344a47164cd085dddba3"
    "python_control/xvf_host.py 5772c25f789288db9c4c9a1f74d98736f96b9289acea990194dcbfec220424f7"
)

[ "$(uname -m)" = aarch64 ] || { echo "these binaries are aarch64; this is $(uname -m)" >&2; exit 1; }
mkdir -p "$DEST"
for entry in "${FILES[@]}"; do
    path="${entry% *}"
    sha="${entry#* }"
    target="$DEST/$(basename "$path")"
    curl -fsSL "$BASE/$path" -o "$target.part"
    echo "$sha  $target.part" | sha256sum --check --quiet -
    mv "$target.part" "$target"
done
chmod +x "$DEST/xvf_host"

# The C tool loads its libraries and transport_config.yaml from beside itself.
cat > /usr/local/bin/xvf_host <<EOF
#!/bin/sh
cd $DEST && LD_LIBRARY_PATH=$DEST exec ./xvf_host "\$@"
EOF
cat > /usr/local/bin/xvf_host.py <<EOF
#!/bin/sh
exec $PYTHON $DEST/xvf_host.py "\$@"
EOF
chmod +x /usr/local/bin/xvf_host /usr/local/bin/xvf_host.py

missing=$(LD_LIBRARY_PATH=$DEST ldd "$DEST/xvf_host" "$DEST"/lib*.so | grep "not found" || true)
if [ -n "$missing" ]; then
    echo "xvf_host is missing libraries (apt install libusb-1.0-0?):" >&2
    echo "$missing" >&2
    exit 1
fi
echo "installed Seeed's xvf_host (commit ${COMMIT:0:7}) into $DEST: try 'xvf_host VERSION'"
