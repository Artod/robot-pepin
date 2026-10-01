#!/bin/bash
# Build ~/Applications/Pepin.app: a launcher for the menu-bar app (apps/macos/tray.py) that
# Spotlight finds under "Pepin" and Finder can open. The bundle's executable is a shell script
# that runs the tray through uv, named in full — a Finder-launched app has no PATH — with its
# output appended to ~/Library/Logs/Pepin.log. No Dock icon (LSUIElement): the app lives in the
# menu bar. The icon is apps/macos/icon_template.png scaled into an .icns when sips and iconutil
# are at hand; without them the bundle gets the generic icon.
# Usage:
#   apps/macos/install_app.sh [DEST_DIR]   build DEST_DIR/Pepin.app (default ~/Applications)
#   apps/macos/install_app.sh --desktop    ...and an alias on the Desktop (Finder makes it, and
#                                          macOS may ask once to let the shell control Finder)
#   PEPIN_REPO=/path/to/pepin apps/macos/install_app.sh   the checkout the app runs (default:
#                                          the one this script is in)
# Rerunning replaces the bundle. `mdimport` is run on it so Spotlight sees it within seconds.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${PEPIN_REPO:-$(cd "$HERE/../.." && pwd)}"
UV=/opt/homebrew/bin/uv
DEST="$HOME/Applications"
DESKTOP=0
for arg in "$@"; do
    case "$arg" in
        --desktop) DESKTOP=1 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) DEST="$arg" ;;
    esac
done
[ -f "$REPO/apps/macos/tray.py" ] || { echo "no tray at $REPO/apps/macos/tray.py (PEPIN_REPO?)" >&2; exit 1; }
[ -x "$UV" ] || echo "warning: $UV is not there (brew install uv): the app will not start until it is" >&2

APP="$DEST/Pepin.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleName</key>
	<string>Pepin</string>
	<key>CFBundleDisplayName</key>
	<string>Pepin</string>
	<key>CFBundleIdentifier</key>
	<string>com.artem.pepin.tray</string>
	<key>CFBundleExecutable</key>
	<string>Pepin</string>
	<key>CFBundleIconFile</key>
	<string>Pepin</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleVersion</key>
	<string>1</string>
	<key>CFBundleShortVersionString</key>
	<string>0.1</string>
	<key>LSUIElement</key>
	<true/>
	<key>NSHighResolutionCapable</key>
	<true/>
</dict>
</plist>
PLIST

# The executable: $REPO is baked in here; everything else is left for the launch to expand.
cat > "$APP/Contents/MacOS/Pepin" <<LAUNCHER
#!/bin/bash
# Pepin.app's executable, written by apps/macos/install_app.sh: the menu-bar app through uv.
# Finder gives an app no PATH, so uv is named in full, and the tools the tray spawns (docker,
# uv for the dashboard) are put on the PATH it passes down.
export PATH="/opt/homebrew/bin:/usr/local/bin:\$PATH"
mkdir -p "\$HOME/Library/Logs"
exec /opt/homebrew/bin/uv run --directory "$REPO" --group macos python apps/macos/tray.py "\$@" >> "\$HOME/Library/Logs/Pepin.log" 2>&1
LAUNCHER
chmod 755 "$APP/Contents/MacOS/Pepin"

# The icon: the 36 px template scaled to every size iconutil wants. Blurry at 512, legible at 32.
if command -v sips >/dev/null && command -v iconutil >/dev/null; then
    ICONSET="$(mktemp -d)/Pepin.iconset"
    mkdir -p "$ICONSET"
    for size in 16 32 128 256 512; do
        sips -z "$size" "$size" "$HERE/icon_template.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null 2>&1
        double=$((size * 2))
        sips -z "$double" "$double" "$HERE/icon_template.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null 2>&1
    done
    iconutil -c icns -o "$APP/Contents/Resources/Pepin.icns" "$ICONSET" 2>/dev/null \
        || echo "no icon (iconutil failed); the bundle gets the generic one" >&2
    rm -rf "$(dirname "$ICONSET")"
fi

mdimport "$APP" 2>/dev/null || true
echo "installed $APP"
echo "  runs: $REPO (apps/macos/tray.py through $UV); log: ~/Library/Logs/Pepin.log"
echo "  Spotlight: command-space, 'Pepin'  —  check: $APP/Contents/MacOS/Pepin --check"
if [ "$DESKTOP" = 1 ]; then
    osascript -e "tell application \"Finder\" to make alias file to (POSIX file \"$APP\") at (path to desktop folder)" >/dev/null \
        && echo "  alias on the Desktop" || echo "  no Desktop alias (Finder refused; allow the shell to control Finder and rerun with --desktop)" >&2
fi
