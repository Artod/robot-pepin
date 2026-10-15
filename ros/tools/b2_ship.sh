#!/bin/bash
# The robot's recordings to Backblaze B2, so the laptop holds only what is live.
#
# The laptop's folders are a spool: recorders write locally, and finished files (closed for at
# least MIN_AGE) are MOVED to the bucket under the same relative path, "pepin/<path>", by rclone.
# Small text (events.jsonl, summaries, scripts, databases) stays local for the tools; audio, video,
# drive recordings with their video, and logs older than a day go (a MOVE: the local copy leaves,
# the bucket keeps it; nothing is ever pruned). Reading a shipped file: the
# mount (~/B2/pepin/<path>, the same tree) or `restore`.
#
#   ros/tools/b2_ship.sh ship              # move what is finished (what launchd runs every 30 min)
#   ros/tools/b2_ship.sh ship --dry-run    # what would go, nothing moves
#   ros/tools/b2_ship.sh status            # local GB per root, the bucket's GB, the last ship
#   ros/tools/b2_ship.sh restore PATH      # a file or folder back from the bucket (copy, not move)
#   ros/tools/b2_ship.sh mount [umount]    # the bucket as ~/B2 (NFS, no FUSE; write cache capped)
#   ros/tools/b2_ship.sh install           # launchd: ship every 30 min, mount at login
#
# Needs: rclone with a remote named "b2" (`rclone config create b2 b2 account KEYID key KEY`; the key
# lives in ~/.config/secrets/b2.env, never in the repo). Bandwidth: full at night, 3 MB/s by day so
# calls and streams keep their uplink (BWLIMIT). Never two ships at once (a lock directory).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE="${PEPIN_B2_REMOTE:-b2:pepin-robot/pepin}"
MOUNT="${PEPIN_B2_MOUNT:-$HOME/B2}"
LOG="$HOME/Library/Logs/pepin-b2-ship.log"
LOCK="/tmp/pepin-b2-ship.lock"
BWLIMIT="08:00,3M 22:00,off"
MIN_AGE="30m"

# root | min age | rclone filter rules (first match wins; "+" include, "-" exclude; a root
# without rules ships everything). Logs, audio and video (Artem, 2026-10-13): the voice sessions'
# audio, the drive recordings with their video and board logs, the ring recorder, the videos
# folder, the container logs. The servo bus tap's logs (data/bus_tap) are a ring kept three
# days by their sink, never shipped. Scratch, models, databases and Docker never go. Nothing is deleted
# here: what to keep is decided in the bucket.
RULES=(
  "data/voice|30m|+ *.wav|- *"
  "ros/maps/rec|24h|- *.py|+ *"
  "ros/maps/ring|24h|"
  "data/videos|30m|"
  "logs|24h|"
)

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

ship() {
  local dry=""; [ "${1:-}" = "--dry-run" ] && dry="--dry-run"
  if ! mkdir "$LOCK" 2>/dev/null; then echo "a ship is already running ($LOCK)"; exit 0; fi
  trap 'rmdir "$LOCK"' EXIT
  mkdir -p "$(dirname "$LOG")"
  echo "=== ship $(date '+%Y-%m-%d %H:%M:%S') ${dry:-}" >> "$LOG"
  local line root age rules extra
  for line in "${RULES[@]}"; do
    IFS='|' read -r root age rules <<< "$line"
    [ -d "$REPO/$root" ] || continue
    extra=()
    local filters=()
    if [ -n "$rules" ]; then
      IFS='|' read -r -a parts <<< "$rules"
      for r in "${parts[@]}"; do [ -n "$r" ] && filters+=(--filter "$r"); done
    fi
    rclone move "$REPO/$root" "$REMOTE/$root" --min-age "$age" ${extra[@]+"${extra[@]}"} ${filters[@]+"${filters[@]}"} \
      --delete-empty-src-dirs --transfers 4 --checkers 8 --bwlimit "$BWLIMIT" \
      --log-file "$LOG" --log-level INFO --stats 5m --stats-one-line $dry || echo "ship: $root failed (see $LOG)"
  done
  echo "=== done $(date '+%H:%M:%S')" >> "$LOG"
  if [ -n "$dry" ]; then  # macOS bash 3.2: no mapfile; the sizes come as 1.2Mi / 3.4Gi
    awk '/^=== ship/{p=1} p' "$LOG" | grep -E "Skipped move" \
      | sed -E 's/.*\(size ([0-9.]+)([KMG]?i?)\).*/\1 \2/' \
      | awk '{v=$1; u=$2; if(u=="Gi") b=v*1024; else if(u=="Mi") b=v; else if(u=="Ki") b=v/1024; else b=v/1048576; s+=b; n++} END{printf "dry run: %d files, %.1f GB would move\n", n, s/1024}'
  fi
  return 0
}

status() {
  echo "local (GB, what the spool still holds):"
  local line root age rules
  for line in "${RULES[@]}"; do
    IFS='|' read -r root age rules <<< "$line"
    [ -d "$REPO/$root" ] && printf '  %-14s %s\n' "$root" "$(du -sh "$REPO/$root" 2>/dev/null | cut -f1)"
  done
  echo "bucket:"
  rclone size "$REMOTE" 2>/dev/null | sed 's/^/  /' || echo "  (rclone remote b2 not reachable)"
  echo "last ship: $(grep -E '^=== (ship|done)' "$LOG" 2>/dev/null | tail -2 | tr '\n' ' ')"
  mount | grep -q " $MOUNT " && echo "mount: $MOUNT is mounted" || echo "mount: not mounted (ros/tools/b2_ship.sh mount)"
}

restore() {
  local path="${1:?restore PATH (relative to the repo, e.g. data/voice/20261012)}"
  rclone copy "$REMOTE/$path" "$REPO/$path" -P
}

do_mount() {
  if [ "${1:-}" = "umount" ]; then umount "$MOUNT" && echo "unmounted $MOUNT"; return; fi
  mkdir -p "$MOUNT"
  if mount | grep -q " $MOUNT "; then echo "$MOUNT already mounted"; return; fi
  rclone nfsmount "${REMOTE%%/*}" "$MOUNT" --vfs-cache-mode full --vfs-cache-max-size 10G \
    --vfs-cache-max-age 2h --daemon && echo "mounted $MOUNT (the bucket's root; the repo's tree is under pepin/)"
}

install() {
  local agents="$HOME/Library/LaunchAgents"; mkdir -p "$agents"
  cat > "$agents/com.pepin.b2ship.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.pepin.b2ship</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$REPO/ros/tools/b2_ship.sh</string><string>ship</string></array>
  <key>StartInterval</key><integer>1800</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/pepin-b2-ship.out</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/pepin-b2-ship.out</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/sbin:/usr/sbin</string></dict>
</dict></plist>
EOF
  cat > "$agents/com.pepin.b2mount.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.pepin.b2mount</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$REPO/ros/tools/b2_ship.sh</string><string>mount</string></array>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/pepin-b2-mount.out</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/pepin-b2-mount.out</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/sbin:/usr/sbin</string></dict>
</dict></plist>
EOF
  for l in com.pepin.b2ship com.pepin.b2mount; do
    launchctl bootout "gui/$(id -u)/$l" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$agents/$l.plist" && echo "installed $l"
  done
}

case "${1:-}" in
  ship) ship "${2:-}" ;;
  status) status ;;
  restore) restore "${2:-}" ;;
  mount) do_mount "${2:-}" ;;
  install) install ;;
  *) usage ;;
esac
