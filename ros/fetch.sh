#!/bin/bash
# Bring home every recording the board still holds (a run whose cleanup never ran leaves them there).
# Usage: ros/fetch.sh
set -uo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/maps/rec"
before=$(ls "$HERE/maps/rec" | wc -l)
rsync -a --info=stats1 "root@$BOARD:/root/pepin-ros/maps/rec/" "$HERE/maps/rec/" | grep -E "Number of regular files transferred" || true
echo "local files: $before -> $(ls "$HERE/maps/rec" | wc -l)"
ssh "root@$BOARD" "du -sh /root/pepin-ros/maps/rec; docker exec pepin-ros pgrep -cf session_logger.py || echo 'recorders running: 0'"
