#!/bin/bash
# Bring home every recording the board still holds (a run whose cleanup never ran leaves them
# there), and the places books the board writes (`ros/go.sh mark` edits <map>.places.yaml on the
# board; the tracked copies here went stale without this).
# Usage: ros/fetch.sh
set -uo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/maps/rec"
before=$(ls "$HERE/maps/rec" | wc -l)
rsync -a --info=stats1 "root@$BOARD:/root/pepin-ros/maps/rec/" "$HERE/maps/rec/" | grep -E "Number of regular files transferred" || true
echo "local files: $before -> $(ls "$HERE/maps/rec" | wc -l)"
# The books, never pushed back (ros/sync.sh excludes them): the board's copy is the truth.
rsync -a --info=stats1 --include='*.places.yaml' --exclude='*' "root@$BOARD:/root/pepin-ros/maps/" "$HERE/maps/" | grep -E "Number of regular files transferred" || true
git -C "$HERE" status --short -- 'maps/*.places.yaml' 2>/dev/null | sed 's/^/places changed: /'
ssh "root@$BOARD" "du -sh /root/pepin-ros/maps/rec; docker exec pepin-ros pgrep -cf session_logger.py || echo 'recorders running: 0'"
