#!/bin/bash
# Kill ros2 command-line tools (topic/param/run/node/service echo|hz|get|set...) left behind
# inside the pepin-ros container: a `timeout` around them ends the tool but not always its DDS
# shutdown, and a dozen leftovers took the Orange Pi to load 12 (2026-09-13). Nodes are never
# touched: only the CLI verbs, and only when older than MAX_AGE_S.
set -u
MAX_AGE_S=${MAX_AGE_S:-90}
docker exec pepin-ros bash -c '
  ps -eo pid,etimes,args | awk -v max='"$MAX_AGE_S"' '"'"'
    $2 > max && $3 ~ /(^|\/)(python3|ros2)$/ && $0 ~ /ros2 (topic|param|run|node|service|action|interface) / { print $1 }
  '"'"' | while read -r pid; do kill -KILL "$pid" 2>/dev/null && echo "reaped $pid"; done
' 2>/dev/null
exit 0
