#!/bin/bash
# The head camera's MJPEG stream copied into FILE (mkv, copied as-is: no re-encoding, no CPU)
# until this script gets TERM; then one closing line: the clip's size, or a loud line when there
# is no clip. ros/goto.sh films every drive with it: any goal may be the clip worth posting.
#   ros/clip.sh FILE URL
# A clip that has not started PEPIN_CLIP_CHECK_S (4) seconds in is said aloud and started once
# more, and a second miss is said aloud too: the drives of 2026-09-28 evening have no clip and
# nobody was told. A stream that ends before the drive does is said aloud as well. ffmpeg's own
# words go to FILE's .ffmpeg.log beside it, kept only when it said something.
# TERM, not INT, ends it: a script's background job ignores SIGINT, and ros/goto.sh runs this in
# a process group of its own so the operator's Ctrl-C reaches the goal and not the film. A clip
# nobody stops is a bug: ffmpeg stops by itself after PEPIN_CLIP_MAX_S (1800), and this script
# when its parent is gone.
set -u
[ $# -eq 2 ] || { echo "usage: ros/clip.sh FILE URL"; exit 2; }
CAM="$1"
URL="$2"
CHECK_S="${PEPIN_CLIP_CHECK_S:-4}"
CAMLOG="${CAM%.*}.ffmpeg.log"
FF=""
said() { tail -1 "$CAMLOG" 2>/dev/null | grep . || echo "ffmpeg said nothing"; }
film() {
    ffmpeg -nostdin -loglevel error -y -f mjpeg -use_wallclock_as_timestamps 1 -i "$URL" \
        -c copy -t "${PEPIN_CLIP_MAX_S:-1800}" "$CAM" 2>>"$CAMLOG" &
    FF=$!
}
stop_ffmpeg() {  # INT is ffmpeg's own clean stop (it writes the index); KILL after 5 s
    kill -INT "$FF" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$FF" 2>/dev/null || break
        sleep 0.5
    done
    kill -KILL "$FF" 2>/dev/null
    wait "$FF" 2>/dev/null
}
finish() {
    stop_ffmpeg
    if [ -s "$CAM" ]; then
        bytes=$(wc -c <"$CAM" | tr -d ' ')
        echo "clip: $(awk -v b="$bytes" 'BEGIN { printf "%.1f", b / 1048576 }') MB ($bytes bytes), $CAM"
        [ -s "$CAMLOG" ] || rm -f "$CAMLOG"
    else
        echo "!! no camera clip for this drive ($(said); $URL)"
        rm -f "$CAM"
    fi
    exit 0
}
trap finish TERM HUP
pause() { sleep "$1" & wait $!; }  # a wait that TERM interrupts
film
pause "$CHECK_S"
if ! [ -s "$CAM" ]; then
    echo "!! the camera clip is NOT recording ($(said)): starting it again"
    stop_ffmpeg
    film
    pause "$CHECK_S"
    [ -s "$CAM" ] || echo "!! still no camera clip: this drive has no picture ($URL)"
fi
wait "$FF"
[ -s "$CAM" ] && echo "!! the camera stream ended before the drive did ($(said))"
while kill -0 "$PPID" 2>/dev/null; do pause 1; done
finish
