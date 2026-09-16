#!/bin/sh
# Restart the board's zenoh bridge because the laptop asked (board/pepin-bridge-kick.path saw
# the flag file appear). The laptop cannot do this itself: its container has no ssh key and must
# not have one, so pepin_bringup.bridge_watch publishes one message, the board's run recorder
# (pepin_bringup.bridge_kick) writes /run/pepin/bridge_kick, and this runs with the board's own
# privileges and restarts exactly one unit.
#
# Why it is needed at all: of two zenoh bridges the one that starts LAST gets working routes — a
# route's DDS endpoint is built when the route is created and only while the far bridge is
# already announcing. On 2026-09-15 a wireless stall made this bridge close its transport and
# reconnect with the same zenoh id and thirteen pub routes whose dds_reader was empty; nothing
# crossed from the board until the unit was restarted by hand. The laptop restarts its own
# bridge first and asks for this one second, which is the order ros/laptop.sh has always used
# (settle_bridge).
set -u
FLAG=/run/pepin/bridge_kick
STAMP=/run/pepin/bridge_kick.at
COOLDOWN_S=${COOLDOWN_S:-120}

REASON="$(head -c 400 "$FLAG" 2>/dev/null | tr -d '\n')"
rm -f "$FLAG"   # first, always: the path unit re-triggers while the file is there
[ -n "$REASON" ] || REASON="no reason given"

NOW=$(date +%s)
LAST=$(cat "$STAMP" 2>/dev/null || echo 0)
case "$LAST" in *[!0-9]*) LAST=0 ;; esac
if [ $((NOW - LAST)) -lt "$COOLDOWN_S" ]; then
    echo "kick ignored ($((NOW - LAST)) s since the last one, cooldown ${COOLDOWN_S} s): $REASON"
    exit 0
fi
# A bridge that is not running is not a bridge this cures: on side=all with PEPIN_BRIDGE=off the
# unit sleeps by design (board/pepin-bridge.service), and starting it here would contradict
# ros/thin.sh.
if ! systemctl is-active --quiet pepin-bridge; then
    echo "kick ignored (pepin-bridge is not active; ros/thin.sh on|vision|slam owns that): $REASON"
    exit 0
fi
echo "$NOW" > "$STAMP"
echo "kick: restarting pepin-bridge — $REASON"
systemctl restart pepin-bridge
echo "kick: pepin-bridge is $(systemctl is-active pepin-bridge)"
