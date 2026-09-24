#!/bin/bash
# One clock for the robot, from the laptop: the board's chrony (board/chrony.sh) and this laptop's
# time server (the pepin-chrony container, ros/chrony). Usage:
#
#   ros/time.sh install          the board on chrony, pointed at this laptop's address (apt on the
#                                board: ~1 min of its CPU, a PARKED robot), then the server here
#   ros/time.sh uninstall        the board back on systemd-timesyncd, exactly as before
#   ros/time.sh source laptop|pool
#                                THE SWITCH (PEPIN_TIME_SOURCE), live on the board: its sources
#                                are reloaded, chronyd is not restarted and the clock is not
#                                stepped. pool = the references timesyncd had. Set the same value
#                                in this shell (ros/lib.sh) so ros/laptop.sh starts or skips the
#                                server to match
#   ros/time.sh point            re-point the board at this laptop's current address (DHCP moved it)
#   ros/time.sh offset           the board's clock minus the laptop's, one line; exit 0 within
#                                PEPIN_CLOCK_WARN_MS (100), 1 over, 2 not measured, 3 no time
#                                server on this laptop. ros/restart.sh's check 1.15
#   ros/time.sh status           the board's chronyc tracking and sources, and this server's
#   ros/time.sh server [stop]    this laptop's server alone (ros/laptop.sh starts it with the router)
#
# Nothing here moves the robot, and nothing is a drive gate: a clock that is off is a WARN line.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$HERE/lib.sh"  # multiplexed ssh; PEPIN_TIME_SOURCE, pepin_laptop_ip, pepin_timeserver_up
REMOTE=/tmp/pepin-chrony  # where the installer is copied; it keeps itself at /usr/local/sbin after
BOARD_TOOL=/usr/local/sbin/pepin-chrony

laptop_ip() {
    local ip
    ip="$(pepin_laptop_ip "$BOARD" || true)"
    [ -n "$ip" ] || { echo "no route from this laptop to the board at $BOARD: cannot tell it where to find the clock" >&2; return 1; }
    printf '%s\n' "$ip"
}

case "${1:-}" in
    install)
        IP="$(laptop_ip)"
        ssh "root@$BOARD" "rm -rf $REMOTE && mkdir -p $REMOTE/chrony"
        scp -q "$HERE/../board/chrony.sh" "root@$BOARD:$REMOTE/chrony.sh"
        scp -q "$HERE/../board/chrony/chrony.conf" "root@$BOARD:$REMOTE/chrony/chrony.conf"
        PEPIN_TIME_SOURCE=laptop pepin_timeserver_up  # the board's first sync should find it
        ssh "root@$BOARD" "sh $REMOTE/chrony.sh install $IP"
        ;;
    uninstall)
        ssh "root@$BOARD" "sh $BOARD_TOOL uninstall"
        ;;
    source)
        case "${2:-}" in laptop | pool) ;; *) echo "usage: ros/time.sh source laptop|pool"; exit 2 ;; esac
        if [ "$2" = laptop ]; then
            IP="$(laptop_ip)"
            PEPIN_TIME_SOURCE=laptop pepin_timeserver_up
            ssh "root@$BOARD" "sh $BOARD_TOOL source laptop $IP"
        else
            ssh "root@$BOARD" "sh $BOARD_TOOL source pool"
        fi
        echo "set PEPIN_TIME_SOURCE=$2 in this shell too, so ros/laptop.sh agrees with the board"
        ;;
    point)
        IP="$(laptop_ip)"
        ssh "root@$BOARD" "sh $BOARD_TOOL source laptop $IP"
        ;;
    offset)
        # The board is the NTP client and this laptop's server the reference, so the number is
        # exactly "board minus the clock the laptop's ROS nodes stamp with". The measurement is
        # src/pepin/timesync.py, piped into the board's own python3: nothing to install there.
        # No server running here is answered at once, not after eight 1-s timeouts on the board
        # (ros/restart.sh asks this on every restart, deployed or not), and with its own exit
        # code: under PEPIN_TIME_SOURCE=pool it is the configuration, not a fault.
        if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$PEPIN_TIMESERVER"; then
            echo "no time server on this laptop ($PEPIN_TIMESERVER is not running: ros/time.sh server)"
            exit 3
        fi
        IP="$(laptop_ip)" || exit 2
        ssh "root@$BOARD" "python3 - $IP --warn-ms ${PEPIN_CLOCK_WARN_MS:-100}" \
            < "$HERE/../src/pepin/timesync.py"
        ;;
    status)
        ssh "root@$BOARD" "sh $BOARD_TOOL status" || echo "(no chrony on the board: ros/time.sh install)"
        echo
        docker exec "$PEPIN_TIMESERVER" chronyc -n tracking 2>/dev/null || echo "(no time server on this laptop: ros/time.sh server)"
        ;;
    server)
        if [ "${2:-}" = stop ]; then
            pepin_remove_container "$PEPIN_TIMESERVER"
            echo "laptop time server stopped: the board falls back to the pool"
        else
            PEPIN_TIME_SOURCE=laptop pepin_timeserver_up
        fi
        ;;
    *)
        echo "usage: ros/time.sh install | uninstall | source laptop|pool | point | offset | status | server [stop]"
        exit 2
        ;;
esac
