#!/bin/sh
# One clock for the robot, the board's half: chrony in place of systemd-timesyncd, the laptop's
# time server first and the internet pool as the fallback (board/chrony/chrony.conf says why).
# Runs ON the board as root. `ros/time.sh install` copies this file and board/chrony/ to
# /tmp/pepin-chrony/ and calls it; the install keeps a copy of itself at
# /usr/local/sbin/pepin-chrony for every later call (never under /root/pepin-ros, which
# ros/sync.sh rsyncs with --delete). Usage:
#
#   chrony.sh install LAPTOP_HOST   chrony from apt — Debian removes systemd-timesyncd with it, so
#                                   its .deb is kept first and an uninstall works offline — our
#                                   chrony.conf, and PEPIN_TIME_SOURCE=laptop + PEPIN_LAPTOP_HOST
#                                   in /etc/default/pepin-ros
#   chrony.sh source laptop|pool [LAPTOP_HOST]
#                                   THE SWITCH (CLAUDE.md rule 19), live: rewrites the laptop's
#                                   source file and `chronyc reload sources` — no daemon restart,
#                                   no clock step. pool = the references timesyncd had
#   chrony.sh status                chronyc tracking and sources
#   chrony.sh uninstall             chrony purged, systemd-timesyncd reinstalled and enabled: the
#                                   board exactly as it was before the install
#
# Cost on the board (a new daemon, CLAUDE.md rule 20): chronyd is one small C process — expected
# well under 1 % of a core and a few MB, one 48-byte UDP exchange with the laptop every 4-16 s and
# with each pool server every 1-17 min. NOT measured on this board yet: `ros/board.sh census`
# after the install is the measurement (config/board_manifest.json carries it as a guess).
# WiFi loss: the laptop and the pool go unreachable together and chronyd free-runs on the
# frequency it learned (the drift file), milliseconds an hour; nothing else on the board waits.
set -eu

DEFAULTS=/etc/default/pepin-ros
SOURCES_DIR=/etc/chrony/sources.d
LAPTOP_SOURCES=$SOURCES_DIR/pepin-laptop.sources
KEEP=/var/cache/pepin-chrony  # the systemd-timesyncd .deb, for an uninstall with apt offline
SELF=/usr/local/sbin/pepin-chrony
SHARE=/usr/local/share/pepin-chrony
HERE=$(cd "$(dirname "$0")" && pwd)
export DEBIAN_FRONTEND=noninteractive

die() { echo "pepin-chrony: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "run as root"

default_get() {  # VAR -> its value in /etc/default/pepin-ros, or nothing
    [ -f "$DEFAULTS" ] || return 0
    sed -n "s/^$1=//p" "$DEFAULTS" | tail -1
}
default_put() {  # VAR [VALUE]: replace VAR's line in /etc/default/pepin-ros; no VALUE removes it
    touch "$DEFAULTS"
    grep -v "^$1=" "$DEFAULTS" > "$DEFAULTS.new" || true
    [ $# -lt 2 ] || echo "$1=$2" >> "$DEFAULTS.new"
    mv "$DEFAULTS.new" "$DEFAULTS"
}

write_sources() {  # SOURCE [HOST]: the laptop's server line (laptop), or no file at all (pool)
    mkdir -p "$SOURCES_DIR"
    if [ "$1" = pool ]; then
        rm -f "$LAPTOP_SOURCES"
        return 0
    fi
    [ -n "${2:-}" ] || die "PEPIN_TIME_SOURCE=laptop needs the laptop's address (chrony.sh source laptop HOST)"
    cat > "$LAPTOP_SOURCES.new" <<EOF
# Written by board/chrony.sh for PEPIN_TIME_SOURCE=laptop: the laptop's time server (ros/laptop.sh's
# pepin-chrony container), which serves the Docker VM's clock — the one every ROS node there stamps
# with. prefer: followed whenever it agrees with the pool; never trust, so a laptop clock that is
# minutes off after a Mac wake is outvoted. Every 4-16 s over the LAN; a round trip over 0.5 s (this
# radio's stalls) is not a sample.
server $2 iburst minpoll 2 maxpoll 4 maxdelay 0.5 prefer
EOF
    mv "$LAPTOP_SOURCES.new" "$LAPTOP_SOURCES"
}

show() {
    chronyc -n tracking || true
    echo
    chronyc -n sources || true
    echo "NTPSynchronized=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo ?)"
}

do_install() {
    host=${1:-}
    [ -n "$host" ] || die "usage: chrony.sh install LAPTOP_HOST"
    [ -f "$HERE/chrony/chrony.conf" ] || die "no chrony/chrony.conf beside $0"
    # The way back first: keep the timesyncd package this board runs, so that `uninstall` can
    # restore it without the internet. Nothing is changed if that fails.
    mkdir -p "$KEEP"
    if dpkg -s systemd-timesyncd >/dev/null 2>&1 && ! ls "$KEEP"/systemd-timesyncd_*.deb >/dev/null 2>&1; then
        (cd "$KEEP" && apt-get download systemd-timesyncd) \
            || die "could not keep the systemd-timesyncd .deb (is apt online?); nothing changed"
    fi
    apt-get install -y --no-install-recommends chrony
    systemctl disable --now systemd-timesyncd >/dev/null 2>&1 || true  # gone already if apt removed it
    [ -f /etc/chrony/chrony.conf.debian ] || cp /etc/chrony/chrony.conf /etc/chrony/chrony.conf.debian
    install -m 644 "$HERE/chrony/chrony.conf" /etc/chrony/chrony.conf
    mkdir -p "$SHARE"
    install -m 644 "$HERE/chrony/chrony.conf" "$SHARE/chrony.conf"
    [ "$HERE/$(basename "$0")" = "$SELF" ] || install -m 755 "$HERE/$(basename "$0")" "$SELF"
    default_put PEPIN_TIME_SOURCE laptop
    default_put PEPIN_LAPTOP_HOST "$host"
    write_sources laptop "$host"
    systemctl enable chrony >/dev/null 2>&1
    systemctl restart chrony
    if chronyc waitsync 30 0 0 1 >/dev/null 2>&1; then
        echo "chrony synchronised (the laptop at $host first, the pool beside it)"
    else
        echo "chrony not synchronised after 30 s: is the laptop's server up (ros/time.sh server) and the pool reachable?"
    fi
    show
}

do_source() {
    src=${1:-}
    host=${2:-$(default_get PEPIN_LAPTOP_HOST)}
    case "$src" in laptop | pool) ;; *) die "usage: chrony.sh source laptop|pool [LAPTOP_HOST]" ;; esac
    systemctl is-active --quiet chrony || die "chrony is not running here (ros/time.sh install first)"
    write_sources "$src" "$host"
    default_put PEPIN_TIME_SOURCE "$src"
    [ "$src" = pool ] || default_put PEPIN_LAPTOP_HOST "$host"
    chronyc reload sources >/dev/null
    echo "PEPIN_TIME_SOURCE=$src$([ "$src" = pool ] || echo " (the laptop at $host)"): sources reloaded, no restart, no step"
    chronyc -n sources || true
}

do_uninstall() {
    # chrony out first (purge works offline: it only removes), then timesyncd back — from apt, or
    # from the .deb kept at install when apt cannot reach its mirror.
    systemctl disable --now chrony >/dev/null 2>&1 || true
    apt-get purge -y chrony || die "apt-get purge chrony failed; chrony is stopped, nothing else changed"
    if ! apt-get install -y systemd-timesyncd; then
        deb=$(ls "$KEEP"/systemd-timesyncd_*.deb 2>/dev/null | tail -1)
        { [ -n "$deb" ] && dpkg -i "$deb"; } \
            || die "systemd-timesyncd could not be reinstalled (apt offline, no kept .deb): the board has NO time daemon"
    fi
    systemctl enable --now systemd-timesyncd
    timedatectl set-ntp true
    default_put PEPIN_TIME_SOURCE
    default_put PEPIN_LAPTOP_HOST
    rm -rf "$SHARE"
    echo "back on systemd-timesyncd: $(timedatectl show -p NTPSynchronized)"
    rm -f "$SELF"
}

case "${1:-}" in
    install) do_install "${2:-}" ;;
    source) do_source "${2:-}" "${3:-}" ;;
    status) show ;;
    uninstall) do_uninstall ;;
    *) die "usage: chrony.sh install LAPTOP_HOST | source laptop|pool [LAPTOP_HOST] | status | uninstall" ;;
esac
