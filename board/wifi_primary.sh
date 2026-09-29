#!/bin/bash
# Which WiFi radio carries the board's network identity (the DHCP lease of 10.0.0.187): the USB
# dongle (MT7921AU, wlx*) or the onboard uwe5622 (wlan0). The primary takes the identity the router
# already knows — the onboard radio's factory MAC and its DHCP IAID, so the client ID (IAID + DUID-LL
# of the MAC) is byte-identical and the router hands out the same lease with no router setup. The
# other radio gets a MAC of its own and a lower route priority and stays up as the fallback link.
#
# Usage (on the board, as root): wifi_primary.sh dongle|onboard|status|confirm
#   dongle, onboard  switch; a timer returns to `onboard` in ROLLBACK_MIN minutes (default 5) unless
#                    `confirm` runs first — run the switch detached (systemd-run), the ssh link drops
#   confirm          keep the current switch (stops the rollback timer)
#   status           the radios, their addresses and the timer
#   dryrun           what `dongle` would generate, into a scratch root (password masked); no change
# The WiFi password is never printed: the access-point block is carried over from the current
# /etc/netplan/30-wifi.yaml verbatim.
set -uo pipefail

ONBOARD=wlan0
DONGLE="$(ls /sys/class/net | grep -m1 '^wlx' || true)"
IDENTITY_MAC=6c:35:cd:01:44:cb       # wlan0's factory MAC: the router's lease of 10.0.0.187
IDENTITY_IAID=3539535402             # 0xd2f9062a: the IAID networkd derived for wlan0
ONBOARD_SPARE_MAC=6e:35:cd:01:44:b0  # locally administered, wlan0's while the dongle is primary
DONGLE_OWN_MAC=74:19:f8:17:5b:39
ROLLBACK_MIN="${ROLLBACK_MIN:-5}"
NETPLAN="${NETPLAN:-/etc/netplan}"
BACKUP=/root/wifi-primary
DROPIN_DIR=/etc/systemd/network/10-netplan-${DONGLE:-wlx}.network.d

log() { echo "wifi_primary: $*"; logger -t wifi_primary "$*"; }

access_points() {  # the access-point block of the current onboard config, verbatim
    awk '/^      access-points:/{f=1} f' "$NETPLAN/30-wifi.yaml"
}

write_netplan() {  # FILE IFACE MAC METRIC OPTIONAL(true|false) ACTIVE(true|false)
    local file="$1" iface="$2" mac="$3" metric="$4" optional="$5" active="$6" aps
    aps="$(access_points)"
    [ -n "$aps" ] || { log "no access-points block in 30-wifi.yaml: refusing"; exit 1; }
    {
        printf 'network:\n  version: 2\n  renderer: networkd\n  wifis:\n    %s:\n' "$iface"
        [ "$optional" = true ] && printf '      optional: true\n'
        [ "$active" = true ] || printf '      activation-mode: off\n'
        printf '      macaddress: "%s"\n      dhcp4: yes\n' "$mac"
        printf '      dhcp4-overrides:\n        route-metric: %s\n' "$metric"
        printf '%s\n' "$aps"
    } > "$file.new"
    chmod 600 "$file.new"
    mv "$file.new" "$file"
}

install_invariants() {  # the parts both modes share: no ARP flux, power save off on every radio
    cat > /etc/sysctl.d/60-pepin-two-wifi.conf <<'EOF'
# Two radios on one subnet: answer ARP only on the interface that owns the address, and announce
# from it, so the router never learns one radio's address at the other's MAC.
net.ipv4.conf.all.arp_ignore = 1
net.ipv4.conf.all.arp_announce = 2
EOF
    sysctl -q -p /etc/sysctl.d/60-pepin-two-wifi.conf
    cat > /etc/udev/rules.d/11-pepin-wifi-dongle-powersave.rules <<'EOF'
# The USB WiFi dongle is renamed wlan1 -> wlx<mac>; power save goes off under its final name.
ACTION=="move", SUBSYSTEM=="net", KERNEL=="wlx*", RUN+="/usr/sbin/iw dev %k set power_save off"
EOF
}

bring_up() {  # IFACE MAC: down, new MAC, up, supplicant, DHCP; returns 1 if the MAC is refused
    local iface="$1" mac="$2"
    systemctl stop "netplan-wpa-$iface.service" 2>/dev/null
    ip link set "$iface" down
    if ! ip link set "$iface" address "$mac"; then
        log "$iface refused MAC $mac"
        return 1
    fi
    ip link set "$iface" up
    systemctl start "netplan-wpa-$iface.service"
    iw dev "$iface" set power_save off 2>/dev/null
    return 0
}

switch_to() {  # dongle|onboard
    local mode="$1" onboard_ok=0
    [ -n "$DONGLE" ] || { log "no wlx* dongle present: refusing"; exit 1; }
    mkdir -p "$BACKUP"
    [ -f "$BACKUP/30-wifi.yaml.orig" ] || cp -a "$NETPLAN/30-wifi.yaml" "$BACKUP/30-wifi.yaml.orig"
    install_invariants
    if [ "$mode" = dongle ]; then
        write_netplan "$NETPLAN/31-wifi-dongle.yaml" "$DONGLE" "$IDENTITY_MAC" 600 false true
        write_netplan "$NETPLAN/30-wifi.yaml" "$ONBOARD" "$ONBOARD_SPARE_MAC" 700 true true
        mkdir -p "$DROPIN_DIR"
        printf '[DHCPv4]\n# the onboard radio'"'"'s IAID: with its MAC, the router sees the same client\nIAID=%s\n' \
            "$IDENTITY_IAID" > "$DROPIN_DIR/pepin-identity.conf"
    else
        write_netplan "$NETPLAN/30-wifi.yaml" "$ONBOARD" "$IDENTITY_MAC" 600 false true
        write_netplan "$NETPLAN/31-wifi-dongle.yaml" "$DONGLE" "$DONGLE_OWN_MAC" 700 true true
        rm -rf "$DROPIN_DIR"
    fi
    ip rule del priority 1000 2>/dev/null  # the A/B's source rule for the dongle's old address
    # The radio receiving the identity MAC stays off the air (no supplicant) until the other one has
    # let go of it: networkd applies the new MACs on reload, in its own order.
    systemctl stop "netplan-wpa-$([ "$mode" = dongle ] && echo "$DONGLE" || echo "$ONBOARD").service"
    netplan generate && systemctl daemon-reload && networkctl reload
    if [ "$mode" = dongle ]; then
        # the identity leaves wlan0 first, so the two radios never share a MAC on the air
        bring_up "$ONBOARD" "$ONBOARD_SPARE_MAC" && onboard_ok=1
        if [ "$onboard_ok" = 0 ]; then  # the driver keeps its MAC: wlan0 stays off, at boot too
            ip link set "$ONBOARD" down
            write_netplan "$NETPLAN/30-wifi.yaml" "$ONBOARD" "$ONBOARD_SPARE_MAC" 700 true false
            netplan generate && systemctl daemon-reload
        fi
        bring_up "$DONGLE" "$IDENTITY_MAC" || exit 1
    else
        bring_up "$DONGLE" "$DONGLE_OWN_MAC"
        bring_up "$ONBOARD" "$IDENTITY_MAC" || exit 1
    fi
    networkctl reload
    networkctl reconfigure "$DONGLE" "$ONBOARD" 2>/dev/null
    for _ in $(seq 1 40); do
        ip -4 -br addr | grep -q "10\.0\.0\.187" && break
        sleep 1
    done
    log "switched to $mode (onboard radio $([ "$mode" = onboard ] || [ "$onboard_ok" = 1 ] && echo up || echo off))"
    status
}

status() {
    ip -4 -br addr show | grep -E "^(wlan|wlx)"
    ip route get 10.0.0.1 | head -1
    for i in "$ONBOARD" "$DONGLE"; do
        [ -n "$i" ] && echo "$i: $(cat /sys/class/net/$i/address) $(iw dev $i get power_save 2>/dev/null)"
    done
    systemctl list-timers wifi-primary-rollback.timer --no-legend 2>/dev/null
}

case "${1:-status}" in
    dongle | onboard)
        if [ "${2:-}" != --no-rollback ]; then
            systemctl stop wifi-primary-rollback.timer 2>/dev/null
            systemd-run --quiet --unit=wifi-primary-rollback --on-active="${ROLLBACK_MIN}min" \
                "$(readlink -f "$0")" onboard --no-rollback
            log "rollback to onboard armed in $ROLLBACK_MIN min (confirm to keep)"
        fi
        switch_to "$1"
        ;;
    dryrun)
        root=/tmp/wifi-primary-dryrun
        rm -rf "$root" && mkdir -p "$root/etc/netplan"
        cp -a /etc/netplan/. "$root/etc/netplan/"
        NETPLAN="$root/etc/netplan"
        write_netplan "$NETPLAN/31-wifi-dongle.yaml" "$DONGLE" "$IDENTITY_MAC" 600 false true
        write_netplan "$NETPLAN/30-wifi.yaml" "$ONBOARD" "$ONBOARD_SPARE_MAC" 700 true true
        for f in "$NETPLAN"/3*.yaml; do echo "== $f"; sed -E 's/(password:).*/\1 ***/' "$f"; done
        netplan generate --root-dir "$root" || echo "netplan generate FAILED"
        for f in "$root"/run/systemd/network/*wl*; do echo "== $f"; cat "$f"; done
        grep -h ExecStart "$root"/run/systemd/system/systemd-networkd-wait-online.service.d/*.conf 2>/dev/null
        rm -rf "$root"
        ;;
    confirm) systemctl stop wifi-primary-rollback.timer && log "confirmed, rollback disarmed" ;;
    status) status ;;
    *) echo "usage: $0 dongle|onboard|status|confirm" >&2; exit 2 ;;
esac
