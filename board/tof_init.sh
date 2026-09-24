#!/bin/bash
# Assign unique I2C addresses to the three VL53L1X ToF sensors at boot.
#
# All sensors power up at the factory address 0x29 (volatile). Two of them
# have controllable XSHUT lines (PC5 = line 69, PC6 = line 70 on gpiochip1);
# the third one sits on PC9, which the kernel refuses to drive (IRQ-tied),
# so that sensor is always awake and gets its address first.
#
# Target map: PC9-sensor -> 0x30, PC5-sensor -> 0x31, PC6-sensor -> 0x32.
#
# NOTE: this board's pinctrl retains the last driven level after a GPIO line
# is released, so XSHUT must be explicitly driven high to wake a sensor —
# releasing the line is not enough.
#
# VERIFIED AND RETRIED (2026-09-23). All three sensors read status 255 (no I2C
# answer) for a whole day: the bus showed one sensor at the factory 0x29 and
# none at 0x30-0x32, and a manual `systemctl restart tof-init` (then pepin-tof)
# fixed it at once. The sequence used to run once and exit 0 whatever it
# found, printing into a journal the stack's own logging rotates out within
# hours (SystemMaxUse=20M: nothing of that day's tof-init was left to read).
# Now every target must answer with the VL53L1X model id (0xEA 0xCC at
# register 0x010F); if one does not, the whole sequence runs again, up to
# TOF_INIT_ATTEMPTS times (default 3), and the outcome is one line in the
# journal AND in TOF_INIT_LOG (default /var/log/tof_init.log, last 200 lines).
# The exit stays 0 on a partial result, on purpose: pepin-tof Requires= this
# unit, streams the sensors that answer and reports the rest dead (status
# 255), so a failed unit would take the good sensors down with the bad one.
#
# A RETRY MUST BE ABLE TO CONVERGE. If the PC9 sensor missed its address and
# the PC5 one then woke at 0x29 beside it, one write moves both to 0x31; the
# next attempt holds PC5/PC6 low, finds nothing at 0x29 and would leave the PC9
# sensor at 0x31 for good. So phase 1 also takes the PC9 sensor from 0x31 or
# 0x32 — where, with the other two held in reset, nothing else can answer —
# whenever 0x30 is empty. The XSHUT sensors need nothing of the kind: holding
# XSHUT low resets them to 0x29, so every attempt re-addresses them afresh.

set -u
BUS=2
CHIP=gpiochip1
PC5=69
PC6=70
TARGETS="30 31 32"
MODEL_ID="0xea 0xcc"  # VL53L1X IDENTIFICATION__MODEL_ID, register 0x010F, as i2ctransfer prints it
ATTEMPTS="${TOF_INIT_ATTEMPTS:-3}"
LOG="${TOF_INIT_LOG:-/var/log/tof_init.log}"

present() { i2cdetect -y "$BUS" 2>/dev/null | grep -q " $1 "; }
answers() {  # ADDR: a VL53L1X answers there with its model id
  [ "$(i2ctransfer -y "$BUS" "w2@0x$1" 0x01 0x0f r2 2>/dev/null | tr 'A-F' 'a-f')" = "$MODEL_ID" ]
}
readdr() {  # FROM TO: the sensor at 0xFROM takes 0xTO until its next reset
  if i2ctransfer -y "$BUS" "w3@0x$1" 0x00 0x01 "0x$2"; then
    sleep 0.2
  else
    echo "tof_init: 0x$1 -> 0x$2 not acknowledged"
  fi
}

hold() { # hold given line=value pairs in background, remember pid
  gpioset -c "$CHIP" "$@" &
  GP=$!
  sleep 0.8
  kill -0 "$GP" 2>/dev/null || echo "tof_init: gpioset $* did not hold its lines"
}
release() { kill "$GP" 2>/dev/null; wait 2>/dev/null; }

sequence() {  # the three phases, once
  # Phase 1: only the always-on (PC9) sensor awake.
  hold "$PC5=0" "$PC6=0"
  if present 29; then
    readdr 29 30
  elif ! present 30; then
    for from in 31 32; do
      if present "$from"; then readdr "$from" 30; break; fi
    done
  fi
  release

  # Phase 2: wake the PC5 sensor, it boots at 0x29.
  hold "$PC5=1" "$PC6=0"
  if present 29; then readdr 29 31; fi
  release

  # Phase 3: wake the PC6 sensor.
  hold "$PC5=1" "$PC6=1"
  if present 29; then readdr 29 32; fi
  release
}

say() {  # one line to the journal and to the log file, which keeps its last 200 lines
  echo "$1"
  {
    printf '%s %s\n' "$(date -Is)" "$1" >> "$LOG" &&
      tail -n 200 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
  } 2>/dev/null || true
}

attempt=0
silent="$TARGETS"
while [ -n "$silent" ] && [ "$attempt" -lt "$ATTEMPTS" ]; do
  attempt=$((attempt + 1))
  sequence
  silent=""
  for a in $TARGETS; do answers "$a" || silent="${silent:+$silent }$a"; done
  msg="all three answer"
  [ -z "$silent" ] || msg="silent at 0x${silent// / 0x}"
  echo "tof_init: attempt $attempt/$ATTEMPTS: $msg"
done

echo "ToF bus state:"
i2cdetect -y "$BUS" | grep -E "20:|30:"
for a in $TARGETS; do
  case " $silent " in *" $a "*) echo "0x$a MISSING" ;; *) echo "0x$a OK" ;; esac
done
if [ -z "$silent" ]; then
  say "tof_init: 0x30 0x31 0x32 answer (VL53L1X model id) after $attempt attempt(s)"
else
  say "tof_init: FAILED after $attempt attempts: silent at 0x${silent// / 0x} (0x29 $(present 29 && echo answers || echo silent)); pepin-tof streams the others and reports these dead"
fi
exit 0
