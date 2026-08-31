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

set -u
BUS=2
CHIP=gpiochip1
PC5=69
PC6=70

present() { i2cdetect -y "$BUS" 2>/dev/null | grep -q " $1 "; }
readdr() { i2ctransfer -y "$BUS" "w3@0x$1" 0x00 0x01 "0x$2" && sleep 0.2; }

hold() { # hold given line=value pairs in background, remember pid
  gpioset -c "$CHIP" "$@" &
  GP=$!
  sleep 0.8
}
release() { kill "$GP" 2>/dev/null; wait 2>/dev/null; }

# Phase 1: only the always-on (PC9) sensor awake.
hold "$PC5=0" "$PC6=0"
if present 29; then readdr 29 30; fi
release

# Phase 2: wake the PC5 sensor, it boots at 0x29.
hold "$PC5=1" "$PC6=0"
if present 29; then readdr 29 31; fi
release

# Phase 3: wake the PC6 sensor.
hold "$PC5=1" "$PC6=1"
if present 29; then readdr 29 32; fi
release

echo "ToF bus state:"
i2cdetect -y "$BUS" | grep -E "20:|30:"
for a in 30 31 32; do
  present "$a" && echo "0x$a OK" || echo "0x$a MISSING"
done
