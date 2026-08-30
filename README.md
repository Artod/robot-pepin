# Pepin

A mobile home robot: an IKEA RASKOG cart on two driven wheels, a 6-DoF arm,
a lidar, and a phone for a face.

Built on the open-source [XLeRobot](https://github.com/Vector-Wangel/XLeRobot)
platform (dual-wheel variant) and the
[LeRobot](https://github.com/huggingface/lerobot) ecosystem.

**Status: hardware bring-up.** First scans and first motion are in progress —
nothing to show yet.

## Hardware

- IKEA RASKOG cart, differential drive (2x Feetech STS3215, 5" wheels)
- SO-ARM101 6-DoF arm (6x Feetech STS3215)
- LDRobot LD19 lidar
- 4x VL53L1X time-of-flight sensors
- Orange Pi Zero 3 as the onboard relay; a laptop as the brain
- 5S li-ion pack (74 Wh) with a 12 V main bus
