# Pepin

A home robot on an IKEA RASKOG cart: differential drive, a 6-DoF arm, a 360-degree
lidar, and a phone for a face — with the brain on a laptop and a thin relay on board.

_Demo video placeholder: a driving clip goes here once the first mapping run is recorded._

## What it is

Pepin is a full-stack mobile manipulator built from parts you can buy: a steel kitchen
cart, ten serial-bus servos, a lidar, and a small ARM board. The board does nothing
clever — it exposes the servo bus, the lidar, the cameras, and the ToF sensors over the
network. Everything else (kinematics, odometry, scan processing, control) runs on a
laptop over wifi, so the robot's software stack is developed, profiled, and debugged on
a real machine instead of a microcontroller.

The result is a robot that drives under keyboard control today, records synchronized
odometry and lidar sessions, and is wired for mapping and autonomous navigation next.

## Architecture

```
 18 V tool battery ──┬── 12 V rail ────────── 10x STS3215 servos (one bus, IDs 1-10)
   (XT60 inline sw)  ├── 5 V (isolated) ───── Orange Pi Zero 3
                     └── 5 V (isolated) ───── powered USB hub

  ┌──────────────┐         wifi          ┌──────────────────────┐
  │   Laptop     │  ◄──── TCP ────►      │ Orange Pi Zero 3     │
  │  "the brain" │                       │ Armbian + ser2net    │
  │              │                       │  :servo bus  (TTL)   │
  │ kinematics   │                       │  :lidar      (LD19)  │
  │ odometry     │                       │  USB cameras         │
  │ lidar scans  │                       │  I2C ToF sensors     │
  │ teleop + viz │                       └──────────┬───────────┘
  └──────────────┘                                  │
                                    ┌───────────────┼───────────────┐
                                 drive base      6-DoF arm       2-DoF neck
                                 (2 servos)     (6 servos)      (2 servos)
                                 + encoders    + wrist cam      + phone face
```

## Measured

| What | Value |
| --- | --- |
| Servo transaction round trip (laptop → wifi → bus → back) | ~9 ms median |
| Lidar scan rate | 10 rev/s, ~500 points per revolution |
| Lidar frame CRC pass rate | 100% |
| Odometry closure, forward/back run | ~10 mm |
| Wheel track | ~505 mm |
| Servos on a single bus | 10 (IDs 1-10) |

## Mapping: odometry, scan matching, loop closure

Drives of the same flat, recorded with `scripts/drive.py` and replayed with
`scripts/build_map.py`. The 33 m loop below (lap3) returned to its exact starting
point, so the distance between the end of the estimated path and its start is the
honest error of each method.

| Wheel odometry only | + correlative scan matching | + pose-graph loop closure |
| --- | --- | --- |
| ![lap3, odometry only](docs/figures/lap3_odometry_only.png) | ![lap3, scan matched](docs/figures/lap3_scan_matched.png) | ![lap3, loop closed](docs/figures/lap3_loop_closed.png) |
| path ends 8 m from the start | 0.73 m | **0.05 m** |

Left: every scan placed where the wheel encoders say the robot was — carpet slip
over-counts turns by ~14%, and dead reckoning wanders off by meters. Middle: each
keyframe's pose corrected by a brute-force correlative matcher against the map
built so far (`src/pepin/scanmatch.py`) — straight walls, rooms, doorways, but the
accumulated drift is still there. Right: keyframes as nodes of a pose graph, scan
matches as edges, revisits detected and verified (`src/pepin/slam.py`), Gauss-Newton
over SE(2) with the start pinned (`src/pepin/posegraph.py`) — the loop closes to 5 cm.

Earlier, smaller drives of the same flat:

| Wheel odometry only | + correlative scan matching |
| --- | --- |
| ![lap2, odometry only](docs/figures/lap2_odometry_only.png) | ![lap2, scan matched](docs/figures/lap2_scan_matched.png) |
| ![lap1, odometry only](docs/figures/lap1_odometry_only.png) | ![lap1, scan matched](docs/figures/lap1_scan_matched.png) |

## Hardware

- IKEA RASKOG cart, differential drive: 2x Feetech STS3215 in continuous-rotation mode,
  5" wheels, ~505 mm track
- SO-ARM101 6-DoF arm (6x STS3215) with a wrist camera
- 2-DoF neck (pan/tilt) carrying the phone that will be the robot's face
- LDRobot LD19 360-degree lidar, mounted under the mid shelf
- 3x VL53L1X time-of-flight sensors for near-field obstacles
- 4K overview camera
- Orange Pi Zero 3 (Armbian) as the onboard relay; laptop as the compute
- Single 18 V tool battery → 12 V servo rail + two isolated 5 V converters, inline
  XT60 switch

## Software

Python 3.12+, `uv`, src layout, package `pepin`.

- **Feetech bus client, written from scratch** for a lossy link: framing, checksums,
  per-request deadlines, retries, and latency telemetry over raw TCP.
- **Diff-drive kinematics and wheel-encoder odometry** using the exact arc model.
- **LD19 driver**: frame parsing, per-revolution scan assembly, mount geometry with
  mirroring and masked sectors.
- **Teleop**: keyboard driving with a live [rerun.io](https://rerun.io) view.
- **Session recording** to JSON lines for offline mapping.
- **Launch-readiness health check** that polls every subsystem before a run.
- Per-run file logging; ruff, mypy strict, pytest split into unit and hardware tiers,
  and a pre-commit hook that runs all of it in under a second.

## Quick start

```bash
uv sync
git config core.hooksPath .githooks

uv run pytest                                # unit tier, no robot needed
uv run python scripts/health_check.py        # poll every subsystem
uv run python scripts/drive.py --name lap1   # keyboard driving + recording
```

## Repo layout

```
src/pepin/   laptop-side package: bus, feetech, lidar, kinematics, odometry,
             teleop, recording, telemetry, geometry, logging
board/       Orange Pi configs: ser2net, udev rules, ToF boot init
config/      robot geometry and sensor maps (JSON)
scripts/     bring-up and operations: health_check, drive, base_smoke, lidar_scan,
             jog, calibrate_neck, scan_bus, setup_motor_id
tests/       unit and hardware tiers
```

## Status and roadmap

**September 2026 — hardware integrated, first maps built.** The base drives under
keyboard control, odometry closes a forward/back run to within ~10 mm, the lidar and
servo bus are served over the air simultaneously, and two recorded drives produce
clean occupancy maps once scan matching corrects the wheel odometry, and a pose graph
closes a 33 m loop to 5 cm (see above).
Next:

1. Live localisation against a saved map, then autonomous navigation
2. Mobile manipulation with the arm
3. The phone face and voice

## Credits

Built on the open-source [XLeRobot](https://github.com/Vector-Wangel/XLeRobot) platform
(dual-wheel variant) and the [LeRobot](https://github.com/huggingface/lerobot) ecosystem
for arm calibration.

License: to be added.
