# Pepin

A differential-drive indoor cart that drives itself to a named place on a saved map —
localisation, planning, control and recording all on a $35 single-board computer.

![Pepin navigating: lidar scan on the map, ToF marks, local costmap and the tracked pose in Foxglove](docs/figures/nav_live.png)

*One goal in flight: the static map underneath, the live scan on top of it, the three ToF cones
marked into the local costmap, and the pose the scan matcher is holding.*

## What it is

Say `ros/go.sh printer` and the cart goes to the printer. It knows where it is because it
matches every lidar revolution against a map of the flat; it knows what is in front of it
because the lidar and three time-of-flight sensors write into the costmaps ten times a second;
it stops with its bumper against the furniture, because that is where a manipulator has to
reach.

Pepin is the mobility organ of a companion robot that does not exist yet. Everything the arm
and the face will one day need — a frame that stays true, a body that fits through doorways,
a name for every place in the flat, a recording of every drive — is being built here first,
on an IKEA cart with two servos for wheels.

The whole autonomy runs on the robot. There is no laptop in the control loop: the board is an
Orange Pi Zero 3 with four Cortex-A53 cores and 1.5 GB of RAM, and it carries ROS 2 Jazzy,
Nav2, an EKF, the scan-matching tracker and the run recorder at once. The laptop watches
through Foxglove and sends one line of JSON when it wants the robot to go somewhere.

## Numbers

Measured on the robot, on the board, during real drives.

| What | Value |
| --- | --- |
| Scan match against the map, on the board | 37–40 ms mean |
| Scan-to-odometry wait (EKF latency) | 43–50 ms mean |
| Heading-rate residual vs commanded, p90 | 6 °/s (26 °/s before the timeline module) |
| Cross-track error, p90 | 4 cm |
| Top speed | 0.30 m/s — the base's own cap, and the wheels reach it |
| A 2.5 m printer → home leg | 16–21 s |
| Localisation confidence | 0.68–0.70 standing, 0.84 driving |
| Loop closure over a 33 m lap (mapping) | 5 cm |
| Goal tolerance | 0.10 m / 0.20 rad |
| Unit tests | 361, mypy strict, 86% coverage floor |

## Architecture

```
 LAPTOP                                  │ BOARD — Orange Pi Zero 3 (4x Cortex-A53, 1.5 GB)
                                         │ one Docker container: ROS 2 Jazzy + Nav2 1.3.12
 ros/go.sh printer ─── TCP 3337 ─────────┼─► goal_server ──► bt_navigator ──► planner_server
        (JSON lines: run number,         │        │              │            global costmap 0.5 Hz
         events, tape path)              │        │              │ /plan
                                         │        │              ▼
 Foxglove ◄── ws 8765 ── foxglove_bridge │        │        controller_server   RPP 10 Hz
                                         │        │        local costmap 3x3 m @ 5 cm, 3 Hz
 rsync   ◄── camera clip (curl on the board) │        │              │ /cmd_vel_nav
 rsync   ◄── run .jsonl, board log ──────┼─ run_recorder         ▼
                                         │        ▲        velocity_smoother   10 Hz
                                         │        │              │ /cmd_vel
                                         │        │        base_bridge (C++)
                                         │        │              │ TCP 3336, JSON lines
                                         │        │        pepin-base   50 Hz, deadman 0.5 s
                                         │        │              │ ser2net :3333
                                         │        │        2x Feetech STS3215
                                         │        │
                                         │ ─── sensing ─────────────────────────────────────
                                         │ LD19 ─► ldlidar ─► hull box filter ─► /scan  10 Hz
                                         │ encoders ─► pepin-base ─────────────► /odom  16 Hz
                                         │ MPU6050 ─► base_bridge ──────► /imu/data_raw  44 Hz
                                         │ /odom + /imu ─► ekf_node ─► odom→base_link    20 Hz
                                         │ /scan + map ─► relocalizer ─► map→odom        20 Hz
                                         │ 3x VL53L1X ─► pepin-tof :3335 ─► tof_bridge
                                         │              ─► /tof/{front,left,right}       14 Hz
```

The sensor nodes and the base bridge are composed into one container process at `nice -10`; Nav2
runs in a second one at `nice +5`, so a busy planner can never starve the lidar. Every process
that could be a separate node and is not saves about 140 MB on a 1.5 GB board.

### Timing

| Loop | Rate |
| --- | --- |
| Lidar revolution | 10 Hz, 455 beams |
| Wheel odometry (`/odom`) | 16 Hz |
| Gyro (`/imu/data_raw`) | 44 Hz |
| EKF, `odom → base_link` | 20 Hz |
| Scan match | every scan while moving, ≥1 Hz standing still |
| `map → odom` broadcast | 20 Hz |
| Lost check (is the pose still trustworthy?) | 1 Hz |
| Tracker timing report | every 30 s |
| Controller (Regulated Pure Pursuit) | 10 Hz |
| Velocity smoother | 10 Hz |
| Local costmap | update 3 Hz, publish 1 Hz |
| Global costmap | 0.5 Hz |
| Planner | up to 2 Hz |
| ToF, per sensor | 14 Hz |
| Wheel loop on the board | 50 Hz, 0.5 s deadman |
| Laptop heartbeat (split mode) | 2 Hz |

## How it drives

One command, end to end:

1. **`ros/go.sh printer`** opens a socket to the goal server on the board and writes one JSON
   line. No client boots, no ROS process starts: a goal costs a socket write, not the 8–15 s
   an ssh-and-import client used to cost.
2. **The goal server** looks the name up in the map's own places book (`<map>.places.yaml`),
   takes the next run number, opens the tape — which already holds the last 15 s of every topic,
   so the file begins *before* the command did — and sends the pose to Nav2. It answers with one
   JSON line per event: accepted, feedback, arrival, done.
3. **The behaviour tree** asks the selected planner for a path, with NavFn standing behind it:
   a footprint-checking planner can refuse to plan from a cart parked against a table, and NavFn
   cannot, so the robot is never stranded by the planner it happens to be using. Stale cells are
   forgotten on a timer (local 1 Hz, global 0.1 Hz), and a failure is answered by backing up,
   clearing, spinning and waiting in a round robin — a different answer on each retry, twenty
   retries before giving up.
4. **Regulated Pure Pursuit** follows the path at 10 Hz, slowing on curvature and on costmap cost.
5. **The velocity smoother** limits acceleration, and **the C++ base bridge** turns `/cmd_vel`
   into a twist on a TCP socket.
6. **`pepin-base`** on the host owns the wheels: it applies the twist and reads the encoders at
   50 Hz next to the UART, and a 0.5 s deadman cuts the motors if anything upstream goes quiet.
   The two **Feetech STS3215** run in wheel mode on a single serial bus.

And back:

- Encoders → **wheel odometry** (16 Hz) and the MPU6050 → **gyro yaw rate** (44 Hz) are fused by
  `robot_localization`'s EKF at 20 Hz into `odom → base_link`. The filter takes the wheels'
  forward speed and the gyro's yaw rate — and nothing else. On carpet the wheels over-report
  rotation by 10–25% and their integrated x/y carries that error with it, so fusing the wheel
  *position* would drag the filter back onto the very trajectory the gyro exists to correct.
- Lidar → **hull box filter** (the cart's own posts and cables, +5 cm, cut out of every scan) →
  **the tracker**, which publishes `map → odom` at 20 Hz.

## Localisation

There is no AMCL in the loop. AMCL's parameters are still in the file and it can be run for
comparison, but `tf_broadcast` is false: the frame belongs to our own tracker, because AMCL only
searches where its particles already are, and a cart that gets carried across the room needs to
find itself again with nobody's help.

**The tracker** is a correlative scan matcher against the static map: a brute-force search over
a ±9 cm / ±9° window in 3 cm and 1.5° steps, 120 beams, about 40 ms per match on an A53 core.
The best pose is not adopted whole — only half of the residual is (correction gain 0.5), because
at full gain the matcher's own noise reached the wheels through `map → odom` ten times a second
and the cart weaved.

**Time is fixed before the match, not after** (`src/pepin/timeline.py`). A lidar revolution is not
a photograph:

- The odometry history is interpolated **at the scan's own timestamp**, never at the newest pose.
  The EKF runs 43–50 ms behind the scan, and in a pivot that gap is a degree or two of heading
  that used to go straight into the correction.
- Every beam is **deskewed** to its own instant. The LD19 stamps the *end* of the revolution and
  the newest beam sits at index 0; over 100 ms of turning the first and last beams are 3–4°
  apart.
- A **scan gate** releases a scan only once odometry covers its whole revolution — a scan is
  matched against the pose it was taken at, or not at all.
- A **motion filter** rests the matcher while the cart stands still.

Together these took the heading-rate residual against the commanded rate from 26 °/s to 6 °/s
at p90.

**When the pose is wrong, it says so.** A fit score is computed every second: above 0.50 the
robot drives, and three consecutive checks below 0.55 (or a collapse, which is what being lifted
looks like) trigger a whole-map FFT search in a worker thread. A candidate is never adopted on
its own word — it must be confirmed by a second search on a *new* scan before the pose is
allowed to move. Failed searches back off exponentially so a lost robot does not saturate a
board that is also driving. A separate check watches for **slip**: the wheels report motion and
two consecutive scans show the same picture, which on this cart is the only honest slip signal
there is.

`map → odom` is future-dated by 0.5 s, the way AMCL does it, so consumers always have a
transform for "now".

## The map it runs on

The map is built once, from a recorded drive, and then frozen. The 33 m lap below returned to
its exact starting point, so the distance between the end of each estimated path and its start
is that method's honest error.

| Wheel odometry only | + correlative scan matching | + pose-graph loop closure |
| --- | --- | --- |
| ![lap3, odometry only](docs/figures/lap3_odometry_only.png) | ![lap3, scan matched](docs/figures/lap3_scan_matched.png) | ![lap3, loop closed](docs/figures/lap3_loop_closed.png) |
| path ends 8 m from the start | 0.73 m | **0.05 m** |

Left: every scan placed where the encoders say the robot was — carpet slip over-counts turns and
dead reckoning wanders off by metres. Middle: each keyframe corrected by the correlative matcher
against the map built so far (`src/pepin/scanmatch.py`) — straight walls, rooms, doorways, and
the accumulated drift still there. Right: keyframes as pose-graph nodes, matches as edges,
revisits detected and verified (`src/pepin/slam.py`), Gauss-Newton over SE(2) with the start
pinned (`src/pepin/posegraph.py`) — 5 cm.

`slam_toolbox` is also wired in (`ros/mode.sh slam` + `ros/savemap.sh NAME`) for building a map
live while teleoperating.

## Perception and costmaps

**LD19 lidar**, 10 Hz, 455 beams, mounted upside down at 0.20 m with a yaw of −87.5°. The driver
emits a counter-clockwise scan for an upright sensor, so the static transform carries a roll of
π to mirror it back. A box filter removes the cart's own hull plus 5 cm of margin — those
returns are its posts and cables, they travel with it, and the costmap used to turn them into a
wall that made every in-place turn a collision.

**Three VL53L1X time-of-flight sensors** (front, left, right) look where the lidar's single
horizontal slice cannot: at the height of a shoe, a cable, a cat. Medium ranging, ~14 Hz each.
Each is believed only as far as its own 27° cone stays off the floor — 0.96 m for the front
sensor at 0.27 m — because beyond that a grazing return off the carpet is indistinguishable from
a wall. A real return is held for 1.2 s so one dropped frame cannot erase a mark the sensor just
made, and a sensor that dies is re-addressed and re-initialised by systemd before every restart.

**The footprint is the true polygon**: 0.0625 m front, 0.30 m rear, 0.275 m half-width, with
`footprint_padding: 0.0`. Nav2's 1 cm default padding was silently in force and it is what made
the cart timid — a centimetre of phantom hull in front of a bumper that is only 6.25 cm ahead of
`base_link` turned "parked against the table" into "in collision". Bumper-to-furniture parking is
the working case, not the edge case.

**Local costmap**: 3×3 m rolling window at 5 cm, updated at 3 Hz in the *map* frame (a slipping
wheel feeds `odom` a metre of motion the robot never made). Layers: the lidar obstacle layer,
then one range layer *per ToF sensor* — one shared layer let the front sensor clear the marks the
side sensors had just written wherever the cones overlap — then inflation at 0.35 m / 5.0.

**Global costmap**: the static map at 0.5 Hz, inflation 0.55 m / 2.0, and the ToF layers are in
here too, because a plan is what routes *around* an obstacle and a sensor whose returns never
reach the planner can only ever stop the robot. `track_unknown_space` is false and planners may
cross unknown cells: the survey does not cover every corner of the flat, and a cell nobody
looked at is floor until the lidar says otherwise — a robot must always be able to plan its way
out of where it already stands.

## Planning and control

**Planners are selectable per run** — `ros/go.sh planner navfn|theta|smac|lattice`:

| Plugin | What it gives |
| --- | --- |
| `GridBased` — NavFn (default) | grid search, never refuses a start cell, the fallback behind every other one |
| `Smac2D` | A* with costmap cost in the metric and a smoother after |
| `ThetaStar` | any-angle: long straight segments, which is what a cart that turns in place wants |
| `SmacPlannerLattice` | a state lattice on the diff-drive control set — real motions, including turning on the spot |

**Behaviour tree** (`ros/params/pepin_nav_to_pose.xml`): the stock replanning-and-recovery tree
with the planner wrapped in a fallback to NavFn, stale-costmap clearing on a timer, and the
recovery round robin reordered so the cart backs up before it tries to spin — a cart wedged
between the sofa and the table cannot spin, and the spin is exactly the move that slips the
wheels on carpet.

**Progress** is judged by `PoseProgressChecker`: 0.15 m *or* 0.5 rad within 6 s. A legal pivot in
place is progress; a translation-only checker aborted goals mid-turn. Goal tolerance is 0.10 m
and 0.20 rad.

**Controller: Regulated Pure Pursuit** at 10 Hz — a fraction of MPPI's CPU on four A53 cores, and
enough for a cart at 0.30 m/s. Lookahead 0.40–0.80 m, velocity-scaled. Speed is regulated by
curvature and by costmap cost, with the controller's cost model matched exactly to the local
inflation layer (0.30 m / 5.0) so it reads its own clearance correctly instead of crawling past
walls for no reason. The controller carries no veto of its own (`use_collision_detection` is
off): its stock check judges the current pose first, and one lethal cell against the footprint
outline — a printer corner 5 cm away after a pivot — refused every command, including the one
that drives away. Obstacles are cost: the planner routes around lethal cells, the regulated speed
crawls near them, and the recovery behaviours keep their own collision checks, each with a time
allowance of about twice its nominal duration so a blocked retreat fails instead of pushing.
`rotate_to_heading_min_angle` is 1.2 rad: anything under 69° is driven out
as an arc rather than pivoted, because `base_link` sits 0.30 m ahead of the rear corners and a
pivot sweeps 0.41 m — the arc is what a person does in a tight corner. Pivots, when they happen,
run at 0.5 rad/s.

**Velocity smoother**: 0.30 m/s ceiling, 0.5 m/s² acceleration, 1.0 m/s² braking. The base's cap,
the controller's target and the smoother's ceiling are one number, held equal by a test.

## Every drive is recorded

There is no "record this run" flag. Runs are numbered (`0001`, `0002`, …) and every goal writes
a tape, because any run may turn out to be the one worth showing.

A tape is a `.jsonl` file that opens with a **15 s prelude** — the recorder keeps the last seconds
of every topic in memory, so the file starts before the command did and the initial turn, the
part actually worth watching, is on it. It carries the raw lidar scans, wheel odometry, EKF
odometry, gyro, the tracker's pose and fit, the plan, the local costmap, the three ToF ranges and
the commanded twist. The costmap matters: it is the proof of what the robot itself believed —
which cells it held for occupied when it refused to move.

`ros/go.sh` brings the run home: the tape, a slice of the board's container log, and the camera
clip — captured on the board itself by the goal server (a plain copy of ustreamer's MJPEG stream,
no re-encoding, no laptop in the loop) and wrapped into mkv on the laptop. It ends with a verdict line naming the run number and which planner actually
drove — the tree can substitute NavFn invisibly, and a good drive credited to the wrong planner
is worse than no measurement.

## Hardware

- **IKEA RASKOG cart**, differential drive: 2× Feetech STS3215 in wheel mode, 0.125 m wheels,
  0.505 m track, 4096 ticks/rev, one serial bus
- **Orange Pi Zero 3** (Armbian, Debian trixie): 4× Cortex-A53 at 1.4 GHz, 1.5 GB RAM — the whole
  autonomy runs here
- **LDRobot LD19** 360° lidar under the mid shelf, upside down at 0.20 m
- **3× VL53L1X** time-of-flight sensors: front at 0.27 m, left and right at ~0.16 m, all level
  and facing forward
- **MPU6050** IMU on I²C, bolted flat over `base_link`; only its yaw rate is used
- **Overview camera** served as MJPEG by ustreamer
- Single 18 V tool battery → 12 V servo rail + two isolated 5 V converters, inline XT60 switch

The cart also carries a 6-DoF SO-ARM101 and a 2-DoF neck for the phone that will be the robot's
face; neither is part of the navigation stack described here.

## Software

**On the board**, one Docker container (`ros:jazzy-ros-base`, pinned by digest) with Nav2 1.3.12,
`slam_toolbox`, `robot_localization`, `laser_filters`, `foxglove_bridge`, the LD19 driver built
from source, and our `pepin_bringup` (Python) and `pepin_base_cpp` (C++, ~25 MB against the Python
bridge's ~190) packages. Host systemd units outside the container: `pepin-base` (wheels, :3336),
`pepin-tof` (:3335), `pepin-camera` (ustreamer, :8080), `pepin-ros` (the container itself),
`ser2net` (:3333, the servo bus) and `tof-init`.

**On the laptop**: Foxglove Studio on `ws://<board>:8765`, and `ros/go.sh` and its siblings —
short shell scripts, one job each, no client to boot.

**The Python library** (`src/pepin`, Python 3.12, numpy) is imported by the ROS package and never
imports it back. It holds the board servers, the algorithms (scan matching, occupancy mapping,
pose graph, localisation, the timeline, slip detection, ToF horizon, places) and the tape. Every
decision that can be pure is pure, which is why it can be tested without a robot: the unit tests
need no hardware and run on every commit. `ruff`, `mypy --strict` and a pre-commit hook with an 86% coverage floor
run on every commit; hardware tests live in `tests/hardware` behind `--hardware`.

## Quick start

```bash
# once per clone
uv sync
git config core.hooksPath .githooks
uv run pytest && uv run mypy && uv run ruff check .

# the robot
ros/build.sh                            # sync ros/ + src/pepin to the board, build the image there
ros/sync.sh                             # push code changes, restart the container (~20 s, no rebuild)
ros/mode.sh nav /maps/<map>.yaml        # Nav2 + the tracker on a saved map
ros/mode.sh sensors                     # lidar, base bridge, Foxglove — nothing that localises

# driving
ros/go.sh printer                       # go to a named place; Ctrl-C cancels
ros/go.sh -1.0 0.3 90                   # ...or to map coordinates, with a heading
ros/go.sh mark sofa                     # name the spot the robot is standing on
ros/go.sh where | places | cancel
ros/go.sh planner theta                 # swap the planner for the next run
ros/go.sh trip                          # printer, then home
ros/stop.sh                             # the red button: wheels stopped within a second

# a new map
ros/mode.sh slam && ros/teleop.sh       # build a map while driving it by hand
ros/savemap.sh flat3                    # save it on the board and fetch it here
```

Watch it in Foxglove (`brew install --cask foxglove-studio`) on `ws://<board>:8765`: a 3D panel
with `/map`, `/scan`, `/tf`, both costmaps and `/plan`.

## Repo layout

```
src/pepin/     the Python library: board servers (base, ToF), drivers and links, and the
               algorithms — mapping, scanmatch, posegraph, slam, localization, timeline,
               watch, slip, tof_horizon, places, tape, deployment
ros/           the ROS 2 side: pepin_bringup (base/ToF bridges, relocalizer, goal server,
               run recorder, link watch, launch files), pepin_base_cpp, params/, maps/,
               tools/, Dockerfile, and the shell scripts that drive the robot
board/         Orange Pi: systemd units, ser2net, udev rules, ToF init
config/        base geometry and speed caps, lidar and ToF mounts (JSON)
scripts/       laptop entry points: drive, build_map, render_slam, replay_nav, health_check
tests/         unit (fast, no robot) and hardware (--hardware) tiers
docs/          figures
```

## Status

**Working**: the cart drives between marked places on a static map of a real flat, parks against
furniture, recovers from being carried or pushed by searching the whole map for itself, and
records every drive with its scans, poses, plan, costmap, log and video.

**Next**: making the thin-client split the daily mode — the planner and the recorder on the
laptop over zenoh while the board keeps the reflexes — and goals named from what the camera sees
instead of from a hand-written places book.

## Credits

Built on the open-source [XLeRobot](https://github.com/Vector-Wangel/XLeRobot) platform
(dual-wheel variant) and the [LeRobot](https://github.com/huggingface/lerobot) ecosystem
for arm calibration.

License: to be added.
