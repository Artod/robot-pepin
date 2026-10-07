# Pepin on ROS 2 Jazzy

Navigation runs **on the Mac**: planner, controller, both costmaps, the behaviour tree, the goal
server and the run recorder in one container, `pepin-macnav` (`ros/laptop.sh nav`), beside the
camera's mapping in `pepin-vslam` (`ros/laptop.sh vslam`). The board is a sensor box: lidar, ToF,
wheels, IMU and neck, the C++ base bridge and the EKF in its own container, `pepin-ros` (Ubuntu
24.04 on `ros:jazzy-ros-base`; Armbian has no Jazzy binaries). The velocity goes back to it on
`/cmd_vel`, and the base server's own 0.5 s deadman is what a WiFi stall meets.

```
laptop (Mac)                                  board (Orange Pi Zero 3, 1.5 GB + zram)
pepin-macnav: Nav2 (planner, controller,      docker pepin-ros: ldlidar_node -> laser_filters
  costmaps, behaviour tree, behaviours,         box filter (/scan), base_bridge (C++: /odom,
  velocity smoother), goal_server               /imu/data_raw, /zupt, /neck/state, the camera's
  (127.0.0.1:3337), run_recorder, gaze          tf), EKF (odom -> base_link), rf2o, tof_bridge
  (the head's one owner, 127.0.0.1:3339)
pepin-vslam: RTAB-Map (map -> odom), the      host: pepin-base.service (:3336, the wheels),
  camera, depth, visual odometry, the           pepin-tof.service (:3335), ser2net (:3333,
  volume, foxglove_bridge (ws 8765)             the servo bus, the base server's own link)
        /cmd_vel ───────── zenoh, WiFi ──────►  base_bridge -> base server
        ◄──── /scan, /odom, ToF, IMU, neck, tf
```

## Layout

| Path | What |
| --- | --- |
| `ros/Dockerfile` | the board's sensor image: Jazzy base + nav2-lifecycle-manager (the lidar's only), laser-filters, robot-localization, rmw_zenoh; the LD19 driver ([Myzhar/ldrobot-lidar-ros2](https://github.com/Myzhar/ldrobot-lidar-ros2)) and rf2o built from source; our `pepin_bringup` and `pepin_base_cpp` |
| `ros/Dockerfile.laptop` | the laptop's image (`ros/laptop-build.sh`): the board's plus Nav2, RTAB-Map and the image pipeline |
| `ros/Dockerfile.xfeat`, `ros/xfeat/` | the laptop image with RTAB-Map rebuilt for XFeat + LighterGlue ("The XFeat image") |
| `ros/run.sh` | the board's `docker run`: host networking, the lidar device, `ros/maps` (recordings) and `ros/params` mounted |
| `ros/pepin_bringup/` | ament_python package: every Python node and the launch files (`robot` / `bringup` on the board, `vslam` and `nav` on the Mac) |
| `ros/pepin_base_cpp/` | ament_cmake package: the board's base bridge in C++ ([its README](pepin_base_cpp/README.md)) |
| `ros/pepin_gaze_bt/` | ament_cmake package: `AskGaze`, the behaviour tree's door to the gaze arbiter ("Gaze") |
| `ros/params/` | Nav2 parameters for this cart (footprint, speeds, rates), the EKF, the behaviour tree, the bag QoS |
| `ros/zenoh/router.json5` | both zenoh routers' configuration ("The zenoh routers") |
| `ros/models.sh`, `ros/depth_host.sh` | the model services on the Mac's GPU ("The model services") |
| `ros/tools/` | one-job tools: goal client, planner check, `map -> odom` reader, bag converter, flags doc, stereo calibration, `npz_to_map.py` (an old `data/maps/*.npz` grid to map_server format) |
| `ros/calibrate.sh` | checkerboard calibration of the stereo head and the mono camera ("Camera calibration") |
| `ros/replay.sh`, `ros/replay/` | the recorded drives through Nav2's own costmaps with candidate parameters ("Replay") |
| `ros/sim.sh`, `ros/sim/` | the kinematic simulator: our Nav2 on a simulated cart in RTAB-Map's saved room ("Simulation") |

## Deploying a change

The containers mount the code from the host (`run.sh` on the board, `laptop.sh` here), so a
Python change needs no image, only the processes that hold the old code restarted.

| Change | Command | What restarts |
| --- | --- | --- |
| Python in `src/pepin` or `ros/pepin_bringup/pepin_bringup` | `ros/push.sh FILE...` | the running nodes that import it, on both halves |
| a launch file, `ros/params`, `config/`, a module a launch file imports | `ros/restart.sh board --deploy`, `laptop` or `both --deploy` | the half's whole stack |
| the whole tree, nothing restarted | `ros/sync.sh` (`--restart`: the board's stack too) | nothing |
| one node by hand | `ros/board.sh kick NODE`, `ros/laptop.sh kick NODE` (no name: the list) | that node |
| the Dockerfiles, the C++ packages, rf2o's patch | `ros/build-image.sh`, `ros/laptop-build.sh` | see "Building the image" |

`ros/push.sh` takes its plan from `pepin.push`: the nodes whose Python imports a changed module
through any chain of imports (`uv run python -m pepin.push plan FILE` prints each chain). What a
kick cannot deliver is refused before anything is touched, with the restart that delivers it.
Otherwise the files go to the board by rsync (exactly those, never `--delete`) and the nodes of
both halves are kicked at once, one line each. `--dry-run` or `PEPIN_PUSH_DRY=1` prints the plan
and touches nothing. A kick ends the node with SIGINT, the launch respawns it two seconds later
and the kick waits for the new pid's ready line (`ros/kick_ready.awk`): push at rest, not
mid-drive.

## Driving

Nav2 runs on this Mac and every command speaks to its goal server on `127.0.0.1:3337`:

```bash
ros/laptop.sh vslam                 # the camera's mapping: RTAB-Map is the map and owns map -> odom
ros/laptop.sh nav                   # Nav2 in pepin-macnav (controller mppi); `nav down`, `nav logs`
ros/ready.sh                        # the cart put on its base: seeded there, voxels and costmaps
                                    # emptied, one plan proven, the pose read back ([X Y YAW]: elsewhere)
ros/preflight.sh                    # ready for a goal? pose, lidar, planner, snapshots, the three
                                    # ToF (each at its rate, not all "unknown"), recognition,
                                    # Foxglove, one plan; --no-plan during a drive
ros/goto.sh printer                 # a goal; Ctrl-C cancels it; exit 0 only when reached
ros/goto.sh -1.0 0.3 90             # map coordinates and a heading
ros/goto.sh cancel | where | places | mark NAME | seed X Y [YAW] | planner NAME
ros/reset_world.sh                  # empty the voxels and both costmaps, nothing restarted
ros/watch.sh                        # Nav2's own words, live
ros/speed.sh [X]                    # the one speed (config/base.json max_wheel_speed_m_s) in the base
                                    # and every Nav2 limit; X sets them all live until a restart
ros/gaze_gate.sh [on|off|KNOB V]    # the gaze gate (frames of a head saccade or a fast yaw dropped)
                                    # in depth_stream, sensor_pack and visual_odometry at once
ros/exposure.sh [show|auto|manual MS [GAIN]|capped MS]
                                    # the head camera's exposure, live (config/camera.json's block
                                    # is what pepin-camera applies at its start)
ros/stop.sh                         # the red button (below)
```

`ros/laptop.sh nav` starts `nav.launch.py` in `pepin-macnav` on `pepin-net`, with the zenoh
session's tx queues raised to 16 batches like every laptop container (`ZENOH_CONFIG_OVERRIDE`; 32
panics zenoh) and port 3337 published on localhost. `PEPIN_MAP` (`/maps/flat3_straight.yaml` by
default) names only the places book beside a map file: no map_server runs, and both costmaps read
RTAB-Map's `/map`. `PEPIN_RECORDER=bag` starts the bag recorder instead of the JSONL one ("Two
recorders"). The behaviour-tree watcher (`ros/tools/bt_watch.py`) appends to
`ros/maps/rec/bt_live.log`.

Every goal of `ros/goto.sh` leaves, under `ros/maps/rec/`: `<stamp>_goto.log` (the goal's own
lines), `<stamp>_goto.nav2.log` (Nav2's reasons as they happen, `nav2|`, and the tree's
transitions, `bt|`, also printed in the terminal), the numbered tape `NNNN_<utc>Z_<place>.jsonl`
and the run's camera clip `NNNN_<utc>Z_<place>_cam.mjpeg`. The clip belongs to the run's
recorder, not to goto.sh: every run has one whoever sent the goal (the tray, the voice tools), it
is named at the end of the drive and missed aloud; a clip that has not started 4 s in is said
in the recorder's log and started once more. `PEPIN_GOTO_CLIP=1` adds goto.sh's own ffmpeg film
(`<stamp>_goto_cam.mkv`, `ros/clip.sh`), a second reader on the board's WiFi. goto.sh records
into the directory `pepin-macnav` mounts as `/maps`, whichever checkout it runs from. Ctrl-C, a closed terminal and a kill each
cancel the goal. `round [NAME]` and `move NAME SEG...` are the two measured motions without the
planner; they run on the board and go only while the goal server's `where` says `"navigating":
false`.

`where` also says `lidar`: `ok`, or how long no scan has reached Nav2. A lidar plugged in after
the board's stack started needs that stack restarted (`ros/restart.sh board`).

**The red button**, `ros/stop.sh` (the tray's first item runs it): the base server's own stop at
once, every goal cancelled through the goal server (confirmed by a navigator within 3 s), and the
base's stop again, believed only from its state stream: the last state lines of a one-second
window must say the wheels are commanded still. Cancelled and still is the whole stop, and Nav2
stays up. Otherwise what commands the wheels is killed first (Nav2's composed container, which
its launch respawns idle, and a measured motion on the board), because the base's stop is not
latched; then the base's stop and its confirmation once more, then `pepin-macnav` stopped. It
exits 0 only when the wheels read still, and never restarts the board, whose odometry the map is
tied to; `ros/laptop.sh nav` brings Nav2 back.

**Game teleop**, `uv run python -m pepin.teleop --game [--host 10.0.0.187]`: a small window where
keys act only while held. Arrows drive the wheels, W/S tilt and A/D pan the head (the base
server's `neck_jog`, [board/README.md](../board/README.md)), Shift is slow, Space stops
everything. It speaks to the base server on :3336 directly, past ROS, so the drive is not
recorded; `ros/teleop.sh NAME` is the recorded way (see its header).

## Restarting

`ros/restart.sh board|laptop|both [--deploy] [--fresh-graph] [--no-check] [--fast] [--dry-run]`
brings the robot up working and ends on one line: `green: N checks, none failed`, or `red:` with
the failing lines above it. `board` restarts the board's sensor stack with its zenoh router
(every goal cancelled first; `--deploy` goes through `ros/sync.sh --restart`), then the laptop's
vslam if it was up, because a board restart re-zeroes the odometry RTAB-Map runs on. `laptop`
is `ros/laptop.sh vslam` then `ros/laptop.sh nav`; `both` is board first, always. Then it proves
the planner plans (4.1) and repairs it when it does not: the Nav2 container alone, then both
halves once more. A planner that answers "no path" in every direction is reported, never
restarted.

`--no-check` restarts only; `--fast` also returns without waiting for either half's first report
line; `--fresh-graph` starts the camera half on an empty RTAB-Map database; `--dry-run` prints
the order. `PEPIN_RESTART_WAIT_S`, `PEPIN_RESTART_POLL_S` and `PEPIN_PLANNER_WAIT_S` change the
waits. Every check is one `PASS`/`FAIL` line with its number; a `WARN` never fails the run, and
a check that cannot be answered says so.

| # | check |
| --- | --- |
| 4.1 | the planner: from `pepin-vslam`, `ros/tools/planner_check.py` sees a new `/global_costmap/costmap` within 10 s and gets a path 0.5 m ahead (then behind, left, right) — a plan, never motion; asked again for up to 180 s while Nav2 comes up |
| 1.1 | `ros/board.sh census`: every board process accounted for, every budget kept |
| 1.3 | the pose: `ros/goto.sh where` answers with `"pose": "tf"` |
| 1.4, 1.5 | no `Failed to meet update rate`, `Extrapolation`, `out of map bounds` or `Off Grid` in Nav2's log in the last 60 s |
| 1.8 | `/vo`, the camera's odometry the EKF fuses, reaches the board (`ros/tools/topic_rate.py`, one 5 s measurement) |
| 1.10 | `pepin-base` is active and no `torque on` is left standing in its journal |
| 1.13 | who corrects the pose: `map -> odom` is fresh (re-broadcast at 20 Hz) and not the identity (`ros/tools/map_odom.py` in the laptop's container). The identity is a `WARN` while the laptop half is under 60 s old and a `FAIL` after |
| 1.14 | `/odom_laser` (the EKF's `odom3`) is flowing; a `WARN` when `PEPIN_LASER_ODOM=false` |
| 1.15 | informational: the board's clock minus the laptop's (`ros/time.sh offset`); `WARN` over `PEPIN_CLOCK_WARN_MS` (100 ms) or when not measured; under `PEPIN_TIME_SOURCE=pool` with no server here a `PASS` "not measured, as configured" ("One clock") |
| 2.1 | `pepin-vslam` is running |
| 2.2, 2.3, 2.4 | `depth_stream` over 5 frames/s with a fitted law in its line; `depth_fusion` receiving frames, `at bound 0`; `visual_odometry` over 5 poses/s |
| 2.6, 2.7, 2.10 | the rtabmap process alive; `sensor_pack` feeding it snapshots; `rtabmap_frame` hearing its updates |
| 2.8 | no `process has died` in the container since it started |
| 2.9 | Foxglove: `ros/foxglove.sh check` (the bridge answers on `ws://localhost:8765` and advertises every topic the layout draws) |
| 2.11 | informational: `marks_audit`'s last line (the local costmap's lethal cells split into lidar-backed, camera-only and unexplained) |
| 2.12 | `pepin-vslam` carries the XFeat adapters (`/opt/xfeat/rtabmap_xfeat.py`); passes on another image only under `PEPIN_XFEAT=0` |
| 2.13 | the localisation service answers `/health` on `:8791`; down is a `WARN` (said loudly under `place_recognition descriptor`), a `FAIL` only under `registration_backend service` |
| 3.x | every live flag of the restarted half is its `FLAGS` table default (`ros/flags.sh drift`); a difference is a `WARN` |

### A node that is quietly stuck: take its stacks before the kick

Alive, silent, a percent of CPU: a kick or a restart destroys exactly the state that says why
(OpenVINS 2026-10-04: one thread asleep in rmw_zenoh's `rmw_wait` with no timeout; the gaze
arbiter froze the same way at 22:22 and nobody took its stack). First:

    ros/tools/stack.sh gaze                               # a pepin_bringup node, by module name
    ros/tools/stack.sh run_subscribe_msckf pepin-vio      # any process, by name, in a container

It writes `ros/maps/rec/stacks/<node>-<UTC>.txt`: for a Python node the faulthandler dump that
node_kit answers SIGUSR2 with (every thread's Python stack, from the container's log) and, on the
laptop, `py-spy dump --native`; for a C++ process `gdb thread apply all bt`. py-spy and gdb run in
a sidecar (`pepin-stack:latest`, built on first use) that shares the container's PID namespace,
since no image carries them; the process stops only for the second they read it. On the board
only the Python dump works (no sidecar there). Then kick.

## One localiser

RTAB-Map on the laptop owns `map -> odom`: it publishes the transform itself (`publish_tf`,
re-broadcast at 20 Hz, stamped 0.5 s ahead), and every consumer composes it with the board's own
`odom -> base_link`. Both costmaps' static layers read `/map`, RTAB-Map's grid relayed by
`pepin_bringup.rtabmap_frame`. What belongs on the board is odometry (wheels, gyro, visual
odometry, laser odometry), because that is what must survive a WiFi loss; `map -> odom` is a slow
correction. The board's scan-matching tracker that owned the edge before is on the tag
`alt/tracker-2026-09-22`.

RTAB-Map starts **localising** on its one database (`ros/maps/rtabmap.db`): a start in mapping
mode opened a new session per restart, and seventeen sessions from one evening's restarts made
the costmaps' map change thirty times in 800 s. The price is that a loaded map is not extended;
`ros/laptop.sh vslam --fresh` starts an empty one. A fresh edge is not a placed start: after a
restart RTAB-Map publishes the pose it saved at its last shutdown, so `rtabmap_frame` says on
`/localization/placement` (latched) whether this start has recognised a node of the loaded map or
been seeded (`ros/goto.sh seed X Y YAW`), and the goal server refuses goals until it has
(`ros/flags.sh set goal_server start_needs_placement false` is the switch back).

The board's EKF also takes a zero-velocity input, `/zupt`, from the C++ base bridge while its own
witnesses say the cart is certainly still ([pepin_base_cpp/README.md](pepin_base_cpp/README.md),
"Zero-velocity update").

## The zenoh routers

There is no bridge: one `rmw_zenohd` router per machine, the board's `pepin-zrouter`
(`board/pepin-zrouter.service`, `--network host`) and the laptop's `pepin-zrouter-laptop`
(`ros/laptop.sh`'s `zrouter_up`), joined by ONE TCP link that the laptop dials (`ros/lib.sh`'s
`pepin_zenoh_router_override`). Every node is a zenoh peer of its own machine's router; only the
router-to-router link crosses the WiFi.

The board's radio pings the laptop at 80-180 ms with spikes of 0.4-1.2 s even with the stack
stopped (2026-09-22), and the shipped router config closed the whole transport session after 5 s
of a blocked RELIABLE push. So both routers start on `ros/zenoh/router.json5`, passed as
`ZENOH_ROUTER_CONFIG_URI` (`ZENOH_CONFIG_OVERRIDE` is applied on top of it, which is where the
laptop's `connect/endpoints` lives). Three values, all under `transport/link/tx`:

| key | shipped | ours | what it does |
| --- | --- | --- | --- |
| `queue/congestion_control/block/wait_before_close` | 5 s | **20 s** | how long a RELIABLE push waits for a free batch before killing the session; 4x the worst measured stall |
| `queue/congestion_control/drop/wait_before_drop` | 1 ms | **50 ms** | how long a BEST_EFFORT sample waits before being dropped: the period of the fastest thing that crosses |
| `keep_alive` | 2 | **4** | keep-alives per 60 s lease, for a link that loses packets |

The file is a full copy of the image's `DEFAULT_RMW_ZENOH_ROUTER_CONFIG.json5` with every change
marked `PEPIN`, because the variable replaces the configuration instead of merging into it; its
header has the `diff` that shows the delta. The way back needs no rebuild: on the laptop
`PEPIN_ZROUTER_CONFIG= ros/laptop.sh vslam`, on the board the unit mounts the file only when
`/root/pepin-ros/zenoh/router.json5` exists (`ros/sync.sh` puts it there).

**The node sessions keep the shipped default**, on purpose: their `wait_before_close` is already
60 s, and no session link crosses the WiFi (the board's nodes are peers over loopback, the
laptop's over `pepin-net`). If a session is ever seen closing, the same file goes to the
containers as `ZENOH_SESSION_CONFIG_URI`.

**RX workers** (`PEPIN_ZROUTER_RX_WORKERS`): a push to a peer that stopped reading waits
`wait_before_close` on one of the router's RX workers, and the close it schedules needs a free
one. With zenoh's default of 2 a frozen peer wedged both and the dead transport lingered for
minutes (reproduced with a paused router, 2026-09-23). So each router runs more workers than the
sessions it holds: 16 on the board (`board/pepin-zrouter.service`), 32 on the laptop.
`PEPIN_ZROUTER_LOG` logs the transport lifecycle at debug. The laptop router takes a new value at
its next re-creation (`ros/laptop.sh stop`, then `vslam`); the board's once the unit is copied
and the router restarted.

**Deploying a change to `router.json5`**: `ros/sync.sh`, then the routers in the clean order:
laptop half down, laptop router down, board router, board stack, laptop router, laptop half. A
router restarted under a live peer has produced a one-way link before.

**Every container's log outlives its restart** (`PEPIN_LOG_ARCHIVE`, on by default). On the board
each unit's next start copies the previous container's log to `/root/pepin-ros/logs/`; on the
laptop `ros/lib.sh`'s `pepin_remove_container` copies `docker logs -t` to
`logs/containers/<UTC>_<name>.log` before it removes a container.

**A topic that crosses between the machines carries the same reliability and depth on both ends**
(`pepin.deployment.BRIDGED_QOS`, handed to every reader by `node_kit.bridged_qos_profile`):
`/imu/data_raw`, `/vo` and `/odom`, all RELIABLE ten deep. A reader and a writer that disagree on
reliability do not match at all.

## One clock (`PEPIN_TIME_SOURCE`, chrony)

Every ROS stamp on the laptop is the Docker VM's clock; every stamp on the board is the board's.
Scans, transforms and maps cross between the two, so the robot runs on one time base:

- **the laptop serves its VM's clock**: the `pepin-chrony` container (`ros/chrony/`: alpine,
  `chronyd -d -x`, no sources, `local stratum 10`) on `udp/123`, started by `ros/time.sh
  install|server` and by `ros/laptop.sh` beside the zenoh router under `PEPIN_TIME_SOURCE=laptop`
  (the default, `ros/lib.sh` and `scripts/timesync.py`);
- **the board follows it**: chrony instead of systemd-timesyncd (`board/chrony.sh`,
  `board/chrony/chrony.conf`), the laptop as a `prefer` source polled every 4-16 s and the
  internet pool beside it, the fallback when the laptop is away and a vote against a laptop clock
  that is wrong. `makestep 1 3`: stepped only within the first three updates, slewed after;
- **the board learns the laptop's address from the laptop**: `ros/time.sh` writes it to the
  board's `/etc/default/pepin-ros` (`PEPIN_LAPTOP_HOST`) and its chrony sources; if DHCP moves the
  Mac, `ros/time.sh point` re-points it.

The laptop's server does not discipline the VM clock (no `CAP_SYS_TIME`): Docker Desktop steps it
back to the Mac's clock by itself 10-26 s after each wake, and a second agent stepping the one
kernel clock every container shares would race that. A laptop sample minutes off after a wake is
a falseticker against the pool's servers, and `prefer` (never `trust`) lets the vote win.
`board/pepin-ros.service` waits up to 90 s for `NTPSynchronized` at boot, which holds under
chrony unchanged.

```bash
ros/time.sh status            # the board's chronyc tracking and sources, and this server's
ros/time.sh offset            # board minus laptop, one line (restart check 1.15)
ros/time.sh source pool       # the board on the pool alone, live; `source laptop` is the way back
ros/time.sh uninstall         # the board back on systemd-timesyncd exactly as before
```

## A new room

There is one arrangement: RTAB-Map on the laptop is the map and the owner of `map -> odom`,
whether the room is known or new. A new room is an empty database:

```bash
ros/restart.sh laptop --fresh-graph   # empty RTAB-Map database
ros/teleop.sh                         # or drive by goal: the map grows as the cart moves
ros/map.sh save flat3_new             # freeze the grid into ros/maps/flat3_new.{yaml,pgm}
```

Goals work with no places book: `ros/goto.sh -1.0 0.3 90` drives to map coordinates, and a click
in Foxglove (Publish → `/goal_pose`, frame `map`) does the same without a tape. The goal server
takes the cart's pose from `map -> base_link` and accepts a goal while that edge is younger than
1 s. `ros/goto.sh mark NAME` names the spot the cart stands on: a labelled RTAB-Map node plus the
cart's offset from it, so the place rides the node when a loop closure bends the map.

Watch the map grow with the `ros/foxglove/pepin_slam.json` layout at `ws://localhost:8765`.

## Camera rigs

The head is a RIG chosen by name. `config/camera.json` holds the cameras as named blocks and one
key, `"active"`, says which the robot is wearing (`stereo` today):

| rig | what it is | what the laptop publishes |
| --- | --- | --- |
| `stereo` | the global-shutter stereo module: ONE 1600x600 side-by-side frame at 10 fps, 800x600 an eye, taped upside down | `/camera/image` + `/camera/camera_info` for the LEFT eye (rectified), plus `/camera/right/image` (mono8) + `/camera/right/camera_info`, all four under one stamp in the left eye's optical frame |
| `overview` | the mono AC310 webcam, 1280x720, checkerboard-calibrated | `/camera/image` (half size by default) + `/camera/camera_info` |

**Switching rigs takes two edits, one per half.** The board: `/etc/default/pepin-camera`
(`PEPIN_CAMERA_DEVICE`, a `/dev/v4l/by-id` path; `PEPIN_CAMERA_RESOLUTION`, `1600x600` or
`1280x720`; `PEPIN_CAMERA_ENCODER=HW` passes the camera's own MJPEG through), then `systemctl
restart pepin-camera`. The laptop: `"active"` in `config/camera.json`, or
`PEPIN_CAMERA=overview ros/laptop.sh vslam` for one container (`pepin.camera.active_camera`).
They must agree; a mismatch is said in every report line of the camera node.

**What stereo publishes.** The laptop decodes the side-by-side frame once, cuts it into the two
eyes as the robot sees them (`pepin.stereo.SideBySide`: an upside-down module's halves are turned
back and swapped) and, with `config/stereo_calibration.json`, rectifies both onto one pinhole with
the rows aligned. The right eye's `P[0,3] = -fx * baseline` carries the baseline on the wire.
Without that file only the left eye goes out, unrectified, and the report line says `NOT
RECTIFIED ... depth has no source`. The rectifier is rebuilt when the calibration file's mtime
moves, so a calibration finished while the robot runs is picked up without a restart.

### The two stereo matchers

The depth node's live `stereo_matcher` flag picks what turns the two eyes into a disparity. Both
engines are built at start, so an A/B is one `ros2 param set`.

| matcher | what it is | where it runs | ms a pair, 800x600 |
| --- | --- | --- | --- |
| `raft` (default) | RAFT-Stereo, `raftstereo-realtime.pth`, 7 iterations | natively on the laptop's GPU, over HTTP (`ros/depth_host.sh stereo` or the `depth` model service) | 89 |
| `sgbm` | OpenCV's semi-global block matcher, `pepin.stereo_depth.StereoMatcher` | in the depth node's own container | 17 |

Docker on macOS cannot see Metal, so RAFT runs where the GPU is, in the native process that also
serves the mono depth network (one port, one inference lock). The node sends two rectified grey
eyes raw and gets float16 disparity back, compressed losslessly when it asks (0.34 of the bytes).
On the parked robot's glossy parquet (2026-09-22, 8 pairs) RAFT left 94 airborne phantom blobs
where SGBM left 450, and 4.7x as many floor points within 5 cm of the plane.

**When the host does not answer**, the pair goes to SGBM and is counted; after three failures in
a row the host is left alone and probed again every 30 s. **The checkpoint** lives in `models/`
(gitignored; RAFT-Stereo's own `download_models.sh`); which file, how many iterations and which
device are in `config/camera.json`'s `net` block. The model's source is vendored under
`src/pepin/vendor/raft_stereo/` (MIT, princeton-vl/RAFT-Stereo at 6e93ed2).

## The head IMU and the visual-inertial odometry

The head ESP32 reads an MPU6050 glued to the stereo module; `pepin.head_server` (board,
`pepin-head`, TCP 3340) owns its serial port and maps its clock onto the board's. The base bridge
subscribes (`head_imu:=true`, `ros/feature.sh head_imu on`) and is the one publisher of what
follows from it:

- `/head/imu` (sensor_msgs/Imu, frame `head_imu`, the chip's axes), every sample under a 200 Hz cap,
  dated by the sample's data-ready edge less the chip's filter delay (the `imu_config` line);
  `head_imu_publish` mutes it live;
- `/mast/state` (JointState `mast_roll|pitch|yaw`, the sway's angles and rates, NaN while the neck
  moves) once the IMU's extrinsics are in `config/camera.json` (`stereo.head_imu`); the gaze gate
  drops a frame on it with `gate_sway_dps`/`gate_sway_deg` (both 0, off, as shipped);
- the sway composed into `base_link -> camera_link` only under `mast_sway` (off until its sign is
  checked, docs/head_imu_calibration.md section 7).

The minute line says it all: `head imu: 200.0 Hz published of 200.0 received, link up, server clock
ready spread 0.21 ms skew +4.5 ppm, 0 gaps > 1.5 periods ...; mast: armed, peak 0.080 deg, ...`.

OpenVINS runs on the laptop in its own container (`pepin-laptop:vio`, OpenVINS pinned to master
2025-11-30 plus PR #500, its own simulator as the build's gate) and enters the board's EKF through
the visual odometry's one slot:

```bash
uv run python ros/tools/vio_config.py          # OpenVINS's files from the repo's numbers (ros/maps/vio)
ros/laptop-build.sh vio                        # once: 16 min of build, 34 s of gate
ros/laptop.sh vio                              # OpenVINS in pepin-vio; vio logs | down | kick (at rest)
ros/laptop.sh vslam --vo-vio                   # the relay reads /ov_msckf/poseimu instead of rtabmap
ros/flags.sh set visual_odometry vo_input vio  # the same, live, with stereo_odometry left running
ros/flags.sh set visual_odometry vo_input stereo  # back to the eyes' odometry, live
```

The relay composes base_link's pose through TF (`head_imu <- base_link`: camera_stream's static
edge from the extrinsics, the neck chain from the board), gives it the per-step `vio` covariance
(`vio_step_fraction` of the step, floored at `vo_sigma_m`), counts OpenVINS's re-inits and withholds
a lost VIO (`vio_lost_*` knobs). Every sample first passes a plausibility guard on the composed
base velocity (`vio_max_speed_m_s` 1.0, `vio_wheel_diff_m_s` 0.5 against the wheels); after
`vio_restart_rejects` (20) refusals in a row with the wheels at rest for 2 s the relay asks
`/vio/restart` (pepin_bringup.vio_keeper in pepin-vio), at most once in 10 s.

OpenVINS recovers by itself (ros/patches/openvins-reset.patch, Dockerfile.vio's RESET=1): its
`/ov_msckf/reset` makes a new filter in the same process and seeds it warm on the first frame with
persistent tracks (the old filter's biases from its last healthy update, gravity from the
accelerometer turned by the gyro, the EKF's twist carried to the head IMU on `/vio/seed_twist`);
it says per frame on `/ov_msckf/health` what it sees, and its poses' frame is renamed at each
reset (`global`, `global_1`, ...), which the relay re-anchors on. The keeper judges that health
(`vio_watch`: a dark stretch of still frames, `vio_dark_*`; a runaway speed) and answers the
relay's `/vio/restart` the same way; `vio_recover restart` is the old kill-and-respawn, also the
fallback when the reset is not served. By hand:
`docker exec pepin-vio /pepin_entrypoint.sh ros2 service call /ov_msckf/reset std_srvs/srv/Trigger`;
the keeper's line: `docker logs pepin-vio 2>&1 | grep -a "vio keeper: " | tail -1`.
Its report line (`docker logs pepin-vslam 2>&1 | grep -a "vo: " | tail -1`) carries the samples in
and out, the guard, the stamp-to-receipt latency, whether `ekf_filter_node` subscribes `/vo`, and
the EKF's own odom -> base_link. Calibration: docs/head_imu_calibration.md. The offline A/B: the
"Replay" section.

Rest and start are OpenVINS's own. Its zero-velocity update is on all day: a camera interval whose
IMU reads rest under the current biases (gyro and accel, chi2 at 95 %) with OpenVINS's own speed
under 0.05 m/s is not propagated, and the biases and the tilt are updated from gravity instead, so
hours on the charger neither move the pose nor walk the biases. The IMU test is what keeps a
turning neck out (a still head passes, a neck turning faster than ~0.8 deg/s never); OpenVINS's
still-picture override is off (its update has no velocity row and froze a wrong speed at rest).
With the ZUPT on, the static initialisation starts from ~1 s of stillness, no jerk or head nod
needed; OpenVINS publishes its first pose at the first visual update, i.e. ~0.5 s into the first
motion after a start (its `initialized()` waits for one). The
start options are vio_config.py's, passed at `ros/laptop.sh vio` (OpenVINS reads its config only
when it starts; the guard's restarts keep it):

```bash
PEPIN_VIO_CONFIG_ARGS=--dyn-init ros/laptop.sh vio  # may also initialise in motion (still: static)
PEPIN_VIO_CONFIG_ARGS=--no-zupt ros/laptop.sh vio   # the first design: no ZUPT, init on a jerk
```

## Camera calibration

**The stereo head** (writes `config/stereo_calibration.json`):

```bash
ros/calibrate.sh --print                              # data/checkerboard.pdf: print at 100 %
ros/calibrate.sh stereo                               # a window with both eyes, ~25 pairs
ros/calibrate.sh stereo --no-window                   # the same over ssh
ros/calibrate.sh stereo --square 0.0245               # the square really on the paper, metres
ros/calibrate.sh stereo --images DIR                  # re-fit a session saved under data/stereo_calib/
ros/calibrate.sh stereo --images DIR --refit-rotation # after a remount: only the eyes' rotation
```

Measure one printed square with a ruler: it is the only length in the whole procedure. No keys to
press: hold the board where the hint says, keep it still, and the pair is taken on its own
countdown. Every accepted pair is saved before anything is solved, so a bad run is re-fitted, not
reshot. A result is refused, with what to reshoot, when its stereo RMS, baseline, rectified
epipolar error or board depth (stereo against left-eye PnP, 0.99..1.01) is bad.
`--refit-rotation` keeps the file's lenses and baseline and refits only the rotation between the
eyes; after the 2026-10-01 remount it brought the board's stereo depth from 0.94 to 1.00 of its
PnP depth with the 61.0 mm baseline kept. The maths lives in `pepin.stereo_calibration`.

**The mono camera** (`overview` rig, writes the `intrinsics` block of `config/camera.json`):
`ros/calibrate.sh` with the same printed board, `--no-window` over ssh, `--images DIR` to re-fit
from `data/camera_calib/<timestamp>/`. It writes only under 0.5 px RMS with the frame covered;
`calibrated: false` switches a bad calibration off and the nominal `hfov_deg` comes back. Restart
the camera node to publish new optics (`ros/laptop.sh kick camera_stream`).

The camera's mount (height, tilt) is `config/camera.json`'s `mount` block and the neck's
`config/neck.json`, measured by tape and encoder, not fitted.

### Camera only

`ros/laptop.sh vslam --camera-only` subscribes to no lidar at all: RTAB-Map's grid is built from
the camera's depth (`Grid/Sensor 1`), ray-traced so the floor it flew over becomes free space.
With the lidar present the 2D grid is the scan's (`Grid/Sensor 0`). Other `vslam` options:
`--fresh` (an empty database), `--no-vo` (no visual odometry), `--vo-depth` (the visual odometry
on the picture and the depth instead of the two eyes), `--fixed-head` (the static camera edge,
for a rig without neck servos; otherwise the board's base bridge publishes it from the neck's
encoders at every state line).

### The depth law

`depth_stream` fits one affine law in inverse depth over the lidar's pairs. Under the stereo head
it only watches (`law_watch`): the depth goes out as measured, and `a 1.00 b +0.000` says the head
is as calibrated. Under the mono network it corrects. The mono network's range law, frame law and
their rulers are on the tag `alt/mono-depth-2026-09-21`.

## The obstacle volume

`pepin_bringup.depth_fusion` fuses the depth frames, the lidar's revolutions and the three ToF
fans into one TSDF (`pepin.tsdf`) on the laptop. It is LOCAL obstacle memory in `odom`: a rolling
window that slides with the cart, placed by `odom -> base_link` and never through `map -> odom`,
born empty at every start. Nothing localises against it. The lidar writes its own layer (rays
carve free space, returns mark a surface), which the camera's depth may not repaint; the ToF fans
go in as rays from their own frames (`tof_rays`); a parked cart's repeated view counts once
(`pepin.worldmap.ViewGate`); and a pixel with no depth carves free space at a lower weight.

The costmaps' camera layer MARKS from the volume's surface sliced around the cart
(`/depth_marks`, `pepin.volume_scan`, at `marks_hz`) and CLEARS from the frame's own fan
(`/depth_scan`): a single stereo frame is not evidence that something is there. The node's
docstring has the reasons and the measurements. The room-sized map-frame volume is on the tag
`alt/volume-map-2026-10-02`.

## The marks audit: who painted the lethal cells, live

`pepin_bringup.marks_audit` answers, once per `/local_costmap/costmap`: of the lethal cells within
`radius_m` of the cart, how many are lidar-backed (a `/scan` return within `match_cells`),
camera-only (only a `/depth_marks` beam) and unexplained. A camera-only cell is a phantom or a
real thing above the lidar's plane, and the node does not guess which. Out go `/marks_audit` (JSON)
and `/marks_audit/phantoms` (a red cloud), and a report line every 10 s: `marks audit: lethal 214
(lidar 97, camera-only 106, unexplained 11), nearest camera-only 0.42 m`. Its `marks_audit` flag
switches it off live.

## Gaze: the head's one owner

`pepin_bringup.gaze` runs beside Nav2 in `pepin-macnav` and is the only thing that moves the
neck (`pepin.gaze`): consumers ask, it decides by band (operator, navigation, the person's word,
sensor checks, driving, idle), TTL and preemption, and the head falls home by itself when nothing
holds it. It speaks to the board's base server on :3336: `neck_target` (renewed every
`target_renew_s`, inside config/neck.json's `motion.lease_s`, so a lost laptop sends the head home)
to a base server whose state lines carry the neck's encoders, `neck_goto`/`neck_home` at rest to
an older one (picked by itself). Its doors: `/gaze/stall_look` (std_srvs/Trigger) for the
behaviour tree, and JSON over HTTP on `127.0.0.1:3339` (`pepin.gaze_link`: `POST /look`,
`POST /renew`, `GET /state`) for the LLM tools. `/gaze/state` (JSON, 10 Hz and on every change)
carries the phase and `since` (a move's write, a settled head's settling reading), the pan and
tilt, the holder and the blind interval (`blind_from`..`blind_until`, the board's clock): the
gaze gate drops the frames inside it ("gaze_gate" in the flags below); both recorders tape it
(`gaze` rows) and every stall look (`/gaze/stall`, `stall` rows).

**The stall look** (`stall_look`, off as shipped): when FollowPath fails, the tree's `StallLook`
asks first. The hull swept along the plan's first `stall_ahead_m` gives the lethal cells that
block; the ones the lidar does not back are looked at — the head saccades to the centroid of what
the volume holds over them (`depth_fusion`'s `/fusion/column`), holds for `frames` fused frames
(`/fusion/frame`), comes home, and the clears and the replan read the volume as it now is. One
log line says it all: `stall look: 2 lethal cells under the hull in the first 1.0 m (lidar 0,
camera-only 2, unexplained 0), the first at 0.30 m; centroid (0.52, 0.04, 0.31) in odom; looked
pan +4 tilt 58 deg: done, 3 frames in 1480 ms; candidates 2/2 cells, weight 61 -> 0/2 cells,
weight 0: carved; lidar-backed cells carved 0; home done in 900 ms`. A blocker under the bumper
(deeper than `stall_max_depression_deg`) answers FAILURE: the tree backs up 0.10 m and asks once
more. An image without `pepin_gaze_bt` (`ros/laptop-build.sh gaze`) runs the tree with
`StallLook` cut out; `nav.launch.py` says which in the log.

```bash
ros/flags.sh set gaze stall_look true     # the look at the next stall (false: the tree as before)
curl -s 127.0.0.1:3339/state              # the head: phase, angles, holder, blind interval
ros/laptop.sh nav logs | grep "stall look"
```

`path_gaze` and `reverse_gaze` (off) need the `neck_target` base server; the tools' `look`,
`look_around` and `find` are requests of the person's band, held ten seconds (`see` renews); a
base server that moves the neck only at rest has them refused during a drive. At a drive's start
every request but the operator's is let go, and a head found more than `drive_home_tol_deg` off
home (a jog, a hand) is sent home.

## Camera grid A/B (2026-09-24)

An alternative to the `/depth_marks` fan, shipped off: `depth_fusion`'s `grid_out` publishes the
volume's current occupied columns as `/camera_grid` (and `/camera_grid_map` on `/map`'s lattice),
and a `camera_grid_layer` (a `StaticLayer`, disabled in both costmaps) draws only the latest
grid. `ros/camera_grid.sh on|status|off` switches the flag and both costmaps' layers in the safe
order (a layer is enabled only after its latched grid is read back). The node's docstring and
`pepin.camera_grid` have the guards.

## Feature flags

Every behaviour that can be switched is a live parameter of the node that owns it, declared
once in that node's `FLAGS` table (`pepin.flags`, declared to ROS by
`pepin_bringup.node_kit.Switches`) with its kind, default and four texts, checked by kind on
every change, and printed in the node's report line. A change lives until the node restarts;
a default changes in the table. This section is generated by `ros/tools/flags_doc.py` (the
pre-push hook keeps it current).

**How to read the flags.** A flag is this robot's A/B switch: a new behaviour ships with the
old one still reachable, so a regression in the field is turned off on the spot instead of
reverted, and two runs a minute apart can be compared without a restart. `ros/flags.sh list
[NODE]` shows every flag with the value the running node holds, `ros/flags.sh get NODE FLAG`
reads one, `ros/flags.sh set NODE FLAG VALUE` changes one live (a value the flag refuses is
refused with the reason, before any host is touched), and `ros/flags.sh flag NODE FLAG` prints
the whole entry from the node's table: what it does, why the default is what it is (the
measurement and the file it was measured in, or `default by design, unmeasured`), when to turn
it on, when to turn it off. Switches live here; the numbers that are not switches (the lidar's
mount, the camera's intrinsics) live in `config/*.json` and are read at start; the numbers a
person tunes live are the config knobs below.

**Muting a sensor live.** A sensor is switched off where it is *published*, by a flag of the node
that publishes it, so the message simply stops and every consumer meets what a dead sensor looks
like — silence, an EKF's `sensor_timeout`, a transform that stops moving — with nothing
restarted and no other live flag lost. `ros/sensor.sh mute imu|odom|vo|lidar` and
`ros/sensor.sh unmute ...` do it in one command and print what to expect; `ros/sensor.sh status`
lists each sensor's mute state. The flags behind them: `base_bridge` `imu_publish` and
`odom_publish` (the board's bridge; `odom_publish` takes the `odom -> base_link` transform with
it, because a transform still broadcast from a silent `/odom` is a state no sensor failure
produces), and `visual_odometry` `vo_publish`. The lidar has none: our own node in its chain is
`scan_filter` (laser_filters, external), and a relay on the board just to drop `/scan` would
cost the board a copy of every scan — so `mute lidar` is the consumer set instead (`lidar_layer`
off on both costmaps, which is `ros/sensor.sh lidar off`), and `ros/sensor.sh lidar off --hard`
is the real absence of a scan. `ros/feature.sh imu off` restarts the board stack instead: a
minute, and every live flag on it back to its default.

| node | flag | kind | default | live | description |
| --- | --- | --- | --- | --- | --- |
| `bag_recorder` | `goal_bag` | choice: record, ring | ring | yes | how a goal's bag is made. record: one `ros2 bag record` of BAG_TOPICS per goal, started on the goal's word and closed at its end (its first message 0.3-0.4 s after the word). ring: one `ros2 bag record` of RING_TOPICS (BAG_TOPICS and OpenVINS's /ov_msckf/poseimu and /ov_msckf/odomimu) runs as long as this node, a file a minute under ring_dir pruned by the knobs ring_keep_h, ring_keep_gb and ring_floor_gb, and a goal's bag is cut out of it, [goal - preroll_s, end + tail_s], into the same /maps/rec/<run>/ (pepin.ring, pepin.bag_slice), whole once the ring is written past the window (tail_s and ~1 s after the goal). A change starts or stops the ring at once; a run keeps the way it began |
| `base_bridge` | `imu_publish` | bool | on | yes | the MPU6050's readings leave the bridge as /imu/data_raw, where the EKF fuses index 11 (the yaw rate) and nothing else; off, the chip is still read and its bias still estimated, but no message is published |
| `base_bridge` | `odom_publish` | bool | on | yes | the base server's state line leaves the bridge as /odom and, while publish_tf is on, as the odom -> base_link transform; off, the wheels are still read and still commanded, and both go silent together — a transform still broadcast from a silent /odom is a state no sensor failure produces |
| `base_bridge` | `odom_stamp` | choice: encoder, arrival | encoder | yes | what /odom and odom -> base_link are dated by: the state line's encoder read carried onto the ROS clock (`encoder`, the stamp /neck/state of the same line carries) or the moment the line reached the bridge (`arrival`); a line older than 0.5 s is dated on arrival either way |
| `base_bridge` | `odom_covariance` | choice: law, constant | law | yes | what /odom's twist covariance says about vx and vyaw: `law` sizes both from the wheels' MEASURED twist while they move (pepin.wheel_noise, config/base.json odometry_noise: over 1 s sigma_v = 0.034 \|w\| + 0.026 m/s, sigma_w = 0.20 \|w\| + 0.038 rad/s, 50 sigma^2 per 50 Hz sample) and for 2 s after their last moving sample, and keeps 0.001 / 0.01 at rest past that; `constant` is 0.001 / 0.01 on every sample, as before 2026-10-06 |
| `base_bridge` | `head_imu_publish` | bool | on | yes | the head IMU's samples (head_server's TCP 3340 stream, under the launch's head_imu:=true) leave the bridge as /head/imu in the chip's axes, dated by the sample's own moment on the board's clock; off, the link, the counters and the mast filter keep running and nothing is published |
| `base_bridge` | `mast_sway` | bool | off | yes | the mast's sway, from the head gyro minus the base gyro's yaw and the neck's joints (mast.hpp), is composed INTO base_link -> camera_link as a rotation about the mast's hinge, so every consumer of that edge gets the corrected camera; off, the edge is the neck's alone, exactly as before, and /mast/state publishes either way |
| `camera_stream` | `camera_stamp` | choice: send, grab | grab | yes | which board moment a frame is stamped with: `send`, ustreamer's X-Timestamp (the write to this client); `grab`, the V4L2 capture moved onto the same realtime clock (grab + X-Timestamp - send, pepin.mjpeg.capture_time; needs ?extra_headers=1 on the stream URL and falls back to send per frame without it, counted in the report line) |
| `camera_stream` | `undistort` | bool | off | yes | the published picture is rectified with the checkerboard calibration (config/camera.json's intrinsics) and its camera_info then says no distortion; a no-op while the camera is uncalibrated, since there is nothing to undo. Rectifying crops to the largest all-valid rectangle, so the field of view narrows. THE MONO RIG's flag: a stereo head is rectified by its own stereo calibration (both eyes onto one pinhole with the rows aligned, which is what a disparity means at all), so the node refuses this one there rather than straighten a picture twice |
| `camera_stream` | `fold_mask` | bool | on | yes | stereo: rectified pixels past a fold of the calibration's undistortion map (the lens corners the board never reached) go out black, as no data, and the depth there is cut; off publishes the mirrored corners as before |
| `camera_stream` | `static_camera_tf` | bool | off | at start | base_link -> camera_link is broadcast from here (ros/laptop.sh vslam --fixed-head); off, the board's base bridge publishes that edge live from the neck's encoders, because two publishers of one edge fight |
| `contact_scan` | `contact_scan` | bool | on | yes | the contact line is published; off, the node is a subscriber that costs nothing — the costmap's own contact_layer.enabled is the other end of the same demo switch, and either one alone takes the camera's floor line out |
| `contact_scan` | `shadow` | bool | on | yes | the last floor pixel on a face stands a band's width UP that face, so its ray lands past the foot: on, that width is taken back off the range (pepin.contact.band_shadow); off is the raw boundary ray |
| `contact_scan` | `imu_lean` | bool | on | yes | the floor plane leans with the gyro as well as the accelerometer (pepin.lean: the lean of a wheel climbing a threshold is followed within a sample instead of being gated away as a push); off, the accelerometer alone, as it always has been |
| `depth_fusion` | `enabled` | bool | on | yes | frames are fused into the model; off, they are dropped |
| `depth_fusion` | `tof_rays` | bool | on | yes | the three ToF fans (/tof/<name>/scan) are written into the volume as rays from their own frames, through the camera's integrator: a return marks a surface at its range and carves the ray free up to it, +inf carves free out to the fan's trusted range (its range_max, pepin.tof_horizon.trusted_max_range), every fan at weight pepin.tof_rays.TOF_WEIGHT. Both costmaps then read what the whiskers saw out of the volume (/depth_marks, /camera_grid) like the camera's; off, the fans touch the volume at all and feed only the local costmap's own layers, as before 2026-10-01 |
| `depth_fusion` | `imu_lean` | bool | on | yes | the cart's lean (pepin.lean, from /imu/data_raw) is followed with the gyro as well as the accelerometer, and a frame is placed with the lean at its stamp composed on base_link before the planar odometry instead of as if the cart stood level; the lidar's scan follows the same switch — its beams are walked as the 3D rays the leaning body sends them along, and lean_gate_deg drops the scans taken too far from level |
| `depth_fusion` | `marks_clear` | bool | off | yes | the fan also says where the volume is KNOWN OPEN: a second, clearing-only scan on /depth_free carrying, per bearing, the range of the last column the volume has observed FREE before the first column it has not (pepin.volume_scan.free_ranges). A bearing the volume cannot vouch for stays NaN, which clears nothing. Off, the topic is silent and the camera layer clears from the single frame alone, as it has since 2026-09-21 |
| `depth_fusion` | `grid_out` | bool | off | yes | publish the volume's current occupied columns (the /depth_marks rule) as grids the costmaps' camera_grid_layer only draws: /camera_grid, a square about the cart in the volume's frame, and /camera_grid_map with its _updates on the lattice of grid_map_topic; off, all three are silent |
| `depth_fusion` | `lidar_layer` | bool | on | yes | /scan is integrated into the volume at the lidar's plane (rays carve free space, returns mark a surface); off, the volume is the camera's alone, as it was |
| `depth_fusion` | `self_filter` | bool | on | yes | the cart's own body (config/body.json: boxes in base_link grown by margin_m) is cut out of every camera frame: a pixel whose depth lies on or past its ray's entry into the body measures no room, and no voxel on or past that entry is written, measured or carved (pepin.body, pepin.tsdf.Tsdf.integrate's clip); off, every ray is written whole, as before |
| `depth_fusion` | `arm_filter` | bool | on | yes | the robot's own arm (config/arm.json: the SO-101's links as boxes posed by its joints through the vendored URDF, grown by margin_m) is cut out of every camera frame and whisker fan as the body is, and every voxel inside a grown link is forgotten after each integration, whoever painted it (pepin.arm, pepin.worldmap.WorldMap.forget); off, the arm is painted like the room, as before |
| `depth_stream` | `edge_filter` | bool | on | yes | flying pixels at object edges are dropped from the published depth and the scan; the law's beam pairs skip them regardless |
| `depth_stream` | `lidar_anchor` | bool | on | yes | the lidar's returns pair with the network's depth and fit the law; off, the last law is held (the failure mode of a lidar that stops) — with no law yet nothing is published until it is back on |
| `depth_stream` | `affine_law` | bool | on | yes | the network's depth through 1 / z = a / D + b, fitted on the pooled pairs; off, the raw network's depth goes out unwithheld |
| `depth_stream` | `floor_anchor` | bool | on | yes | pixels within centimetres of the floor plane snap to it in the published image (the scan is built before it); the plane leans with the cart, from the IMU's up vector |
| `depth_stream` | `depth_backend` | choice: remote, local, auto | local (env PEPIN_DEPTH_BACKEND) | yes | where the network runs: local (the CPU model in this container), remote (the laptop's GPU service, ros/depth_host.sh), auto (the service while it answers, the CPU model while it does not) |
| `depth_stream` | `stereo_matcher` | choice: sgbm, raft | raft (env PEPIN_STEREO_MATCHER) | yes | which engine turns the two eyes into a disparity: sgbm (OpenCV's semi-global block matcher, in this container) or raft (RAFT-Stereo on the laptop's GPU through the same host the mono network uses, ros/depth_host.sh stereo). Both are built at start and this picks which one answers the next pair, so an A/B needs no restart; a pair the host cannot answer falls to sgbm and the report line counts it. Under depth_source: network it does nothing |
| `depth_stream` | `law_watch` | bool | off | yes | the affine law is fitted on the lidar's pairs and printed, and the depth is published exactly as the source measured it; no frame waits for a law |
| `depth_stream` | `imu_lean` | bool | on | yes | the cart's lean (pepin.lean, from /imu/data_raw) is followed with the gyro as well as the accelerometer and carried into the scan's carry and the camera's place in the map; off, the floor plane leans with the accelerometer alone, as it always has, and nothing else is leaned |
| `depth_stream` | `gaze_gate` | bool | on | yes | frames whose exposure window (stamp +- gate_exposure_s) overlaps a head saccade (/gaze/state's blind intervals, + gate_settle_s after settling) or holds a body yaw faster than gate_yaw_dps (/imu/data_raw; 0 is off) are dropped here and counted; off, every frame passes as before |
| `depth_stream` | `gate_dark` | bool | on | yes | a frame whose mean luma (/camera/brightness, camera_stream's per frame 1/8 subsample of the whole stereo frame) is under gate_dark_floor grey is dropped here and counted as dark, and so is every frame after it until one reaches gate_dark_floor + gate_dark_hyst: it writes nothing into the volume, the clearing fan or RTAB-Map, and a look's still frames do not advance on it; only under gaze_gate. The visual odometry never carries it. Off, the brightness is not consulted and every frame the other rules pass goes on as before |
| `gaze` | `stall_look` | bool | on | yes | when the controller fails, the behaviour tree's AskGaze asks for a look at the camera-only and unexplained marks blocking the hull in the plan's first stall_ahead_m: the head saccades to their voxel centroid, holds for 'frames' still frames so the volume carves a phantom or confirms a thing, comes home, and the tree clears and replans; off, AskGaze answers at once and the tree runs as before |
| `gaze` | `path_gaze` | bool | on | yes | while a drive runs, the head looks along the plan path_lookahead_s ahead (pan clamped to path_pan_clamp_deg, a dead-band of path_deadband_deg); only with a base server that moves the neck while driving (neck_target), otherwise idle |
| `gaze` | `reverse_gaze` | bool | on | yes | a reverse leg the plan announces (reverse_min_m) or a recovery drives (reverse_recovery) turns the head reverse_pan_deg toward the side the rear swings to at its first reversing command, any other after reverse_min_s, any with the rear tight at once; with reverse_hold_s the look is let go at the first forward command; only with a base server that moves the neck while driving |
| `gaze` | `face_events` | bool | on | yes | the stall look's moments on the head's face (pepin.face_events.StallFace, through the board's head server on PEPIN_HOST:3340): surprised as the head turns to the blocker, a small grin when the look carved a phantom, worried when it confirmed a thing (config/face.json's events) |
| `goal_server` | `controller` | choice: mppi, rpp, rpp_shim, graceful, dwb, shim_mppi | shim_mppi | yes | what follows the plan: mppi is Nav2's MPPI controller for every planner, held to the mark's heading by the yaw-checking goal checker; rpp is each planner's own Regulated Pure Pursuit from PLANNERS, ending on position alone as before 2026-09-23; rpp_shim is the reversing RPP inside Nav2's RotationShimController, which turns the cart to the mark's heading in place once it is inside the goal tolerance; graceful and dwb are Nav2's Graceful and DWB controllers, for an A/B against mppi; shim_mppi drives each goal on rpp_shim and hands it to mppi for the rest of that goal once the cart is within the knob park_distance_m of the goal (PARKERS) (ros/goto.sh controller NAME). Published latched on controller_selector and goal_checker_selector; the behaviour tree plans at once on a controller change and hands FollowPath the new controller together with that plan |
| `goal_server` | `start_needs_placement` | bool | on | yes | a goal or a mark waits for the laptop's word on /localization/placement (pepin_bringup.rtabmap_frame, latched) that this start of RTAB-Map is PLACED — a node of the loaded map recognised, or an operator's seed — and nothing heard is refused like not placed. Off, a fresh map -> base_link is enough, as before 2026-09-23 |
| `goal_server` | `face_events` | bool | on | yes | the drive's moments on the head's face (pepin.face_events, through the board's head server on PEPIN_HOST:3340): focused while a goal runs, struggling for a moment at each new Nav2 recovery, happy on arrival, sad on an abort or a refusal, a flat line on a cancel (config/face.json's events say what each looks like); and a brain lease every 2 s, so the face falls asleep when this node or the WiFi is gone |
| `marks_audit` | `marks_audit` | bool | on | yes | the audit runs; off, the node keeps its subscriptions and computes, publishes and reports nothing |
| `marks_audit` | `inscribed_counts` | bool | off | yes | the inflation's 99 band (costmap 253, INSCRIBED_INFLATED_OBSTACLE) is judged as a mark too; off, only the 100s a sensor actually wrote |
| `rtabmap_frame` | `visual_features` | choice: orb, xfeat | xfeat | yes | which features RTAB-Map's VISUAL registration (Reg/Strategy 0, the camera-only strategy, and the visual half of 2) matches when it checks a node the words recognised. xfeat: XFeat keypoints matched by LighterGlue, re-extracted from both nodes' stored pictures at loop-closure time (Vis/FeatureType 15, Vis/CorNNType 6, RGBD/LoopClosureReextractFeatures true); the database is only read. orb: the database's own GFTT/ORB words, the launch table's values. Sent with the strategy and changed live; under ICP alone, and unless RTAB-Map is certainly localising (told so and answered, no switch to mapping waiting), the set is always orb, and a switch to mapping waits until orb's set is in force. xfeat needs the pepin-laptop:xfeat image (/opt/xfeat/rtabmap_xfeat.py); in another image this node sends orb and the report line says why |
| `rtabmap_frame` | `visual_confirm` | choice: rtabmap, aggressive, single | aggressive | yes | how RTAB-Map CONFIRMS a localisation while the camera registers alone: RTAB-Map 0.22 delays a first good localisation into its odometry cache and accepts it only with a second one inside RGBD/MaxOdomCacheSize updates, and that second try has to reach Rtabmap/LoopThr. rtabmap: its stock 0.11 and 10. aggressive: Rtabmap/LoopThr 0.05, the threshold the first try already used, so the second comes on the next update; the confirmation stays. single: RGBD/MaxOdomCacheSize 0, the first good localisation is accepted. Sent with the visual strategy while the database localises and changed live; under ICP and while it maps, always rtabmap |
| `rtabmap_frame` | `visual_proximity` | bool | on | yes | whether a localised camera also registers every update against the database nodes near its pose (RGBD/ProximityBySpace), each an XFeat re-extraction and a LighterGlue match; off, only the words' own hypothesis is registered, one at most per update. Sent with the visual strategy while the database localises and changed live; under ICP and while it maps, always on (the lidar's proximity links are cheap and most of the graph's) |
| `rtabmap_frame` | `registration_backend` | choice: service, local, auto | auto (env PEPIN_REGISTRATION_BACKEND) | yes | where RTAB-Map's XFeat keypoints and LighterGlue matches are computed (the xfeat visual features): service — the localisation service on the laptop's GPU (pepin.localization_service, PEPIN_MODELS_URL), no features when it does not answer; local — in RTAB-Map's own process on the Docker VM's CPU, as before 2026-09-24; auto — the service, and a call it does not answer computed locally. Written to /tmp/pepin/registration.json, which RTAB-Map's adapters read on every call (one stat): live, no restart of RTAB-Map |
| `rtabmap_frame` | `place_recognition` | choice: words, descriptor | descriptor | yes | how RTAB-Map finds WHICH database node a picture is (its likelihood, before any registration): words — the ORB bag of words' TF-IDF (Kp/TfIdfLikelihoodUsed true, Rtabmap/VirtualPlaceLikelihoodRatio 0, RTAB-Map's defaults); descriptor — the dot product of the nodes' learned place descriptors as z-scores (false and 1; rtabmap Memory::computeLikelihood -> Signature::compareTo, Rtabmap::adjustLikelihood), which sensor_pack attaches to every snapshot. descriptor is sent ONLY when it cannot abort RTAB-Map: its core carries ros/patches/rtabmap-keep-global-descriptors.patch (the marker /opt/rtabmap_patches/keep-global-descriptors), the snapshots carry one each (/sensor_pack/place) and the database's census at this start (PEPIN_PLACE_CENSUS, taken by the launch before RTAB-Map opens the file) says every node carries exactly one of the same length — and only while the camera snapshots are described (descriptor_null_share); otherwise the words, and the report line says why. Live |
| `rtabmap_frame` | `start_needs_placement` | bool | on | yes | what goes out on /localization/placement (latched) says this start of RTAB-Map is PLACED only once an update has recognised a node of the database it loaded, or an operator's seed (/rtabmap/initialpose) has been heard since its first update — or it loaded an empty database, whose start pose is the map's origin. The goal server (pepin_bringup.goal_server) refuses a goal until then, saying to seed or to let the camera see a mapped place. Off, every start counts as placed: RTAB-Map's pose is taken as it is |
| `sensor_pack` | `sensor_pack` | bool | on | yes | snapshots are published; off, the node subscribes and counts and RTAB-Map is fed nothing at all |
| `sensor_pack` | `sources` | list of: camera, lidar | camera,lidar | yes | which sensors may enter a snapshot: the live A/B for camera-only and lidar-only mapping, with no restart and without muting a publisher |
| `sensor_pack` | `global_descriptor` | choice: auto, on, off | auto (env PEPIN_GLOBAL_DESCRIPTOR) | at start | whether every snapshot carries exactly one rtabmap_msgs/GlobalDescriptor (type 1): the place vector of its picture (place_descriptor), or the null descriptor (zeros) when it has no picture or no vector came in place_timeout_s; said latched on /sensor_pack/place. auto: exactly when this image's RTAB-Map keeps a node's descriptor across the reload of its data (the marker /opt/rtabmap_patches/keep-global-descriptors, ros/xfeat/patch_rtabmap.sh); on: always; off: never, the snapshots of before 2026-09-24 |
| `sensor_pack` | `place_descriptor` | bool | on | yes | a camera snapshot's descriptor is the localisation service's /place vector of its picture (pepin.localization_service, BoQ on the laptop's GPU); off, every snapshot carries the null descriptor and the service is not asked |
| `sensor_pack` | `gaze_gate` | bool | on | yes | frames whose exposure window (stamp +- gate_exposure_s) overlaps a head saccade (/gaze/state's blind intervals, + gate_settle_s after settling) or holds a body yaw faster than gate_yaw_dps (/imu/data_raw; 0 is off) are dropped here and counted; off, every frame passes as before |
| `sensor_pack` | `gate_dark` | bool | on | yes | a frame whose mean luma (/camera/brightness, camera_stream's per frame 1/8 subsample of the whole stereo frame) is under gate_dark_floor grey is dropped here and counted as dark, and so is every frame after it until one reaches gate_dark_floor + gate_dark_hyst: it writes nothing into the volume, the clearing fan or RTAB-Map, and a look's still frames do not advance on it; only under gaze_gate. The visual odometry never carries it. Off, the brightness is not consulted and every frame the other rules pass goes on as before |
| `vio_feed` | `rate_gate` | bool | on | yes | a pair whose exposure window (gate_exposure_s, gate_stamp_end) holds a /head/imu sample turning faster than head_rate_dps (the gyro's norm) is held back from OpenVINS, once OpenVINS has initialised; off, every pair passes (a plain relay) |
| `vio_keeper` | `vio_recover` | choice: reset, restart, off | reset | yes | how a lost OpenVINS (the keeper's own verdict, or the relay's on /vio/restart) is brought back: `reset`, a new filter in the same process (/ov_msckf/reset, openvins-reset.patch) seeded warm on the first frame with a picture; `restart`, the launch's SIGINT to the process (respawned in 2 s, initialises after ~1 s at rest); `off`, logged only |
| `vio_keeper` | `vio_watch` | bool | on | yes | the keeper's own failure rules on /ov_msckf/health call for a recovery: a velocity disagreeing with the EKF's at the IMU (vio_disagree_m_s for vio_disagree_s), a dark stretch (vio_dark_tracks, vio_dark_s, swings excused by vio_swing_rad_s) and a runaway speed (vio_max_speed_m_s for vio_speed_s); off, only the relay's /vio/restart does |
| `vio_keeper` | `vio_dark_at_rest` | bool | off | yes | the dark rule also judges frames OpenVINS held with its zero-velocity update (the cart at rest): a still cart staring at a blank wall is then lost too |
| `vio_keeper` | `vio_restart_fallback` | bool | on | yes | when /ov_msckf/reset is not served or does not answer within 2 s, the process is restarted instead (the launch's SIGINT) |
| `vio_keeper` | `seed_warm` | bool | on | yes | a reset's new filter is seeded warm (kept biases, gravity from the accelerometer, /vio/seed_twist's velocity) on the first frame with a picture; off, it waits for OpenVINS's own initialiser (static at rest). Pushed to OpenVINS's parameters |
| `vio_keeper` | `seed_dyn_init` | bool | off | yes | a reset's new filter may also initialise dynamically (in motion) when the warm seed cannot (OpenVINS's init_dyn_use for the reset only; the config's is the cold start's). Pushed to OpenVINS's parameters |
| `visual_odometry` | `vo_input` | choice: stereo, depth, vio | vio | yes | what the relay reads: `stereo` or `depth`, rtabmap's /vo/raw from the odometry node vslam.launch.py started (its own vo_input argument); `vio`, OpenVINS's /ov_msckf/poseimu (ros/laptop.sh vio) composed into base_link through the neck's TF, every sample past the plausibility guard (vio_max_speed_m_s, vio_wheel_diff_m_s) and OpenVINS restarted at rest after vio_restart_rejects rejections in a row. Live between the launched rtabmap input and vio: the gate and the track's anchor restart, the published track carries on |
| `visual_odometry` | `vo_publish` | bool | on | yes | the gated visual odometry leaves this laptop as /vo, where the board's EKF fuses it as a third input beside the wheels and the gyro; off, the node still measures and reports and the EKF is exactly what it was without it |
| `visual_odometry` | `vo_covariance` | choice: dynamic, constant, rtabmap, vio | dynamic | yes | whose covariance rides on the published pose: `dynamic`, the registration's own sigma and the depth scale's share of the step just taken added in quadrature (pepin.visual_odometry.scaled_covariance); the documented constant (vo_sigma_m, vo_yaw_sigma_deg); the one rtabmap's registration computed, untouched; or `vio`, a per-step model for the visual-inertial input (vio_step_fraction of the step, floored at vo_sigma_m = one wheel sample; vo_input vio pins it, since OpenVINS's own covariance is the MARGINAL of an unobservable global pose and only grows) |
| `visual_odometry` | `vo_output` | choice: pose, twist | twist | yes | how the VIO reaches the board's EKF: `pose`, the admitted track on /vo with the vo_covariance model, differenced by the EKF (odom1); `twist`, the same admitted step as a body velocity (vx, vy, vyaw) on /vo_twist (twist0) with OpenVINS's own velocity covariance from /ov_msckf/odomimu at the pose's stamp turned into base_link's axes, no floor, while /vo carries the track weightless (1e6). A sample whose covariance is missing or not one is withheld and counted. Live; vo_input vio only (under rtabmap's inputs it reads as `pose`) |
| `visual_odometry` | `vio_twist_source` | choice: step, imu | step | yes | vo_output twist only: where the body velocity comes from. `step`, the SE(2) log of two composed poseimu poses (one per camera frame, 9.5-11.6 Hz, the neck subtracted by the composition); `imu`, every odomimu sample vio_publish_hz takes (50 Hz): OpenVINS's IMU velocity STATE and bias-corrected gyro carried into base_link through the neck chain (its rate differenced from TF over the step, pepin.visual_odometry.imu_base_twist), its covariance odomimu's own with the lever arm, processed 0.1 s behind the newest sample so the TF lookup never waits. Under lost both send the yaw rate alone |
| `visual_odometry` | `vio_guard` | bool | off | yes | the plausibility guard's wheel rule: a composed base velocity farther than vio_wheel_diff_m_s from the wheels' is refused; off, the guard judges the speed alone (vio_max_speed_m_s, always on), so the VIO may disagree with slipping wheels |
| `visual_odometry` | `vio_lost_rule` | choice: used, tracked | used | yes | which features the lost rule counts (fewer than vio_min_features for vio_lost_s while the base moves, never at rest): `used`, the features OpenVINS's last update used, /ov_msckf/points_msckf + points_slam; `tracked`, the tracker's persistent tracks on /ov_msckf/health (features seen in 3+ frames, the count the keeper's dark rule reads). Under tracked the yaw-only bias walk counts from the last sample whose update used vio_min_features, not from the last one that passed. The wheel and rest rules and the guard are the same under both. Live |
| `visual_odometry` | `gaze_gate` | bool | on | yes | frames whose exposure window (stamp +- gate_exposure_s) overlaps a head saccade (/gaze/state's blind intervals, + gate_settle_s after settling) or holds a body yaw faster than gate_yaw_dps (/imu/data_raw; 0 is off) are dropped here and counted; off, every frame passes as before |

### Config knobs

A knob is a number, not a switch: its default, its range and one line of note live in
`config/knobs.json` under the node's name, the node declares it as a live parameter beside its
flags, and `ros/flags.sh get|set NODE KNOB` reaches it like a flag (a value outside the range is
refused with the reason). A default moves by editing that file; a change made live lasts until
the node restarts. Where the code already names the number, a unit test holds the two equal
(tests/unit/test_knobs.py).

| node | knob | kind | default | note |
| --- | --- | --- | --- | --- |
| `bag_recorder` | `ring_keep_h` | number 0.1..168 | 6.0 | goal_bag ring: a ring file last written longer ago than this (hours) is deleted at the next pass (every 10 s); 6 h is ~9 GB at the ~26 MB a minute estimated from drive bags |
| `bag_recorder` | `ring_keep_gb` | number 0.5..500 | 10.0 | goal_bag ring: the oldest ring files go while the ring holds more than this (GB), whatever their age; the file being written never goes |
| `bag_recorder` | `ring_floor_gb` | number 1..500 | 15.0 | goal_bag ring: ring files go while the Mac's disk has less than this free (GB), and the ring's recorder stops while even an empty ring leaves it under |
| `bag_recorder` | `preroll_s` | number 0..120 | 15.0 | goal_bag ring: a goal's bag begins this many seconds before the goal's word (the JSONL tape's prelude, pepin.tape.PRELUDE_S); a cold OpenVINS replay needs at least init_window_time (1 s) of rest in it |
| `bag_recorder` | `tail_s` | number 0..60 | 2.0 | goal_bag ring: a goal's bag ends this many seconds after the goal ended (the cart settling); the bag is cut this long and ~1 s after the end |
| `camera_stream` | `scale` | number 0..1 | 0.5 | the published picture as a fraction of the camera's own 1280x720, its optics scaled with it; a change takes the next frame. THE MONO RIG's flag: a stereo head publishes at its calibration's own size (the size the remap tables were built for, the size a matcher's disparity is in pixels of), so the node pins this to 1.0 there and refuses any other value with that reason |
| `camera_stream` | `camera_stamp_lag_s` | number 0..0.3 | 0.09 | camera_stamp grab only: every grab stamp is dated this many seconds earlier, to the exposure the V4L2 capture stamp sits behind (about one frame period at 10 fps whatever the light: a buffered frame, not the exposure); the send fallback is left alone. Measured with the head's gyro against the pictures' own motion (image flow vs /head/imu, scratch/vio_night/cam_imu_lag.py): 90/90 ms by day, 92/95 ms at night (2026-10-04); with 0.09 live the residual read +1 ms. config/camera.json's head_imu.time_offset_s is measured against the stamps at this knob's default, and ros/laptop.sh vio reads the live value at its start: restart the VIO after a change |
| `contact_scan` | `max_range` | number 0.1..10 | 2.0 | metres past which a column is called clear instead of ended; the costmap's contact_layer.obstacle_max_range must match it |
| `depth_fusion` | `tf_static_wait_s` | number 1..600 | 15.0 | seconds after the start before a WARN names the /tf_static edges still missing, the usual cause (a silent zenoh peer: a publisher cache that never answers the history query) and the debug recipe (pepin.static_facts); repeated every 4x this while one is missing |
| `depth_fusion` | `lean_gate_deg` | number 0..90 | 3.0 | a scan taken while the cart leans more than this many degrees is not integrated into the map; only with imu_lean on, which is where the lean is known at all |
| `depth_fusion` | `lean_min_quality` | number 0..1 | 0.5 | how much of the lean gravity must have voted for (pepin.lean's quality, printed beside the lean in this line) before a frame or a scan is placed by it: below it the lean is treated as unknown — the measurement is placed level and the scan gate admits it |
| `depth_fusion` | `min_weight` | number 0..100 | 4.0 | observations a voxel needs before it is shown in /fusion/surface and read out as /depth_marks, the two things this node publishes about the room |
| `depth_fusion` | `marks_min_z` | number 0..1 | 0.15 | the floor of the height band /depth_marks reads the volume in, metres above the cart's own floor plane; the band's top is the volume's own camera band (config/fusion.json's camera_band_m) |
| `depth_fusion` | `marks_hz` | number 0..30 | 5.0 | the cap on how often /depth_marks is PUBLISHED, in hertz; 0 publishes every frame, which is what this topic did until 2026-09-22. Only the publication is thinned: every frame and every revolution is still fused into the volume, and a slice that is not published is not computed either (the gate is read before the crossing search) |
| `depth_fusion` | `grid_hz` | number 0.5..10 | 3.0 | the cap on how often the camera grids are published, hertz |
| `depth_fusion` | `grid_size_m` | number 1..20 | 6.0 | the side of /camera_grid's square about the cart, metres |
| `depth_fusion` | `grid_resolution_m` | number 0.02..0.5 | 0.05 | the cell of /camera_grid, metres; /camera_grid_map always takes the map's |
| `depth_fusion` | `surface_hz` | number 0.1..10 | 1.0 | how often /fusion/surface is published (the crossing search costs a fraction of a second) |
| `depth_fusion` | `band_half_z` | number 0.02..0.5 | 0.125 | half the height band around the lidar's plane a frame is seated on, metres (config/fusion.json's band_half_z_m is the default); the band's centre is the plane the published base_link -> laser edge names, and both are printed in the report line |
| `depth_fusion` | `no_depth_weight` | number 0..2 | 0.5 | what a depthless ray's carve weighs, as a share of what a measurement AT the source's reach weighs (0.67 at the stereo rig's own 2.44 m, so 0.34 by default); 0 carves nothing, 1 makes a NaN as convincing as a measurement |
| `depth_fusion` | `no_depth_reach_m` | number 0..12 | 0.0 | the reach a depthless ray carves to, metres, when it must be stated; 0 (the default) MEASURES it from the frames themselves — the largest finite depth seen in the last 60 — and the report line prints what it found |
| `depth_stream` | `tf_static_wait_s` | number 1..600 | 15.0 | seconds after the start before a WARN names the /tf_static edges still missing, the usual cause (a silent zenoh peer: a publisher cache that never answers the history query) and the debug recipe (pepin.static_facts); repeated every 4x this while one is missing |
| `depth_stream` | `scale_ceiling` | number 0.5..20 | 5.0 | the largest 1 / scale the law may be fitted to (the upper half of pepin.depth.A_BOUNDS); a law that lands on a bound prints AT BOUND |
| `depth_stream` | `carry_max_speed_mps` | number 0.1..20 | 1.0 | metres per second the carry from the scan's moment to the frame's may imply before the frame's lidar beams are thrown away instead of anchoring the law; the frame still publishes its depth, it simply judges nothing |
| `depth_stream` | `lean_min_quality` | number 0..1 | 0.5 | how much of the lean gravity must have voted for (pepin.lean's quality, printed beside the lean in this line) before a frame is placed by it: below it the lean is treated as unknown and the frame is placed level |
| `depth_stream` | `tf_dead_s` | number 0..600 | 3.0 | how far behind a frame's stamp TF's newest edge may be before that edge is taken for dead and no lookup on the frame's path waits for it: the camera pose falls to config/camera.json's mount and the lidar's scan passes uncarried, both at once and both counted. 0 turns the guard off — every lookup waits CARRY_WAIT_S again |
| `depth_stream` | `depth_reach_m` | number 0.3..12 | 4.0 | metres past which the published depth is NaN; the same number /depth_scan is capped at |
| `depth_stream` | `scan_hz` | number 0..30 | 5.0 | the cap on how often /depth_scan is PUBLISHED, in hertz; 0 publishes one fan per frame, which is what this topic did until 2026-09-22. The cap is on the publisher alone: every frame still goes through the network and the whole pipeline, every law is still fitted from it, and the depth image on /camera/depth is not thinned at all |
| `depth_stream` | `gate_exposure_s` | number 0..0.2 | 0.035 | half the exposure window the gaze gate judges a frame by: stamp +- this, because whether ustreamer stamps the start or the end of the exposure is not known; 0.035 covers an auto exposure up to half a 15 fps frame. With a manual exposure (config/camera.json's exposure block) set it to that exposure |
| `depth_stream` | `gate_settle_s` | number 0..1 | 0.1 | how long after the head settled (the since of the phase it settled into, /gaze/state) a frame is still blind: the gaze contract's one frame period, 0.105 s at 9.5 fps; raise it if the mast is seen ringing after a saccade |
| `depth_stream` | `gate_yaw_dps` | number 0..360 | 0.0 | a frame whose exposure window holds an IMU sample turning faster than this about base_link z (deg/s) is dropped like a saccade frame; 0 is off (as shipped: the base turns at most 57 deg/s and nothing has measured where its blur starts to cost) |
| `depth_stream` | `gate_stamp_end` | integer 0..1 | 1 | 1: a frame's stamp is the END of its exposure (camera_stream's camera_stamp grab: the V4L2 capture stamp sits after the exposure) and the window is [stamp - 2 * gate_exposure_s, stamp + 10 ms]; 0: stamp +- gate_exposure_s, for ustreamer's send stamp whose relation to the exposure is unknown (1-68 ms late, 2026-10-02). Set it to 1 together with camera_stamp grab |
| `depth_stream` | `gate_sway_dps` | number 0..360 | 6.0 | a frame whose window holds a mast sway rate above this (deg/s, the norm of /mast/state's velocities, the base bridge's head gyro minus base yaw and neck) is dropped as swaying; small sway is kept because the bridge's TF corrects it. 0 is off (as shipped: /mast/state does not exist before the head IMU); vio.md proposes 10 (the 5.3 Hz ring peaks at 6.7 deg/s, a knock is faster) |
| `depth_stream` | `gate_sway_deg` | number 0..10 | 1.0 | a frame whose window holds a mast sway angle above this (deg, the norm of /mast/state's positions) is dropped as swaying: beyond what the TF correction is trusted for (a knock, a runaway). 0 is off (as shipped); vio.md proposes 1.0 |
| `depth_stream` | `gate_dark_floor` | number 0..255 | 10.0 | gate_dark: a frame whose mean luma (/camera/brightness, camera_stream's 1/8 subsample of the whole stereo frame, 0-255) is under this is dropped as dark, and so is every frame after it until one reaches this plus gate_dark_hyst. 10 sits above the bench's collapse and under every frame of the 2026-10-06 evening drives (darkest 11.1): manual 8.3 ms on a dark doorway (mean estimated <= 8 from the lit-room ladder: 8.3 ms gives 0.145 of the auto picture, the auto doorway 57) left RTAB-Map 0-1 stereo inliers and made 156 camera-only births in 20 s against 10-30 under auto; 33 ms (<= 18) 45 inliers, 96 births; 67 ms (<= 21) 138, 34. On the 16 evening drives frames at 11-30 grey made 1.3-2.8 camera-only births per integrated frame against 4.1-4.6 above 40, the lidar's own births falling alike (0.7-1.9 vs 1.8-2.7): no excess from the dim frames, so the floor is not set at the dim room (scratch/depth_dark_floor/floor.py, replay.py) |
| `depth_stream` | `gate_dark_hyst` | number 0..50 | 2.0 | gate_dark: grey levels above gate_dark_floor a frame must reach before the dark ends, so a picture hovering at the floor does not flicker in and out (frame to frame the brightness moves p50 0.25-0.65 grey on the evening drives, p99 6.5-28 on turns) |
| `gaze` | `frames` | integer 0..20 | 3 | still depth frames a look waits for at each view: fused into the volume and stamped later than one frame period after the head settled; a request may ask for its own number |
| `gaze` | `settle_tol_deg` | number 0.1..10 | 1.0 | how close, in degrees, the encoders must read to the target for the head to have arrived |
| `gaze` | `move_timeout_s` | number 0.5..10 | 3.0 | a move not settled by then ends unreached (the encoders say where the head is), and a write the base server keeps refusing for this long is denied |
| `gaze` | `frame_period_s` | number 0..1 | 0.105 | the blind tail after the head settles, one depth frame period at 9.5 fps: /gaze/state's blind_until is the settle plus this |
| `gaze` | `ttl_operator_s` | number 0.1..10 | 0.5 | the default life of an operator's request (band 0), renewed while keys are held |
| `gaze` | `ttl_navigation_s` | number 0.5..30 | 3.0 | the default life of a navigation request (band 1): the stall look's saccade and frames must fit in it |
| `gaze` | `ttl_person_s` | number 1..120 | 10.0 | the default life of the person's request (band 2: look, look_around, find); see renews it |
| `gaze` | `ttl_sensor_s` | number 0.5..30 | 2.0 | the default life of a sensor-triggered check (band 3) |
| `gaze` | `ttl_driving_s` | number 0.2..5 | 0.5 | the life of path gaze and reverse gaze (band 4), renewed every path_period_s while they apply |
| `gaze` | `ttl_idle_s` | number 1..300 | 20.0 | the default life of an idle request (band 5) |
| `gaze` | `drive_home_tol_deg` | number 0.5..45 | 5.0 | at a drive's start a head the encoders read further than this from where the arbiter last put it (a jog, a hand, ros/neck.sh) is sent home: no drive starts with a crooked head |
| `gaze` | `stall_ahead_m` | number 0.2..3 | 1.0 | how far along the plan the hull is swept for blockers at a stall |
| `gaze` | `stall_margin_m` | number 0..0.3 | 0.05 | the hull grown by this much on every side for the sweep (one costmap cell) |
| `gaze` | `stall_cluster_m` | number 0.05..1 | 0.25 | candidates first reached within this much of the nearest one are one blocker: the look aims at their centroid |
| `gaze` | `stall_match_cells` | number 0.5..5 | 1.5 | how near a lidar return or a camera mark must land to a blocking cell, in costmap cells, to account for it (marks_audit's match_cells) |
| `gaze` | `stall_column_bottom_m` | number 0..1 | 0.15 | the bottom of the columns asked of /fusion/column over the blockers, metres above the cart's floor plane: depth_fusion's marks_min_z, where the camera's marks start, so the floor's own surface is never a blocker |
| `gaze` | `stall_column_top_m` | number 0.3..2 | 1.3 | the top of the columns asked of /fusion/column over the blockers, metres above the cart's floor plane: the top of the band the marks are read in (pepin.volume_scan.MARKS_MAX_Z_M) |
| `gaze` | `stall_max_depression_deg` | number 45..95 | 85.0 | a blocker deeper below the lens than this is in the frame's last rows: the look answers 'back off' and the tree backs up before asking again |
| `gaze` | `slow_deg_s` | number 1..60 | 20.0 | the head's speed for a 'slow' request (a detector that wants unblurred frames), with neck_target; a saccade goes at the board's own top speed |
| `gaze` | `target_renew_s` | number 0.1..5 | 0.5 | how often a held neck_target is sent again, and never slower than half config/neck.json's motion.lease_s: the lease's lapse sends the head home |
| `gaze` | `path_period_s` | number 0.05..2 | 0.2 | how often path gaze and reverse gaze are recomputed and renewed while a drive runs |
| `gaze` | `path_lookahead_s` | number 0.5..5 | 2.0 | path gaze looks at the plan's point this many seconds of the current speed ahead |
| `gaze` | `path_min_m` | number 0.2..3 | 0.6 | ...but at least this far along the plan |
| `gaze` | `path_max_m` | number 0.3..5 | 1.5 | ...and at most this far |
| `gaze` | `path_deadband_deg` | number 0..45 | 22.0 | the zone path gaze follows in: the held aim stays while the plan's aim is within this of it, saccade and hold, no creeping (8 until 2026-10-05; uv run python ros/tools/gaze_preset.py baseline) |
| `gaze` | `path_pan_clamp_deg` | number 0..150 | 60.0 | path gaze's pan limit while driving: a goal behind is turned to by the body, not the head |
| `gaze` | `path_near_m` | number 0.2..3 | 1.0 | a path point nearer than this tilts the head below home |
| `gaze` | `path_near_offset_deg` | number 0..45 | 15.0 | ...to atan(lens height / distance) less this, so the floor under the point sits in the lower third of the picture |
| `gaze` | `path_hyst_s` | number 0..3 | 0.3 | a plan aim outside the zone moves the head only after it has stayed outside this long (0: at once, as before 2026-10-05) |
| `gaze` | `path_cooldown_s` | number 0..10 | 2.0 | path gaze moves the head at most once in this many seconds; a path look that lost the head to another look is aimed at once (0: no cooldown, as before 2026-10-05) |
| `gaze` | `path_tail_s` | number 0..5 | 0.5 | no new path saccade once the plan's remaining arc at the current speed is this many seconds or less: the drive's end is reached with the head still (0: off, as before 2026-10-05) |
| `gaze` | `path_tail_m` | number 0..2 | 0.35 | ...nor once the plan's remaining arc is this short, whatever the speed: parking slows the cart, so the seconds alone read the end as far off (drive 0330: 0.16 m at 0.06 m/s read 2.5 s, a 40 deg saccade 0.5 s before the end) (0: off, as before 2026-10-06) |
| `gaze` | `path_hold_s` | number 0..300 | 60.0 | while path gaze has no aim (reversing before a reverse look, no plan point ahead) its look is renewed at its aim for up to this long, instead of lapsing home after ttl_driving_s (0: lapses, as before 2026-10-05) |
| `gaze` | `path_still_m_s` | number 0..0.2 | 0.02 | the cart stands while its command is slower than this (and turns slower than 0.1 rad/s): path gaze then makes no saccade unless the plan's aim is more than path_still_deg off the head, and a path look that lost the head takes it where it is, the first saccade waiting for the command that moves the cart (drives 0348-0371: 59 fresh path looks made standing, 21 in 0351's recovery loop) (0: off, as on drives 0348-0371) |
| `gaze` | `path_still_deg` | number 0..180 | 60.0 | ...unless the plan's aim is more than this off where the head is (the head looking back or down at a blocker while the plan turns away) |
| `gaze` | `path_stall_guard_s` | number 0..3 | 0.5 | for this long after the command dropped to zero from a moving one, path gaze makes no saccade at all: the controller stopped, a stall look is coming and takes the head from where it is (drives 0348-0371: 29 path looks held still under 0.5 s, 18 frames in all) (0: off, as on drives 0348-0371) |
| `gaze` | `path_recentre_s` | number 0..10 | 1.0 | once the plan's aim has been within path_recentre_deg of straight ahead this long while the held aim is more than half the zone from it, the head moves once to the plan's aim, the cooldown permitting: the cart's own turn carried the path ahead and the zone alone keeps the head aside (drive 0365: 24 deg right for 2.4 s along a straight path) (0: off, as on drives 0348-0371) |
| `gaze` | `path_recentre_deg` | number 0..30 | 5.0 | ...the band around straight ahead the plan's aim must stay in |
| `gaze` | `path_bend_deg` | number 0..90 | 30.0 | STRAIGHT AHEAD: path gaze looks straight ahead unless the plan point ahead is more than this off the nose (a bend look; it ends once the point is back within 0.6 of this), and a head left aside comes back straight ahead once the bend is passed, the hysteresis and the cooldown permitting. Drives 0376-0379: the plan point a median 5 deg off on straights (p90 13), 37 in bends over 30 deg; on the eleven tapes 0330-0379 (replays) 20/25/30/35/40 deg made 90/89/86/85/84 writes (88 before) with the next 2 s of plan inside the frame alike (88.5-89.0 %); at 30 with the parking: 83 writes, the head straight ahead 60 % of forward driving (43 before), path frames 837 (822), stall looks' frames unchanged (0: the plan point itself, as before 2026-10-07) |
| `gaze` | `path_bend_lead_s` | number 0..10 | 0.0 | ...the plan point the bend is read at and looked to is this many seconds of the current speed ahead, within path_min_m..path_max_m (0: path_lookahead_s's point, 0.6 m at the cart's <= 0.3 m/s). The 2 s point is early enough: a bend look settles a median 0.63 s before the cart has turned 15 deg (p10 -0.16 s, 3 of 24 late); 3 s or 4 s the same (0.64/0.69 s), bends are taken at pivot speeds where path_min_m binds, and 4 s adds 2 writes on the tapes |
| `gaze` | `path_park_ahead_m` | number 0..5 | 1.0 | PARKING: from the goal server's hand-over to the parker (FollowPathMPPI on /controller_selector, mid-drive) or the first time the plan left is this short, path gaze looks at the parking spot (path_park_beyond_m past that plan's end along its last step, fixed for the rest of the drive: the final headings were within 4-11 deg of that step) at the driving tilt, the straight-ahead rule the same, and the tail no longer freezes the head aside: it comes back straight ahead once the cart faces the spot (drive 0376: frozen 53 deg right for the last 3.4 s); on the tapes the spot inside the calibrated frame from the hand-over to the end 85 -> 95 % (0376 27 -> 100 %) (0: off, the plan point to the end, as before 2026-10-07) |
| `gaze` | `path_park_beyond_m` | number 0..2 | 0.5 | ...how far past the plan's end the spot is: what a cart parked nose-to-furniture stands against |
| `gaze` | `reverse_pan_deg` | number 60..156 | 150.0 | reverse gaze's pan toward the rear, on the side the rear swings to |
| `gaze` | `reverse_tilt_deg` | number 0..63 | 23.8 | reverse gaze's tilt: home's until the body self-filter lands, since deeper looks back see the cart's own top shelf |
| `gaze` | `reverse_min_s` | number 0..10 | 1.0 | a reverse leg nobody announced (neither reverse_min_m nor reverse_recovery) must last this long before the head turns back (unless the rear is tight) |
| `gaze` | `reverse_min_m` | number 0..2 | 0.15 | a reverse leg the plan announces — /plan leaves the cart backwards (its first step behind the cart's heading) for this long up to its first cusp — turns the head back at the leg's first reversing command, not after reverse_min_s (drives 0348-0363: requested 1.17 s into the leg, settled after 9 of 12 1-2 s legs had ended); Hybrid-A*'s reverse legs come in 0.078 m steps, so 0.15 takes two or more (0: off, as before 2026-10-06) |
| `gaze` | `reverse_park_min_m` | number 0..2 | 0.3 | while parking (path_park_ahead_m), the plan announces a reverse leg only from this long, and a leg nobody announced gets no look by reverse_min_s: the parker's fresh plans carry 0.08-0.16 m reverse stubs and it creeps back at 0.02-0.03 m/s while it pivots (drive 0379: a 150 deg look back 0.41 s before 'Reached the goal'), and only a behaviour server's goal is a recovery (drive 0373: the controller's goal ended at the goal with the creep, v -0.04, still the command: a 150 deg look 0.1 s before the end); a tight rear or a BackUp still turns the head (0: the reverse look's rules in the parking too, as before 2026-10-07) |
| `gaze` | `reverse_recovery` | integer 0..1 | 1 | 1: a reverse leg the tree's recovery drives (a /backup or /drive_on_heading goal runs, or /follow_path's status was heard and runs none) turns the head back at its first reversing command (0: only by reverse_min_m or reverse_min_s, as before 2026-10-06) |
| `gaze` | `reverse_hold_s` | number 0..60 | 1.5 | the reverse look is held through a stand after its leg for up to this long (a reverse within it is the same leg, same side) and let go at once at the first forward command or the stand's end, a glance under way too: drives 0330 and 0348-0363 stood <= 1.5 s after 60 of 73 legs, and 3-68 s (spinning recoveries) after 11 (0: let go when the reversing ends, after its glance, as before 2026-10-06) |
| `gaze` | `reverse_rear_m` | number 0..1 | 0.3 | a lethal cell this close behind the hull makes the rear tight: the head turns back from the first reversing twist |
| `gaze` | `reverse_frames` | integer 0..20 | 3 | the clean (gate-passed, fused) frames the reverse look waits for once settled before the path look may take the head (0: answered on arrival, as before 2026-10-05) |
| `gaze` | `glance_dwell_s` | number 0..3 | 0.6 | above 0 the reverse and stall looks are atomic glances: nothing but a better band takes the head before they are done, the reverse look at reverse_frames or this long after settling, the stall look at frames or ttl_navigation_s; the reverse glance takes the head from the path look at once, even mid-swing, and with reverse_hold_s it is cut at the first forward command (0: no glance, the reverse look waits for a path look to arrive, as before 2026-10-05) |
| `gaze` | `return_deg_s` | number 0..300 | 45.0 | after a drive the head goes home at this speed, with neck_target (0: a saccade at the board's top speed, as before 2026-10-05) |
| `gaze` | `gate_dark_patience_s` | number 0..30 | 1.5 | a look whose frames depth_stream's gate has been dropping as dark (/depth/dark) for this long, board seconds from the first dark frame after the last fused one, gives up as 'dark, no frames' (expired, counted dark in the report line) instead of holding the head for its TTL; the auto exposure catches up in 1-2 s after a turn from a window (the 2026-10-06 bench), and the look ends early anyway once its frames are fused. 0 waits for the TTL |
| `goal_server` | `park_distance_m` | number 0..10 | 1.0 | under the controller flag's shim_mppi: within this many metres of the goal (straight line from map -> base_link) the drive is handed from the shim to MPPI for the rest of that goal; 0 never hands over |
| `marks_audit` | `tf_static_wait_s` | number 1..600 | 15.0 | seconds after the start before a WARN names the /tf_static edges still missing, the usual cause (a silent zenoh peer: a publisher cache that never answers the history query) and the debug recipe (pepin.static_facts); repeated every 4x this while one is missing |
| `marks_audit` | `radius_m` | number 0.2..3 | 2.0 | how far around the cart a lethal cell is judged, metres |
| `marks_audit` | `match_cells` | number 0.5..5 | 1.5 | how near a beam must land to a cell, in costmap cells, to account for it |
| `rtabmap_frame` | `pnp_reproj_px` | number 1..4 | 2.0 | Vis/PnPReprojError: how far, in pixels, a database point may reproject from its match in the current picture and still count as an inlier of the visual registration (20 inliers accept it, Vis/MinInliers). Sent with the strategy's parameters and changed live |
| `rtabmap_frame` | `registration_timeout_s` | number 0.05..5 | 1.0 | how long RTAB-Map's adapters wait for one answer of the localisation service before it counts as none (auto then computes it locally, and the service is left alone for 10 s) |
| `rtabmap_frame` | `xfeat_top_k` | integer 256..8192 | 2048 | the most XFeat keypoints a picture keeps, best score first, wherever XFeat runs; RTAB-Map applies no cap of its own to a Python detector |
| `rtabmap_frame` | `descriptor_null_share` | number 0..1 | 0.5 | place_recognition descriptor falls back to the words while more than this share of the last 10 camera snapshots carried the null descriptor (the localisation service down, late or answering nonsense), and goes back once they are described again; 1.0 never falls back |
| `sensor_pack` | `pack_hz` | number 0.1..15 | 1.0 | at most this many snapshots a second of SENSOR time (the stamps' own clock, not this laptop's) |
| `sensor_pack` | `pair_periods` | number 0.5..10 | 1.5 | how many of its OWN measured periods a source's message may be from the snapshot's stamp and still be paired with it (pepin.snapshot) |
| `sensor_pack` | `place_timeout_s` | number 0.05..0.95 | 0.5 | the most a snapshot waits for its place vector, from the moment it is packed — its wait in the worker's queue included — before it goes out with the null descriptor |
| `sensor_pack` | `gate_exposure_s` | number 0..0.2 | 0.035 | half the exposure window the gaze gate judges a frame by: stamp +- this, because whether ustreamer stamps the start or the end of the exposure is not known; 0.035 covers an auto exposure up to half a 15 fps frame. With a manual exposure (config/camera.json's exposure block) set it to that exposure |
| `sensor_pack` | `gate_settle_s` | number 0..1 | 0.1 | how long after the head settled (the since of the phase it settled into, /gaze/state) a frame is still blind: the gaze contract's one frame period, 0.105 s at 9.5 fps; raise it if the mast is seen ringing after a saccade |
| `sensor_pack` | `gate_yaw_dps` | number 0..360 | 0.0 | a frame whose exposure window holds an IMU sample turning faster than this about base_link z (deg/s) is dropped like a saccade frame; 0 is off (as shipped: the base turns at most 57 deg/s and nothing has measured where its blur starts to cost) |
| `sensor_pack` | `gate_stamp_end` | integer 0..1 | 1 | 1: a frame's stamp is the END of its exposure (camera_stream's camera_stamp grab: the V4L2 capture stamp sits after the exposure) and the window is [stamp - 2 * gate_exposure_s, stamp + 10 ms]; 0: stamp +- gate_exposure_s, for ustreamer's send stamp whose relation to the exposure is unknown (1-68 ms late, 2026-10-02). Set it to 1 together with camera_stamp grab |
| `sensor_pack` | `gate_sway_dps` | number 0..360 | 6.0 | a frame whose window holds a mast sway rate above this (deg/s, the norm of /mast/state's velocities, the base bridge's head gyro minus base yaw and neck) is dropped as swaying; small sway is kept because the bridge's TF corrects it. 0 is off (as shipped: /mast/state does not exist before the head IMU); vio.md proposes 10 (the 5.3 Hz ring peaks at 6.7 deg/s, a knock is faster) |
| `sensor_pack` | `gate_sway_deg` | number 0..10 | 1.0 | a frame whose window holds a mast sway angle above this (deg, the norm of /mast/state's positions) is dropped as swaying: beyond what the TF correction is trusted for (a knock, a runaway). 0 is off (as shipped); vio.md proposes 1.0 |
| `sensor_pack` | `gate_dark_floor` | number 0..255 | 10.0 | gate_dark: a frame whose mean luma (/camera/brightness, camera_stream's 1/8 subsample of the whole stereo frame, 0-255) is under this is dropped as dark, and so is every frame after it until one reaches this plus gate_dark_hyst. 10 sits above the bench's collapse and under every frame of the 2026-10-06 evening drives (darkest 11.1): manual 8.3 ms on a dark doorway (mean estimated <= 8 from the lit-room ladder: 8.3 ms gives 0.145 of the auto picture, the auto doorway 57) left RTAB-Map 0-1 stereo inliers and made 156 camera-only births in 20 s against 10-30 under auto; 33 ms (<= 18) 45 inliers, 96 births; 67 ms (<= 21) 138, 34. On the 16 evening drives frames at 11-30 grey made 1.3-2.8 camera-only births per integrated frame against 4.1-4.6 above 40, the lidar's own births falling alike (0.7-1.9 vs 1.8-2.7): no excess from the dim frames, so the floor is not set at the dim room (scratch/depth_dark_floor/floor.py, replay.py) |
| `sensor_pack` | `gate_dark_hyst` | number 0..50 | 2.0 | gate_dark: grey levels above gate_dark_floor a frame must reach before the dark ends, so a picture hovering at the floor does not flicker in and out (frame to frame the brightness moves p50 0.25-0.65 grey on the evening drives, p99 6.5-28 on turns) |
| `vio_feed` | `gate_exposure_s` | number 0..0.2 | 0.035 | half the exposure window a pair is judged by, as in the gated camera nodes (with gate_stamp_end 1: [stamp - 2 * this, stamp + 10 ms]); with a manual exposure set it to that exposure |
| `vio_feed` | `gate_stamp_end` | integer 0..1 | 1 | 1: a frame's stamp is the END of its exposure (camera_stream's camera_stamp grab) and the window is [stamp - 2 * gate_exposure_s, stamp + 10 ms]; 0: stamp +- gate_exposure_s. Set it to 1 together with camera_stamp grab, which the VIO always runs on |
| `vio_feed` | `head_rate_dps` | number 0..1000 | 60.0 | a pair whose exposure window holds a /head/imu sample turning faster than this (deg/s, the gyro's norm) is held back from OpenVINS; 0 holds back nothing. 60 is three times the arbiter's slow sweep (20 deg/s, whose frames OpenVINS needs) and a fifth of a saccade (290 deg/s, whose frames threw it metres off on the run4 replay) |
| `vio_keeper` | `vio_disagree_m_s` | number 0..10 | 0.2 | the disagreement rule: a frame whose OpenVINS velocity at the IMU (/ov_msckf/health v_I) is farther than this (m/s) from the EKF's carried to the IMU (/vio/seed_twist, sent only while the head is still, used within 0.3 s) adds its interval to a run; 0 turns the rule off. Replayed divergences ran 0.2 -> 0.5 m/s off within 1 s, healthy filters within 0.02-0.17 of the wheels (scratch/vio_recover r4) |
| `vio_keeper` | `vio_disagree_s` | number 0.1..30 | 0.7 | the disagreement rule: this many seconds of disagreeing frames (frames without a fresh reference, or taken in a swing, not counted) make OpenVINS lost. 0.7 since 2026-10-06 (scratch/vio_push/keeper.py, the keeper replayed on its own seeds over drives 0331-0363, its 17 live resets matched): 0.6-0.8 also reset 0361 +25.4 (0.3-0.9 m/s off the truth for 3 s, its run cut at 0.8 s by one frame at 0.17) and 0349 +26.6, with no reset outside a divergence; 0.5 adds one (0351 +134.0), 1.0 missed both |
| `vio_keeper` | `vio_dark_tracks` | integer 0..500 | 10 | the dark rule: a frame with fewer persistent tracks than this (the left eye's features seen in seed_track_frames frames, /ov_msckf/health) adds its interval to a run, unless OpenVINS held it with its zero-velocity update (vio_dark_at_rest) or the camera was swinging (vio_swing_rad_s); 0 turns the rule off. Not the features the update used: 0-8 in good light with the gaze moving (drives 0329/0330 replayed), while the tracker kept 20-45 persistent tracks in the dimmest stretch |
| `vio_keeper` | `vio_dark_s` | number 0.1..30 | 1.0 | the dark rule: this many seconds of such frames (swing frames not counted) make OpenVINS lost, and vio_recover brings it back |
| `vio_keeper` | `vio_swing_rad_s` | number 0.1..20 | 1.0 | a frame whose largest bias-corrected gyro rate since the last frame is over this (rad/s, /ov_msckf/health's gyro) was taken in a swing: the dark rule neither counts nor clears it. Saccades reach 5 rad/s, the cart turns at 0.5 at most (drives 0329/0330) |
| `vio_keeper` | `vio_max_speed_m_s` | number 0.1..100 | 1.0 | the runaway rule: OpenVINS's own IMU speed over this (m/s) for vio_speed_s makes it lost; the cart's cap is 0.30 m/s, a head pan moves the IMU a few cm/s |
| `vio_keeper` | `vio_speed_s` | number 0..10 | 0.3 | the runaway rule: how long the speed must stay over vio_max_speed_m_s, seconds (frame time) |
| `vio_keeper` | `vio_grace_s` | number 0..30 | 2.0 | after a reset nothing is judged until the new filter publishes and this long has passed (ModalAI voxl-open-vins-server's ok_state_grace_timeout_s 2.0): its first second has few SLAM features by construction |
| `vio_keeper` | `head_still_s` | number 0.05..5 | 0.3 | the velocity seed: the head counts as still when its orientation in base_link (TF head_imu <- base_link: the neck's encoders and the mast) turned no more than head_still_deg over this many seconds; only then is the EKF's twist carried to the head IMU as one rigid body |
| `vio_keeper` | `head_still_deg` | number 0..10 | 1.0 | the velocity seed: the largest turn of the head in base_link (degrees) over head_still_s that still counts as still (encoder jitter ~0.1 deg; the mast's sway, 6.7 deg/s at its 5.3 Hz peak, moves the IMU ~9 cm/s at 0.8 m) |
| `vio_keeper` | `seed_vz_sigma_m_s` | number 0..1 | 0.02 | the velocity seed: the sigma of the cart's vertical speed (m/s), which the planar EKF does not estimate (zero; the mast sways) |
| `vio_keeper` | `seed_min_features` | integer 0..500 | 15 | OpenVINS's warm seed waits for this many features in the left eye tracked over seed_track_frames frames: a picture first, a seed into the dark would only propagate the IMU (pushed to OpenVINS's parameters) |
| `vio_keeper` | `seed_track_frames` | integer 1..20 | 3 | OpenVINS's warm seed: a feature counts toward seed_min_features once seen in this many frames (noise corners of a dark frame rarely survive the tracker's RANSAC that long) |
| `vio_keeper` | `seed_bias_max_age_s` | number 0..3600 | 60.0 | OpenVINS's warm seed keeps the IMU biases of the old filter's last healthy update (enough features, or a ZUPT) at most this old (s); older, the reset waits for OpenVINS's own initialiser (static at rest). The head IMU's random walk adds 0.8 mrad/s and 0.012 m/s^2 over 60 s |
| `vio_keeper` | `seed_bias_lag_s` | number 0..60 | 2.0 | OpenVINS's warm seed takes the bias snapshot from at least this long before the reset (s): a divergence corrupts the biases before anyone calls it one |
| `vio_keeper` | `seed_accel_window_s` | number 0.05..3 | 0.5 | OpenVINS's warm seed: gravity is the accelerometer's mean over this window before the frame, each sample turned into the frame's axes by the gyro (s); the cart's vibration (1.5 m/s^2 spread per sample while driving) averages out |
| `vio_keeper` | `seed_accel_allow_m_s2` | number 0..5 | 0.3 | OpenVINS's warm seed: the unmodelled acceleration allowed in that mean (m/s^2): the tilt's sigma is (spread / sqrt(n) + this) / g, never under seed_sigma_q_rad |
| `vio_keeper` | `seed_gyro_max_rad_s` | number 0..20 | 1.0 | OpenVINS's warm seed waits while the bias-corrected gyro over the window exceeds this (rad/s): a swing's centripetal acceleration is not gravity |
| `vio_keeper` | `seed_g_tol_m_s2` | number 0..5 | 0.5 | OpenVINS's warm seed waits while the window's mean specific force is farther than this from gravity's magnitude (m/s^2) |
| `vio_keeper` | `seed_vel_max_age_s` | number 0..5 | 0.2 | OpenVINS's warm seed takes the /vio/seed_twist nearest the frame's IMU time (the last 3 s are kept), only within this (s); none (the head moving, the EKF silent), it waits. The newest seed was 0.2-0.4 s ahead of the frame in the 0329/0330 replay |
| `vio_keeper` | `seed_sigma_q_rad` | number 0..1 | 0.02 | OpenVINS's warm seed: the orientation's least sigma (rad), as the static initialiser's 0.02; the yaw is the new frame's gauge |
| `vio_keeper` | `seed_sigma_p_m` | number 0..10 | 0.05 | OpenVINS's warm seed: the position's sigma (m), the static initialiser's 0.05; the position is the new frame's origin (the relay re-anchors) |
| `vio_keeper` | `seed_sigma_v_m_s` | number 0..1 | 0.03 | OpenVINS's warm seed: added in quadrature to the seed velocity's own sigma (m/s): the EKF's lag at the cart's 0.5 m/s^2 |
| `vio_keeper` | `seed_bias_inflation` | number 0.1..100 | 1.0 | OpenVINS's warm seed: the kept biases' sigma (their snapshot's marginal plus the random walk since) times this; ModalAI inflates 3x after a divergence |
| `vio_keeper` | `seed_sigma_bg_floor` | number 0..1 | 0.005 | OpenVINS's warm seed: the gyro bias's least sigma (rad/s), ModalAI's floor |
| `vio_keeper` | `seed_sigma_ba_floor` | number 0..5 | 0.02 | OpenVINS's warm seed: the accelerometer bias's least sigma (m/s^2), ModalAI's floor |
| `vio_keeper` | `reset_frame_gap_s` | number 0.2..600 | 2.0 | OpenVINS resets itself (openvins-reset.patch) when a frame comes this long after its last (s): propagating across the gap and its zero-velocity check over every IMU sample in it (a dense 6(n-1)-square matrix: 24000 rows for 20 s) would follow |
| `vio_keeper` | `odomimu_delay_s` | number 0..10 | 0.5 | OpenVINS publishes /ov_msckf/odomimu (the relay's twist covariance) this long after it initialised (s; upstream 1.0): a warm seed publishes poses 0.5 s after the seed (five clones), and their twists need it |
| `visual_odometry` | `vo_sigma_m` | number 0.001..1 | 0.07 | the constant position sigma of one visual-odometry pose, in metres; the EKF differences two of them into a velocity and the covariance rides along — as (this pose's + the previous pose's) TIMES the gap, so what the filter actually weighs is a velocity variance of 2 * sigma^2 * dt |
| `visual_odometry` | `vo_yaw_sigma_deg` | number 0.1..180 | 5.0 | the constant yaw sigma of one visual-odometry pose, in degrees; since 2026-09-15 the board's EKF fuses this yaw differentially (ekf.yaml odom1_config index 5), so this number is what sizes a second heading source against the gyro |
| `visual_odometry` | `vo_max_speed` | number 0.05..10 | 1.0 | a step between two visual-odometry poses faster than this, in m/s, is dropped: rtabmap restarting its tracking moves the pose without moving the cart |
| `visual_odometry` | `vo_max_gap_s` | number 0.1..60 | 1.0 | a pose that arrives more than this many seconds after the previous one is dropped and becomes the new anchor: across a gap the speed and turn ceilings are ratios and measure nothing |
| `visual_odometry` | `vo_max_turn` | number 5..720 | 180.0 | a turn between two visual-odometry poses faster than this, in deg/s, is dropped, for the same reason as vo_max_speed |
| `visual_odometry` | `vo_reset_radius_m` | number 0..1 | 0.05 | a pose that lands this close to rtabmap's own origin while the previous one was farther out is its re-initialisation, not a drive, and is dropped; 0 turns the check off |
| `visual_odometry` | `vo_publish_hz` | number 0..30 | 10.0 | rtabmap's inputs (stereo, depth): how often a gated pose may leave for the board's EKF, in hertz; 0 publishes every one of them. Under vo_input vio the rate is vio_publish_hz |
| `visual_odometry` | `vio_publish_hz` | number 0..200 | 50.0 | vo_input vio: how often a sample may leave for the board's EKF, in hertz (the /vo pose and the /vo_twist twist); 0 sends every one. 50 = the wheels' /odom rate: the EKF weighs a source by information per SECOND (samples/s over variance), not per sample, so at 10 Hz against the wheels' 50 the VIO carried ~1/3 of the velocity vote at best (2026-10-05 drives). Only vio_twist_source imu reaches it: poseimu (the step source) comes once per camera frame, 9.5-11.6 Hz, and this only stops the cap from skipping one. The imu source's samples between two camera updates share that update's velocity error (odomimu is the last update's state propagated on the IMU), so 50 Hz does not mean 5x the independent information: its sigma is OpenVINS's own, unscaled, until drives measure the vote. Board EKF: ~25 % of a core at 10 Hz twist (2026-10-05, 15-25 % before); +40 corrections/s at 50 Hz, est. 28-33 % |
| `visual_odometry` | `vio_sigma_scale` | number 0.2..10 | 1.75 | vo_output twist: OpenVINS's linear-velocity sigma (vx, vy) multiplied by this before the EKF (cross terms by the product of the two scales). Measured on drives 0329/0330 (step source, vs the wheels): z p50 1.2, p90 2.9-3.9 moving (sigma ~2x small), 0.03-0.12 at rest (~10x big); offline the imu source reads z p50 1.3-1.6, p90 3.8-4.7 moving, 0.6-1.1 at rest. 1.0 until more drives say one number for both: no fudge from two drives |
| `visual_odometry` | `vio_yaw_sigma_scale` | number 0.2..10 | 2.5 | vo_output twist: OpenVINS's yaw-rate sigma multiplied by this (its reported sigma_w^2/dt is 0.96 deg/s, the x10 noise block at 200 Hz); under lost the bias walk sqrt(gyro_random_walk^2 * t) is added after the scale. Offline vs the base gyro: z p50 1.3 moving with the head still (imu source), 0.1-0.2 at rest. 1.0 until drives measure it |
| `visual_odometry` | `vio_step_fraction` | number 0..1 | 0.03 | vo_covariance vio: the VIO's drift per metre of step, added in quadrature to the vo_sigma_m floor (one wheel sample at 10 Hz). 0.03 is an ESTIMATE (vio.md M3) until the offline A/B measures it |
| `visual_odometry` | `vio_lost_speed_m_s` | number 0.01..1 | 0.1 | vo_input vio: the composed base speed disagreeing with the wheels' by more than this (m/s) for vio_lost_s while /zupt is silent marks the VIO lost (it never resets itself; a drift at 0.5 m/s would pass vo_max_speed) |
| `visual_odometry` | `vio_lost_s` | number 0.1..10 | 1.0 | vo_input vio: how long the wheel disagreement, or a feature count under vio_min_features while the base moves, must last before the VIO is lost, seconds |
| `visual_odometry` | `vio_min_features` | integer 0..500 | 20 | vo_input vio: fewer features than this for vio_lost_s while the base moves (/zupt silent) marks the VIO lost, never at rest; which features is vio_lost_rule: used, the ones OpenVINS's last update used (/ov_msckf/points_msckf + points_slam; 0 at rest with the head still, 0-8 in a head swing), or tracked, the tracker's persistent tracks on /ov_msckf/health (20-45 while moving outside a swing; drives 0343-0363 moving: 7 % of the frames under 20 outside a swing, 2.8 % under 10); under tracked also the update count the yaw-only bias walk restarts at; 0 turns the rule off |
| `visual_odometry` | `vio_max_speed_m_s` | number 0.1..10 | 1.0 | vo_input vio, the guard: a composed base velocity between two OpenVINS samples faster than this (m/s) is not sent (the wheels top out at 0.32; a diverged OpenVINS runs at metres per second) |
| `visual_odometry` | `vio_wheel_diff_m_s` | number 0.05..5 | 0.2 | vo_input vio, the guard: a composed base velocity farther than this (m/s) from the wheels' (forward speed, no sideways motion) is not sent; judged per sample, no duration (vio_lost_speed_m_s is the slow rule) |
| `visual_odometry` | `vio_restart_rejects` | integer 0..1000 | 20 | vo_input vio: this many guard rejections in a row (2 s at 10 Hz) with the wheels at rest for 2 s restart OpenVINS through /vio/restart (pepin_bringup.vio_keeper), at most once in 10 s; 0 never restarts |
| `visual_odometry` | `gate_exposure_s` | number 0..0.2 | 0.035 | half the exposure window the gaze gate judges a frame by: stamp +- this, because whether ustreamer stamps the start or the end of the exposure is not known; 0.035 covers an auto exposure up to half a 15 fps frame. With a manual exposure (config/camera.json's exposure block) set it to that exposure |
| `visual_odometry` | `gate_settle_s` | number 0..1 | 0.1 | how long after the head settled (the since of the phase it settled into, /gaze/state) a frame is still blind: the gaze contract's one frame period, 0.105 s at 9.5 fps; raise it if the mast is seen ringing after a saccade |
| `visual_odometry` | `gate_yaw_dps` | number 0..360 | 0.0 | a frame whose exposure window holds an IMU sample turning faster than this about base_link z (deg/s) is dropped like a saccade frame; 0 is off (as shipped: the base turns at most 57 deg/s and nothing has measured where its blur starts to cost) |
| `visual_odometry` | `gate_stamp_end` | integer 0..1 | 1 | 1: a frame's stamp is the END of its exposure (camera_stream's camera_stamp grab: the V4L2 capture stamp sits after the exposure) and the window is [stamp - 2 * gate_exposure_s, stamp + 10 ms]; 0: stamp +- gate_exposure_s, for ustreamer's send stamp whose relation to the exposure is unknown (1-68 ms late, 2026-10-02). Set it to 1 together with camera_stamp grab |
| `visual_odometry` | `gate_sway_dps` | number 0..360 | 6.0 | a frame whose window holds a mast sway rate above this (deg/s, the norm of /mast/state's velocities, the base bridge's head gyro minus base yaw and neck) is dropped as swaying; small sway is kept because the bridge's TF corrects it. 0 is off (as shipped: /mast/state does not exist before the head IMU); vio.md proposes 10 (the 5.3 Hz ring peaks at 6.7 deg/s, a knock is faster) |
| `visual_odometry` | `gate_sway_deg` | number 0..10 | 1.0 | a frame whose window holds a mast sway angle above this (deg, the norm of /mast/state's positions) is dropped as swaying: beyond what the TF correction is trusted for (a knock, a runaway). 0 is off (as shipped); vio.md proposes 1.0 |
| `visual_odometry` | `vio_gate_sway_dps` | number 0..360 | 20.0 | vo_input vio: the mast-sway rate (deg/s) over which a pose's frame is withheld, in place of gate_sway_dps (rtabmap's inputs and the depth consumers keep theirs): OpenVINS's velocity rides the IMU and the composition subtracts the sway through TF. Drives 0331-0363 replayed (scratch/vio_push/gate.py): the twists the 6 deg/s gate withheld scored a vx sigma multiplier 1.05 / 1.19 / 1.27 at 6-10 / 10-15 / 15-20 deg/s, 1.68 at 20-30, 3.0 beyond (the passed 0.80); at 20 / 1 the full twists per moving second 2.71 -> 5.26, the added ones s50 1.04, 1.2 % of them off the truth by > 0.2 m/s, 1.1 % of all sent (0.9 % before; scratch/vio_push/verdict.py); 0 is off |
| `visual_odometry` | `vio_gate_sway_deg` | number 0..10 | 1.0 | vo_input vio: the mast-sway angle (deg) over which a pose's frame is withheld, in place of gate_sway_deg; the 12 moving frames over 1 deg of drives 0331-0363 scored a vx multiplier 2.74, so it stays at the gate's 1; 0 is off |
| `visual_odometry` | `vio_coast_per_s` | number 0..10 | 0.5 | vo_input vio, vo_output twist: vx and vy sigma x min(1 + this * max(age - vio_coast_free_s, 0), vio_coast_max), age = seconds since OpenVINS's last update that used vio_min_features (/ov_msckf/health msckf + slam); the coasting filter's velocity drifts faster than its covariance says (drives 0331-0363, scratch/vio_push/coast_fit.py: vx s90 1.14 fresh, 1.86 at 1.0-1.5 s, 2.31 at 1.5-2.5 s, 2.57 past 4 s; with the law 1.05-1.27 per bin); 0 is off |
| `visual_odometry` | `vio_coast_free_s` | number 0..10 | 0.15 | vo_input vio: the coasting age (s) under which the sigma is not inflated: the update of the frame before (a pose is judged before its own frame's health arrives) |
| `visual_odometry` | `vio_coast_max` | number 1..10 | 2.25 | vo_input vio: the coasting multiplier's ceiling (the s90 past 2.5 s, 2.36-2.57, over the fresh 1.14) |

## Two recorders

Every drive is written down, two ways. Which one runs is `PEPIN_RECORDER` when Nav2 starts
(`PEPIN_RECORDER=bag ros/laptop.sh nav`, the `recorder` argument of `nav.launch.py`), and both
roads end in the same file, `ros/maps/rec/NNNN_<utc>Z_<goal>.jsonl`.

| | `PEPIN_RECORDER=jsonl` (default) | `PEPIN_RECORDER=bag` |
| --- | --- | --- |
| who writes | `pepin_bringup.run_recorder`, beside Nav2 in `pepin-macnav` | `ros2 bag record` (MCAP, no compression), started by `pepin_bringup.bag_recorder` |
| what lands there | the tape itself | `NNNN_<utc>Z_<goal>/` (MCAP) |
| the tape | already written | made by `ros/tools/bag_to_tape.py` in `pepin-vslam`, which `ros/goto.sh` runs for you |
| prelude | the 15 s before the goal (`pepin.tape.RunTape`) | none under `goal_bag record`; `preroll_s` (15 s) under `goal_bag ring` |

Both answer the same protocol (`pepin.runlink`): the goal server publishes `{"cmd": "start",
"name": ...}` on `pepin/run` and reads the run's number and path back from the latched
`pepin/run_status`. The bag records every topic the JSONL recorder subscribes to
(`pepin.tape_rows.TOPIC_RECORDS`) plus `/tf`, `/tf_static` and `/odom_laser`, with
`--include-hidden-topics` for Nav2's action status and a QoS override for the two latched ones
([`params/rosbag_qos.yaml`](params/rosbag_qos.yaml)). The converter rebuilds every record with the
live recorder's own functions (`pepin.tape_rows`); the `loc` records are composed from `/tf`
(`map -> odom` times `odom -> base_link`) at 5 Hz.

```bash
# by hand, if a bag was left without its conversion:
docker exec pepin-vslam /pepin_entrypoint.sh python3 /tools/bag_to_tape.py /maps/rec/0251_... --force
```

### The ring (`bag_recorder`'s `goal_bag ring`, off by default)

Under `PEPIN_RECORDER=bag`, `ros/flags.sh set bag_recorder goal_bag ring` makes the bag recorder
keep ONE `ros2 bag record` running for as long as it lives: the bag's topics plus OpenVINS's
`/ov_msckf/poseimu` and `/ov_msckf/odomimu`, one uncompressed MCAP file a minute (256 KiB chunks,
[`params/ring_mcap.yaml`](params/ring_mcap.yaml)) under `ros/maps/ring/<UTC start>/`, the oldest
files deleted past `ring_keep_h` (6 h), `ring_keep_gb` (10 GB) or under `ring_floor_gb` (15 GB)
free. A goal only marks its start and end: its bag is cut out of the ring,
`[goal - preroll_s, end + tail_s]` (15 s, 2 s), into the same `ros/maps/rec/NNNN_<utc>Z_<goal>/`
(`pepin.ring`, `pepin.bag_slice`, the `mcap` library: `ros2 bag` has no cut), with the latched
`/tf_static` and the last global costmap carried to its start. The bag appears whole about
`tail_s` + 1 s after the goal ends; `ros/goto.sh` waits for it before converting. Needs `mcap` in
the image (`ros/laptop-build.sh gaze`); without it the switch is refused. `goal_bag record` stops
the ring and makes each goal's bag the proven way.

### The board's own recording (`board_bag`, off by default)

Both recorders above write on the laptop, out of what crossed the WiFi, so a stalled link leaves
its hole in the drive's record. `ros/feature.sh board_bag on` adds a third, on the board:
[`pepin.board_bag`](../src/pepin/board_bag.py) keeps one `ros2 bag record` running for as long as
the stack does (one zenoh session, opened once) and writes one uncompressed MCAP file a minute
under `/root/pepin-ros/maps/board_rec/<UTC start>/`.

- **What**: `/scan`, `/odom`, `/imu/data_raw`, `/odom_laser`, `/zupt`, `/tof/<n>` and
  `/tof/<n>/scan`, `/neck/state`, `/tf_static`, and `/cmd_vel` and `/vo` as they arrived: every
  input of the board's EKF, so its output replays from them. No `/tf` and no camera.
- **The cap**: every 30 s the oldest minute files go while the directory holds more than 20 GB or
  the card has under 10 GB free; the file being written is never touched. 20 GB is ~38 h at the
  0.14 MB/s these topics measure.
- **Cost**, estimated on a network-isolated replica on the laptop and scaled by the EKF's
  board/laptop ratio: the recorder ~25 % of one A53 core and 70 MB, its supervisor ~0 % and 15 MB,
  niced 10. A census with it on replaces these numbers.
- **Reading it**: each minute file is a whole MCAP (`ros2 bag info FILE.mcap`); fetch with
  `rsync -a root@10.0.0.187:/root/pepin-ros/maps/board_rec/ ros/maps/board_rec/`.

## Replay

A costmap change is judged on the recorded drives before it is driven. `ros/replay.sh` runs the
recorded bags (`ros/maps/rec/NNNN_*/`, MCAP) through the stock `nav2_costmap_2d` local and global
costmaps, configured from `ros/params/nav2_params.yaml` and whatever is layered on top, and prints
one score row per drive.

```bash
ros/replay.sh 483-498                                       # the parameters as they are
ros/replay.sh 483-498 --set both.camera_layer.enabled=false \
    --against ros/replay/baselines/0483-0498.json           # candidate minus baseline, per drive
ros/replay.sh 483-498 --params candidate.yaml               # a params file layered last
ros/replay.sh 483-498 --save ros/replay/baselines/NAME.json # a new baseline (+ NAME.txt)
```

It runs in a throwaway container with `--network none`; `ros/replay/engine` (C++) is the
costmaps' clock and loop, so the same bag gives the same costmap bit for bit. The camera volume
is not re-fused (`/depth_marks` is replayed as recorded) and the loop is open: the recorded plan
and motion, no planner or controller run. The columns are in `ros/replay/score.py`.

### The camera of a drive (`ros/clip_to_bag.sh`)

The drives record no image topics; the recorder's clip `<run>_cam.mjpeg` beside each bag is the
camera's record (`pepin_bringup.camera_clip`, curl's raw copy, which keeps every part's headers;
its URL is `config/camera.json`'s stream on `PEPIN_HOST` and asks for `?extra_headers=1`, so the
V4L2 capture stamp rides along). `ros/clip_to_bag.sh`
turns it into a camera bag of the four stereo topics, rectified exactly as `camera_stream` does
(`pepin_bringup.stereo_frames`) and dated by the capture (`pepin.mjpeg.capture_time` grab):

```bash
ros/clip_to_bag.sh 0512                  # ros/maps/rec/0512_*_cam.mjpeg -> 0512_*_cam.bag
ros/clip_to_bag.sh 0512 --require-grab   # refuse a clip recorded without the capture stamps
ros2 bag play --clock 100 -i ros/maps/rec/0512_*/ -i ros/maps/rec/0512_*_cam.bag  # drive + camera
```

It prints the frames, the parts that fell back to the send stamp and the send-grab lag. The raw
rectified eyes are ~17 MB/s with MCAP's zstd (a 4-minute drive is ~4 GB): convert the drives being
replayed, not the archive. Clips recorded before 2026-10-03 carry no grab headers (send stamps,
1-68 ms late and bimodal): fine for looking, not for a VIO or a Kalibr run.

### The visual odometry's A/B (`ros/vio_replay.sh`, `ros/tools/vio_score.py`)

One drive, one arm, the board's EKF (`ros/params/ekf.yaml`) replayed on the drive's own inputs
plus the arm's `/vo`, in real time (OpenVINS has no ROS 2 serial reader) in a throwaway container
with no network; then every arm's `/odometry/filtered` scored against the lidar truth with the
metrics fixed before the drives (vio.md section 6: the median relative pose error per metre over
1 m segments, the paired bootstrap CI against B, the per-drive wins):

```bash
ros/vio_replay.sh 0601 --arm A      # the EKF with no /vo
ros/vio_replay.sh 0601 --arm B      # stereo_odometry + the relay (vo_input stereo): the baseline
ros/vio_replay.sh 0601 --arm E      # OpenVINS + the relay (vo_input vio); E0: without its ZUPT
uv run python ros/tools/vio_score.py runs/06* --arms A B E --baseline B --s3 0601 0602 0603 0604 0605 0606
```

Each run leaves `<run>_arm_<ARM>.bag` and `.csv` beside the drive; the scorer reads a directory per
drive holding `truth.csv` (the lidar truth) and one `<ARM>.csv` per arm.

## Simulation

A kinematic simulator for Nav2's behaviour on the Mac: `ros/sim.sh` runs our Nav2
(`nav.launch.py` included whole by `ros/sim/sim_nav.launch.py`) against a simulated cart in the
room RTAB-Map saved, and tapes every goal like a drive.

```bash
ros/sim.sh up                 # ~10 s to "Managed nodes are active"; the cart starts at home
ros/sim.sh goal printer       # or X Y [YAW_DEG]; prints goto's lines, then a SCORE line
ros/sim.sh scenario tour.yaml # ros/sim/scenarios: place, furnish, drive each leg, score it
ros/sim.sh place bookshelf    # teleport, standing still; both costmaps are emptied
ros/sim.sh up --rate 2        # the world's clock at 2x the wall, every node on use_sim_time
ros/sim.sh down               # gentle stop, logs kept in logs/containers
ros/sim.sh map                # re-export ros/sim/worlds/flat from ros/maps/rtabmap.db
```

It models the hull, the base's limits and deadman, and the LD19's raycast scan through its own
mount and filter (`pepin.sim`, `ros/sim/sim_world.py`). It does not model wheel slip, localisation
error, lidar noise, the camera, the ToF sensors, WiFi or the board's CPU. `ros/sim.sh` and
`ros/replay.sh` run this Mac's older `pepin-ros:latest` image, which still carries Nav2.

## What runs on the board

Four A53 cores and 1.5 GB, a sensor box since 2026-10-01. Everything the board may run is declared
once, with a budget and the reason it is on the board (real-time, survives a WiFi loss, wired to
its pins), in [`config/board_manifest.json`](../config/board_manifest.json). Nothing is added to
the board without an entry with a measured budget; a feature that only works while the laptop is
alive belongs on the laptop.

```bash
ros/board.sh census          # the manifest against a live ps: the table and the verdict; exit 1 when red
ros/board.sh census --json   # the same as data
ros/board.sh manifest        # the registry itself: what we run there and why (touches no host)
```

`ros/sync.sh` ends with a census; it never fails the deploy. A census is one `ps` over the
multiplexed ssh, with no `ros2` CLI and no `docker exec`. The verdict is red when anything is
**OVER** its budget, **MISSING** (expected always, not running), **FORBIDDEN** (declared
`expected: false` and running) or **UNLISTED** (a process above 1 % CPU that no entry claims,
usually a forgotten `ros2 topic hz`). Entries marked `sometimes` are **IDLE** when absent. ps's
`%CPU` is the average over the process's whole life and a percentage of one core (400 % is the
whole board).

### Laser odometry on the board

`rf2o_laser_odometry` matches each LD19 scan against the one before it (no map) and publishes
`/odom_laser` at 10 Hz, which the EKF fuses as `odom3`, twist only (`vx`, `vyaw`;
`ros/params/ekf.yaml` says why). It publishes no transform. `ros/feature.sh laser_odom on|off` is
the switch, on by default. It is built into the board image from a pinned commit of the MAPIRlab
ROS 2 port with `ros/patches/rf2o-base-twist.patch` (upstream publishes the twist in the laser's
frame, an empty covariance and two `INFO` lines per scan). Its variances
(`pepin.deployment.LASER_ODOM_TWIST_VARIANCE`) are sized to be quiet until a drive measures them:
push the cart forward by hand (`/odom_laser`'s `vx` positive and near `/odom`'s) and turn it left
(`vyaw` positive and near the gyro's), because the LD19 hangs upside down and yawed and a sign
error here is the one failure that would matter.

## Building the image

The board's image is `pepin-ros`, built from `ros/Dockerfile` on the Mac (the same arm64
architecture) and loaded on the board over ssh. It carries the sensor stack only: no Nav2 (the
lidar's lifecycle manager is the one Nav2 package in it), no slam_toolbox, no rviz.

```bash
ros/build-image.sh                    # build pepin-ros:sensors here; the board untouched
PEPIN_BUILD_CPUS=4 ros/build-image.sh # the same on 4 of the Docker VM's CPUs at low weight
ros/build-image.sh --ship             # build, then load it on the board (stack stopped first)
ros/build-image.sh --ship-only        # load the image already built here
ros/laptop-build.sh                   # the laptop's image on top of it, with Nav2 and RTAB-Map
ros/laptop-build.sh gaze              # pepin-laptop:gaze: that image plus the AskGaze BT node
```

The load tags it `pepin-ros:latest` and `pepin-ros:zenoh` on the board, the names `ros/run.sh`
and `pepin-zrouter.service` ask for; the board's image from before the first sensors-only load
keeps the tag `pepin-ros:pre-sensors-2026-10-01`, and the script's header has the one line that
puts it back. A ship is refused under a running stack (unless `--force`), on a board whose
`/etc/default/pepin-ros` still says `PEPIN_NAV=true` or `PEPIN_SLAM_TOOLBOX=true`, and under 2 GB
free on docker's root. It installs `board/pepin-ros.service` (daemon-reload, no restart) and
prints the restart. Code, `params/ekf.yaml` and `config/` travel with `ros/sync.sh`. The rf2o
layer's compiler count is the `RF2O_JOBS` build argument (a `-j4` build put the board into swap
once).

## The XFeat image

`pepin-laptop:xfeat` is the laptop image with RTAB-Map able to run `rtabmap_frame`'s
`visual_features` default: XFeat keypoints matched by LighterGlue in the visual registration. It
is `pepin-laptop:zenoh` plus RTAB-Map 0.22.1 and the rtabmap_ros packages rebuilt from pinned
commits with Python (the apt build has none), XFeat and LighterGlue at a pinned commit with their
weights, the two adapters RTAB-Map loads from `/opt/xfeat` (`ros/xfeat/`), and
`ros/patches/rtabmap-keep-global-descriptors.patch`.

**Build**: `ros/laptop-build.sh xfeat`, about an hour, 2 jobs per phase (`PEPIN_CORE_JOBS`,
`PEPIN_ROS_JOBS`), cancelled once more than 75 % of the VM's memory is in use
(`PEPIN_BUILD_MAX_USED_PCT`): the VM also runs the live stack, and nine parallel compiles once got
the stack's processes OOM-killed instead of the compilers.

**Run**: `ros/laptop.sh vslam` (and so `ros/restart.sh laptop`) starts `pepin-vslam` on
`pepin-laptop:xfeat` whenever that image exists and was built on the current laptop image;
otherwise the usual image runs, `visual_features` falls back to ORB and the script says so.

| setting | image | visual registration |
| --- | --- | --- |
| unset | `pepin-laptop:xfeat` if built on the current base, else the base with a stderr line | xfeat, else ORB |
| `PEPIN_XFEAT=0` | the base (`pepin-laptop:zenoh`), the rollback | ORB, and check 2.12 passes |
| `PEPIN_XFEAT=1` | `pepin-laptop:xfeat`, or the command refuses | xfeat |
| `PEPIN_IMAGE=<tag>` | that tag, whatever the above | xfeat if the tag carries `/opt/xfeat` |

`ros/laptop.sh vslam` mounts the checkout's two adapter files over the image's, so an adapter
change needs a vslam restart and no rebuild (`PEPIN_ADAPTERS_MOUNT=0`: the image's). **Known
broken in this image**: the GUI parts (`rtabmap_viz`, the rviz plugins, the `rtabmap` /
`rtabmap-databaseViewer` desktop tools) still link apt's `librtabmap_gui`, built against the old
core; nothing in `vslam.launch.py` starts them. Open a database with the tools of
`pepin-laptop:zenoh`.

## The model services

Docker on macOS has no GPU, so the learned models run on the laptop as two host processes (two
failure domains), which the containers reach as `http://host.docker.internal:<port>` (bound to
127.0.0.1):

| job | process | port | models (device from `config/models.json`) |
| --- | --- | --- | --- |
| `depth` | `pepin.depth_service` | 8790 | the mono depth network, RAFT-Stereo (as `ros/depth_host.sh`) |
| `localization` | `pepin.localization_service` | 8791 | XFeat (`/xfeat`, cpu), LighterGlue (`/match`, cpu), BoQ-DINOv2 place descriptors (`/place`, mps) |

Both are launchd jobs run from the repository's uv environment and loaded on demand: their plists
live in `~/Library/Application Support/pepin/launchd` (`PEPIN_LAUNCHD_DIR`), so nothing loads them
at login. While a job is loaded launchd restarts it when it dies. `ros/laptop.sh vslam` starts
both and `ros/laptop.sh stop` stops them.

| command | what it does |
| --- | --- |
| `ros/models.sh install [depth\|localization\|all]` | write the job's plist and start it; waits for `/health` |
| `ros/models.sh start\|stop\|restart [...]` | load / unload / kill-and-respawn the job |
| `ros/models.sh status [...]` | launchd's word and each `/health`: every model's tag (its weights' hash), device, requests and ms |
| `ros/models.sh logs [depth\|localization]` | follow `logs/models_<job>.out` |
| `ros/models.sh fetch-xfeat` | clone XFeat at the commit the image pins into `models/accelerated_features` |
| `ros/models.sh uninstall [...]` | stop and remove |

A model that fails to build does not take its process down: its endpoint answers 500 with the
reason and `/health` names it (`failed: ...`); restart check 2.13 reads it. A place vector that is
not a finite unit vector is a 500 too (one NaN aborts RTAB-Map). The localization job takes an
XFeat checkout only at the commit `ros/Dockerfile.xfeat` pins (`PEPIN_XFEAT_UNPINNED=1` takes
another). RTAB-Map's XFeat adapters are the service's clients: `rtabmap_frame`'s
`registration_backend` (`service`, `local`, `auto`) is written to
`/tmp/pepin/registration.json` in the container, and `auto` falls back to the in-process
computation while the service is down.

**Place descriptors.** With `place_recognition descriptor` RTAB-Map finds which node a picture is
by the dot product of BoQ descriptors instead of its ORB words. It aborts when it compares a node
that carries one with a node that does not, so every node carries exactly one:

1. `sensor_pack` attaches one to every snapshot (`global_descriptor auto`: only on an RTAB-Map
   built with the patch): the service's `/place` vector for a camera snapshot, the null
   descriptor (zeros) for a lidar-only one or any failure on the way;
2. `ros/tools/place_backfill.py ros/maps/rtabmap.db` writes one into every node a database already
   holds (the service up, vslam stopped, a timestamped backup first); `--check` prints the census;
3. `vslam.launch.py` takes that census on every start, before RTAB-Map opens the file, and hands it
   to `rtabmap_frame` (`PEPIN_PLACE_CENSUS`), which sends the descriptor likelihood only when the
   census, the snapshots and the RTAB-Map build all say nothing can abort it, and only while the
   recent camera snapshots were described (`descriptor_null_share`); the words otherwise, with
   the reason in its report line;
4. with descriptors on board the launch raises `Mem/RehearsalSimilarity` from 0.30 to 0.92, the
   value at which consecutive daylight pairs are called the same place as often as the words did.

RTAB-Map 0.22.1 drops a node's descriptor the first time it registers against it and then aborts
on the next comparison; the patch keeps them, and `ros/laptop-build.sh xfeat` leaves the marker
`/opt/rtabmap_patches/keep-global-descriptors` that `rtabmap_frame` requires. RTAB-Map scores
places only for a frame with ORB words, so a frame too dark for a single corner is not
recognised by descriptor either.

## Foxglove

The bridge is the laptop's (`pepin-vslam`, `ws://localhost:8765`); the board runs none. Install
the app with `brew install --cask foxglove`. Layouts live in `ros/foxglove/` (`pepin_slam.json` is
the driving one); send a goal with the "Publish" panel on `/goal_pose` (`geometry_msgs/PoseStamped`,
frame `map`).

| command | what it does |
| --- | --- |
| `ros/foxglove.sh check` | one `PASS`/`FAIL` line each: `pepin-vslam` is up, port 8765 accepts, the websocket handshake succeeds, every topic the layout draws is advertised, and how many times the bridge has died since the container started (restart check 2.9) |
| `ros/foxglove.sh reopen` | tells the running desktop app to reconnect, unless it already did; prints the link when the app is not running |
| `ros/foxglove.sh url` | prints that link |

A Foxglove client is bound to ONE bridge process: channel ids start again at 1 when it restarts,
so a recreated `pepin-vslam` leaves the app's panels empty until it reconnects. The scripts print
the link and never open a connection on their own (each deep link opened another, ~26 MB/s
through Docker's proxy each).

## At boot

`board/pepin-zrouter.service` starts the board's zenoh router and `board/pepin-ros.service` the
sensor container (`bringup.launch.py`, which is `robot.launch.py` under the stop window) after the
base and ToF servers, with the switches of `/etc/default/pepin-ros` (`ros/feature.sh`;
[board/README.md](../board/README.md) has the table). Nothing of navigation starts on the board.

```bash
# by hand instead of the unit (systemctl stop pepin-ros first):
ssh root@pepin.local '/root/pepin-ros/run.sh ros2 launch pepin_bringup robot.launch.py'
```

## Bring-up checklist (in this order, each step visible in Foxglove)

1. The board's stack alone: `/scan` at ~10 Hz (the hull box filter's output), `/odom` at 50 Hz,
   `/imu/data_raw` at 100 Hz, TF `odom -> base_link -> laser`. Push the cart forward by hand:
   `/odom` x grows. Turn it left: theta grows.
2. Laser orientation: a wall in front of the cart must draw at +x in `base_link`. The lidar is
   mounted upside down and the LD19 counts angles clockwise; the static transform (roll pi, yaw
   -87.5 degrees) is read from `config/lidar.json` by the board's launch and the laptop's camera
   node alike. If the scan comes out mirrored, fix `roll_deg` there and `ros/sync.sh --restart`.
3. `ros/laptop.sh vslam`: `/map` arrives and restart check 1.13 says `map -> odom` is corrected
   (`ros/goto.sh seed X Y YAW` if RTAB-Map has recognised nothing yet).
4. `ros/laptop.sh nav`, `ros/preflight.sh`, then a short goal.

## Redundancy demo

Both sensors see the room, and either one alone can feed the costmaps, switched while the robot
runs. `ros/sensor.sh` is one command per sensor: what writes into the costmaps (`lidar_layer`,
`camera_layer`, `contact_layer`, on the local and the global costmap). Localisation is RTAB-Map's
either way; what its snapshots carry is `sensor_pack`'s `sources` flag.

| mode | command | costmap layers |
| --- | --- | --- |
| fused | `ros/sensor.sh lidar on` + `ros/sensor.sh camera on` | `lidar_layer`, `camera_layer`, `contact_layer` |
| lidar only | `ros/sensor.sh camera off` | `lidar_layer` |
| camera only | `ros/sensor.sh lidar off` (`--hard` to stop the driver) | `camera_layer`, `contact_layer` |

The camera is one sensor read twice: `depth_scan` is the band 8 cm-1.3 m above the floor (table
tops, seats, a hand) and `contact_scan` is where the floor ends (chair feet, a plinth), so
`camera on` moves two layers at once. `ros/sensor.sh status` lists the layers and what each camera
node last said; `ros/sensor.sh mute|unmute imu|odom|vo|lidar` silences a sensor where it is
published ("Feature flags"). The soft `lidar off` is an ignored scan; `--hard` deactivates the
driver (`ros2 lifecycle set /ldlidar_node deactivate`) and only `ros/sensor.sh lidar on` brings
it back. `--hard` first asks `ros/tools/nav_goal_running.py` and refuses while a goal runs.

Watch it with `ros/foxglove/pepin_nav.json`: `/scan`, `/depth_scan` and `/depth_marks` beside both
costmaps; switching a layer changes the grid within a costmap cycle.

## Frames and conventions

`base_link` sits between the drive-wheel contact points: x forward, y left. Footprint (metres,
`base_link`): front 0.0625, rear 0.30, half-width 0.275, from `config/base.json`.

`odom -> base_link` is planar and stays planar (`ros/params/ekf.yaml`, `two_d_mode`). The few
degrees the body leans over a threshold are a separate number, estimated by `pepin.lean` from
`/imu/data_raw` (the accelerometer for the slow truth, the gyro for the fast part) and composed on
base_link's side of the planar pose by `pepin.frame_pose.FramePoser` behind each node's `imu_lean`
flag. Roll is positive right side down, pitch positive nose down. Every node that estimates it
prints it in its report line (`lean +0.3/-1.8 deg q0.94 bias 0.05 deg/s`); below
`lean_min_quality` (the share of the lean gravity itself voted for) the measurement is placed
level, because a gyro whose zero has drifted reports a tip nobody made. The zero of roll and
pitch and the gyro's starting bias come from `config/imu.json`'s `level` block, measured over
30 s at rest on a level floor (2026-09-12).
