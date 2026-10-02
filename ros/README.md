# Pepin on ROS 2 Jazzy + Nav2 (branch `ros2-nav2`)

The navigation stack moves to Nav2; everything ROS runs **on the board** in one Docker
container (the board is Armbian Debian trixie, which has no Jazzy binaries; the container
is Ubuntu 24.04 with the official `ros:jazzy-ros-base` image). The laptop only watches,
through Foxglove Studio. Reason: the wifi to the board has 300-700 ms latency spikes; a
20 Hz controller on the laptop would drive through them the way our first wheel loop did.

What stays from the Python stack:

- `pepin.base_server` on the board still owns the wheels (50 Hz, 0.5 s deadman). The ROS
  node `base_bridge` maps `/cmd_vel` to it and publishes `/odom` and the `odom -> base_link`
  transform from its state stream.
- `pepin.tof_server` feeds `tof_bridge`, which publishes three `sensor_msgs/Range` topics.
- Saved maps (`data/maps/*.npz`) convert to Nav2 maps with `ros/tools/npz_to_map.py`.

```
laptop (Mac)                      board (Orange Pi Zero 3, 1.5 GB + zram)
Foxglove Studio                   docker: ldlidar_node -> laser_filters box filter (/scan), base_bridge
  ^ ws 8765, foxglove_bridge ON THE LAPTOP  (/odom, tf), tof_bridge, Nav2 (amcl, costmaps,
                                          planner, controller, bt_navigator), slam_toolbox
                                  host:   pepin-base.service (:3336), pepin-tof.service (:3335),
                                          ser2net (:3333 servo bus for bench tools only)
```

## Layout

| Path | What |
| --- | --- |
| `ros/Dockerfile` | Jazzy base + nav2, nav2-bringup, slam-toolbox, foxglove-bridge, CycloneDDS; LD19 driver ([Myzhar/ldrobot-lidar-ros2](https://github.com/Myzhar/ldrobot-lidar-ros2)) built from source; our `pepin_bringup` |
| `ros/run.sh` | `docker run` with host networking, the lidar device and `ros/maps`, `ros/params` mounted |
| `ros/pepin_bringup/` | ament_python package: `base_bridge`, `tof_bridge`, launch files |
| `ros/pepin_base_cpp/` | ament_cmake package: the same base bridge in C++ (`base_bridge_cpp:=true`), ~25 MB instead of ~190 MB |
| `ros/params/` | Nav2 parameters for this cart (footprint, speeds, rates for a weak CPU) |
| `ros/maps/` | Converted maps (`<name>.pgm` + `<name>.yaml`) |
| `ros/tools/npz_to_map.py` | Our occupancy grid -> map_server format |
| `ros/calibrate.sh` | Checkerboard calibration of the neck camera, print to config (see below) |
| `ros/replay.sh`, `ros/replay/` | The recorded drives through Nav2's own costmaps with candidate parameters, one score per drive (see "Replay") |
| `ros/sim.sh`, `ros/sim/` | The kinematic simulator: our Nav2 on a simulated cart in RTAB-Map's saved room ("Simulation") |

## Deploying a change

The containers mount the code from the host (`run.sh` on the board, `laptop.sh` here), so a
Python change needs no image, only the processes that hold the old code restarted.

| Change | Command | What restarts |
| --- | --- | --- |
| Python in `src/pepin` or `ros/pepin_bringup/pepin_bringup` | `ros/push.sh FILE...` | the running nodes that import it, on both halves |
| a launch file, `ros/params`, `config/`, a module a launch file imports | `ros/restart.sh board --deploy`, `laptop` or `both --deploy` | the half's whole stack |
| the whole tree, nothing restarted | `ros/sync.sh` (`--restart`: the board's stack too) | nothing |
| one node by hand | `ros/thin.sh kick NODE`, `ros/laptop.sh kick NODE` (no name: the list) | that node |
| the Dockerfiles, the C++ packages, rf2o's patch | `ros/build-image.sh`, `ros/laptop-build.sh` | see **Building the image** |

`ros/push.sh` takes its plan from `pepin.push`: the nodes whose Python imports a changed module
through any chain of imports (`uv run python -m pepin.push plan FILE` prints each chain). Whatever
a kick cannot deliver is refused before anything is touched, with the restart that delivers it: a
launch, params or unit file, config, a module a launch file imports, an image layer, a process of
ours that the launch does not respawn (checked in its container: refused only while it runs), a
laptop container that mounts another checkout. Otherwise the files go to the board by rsync —
exactly those, never `--delete` — and the nodes of both halves are kicked at once, one line each:
kicked at, ready at (UTC, the container's clock), seconds, the old and the new pid, the ready line.
A node that is not running on its half (the other recorder, the goal server's other side) is
skipped. `--dry-run` or `PEPIN_PUSH_DRY=1` prints the plan and what
would run, and touches nothing.

A kick ends the node with SIGINT, the launch respawns it from the new sources two seconds later,
and the kick waits for the new pid's own ready line (`ros/kick_ready.awk`). The node is gone for
those seconds: push at rest, not mid-drive.

The laptop's SLAM container is its own: `ros/laptop.sh vslam` restarts it with the RTAB-Map
database kept, `ros/laptop.sh vslam --fresh` deletes the database first and starts an empty map
(in SLAM mode the session starts empty anyway: see below).

## Restarting

`ros/restart.sh board|laptop|both [--deploy] [--fresh-graph] [--no-check] [--dry-run]` brings the
robot up working in one command and ends on one line — `green: N checks, none failed`, or `red:`
with the failing lines above it — and that line is all the operator reads. It restarts the board's
stack together with its zenoh router (stack stopped, router restarted, stack started; `--deploy`
does it through `ros/sync.sh --restart`), then the laptop: board first, always, because a board
restart re-zeroes the odometry RTAB-Map runs on (`board` alone restarts only the laptop's vslam, and
only if that half is up). Then it proves the planner plans (4.1) and repairs it when it does not —
the Nav2 container alone, then the board half once more — before it fails; a planner that answers
"no path" in every direction is reported, never restarted. The board's Nav2 waits for the laptop's
map however late it comes (`initial_transform_timeout`), so the start order is no longer a race.
`--dry-run` prints the order and touches nothing, `--no-check` restarts only, `--fresh-graph` starts
the camera half on an empty RTAB-Map database and moves the old frame's volume aside;
`PEPIN_RESTART_WAIT_S`, `PEPIN_RESTART_POLL_S` and `PEPIN_PLANNER_WAIT_S` change how long a half
and the planner are waited for.

Every check is one `PASS`/`FAIL` line with its number, a `WARN` is shown but never fails the run,
and the script exits 1 if anything failed. A check that cannot be answered says so; it never reads
silence as good news. What is checked:

| # | the planner |
|---|---|
| 4.1 | from `pepin-vslam`, `ros/tools/planner_check.py`: `/global_costmap/costmap` (or its updates) delivers something new within 10 s — its latched copy proves nothing — and one `ComputePathToPose` from the cart's pose to 0.5 m ahead (then behind, left, right) returns a path: a plan, never motion. Asked again for up to 180 s while Nav2 comes up, repaired before it fails (above). "No path" in every direction on a corrected pose (`ros/tools/map_odom.py`) fails with no restart: the cart is boxed in or the costmap is full of marks. With the laptop half down it fails: there is no map to plan on |

| # | board |
|---|---|
| 1.1 | `ros/board.sh census`: every process accounted for, every budget kept |
| 1.3 | the pose: `ros/go.sh where` — the goal server's socket, whose answer is composed from `map -> base_link` — answers and says `"pose": "tf"` |
| 1.4 | no `Failed to meet update rate` in the last 60 s |
| 1.5 | no `Extrapolation` / `out of map bounds` / `Off Grid` in the last 60 s |
| 1.6, 1.7 | `/depth_scan` and `/vo` really reach the board (`ros/tools/topic_rate.py`, one 5 s measurement each — not `ros2 topic hz`, which costs ~4.5 s of A53 before it measures anything) |
| 1.8 | `pepin-base` is active and no `torque on` is left standing in its journal |
| 1.11 | Nav2 is **active**, not merely running: the lifecycle manager got `planner_server connected with bond`, and the log carries zero `Range sensor layer can't transform` lines. A `planner_server` that activated and never bonded is wedged inside its global costmap's first update — tf2's `canTransform` costs a whole `transform_tolerance` per untransformable Range and the three ToF layers deliver 15 Hz each, so the backlog outgrows the drain and the update never ends (`scratch/nav2_hang/wedge_gain.py`; the fix was to stop feeding that plugin — the whiskers are `ObstacleLayer`s now, which drop what they cannot place). Goals are then accepted and nothing is planned. A board that runs no Nav2 is a `WARN`, never a failure. Since 2026-09-21 no costmap lists a `RangeSensorLayer` at all (the whiskers arrive as scan fans, `tof_bridge`'s `range_as`), so one of those lines now means the board is running a `nav2_params.yaml` older than this checkout — still worth a `FAIL` |
| 1.13 | **who is correcting the pose**: `map -> odom` is in TF and read where its publisher is — one rclpy node in the laptop's own container (`ros/tools/map_odom.py`), never on the board, whose /tf would cost it ~100 messages a second (CLAUDE.md rule 20). Two readings: the transform is **fresh** (re-broadcast at 20 Hz, so seconds of silence is a publisher that is gone) and it is **not the identity** (a localiser that has recognised nothing publishes `map == odom`, and every pose composed from it is simply the odometry's). The identity is a `WARN` while the laptop half is under 60 s old and a `FAIL` after that |
| 1.12 | no thread of the Nav2 container is pegged: `ps -L` over ssh, the busiest thread's cumulative CPU time over the process's own lifetime. `range_sensor_layer.cpp:362-369` clamps its cell bounds and then walks them as `unsigned`, so a cone that falls off the grid's left or bottom edge runs ~4e9 iterations under the costmap mutex and writes **no log line at all** — one thread at 100 %, "Pose Goes Off Grid", services timing out, zero plans (reproduced 2026-09-21 with a kick of the board's old tracker: tid 191, 415 s of CPU in 700 s). `FAIL` above 0.90, `WARN` above 0.50 (nobody has yet measured what a healthy container's busiest thread costs — tighten it once a few restarts have printed theirs), `WARN` when the board could not be read |
| 1.14 | `/odom_laser` is flowing (the EKF's `odom3`), measured the same way. Skipped with a `WARN` when `PEPIN_LASER_ODOM=false` on the board |
| 1.15 | **informational (`PASS`/`WARN`, never fails, never a drive gate)**: the board's clock minus the laptop's — the Docker VM's, which every laptop ROS node stamps with — over NTP from the board to the laptop's time server (`ros/time.sh offset`: `scripts/timesync.py` piped into the board's `python3`, best of eight round trips). `WARN` over `PEPIN_CLOCK_WARN_MS` (100 ms), `WARN` "not measured" when the board cannot reach the server or `PEPIN_TIME_SOURCE=laptop` and no server runs here, `WARN` on a value that is neither `laptop` nor `pool`. Under `PEPIN_TIME_SOURCE=pool` (the default) with no server here it is a `PASS` "not measured, as configured": that is the configuration, not a fault. See "One clock" below |

`ros/tools/coldstart_soak.sh [N]` is the acceptance test behind checks 1.11 and 1.12: N cold starts of the
board half (10 by default), each timed from `Activating planner_server` to the bond, with the
range-layer and `Invalid frame ID` counts beside it, one row per start and a non-zero exit unless
every start passed. It restarts processes and reads logs — **the robot does not move**, and it
refuses to begin while a navigation goal is running. The hang appeared on 4 of 7 starts on
2026-09-21, which is why one green restart is not an answer.

| # | laptop |
|---|---|
| 2.2 | `depth_stream` over 5 frames/s, with a fitted law in its line |
| 2.3 | `depth_fusion` over 5 frames/s, `at bound 0` |
| 2.4 | `visual_odometry` over 5 poses/s from rtabmap |
| 2.6 | the rtabmap process is alive in `pepin-vslam` |
| 2.7 | `sensor_pack` feeds RTAB-Map: its snapshots per second, over 0 |
| 2.10 | `rtabmap_frame` hears RTAB-Map: `N updates` with N > 0 in its report line |
| 2.8 | no `process has died` in the container since it started |
| 2.9 | Foxglove: the bridge answers on `ws://localhost:8765` and advertises every topic the layout draws (`ros/foxglove.sh check`; its failing lines are indented under this one) |
| 2.11 | **informational (`WARN`, never fails)**: `marks_audit`'s last line — the local costmap's lethal cells split into lidar-backed, camera-only and unexplained. There is no healthy value (a room with a table in it should show camera-only cells); it is printed so the split is in front of you before the first goal |
| 2.12 | `pepin-vslam` carries the XFeat adapters (`/opt/xfeat/rtabmap_xfeat.py`), so `rtabmap_frame`'s `visual_features` default can run; on any other image the visual registration is ORB's. Passes on such an image only under `PEPIN_XFEAT=0`, which says ORB is meant (see "The XFeat image") |
| 2.13 | the localisation service answers `/health` on `:8791` with `xfeat`, `match` and `place`, none of them `failed: ...` (see "The model services"). Down is a `WARN` (RTAB-Map's adapters compute in its own process and the snapshots carry null place descriptors), said loudly under `place_recognition descriptor`, which is then back on the words until the service describes again, and a `FAIL` only under `registration_backend service`, where a registration it does not answer finds no features |

| # | flags |
|---|---|
| 3.x | every live flag of the restarted half is its `FLAGS` table default (`ros/flags.sh drift`). A difference is a `WARN`, not a failure: a flag set on purpose is legitimate, but a restart puts every flag back to its default, so this is where a switch you meant to keep shows up as gone. |

## One localiser

RTAB-Map on the laptop owns `map -> odom` (since 2026-09-22): it publishes the transform itself
(`publish_tf`, re-broadcast at 20 Hz, stamped 0.5 s ahead) and every consumer on the board composes
it with the board's own `odom -> base_link`. Both costmaps' static layer reads `/map`, RTAB-Map's
grid relayed by `pepin_bringup.rtabmap_frame`. RTAB-Map starts **localising** on a loaded database.
The board's scan-matching tracker that owned the edge before, its laptop-side watchdog and the
message-path owner (`slam_frame`) are on the tag `alt/tracker-2026-09-22`, whose message says how
to bring them back.

**Why.** Two owners of the truth is a race, not a redundancy. On 2026-09-21/22 the tracker
trusted its own whole-map search on a fragment grid (fit 0.96 on the wrong place), collapsed its
sigma and gated RTAB-Map's correct words out; the pose jumped 3.4 m. What belongs on the board is
**odometry** — wheels, gyro, visual odometry, laser odometry — because that is what must survive a
WiFi loss and close a loop in milliseconds. `map -> odom` is a slow correction every consumer
composes with `odom -> base_link`.

**The zero-velocity update.** The tracker was also the only publisher of the EKF's zero-velocity
input (`/zupt`, `odom2` in `ros/params/ekf.yaml`), so a parked cart's filter heard nothing but its
sources' own drift and its heading followed rf2o's +1.5 deg/min at rest: ~5 deg an hour
(2026-09-24). The C++ base bridge now publishes that input itself while its own rest witness says
the cart is certainly still — wheels at rest past `imu_bias_s`, no fresh `/cmd_vel`, the gyro
quiet — and nothing otherwise (`base_bridge` `zupt_publish`, "The base bridge's zero-velocity
update" below).

**Why localising and not mapping.** A start in mapping mode opens a new session per restart, and
the grid RTAB-Map publishes is the connected component of the *current node inside working
memory* (`Rtabmap.cpp:3941/4111/5444`) — so seventeen sessions from one evening's restarts made
the map the costmaps read change thirty times in 800 s. Localising writes nothing, so no restart
can add a session. The price is that a loaded map is not extended; `ros/laptop.sh vslam --fresh`
starts an empty one.

**What the first seconds look like — UNVERIFIED.** Read from rtabmap_ros's sources rather than
measured: `CoreWrapper` broadcasts `mapToOdom_`, which is initialised to the **identity** and
replaced only when a localisation or a graph optimisation lands. So the expectation is that a
start on a loaded database publishes `map == odom` — the pose is the odometry's — until RTAB-Map
recognises a node, *not* the last saved correction. Nobody has watched this on the robot yet.
Check 1.13 and the procedure below are where it gets measured.

### Parked acceptance, 120 s

With the cart parked where it can see the room, both halves up, and **no goal sent**:

1. `ros/restart.sh both` — 1.13 must end green (`corrected`, stamped well under 2 s ago). Note
   how long after the start it stopped saying `identity`: that is the answer to the paragraph
   above, and it belongs in the journal.
2. `docker exec pepin-vslam /pepin_entrypoint.sh python3 /tools/map_odom.py 5`, three times over
   two minutes — the shift must not wander between readings on a cart that is not moving.
3. `docker logs pepin-vslam | grep 'rtabmap frame:'` — **at most two** distinct map ids in the
   window. More than that is the churning grid this switch exists to stop, and means the session
   is not localising after all.
4. `ros/go.sh where` — `"pose": "tf"`, no `fit` in the answer, and coordinates that match where
   the cart really stands to a few centimetres.
5. `ros/restart.sh board` alone, then Nav2: `planner_server connected with bond` and no
   `Pose Goes Off Grid` — the board must come up and plan with the transform arriving from the
   other machine.
6. `ros/board.sh census` — the board's CPU. The old tracker measured 60 % of a core standing
   still; this is where that shows as freed.

## The zenoh routers

Under `rmw_zenoh_cpp` there is no bridge at all: one `rmw_zenohd` router per machine — the board's
`pepin-zrouter` (a systemd unit, `board/pepin-zrouter.service`, `--network host`) and the laptop's
`pepin-zrouter-laptop` (`ros/laptop.sh`'s `zrouter_up`) — joined by ONE TCP link that the laptop
dials (`ros/lib.sh`'s `pepin_zenoh_router_override`). Every node is a zenoh peer of its own
machine's router; only the router-to-router link crosses the WiFi.

**The link is now allowed to stall without dying.** Measured on 2026-09-22: the board's radio
(unisoc uwe5622, -35..-42 dBm on 5785 MHz) pings the laptop at 80-180 ms on average with spikes of
0.4-1.2 s **with the whole stack stopped** — a driver/radio property, not our load. A ROS RELIABLE
publication is pushed to zenoh as `CongestionControl::Block`, and the shipped router config gives
such a push 5 s to find a free batch before it **closes the whole transport session**: `Unable to
push non droppable network message to <zid>. Closing transport!`, 8-12 times a minute, each one
followed by seconds of re-discovery with no `/tf`, no `/map` and no services. So both routers now
start on `ros/zenoh/router.json5`, passed as `ZENOH_ROUTER_CONFIG_URI` (the variable
`rmw_zenohd` itself reads; `ZENOH_CONFIG_OVERRIDE` is applied **on top** of it, which is why the
laptop's `connect/endpoints` still lives there and one file serves both sides). Three values,
all under `transport/link/tx`:

| key | shipped | ours | what it does |
| --- | --- | --- | --- |
| `queue/congestion_control/block/wait_before_close` | 5 s | **20 s** | how long a RELIABLE push waits for a free batch before killing the session. This is the one that stops the closures; 4x the worst measured stall. A peer that is truly gone is buried by the close it triggers — provided the router has an RX worker free to run that close (see "A frozen peer" below). |
| `queue/congestion_control/drop/wait_before_drop` | 1 ms | **50 ms** | how long a BEST_EFFORT sample waits before being dropped. 50 ms is the period of the fastest thing that crosses, so a sample is never held past its own successor. |
| `keep_alive` | 2 | **4** | keep-alives per `lease` (60 s, unchanged): every 15 s instead of 30, which is what zenoh's own note asks for on a link that loses packets. |

The queue **depth** (`queue/size`, 2 batches = 128 kB per priority) is deliberately untouched:
even zenoh's maximum of 16 is 1 MB, a few hundred milliseconds of this link, so it cannot hold a
2 s stall — and it would cost the 1.5 GB board memory on every link its router holds. The file is
a full copy of the image's `DEFAULT_RMW_ZENOH_ROUTER_CONFIG.json5` with every change marked
`PEPIN`, because `ZENOH_ROUTER_CONFIG_URI` *replaces* the configuration instead of merging into
it; its header has the one-line `diff` that shows our delta.

**The way back is the variable, unset.** Neither starter needs a rebuild: on the laptop
`PEPIN_ZROUTER_CONFIG= ros/laptop.sh vslam` starts the router on the shipped default, and on the
board the unit mounts the file only when `/root/pepin-ros/zenoh/router.json5` exists (deleting it,
or simply not syncing it, is the same switch). The file reaches the board through `ros/sync.sh`,
which rsyncs the whole of `ros/` to `/root/pepin-ros/`.

**The node SESSIONS keep the shipped default**, and on purpose. Their `wait_before_close` is
already 60 s (`DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5` raises it where the router's stayed low),
and no session link crosses the WiFi: the board's nodes are peers over one loopback, the laptop's
over `pepin-net`, and the radio hop belongs to the routers alone. If a measurement ever shows a
session closing, the same file is passed to the containers as `ZENOH_SESSION_CONFIG_URI` — the
mechanism is identical — but nothing has asked for it.

**Deploying a change to `ros/zenoh/router.json5`**: `ros/sync.sh` (no restart), then the routers
in the clean order (journal 2026-09-13): laptop half down, laptop router down, board router, board
stack, laptop router, laptop half. A router restarted under a live peer has produced a one-way
link before.

**Every container's log outlives its restart** (`PEPIN_LOG_ARCHIVE`, `on` by default, `off` keeps
nothing as before). On the board, `pepin-ros` and `pepin-zrouter` are started without `--rm`, and
each unit's next start copies the previous container's log to `/root/pepin-ros/logs/`
(`<local time>.log` for the stack, `zrouter_<local time>.log` for the router, `docker logs -t`;
`ros/sync.sh` never deletes `logs/`); the router's output does not reach the journal. On the
laptop every removal goes through `ros/lib.sh`'s `pepin_remove_container`, which stops the
container, copies `docker logs -t` to `logs/containers/<UTC>_<name>.log` in this checkout
(`PEPIN_LOG_DIR`), and only then removes it — `ros/laptop.sh stop`, a `vslam` restart and a
router re-creation used to take the laptop router's and `pepin-vslam`'s logs with them. Docker's
own receive stamps are kept on purpose: after a Mac wake the VM clock's step is visible in them.
The unit change reaches the board by `scp board/pepin-zrouter.service root@<board>:/etc/systemd/system/`
and `systemctl daemon-reload`; it takes effect at the router's next restart.

**A frozen peer, and the 2026-09-23 wake** (`PEPIN_ZROUTER_RX_WORKERS`, `PEPIN_ZROUTER_LOG`). The
Mac idle-slept at 20:31Z with the stack up and woke three times for maintenance before the user
woke it at 21:27Z. Each sleep, the board router logged `Unable to push non droppable network
message to <laptop router>. Closing transport!` every 20 s for ~7 min, and the board's own nodes'
sessions timed out on their router; after each dark wake the link was back within seconds, after
the user wake it never came back (0 board samples reached the laptop router, all 89 board topics
left the laptop's graph 13 s after the wake and none returned in 28 min, while the camera's own
TCP connection through the same Docker VM kept delivering; `scratch/link_autopsy/wake_nonrecovery.py`).
The clock was not the cause: the dark wakes carried data under a 898-s VM lag. Neither router
wrote a line after the wake, so which end refused the new session is not in the saved logs.
What IS established, from zenoh 2687c51's source and reproduced in statics
(`scratch/link_autopsy/wedge_repro.py`: two routers, four RELIABLE publishers, the far router
`docker pause`d): a session's RX task routes its messages on zenoh's RX runtime, a push to a peer
that stopped reading waits `wait_before_close` (20 s) *on that worker* holding the priority
queue's mutex, and the `close()` that the failure schedules is spawned on the same runtime. With
zenoh's default of 2 RX workers both sat in 20-s pushes and the close started only when something
else broke the loop — 620 s after the first closure in one run (33 closures, the publishers' own
sessions dropped 24 times: the board router's log of that night line for line), 130 s in another
(at the thaw; 7 closures, 4 drops). With 8 workers the close started 9 ms after the first closure,
the dead transport was gone 35 s later and no local session dropped; 16 behaved the same (2 ms,
2 closures, recovery 0.3 s after the thaw) at 21 threads and 6.4 MB against 7 and 5.7 MB. The
count is set by the sessions: each one pushing toward the frozen peer holds a worker and the close
needs one more — four publishers wedged 4 workers (the close only at the thaw, 4 drops) and not
5, 6 or 32 (`scratch/link_autopsy/wedge_threshold.py`; one run per count). So each router runs
more RX workers than the sessions its census counted (2026-09-24, TCP sessions on 7447): the
board's `ZENOH_RUNTIME=(rx: (worker_threads: 16))` for its 10 nodes and the laptop's link, the
laptop's 32 for `pepin-vslam`'s 14 sessions, the link, and `pepin-laptop`'s nodes and the tools on
top (`PEPIN_ZROUTER_RX_WORKERS=2` is zenoh's default and the old behaviour), and both log the
transport lifecycle and the connector at
debug (`PEPIN_ZROUTER_LOG`; `info` is the old level): the next wake names the refusing end in one
line, at 19-28 lines a router per statics freeze-and-thaw. Not fixed and not claimed: the non-recovery
itself was not reproduced in statics (every thaw there recovered within 7 s, including one 25 s
after the last closure, as the user wake was), so a Mac wake with the stack up stays a case to
watch — the heal that worked on the night is the clean-order router restart above. **When each
side takes it:** the laptop router at its next re-creation (`ros/laptop.sh stop`, then `start` or
`vslam`; a running router is left alone), the board's once `board/pepin-zrouter.service` is
copied (above) and the router restarted. In between the two routers run different worker counts
and log levels, which is safe — a router's workers guard only its own sessions — but a wake then
is logged at debug on the laptop's side only.

## One clock (`PEPIN_TIME_SOURCE`, chrony)

Every ROS stamp on the laptop is the **Docker VM's** clock (all its containers share one kernel);
every stamp on the board is the board's. Scans, transforms and maps cross between the two, so the
robot runs on one time base:

- **the laptop serves its VM's clock**: the `pepin-chrony` container (`ros/chrony/`: alpine,
  `chronyd -d -x`, no sources, `local stratum 10`), published on `udp/123`, started by
  `ros/time.sh install|server` (and idempotently by `ros/laptop.sh` beside the zenoh router under
  `PEPIN_TIME_SOURCE=laptop`), `--restart unless-stopped`, left running across `ros/laptop.sh stop`.
  Measured in statics on 2026-09-23: the image builds in 12 s, chronyd 4.5 answers from the first
  second (stratum 10, leap normal, 1 MB of memory), 8 of 8 SNTP exchanges at a 1 ms round trip both
  on `127.0.0.1` and on the Mac's own LAN address — both stay inside the Mac, neither is the WiFi —
  through Docker Desktop 4.79's port publishing (Docker Engine 29.5.3; no sudo; the macOS firewall
  is off), and the VM's clock read 15 ms behind the Mac's. The path the design rests on, the board
  to the Mac's `udp/123` over the WiFi, measured on 2026-09-24 04:00Z with a throwaway server of the
  same image and `laptop.conf` (on the board `python3 - 10.0.0.167 < scripts/timesync.py`, which
  changes nothing there): 8 of 8 answers in each of three runs, best round trip 5-6 ms, board minus
  laptop +20.7 to +21.1 ms with the board still on systemd-timesyncd. chronyd logged all 32 queries
  (the Mac's 8, the board's 24) as coming from one address, 151.101.138.132, neither the Mac's nor
  the board's: the port proxy rewrites the source, which is why `laptop.conf` allows any;
- **the board follows it**: chrony instead of systemd-timesyncd (`board/chrony.sh`,
  `board/chrony/chrony.conf`), the laptop as a `prefer` source polled every 4-16 s, the internet
  pool (the servers timesyncd used) beside it — the fallback when the laptop is away and a vote
  against a laptop clock that is wrong. `makestep 1 3`: the clock is stepped only within the
  first three updates after chronyd starts (normally the boot, before `pepin-ros`'s wait ends) and
  slewed afterwards; a board that syncs only after that 90-s wait, or a chronyd restart (an apt
  upgrade, a re-run of `ros/time.sh install`), can still step it once under a running stack;
- **the board learns the laptop's address from the laptop**: `ros/time.sh` asks this Mac for the
  address of the interface that routes to the board (`ros/lib.sh`'s `pepin_laptop_ip`) and writes
  it to the board's `/etc/default/pepin-ros` (`PEPIN_LAPTOP_HOST`) and to
  `/etc/chrony/sources.d/pepin-laptop.sources`. Nothing on the board ever had the laptop's address
  before (the laptop dials the board, never the reverse); if DHCP moves the Mac, `ros/time.sh
  point` re-points it and check 1.15 says "not measured" until then.

**Why the laptop's server does not discipline the VM clock** (no `CAP_SYS_TIME`). The VM's clock
stops while the Mac sleeps and resumes where it stopped, minutes behind, and Docker Desktop steps
it back to the Mac's clock by itself 10-26 s after each wake (measured 2026-09-23 on four wakes:
+26, +10, +22, +19 s). Through three 15-minute-lag windows that day the link and RTAB-Map kept
working (data under replaced zenoh timestamps, RTAB-Map at `delay=-898 s`); what left the link
dead after the fourth wake was the router-to-router zenoh session, not the clock
(`scratch/link_autopsy/wake_chain.py`). A second agent stepping the one kernel clock every
container shares would race Docker's resync and could step every laptop node's time backwards.
The window is the board's to survive instead: a laptop sample minutes off is a falseticker
against the pool's three servers, and `prefer` (never `trust`) lets the vote win. What it costs:
with the internet down AND the Mac just awake, the board has only the laptop to believe for those
seconds (`maxupdateskew 100` still keeps a source with a jumping frequency estimate from moving
the clock).

**The boot wait holds under chrony unchanged.** `board/pepin-ros.service` waits up to 90 s for
`NTPSynchronized`, which timedated reads from the kernel (`adjtimex().maxerror < 16 s`, systemd
257); chronyd writes its real error bound there at every update and 16 s while unsynchronised, and
its first update after boot is the `makestep` step itself. That holds when the sync comes within
the 90 s: a board with no network at boot starts the stack anyway and takes the step later, under
it — no worse than systemd-timesyncd, which steps at any update over 0.4 s.

**The switch** (CLAUDE.md rule 19): `ros/time.sh source pool` puts the board on the pool alone —
live, a `chronyc reload sources`, no restart and no step — and `ros/time.sh source laptop` is the
way back. On the laptop the variable only decides whether `ros/laptop.sh` (re)starts the server;
its default is `pool` — what the board runs today under systemd-timesyncd — until the deploy
below has measured the laptop source on the robot, and then it flips to `laptop` in `ros/lib.sh`
and `scripts/timesync.py` together. `ros/time.sh uninstall` puts the board on
systemd-timesyncd exactly as before (the package's `.deb` is kept at install, so it works with
apt offline; our sources file and the Debian config backup go with chrony). Check 1.15 answers at
once while no server runs here (`PASS` "not measured, as configured" under `pool`), so an
undeployed laptop costs a restart nothing. A value that is neither `laptop` nor `pool` is refused
by `ros/laptop.sh` before a half starts (`ros/lib.sh`'s `pepin_time_source_check`).

**Deploy (not done yet; a parked robot, the owner's go):**

1. `ros/time.sh install` — builds the `pepin-chrony` image on first use (alpine from Docker Hub),
   starts the server, copies `board/chrony.sh` and its config to the board and runs `install`
   there: `apt-get install chrony` (about a minute of the board's CPU; Debian removes
   systemd-timesyncd with it), our `chrony.conf`, the laptop's source, `systemctl restart chrony`.
   No sudo on the Mac; root on the board through the existing key. The stack is not restarted.
   The clock is adjusted once by up to the current board-laptop difference (a step only if it
   exceeds 1 s, which it has not since the boot waits); do it parked.
2. `ros/time.sh status` — the laptop's source marked `*` (selected), the pool `+`/`-`.
3. `ros/time.sh offset` — under 100 ms; then `ros/board.sh census` for chronyd's real cost
   (`config/board_manifest.json` carries a guess).
4. With the offset measured: `PEPIN_TIME_SOURCE` defaults to `laptop` in `ros/lib.sh` and
   `DEFAULT_TIME_SOURCE` in `scripts/timesync.py` (and its test), one commit with the numbers.
5. Not verified from the laptop alone (the board is the client): that the board's chrony selects
   the laptop (`*` in `ros/time.sh status`) and that `timedatectl show -p NTPSynchronized` reads
   `yes` under chrony within the first minute after a reboot — the boot wait's premise.

## The retired zenoh bridge

Until 2026-09-20 the two halves ran CycloneDDS joined by two `zenoh-bridge-ros2dds` sidecars with
generated one-way allow-lists, a watch of the routes' flow and a kick that let the laptop restart
the board's bridge. All of it is on the tag `alt/cyclone-bridges-2026-09-20`, whose message says
how to bring it back. One thing of it stays: **a topic that crosses between the machines carries
the same reliability and depth on both ends** (`pepin.deployment.BRIDGED_QOS`, handed to every
reader by `node_kit.bridged_qos_profile`): `/imu/data_raw`, `/vo` and `/odom`, all RELIABLE ten
deep. A reader and a writer that disagree on reliability do not match at all.

## A new room

There is one arrangement (World R, one localiser): RTAB-Map on the laptop is the map and the owner
of `map -> odom`, whether the room is known or new. A new room is an empty database:

```bash
ros/restart.sh laptop --fresh-graph   # empty RTAB-Map database, the old volume moved aside
ros/teleop.sh                         # or drive by goal — the map grows as the cart moves
ros/map.sh save flat3_new             # freeze the grid into ros/maps/flat3_new.{yaml,pgm}
```

`ros/teleop.sh` latches a key until the next one (a terminal has no key-up): arrows drive at full
speed, Shift+arrows slow, space stops. The game mode, `uv run python -m pepin.teleop --game
[--host 10.0.0.187]`, is a small window where keys act only while HELD and the window is focused:
arrows the wheels at the same speeds, W/S tilt and A/D pan the head (the base server's `neck_jog`,
board/README.md), Shift slow, Space stops everything, Esc quits. It speaks to the base server on
:3336 directly, past ROS, so a drive made with it is not recorded.

Goals work with no places book: `ros/go.sh -1.0 0.3 90` drives to map coordinates (recorded like
any other drive), and a click in Foxglove (Publish → `/goal_pose`, frame `map`) does the same
without a tape. **The goal server takes the cart's pose from `map -> base_link`** and accepts a
goal while that edge is younger than 1 second; older, or missing, the goal is refused with which
of the two it was. `ros/go.sh where` says `"pose": "tf"`, and `ros/go.sh mark` fills the book on
the same evidence.

**A fresh edge is not a placed start.** RTAB-Map publishes `map -> odom` from the moment it
starts, at the pose it saved at its last shutdown. The laptop's `rtabmap_frame` says on
`/localization/placement` (latched) whether this start has recognised a node of the loaded map or
been seeded (`ros/goto.sh seed X Y YAW`), and both goal paths — the goal server and
`ros/tools/goto_ros.py` — refuse until it has, or when nothing is heard at all. The switch back
works with the laptop's node down or older, because it lives on the goal server and goto_ros
reads it from there: `ros/flags.sh set goal_server start_needs_placement false`.

**The camera's costmap layers stay off for driving.** They marked within 2 cm of the hull and
stalled two drives on 2026-09-13 (the contact ring at 1.2–1.5 m, the depth band beside the hull);
`ros/sensor.sh camera off` before a goal, and `ros/sensor.sh status` to see where they stand.

Watch the map grow with the `ros/foxglove/pepin_slam.json` layout at `ws://localhost:8765`: `/map`
under the fused surface, the graph's path, the head camera.

## Camera rigs

The head is a RIG chosen by name. `config/camera.json` holds the cameras as named blocks and one
key says which of them the robot is wearing:

| rig | what it is | what the laptop publishes |
| --- | --- | --- |
| `overview` | the mono AC310 webcam, 1280x720, checkerboard-calibrated (45 views, 0.23 px) | `/camera/image` (bgr8, half size by default) + `/camera/camera_info` |
| `stereo` | the global-shutter stereo module, ONE 1600x600 side-by-side frame at 10 fps, 800x600 an eye, taped upside down | the same two topics for the LEFT eye (rectified) plus `/camera/right/image` (mono8) + `/camera/right/camera_info`, all four under one stamp in the left eye's optical frame |

**Switching rigs takes two edits, one per half of the robot.** They must agree: `"active":
"stereo"` beside a board still serving the webcam is a node cutting a 1280x720 picture down the
middle (it says so, with both sizes, in every report line).

* **The board** — which device ustreamer opens and at what resolution: `/etc/default/pepin-camera`
  (`PEPIN_CAMERA_DEVICE`, a `/dev/v4l/by-id` path; `PEPIN_CAMERA_RESOLUTION`, `1600x600` for the
  stereo module and `1280x720` for the webcam; `PEPIN_CAMERA_ENCODER=HW`, which passes the
  camera's own MJPEG through instead of re-encoding it on an A53 core). The unit reading it is
  `board/pepin-camera.service`; `systemctl restart pepin-camera` after an edit.
* **The laptop** — which block of `config/camera.json` every node reads: the top-level `"active"`.
  `PEPIN_CAMERA=overview ros/laptop.sh vslam` overrides it for one container (the variable is
  forwarded in), and `camera:=<name>` overrides both for one launch. The order lives in one
  function, `pepin.camera.active_camera`; a name no block answers to stops the launch at start.

**What stereo publishes.** The laptop decodes the side-by-side frame once, cuts it into the two
eyes as the robot sees them (`pepin.stereo.SideBySide`: an upside-down module's halves are turned
back and swapped — verified on a real frame, only then is the disparity of near objects positive)
and, **with `config/stereo_calibration.json`**, rectifies both onto one pinhole with the rows
aligned. Then four messages go out with ONE stamp (the board's capture time) and the LEFT eye's
`camera_optical` frame: the left picture with the rectified `CameraInfo` (`P`'s Tx zero, no
distortion), and the right picture in grey with the same `K` and `P[0,3] = -fx * baseline`. That
is where the baseline lives from then on — on the wire, not in a config. Everything that already
reads `/camera/image` + `/camera/camera_info` sees one ordinary camera.

**Without that file** the head cannot measure: only the left eye goes out, unrectified, with the
nominal one-eye pinhole of `hfov_deg`, nothing is published on the right topics, and the report
line says `NOT RECTIFIED ... depth has no source`. The rectifier costs about a second to build,
so it is built once and rebuilt only when the calibration file's mtime moves — a calibration
finished while the robot is running is picked up without a restart, and the log says so.

On a stereo rig the two mono flags are refused with their reason: `undistort`, because the stereo
calibration is what rectifies here, and any `scale` but 1.0, because the eyes are published at
the size their remap tables were built for, which is the size a disparity is in pixels of.

### The two stereo matchers

What turns the two eyes into a disparity is the depth node's live `stereo_matcher` flag. Both
engines are built when the node starts, so an A/B is one `ros2 param set` and never a restart.

| matcher | what it is | where it runs | ms a pair, 800x600 |
| --- | --- | --- | --- |
| `sgbm` (default) | OpenCV's semi-global block matcher, `pepin.stereo_depth.StereoMatcher` | in the depth node's own container | 17 |
| `raft` | RAFT-Stereo, `raftstereo-realtime.pth`, 7 iterations | natively on the laptop's GPU, over HTTP | 89 |

**Why the network is not in the container.** Docker on macOS cannot see Metal, and at the full
size — the only size worth having, see below — the same forward pass on the container's CPU is
about 3 fps. So RAFT runs where the GPU is, in the *same* native process that already serves the
mono depth network: one port, `POST /disparity` beside `POST /depth`, one inference lock. Two
processes would each hold a Metal context and race for the one GPU with nothing serialising them,
and the node only ever runs one depth source, so the model nobody asks for is simply never built.
The node sends two rectified grey eyes raw (the rectifier stays in the node; JPEG ringing is
exactly the error a subpixel disparity cannot tell from a real shift) and gets float16 disparity
back — 0.06 px of quantisation at the near end, an eighth of the half pixel the rig's reach is
derived from.

```bash
ros/depth_host.sh stereo     # the host with RAFT built at start (~2 s), beside the mono network
ros/depth_host.sh status     # both models: device, pairs served, per-stage ms
ros/flags.sh set depth_stream stereo_matcher raft    # and back to sgbm; no restart either way
```

`ros/depth_host.sh start` serves `/disparity` too — RAFT is then built on the first pair, which
costs that one pair ~2 s inside the lock. `stereo` is the verb that pays it up front.

**What the switch buys and what it costs.** Measured through this path on 2026-09-22, on the 8
rectified pairs of `scratch/stereo_net/frames/pairs.npz` — the parked robot, glossy herringbone
parquet with the ceiling lamp's reflection in view:

| | SGBM | RAFT realtime | script |
| --- | --- | --- | --- |
| airborne phantom pixels (0.5–1.5 m up, 0.15–2.5 m ahead, \|y\| < 0.6 m) | 6651 | 5265 | `host_smoke.py` |
| blobs those form, whole picture | 450 | **94** | `phantom_where.py` |
| of them, 50 px or more | 192 | **37** | `phantom_where.py` |
| phantom pixels in the robot's own path (\|y\| < 0.35 m) | 4486 | **2566** | `phantom_where.py` |
| points within 5 cm of the fitted floor | 204459 | **958432** | `host_smoke.py` |
| ms a pair, end to end | 16.8 | 89.2 | `host_smoke.py` |

The pixel count is not the argument — it falls by a fifth, and RAFT is the worse of the two on
three of the eight pairs. What a costmap receives is: SGBM's airborne mass is 450 separate specks
sprayed over the open parquet, RAFT's is 94 blobs against one bright doorway, and in the robot's
own path the count halves. The floor is the other half of it — 4.7x as many points within 5 cm of
the plane, so it comes back whole instead of in patches. Half size is disqualified, not cheap: the
network's disparity is scaled back up and its error with it, to twelve times SGBM's phantom count.
And 89 ms is 11 fps against a camera that delivers 10 and a `scan_hz` cap of 5 — the whole budget,
with the pose and the correction pipeline still to pay for, and the zenoh routers not yet in the
path. That is why the default stays `sgbm` until a live drive settles it.

> An earlier note (`scratch/stereo_net/matcher_raft.patch`) gives this comparison as 31314 → 5338
> phantoms and 274039 → 1009276 floor points. The RAFT half reproduces; **the SGBM half does
> not** — that note's own script, run unchanged today, prints 6651 and 204459. Nothing from its
> SGBM column is quoted here or in the code.

**When the host does not answer**, the pair goes to the very SGBM object the flag would otherwise
have used and is counted; after three failures in a row the host is left alone rather than costing
a timeout per pair, and probed again every 30 s. The report line says how many pairs fell and for
how long the host has been down — the same shape as `depth_backend: auto`.

**The checkpoint** lives in `models/` at the repo root, which is gitignored (40 MB of binary):
fetch it with RAFT-Stereo's own `download_models.sh`. Which file, how many iterations and which
device are the head's own, in `config/camera.json`'s `net` block, because they are properties of
the camera and not of the code that loads the network; an empty `weights` means this machine has
no checkpoint and the flag cannot be turned on. The model's source is vendored under
`src/pepin/vendor/raft_stereo/` (MIT, princeton-vl/RAFT-Stereo at 6e93ed2) — see that package's
`__init__.py` for the three edits and nothing else.

## Camera calibration

The neck camera's optics were a guess: one field-of-view number (78 deg, fitted against the lidar
on 2026-09-10) standing in for four — `fx`, `fy`, `cx`, `cy` — and a lens that bends straight
lines and was not modelled at all. A checkerboard measures all of it in one sitting. It was run
on 2026-09-13: 45 views, 0.23 px RMS, 82.94 deg wide — so `calibrated: true` and the numbers
below are in the file.

```bash
ros/calibrate.sh --print      # data/checkerboard.pdf: print at 100 %, not "fit to page"
ros/calibrate.sh              # a window, ~25 views, then the fit is written to config/camera.json
ros/calibrate.sh --no-window  # the same over ssh: a text coverage report instead of the window
```

**Print the board.** `--print` writes an A4 (or `--page letter`) sheet of 10x7 squares — 9x6
*inner corners*, which is what the calibration counts — with its square size printed on it. Print
at 100 %; then **measure one square with a ruler** and, if it is not 24.0 mm, pass what it really
is: `ros/calibrate.sh --square 0.0235`. That square is the only length in the whole procedure, and
every metre the camera later reports is wrong by however wrong it is. Tape the sheet to something
rigid — a book, a clipboard. A bent board fits a bent lens.

**Run it.** No keys to press: hold the board where the line at the top says, keep it still, and
the shot is taken on its own countdown (a bar fills; a green border means it was kept). The grid
drawn over the picture is the coverage — each third of the frame wants two views, because the
distortion at a corner of the image is only ever seen by a board that was at that corner. It also
wants six views turned at an angle (about 30 deg or more: fronto-parallel views alone let the
focal length and the distance trade against each other) and three close enough to fill the frame.
It stops by itself at ~25 well-spread views; `q` stops it early.

**What the numbers mean.**

| Number | What it is | What is good |
| --- | --- | --- |
| `fx`, `fy` | focal length in pixels, at 1280x720 | within a few per cent of each other |
| `cx`, `cy` | where the optical axis crosses the sensor | near 640, 360 — tens of pixels off is normal, hundreds is a bad fit |
| `dist` | plumb_bob: `k1 k2 p1 p2 k3` | `k1` around -0.3 for a wide webcam; the tangential `p1 p2` near zero |
| `rms` | mean reprojection error over every corner of every view | **under 0.5 px, or nothing is written** |
| worst view | the view the fit explains worst | one blurred view carries the whole RMS — shoot that place again |
| `hfov_deg` | derived from `fx`, for people to read | inside the block; the config's own `hfov_deg` stays the nominal number |

A poor run refuses to write and says why: too few views, a frame the board never covered, or an
RMS over the bound. Accepted frames are kept under `data/camera_calib/<timestamp>/`, so a run can
be re-fitted without the camera: `ros/calibrate.sh --images data/camera_calib/20260912-181500`.

**After it is written.** `config/camera.json` gets an `intrinsics` block beside `calibrated: true`.
The config's own `hfov_deg` is left alone — it is the nominal field of view the stack falls back
to, and the measured one lives inside the block. Everything that needs the camera's optics reads
one function,
`pepin.camera.optics` — `camera_stream` publishes the measured `K` and `D` on
`/camera/camera_info` (scaled to whatever `scale` publishes), and `depth_stream` uses the same
numbers as its fallback until a `camera_info` arrives. Restart the node to pick them up
(`ros/laptop.sh kick camera_stream`); its report line then says `optics: calibrated 2026-09-12 on
9x6 ..., rms 0.23 px, 82.9 deg wide` instead of `optics: nominal 83 deg field of view
(uncalibrated)`. The `undistort` flag publishes a rectified picture (`ros/flags.sh set
camera_stream undistort true`); it is off until the straightened picture has been measured against
the raw one on the robot. A calibration that turns out bad is switched off with one boolean —
`calibrated: false` — and the block stays in the file as history, the nominal pinhole coming back
untouched. That nominal is `hfov_deg` 82.94: a calibration never rewrites it, but the best field
of view known is what a bad calibration must be switched off onto, so it was set by hand to what
the checkerboard measured when the 78 it held was retired.

**The tilt was re-measured, and not by a fit.** `mount.pitch_deg` used to be 26 deg out of the
same lidar fit as the 78 deg field of view, where the two traded against each other (78/26 with a
1.2 % residual, the nominal 70/28 with 3.0 %) — so pinning `fx` with a checkerboard left that
tilt standing on a focal length no longer in use. It is now 23.8 deg, from the neck's own encoder
against four still frames of one room: head level by eye the tilt servo reads 2068 ticks and the
picture is 1.0 deg down (±1.5), and the reference pose sits 243 ticks further down at the plain
360/4096 deg a tick — which those frames also verify (1.07 true degrees per commanded degree,
1.03..1.09 over every variant, so `pepin.neck.RAD_PER_TICK` needs no correction).
`scratch/neck_tilt_scale.txt` is the whole run. The height came off a tape at the same pose
(floor to tilt axis 1.134 m, lens 0.086 m above it and 0.025 m in front: 1.203 m), which is also
where `config/neck.json`'s `pivot` block stopped being zero. `camera_stream` broadcasts
`mount.pitch_deg` as the static `base_link -> camera_link` edge, so both numbers moved the whole
camera in TF with them.

**When to redo it.** After anything that changes the optics or the sensor's relation to them: a
different lens or camera, a knocked or re-seated lens barrel, a re-mounted head that required
touching the camera body, a change of capture resolution. Re-seating the *neck* changes the mount,
not the intrinsics — that is `config/camera.json`'s `mount` block and a different measurement.

### Camera only

`ros/laptop.sh vslam --camera-only` does not subscribe to the lidar at all: the grid is built from
the camera's depth (`Grid/Sensor 1`), ray-traced so the floor it flew over becomes free space, and
capped at 3 m — past that this network's metric scale stops being a measurement. It is the honest
test of "the camera as the primary sense"; with the lidar present the 2D grid is the **scan's**
(`Grid/Sensor 0`), because the depth's scale is scene-dependent (0.94 to 1.98 across one
afternoon) and a map the cart plans on may not be built out of that.

Other flags: `--resume` continues the session's database instead of starting empty, `--fresh`
deletes it first, `--known-map` forces the old mode regardless of what the board is doing.

### The depth law

`depth_stream` fits one law over the lidar's pairs: the **affine** law in inverse depth over a
600-frame pool, what a restart is seeded with. Under the stereo head it watches (`law_watch`):
the depth goes out as measured, and a 1.00 b +0.000 says the head is as calibrated. Under the
monocular network it corrects. The network's range law, frame law and the rulers that fed them
(floor pairs, wall anchor, parallax anchor) are on the tag `alt/mono-depth-2026-09-21`.

### Not yet verified on the robot

- RTAB-Map's own `map -> odom` quality against the wheels-plus-gyro EKF: on a known map its
  "odometry" was the tracker's centimetre-accurate pose, and the graph rejected closures as
  inconsistent when it was fed raw odometry (2026-09-10). Whether ICP on the scans plus
  `RGBD/NeighborLinkRefining` absorbs that drift is the first thing to look at.
- Nav2 on a growing map: the global costmap's static layer resizes with every new `/map`, and the
  local costmap works in the `map` frame, which jumps at a loop closure. `track_unknown_space:
  false` means unmapped ground is planned through as free floor — which is what exploration needs
  and what a wrong correction would exploit.
- The correction's path (RTAB-Map's `map -> odom`, broadcast on the laptop) adds a wireless hop
  before the transform moves on the board; each broadcast is stamped 0.5 s ahead
  (`tf_tolerance`), so a WiFi stall shorter than that is invisible to the lookups.
- `Grid/RangeMax 8.0` for the lidar grid and the camera-only ground/obstacle heights are
  first guesses from the known-map profile, not measurements.

## The world map

One map for both sensors, always alive. The fused volume (`pepin.tsdf`, built by
`pepin_bringup.depth_fusion` on the laptop) is no longer a picture nobody localises against: the
lidar writes its own layer into it and the volume IS the map (`src/pepin/worldmap.py`).

- **The lidar's layer.** Every `/scan` is integrated at the height `config/lidar.json` calibrates
  and nowhere else (the plane ± one voxel): each beam carves free space along its run and marks a
  surface at its return, on the lidar's own weight channel. The return itself is sampled where it
  came back, not only on the ray's sampling ladder: a wall that falls on a cell boundary has no
  ladder rung within half a voxel of it and would otherwise average out of the map (18 % of such a
  wall left after nine viewpoints; 95-100 % with the return sampled). A beam with **no** return
  writes nothing — a mirror and a black chair leg say the same nothing as an open door — unless
  `no_return_free` is on, and then it carves to the sensor's reach and marks nothing there. The
  camera keeps writing its band
  through the same volume, but inside that layer it may not repaint a cell the lidar has spoken
  for — the network's depth is scale-uncertain, the lidar's returns are metric truth.
- **The whiskers write it too** (`tof_rays`, since 2026-10-01). The three ToF fans
  (`/tof/<name>/scan`) go into the volume as rays from their own frames, through the camera's
  integrator: each fan is a tiny depth image behind a pinhole whose edges are the sensor's cone
  (`pepin.tof_rays`). A return marks a surface at its range and carves the ray free up to it;
  `+inf` — nothing within the fan's trusted range, its `range_max` — carves free out to that
  range; NaN says nothing. Every fan weighs 0.5, hit or miss, instead of the grid's (ref / d)^2
  (a ranger's error does not shrink with distance): at the node's `min_weight` 4.0 a hit speaks
  in `/depth_marks` after 8 fans (0.5 s at 15 Hz), a stared-at pillow saturates at `max_weight`
  20 after 40, and 29 misses (1.9 s) carve a saturated one back out — the hit's own back halo is
  written no farther than the reach, so no voxel a hit leaves outlives the misses that must
  clear it. The camera's rays through the spot carve it at their own weight. This is why a
  pillow below the lidar and too near for the stereo rig now reaches the global planner: both
  costmaps read the volume, so what only the whiskers saw is in the memory they plan on.
- **Slices.** A horizontal band of the volume reads out as an occupancy grid: `lidar_slice()` at
  the lidar's plane, `camera_band_slice()`
  over `camera_band_m` of `config/fusion.json` — the band `/depth_scan` marks in, where seats and
  tabletops the lidar's plane cannot see are. A column is occupied where the field comes within
  half a voxel of a surface, free where it stays a voxel away from anything, unknown between; a
  cell speaks only once it carries enough observations (`map_min_weight` for `/map`, capped at the
  lidar's own weight cap so no value can blank the map the cart drives on; `min_weight` is the
  debug cloud's). **Maturity is the weight in a cell,
  not a flag**: nothing is ever "finished", a cell that stops being observed simply keeps its
  weight and a chair that moves is cleared by the beams that cross it.
- **A matcher gets hard cells only.** `/map_camera` — the band the camera's own scans are matched
  against on the laptop — is cut with its own, far higher threshold (`camera_map_min_weight`, 20
  against the map's 2). A cell the camera painted two frames ago at the pose it is now asking
  about is not evidence about that pose: that circle is how camera-only localisation walked away
  in 20-33 cm and 15-23° steps (2026-09-13). At `weight_ref_m` 2.0 m an observation weighs 1 and
  the stream runs 8-9 frames a second, so 20 is 2.5 s of watching one cell from 2 m — and it is
  also the lidar's own weight cap, so a saturated lidar cell still speaks in the band. Painting is
  untouched: a cell is in the volume from the first frame and simply does not appear in the
  matcher's slice until it is heavy. The report line says what that costs — the share of the
  band's occupied cells the threshold keeps.
- **`/map` has exactly one owner.** Two launch decisions, both told to the node, decide whether
  it may publish: `pepin.deployment.map_owner` per bridge mode — the board's `map_server` in
  `split` and `vision`, the laptop in `slam` — and `world_map:=true`, which is what keeps
  RTAB-Map's grid on `/rtabmap/map` instead of remapping it onto `/map`. With both, and
  `map_source=volume`, the fusion node publishes the lidar slice as `/map`; without either it
  refuses and names the reason in its report line, so setting `map_source volume` live in an
  ordinary SLAM run cannot put a second publisher on `/map` — which is the failure this whole
  table exists to prevent.
- **That `/map` is latched and published on change.** The slice is hashed
  (`OccupancyGridFields.digest`: the geometry and a CRC of the cells) and an identical grid is
  never sent twice — `map_hz` (0.5) is the ceiling on how often a *changed* one may go, not a
  cadence — and the first one is published as soon as the seed is in the volume rather than at
  the timer's first tick. A republication buys a subscriber nothing (transient local already
  serves a late one the last map) and costs every reader a rebuild: a tracker that adopts one
  rebuilds its matcher, its mask and its tracker on four A53 cores, a costmap re-seeds its
  static layer and resizes every other layer with it. With the wifi down the board keeps the
  last `/map` it was handed — a grid in memory does not stop working because its publisher went
  away — so nothing new runs on the board for rule 20.
- **Nothing is painted at a pose nobody trusts.** The volume is written in the MAP frame and a
  TSDF cannot be un-integrated, so an observation placed by a wrong pose does not add noise — it
  deletes the room. Both paint paths ask one predicate first (`pepin.watch.PaintTrust`, flags
  `fit_gate` for the camera and `lidar_fit_gate` for the lidar): the fit is at least `DRIVE_FIT`,
  it was **heard** within the source patience, `sigma_xy` is within `paint_sigma_m` where
  `/localization/sigma` speaks, and the `map -> odom` edge is within a second of the observation.
  Otherwise the observation is withheld and counted, with the reason in the report line
  (`lidar revolutions withheld: N (pose not trusted: ...)`). Until 2026-09-16 only the camera was
  asked, and only about the number — a fit that stops arriving keeps its last good value in the
  node for ever, which is how the session whose routes died at 19:17 on 2026-09-15 went on
  painting at a frozen pose. The cost to a healthy drive was measured on the four tapes of
  2026-09-13 (`scratch/paint_gate_on_a_tape.py`): 452 of 10048 revolutions withheld, an upper
  bound of 4.5 % of which 439 are the tape's own ~2 Hz pose recording against the node's 20 Hz
  edge — 13 revolutions, 0.13 %, where the tracker had really gone quiet, and **not one** refused
  for a low fit. Both gates come up off in SLAM mode (`vslam.launch.py`), where no tracker speaks.
- **Whether the volume may be the default `/map` is a measurement, and today it fails.**
  `scratch/volume_vs_file_seating.py` replays the node's own tracker over the four tapes of
  2026-09-13 once per map. A volume **seeded** from `flat3_straight` and sliced seats a median
  **0.01 cm** from where the file puts it, with the same fit to two decimals — the same map. The
  **live** snapshot of 2026-09-15 (that file plus a day's painting) seats a median **1.90 m**
  away with the fit down **0.295**, re-seating 3.7-3.8 m off on two of the four tapes; only
  52.9 % of the file's walls are still within a cell of it and 2070 of them have been carved
  free. The mechanism is known and not fixed: `fit_gate` holds the **camera** path only, while a
  lidar revolution is integrated at whatever pose TF gives, lost tracker or none. So
  `map_source` stays `file` until the lidar path is gated too.
- **Known room, unknown room, one machine.** The volume is written to `world_path`
  (`/maps/world_live.npz`) every `snapshot_s` and at shutdown, and loaded at start
  (`resume_volume`). A known room is a resumed snapshot — or a saved map seeded into the lidar's
  layer with `ros/laptop.sh vslam --seed-map=/maps/flat3_straight.yaml` — and an unknown one an
  empty volume. There is no mode switch between them, and the map keeps growing either way.
  Seeding also **snaps the volume's grid to that map's own cell lattice** (the box moves by less
  than one voxel): unsnapped, the config's grid sits a third of a cell off `flat3_straight`'s, and
  a tracker matching on the slice answered a median 2.3-2.9 cm from where the very same map as a
  file put it on the four tapes of 2026-09-13 — snapped, the two agree to 0.1 cm. The seeded rows
  become the lidar's own layer at once, so the camera cannot repaint the file's walls before the
  first revolution arrives.

```bash
ros/laptop.sh vslam --world-map            # /map comes from the volume instead of RTAB-Map's grid
ros/flags.sh set depth_fusion map_source file    # back to the old behaviour, live
ros/flags.sh set depth_fusion no_return_free true # beams with no return carve an open door
ros/flags.sh set depth_fusion lidar_layer false  # the volume goes back to being the camera's alone
ros/flags.sh set depth_fusion snapshot_s 30      # write ros/maps/world_live.npz twice a minute
ros/laptop.sh vslam --seed-map=/maps/flat3_straight.yaml  # the volume starts as the served map
```

Offline, `WorldMap.export_pgm_yaml` writes the map_server pair every existing tool already reads
(`ros/mode.sh nav /maps/NAME.yaml`, `pepin.mapping.grid_from_pgm`), so a volume can be frozen into
a file exactly like `ros/map.sh save`.

**The next step, not built:** re-fusing after a loop closure. A TSDF cannot be un-integrated, so a
graph correction leaves the old geometry standing. The cure is to replay the frames at their
corrected poses, and the snapshot already carries the index for it — every integration's stamp,
sensor and pose — while the measurements themselves stay in the run tape, where they already live.

## The marks audit: who painted the lethal cells, live

`pepin_bringup.marks_audit` is the third laptop-side reader of the board's topics, beside the
localizer and the graph's frame node, and it answers one question the operator used to need a tape
for: of the lethal cells in the local costmap right now, how many are the lidar's and how many only
the camera's. Once per published `/local_costmap/costmap` (1 Hz) it takes every lethal cell — ROS
`100`, i.e. costmap `254`; `99`/`253` is the inflation's inscribed band and is judged only with
`inscribed_counts` on — within `radius_m` of the cart, places `/scan` and the camera's
`/depth_marks` fan into the grid's own frame through TF at each message's own stamp (the frame is
read from `header.frame_id`, which became `odom` on 2026-09-22), and calls a cell **lidar-backed**
when a return lands within `match_cells` of it, **camera-only** when only a camera beam does, and
**unexplained** when neither can. A camera-only cell is a phantom *or* a real thing above the
lidar's plane — a table top, a seat — and the node deliberately does not guess which; the operator
is looking at the room. Out go `/marks_audit` (a `std_msgs/String` of JSON: `lethal`,
`lidar_backed`, `camera_only`, `unexplained`, `nearest_camera_only_m`) for a Foxglove plot and
`/marks_audit/phantoms` (a `sensor_msgs/PointCloud2`, red, in the grid's frame) for the 3D panel
beside the costmap, plus a report line every 10 s: `marks audit: lethal 214 (lidar 97, camera-only
106, unexplained 11), nearest camera-only 0.42 m`. It runs in the SLAM half by default
(`marks_audit:=false` leaves it out; the `marks_audit` flag switches it off live) and costs the
laptop a few milliseconds of numpy a second — brute-force distances over a 60x60 grid, no scipy in
the image — and the board nothing at all.

## Camera grid A/B (2026-09-24)

Today the camera reaches Nav2 as the `/depth_marks` fan, and each costmap's `camera_layer` (an
`ObstacleLayer`) accumulates it into a grid of its own: the global one is smeared by every
`map -> odom` jump and both are wiped by the behaviour tree's clears (scratch/costmap_split). The
alternative, behind switches that all ship off: `depth_fusion`'s `grid_out` publishes the volume's
CURRENT occupied columns — the same column rule as the fan — and a `camera_grid_layer`
(`nav2_costmap_2d::StaticLayer`) in each costmap only draws the latest grid (`pepin.camera_grid`).

| topic | type | frame | what |
|---|---|---|---|
| `/camera_grid` | `nav_msgs/OccupancyGrid` | the volume's (`odom`) | a 6 m square about the cart at 5 cm (`grid_size_m`, `grid_resolution_m`), corner snapped to 0.5 m; 100 occupied, 0 otherwise; latched, at most `grid_hz` (3 Hz), stamped with the observation; ~14.5 kB, ~43 kB/s to the board |
| `/camera_grid_map` | `nav_msgs/OccupancyGrid` | `map` | EMPTY, on exactly the lattice of `grid_map_topic` (`/map`); latched, once per map geometry; nothing until that map is heard |
| `/camera_grid_map_updates` | `map_msgs/OccupancyGridUpdate` | `map` | the same cells drawn through this laptop's `map -> odom` at the grid's stamp, one rectangle covering the last window and this one, at `grid_hz`; held 1 s after a new geometry |

Nav2 must load the new `nav2_params.yaml` once (the plugin list changed): restart the Nav2 halves.
After that the A/B is live and reversible:

```bash
ros/camera_grid.sh on       # grid_out, then per costmap: camera_grid_layer on, camera_layer off
ros/camera_grid.sh status   # the flag, both costmaps' two layers, the fusion's last "grid:" report
ros/camera_grid.sh off      # camera_layer on, camera_grid_layer off, then grid_out off
```

By hand, the same switches (the global costmap is the laptop's `pepin-laptop` under the split,
`PEPIN_SIDE=board`, and the board's `pepin-ros` otherwise):

```bash
ros/flags.sh set depth_fusion grid_out true
ssh root@10.0.0.187 docker exec pepin-ros /pepin_entrypoint.sh ros2 param set /local_costmap/local_costmap camera_grid_layer.enabled true
ssh root@10.0.0.187 docker exec pepin-ros /pepin_entrypoint.sh ros2 param set /local_costmap/local_costmap camera_layer.enabled false
docker exec pepin-laptop /pepin_entrypoint.sh ros2 param set /global_costmap/global_costmap camera_grid_layer.enabled true
docker exec pepin-laptop /pepin_entrypoint.sh ros2 param set /global_costmap/global_costmap camera_layer.enabled false
```

and back with the four `enabled` values swapped and `grid_out false`. What guards what:

- **A layer enabled before its first grid is never current**, and the controller and the planner
  then answer "Costmap timed out waiting for update". `camera_grid.sh on` enables a layer only
  after reading its latched grid back from the topic, and leaves `camera_layer` on where it could
  not. A disabled layer, or one whose topic is silent, never blocks activation.
- **The last grid stays when the fusion stops** — killed, kicked, crashed or cut off by the link
  (a StaticLayer has no timeout, the tree's `ClearEntireCostmap` resets a StaticLayer without
  erasing its cells, and after a SIGINT rclpy can no longer publish). Only `grid_out false` (so
  `camera_grid.sh off`) sends empty grids first. With the laptop gone the board-side switch is
  the way out: `ros2 param set /local_costmap/local_costmap camera_grid_layer.enabled false` on
  `pepin-ros`. The cells stay where they were in `odom`; what is lost is their clearing.
- **The global costmap resizes for a map of another geometry**, dropping every layer's marks. The
  map grid copies `grid_map_topic`'s origin, size and resolution exactly (Nav2's own 1e-5
  tolerance), so it resizes nothing the map itself does not. An update is always inside the
  geometry it was drawn for; the one unguarded case is an update already in flight when the map
  SHRINKS — switch the grid off before swapping databases.
- `use_maximum: True` on both costmaps is what lets the grid's zeros leave the other layers alone;
  for the global `static_layer` it is a no-op (first plugin, `track_unknown_space: false`), which a
  contract test holds.

## Feature flags

Every behaviour that can be switched is a live parameter of the node that owns it, declared
once in that node's `FLAGS` table (`pepin.flags`, declared to ROS by
`pepin_bringup.node_kit.Switches`) with its kind, default and four texts, checked by kind on
every change, and printed in the node's report line (CLAUDE.md rule 19). A change lives until
the node restarts; a default changes in the table. This section is generated by
`ros/tools/flags_doc.py` (the pre-push hook keeps it current).

**How to read the flags.** A flag is this robot's A/B switch: a new behaviour ships with the
old one still reachable, so a regression in the field is turned off on the spot instead of
reverted, and two runs a minute apart can be compared without a restart. `ros/flags.sh list
[NODE]` shows every flag with the value the running node holds, `ros/flags.sh get NODE FLAG`
reads one, `ros/flags.sh set NODE FLAG VALUE` changes one live (a value the flag refuses is
refused with the reason, before any host is touched), and `ros/flags.sh flag NODE FLAG` prints
the whole entry below: what it does, why the default is what it is, when to turn it on, when
to turn it off. The *Default* line carries the measurement the default rests on and the file it
was measured in, or says `default by design, unmeasured` when there is none — a flag never
argues from taste. Switches live here; the numbers that are not switches (the lidar's mount,
the camera's intrinsics) live in `config/*.json` and are read at start; the numbers a person
tunes live are the config knobs below.

**Muting a sensor live.** A sensor is switched off where it is *published*, by a flag of the node
that publishes it, so the message simply stops and every consumer meets what a dead sensor looks
like — silence, an EKF's `sensor_timeout`, a transform that stops moving — with nothing
restarted and no other live flag lost. `ros/sensor.sh mute imu|odom|vo|lidar` and
`ros/sensor.sh unmute ...` do it in one command and print what to expect; `ros/sensor.sh status`
lists each sensor's mute state. The flags behind them: `base_bridge` `imu_publish` and
`odom_publish` (the board's bridge; `odom_publish` takes the `odom -> base_link` transform with
it, because a transform still broadcast from a silent `/odom` is a state no sensor failure
produces), and `visual_odometry` `vo_publish`. The lidar has none: our own node in its chain is
`scan_filter` (laser_filters, external), and a relay on the board that could drop `/scan` is
what CLAUDE.md rule 20 refuses — so `mute lidar` is the consumer set instead (`lidar_layer` off
on both costmaps, which is `ros/sensor.sh lidar off`), and `ros/sensor.sh lidar off --hard` is
the real absence of a scan. The old way — `ros/feature.sh
imu off` — restarts the board stack: a minute, and every live flag on it back to its default.

| node | flag | kind | default | live | description |
| --- | --- | --- | --- | --- | --- |
| `base_bridge` | `imu_publish` | bool | on | yes | the MPU6050's readings leave the bridge as /imu/data_raw, where the EKF fuses index 11 (the yaw rate) and nothing else; off, the chip is still read and its bias still estimated, but no message is published. THE PYTHON BRIDGE PUBLISHES NO IMU AT ALL — here the flag only exists so the node's table is the same table whichever bridge robot.launch.py started; the C++ bridge is the one that reads the chip |
| `base_bridge` | `odom_publish` | bool | on | yes | the base server's state line leaves the bridge as /odom and, while publish_tf is on, as the odom -> base_link transform; off, the wheels are still read and still commanded, and both go silent together — a transform still broadcast from a silent /odom is a state no sensor failure produces |
| `camera_stream` | `undistort` | bool | off | yes | the published picture is rectified with the checkerboard calibration (config/camera.json's intrinsics) and its camera_info then says no distortion; a no-op while the camera is uncalibrated, since there is nothing to undo. Rectifying crops to the largest all-valid rectangle, so the field of view narrows. THE MONO RIG's flag: a stereo head is rectified by its own stereo calibration (both eyes onto one pinhole with the rows aligned, which is what a disparity means at all), so the node refuses this one there rather than straighten a picture twice |
| `camera_stream` | `fold_mask` | bool | on | yes | stereo: rectified pixels past a fold of the calibration's undistortion map (the lens corners the board never reached) go out black, as no data, and the depth there is cut; off publishes the mirrored corners as before |
| `camera_stream` | `static_camera_tf` | bool | on | at start | base_link -> camera_link is broadcast from here; it goes off (ros/laptop.sh vslam --neck) when the board's neck node publishes that edge live from the servo encoders (neck_state, flag neck_tf), because two publishers of one edge fight |
| `contact_scan` | `contact_scan` | bool | on | yes | the contact line is published; off, the node is a subscriber that costs nothing — the costmap's own contact_layer.enabled is the other end of the same demo switch, and either one alone takes the camera's floor line out |
| `contact_scan` | `shadow` | bool | on | yes | the last floor pixel on a face stands a band's width UP that face, so its ray lands past the foot: on, that width is taken back off the range (pepin.contact.band_shadow); off is the raw boundary ray |
| `contact_scan` | `imu_lean` | bool | on | yes | the floor plane leans with the gyro as well as the accelerometer (pepin.lean: the lean of a wheel climbing a threshold is followed within a sample instead of being gated away as a push); off, the accelerometer alone, as it always has been |
| `depth_fusion` | `enabled` | bool | on | yes | frames are fused into the model; off, they are dropped |
| `depth_fusion` | `tof_rays` | bool | on | yes | the three ToF fans (/tof/<name>/scan) are written into the volume as rays from their own frames, through the camera's integrator: a return marks a surface at its range and carves the ray free up to it, +inf carves free out to the fan's trusted range (its range_max, pepin.tof_horizon.trusted_max_range), every fan at weight pepin.tof_rays.TOF_WEIGHT. Both costmaps then read what the whiskers saw out of the volume (/depth_marks, /camera_grid) like the camera's; off, the fans touch the volume at all and feed only the local costmap's own layers, as before 2026-10-01 |
| `depth_fusion` | `volume_frame` | choice: odom, map | odom | yes | which frame the volume is painted in. odom: every frame and revolution is placed by odom -> base_link (plus the camera's own edge) and nothing in the paint path reads map -> odom at all — no snapshot is read or written, align, the paint gates (fit_gate, lidar_fit_gate, paint_sigma_m) and follow_correction are inert, and the volume is a rolling window that slides onto the cart past window_recentre_m of config/fusion.json and forgets what leaves it. map: the room-sized model of before 2026-09-22, resumed from and saved to world_path, seated by the yaw search, gated by the tracker's fit and sigma and carried by the graph's bend. /fusion/surface carries this frame's own name; /depth_marks is in base_link either way. CHANGING THIS EMPTIES THE VOLUME: voxels painted in the other frame are a room drawn in coordinates nothing here shares |
| `depth_fusion` | `fit_gate` | bool | on | yes | camera frames are fused only while the tracker's pose is trusted (pepin.watch.PaintTrust: /localization_fit >= 0.50, HEARD within the source patience, sigma_xy <= paint_sigma_m where a sigma is published, and a map -> odom edge within a second of the frame); off, every frame is fused |
| `depth_fusion` | `lidar_fit_gate` | bool | on | yes | lidar revolutions are integrated only while the tracker's pose is trusted — the very test fit_gate applies to a camera frame (pepin.watch.PaintTrust); off, every revolution is integrated at whatever pose TF gives, which is what this node did until 2026-09-16. A withheld revolution is counted and never painted |
| `depth_fusion` | `imu_lean` | bool | on | yes | the cart's lean (pepin.lean, from /imu/data_raw) is followed with the gyro as well as the accelerometer, and a frame is placed with the lean at its stamp composed on base_link before the planar odometry instead of as if the cart stood level; the lidar's scan follows the same switch — its beams are walked as the 3D rays the leaning body sends them along, and lean_gate_deg drops the scans taken too far from level |
| `depth_fusion` | `self_heal` | bool | off | yes | a streak of 30 frames refused at the alignment bound empties the model, so it re-seeds from the next frame instead of staying frozen until a human resets it |
| `depth_fusion` | `align` | bool | on | yes | frame-to-model: a frame's lidar-height band is turned about the cart to fit the model before it is fused, and a frame whose best turn is the search's bound (+-4 deg) is refused |
| `depth_fusion` | `marks_source` | choice: volume, frame | volume | yes | where the camera's MARKS in the costmap come from (/depth_marks): volume, the accumulated model's own surface sliced around the cart at min_weight (pepin.volume_scan — the very surface /fusion/surface draws); frame, the latest /depth_scan relayed unchanged, which is what marked the costmap until 2026-09-21. Either way /depth_scan itself keeps CLEARING the layer: a single frame is the eyewitness of what is open now |
| `depth_fusion` | `marks_clear` | bool | off | yes | the fan also says where the volume is KNOWN OPEN: a second, clearing-only scan on /depth_free carrying, per bearing, the range of the last column the volume has observed FREE before the first column it has not (pepin.volume_scan.free_ranges). A bearing the volume cannot vouch for stays NaN, which clears nothing. Off, the topic is silent and the camera layer clears from the single frame alone, as it has since 2026-09-21 |
| `depth_fusion` | `grid_out` | bool | off | yes | publish the volume's current occupied columns (the /depth_marks rule) as grids the costmaps' camera_grid_layer only draws: /camera_grid, a square about the cart in the volume's frame, and /camera_grid_map with its _updates on the lattice of grid_map_topic; off, all three are silent |
| `depth_fusion` | `lidar_layer` | bool | on | yes | /scan is integrated into the volume at the lidar's plane (rays carve free space, returns mark a surface); off, the volume is the camera's alone, as it was |
| `depth_fusion` | `no_return_free` | bool | off | yes | a beam that came back with nothing carves free space out to the sensor's reach (an open door reads as open); off, it writes nothing at all |
| `depth_fusion` | `no_depth_free` | bool | on | yes | a camera pixel with NO depth carves free space along its own ray, from 0.20 m out to the source's own reach less one truncation, at no_depth_weight of what a measurement at that range weighs; off, a NaN pixel touches nothing at all, which is what this node did until 2026-09-22 |
| `depth_fusion` | `colour_fallback` | bool | on | yes | a surface point whose nearer voxel was never painted by a camera takes the OTHER neighbour's colour on /fusion/surface; off, it keeps the black that means 'no camera ever wrote here', which is what the cloud showed until 2026-09-22 |
| `depth_fusion` | `resume_volume` | bool | on | at start | a volume snapshot at world_path is loaded at start, so a room the cart has painted before comes back as it was left; off, the volume starts empty and grows from the sensors. world_path belongs to the graph DATABASE whose frame the voxels were painted in (rtabmap.db -> rtabmap.world.npz), so a fresh database means a fresh volume |
| `depth_fusion` | `view_gate` | bool | on | yes | a revolution taken from a place the volume has already integrated is not integrated again (pepin.worldmap.ViewGate: the pose must have moved a whole voxel at the scan's own farthest return before it counts as a new view); off, every revolution is painted, which is what this node did until 2026-09-18 |
| `depth_fusion` | `follow_correction` | bool | on | yes | the graph's optimisation moves the voxels, not only the pose: when the accumulated move of RTAB-Map's own node poses (/rtabmap/mapGraph, read at the newest shared node) differs from the one the volume is painted under by more than follow_correction_min_m / _min_deg, the whole content is carried rigidly by that difference before the next observation goes in. Every mode — the graph is the one source of truth about the room under World R, and the node poses are the only signal that says the ROOM moved rather than the cart having been found |
| `depth_fusion` | `follow_correction_law` | choice: blend, nearest | nearest | yes | how the move resamples the volume: blend is the fusion's own weighted average of the four source columns, nearest takes the one column the cell came from |
| `depth_stream` | `edge_filter` | bool | on | yes | flying pixels at object edges are dropped from the published depth and the scan; the law's beam pairs skip them regardless |
| `depth_stream` | `lidar_anchor` | bool | on | yes | the lidar's returns pair with the network's depth and fit the law; off, the last law is held (the failure mode of a lidar that stops) — with no law yet nothing is published until it is back on |
| `depth_stream` | `affine_law` | bool | on | yes | the network's depth through 1 / z = a / D + b, fitted on the pooled pairs; off, the raw network's depth goes out unwithheld |
| `depth_stream` | `floor_anchor` | bool | on | yes | pixels within centimetres of the floor plane snap to it in the published image (the scan is built before it); the plane leans with the cart, from the IMU's up vector |
| `depth_stream` | `depth_backend` | choice: remote, local, auto | local (env PEPIN_DEPTH_BACKEND) | yes | where the network runs: local (the CPU model in this container), remote (the laptop's GPU service, ros/depth_host.sh), auto (the service while it answers, the CPU model while it does not) |
| `depth_stream` | `stereo_matcher` | choice: sgbm, raft | raft (env PEPIN_STEREO_MATCHER) | yes | which engine turns the two eyes into a disparity: sgbm (OpenCV's semi-global block matcher, in this container) or raft (RAFT-Stereo on the laptop's GPU through the same host the mono network uses, ros/depth_host.sh stereo). Both are built at start and this picks which one answers the next pair, so an A/B needs no restart; a pair the host cannot answer falls to sgbm and the report line counts it. Under depth_source: network it does nothing |
| `depth_stream` | `law_watch` | bool | off | yes | the affine law is fitted on the lidar's pairs and printed, and the depth is published exactly as the source measured it; no frame waits for a law |
| `depth_stream` | `imu_lean` | bool | on | yes | the cart's lean (pepin.lean, from /imu/data_raw) is followed with the gyro as well as the accelerometer and carried into the scan's carry and the camera's place in the map; off, the floor plane leans with the accelerometer alone, as it always has, and nothing else is leaned |
| `depth_stream` | `camera_tf_latest` | bool | on | yes | take the newest base_link <- camera_optical edge TF holds (at most 1 s old, and only while the head has not moved over the half second before it) when the frame's own stamp is not covered yet, instead of waiting CARRY_WAIT_S for it; off: the old wait at the exact stamp |
| `depth_stream` | `fan_floor_gate` | choice: band, contact, off | band | yes | what keeps the floor out of /depth_scan: band (the default) raises the band's lower edge with the floor's own noise, 3 sigma of it (pepin.contact.fan_min_z), contact drops every mark nearer than that bearing's floor-contact range (pepin.contact.gate_by_contact, a range no depth law enters), off is the flat 0.15 m edge the fan always had. The report line counts the bearings gated |
| `depth_stream` | `scan_honours_pan` | bool | on | yes | fold /depth_scan onto the floor through the neck's pan: the fan's bearings turn with the head and its angular window turns with them, so angle_min comes out at pan - 40 deg instead of -40. The pan is the yaw of the same base_link <- camera_optical edge the volume path reads (camera_tf_latest); with none at the frame's stamp the last edge TF held stands in, and only while TF never had one the config mount's straight-ahead yaw (the report line counts both). Off: the fan is projected as if the head looked along the cart's x, whatever the encoders say |
| `depth_stream` | `depth_reach` | bool | on | yes | the PUBLISHED depth image is NaN past depth_reach_m: the camera answers for its own data and says nothing where it does not vouch for the range. /depth_scan is unaffected (it is capped at the same range already) and so is every law — the gate is applied to the image on its way out, after the pipeline |
| `goal_server` | `places_from_the_file` | bool | off | yes | before the graph's book of places has been heard, a name is answered from the yaml beside the map (coordinates of the frozen-grid era); off, a name is refused until the book arrives, with that reason |
| `goal_server` | `jump_clear` | bool | off | yes | map -> odom is read from TF five times a second and, when it STEPS further than 0.10 m, Nav2's local costmap is emptied ("/local_costmap/clear_entirely_local_costmap", asynchronously, at most once per 1 s): the marks in that grid were laid where the cart used to be. The step in that edge is the correction alone — the cart's own motion lives in odom -> base_link — whoever published it. Off, nothing reads the edge and no listener is started for it |
| `goal_server` | `pose_topic` | bool | on | yes | the pose this node reads out of TF is republished as /pose (geometry_msgs/PoseStamped in map, 5 Hz, stamped with the transform's own stamp), so the other nodes on this board can have the pose without a TF listener of their own. Inert on a split stack, where the reader is on the other machine and reads the edge itself. Off, nothing is published and this node's listener goes back to being started on the first ask |
| `goal_server` | `controller` | choice: mppi, rpp, rpp_shim | rpp_shim | yes | what follows the plan: mppi is Nav2's MPPI controller for every planner, held to the mark's heading by the yaw-checking goal checker; rpp is each planner's own Regulated Pure Pursuit from PLANNERS, ending on position alone as before 2026-09-23; rpp_shim is the reversing RPP inside Nav2's RotationShimController, which turns the cart to the mark's heading in place once it is inside the goal tolerance. Published latched on controller_selector and goal_checker_selector, so a change is read by the behaviour tree at its next tick |
| `goal_server` | `start_needs_placement` | bool | on | yes | a goal or a mark waits for the laptop's word on /localization/placement (pepin_bringup.rtabmap_frame, latched) that this start of RTAB-Map is PLACED — a node of the loaded map recognised, or an operator's seed — and nothing heard is refused like not placed. ros/tools/goto_ros.py asks this node for this same flag before its own preflight. Off, a fresh map -> base_link is enough, as before 2026-09-23 |
| `goal_server` | `cancel_every_goal` | bool | on | yes | a cancel on the socket also asks both navigators' own cancel services (navigate_to_pose and navigate_through_poses, <action>/_action/cancel_goal) for EVERY goal — a zero goal id, whoever sent it — and answers what each said (``navigators``); off, it cancels only the goal this node sent, as before 2026-09-25 |
| `goal_server` | `lidar_watch` | bool | on | yes | the lidar's driver is restarted when its port is there and no scan has come for 5 s, and once per absence of the port so it respawns idle instead of spinning a core (pepin.lidar_watch); `where` says `lidar` either way |
| `marks_audit` | `marks_audit` | bool | on | yes | the audit runs; off, the node keeps its subscriptions and computes, publishes and reports nothing |
| `marks_audit` | `inscribed_counts` | bool | off | yes | the inflation's 99 band (costmap 253, INSCRIBED_INFLATED_OBSTACLE) is judged as a mark too; off, only the 100s a sensor actually wrote |
| `marks_audit` | `phantom_cloud` | bool | on | yes | the camera-only cells are published as red points on /marks_audit/phantoms; off, only the counts go out |
| `neck_state` | `neck_tf` | bool | on | yes | base_link -> camera_link is published live from the neck's encoders; the laptop's camera node must then run with ros/laptop.sh vslam --neck, or two nodes publish that edge |
| `places` | `publish_places` | bool | on | yes | the resolved places are published on /places whenever the graph moves; off, the book is still kept and marked but nothing is published and every consumer falls back to the coordinates beside the map |
| `places` | `label_nodes` | bool | on | yes | a mark also sets RTAB-Map's own label on the node (set_label), which is what makes the place a thing in its tools and in its set_goal; off, only our own book records the node id and the offset |
| `rtabmap_frame` | `registration_follows_snapshots` | bool | on | yes | RTAB-Map's Reg/Strategy follows what the snapshots carry (/sensor_pack/state): a scan and a picture mean visual then ICP (2), a scan alone means ICP (1), no scan means visual (0), switched live through the node's own parameter path on a change that has held for the hold the state carries. Off, the strategy stays whatever the launch table set and this node only reports what it would have asked for |
| `rtabmap_frame` | `atomic_parameter_sets` | bool | on | yes | a parameter set this node hands RTAB-Map (a strategy with its visual features, a visual set alone, a memory mode's pair) goes as ONE /rtabmap/rtabmap/set_parameters_atomically request, which rtabmap_slam applies as one /parameter_events notification and one parseParameters. Off, it goes as one set_parameters request, which rclcpp applies one parameter at a time — each its own event and its own parseParameters — with RGBD/LoopClosureReextractFeatures ordered first when it turns off and last when it turns on |
| `rtabmap_frame` | `visual_features` | choice: orb, xfeat | xfeat | yes | which features RTAB-Map's VISUAL registration (Reg/Strategy 0, the camera-only strategy, and the visual half of 2) matches when it checks a node the words recognised. xfeat: XFeat keypoints matched by LighterGlue, re-extracted from both nodes' stored pictures at loop-closure time (Vis/FeatureType 15, Vis/CorNNType 6, RGBD/LoopClosureReextractFeatures true); the database is only read. orb: the database's own GFTT/ORB words, the launch table's values. Sent with the strategy and changed live; under ICP alone, and unless RTAB-Map is certainly localising (told so and answered, no switch to mapping waiting), the set is always orb, and a switch to mapping waits until orb's set is in force. xfeat needs the pepin-laptop:xfeat image (/opt/xfeat/rtabmap_xfeat.py); in another image this node sends orb and the report line says why |
| `rtabmap_frame` | `visual_confirm` | choice: rtabmap, aggressive, single | aggressive | yes | how RTAB-Map CONFIRMS a localisation while the camera registers alone: RTAB-Map 0.22 delays a first good localisation into its odometry cache and accepts it only with a second one inside RGBD/MaxOdomCacheSize updates, and that second try has to reach Rtabmap/LoopThr. rtabmap: its stock 0.11 and 10. aggressive: Rtabmap/LoopThr 0.05, the threshold the first try already used, so the second comes on the next update; the confirmation stays. single: RGBD/MaxOdomCacheSize 0, the first good localisation is accepted. Sent with the visual strategy while the database localises and changed live; under ICP and while it maps, always rtabmap |
| `rtabmap_frame` | `visual_proximity` | bool | on | yes | whether a localised camera also registers every update against the database nodes near its pose (RGBD/ProximityBySpace), each an XFeat re-extraction and a LighterGlue match; off, only the words' own hypothesis is registered, one at most per update. Sent with the visual strategy while the database localises and changed live; under ICP and while it maps, always on (the lidar's proximity links are cheap and most of the graph's) |
| `rtabmap_frame` | `registration_backend` | choice: service, local, auto | auto (env PEPIN_REGISTRATION_BACKEND) | yes | where RTAB-Map's XFeat keypoints and LighterGlue matches are computed (the xfeat visual features): service — the localisation service on the laptop's GPU (pepin.localization_service, PEPIN_MODELS_URL), no features when it does not answer; local — in RTAB-Map's own process on the Docker VM's CPU, as before 2026-09-24; auto — the service, and a call it does not answer computed locally. Written to /tmp/pepin/registration.json, which RTAB-Map's adapters read on every call (one stat): live, no restart of RTAB-Map |
| `rtabmap_frame` | `place_recognition` | choice: words, descriptor | descriptor | yes | how RTAB-Map finds WHICH database node a picture is (its likelihood, before any registration): words — the ORB bag of words' TF-IDF (Kp/TfIdfLikelihoodUsed true, Rtabmap/VirtualPlaceLikelihoodRatio 0, RTAB-Map's defaults); descriptor — the dot product of the nodes' learned place descriptors as z-scores (false and 1; rtabmap Memory::computeLikelihood -> Signature::compareTo, Rtabmap::adjustLikelihood), which sensor_pack attaches to every snapshot. descriptor is sent ONLY when it cannot abort RTAB-Map: its core carries ros/patches/rtabmap-keep-global-descriptors.patch (the marker /opt/rtabmap_patches/keep-global-descriptors), the snapshots carry one each (/sensor_pack/place) and the database's census at this start (PEPIN_PLACE_CENSUS, taken by the launch before RTAB-Map opens the file) says every node carries exactly one of the same length — and only while the camera snapshots are described (descriptor_null_share); otherwise the words, and the report line says why. Live |
| `rtabmap_frame` | `start_needs_placement` | bool | on | yes | what goes out on /localization/placement (latched) says this start of RTAB-Map is PLACED only once an update has recognised a node of the database it loaded, or an operator's seed (/rtabmap/initialpose) has been heard since its first update — or it loaded an empty database, whose start pose is the map's origin. The board's goal clients (ros/tools/goto_ros.py, pepin_bringup.goal_server) refuse a goal until then, saying to seed or to let the camera see a mapped place. Off, every start counts as placed: RTAB-Map's pose is taken as it is |
| `run_recorder` | `planner_records` | bool | on | yes | what the PLANNER saw goes on the tape too: the global costmap (run-length encoded, at most one grid per new plan), the goal status of Nav2's three actions (navigate_to_pose, compute_path_to_pose, follow_path); off, the tape holds what it held before 2026-09-18 |
| `run_recorder` | `loc_from` | choice: pose_topic, tf | pose_topic | yes | where the tape's `loc` rows come from: pose_topic, the goal server's /pose (it parses /tf for navigation anyway and republishes what it reads); tf, this node's own TF listener at 0.2 s, which is what it ran until 2026-09-22. The rows are identical either way — same fields, same 5 Hz, source 'tf' in both, because both are that same edge |
| `sensor_pack` | `sensor_pack` | bool | on | yes | snapshots are published; off, the node subscribes and counts and RTAB-Map is fed nothing at all |
| `sensor_pack` | `sources` | list of: camera, lidar | camera,lidar | yes | which sensors may enter a snapshot: the live A/B for camera-only and lidar-only mapping, with no restart and without muting a publisher |
| `sensor_pack` | `tf_retry` | bool | on | yes | a member whose transform TF cannot answer for yet does not cost the snapshot: the moment is put back and offered again on the next arrival, until it has left that member's own pairing patience (pair_periods of its measured period). Off, the member is dropped at once and the snapshot goes out without it — the behaviour of before 2026-09-19 |
| `sensor_pack` | `global_descriptor` | choice: auto, on, off | auto (env PEPIN_GLOBAL_DESCRIPTOR) | at start | whether every snapshot carries exactly one rtabmap_msgs/GlobalDescriptor (type 1): the place vector of its picture (place_descriptor), or the null descriptor (zeros) when it has no picture or no vector came in place_timeout_s; said latched on /sensor_pack/place. auto: exactly when this image's RTAB-Map keeps a node's descriptor across the reload of its data (the marker /opt/rtabmap_patches/keep-global-descriptors, ros/xfeat/patch_rtabmap.sh); on: always; off: never, the snapshots of before 2026-09-24 |
| `sensor_pack` | `place_descriptor` | bool | on | yes | a camera snapshot's descriptor is the localisation service's /place vector of its picture (pepin.localization_service, BoQ on the laptop's GPU); off, every snapshot carries the null descriptor and the service is not asked |
| `tof_bridge` | `range_as` | choice: scan, range | scan | yes | what Nav2 is fed with: scan also publishes each cone on /tof/<name>/scan as a small LaserScan fan for an ObstacleLayer; range publishes nothing there, which is the node before 2026-09-21 and needs the three RangeSensorLayer blocks back in the local costmap's plugins list. The sensor_msgs/Range topics are published either way |
| `visual_odometry` | `vo_publish` | bool | on | yes | the gated visual odometry leaves this laptop as /vo, where the board's EKF fuses it as a third input beside the wheels and the gyro; off, the node still measures and reports and the EKF is exactly what it was without it |
| `visual_odometry` | `vo_covariance` | choice: dynamic, constant, rtabmap | dynamic | yes | whose covariance rides on the published pose: `dynamic`, the registration's own sigma and the depth scale's share of the step just taken added in quadrature (pepin.visual_odometry.scaled_covariance); the documented constant (vo_sigma_m, vo_yaw_sigma_deg); or the one rtabmap's registration computed, untouched |

### The flags one by one

#### `base_bridge`

- **`imu_publish`** — bool, default on
  - *What:* the MPU6050's readings leave the bridge as /imu/data_raw, where the EKF fuses index 11 (the yaw rate) and nothing else; off, the chip is still read and its bias still estimated, but no message is published. THE PYTHON BRIDGE PUBLISHES NO IMU AT ALL — here the flag only exists so the node's table is the same table whichever bridge robot.launch.py started; the C++ bridge is the one that reads the chip
  - *Default:* on — on, because the gyro is the heading: the wheels over-report a turn in place by 10-25 % on carpet, and odom0's vyaw — the only other yaw-rate source, live since 2026-09-15 — carries about 4 % of the weight beside it (ros/params/ekf.yaml)
  - *On when:* always, unless the point of the run is what the stack does without a gyro
  - *Off when:* for one test of the heading on the wheels alone, or to see an EKF meet its sensor_timeout on a source that is simply gone; unmute and the rate is back within one IMU period (50 Hz)
- **`odom_publish`** — bool, default on
  - *What:* the base server's state line leaves the bridge as /odom and, while publish_tf is on, as the odom -> base_link transform; off, the wheels are still read and still commanded, and both go silent together — a transform still broadcast from a silent /odom is a state no sensor failure produces
  - *Default:* on — on, because /odom is the only source of speed this filter has: odom0 fuses vx and vy at 0.001 (m/s)^2 and, since 2026-09-15, vyaw; ax and ay are off (a mount bias of -0.229 to +0.066 m/s^2 that no covariance can answer), so with /odom silent past the EKF's sensor_timeout of 0.5 s the filter has no velocity measurement left at all
  - *On when:* always, unless the run is about what the stack does with dead wheel odometry
  - *Off when:* to watch a consumer meet a silent odometry — the EKF's sensor_timeout, Nav2's TF lookups, the tracker's dead reckoning — without stopping the base server; unmute and /odom is back on the next state line (20 Hz)

#### `camera_stream`

- **`undistort`** — bool, default off
  - *What:* the published picture is rectified with the checkerboard calibration (config/camera.json's intrinsics) and its camera_info then says no distortion; a no-op while the camera is uncalibrated, since there is nothing to undo. Rectifying crops to the largest all-valid rectangle, so the field of view narrows. THE MONO RIG's flag: a stereo head is rectified by its own stereo calibration (both eyes onto one pinhole with the rows aligned, which is what a disparity means at all), so the node refuses this one there rather than straighten a picture twice
  - *Default:* off — default by design, unmeasured: the camera is calibrated (45 views, rms 0.230 px, fx 724.1, fy 726.8, cx 652.0, cy 374.0, k1 -0.150, k2 -0.129, k3 +0.092, HFOV 82.9 deg, 2026-09-13), but the straightened picture has never been compared with the raw one on the robot, and nobody has measured what the crop costs in field of view
  - *On when:* when a consumer needs straight lines — a checkerboard, a marker, a recogniser that assumes a pinhole
  - *Off when:* wherever a consumer was measured in the raw picture's optics (the depth pipeline's law was fitted there), and wherever the field of view matters more than straight lines
- **`fold_mask`** — bool, default on
  - *What:* stereo: rectified pixels past a fold of the calibration's undistortion map (the lens corners the board never reached) go out black, as no data, and the depth there is cut; off publishes the mirrored corners as before
  - *Default:* on — measured 2026-09-24 (scratch/stereo/fold_check.py): the left eye's map folds back in both bottom corners, 0.4 % of the picture, the 'crack' Artem saw, and camera phantoms lined up along it
  - *On when:* always on a stereo head calibrated without the corners
  - *Off when:* a calibration that covers the corners (fold_check.py finds no fold)
- **`static_camera_tf`** — bool, default on, not live
  - *What:* base_link -> camera_link is broadcast from here; it goes off (ros/laptop.sh vslam --neck) when the board's neck node publishes that edge live from the servo encoders (neck_state, flag neck_tf), because two publishers of one edge fight (not live: set at the next start)
  - *Default:* on — default by design, unmeasured: an ownership rule rather than a tuning — one edge, one publisher. Not live because a static transform cannot be withdrawn once it is sent, so the choice is made at start
  - *On when:* when the neck does not publish the edge: a fixed head, or the neck node down
  - *Off when:* whenever neck_state runs with neck_tf on — at start, since this one cannot be taken back

#### `contact_scan`

- **`contact_scan`** — bool, default on
  - *What:* the contact line is published; off, the node is a subscriber that costs nothing — the costmap's own contact_layer.enabled is the other end of the same demo switch, and either one alone takes the camera's floor line out
  - *Default:* on — default by design, unmeasured as a switch: it is one end of a pair, so the line can be taken out of the map in one command from either side. The line's own accuracy is measured (see max_range), but the validation drive that was asked for — an open floor showing about 0 % marks and a taped box at 1.2, 1.6 and 2.0 m landing within 10 cm — has not been run
  - *On when:* wherever the camera's floor line should be seen: the redundancy demo, or a low obstacle the lidar's plane looks over
  - *Off when:* to take the line out in one command, and on a run where this node's cost must be zero
- **`shadow`** — bool, default on
  - *What:* the last floor pixel on a face stands a band's width UP that face, so its ray lands past the foot: on, that width is taken back off the range (pepin.contact.band_shadow); off is the raw boundary ray
  - *Default:* on — the uncorrected ray reports an obstacle about 10 % of its range too far — at 1.5 m the band is 0.120 m tall and the raw ray lands at 1.66 m, 16 cm of phantom clearance, in the direction a costmap pays for. That 10 % is geometry on this mount, not a field A/B: no run compares the line against the lidar with the correction off
  - *On when:* wherever the line feeds a costmap: an obstacle reported too far is the failure a bumper pays for
  - *Off when:* to see the raw boundary ray, or on a mount whose band is thin enough that the correction is inside the noise
- **`imu_lean`** — bool, default on
  - *What:* the floor plane leans with the gyro as well as the accelerometer (pepin.lean: the lean of a wheel climbing a threshold is followed within a sample instead of being gated away as a push); off, the accelerometer alone, as it always has been
  - *Default:* on — on: the gyro's sign was verified by hand on 2026-09-13 (the cart tipped nose-down read pitch +10.5 deg, left-side-down read roll -9.4 deg, the fast path following at once with quality 0.9 while held; scratch/lean_tip_test.txt), and the gyro's zero offset is learned. Off, the floor anchor keeps the accelerometer-only lean, which by design ignores any tip shorter than 10 s
  - *On when:* after a hand tip through a known angle shows the reported lean following it the right way; the gain is the threshold case, where a real lean is followed within a sample
  - *Off when:* wherever the reported lean disagrees with the cart's visible attitude

#### `depth_fusion`

- **`enabled`** — bool, default on
  - *What:* frames are fused into the model; off, they are dropped
  - *Default:* on — default by design, unmeasured: the kill switch, so the model can be stopped growing without stopping the node, the camera or the report line
  - *On when:* whenever the fused surface or the volume's map is wanted
  - *Off when:* to freeze the model where it stands — a snapshot to save, a picture to read, a run where the camera is carried by hand
- **`tof_rays`** — bool, default on
  - *What:* the three ToF fans (/tof/<name>/scan) are written into the volume as rays from their own frames, through the camera's integrator: a return marks a surface at its range and carves the ray free up to it, +inf carves free out to the fan's trusted range (its range_max, pepin.tof_horizon.trusted_max_range), every fan at weight pepin.tof_rays.TOF_WEIGHT. Both costmaps then read what the whiskers saw out of the volume (/depth_marks, /camera_grid) like the camera's; off, the fans touch the volume at all and feed only the local costmap's own layers, as before 2026-10-01
  - *Default:* on — a pillow under the cart's nose is below the lidar's plane and too near for the stereo rig, so only the ToF saw it, only the local costmap's reflex layers knew, and the global planner kept routing through it (2026-10-01). The volume is the one memory of what is where and both costmaps already read it, so the whiskers write it too. At weight 1.0 a hit speaks in the marks after 4 fans (the node's min_weight 4.0, 0.27 s at 15 Hz) and a saturated one (max_weight 20) is carved by 15 misses, 1.0 s (pepin.tof_rays). Checked live 2026-10-01 with the camera's frames off: the front whisker at 0.75 m put its marks at 0.75-0.78 m inside its cone, none under the cart; a hand at 0.15 m marks (the ToF's own near limit, 0.08 m)
  - *On when:* on every drive: the planner must know what the whiskers saw
  - *Off when:* a sensor reading its own surroundings (a cable hanging in its cone) paints phantoms into the volume that the planner cannot pass; the local costmap's own layers still get the fans either way
- **`volume_frame`** — choice: odom, map, default odom
  - *What:* which frame the volume is painted in. odom: every frame and revolution is placed by odom -> base_link (plus the camera's own edge) and nothing in the paint path reads map -> odom at all — no snapshot is read or written, align, the paint gates (fit_gate, lidar_fit_gate, paint_sigma_m) and follow_correction are inert, and the volume is a rolling window that slides onto the cart past window_recentre_m of config/fusion.json and forgets what leaves it. map: the room-sized model of before 2026-09-22, resumed from and saved to world_path, seated by the yaw search, gated by the tracker's fit and sigma and carried by the graph's bend. /fusion/surface carries this frame's own name; /depth_marks is in base_link either way. CHANGING THIS EMPTIES THE VOLUME: voxels painted in the other frame are a room drawn in coordinates nothing here shares (one of: odom, map)
  - *Default:* odom — MEASURED 2026-09-21, and it is why this flag exists. The volume's job is LOCAL OBSTACLE MEMORY (nvblox's local mapper beside a pose graph, STVL's pattern), and local memory that depends on global localisation inherits every one of its mistakes. Painted in map and kept, it accumulated the walls of some twenty re-seatings of the tracker in one evening: the aligner sat at its +-4 deg bound (240 refusals), revolutions were withheld at sigma 2.98 m while the slice kept publishing, and /depth_marks put 300-650 lethal cells around the cart that the lidar never saw — 199 of 219 outside the camera's own 94 deg cone and 116 of them BEHIND the cart, measured layer by layer against Nav2's own grids (scratch/nav2_hang/layer_blame.py). With the file set aside the same drive marked 0 alien cells. In odom there is no global pose in the path to be wrong about: the odometry pose is always the pose, which is why the gates that ask whether the tracker is trustworthy have nothing to judge here and say so in the report line instead of silently passing. What the window costs when it slides is a copy of the overlap: 0.3-0.9 ms measured on the node's 120x120x34 test grid and 4.7 ms on the live 280x250x34 one (tests/unit/test_tsdf.py), timed into that same line
  - *On when:* odom, everywhere the cart drives: the costmap's camera marks then remember only what this odometry run has seen around the cart, and a re-seating of the global pose cannot move a single voxel of them
  - *Off when:* map for a mapping run whose product is the painted room itself — a surface to look at in Foxglove, a volume to resume tomorrow — and for the A/B of 2026-09-21, on a cart nobody is driving
- **`fit_gate`** — bool, default on
  - *What:* camera frames are fused only while the tracker's pose is trusted (pepin.watch.PaintTrust: /localization_fit >= 0.50, HEARD within the source patience, sigma_xy <= paint_sigma_m where a sigma is published, and a map -> odom edge within a second of the frame); off, every frame is fused
  - *Default:* on — a fit that STOPS arriving leaves its last good value in this node for ever, so the gate asks when it was heard as well as what it said: the session whose routes died at 19:17 on 2026-09-15 went on fusing at a frozen pose for hours. Otherwise: where a tracker speaks, and the off state is measured: in the first online-SLAM session, where nobody publishes /localization_fit and the gate had to come off, the fused floor came out rough — offset +4.7 cm, sd 5.5 cm, 34 % within 3 cm at a tilt of 0.61 deg — against sd 3.3 cm and 71 % within 3 cm in the known-map mode with a centimetre tracker pose. The 0.50 itself is the drive rung of the tracker's own ladder (pepin.watch: blind 0.30, drive 0.50, lost 0.55), inherited, not swept for fusion
  - *On when:* in the known-map modes (split, vision), where the board's tracker publishes the fit: it keeps a frame taken while the pose was wrong out of the model. The launch brings it up on there and off in SLAM mode; this is how to put it back on by hand
  - *Off when:* in SLAM mode, where RTAB-Map owns the pose and no tracker speaks — with the gate on nothing is ever fused there. vslam.launch.py passes fit_gate:=false in that mode, so nobody has to remember it at the start of a session
- **`lidar_fit_gate`** — bool, default on
  - *What:* lidar revolutions are integrated only while the tracker's pose is trusted — the very test fit_gate applies to a camera frame (pepin.watch.PaintTrust); off, every revolution is integrated at whatever pose TF gives, which is what this node did until 2026-09-16. A withheld revolution is counted and never painted
  - *Default:* on — measured by its absence, on the volume itself. A revolution places a wall and carves free space along its beams, in the MAP frame, and a TSDF cannot be un-integrated — so a revolution written at a wrong pose does not add noise, it deletes the room. The camera path has been gated since the beginning and the lidar path never was; on 2026-09-15 the laptop's routes from the board died at 19:17 and the fusion went on integrating at the last pose TF held, and the snapshot at the end of that session keeps 52.9 % of the saved map's walls, has carved 2070 of them free, and the node's own tracker replayed on its slice seats a median 1.90 m from where the file puts it, re-seating 3.7-3.8 m off on two of four tapes (scratch/volume_vs_file_seating.py). That measurement is also part of why nothing localises against this volume any more. What it costs a HEALTHY drive was measured too, on the same four tapes with the predicate applied to every recorded revolution (scratch/paint_gate_on_a_tape.py): 452 of 10048 withheld, 4.5 %, and that is an upper bound — 439 of them are the recorded pose's own gap (the tape carries the tracker's pose at about 2 Hz while the node reads a map -> odom edge broadcast at 20 Hz), leaving 13 revolutions, 0.13 %, where the tracker really had gone quiet past the source patience. Not one revolution of the four drives was refused for a LOW fit: a drive the stack was willing to make paints as it always did
  - *On when:* the default, wherever a tracker publishes a fit: the volume then holds only what was seen from a pose the stack was willing to drive on
  - *Off when:* in SLAM mode, where no tracker speaks and the gate would integrate nothing at all (vslam.launch.py passes lidar_fit_gate:=false there, as it does fit_gate) — and to reproduce the old behaviour for a comparison, on a volume nobody will navigate on
- **`imu_lean`** — bool, default on
  - *What:* the cart's lean (pepin.lean, from /imu/data_raw) is followed with the gyro as well as the accelerometer, and a frame is placed with the lean at its stamp composed on base_link before the planar odometry instead of as if the cart stood level; the lidar's scan follows the same switch — its beams are walked as the 3D rays the leaning body sends them along, and lean_gate_deg drops the scans taken too far from level
  - *Default:* on — on: the gyro's sign was verified by hand on 2026-09-13 (the cart tipped nose-down read pitch +10.5 deg, left-side-down read roll -9.4 deg, the fast path following at once with quality 0.9 while held; scratch/lean_tip_test.txt), and the gyro's zero offset is learned. Off, the floor anchor keeps the accelerometer-only lean, which by design ignores any tip shorter than 10 s
  - *On when:* after a hand tip through a known angle shows the reported lean following it the right way and returning to zero
  - *Off when:* wherever the lean in the report line disagrees with the cart's visible attitude; off, the lean is still estimated and reported, only not applied
- **`self_heal`** — bool, default off
  - *What:* a streak of 30 frames refused at the alignment bound empties the model, so it re-seeds from the next frame instead of staying frozen until a human resets it
  - *Default:* off — off: measured harmful in its first evening (2026-09-11). It was written after one freeze (the head two hours at 31.5 deg against the config's 26, the law swinging 0.94-1.60: 284 refusals in one 30 s window, a surface 40 s stale, one hand-sent /fusion/reset brought back 273 frames per 30 s) — and then fired three times in the next two hours (19:54, 19:55, 20:05) on ordinary turns and a lidar-off test, wiping a good model each time: 30 frames at the bound is 3 s, which any pivot reaches. The cure for the freeze it was written for was the mount (0.383 m) and the TF camera pose, not the wipe
  - *On when:* only with a much longer streak (a minute) and only at rest — as written it is a model-wiper; until then a stale surface is reset by hand (/fusion/reset)
  - *Off when:* always, as shipped: the report line keeps 'at bound N' visible, and a model that stops accepting frames is a mount or pose problem to fix, not to hide
- **`align`** — bool, default on
  - *What:* frame-to-model: a frame's lidar-height band is turned about the cart to fit the model before it is fused, and a frame whose best turn is the search's bound (+-4 deg) is refused
  - *Default:* on — every A/B favours it by a centimetre or two of local surface thickness — 14.0 cm off against 12.1 on after the scan-carry fix, 12.6 against 11.6 in the demo, with RTAB-Map's cloud on the same turns at 17.5-20.0 cm — and the score curve on live frames peaks where it should (0.498 at 0 deg against 0.150 at either +-4 bound). The win is small, and it was once entirely fake: before the carry fix 89 % of frames answered AT the bound with a 4.00 deg median turn
  - *On when:* on for a model that must stay thin enough to read a wall's face
  - *Off when:* where the pose is already better than the search can be (a graph's corrections in SLAM mode), or to prove that a thick surface is the pose's fault: off, no frame is turned and none is refused
- **`marks_source`** — choice: volume, frame, default volume
  - *What:* where the camera's MARKS in the costmap come from (/depth_marks): volume, the accumulated model's own surface sliced around the cart at min_weight (pepin.volume_scan — the very surface /fusion/surface draws); frame, the latest /depth_scan relayed unchanged, which is what marked the costmap until 2026-09-21. Either way /depth_scan itself keeps CLEARING the layer: a single frame is the eyewitness of what is open now (one of: volume, frame)
  - *Default:* volume — the first stereo drive measured what one frame is worth as a mark (tape ros/maps/rec/0415_*): SGBM on the herringbone parquet answers small blobs of disparity 2-5 px too large, which lift FLOOR pixels to 0.15-0.24 m — inside the band the fan marks in — at about one false bearing a frame, a different bearing each time. In the costmap that is 100-300 lethal cells the lidar never saw, 115 'collision ahead' a minute and 44 recoveries in one drive. The same frames fused into the volume look clean, because fusing is what a single opinion cannot survive: a weighted average and the free space every later ray carves through the blob. So the marks come from the model and the clearing stays with the frames — the nvblox arrangement (a probabilistic volume, a 2D slice of it, the costmap), and no floor-specific rule anywhere in it
  - *On when:* volume: wherever the camera layer marks at all. A mark then needs the same agreement a point of /fusion/surface needs, and the bearings behind the head are answered too — the volume remembers the table the cart has driven past
  - *Off when:* frame reproduces the pre-2026-09-21 costmap exactly (the fan itself, marks and all) without a restart: the A/B for whether a missing mark is the volume's fault, and the way back if the volume is ever seen to hold a ghost
- **`marks_clear`** — bool, default off
  - *What:* the fan also says where the volume is KNOWN OPEN: a second, clearing-only scan on /depth_free carrying, per bearing, the range of the last column the volume has observed FREE before the first column it has not (pepin.volume_scan.free_ranges). A bearing the volume cannot vouch for stays NaN, which clears nothing. Off, the topic is silent and the camera layer clears from the single frame alone, as it has since 2026-09-21
  - *Default:* off — OFF, because the measurement that would justify it says it would do almost nothing. The case FOR it is real: the fan marks over the whole turn while /depth_scan clears only the head's forward 83 deg, so on run 0434's turn unbacked lethal cells were born at 109/s against 27/s standing, 52 % of them BEHIND the cart where nothing can ever raytrace them away, and the count climbed 24 -> 904 in 38 s (scratch/one_localiser/tape_0434_turn.py). But a clearing ray stops at the first column that is not open, and on the parked cart of 2026-09-23 the volume held an occupied column on 717 of 720 bearings: where the camera's own frame shares a bearing with a mark it AGREES with it within 0.20 m 96 % of the time and sees past it 3 % (scratch/one_localiser/live_fan_vs_lidar.py), and on the saved room volume the walk could vouch for 87 bearings of 720. The marks are not stale memory the volume has already carved — they are what the volume currently holds, and clearing cannot remove what the model still believes
  - *On when:* after the near marks themselves are answered: with the volume no longer holding a shell at 0.5-0.75 m on every bearing, the walk reaches past it and this is what stops the ratchet behind the cart. Turn it on together with the yaml's depth_free source and watch 'clear' in the report line rise off its floor
  - *Off when:* as shipped, and whenever a cell must not be erased by the camera's own memory: silent topic, and the layer clears from /depth_scan as it did before
- **`grid_out`** — bool, default off
  - *What:* publish the volume's current occupied columns (the /depth_marks rule) as grids the costmaps' camera_grid_layer only draws: /camera_grid, a square about the cart in the volume's frame, and /camera_grid_map with its _updates on the lattice of grid_map_topic; off, all three are silent
  - *Default:* off — off until a drive has measured it: the nvblox pattern against camera_layer's own copies of the memory, smeared by map -> odom jumps and wiped by the tree's clears (journal 2026-09-24, scratch/costmap_split)
  - *On when:* with camera_grid_layer on and camera_layer off in both costmaps: ros/camera_grid.sh on
  - *Off when:* ros/camera_grid.sh off, back to /depth_marks alone; turning it off publishes one empty grid on each topic so a layer left on holds nothing stale
- **`lidar_layer`** — bool, default on
  - *What:* /scan is integrated into the volume at the lidar's plane (rays carve free space, returns mark a surface); off, the volume is the camera's alone, as it was
  - *Default:* on — replayed from tape 0171 into an empty volume the lidar layer reproduces the saved map's walls to a median 0.0 cm, p90 13.0 cm, 79.3 % within one cell, and where the volume says free the saved map agrees 91.3 % of the time — for 1 ms a scan (575 scans in 0.8 s). It is also protected from the camera: 0 of 13499 lidar cells were changed by depth, while the camera filled 922 cells the lidar never reached
  - *On when:* on wherever the surface must show what the lidar knows, which is every mode: the beams are the only metric truth in the volume
  - *Off when:* to measure the camera alone — what the depth adds, and where it lies
- **`no_return_free`** — bool, default off
  - *What:* a beam that came back with nothing carves free space out to the sensor's reach (an open door reads as open); off, it writes nothing at all
  - *Default:* off — default by design, unmeasured: no false-carve rate was ever taken, and with the real /scan the branch is unreachable anyway — pepin.msgs.scan_arrays turns everything past range_max into NaN and config/lidar.json's max_range_m is that same 12.0 m, so a doorway carved nothing and stayed unknown. It stays off because a mirror, a black chair leg and anything nearer than the 0.05 m minimum all say the identical nothing, and carving them out to 12 m would rub out the wall behind them
  - *On when:* when a beam carries something that separates an open bearing from a mirror or a black surface — return quality, or the same emptiness confirmed from several viewpoints; nothing on this robot does today
  - *Off when:* leave it off: an open door stays unknown, which a planner may be told to cross (allow_unknown) rather than being told a lie
- **`no_depth_free`** — bool, default on
  - *What:* a camera pixel with NO depth carves free space along its own ray, from 0.20 m out to the source's own reach less one truncation, at no_depth_weight of what a measurement at that range weighs; off, a NaN pixel touches nothing at all, which is what this node did until 2026-09-22
  - *Default:* on — THE PHANTOMS THAT NEVER DECAYED. Only a ray that MEASURED something moved any voxel (pepin.tsdf.Tsdf.integrate), and the stereo depth is cut at the rig's own reach (2.46 m, pepin.stereo_depth), so anything standing in front of something farther than that had NaN at every one of its pixels and could never be carved. Measured on the live volume (scratch/one_localiser/black_voxels.py, 2026-09-22): an airborne cluster of 133 voxels at camera height, 94 % of its pixels NaN, 0 % ever seen free, not one voxel rewritten in 60 s; the operator's face 175 voxels, all 175 at the same millimetre a minute later; the same person at 2 m with a wall behind him cleared in about 3 s. AND THE A/B, 60 s of live frames replayed into two volumes off the robot (scratch/one_localiser/volume_ab.py: 374 frames, 297 revolutions, the cart parked): a saturated obstacle painted 0.4 m ahead is 90 % gone in 18 frames, 2.8 s, with the carve and NEVER without it (94 of its 330 voxels still a surface after the whole minute), and max_weight 20 alone changes nothing about that — which is the mechanism, nothing ever touches those voxels. The cost on the same minute: the camera band keeps 81 % of its occupied cells (99 lost, 41 new), and 3 of the 99 lost are cells the lidar's own returns mark occupied. It costs this node 2.6 ms a frame on the live grid and the live 800x600 image, 10.1 -> 12.7 ms, writing 37628 voxels instead of 8705 (scratch/one_localiser/carve_cost.py) — the laptop's, where every camera feature lives
  - *On when:* on: it is the only thing that carves a phantom whose background lies beyond the rig's reach, and the lidar's own layer is protected from it by lidar_layer
  - *Off when:* off to reproduce the pre-2026-09-22 volume exactly, or if a textureless near wall (a matcher refusal, not an empty ray) is ever seen to be eaten out of the surface — the marks audit counts what the camera holds that the lidar does not. The published depth cannot say WHY a pixel is NaN (beyond the reach, the edge filter's flying pixels, a rectification margin, a refused match all read the same), so the weight is the whole of the defence and no_depth_weight is where to turn it down
- **`colour_fallback`** — bool, default on
  - *What:* a surface point whose nearer voxel was never painted by a camera takes the OTHER neighbour's colour on /fusion/surface; off, it keeps the black that means 'no camera ever wrote here', which is what the cloud showed until 2026-09-22
  - *Default:* on — the lidar writes field and weight but no colour (a beam has no colour to give), and the readout takes the colour of the neighbour nearer the surface — for a beam's own return always the uncoloured one: 1532 pure-black points on the live volume, 100 % of them in the lidar's own height band and every one a real wall (scratch/one_localiser/depth_nan_why.py). AND THE FALLBACK IS A SMALL FIX, measured: on the taped minute of 2026-09-22 painted with every revolution (21777 lidar voxels, 5682 surface points) it recovers 16 of 987 black points, 1.6 % (scratch/one_localiser/volume_ab.py). The other 971 are crossings NEITHER of whose voxels a camera ever painted — the camera's own surface sits in other voxels than the beams' — so their black is the truth about them, and the only way to colour them would be to invent a colour no camera saw. That is why the scan path still writes none
  - *On when:* on: it costs nothing at the readout, and every colour it hands out is one the camera really wrote into that voxel
  - *Off when:* off to see exactly which points only the lidar holds — the black IS that measurement, and it is how the 1532 were found
- **`resume_volume`** — bool, default on, not live
  - *What:* a volume snapshot at world_path is loaded at start, so a room the cart has painted before comes back as it was left; off, the volume starts empty and grows from the sensors. world_path belongs to the graph DATABASE whose frame the voxels were painted in (rtabmap.db -> rtabmap.world.npz), so a fresh database means a fresh volume (not live: set at the next start)
  - *Default:* on — resuming its own snapshot is what makes yesterday's painting yesterday's surface instead of a picture thrown away every morning. Measured, on the four tapes of 2026-09-13 painted through scratch/volume_drive_regression.py: a volume seeded from flat3_straight and driven through a whole tape keeps 79.8 % of the walls it LOOKED at (the two thirds of the flat a single errand never enters are not counted), carves 447-791 of them free per tape, and chaining all four drives through one volume instead of re-seeding each time costs 2.8 points of that share (77.0 %) — the erosion does not run away, and ten passes of one tape cost 3.9 more points and nothing after that. The one hard rule around it is a guard: a snapshot is resumed only onto the grid config/fusion.json describes, so a changed grid starts empty instead of resuming into the wrong place
  - *On when:* on in the room the snapshot was taken in, beside the database it was painted in — the default
  - *Off when:* off for a new room, beside a fresh database, or to measure how fast the volume fills from nothing (ros/laptop.sh --fresh passes it off)
- **`view_gate`** — bool, default on
  - *What:* a revolution taken from a place the volume has already integrated is not integrated again (pepin.worldmap.ViewGate: the pose must have moved a whole voxel at the scan's own farthest return before it counts as a new view); off, every revolution is painted, which is what this node did until 2026-09-18
  - *Default:* on — A VIEW IS EVIDENCE ONCE, and it was being counted ten times a second: a parked cart sends the same revolution ten times a second and every one of them used to weigh as an independent observation, so a standing cart's own paint outgrew everything else in the volume within a second. Measured on the robot while the board's tracker still matched the volume it was painting: a cart parked with its wheels blocked walked 7 degrees and 5-7 cm in 35 minutes at fit 0.97-0.99, every step under a tenth of a degree. That closed loop is gone — nothing localises against this volume now — and the gate stays because what it measures is the volume's own honesty: a weight that counts one view a thousand times calls a single glance a wall the room agrees on. Offline (scratch/volume_closed_loop.py, 2000 revolutions of a standing cart from tape 20260913_190422) the old law drifts 0.268 deg/min at fit 1.00 and this gate alone holds 1993 of the 2000 revolutions and drifts 0.018 deg/min, with 100 % of the room's walls kept against 94.2 %. The threshold is not one: a return at range r moves in the map by the translation plus r times the turn, so 'a new view' is 'no return of this scan stays in the cell it was in', which is the grid's voxel and the scan's own reach and nothing chosen
  - *On when:* always: it is what keeps a weight a count of observations of the room rather than a count of seconds parked
  - *Off when:* to reproduce the drift for a comparison, or where the volume must integrate a long stare on purpose (a mapping run of one corner with the cart on a tripod)
- **`follow_correction`** — bool, default on
  - *What:* the graph's optimisation moves the voxels, not only the pose: when the accumulated move of RTAB-Map's own node poses (/rtabmap/mapGraph, read at the newest shared node) differs from the one the volume is painted under by more than follow_correction_min_m / _min_deg, the whole content is carried rigidly by that difference before the next observation goes in. Every mode — the graph is the one source of truth about the room under World R, and the node poses are the only signal that says the ROOM moved rather than the cart having been found
  - *Default:* on — the correction never reached the voxels (2026-09-13): RTAB-Map closed a loop, the cloud moved with the graph, and the painted room stayed where the pose used to be, so no loop drive could close in the map itself. The move is not free: on the live 280x250x34 grid as it stood on 2026-09-14 (2.4 M voxels, 297 k of them painted, scratch/volume_shift_cost.py) one move of 10 cm / 3 deg costs 108 ms of the worker thread under the default law and leaves 91 % of the occupied cells, each of the 99th percentile 9.7 cm from where the correction points — a resampled field is a weighted average and a surface averaged with the free space in front of it thins. The sensors repaint what thins within a second of driving; a map left behind the graph never comes back. Which law pays best is follow_correction_law's question, not this one's. What is NOT measured is a bend of the node poses on this database: beside a loaded one nothing is written, so nothing optimises and nothing should ever move — which is itself the live check
  - *On when:* always: any drive where the memory rule lets RTAB-Map learn is a drive where a closure can land, and a map left behind the graph never comes back
  - *Off when:* to see the old behaviour under the same graph — the graph and the pose move, the voxels stay — or if a closure is ever seen to smear the map instead of moving it
- **`follow_correction_law`** — choice: blend, nearest, default nearest
  - *What:* how the move resamples the volume: blend is the fusion's own weighted average of the four source columns, nearest takes the one column the cell came from (one of: blend, nearest)
  - *Default:* nearest — MEASURED 2026-09-18, and it reversed the default. blend was chosen because a weighted average THINS a wall and the sensors repaint a thin wall, while a quantisation bias is never repainted. LidarLaw.beam_footprint took that premise away: a far crossing now weighs only the share of its own disc that the voxel covers, so the free space in front of a wall is weak while the return is full weight, and the average is pulled INTO the wall. On the synthetic box, one move of 10 cm / 3 deg: blend 430 occupied cells -> 516 (+20 %, a wall two cells thick) with its worst cell 5.85 cm from where the correction points — past the voxel — against nearest 430 -> 431 with its worst at exactly 5.00 cm, half a voxel, which is its whole documented cost (tests/unit/test_follow_correction.py). A widened wall is a bias in the layer the cart drives by, which is the failure the old default existed to avoid. Beside that, the cost: on the live snapshot of 2026-09-14 (280x250x34, 297 k painted voxels) one move of 10 cm / 3 deg costs blend 108 ms and leaves the 99th occupied cell 9.7 cm from where the correction points (91 % of the cells survive); nearest costs 9 ms, keeps every cell and puts the 99th within half a voxel, 2.5 cm — sharp, cheap, and systematically quantised, which is the bias class the grid snapping of 2026-09-13 was written to kill. blend is the default because a bias is not repainted by the sensors and a thinned wall is; the 9.7 cm says that argument is not settled, and the morning's loop drive is what settles it (scratch/volume_shift_cost.py)
  - *On when:* nearest: every cell kept, each within half a voxel, and 9 ms instead of 108
  - *Off when:* blend to reproduce the old default, or on a volume painted with beam_footprint off, where its thinning argument still holds — the A/B is a live flag, no restart

#### `depth_stream`

- **`edge_filter`** — bool, default on
  - *What:* flying pixels at object edges are dropped from the published depth and the scan; the law's beam pairs skip them regardless
  - *Default:* on — the band probe found 21 % of the band's pixels more than 15 cm from any lidar return, with no bearing trend to blame the focal length or the mount yaw on — flying pixels; the 8 % step that drops them costs 1.2-1.6 ms a frame and 2 % of the pixels (scratch/pipeline_vs_truth.txt, scratch/band_frame_probe.py). The cause is measured, the benefit is not: the one A/B on the robot, one turn each way, showed no difference
  - *On when:* whenever the scan feeds a costmap: the pixels it drops are the ones that become an obstacle with nothing behind them
  - *Off when:* to see the raw band's tail in Foxglove, or when a thin real object (a chair leg at range) is missing from the scan and the 8 % step is the suspect
- **`lidar_anchor`** — bool, default on
  - *What:* the lidar's returns pair with the network's depth and fit the law; off, the last law is held (the failure mode of a lidar that stops) — with no law yet nothing is published until it is back on
  - *Default:* on — the beams are the only metric ruler on board. Without them a floor-only fit reads the lidar's own row 1.98-2.46x too far and the raw network 1.62-1.96x (scratch/pipeline_vs_truth.txt, scratch/lidar_height_check.txt), and the pairing costs 0.0-0.1 ms a frame. The one on/off turn on the robot is split: a held law was better above the band (9.6-19.3 cm against 16-46 cm) and worse at it (12.0 cm against 7-9 cm), and the band is the row the costmap drives on
  - *On when:* whenever the lidar spins — it is what makes the network's depth metric. It can only judge past the range at which its own plane enters the picture (pepin.depth.plane_in_view_from: 0.71 m with the head 23.7 deg down, the lens 0.82 m above the plane, a 640x360 frame). Parked closer than that — the working case at a desk — not one beam lands in the image, the report says so (lidar plane out of the picture N frames) instead of asking whether /scan is alive, and the law is held
  - *Off when:* to rehearse a lidar that dies mid-run (the law freezes, nothing else changes), or to compare the slices above the band, where the frozen law measured better
- **`affine_law`** — bool, default on
  - *What:* the network's depth through 1 / z = a / D + b, fitted on the pooled pairs; off, the raw network's depth goes out unwithheld
  - *Default:* on — the raw network is 1.6-2.0x too far — the lidar's own row reads 1.958 raw against 1.033 through the law, and the band against the beams 35.5/112.0 cm against 3.8/20.6 cm, 14 % -> 57 % of it within 5 cm (scratch/lidar_height_check.txt). The fit costs 0.4 ms
  - *On when:* always, to drive
  - *Off when:* as an A/B measure of the correction, at rest — never a way to drive: every published metre is then 1.6-2.0x long
- **`floor_anchor`** — bool, default on
  - *What:* pixels within centimetres of the floor plane snap to it in the published image (the scan is built before it); the plane leans with the cart, from the IMU's up vector
  - *Default:* on — measured on the robot, one turn each way — the floor's sd 5.8 -> 3.3 cm and 43 % -> 71 % of it within 3 cm, for 0.4-0.6 ms a frame. The scan is built before this stage on purpose: at 5.8 cm of patchy floor noise a probe called 63 of 161 bearings lethal, which is where the scan's 0.15 m floor cut comes from
  - *On when:* whenever the floor should read as floor in the published depth and in /fusion/surface
  - *Off when:* to measure the raw floor's noise again (the number the 0.15 m scan cut was chosen from), or where the floor is not a plane — a ramp, a threshold — and snapping would invent one
- **`depth_backend`** — choice: remote, local, auto, default local
  - *What:* where the network runs: local (the CPU model in this container), remote (the laptop's GPU service, ros/depth_host.sh), auto (the service while it answers, the CPU model while it does not) (one of: remote, local, auto) (PEPIN_DEPTH_BACKEND overrides the default at start)
  - *Default:* local — default by design, unmeasured as a choice: local is the value that needs nothing else running, and ros/laptop.sh exports PEPIN_DEPTH_BACKEND=auto whenever the laptop's Metal backend answers, so the field default is auto. What the choice is worth is measured: Depth Anything V2 Small is 20.6 ms a frame on MPS against 172-204 ms on the container's CPU, 26 ms end to end from the container through the JPEG service — 6.6x (scratch/depth_backend_bench.py), and on the robot the remote backend published 9.5 fps with 0 frames falling back to local
  - *On when:* remote while the GPU service is up and the frame rate matters; auto for a run that must survive the service dying mid-drive
  - *Off when:* local when the laptop's service is not there or is being restarted, or to measure the container's own worst case (5.8 fps)
- **`stereo_matcher`** — choice: sgbm, raft, default raft
  - *What:* which engine turns the two eyes into a disparity: sgbm (OpenCV's semi-global block matcher, in this container) or raft (RAFT-Stereo on the laptop's GPU through the same host the mono network uses, ros/depth_host.sh stereo). Both are built at start and this picks which one answers the next pair, so an A/B needs no restart; a pair the host cannot answer falls to sgbm and the report line counts it. Under depth_source: network it does nothing (one of: sgbm, raft) (PEPIN_STEREO_MATCHER overrides the default at start)
  - *Default:* raft — raft since 2026-09-23, when the drive settled it: six legs of that day (three with the lidar, three camera-only) all ran on raft at 101-111 ms a pair, 5.8 published frames/s, 71 % of the pixels valid, and reached every mark; a restart that fell back to the old sgbm default was a silent change of the sensor under the next test. What raft buys was measured before that through THIS path on 2026-09-22, on the 8 rectified pairs of scratch/stereo_net/frames/pairs.npz (host_smoke.py, sgbm_vs_raft.py, phantom_where.py): the airborne phantom PIXELS the lamp's reflection on the parquet puts 0.5-1.5 m up and under 2.5 m ahead fall only 6651 -> 5265, and raft is the worse of the two on three of the eight pairs — but what a costmap is told falls 450 separate specks -> 94 blobs (192 -> 37 of 50 px or more), halves in the robot's own path (4486 -> 2566), and the floor comes back whole: 4.7x as many points within 5 cm of the fitted plane. What it costs is measured too: 89.2 ms a pair end to end through the host against SGBM's 16.8 here, so 11 fps against the camera's 10 and the scan_hz cap's 5 — the whole budget, with the pose and the pipeline still to pay for, and the routers not yet in the path. The drive settles it
  - *On when:* on the parquet, where SGBM's reflection phantoms are the thing marking the costmap, and with ros/depth_host.sh stereo up — watch the report's ms a pair and the published frames/s before trusting it
  - *Off when:* whenever the frame rate matters more than the phantoms, when the host is not running (it falls back by itself, but paying a round trip per pair to be refused is not free), and as the A/B half of any claim about either engine
- **`law_watch`** — bool, default off
  - *What:* the affine law is fitted on the lidar's pairs and printed, and the depth is published exactly as the source measured it; no frame waits for a law
  - *Default:* off — off for the network, whose depth is 1.6-2.0x long until the law corrects it. ON under depth_source stereo (STEREO_DEFAULTS): a calibrated head is metric by construction — the checkerboard calibration of 2026-09-21 reads a printed board's span to +0.8 % on frames it never saw (scratch/stereo/board_metric_check.py) — and the law fitted on a cluttered room pulled b to its -0.200 bound, because the lidar's plane is 0.8 m under the lens and its far returns project onto whatever stands in front of them. Watching, the same fit is the head's health line: a 1.00 while the rig is as calibrated, anything else once it has been knocked. It costs the fit alone
  - *On when:* the source is metric (stereo) and the lidar is a witness, not a ruler
  - *Off when:* the depth needs the lidar's scale (the mono network), or as an A/B of what the law would do to a stereo depth
- **`imu_lean`** — bool, default on
  - *What:* the cart's lean (pepin.lean, from /imu/data_raw) is followed with the gyro as well as the accelerometer and carried into the scan's carry and the camera's place in the map; off, the floor plane leans with the accelerometer alone, as it always has, and nothing else is leaned
  - *Default:* on — on: the gyro's sign was verified by hand on 2026-09-13 (the cart tipped nose-down read pitch +10.5 deg, left-side-down read roll -9.4 deg, the fast path following at once with quality 0.9 while held; scratch/lean_tip_test.txt), and the gyro's zero offset is learned. Off, the floor anchor keeps the accelerometer-only lean, which by design ignores any tip shorter than 10 s
  - *On when:* after a hand tip through a known angle shows the reported lean following it the right way and returning to zero
  - *Off when:* the moment the lean in the report line disagrees with the cart's visible attitude
- **`camera_tf_latest`** — bool, default on
  - *What:* take the newest base_link <- camera_optical edge TF holds (at most 1 s old, and only while the head has not moved over the half second before it) when the frame's own stamp is not covered yet, instead of waiting CARRY_WAIT_S for it; off: the old wait at the exact stamp
  - *Default:* on — on since 2026-09-15 06:00: the neck's edge crosses the bridge late (bursts of +0.75 s) and the head stands still while the cart drives; waiting for the exact stamp cost 0.2 s on every frame ('Extrapolation ... into the future' x104 a window), the stream fell to 3.5 frames/s and rgbd_odometry starved (0 poses/s)
  - *On when:* always: a turning head is recognised (its edge moved within the last half second) and waited for at the exact stamp instead (2026-09-30)
  - *Off when:* to measure what the shortcut is worth: every uncovered frame then waits
- **`fan_floor_gate`** — choice: band, contact, off, default band
  - *What:* what keeps the floor out of /depth_scan: band (the default) raises the band's lower edge with the floor's own noise, 3 sigma of it (pepin.contact.fan_min_z), contact drops every mark nearer than that bearing's floor-contact range (pepin.contact.gate_by_contact, a range no depth law enters), off is the flat 0.15 m edge the fan always had. The report line counts the bearings gated (one of: band, contact, off)
  - *Default:* band — a floor pixel stands camera_height * (relative depth error) above the floor at every range, so 12.5 % short is exactly the 0.15 m edge on this mount while the per-frame law's own median residual is 10.2 % — the floor marks itself as an obstacle, and it is why the fan read 0.58 of the lidar at the working pitch on 2026-09-15 (scratch/fan_floor_leak.py). Measured on the three pitch tapes of 2026-09-12 (scratch/fan_gate_offline.py, the per-frame law, 9 frames each): band moves k = fan / lidar 0.499 -> 0.595 at 25.8 deg without removing a single bearing (the mark simply lands on the real obstacle instead of the floor in front of it), and does nothing at 11.1 (0.851 -> 0.853) or 40.9 (0.232). contact is the aggressive one and overshoots: it removes 455 of 1399 marks at 11.1 deg and takes k past 1 to 1.215, and at 25.8 and 40.9 it removes every mark there is
  - *On when:* band as shipped; contact only against a scene where the floor plane is trusted and the fan is known to be floor — and never without reading the bearings gated
  - *Off when:* off to reproduce a costmap from before this gate
- **`scan_honours_pan`** — bool, default on
  - *What:* fold /depth_scan onto the floor through the neck's pan: the fan's bearings turn with the head and its angular window turns with them, so angle_min comes out at pan - 40 deg instead of -40. The pan is the yaw of the same base_link <- camera_optical edge the volume path reads (camera_tf_latest); with none at the frame's stamp the last edge TF held stands in, and only while TF never had one the config mount's straight-ahead yaw (the report line counts both). Off: the fan is projected as if the head looked along the cart's x, whatever the encoders say
  - *Default:* on — the fan carried no pan at all until 2026-09-15, and the report line said so ('head panned N frames (projected as if not)'). At rest that is not nothing: the pan reference measured that day (config/neck.json pan_note, scripts/extrinsics.py's pan_from_bearings, six windows in four scenes) puts the resting head +0.79 deg left of the cart's x, which is 4 cm of bearing error at 3 m — under PAN_NOTICE_RAD, so the old fan did not even count it. A head panned on purpose puts the whole fan in the wrong place: 20 deg of neck is 20 deg of costmap, one metre sideways at 3 m
  - *On when:* always once the neck's edge is in TF — a scan whose bearings are the cart's is what Nav2's obstacle layer assumes it is being handed
  - *Off when:* to reproduce a costmap from before 2026-09-15, or to read a fan against a measurement taken while the projection ignored the pan (the yaw-offset probes of config/neck.json's pan_note were)
- **`depth_reach`** — bool, default on
  - *What:* the PUBLISHED depth image is NaN past depth_reach_m: the camera answers for its own data and says nothing where it does not vouch for the range. /depth_scan is unaffected (it is capped at the same range already) and so is every law — the gate is applied to the image on its way out, after the pipeline
  - *Default:* on — a NaN depth pixel makes no point in any consumer: rtabmap drops it from the cloud before the grid (pcl::isFinite, rtabmap/core/util3d.cpp:644), Nav2's obstacle layer neither marks nor raytraces it (verified against obstacle_layer.cpp 1.3.12 on 2026-09-11), and pepin.tsdf integrates only finite depths. Without the gate the camera's far half is a fiction that outvotes the lidar: 44 % of the camera's costmap marks within 2.5 m were BEHIND the wall the lidar sees (run 0224, camera layer alone, 2026-09-11), and the wall-truth eval put the network 1.24-1.27 of the truth in the middle and top thirds of the picture against 0.998 at the beams (errand 0313, scratch/wall_truth_eval.py). It is what lets ONE Grid/RangeMax serve both sensors in vslam.launch.py: the lidar's 8 m, with the camera's reach carried in the camera's data
  - *On when:* always while any grid, costmap or volume is built from BOTH this depth and the lidar — which is every mode since 2026-09-19
  - *Off when:* to measure the network past its reach (a range-law session that wants the far bins), and to reproduce a volume or a costmap from before this gate

#### `goal_server`

- **`places_from_the_file`** — bool, default off
  - *What:* before the graph's book of places has been heard, a name is answered from the yaml beside the map (coordinates of the frozen-grid era); off, a name is refused until the book arrives, with that reason
  - *Default:* off — that file holds coordinates of a frame that no longer exists, and the board keeps its own stale copy (the sync excludes it). Twice a name was answered from it and sent the cart at a point outside the map: `home` -> (-9.39, +2.53) on 2026-09-19, `printer` -> (-11.38, +0.77) on 2026-09-21, 2.1 s after RTAB-Map's first graph of a cold start of both halves — the planner said 'outside bounds', the behaviour tree ran 33 recoveries in 28 s and backed the cart into a sofa. A refusal costs a second try a minute later
  - *On when:* only on a robot driven without the laptop's graph at all, on the old frozen map
  - *Off when:* always under World R: a place rides a graph node, and only the graph can say where that node is now
- **`jump_clear`** — bool, default off
  - *What:* map -> odom is read from TF five times a second and, when it STEPS further than 0.10 m, Nav2's local costmap is emptied ("/local_costmap/clear_entirely_local_costmap", asynchronously, at most once per 1 s): the marks in that grid were laid where the cart used to be. The step in that edge is the correction alone — the cart's own motion lives in odom -> base_link — whoever published it. Off, nothing reads the edge and no listener is started for it
  - *Default:* off — OFF, AND SINCE 2026-09-22 ITS PREMISE IS GONE: the local costmap is built in the ODOM frame (ros/params/nav2_params.yaml), and a step in map -> odom does not move a grid that is not drawn in map — the marks stay exactly where the cart saw them. A clear would now throw away good evidence for nothing. The flag stays because the frame is one word away from being map again, and there it is the right behaviour: the lidar tracker did exactly this while it owned map -> odom (pepin.watch.JumpClear, written for the camera-only return of 2026-09-16, where the pose lagged 1.4 m behind the cart and Nav2 spent 29 recoveries fighting marks placed at the poses before each correction). Even then the other side of the trade was unmeasured: RTAB-Map corrects in centimetres at a loop closure, which the costmap absorbs, and the raytracing of the live scans re-clears a stranded mark within seconds anyway. The threshold and the gap are the tracker's measured ones, inherited unchanged
  - *On when:* only together with a local costmap put back into the map frame, and then when a drive is seen fighting a second copy of the room after a correction: recoveries at obstacles that are not there, the grid holding marks offset from the live scans by the size of the last jump
  - *Off when:* the shipped state, and the only sane one while that costmap is in odom: a clear there costs a controller its picture of the room and buys nothing
- **`pose_topic`** — bool, default on
  - *What:* the pose this node reads out of TF is republished as /pose (geometry_msgs/PoseStamped in map, 5 Hz, stamped with the transform's own stamp), so the other nodes on this board can have the pose without a TF listener of their own. Inert on a split stack, where the reader is on the other machine and reads the edge itself. Off, nothing is published and this node's listener goes back to being started on the first ask
  - *Default:* on — a TF listener is a subscription to the whole /tf stream — RTAB-Map's map -> odom at 20 Hz plus the board's odom -> base_link at 50 Hz plus the statics — deserialised in Python whatever the reader wanted out of it. Two of them ran on a 4-core A53 to read one pose now and then: this node's, for `where`, the preflight and the jump watch (~22 % of a core), and pepin_bringup.run_recorder's 5 Hz read for the tape's `loc` rows (~34 %), on a board measured at 252 % with the real-time loops starving (2026-09-22). This node owns navigation and the jump watch, so its listener is the one that stays and the tape reads the topic instead (run_recorder's loc_from)
  - *On when:* always where this node and the tape recorder share a machine: it is what lets every other node there read the pose for the price of a 5 Hz PoseStamped
  - *Off when:* to put the two independent listeners back for a comparison — turn this off here and run_recorder's loc_from to tf, or the tape loses its pose rows
- **`controller`** — choice: mppi, rpp, rpp_shim, default rpp_shim
  - *What:* what follows the plan: mppi is Nav2's MPPI controller for every planner, held to the mark's heading by the yaw-checking goal checker; rpp is each planner's own Regulated Pure Pursuit from PLANNERS, ending on position alone as before 2026-09-23; rpp_shim is the reversing RPP inside Nav2's RotationShimController, which turns the cart to the mark's heading in place once it is inside the goal tolerance. Published latched on controller_selector and goal_checker_selector, so a change is read by the behaviour tree at its next tick (one of: mppi, rpp, rpp_shim)
  - *Default:* rpp_shim — the six legs of 2026-09-23 all stopped 12-60 deg short of the mark's heading: the reversing RPP cannot rotate in place, the tree ended the drive on position (xy_only_goal_checker), and the pivot that finishes the heading lives here in _pivot_to, which drives sent through goto_ros.py never reach. Each leg also ran 5-9 recoveries, which is what an RPP answers a refused arc with; MPPI samples another trajectory instead
  - *On when:* rpp_shim by default since the evening of 2026-09-23 (six legs: 21-33 s, 6-11 recoveries, 4-10 cm and 2-5 deg at the mark; MPPI on the same board crawled at a median 0.06 m/s); mppi where its sampling is wanted, rpp for the position-only drives of before
  - *Off when:* rpp for an A/B against the RPP drives of before, or if MPPI's cycle does not fit the board's control period (the controller server's 'Control loop missed its desired rate')
- **`start_needs_placement`** — bool, default on
  - *What:* a goal or a mark waits for the laptop's word on /localization/placement (pepin_bringup.rtabmap_frame, latched) that this start of RTAB-Map is PLACED — a node of the loaded map recognised, or an operator's seed — and nothing heard is refused like not placed. ros/tools/goto_ros.py asks this node for this same flag before its own preflight. Off, a fresh map -> base_link is enough, as before 2026-09-23
  - *Default:* on — on, measured 2026-09-23: after a restart RTAB-Map publishes map -> odom from the pose it SAVED at its last shutdown, and a fresh transform was taken for a localisation — 'at home' at the bookshelf with 0 of 198 updates recognised, then 76 cm off inside the table. The laptop's flag of the same name only changes what rtabmap_frame SAYS: it cannot lift a refusal of silence, from a node that is down, respawning or running code from before the word existed. This one is the board-side switch the refusal itself answers to
  - *On when:* always: a pose nobody has vouched for since RTAB-Map's start is not a pose to drive on
  - *Off when:* when the word cannot come and the cart is known to stand where RTAB-Map's pose says: pepin-vslam down or started before this build (ros/laptop.sh vslam restarts it on the checkout), or a dark room with no seed at hand
- **`cancel_every_goal`** — bool, default on
  - *What:* a cancel on the socket also asks both navigators' own cancel services (navigate_to_pose and navigate_through_poses, <action>/_action/cancel_goal) for EVERY goal — a zero goal id, whoever sent it — and answers what each said (``navigators``); off, it cancels only the goal this node sent, as before 2026-09-25
  - *Default:* on — ros/goto.sh drives through ros/tools/goto_ros.py, a goal this node never sent, so this cancel reached nothing, and goto.sh's own cancel started goto_ros.py on the board mid-drive: a new ROS process is a new zenoh session, and each one stalled all laptop -> board delivery for 2.6-3.1 s about 1.5 s after it started (34 of 39 cases, journal 2026-09-25). Asked from this long-lived node the same cancel costs a socket write; one 30 s deadline is shared by both navigators, as in goto_ros.py
  - *On when:* always: the operator's cancel means every goal on the board, whichever client sent it
  - *Off when:* to put the old answer back for a comparison — pepin.goal_link then finds no navigators in the answer and ros/goto.sh cancel falls back to goto_ros.py
- **`lidar_watch`** — bool, default on
  - *What:* the lidar's driver is restarted when its port is there and no scan has come for 5 s, and once per absence of the port so it respawns idle instead of spinning a core (pepin.lidar_watch); `where` says `lidar` either way
  - *Default:* on — the driver opens its port once: a lidar re-plugged or plugged in after the start stayed dead until a stack restart (journal 2026-09-28)
  - *On when:* always
  - *Off when:* while the lidar is deliberately held silent with its port present

#### `marks_audit`

- **`marks_audit`** — bool, default on
  - *What:* the audit runs; off, the node keeps its subscriptions and computes, publishes and reports nothing
  - *Default:* on — on, and the cost is bounded by the grid rather than estimated: the local costmap is 3 x 3 m at 5 cm published at 1 Hz (ros/params/nav2_params.yaml), so the very worst frame this can be handed is 3600 lethal cells against a 450-return revolution — 1.6 M distances of numpy once a second, on the laptop, while the board waits for none of it. It is on because the drive that needs the number is the drive nobody knew would go wrong: run 0431 took a tape and an hour of replay to say that 54-73 % of its near-ahead lethal cells had no lidar behind them
  - *On when:* always, and especially on any drive where the camera layer is marking
  - *Off when:* to take this laptop's last percent back for a profile of something else
- **`inscribed_counts`** — bool, default off
  - *What:* the inflation's 99 band (costmap 253, INSCRIBED_INFLATED_OBSTACLE) is judged as a mark too; off, only the 100s a sensor actually wrote
  - *Default:* off — off, which is the threshold the analysis this node reproduces used: 100 for the LOCAL grid's marks (scratch/one_localiser/tape_0431_phantoms.py, LETHAL = 100), with the 99 band reserved for the reduced GLOBAL grid, whose cells are classes and not costs. An inscribed cell (99, costmap 253) is the inflation layer's arithmetic around a mark (100, costmap 254) and not an observation, so asking which sensor painted it answers a question nobody asked while multiplying every count by the inflation's own area
  - *On when:* to see the band the planners' footprint check really refuses — the cells that made Hybrid-A* call a start pose blocked
  - *Off when:* whenever the question is which sensor SAW something
- **`phantom_cloud`** — bool, default on
  - *What:* the camera-only cells are published as red points on /marks_audit/phantoms; off, only the counts go out
  - *Default:* on — on, and the bandwidth is bounded by the grid: the cloud can never hold more than the 3600 cells of a 3 x 3 m grid at 5 cm, 32 bytes each in the PCL layout (pepin_bringup.msgs.cloud_from_points), so 115 kB/s at the impossible worst and about 3 kB/s for the hundred cells a real frame has — and none of it crosses the radio, since both this node and Foxglove's bridge are on the laptop. The counts alone do not say WHERE the phantoms are, and where is what sends the operator to the right furniture
  - *On when:* always, while looking at the 3D panel
  - *Off when:* on a link already saturated by the surface cloud

#### `neck_state`

- **`neck_tf`** — bool, default on
  - *What:* base_link -> camera_link is published live from the neck's encoders; the laptop's camera node must then run with ros/laptop.sh vslam --neck, or two nodes publish that edge
  - *Default:* on — the encoders are honest and their signs are checked by hand: the tilt reads 26 -> 103 degrees as the head goes down and the pan 0 -> -124 degrees to the left (config/neck.json's tilt_sign +1, pan_sign -1), 50 reads of a still head gave the same ticks every time, a read costs 9.7 ms, and the tick scale solved from the level frames is 1.067 true degrees per commanded degree, so 360/4096 stands (scratch/neck_tilt_scale.txt). At the reference pose the live transform equals the static one, so turning it on moves nothing until the head does
  - *On when:* whenever the head moves at all: with it off a turned head is a camera the map places where it is not
  - *Off when:* when the laptop broadcasts the static edge instead (camera_stream's static_camera_tf), or when the neck bus is suspect and a frozen edge is better than a wrong one

#### `places`

- **`publish_places`** — bool, default on
  - *What:* the resolved places are published on /places whenever the graph moves; off, the book is still kept and marked but nothing is published and every consumer falls back to the coordinates beside the map
  - *Default:* on — default by design, unmeasured: it is the one output of this node. The switch exists to prove which book a drive resolved a name from — with it off, ros/go.sh printer must print the fallback warning and reach the same furniture, which is the A/B between a place that rides its node and a coordinate that does not
  - *On when:* always: a coordinate written before a loop closure names a spot beside the furniture rather than in front of it
  - *Off when:* for that A/B, and if a resolved place is ever seen further from the furniture than the file's own coordinate
- **`label_nodes`** — bool, default on
  - *What:* a mark also sets RTAB-Map's own label on the node (set_label), which is what makes the place a thing in its tools and in its set_goal; off, only our own book records the node id and the offset
  - *Default:* on — on, because the label costs one service call and is the only name RTAB-Map itself understands. It is NOT the storage and this node never reads it back as one: beside a loaded database nothing is written (Mem/IncrementalMemory false) and a label on a node in the working memory only flips a dirty bit (Signature.h:76), so a label set while localising never reaches the file. list_labels is still called with it off — it is the only way to learn which node RTAB-Map considers the current one
  - *On when:* always beside a database that may be written; it costs nothing where it cannot
  - *Off when:* on a database that must not be touched at all, even by a dirty bit

#### `rtabmap_frame`

- **`registration_follows_snapshots`** — bool, default on
  - *What:* RTAB-Map's Reg/Strategy follows what the snapshots carry (/sensor_pack/state): a scan and a picture mean visual then ICP (2), a scan alone means ICP (1), no scan means visual (0), switched live through the node's own parameter path on a change that has held for the hold the state carries. Off, the strategy stays whatever the launch table set and this node only reports what it would have asked for
  - *Default:* on — on, because under ICP a camera-only cart cannot localise AT ALL, and that is measured rather than reasoned: in a minute of camera-only snapshots on 2026-09-18 RTAB-Map logged 28 'Missing visual features or missing raw data to compute them' and 56 'Requested laser scan data, but the sensor data doesn't have laser scan', and not one update named a node. The strategy is one object for the process (the pipeline is deleted and re-created when the parsed value differs from the one in hand, rtabmap/core/Memory.cpp:721-731), so one table cannot serve a node with a scan and a node without one — and which a node has is now data, not config. What is NOT measured yet is that strategy 0 makes a camera-only link on THIS database: that is the live check
  - *On when:* always beside a known map, and above all in a camera-only test — it is the whole difference between a camera that recognises a place and one that can act on it
  - *Off when:* to reproduce the stage-1 behaviour (ICP throughout) under the same snapshots, or if a live switch is ever seen to cost RTAB-Map its working memory
- **`atomic_parameter_sets`** — bool, default on
  - *What:* a parameter set this node hands RTAB-Map (a strategy with its visual features, a visual set alone, a memory mode's pair) goes as ONE /rtabmap/rtabmap/set_parameters_atomically request, which rtabmap_slam applies as one /parameter_events notification and one parseParameters. Off, it goes as one set_parameters request, which rclcpp applies one parameter at a time — each its own event and its own parseParameters — with RGBD/LoopClosureReextractFeatures ordered first when it turns off and last when it turns on
  - *Default:* on — on, measured 2026-09-24 in a throwaway container of pepin-laptop:xfeat (scratch/xfeat_critic/atomic_set.sh: the node on an empty database, its own ROS domain, no network). rtabmap_slam applies every parameter event it hears on its own node (CoreWrapper.cpp:907-970). Five parameters in one SetParameters request (Reg/Strategy 0 and the xfeat set) arrived as 5 events and 5 parseParameters over 61 ms, Reg/Strategy first and re-extraction fourth, with 49 ms after Vis/FeatureType 15 alone while its detector was built; the same five in one SetParametersAtomically request arrived as 1 event and 1 parseParameters. One by one, leaving xfeat for ICP passes through ICP with re-extraction still on, and every intermediate state is a registration nobody chose
  - *On when:* always: every parameter this node sets is declared by rtabmap_slam (CoreWrapper.cpp:362-364), so an atomic set is refused only as a whole
  - *Off when:* to reproduce the behaviour before 2026-09-24, or if RTAB-Map's node is ever seen to refuse an atomic set (one unknown name fails the WHOLE request, where set_parameters fails that parameter alone)
- **`visual_features`** — choice: orb, xfeat, default xfeat
  - *What:* which features RTAB-Map's VISUAL registration (Reg/Strategy 0, the camera-only strategy, and the visual half of 2) matches when it checks a node the words recognised. xfeat: XFeat keypoints matched by LighterGlue, re-extracted from both nodes' stored pictures at loop-closure time (Vis/FeatureType 15, Vis/CorNNType 6, RGBD/LoopClosureReextractFeatures true); the database is only read. orb: the database's own GFTT/ORB words, the launch table's values. Sent with the strategy and changed live; under ICP alone, and unless RTAB-Map is certainly localising (told so and answered, no switch to mapping waiting), the set is always orb, and a switch to mapping waits until orb's set is in force. xfeat needs the pepin-laptop:xfeat image (/opt/xfeat/rtabmap_xfeat.py); in another image this node sends orb and the report line says why (one of: orb, xfeat)
  - *Default:* xfeat — xfeat, measured 2026-09-24 in RTAB-Map's OWN registration: RegistrationVis of pepin-laptop:xfeat on evening frames of runs 0455-0465, each against the daylight database's node nearest the lidar-held truth (scratch/xfeat/probe/probe_report.py; Vis/Iterations 300, 2 px, 20 inliers; judged while turning slower than 0.35 rad/s, right within 0.30 m and 10 deg). Of 219 frames ORB recognised 7 (3 %), xfeat 197 (90 %): 139 right and 18 wrong — 11.5 % of the 157 judged — median error 0.12 m / 3.1 deg, p90 0.30 m / 9.2 deg (scratch/xfeat_critic/rtabmap_own_figures.py). On the camera-only run 0466, against the nodes RTAB-Map itself proposed, ORB recognised 0 of 136 frames and xfeat 133. The whole node replaying 0455-0465 with the stock table (scratch/xfeat/rtabmap_replay.py, replay_summary.py) accepted 27 of 425 updates — 8 loop closures, 19 proximity links — and 19 of the 21 judged were right; the 2 wrong are loop closures on node 641 (the bookshelf), all five of whose accepts sit at 0.24-0.31 m / 7-11 deg. In the rtabmap role an accepted wrong registration moves map -> odom (RGBD/OptimizeMaxError 0 there: the graph's error rejects nothing), so the 11.5 % is the number to watch on a drive. Secondary, the Python emulation over the 3 nearest nodes (scratch/xfeat/xfeat_bench.py): ORB 10 (5 %), xfeat 167 (76 %), 6 of 133 judged wrong, four of them node 492/493 where the daylight picture's rocking chair had moved by the evening (scratch/xfeat/node_consistency.py); on 0466 its 93 recognitions with an EKF sample gave a map -> odom within 4.0 cm / 0.45 deg of a running median (scratch/xfeat/bench_analyse.py)
  - *On when:* xfeat whenever the camera has to localise alone on a map built in other light — the evening, lamps on, a daylight database
  - *Off when:* orb to reproduce RTAB-Map's stock registration (0 of 630 camera-only updates accepted on 2026-09-23 evening), in an image without the adapters, when a drive shows the wrong recognitions (map -> odom jumping by 0.3 m or 10 deg on one word), or if the laptop's CPU cannot spare a registration: 0.67-0.70 s median inside RTAB-Map on the Docker VM once torch is imported with RTLD_DEEPBIND, 1.70 s before (scratch/xfeat/data/probe/out/probe_xfeat_2px_300_6_*.csv), per loop-closure or proximity candidate; in the stock-table replay, run before that fix, an update that registered took 2.5 s median, p90 5.1 s. The benchmark's 75 + 75 + 167 ms is the Python adapters alone, not what RTAB-Map pays
- **`visual_confirm`** — choice: rtabmap, aggressive, single, default aggressive
  - *What:* how RTAB-Map CONFIRMS a localisation while the camera registers alone: RTAB-Map 0.22 delays a first good localisation into its odometry cache and accepts it only with a second one inside RGBD/MaxOdomCacheSize updates, and that second try has to reach Rtabmap/LoopThr. rtabmap: its stock 0.11 and 10. aggressive: Rtabmap/LoopThr 0.05, the threshold the first try already used, so the second comes on the next update; the confirmation stays. single: RGBD/MaxOdomCacheSize 0, the first good localisation is accepted. Sent with the visual strategy while the database localises and changed live; under ICP and while it maps, always rtabmap (one of: rtabmap, aggressive, single)
  - *Default:* aggressive — measured 2026-09-24 parked at the base under the evening lamps, camera only, xfeat (scratch/link_autopsy/confirm_ab.sh, 240 s each): rtabmap accepted 0 of 244 updates — XFeat registered with 83-118 inliers once every 11 updates, the night's ORB hypotheses (0.05-0.07) never reach 0.11 for the confirming try and the cache rolls the first one out (scratch/link_autopsy/localisation_cadence.py); aggressive accepted 41 of 42 and single 44 of 44, all within 7 cm of the seed and within 1.2 deg of the yaw at which the lidar's scan fits the map (scratch/link_autopsy/lidar_yaw_truth.py). aggressive keeps RTAB-Map's own two-localisation rule
  - *On when:* aggressive whenever the camera has to localise alone in other light than the database's; single if a drive shows aggressive still waiting between localisations
  - *Off when:* rtabmap to reproduce RTAB-Map's stock confirmation, or if the extra registrations (one per update with a hypothesis over 0.05) cost the laptop more than it can spare
- **`visual_proximity`** — bool, default on
  - *What:* whether a localised camera also registers every update against the database nodes near its pose (RGBD/ProximityBySpace), each an XFeat re-extraction and a LighterGlue match; off, only the words' own hypothesis is registered, one at most per update. Sent with the visual strategy while the database localises and changed live; under ICP and while it maps, always on (the lidar's proximity links are cheap and most of the graph's)
  - *Default:* on — on, measured 2026-09-24. RTAB-Map in pepin-laptop:xfeat on the evening drives with the lidar-held truth (runs 0455-0465, 425 updates, LoopThr 0.05, scratch/xfeat/REPORT_replay.txt): on accepted 213 (50 %), 51 wrong at 0.25 m / 5 deg, median 0.12 m / 3.0 deg, p90 0.30 m / 8.6 deg, 775 / 1877 ms an update; off accepted 160 (38 %), the same 51 wrong, median 0.10 m / 3.9 deg, p90 0.31 m / 11.0 deg, 551 / 826 ms, and a cluster of parked accepts 49-57 deg off in 0458. Parked at the base on: 137 accepted in 300 s within +0.2..+0.8 deg of the lidar's yaw; off: the accepts walked together to 93 deg (+3.4) within minutes, each node pulling its own bias
  - *On when:* always while its cost fits the laptop: every update then localises, a hypothesis the words miss is still caught by the pose, and the per-node biases average out
  - *Off when:* when the camera's localisations arrive seconds late, or the laptop's CPU is short under a camera-only drive
- **`registration_backend`** — choice: service, local, auto, default auto
  - *What:* where RTAB-Map's XFeat keypoints and LighterGlue matches are computed (the xfeat visual features): service — the localisation service on the laptop's GPU (pepin.localization_service, PEPIN_MODELS_URL), no features when it does not answer; local — in RTAB-Map's own process on the Docker VM's CPU, as before 2026-09-24; auto — the service, and a call it does not answer computed locally. Written to /tmp/pepin/registration.json, which RTAB-Map's adapters read on every call (one stat): live, no restart of RTAB-Map (one of: service, local, auto) (PEPIN_REGISTRATION_BACKEND overrides the default at start)
  - *Default:* auto — auto, measured 2026-09-24 in RTAB-Map itself against the NIGHT branch's in-process adapters under the same conditions (scratch/models/backend_control.sh and .py: the replay of camera-only run 0466, words at LoopThr 0.05, four runs back to back, the live stack beside them): the registration of a localised update took 571 ms median through the service against 1896 and 2326 ms for the night's adapters in the runs before and after it (3.3-4.1x), 2428 ms for this branch's local fallback (the night's path, within that drift), 18 localised updates of 20 in each — one run, no truth, so a speed, not a recognition result; the 0.67 s of 2026-09-23 was a quieter laptop. From inside a container XFeat answers an 800x600 picture in 41 ms, a node picture already seen in 1.5 ms, LighterGlue a pair in 129 ms (scratch/models/endpoint_bench.py). auto keeps the old path as the answer to a service that is down
  - *On when:* service to measure the service alone (a registration it cannot answer then finds no features, which shows as no recognition), auto always otherwise
  - *Off when:* local to reproduce the registration of before 2026-09-24, or when the laptop's GPU is wanted elsewhere
- **`place_recognition`** — choice: words, descriptor, default descriptor
  - *What:* how RTAB-Map finds WHICH database node a picture is (its likelihood, before any registration): words — the ORB bag of words' TF-IDF (Kp/TfIdfLikelihoodUsed true, Rtabmap/VirtualPlaceLikelihoodRatio 0, RTAB-Map's defaults); descriptor — the dot product of the nodes' learned place descriptors as z-scores (false and 1; rtabmap Memory::computeLikelihood -> Signature::compareTo, Rtabmap::adjustLikelihood), which sensor_pack attaches to every snapshot. descriptor is sent ONLY when it cannot abort RTAB-Map: its core carries ros/patches/rtabmap-keep-global-descriptors.patch (the marker /opt/rtabmap_patches/keep-global-descriptors), the snapshots carry one each (/sensor_pack/place) and the database's census at this start (PEPIN_PLACE_CENSUS, taken by the launch before RTAB-Map opens the file) says every node carries exactly one of the same length — and only while the camera snapshots are described (descriptor_null_share); otherwise the words, and the report line says why. Live (one of: words, descriptor)
  - *Default:* descriptor — words until a drive has shown the descriptor on the robot (a default flips after a drive). Measured 2026-09-24 in RTAB-Map itself on the replay of the evening runs against the backfilled daylight database (scratch/models/replay_place.py, replay_matrix.sh, matrix_report.py; xfeat, 2 px, proximity on, a lidar-only snapshot every fourth). Words at the stock Rtabmap/LoopThr 0.11: hypotheses 0.05-0.09, 0 camera updates localised on 0457, 0460 and the camera-only 0466; words at 0.05 (visual_confirm aggressive, this node's default): 9 of 23 (9 of 9 judged right), 3 of 17 (0 of 1) and 53 of 55. Descriptor with the ratio 1 at 0.11: hypotheses 0.42-0.88 (median), 16 of 23 (15 of 15 right), 14 of 17 (11 of 11) and 54 of 55; at 0.05 two of 0460's eleven judged were WRONG — so the descriptor goes with visual_confirm rtabmap, and the report line says so when it does not. Judged counts are 16 or fewer a run and 0466 has no truth. With the ratio 0 the descriptor's hypotheses read 0.01 and nothing localised. The retrieval behind it: BoQ-DINOv2 R@1 0.986 on 219 evening frames (scratch/models/place_parity.py). Without the patch RTAB-Map 0.22.1 aborted at the first comparison after a registration (Signature.cpp:252), which is why the patch is a gate
  - *On when:* descriptor on a backfilled database (ros/tools/place_backfill.py) and a patched core, together with visual_confirm rtabmap: the descriptor's hypotheses reach 0.11 by themselves, and aggressive's 0.05 let the two wrong ones of 0460 through
  - *Off when:* words to reproduce RTAB-Map's stock place recognition, or when a drive shows the descriptor naming the wrong node; an unpatched core, a database not backfilled, snapshots without descriptors or a service that is not describing keep the words by themselves
- **`start_needs_placement`** — bool, default on
  - *What:* what goes out on /localization/placement (latched) says this start of RTAB-Map is PLACED only once an update has recognised a node of the database it loaded, or an operator's seed (/rtabmap/initialpose) has been heard since its first update — or it loaded an empty database, whose start pose is the map's origin. The board's goal clients (ros/tools/goto_ros.py, pepin_bringup.goal_server) refuse a goal until then, saying to seed or to let the camera see a mapped place. Off, every start counts as placed: RTAB-Map's pose is taken as it is
  - *Default:* on — on, measured 2026-09-23: after a restart RTAB-Map publishes map -> odom from the pose it SAVED at its last shutdown, before recognising anything, and the preflight took that fresh transform for a localisation — the cart was 'at home' while standing at the bookshelf (0 recognised a node in 130-198 updates, hypothesis 0.07), and after the next restart 76 cm off, inside the table, where Hybrid refused 'Start occupied' and the recoveries ran 93 times in 81 s (the journal, 19:05 and 20:57). It is the startup-zero trap a third time
  - *On when:* always: a pose nobody has vouched for since the start is not a pose to drive on
  - *Off when:* to drive on the saved start pose anyway — a cart known to stand exactly where RTAB-Map last shut down, with the camera unable to recognise anything (darkness) and no seed at hand. A refusal of SILENCE (this node down, respawning, or older than this flag) is lifted by the goal server's flag of the same name, which both goal clients obey

#### `run_recorder`

- **`planner_records`** — bool, default on
  - *What:* what the PLANNER saw goes on the tape too: the global costmap (run-length encoded, at most one grid per new plan), the goal status of Nav2's three actions (navigate_to_pose, compute_path_to_pose, follow_path); off, the tape holds what it held before 2026-09-18
  - *Default:* on — the tape was blind exactly where the failures were. On 2026-09-17 two legs piled up 78 and 90 recoveries in ~125 s with no path (ros/maps/rec/20260917_192935_goto.log, ..._201425_goto.log) and the tapes could not say why: they carry /plan and the LOCAL costmap, and the planner reads the GLOBAL one. The cart's own footprint was clear in every one of the 1740 taped local grids (scratch/footprint_in_costmap.py), so the answer was in the grid nobody recorded. Cost, measured on those tapes (scratch/costmap_rle_cost.py): the planner's grid is 239x215 = 51385 cells, 195 kB of raw JSON, and 16 kB run-length encoded over the four classes that decide whether the cart FITS (unknown / free / inflated / the 99-100 lethal band) — 12-fold, and the gradient it drops is cost, not feasibility. One grid per plan at the tapes' own 1.2 s plan cadence is 13 kB/s beside the 55 kB/s the scans already write, and one encode of 51k cells, 2.8 ms on the laptop's core. The status topics carry a message per transition and the graph's words arrive at 1 Hz
  - *On when:* always while Nav2 is the thing being debugged
  - *Off when:* on a long autonomy run where the tape must stay small, or to reproduce a tape recorded before 2026-09-18
- **`loc_from`** — choice: pose_topic, tf, default pose_topic
  - *What:* where the tape's `loc` rows come from: pose_topic, the goal server's /pose (it parses /tf for navigation anyway and republishes what it reads); tf, this node's own TF listener at 0.2 s, which is what it ran until 2026-09-22. The rows are identical either way — same fields, same 5 Hz, source 'tf' in both, because both are that same edge (one of: pose_topic, tf)
  - *Default:* pose_topic — an rclpy TF listener deserialises the WHOLE /tf stream — RTAB-Map's map -> odom at 20 Hz, the board's odom -> base_link at 50 Hz, the statics — to read one pose five times a second, and it cost this node ~34 % of a core on a board measured at 252 % with the real-time loops starving (2026-09-22). The goal server owns navigation and the jump watch, so its listener is the one that stays; a 5 Hz PoseStamped costs this node what any other small topic costs it
  - *On when:* wherever the goal server shares this machine (side all), which is where nav.launch.py sets it: one listener for the pose, and that node's pose_topic flag is the other half of the switch
  - *Off when:* tf where this node must read the edge itself: a SPLIT stack, where the goal server is the laptop's and a tape whose pose rows crossed the WiFi is what this recorder is on the board to prevent (nav.launch.py passes it there), or a tape that has to be compared against one written before 2026-09-22. The listener is then started on the next tick and is NOT stopped again by switching back: that costs a restart

#### `sensor_pack`

- **`sensor_pack`** — bool, default on
  - *What:* snapshots are published; off, the node subscribes and counts and RTAB-Map is fed nothing at all
  - *Default:* on — on by design: with subscribe_sensor_data there is no other input, so off is a mapper that stops adding nodes, and at Rtabmap/DetectionRate 1.0 that shows in RTAB-Map's own report within 1 s and in this node's within the 30 s of a window. It is the switch for isolating this node during a live test, not a behaviour A/B: the A/B against the old synchronised triple is the launch argument sensor_pack:=false, which also puts subscribe_depth and subscribe_scan back on RTAB-Map
  - *On when:* always, whenever the map is meant to grow or to localise
  - *Off when:* to prove that a node RTAB-Map reports is this node's and not a leftover subscription, and to take the camera's bytes off DDS while something else is measured
- **`sources`** — list of: camera, lidar, default camera,lidar
  - *What:* which sensors may enter a snapshot: the live A/B for camera-only and lidar-only mapping, with no restart and without muting a publisher (any of: camera, lidar, comma-separated)
  - *Default:* camera,lidar — both, because that is the whole point of the message: a node carries the scan the grid's plane is exact from AND the picture the place is recognised by. Dropping one here is exactly what the stack used to need a different launch and a different parameter table for (the SLAM_LIDAR and SLAM_CAMERA_ONLY tables of before 2026-09-19)
  - *On when:* lidar alone to reproduce the old lidar SLAM, camera alone for the honest camera-only test — the map is then only as true as the network's scale
  - *Off when:* never empty: with no source there is nothing to pack and RTAB-Map starves
- **`tf_retry`** — bool, default on
  - *What:* a member whose transform TF cannot answer for yet does not cost the snapshot: the moment is put back and offered again on the next arrival, until it has left that member's own pairing patience (pair_periods of its measured period). Off, the member is dropped at once and the snapshot goes out without it — the behaviour of before 2026-09-19
  - *Default:* on — about 5 % of camera frames answered 'base_link<-camera_optical: Lookup would require extrapolation into the future' and became lidar-only snapshots (measured live 2026-09-18): the neck's dynamic edge is published on the board and arrives over the bridge tens of milliseconds behind the frame it belongs to. The old answer was a 0.2 s BLOCKING wait inside the subscription callback, which both stalled the executor on every late frame — no picture, no depth and no revolution read while it waited — and still dropped those 5 %. A retry costs nothing and waits longer: arrivals come eighteen a second, so the patience is spent on the bridge rather than on this thread. The bound is the pairing bound itself and not a new number (176 ms for the camera at 8.5 Hz)
  - *On when:* always: a camera frame is the only thing in a snapshot that can recognise a place, and it is the one whose transform is late
  - *Off when:* to measure what the retry is worth — the report line's 'tf waits' against its 'frames TF could not place' is the same comparison with it on
- **`global_descriptor`** — choice: auto, on, off, default auto, not live
  - *What:* whether every snapshot carries exactly one rtabmap_msgs/GlobalDescriptor (type 1): the place vector of its picture (place_descriptor), or the null descriptor (zeros) when it has no picture or no vector came in place_timeout_s; said latched on /sensor_pack/place. auto: exactly when this image's RTAB-Map keeps a node's descriptor across the reload of its data (the marker /opt/rtabmap_patches/keep-global-descriptors, ros/xfeat/patch_rtabmap.sh); on: always; off: never, the snapshots of before 2026-09-24 (one of: auto, on, off) (PEPIN_GLOBAL_DESCRIPTOR overrides the default at start) (not live: set at the next start)
  - *Default:* auto — auto, because a descriptor is safe only on the patched core. RTAB-Map compares two nodes' descriptors only when BOTH carry one and ABORTS on a node with one against a node without (UASSERT, rtabmap Signature.cpp:252), and the unpatched 0.22.1 drops a node's descriptor whenever it reloads the node's data for a registration (Memory.cpp:2902, 2908) — also under the words, because mapping's rehearsal compares the last node with the next (Memory.cpp:3784-3834), and a rehearsal merge at rest makes the last node a saved one, whose data is reloaded (Memory.cpp:2763, 3958): the core aborts and RTAB-Map is not respawned. Measured 2026-09-24 (scratch/models/unpatched_mapping_control.sh, camera-only run 0466 mapping under the words): with descriptors the unpatched core died on the THIRD update — a rehearsal at similarity 0.97 merged the new node into the last, a proximity registration reloaded it, Signature.cpp:252 — and without them it ran all 17 clean; the patched core (the image's patch layer, rebuilt the same day) ran the same 17 with descriptors clean, every one merged at 0.96-0.98 and 15 loop closures. Not live: switched mid-run, the next node would differ from the last. With descriptors on board the launch raises Mem/RehearsalSimilarity to 0.92 (DESCRIPTOR_REHEARSAL in vslam.launch.py: the descriptor's scale, measured)
  - *On when:* on only for a control against an unpatched core (it can abort RTAB-Map); auto otherwise — before place_recognition descriptor, whose gate refuses a core without the patch and a database whose census is not whole
  - *Off when:* off against a database WITHOUT descriptors to reproduce the snapshots of before 2026-09-24 (PEPIN_GLOBAL_DESCRIPTOR=off ros/laptop.sh vslam); never against a backfilled database while RTAB-Map maps
- **`place_descriptor`** — bool, default on
  - *What:* a camera snapshot's descriptor is the localisation service's /place vector of its picture (pepin.localization_service, BoQ on the laptop's GPU); off, every snapshot carries the null descriptor and the service is not asked
  - *Default:* on — on, measured 2026-09-24: through the service the fixed set (scratch/vpr, 147 daylight nodes, 219 evening frames of runs 0455-0465) recognises evening R@1 0.986, R@3 1.000 against the daylight gallery, JPEG as sent here the same as lossless (scratch/models/place_parity.py), where the ORB words' hypotheses read 0.05-0.07 under the evening lamps. The call costs this node nothing: it runs on a worker thread
  - *On when:* whenever place_recognition descriptor is used, or its vectors are to be stored in the database while it maps
  - *Off when:* to take the service out of the loop without breaking the one-descriptor invariant (a sick service, a measurement of the words alone); RTAB-Map then sees every place as equally likely by descriptor, so the words should be in force

#### `tof_bridge`

- **`range_as`** — choice: scan, range, default scan
  - *What:* what Nav2 is fed with: scan also publishes each cone on /tof/<name>/scan as a small LaserScan fan for an ObstacleLayer; range publishes nothing there, which is the node before 2026-09-21 and needs the three RangeSensorLayer blocks back in the local costmap's plugins list. The sensor_msgs/Range topics are published either way (one of: scan, range)
  - *Default:* scan — nav2_costmap_2d::RangeSensorLayer carries two defects that are both unfixed on main and have each stopped this robot. One: it asks tf2 to transform every message at the message's own stamp with transform_tolerance as the timeout, and tf2 blocks the whole timeout on any failure, so a broken chain amplifies 4.5x per update cycle until updateMap never returns (4 of 7 board starts on 2026-09-21; scratch/nav2_hang/wedge_gain.py). Two: after clamping its cell bounds to the grid (range_sensor_layer.cpp:362-369) bx1/by1 stay NEGATIVE when the cone falls off the left or bottom edge, and the loops cast them to unsigned: about 4e9 iterations holding the costmap mutex, one thread at 100 % for ever, 'Pose Goes Off Grid', every service timing out and zero plans. It takes one jump of the pose in map between a reading's stamp and the update — a tracker restart, a relocalisation, the cart carried by hand — and no publisher can gate it, because the jump happens after the reading has left. Reproduced on 2026-09-21 with ros/thin.sh kick relocalizer (tid 191 of the Nav2 container: 415 s of CPU in 700 s). An ObstacleLayer drops what it cannot place instead of blocking on it and walks points instead of a cell rectangle, so it has neither; the fan is what turns one distance into points, one beam per costmap cell of arc at the sensor's ceiling (pepin.tof_horizon.cone_beams: 11 beams front, 7 and 7 at the sides), every beam carrying that distance
  - *On when:* always while Nav2 reads the ToF: it is the arrangement with no unbounded loop and no blocking transform in it
  - *Off when:* to compare against the old plugin, or if an ObstacleLayer ever proves worse at a cone than the range layer was — put tof_front_layer, tof_left_layer and tof_right_layer back into the local costmap's plugins list at the same time, or the whiskers reach no costmap at all

#### `visual_odometry`

- **`vo_publish`** — bool, default on
  - *What:* the gated visual odometry leaves this laptop as /vo, where the board's EKF fuses it as a third input beside the wheels and the gyro; off, the node still measures and reports and the EKF is exactly what it was without it
  - *Default:* on — on since 2026-09-14 13:40: at full rate (vo_publish_hz 10) the board's EKF missed its 20 Hz period 0 times in 130 s at rest and twice in 3 min of driving, |vy| stayed under 0.0003 m/s at rest, the odometry runaway guard counted 0, and the live pose against the lidar truth was 0.9 / 2.1 / 4.2 cm — the same as without (tapes 0261/0262 vs 0265/0266). Before that it shipped off because the half that matters is unmeasured. AT REST it is measured and it passes: on this laptop's live topics, with the launch's own parameters, this node gated 9.4-9.7 poses/s and dropped none, and the drift over 60 s with the wheels reporting a hard zero was 0.2 cm and 0.0 deg (worst stretch 0.4 cm); the same camera read by scratch/vo_probe.py for 85 s gave 9.1 poses/s, 630 inlier features a frame, one lost frame (the first) and 0.48 cm / 0.075 deg — against the centimetre and half-degree a minute this source has to stay under (2026-09-14). IN MOTION nobody has compared it with anything, and the EKF's odom -> base_link is what every other measurement in the stack is carried over: the tracker's scans, the camera's measurements, the costmaps. The scale is why the caution is not ceremony — the translation rgbd_odometry reports is the depth image's, and that depth is a network's corrected by a law fitted against the lidar (0.94 to 1.98 across one afternoon, 2026-09-11)
  - *On when:* after one drive compares odom -> base_link with it on and off over the same path (it is a live flag exactly so the two runs are a minute apart) and the visual odometry did not disagree with a lidar-measured distance by more than the wheels did
  - *Off when:* the moment odom -> base_link must be the wheels and the gyro alone: a dark room, a blank wall, a depth law that has not been fitted this session, or any drive whose odometry is the measurement
- **`vo_covariance`** — choice: dynamic, constant, rtabmap, default dynamic
  - *What:* whose covariance rides on the published pose: `dynamic`, the registration's own sigma and the depth scale's share of the step just taken added in quadrature (pepin.visual_odometry.scaled_covariance); the documented constant (vo_sigma_m, vo_yaw_sigma_deg); or the one rtabmap's registration computed, untouched (one of: dynamic, constant, rtabmap)
  - *Default:* dynamic — the constant, because rtabmap's own number answers the wrong question. Measured at rest on this robot (2026-09-14, scratch/vo_probe.py, 85 s): its registration claimed a position standard deviation of 3.8 mm at the median and 15.9 mm at p90 — an honest spread of the feature matches, and a claim about the PICTURE. The error that matters is the scale of the depth those features sit on, and that scale is a network's law fitted against the lidar (0.94 to 1.98 across one afternoon, 2026-09-11), which no registration can see. So the topic carries a constant a person can argue with, and rtabmap's own is one flag away for the session that wants to compare them — one flag away and worth reading twice before it is turned on with vo_publish: 3.8 mm through robot_localization's differential conversion (2 * sigma^2 * dt) is a velocity sigma of 1.8 mm/s, 325x the wheels' certainty per sample, which is no longer a third opinion but the whole odometry (scratch/vo_weight.py)
  - *On when:* never as such — it is a choice: 'rtabmap' while comparing the two on a tape, and with vo_publish off unless the point of the session is that comparison
  - *Off when:* 'constant' is the shipping value; leave it there unless a session is about the covariance itself

### Config knobs

A knob is a number, not a switch: its default, its range and one line of note live in
`config/knobs.json` under the node's name, the node declares it as a live parameter beside its
flags, and `ros/flags.sh get|set NODE KNOB` reaches it like a flag (a value outside the range is
refused with the reason). A default moves by editing that file; a change made live lasts until
the node restarts. Where the code already names the number, a unit test holds the two equal
(tests/unit/test_knobs.py).

| node | knob | kind | default | note |
| --- | --- | --- | --- | --- |
| `camera_stream` | `scale` | number 0..1 | 0.5 | the published picture as a fraction of the camera's own 1280x720, its optics scaled with it; a change takes the next frame. THE MONO RIG's flag: a stereo head publishes at its calibration's own size (the size the remap tables were built for, the size a matcher's disparity is in pixels of), so the node pins this to 1.0 there and refuses any other value with that reason |
| `contact_scan` | `max_range` | number 0.1..10 | 2.0 | metres past which a column is called clear instead of ended; the costmap's contact_layer.obstacle_max_range must match it |
| `depth_fusion` | `paint_sigma_m` | number 0.01..2 | 0.25 | how sure of itself the tracker must be, in metres of sigma_xy, before this node paints with its pose — read from /localization/sigma, and ignored entirely while nothing publishes that topic (the fit gate stands on its own until it exists) |
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
| `depth_fusion` | `snapshot_s` | number 0..3600 | 60.0 | how often the volume is written to world_path (0: only at shutdown) |
| `depth_fusion` | `follow_correction_min_m` | number 0..5 | 0.05 | how far the graph must have bent before the volume is resampled; smaller bends are kept against the same anchor and move it together when they add up |
| `depth_fusion` | `follow_correction_min_deg` | number 0..180 | 1.0 | how far the graph's bend must have TURNED before the volume is resampled: the other half of the threshold, because a turn moves the far end of the flat metres while the origin stands still |
| `depth_fusion` | `follow_correction_min_s` | number 0..60 | 2.0 | the shortest time between two moves of the volume: a burst of graph optimisations costs one resample, not one each. The correction is not lost (it is owed against the same anchor and applied at the next move) — but the frames and revolutions of that window are not painted, because a volume that owes a move is not the map they were placed in |
| `depth_stream` | `scale_ceiling` | number 0.5..20 | 5.0 | the largest 1 / scale the law may be fitted to (the upper half of pepin.depth.A_BOUNDS); a law that lands on a bound prints AT BOUND |
| `depth_stream` | `law_slew` | number 0..1 | 0.0 | how fast the affine law may move, as the largest relative change of the published inverse depth over the pool's own depth range, per second; 0 applies every fit whole, as the node always did. A law still walking to its fit says so in the report line (slewing to a X b Y) |
| `depth_stream` | `carry_max_speed_mps` | number 0.1..20 | 1.0 | metres per second the carry from the scan's moment to the frame's may imply before the frame's lidar beams are thrown away instead of anchoring the law; the frame still publishes its depth, it simply judges nothing |
| `depth_stream` | `lean_min_quality` | number 0..1 | 0.5 | how much of the lean gravity must have voted for (pepin.lean's quality, printed beside the lean in this line) before a frame is placed by it: below it the lean is treated as unknown and the frame is placed level |
| `depth_stream` | `tf_dead_s` | number 0..600 | 3.0 | how far behind a frame's stamp TF's newest edge may be before that edge is taken for dead and no lookup on the frame's path waits for it: the camera pose falls to config/camera.json's mount and the lidar's scan passes uncarried, both at once and both counted. 0 turns the guard off — every lookup waits CARRY_WAIT_S again |
| `depth_stream` | `lidar_sigma_m` | number 0..0.2 | 0.0 | what one lidar beam's range is trusted to, in metres. 0 (the default) gives every beam the flat weight of 1, the reference pair. Above 0, a beam's weight is 1 / sigma^2 in inverse depth, sigma_m / z^2, which reads as a weight proportional to z^4 |
| `depth_stream` | `depth_reach_m` | number 0.3..12 | 4.0 | metres past which the published depth is NaN; the same number /depth_scan is capped at |
| `depth_stream` | `scan_hz` | number 0..30 | 5.0 | the cap on how often /depth_scan is PUBLISHED, in hertz; 0 publishes one fan per frame, which is what this topic did until 2026-09-22. The cap is on the publisher alone: every frame still goes through the network and the whole pipeline, every law is still fitted from it, and the depth image on /camera/depth is not thinned at all |
| `marks_audit` | `radius_m` | number 0.2..3 | 2.0 | how far around the cart a lethal cell is judged, metres |
| `marks_audit` | `match_cells` | number 0.5..5 | 1.5 | how near a beam must land to a cell, in costmap cells, to account for it |
| `rtabmap_frame` | `pnp_reproj_px` | number 1..4 | 2.0 | Vis/PnPReprojError: how far, in pixels, a database point may reproject from its match in the current picture and still count as an inlier of the visual registration (20 inliers accept it, Vis/MinInliers). Sent with the strategy's parameters and changed live |
| `rtabmap_frame` | `registration_timeout_s` | number 0.05..5 | 1.0 | how long RTAB-Map's adapters wait for one answer of the localisation service before it counts as none (auto then computes it locally, and the service is left alone for 10 s) |
| `rtabmap_frame` | `xfeat_top_k` | integer 256..8192 | 2048 | the most XFeat keypoints a picture keeps, best score first, wherever XFeat runs; RTAB-Map applies no cap of its own to a Python detector |
| `rtabmap_frame` | `descriptor_null_share` | number 0..1 | 0.5 | place_recognition descriptor falls back to the words while more than this share of the last 10 camera snapshots carried the null descriptor (the localisation service down, late or answering nonsense), and goes back once they are described again; 1.0 never falls back |
| `sensor_pack` | `pack_hz` | number 0.1..15 | 1.0 | at most this many snapshots a second of SENSOR time (the stamps' own clock, not this laptop's) |
| `sensor_pack` | `pair_periods` | number 0.5..10 | 1.5 | how many of its OWN measured periods a source's message may be from the snapshot's stamp and still be paired with it (pepin.snapshot) |
| `sensor_pack` | `place_timeout_s` | number 0.05..0.95 | 0.5 | the most a snapshot waits for its place vector, from the moment it is packed — its wait in the worker's queue included — before it goes out with the null descriptor |
| `visual_odometry` | `vo_sigma_m` | number 0.001..1 | 0.07 | the constant position sigma of one visual-odometry pose, in metres; the EKF differences two of them into a velocity and the covariance rides along — as (this pose's + the previous pose's) TIMES the gap, so what the filter actually weighs is a velocity variance of 2 * sigma^2 * dt |
| `visual_odometry` | `vo_yaw_sigma_deg` | number 0.1..180 | 5.0 | the constant yaw sigma of one visual-odometry pose, in degrees; since 2026-09-15 the board's EKF fuses this yaw differentially (ekf.yaml odom1_config index 5), so this number is what sizes a second heading source against the gyro |
| `visual_odometry` | `vo_max_speed` | number 0.05..10 | 1.0 | a step between two visual-odometry poses faster than this, in m/s, is dropped: rtabmap restarting its tracking moves the pose without moving the cart |
| `visual_odometry` | `vo_max_gap_s` | number 0.1..60 | 1.0 | a pose that arrives more than this many seconds after the previous one is dropped and becomes the new anchor: across a gap the speed and turn ceilings are ratios and measure nothing |
| `visual_odometry` | `vo_max_turn` | number 5..720 | 180.0 | a turn between two visual-odometry poses faster than this, in deg/s, is dropped, for the same reason as vo_max_speed |
| `visual_odometry` | `vo_reset_radius_m` | number 0..1 | 0.05 | a pose that lands this close to rtabmap's own origin while the previous one was farther out is its re-initialisation, not a drive, and is dropped; 0 turns the check off |
| `visual_odometry` | `vo_publish_hz` | number 0..30 | 10.0 | how often a gated pose may leave for the board's EKF, in hertz; 0 publishes every one of them |

## The base bridge's zero-velocity update

The C++ base bridge's own switches, beside `imu_publish` and `odom_publish` above; they live here
because the table above is generated from the Python nodes, and the Python bridge has no gyro and
no update. While the cart is certainly standing still the bridge publishes `/zupt` — a twist of
exactly zero, the EKF's `odom2` (vx, vy, vyaw; `ros/params/ekf.yaml`) — and nothing otherwise.
Certainly still is three witnesses at once: the wheels at rest for `zupt_settle_s`, no non-zero
`/cmd_vel` younger than `zupt_cmd_hold_s`, and the bias-corrected gyro under
`zupt_gyro_quiet_rad_s` for that same window. The first moving sample of any of them stops the
update within one period. Why it exists: parked, the EKF's heading crept ~5 deg an hour, pulled
by rf2o's +1.5 deg/min at rest while the gyro read -0.001 (2026-09-24).

**Every tunable is live.** `ros2 param set /base_bridge <name> <value>` takes effect at the next
tick, with no restart and no rebuild; a new `zupt_rate_hz` re-times the timer at once. A value
outside its range, or not a number, is refused: `ros2 param set` prints the reason, the bridge
logs it as a warning, and the value in force stays. The CLI's own quirks are absorbed: `50`
arrives as an integer and `1e-4` as a string (YAML 1.1 wants `1.0e-4`), and both are taken as the
numbers they are. The same range check applies to a value given in a launch file, which falls
back to the default with a warning instead of stopping the base.

| name | default | range | what it does |
| --- | --- | --- | --- |
| `zupt_publish` | true | bool | the update at all; false is the bridge before 2026-09-24: nothing on `/zupt` |
| `zupt_rate_hz` | 10 | 1..100 Hz | how often the update is published while the cart is at rest |
| `zupt_var_linear` | 1e-6 | 1e-9..1 (m/s)^2 | the variance claimed on vx and vy |
| `zupt_var_yaw` | 1e-6 | 1e-9..1 (rad/s)^2 | the variance claimed on vyaw (the gyro's is 4e-4, rf2o's 2.5e-3) |
| `zupt_settle_s` | `imu_bias_s` (2.0) | 0..60 s | witnessed rest before the update, and the hold after a gyro turn; the bias tracker's own window stays `imu_bias_s` |
| `zupt_cmd_hold_s` | `cmd_timeout_s` (0.5) | 0..10 s | how long a non-zero `/cmd_vel` holds the update off; the command's own timeout is not moved |
| `zupt_gyro_quiet_rad_s` | 0.005 | 1e-4..0.5 rad/s | a bias-corrected yaw rate at or above this is a turn (0.005 = 0.29 deg/s, 7.9 sigma of the parked chip's noise) |

The reasons for each default and each range are in
`ros/pepin_base_cpp/include/pepin_base_cpp/zupt.hpp` (its contract: `test/zupt_contract.cpp`).
Two numbers stay fixed, because they are structural: the variance on the three velocities the
update does not claim (1e6: `odom2` fuses none of them, so the EKF never reads it), and how old
the newest gyro sample may be (1 s, the same gap that ends the wheels' witness; at 50 Hz a sample
is that old only after ~50 failed reads in a row).

**What the bridge says.** Each change is one log line (`zupt: zupt_rate_hz 10 -> 50, in force at
the next tick`), each refusal a warning with its reason, and each start or stop of the update a
line with the reason it stopped. The link-up line and the once-a-minute gyro line end with the
update's state and every setting in force:

    zupt publishing for 312 s, 3121 sent [rate 10 Hz, var 1e-06 xy 1e-06 yaw, settle 2 s,
    cmd hold 0.5 s, gyro quiet 0.005 rad/s]

**Parked A/B, one variable at a time** (read the heading creep between each, e.g.
`scratch/link_autopsy/rest_yaw_sources.py`; the simulation in `scratch/zupt/zupt_variance_sim.py`
says the rate matters more than the variance past 1e-5):

```bash
ros2 param set /base_bridge zupt_publish false        # the filter as it was: the baseline creep
ros2 param set /base_bridge zupt_publish true
ros2 param set /base_bridge zupt_rate_hz 10.0         # the default, phase-locked to rf2o's scans
ros2 param set /base_bridge zupt_rate_hz 50           # every gyro sample's worth
ros2 param set /base_bridge zupt_var_yaw 1.0e-6       # the default claim on vyaw
ros2 param set /base_bridge zupt_var_yaw 1.0e-4       # a hundred times looser
ros2 param set /base_bridge zupt_var_linear 1.0e-4    # ...and on vx, vy
ros2 param get /base_bridge zupt_rate_hz              # what is in force
```

## Two recorders

Every drive is written down, and there are two ways to do it. Which one runs is
`PEPIN_RECORDER` on the board (`board/pepin-ros.service`, the `recorder` argument of
`nav.launch.py`); the operator flips it with one command and never touches the analysis side,
because both roads end in the same file — `ros/maps/rec/NNNN_<utc>Z_<goal>.jsonl`.

| | `PEPIN_RECORDER=jsonl` (default) | `PEPIN_RECORDER=bag` |
| --- | --- | --- |
| who writes | `pepin_bringup.run_recorder`, on the board | `ros2 bag record` (MCAP, no compression), started by `pepin_bringup.bag_recorder` |
| board cost | 34-43 % of a core: rclpy deserialises 450 floats 10x/s, the TF buffer runs, `json.dumps` writes | an estimated 8 % for the node plus rosbag2's copy of serialised bytes — nothing deserialises a message |
| what lands there | the tape itself, plus the camera clip | `NNNN_<utc>Z_<goal>/` (MCAP) plus the camera clip |
| the tape | already written | made on the laptop by `ros/tools/bag_to_tape.py`, which `ros/goto.sh` runs for you |
| prelude | the 15 s before the goal are on the tape (`pepin.tape.RunTape`) | none: the bag begins when the goal's word does |

Both recorders answer the same protocol (`pepin.runlink`): the goal server — on the board or on
the laptop — publishes `{"cmd": "start", "name": ...}` on `pepin/run` and reads the run's number
and path back from the latched `pepin/run_status`, so `ros/go.sh` and `ros/goto.sh` are unchanged.
The bag records every topic the JSONL recorder subscribes to (`pepin.tape_rows.TOPIC_RECORDS`)
plus `/tf`, `/tf_static` and `/odom_laser`, with `--include-hidden-topics` for Nav2's action
status topics and a QoS override for the two latched ones
([`ros/params/rosbag_qos.yaml`](params/rosbag_qos.yaml)).

```bash
ros/feature.sh recorder bag      # the board restarts the stack; a drive now writes an MCAP bag
ros/feature.sh recorder jsonl    # back to the tape written on the board
ros/goto.sh home                 # either way: "numbered tape: ros/maps/rec/0251_...jsonl"
# by hand, if a bag was fetched without its conversion (the laptop's ROS container):
docker exec pepin-vslam /pepin_entrypoint.sh python3 /tools/bag_to_tape.py /maps/rec/0251_... --force
```

The converter rebuilds every record from the same functions the live recorder uses
(`pepin.tape_rows`), so a converted tape is byte-identical in everything but the two things it
cannot be: the records with no stamp of their own (`cmd`, `nav`, `meas`, `srcs`) are dated by the
bag's receive time, and there is no prelude. `tests/unit/test_bag_to_tape.py` feeds both paths
the same messages and compares the rows. The `loc` records come from `/tracker_pose` when the bag
holds one and are composed from `/tf` (`map -> odom` times `odom -> base_link`) at 5 Hz when it
does not, which is what the recorder does on a stack whose localiser publishes no pose.

**The default stays `jsonl` until the two are measured on the board** (CLAUDE.md rule 19): the
bag's manifest entries in [`config/board_manifest.json`](../config/board_manifest.json) are
estimates, marked as such, and a census during a bag drive is what replaces them. The image must
carry `ros-jazzy-ros2bag` and `ros-jazzy-rosbag2-storage-mcap` ([`ros/Dockerfile`](Dockerfile));
an image built before they were added logs `ros2 bag record` as not found at the first goal and
the drive goes on unrecorded.

## Replay

A costmap change is judged on the recorded drives before it is driven. `ros/replay.sh` runs every
board bag (`ros/maps/rec/NNNN_*/`, MCAP, since 2026-09-23) through the stock `nav2_costmap_2d`
local and global costmaps of the board's own image, configured from `ros/params/nav2_params.yaml`
and whatever is layered on top, and prints one row per drive.

```bash
ros/replay.sh 483-498                                       # the parameters as they are
ros/replay.sh 483-498 --set both.camera_layer.enabled=false \
    --against ros/replay/baselines/0483-0498.json           # candidate minus baseline, per drive
ros/replay.sh 483-498 --params scratch/candidate.yaml       # a params file layered last
ros/replay.sh 483-498 --save ros/replay/baselines/NAME.json # a new baseline (+ NAME.txt)
```

It runs in a throwaway `pepin-ros` container with `--network none` (nothing reaches the robot or
the live containers); a worktree points `PEPIN_REPLAY_REC` / `PEPIN_REPLAY_MAPS` at the main
checkout's `ros/maps/rec` and `ros/maps`. 16 drives, 841 s of goals: 18 s wall on one job, ~9 s
on six (preparing the bags the first time: +6 s; building the engine: ~10 s once).

**How.** `ros2 bag play --rate N` into the stock nodes does not work: `Costmap2DROS` paces
`updateMap()` with a `WallRate`, so at N times real time each update folds N times more scans and
the result depends on the rate and the scheduler. `ros/replay/engine` (C++, `costmap_replay`) is
the costmaps' clock and loop instead: bag time is their ROS time, `/tf` goes straight into their
buffers, each observation into its ObstacleLayer source's buffer through the layer's own callback
as soon as TF places it (dropped after `transform_tolerance`, as the MessageFilter does), and
`updateMap()` runs at the live 5 Hz / 2 Hz in bag time. Same input, same output, bit for bit.
`ros/replay/prepare.py` rebuilds what the costmaps were fed from what the bag keeps: `/scan` from
the raw LD19 scan through the hull box, the ToF fans from the Ranges (`tof_bridge.fan_scan`),
`/map` from RTAB-Map's saved grid (the bag has none), and the tree's `ClearEntireCostmap` calls
from the action status topics (every 5 s / 10 s from the goal and from each pipeline restart, one
per controller or planner failure).

**Score** (`ros/replay/score.py`), while the goal is active, after a 2 s warm-up per bag:

| column | what |
| --- | --- |
| `stuck_s` | seconds the recorded `/cmd_vel` held zero (or went silent > 0.5 s): ground truth, no parameter moves it |
| `goal_max`, `goal_end` | lethal cells of the global costmap within 0.35 m of the goal: worst snapshot, last one (the map's own floor: printer 25, home 2, bookshelf 1) |
| `corr_p50`, `corr_max` | lethal + inscribed cells of the global costmap within 0.45 m of each recorded `/plan` |
| `cam_mean`, `cam_max` | lethal cells of the local costmap within 2 m with no `/scan` return within 1.5 cells (`pepin.marks_audit`) |
| `fid` | F1 of the replayed local costmap's lethal cells against the recorded `/local_costmap/costmap`: 0.98 on the baseline; for a candidate, how far it moved the costmap |
| `unseen_s` | seconds of the goal before the bag began (the recorder starts late): not scored |

**What it cannot replay.** The camera volume is not re-fused: `/depth_marks` is replayed as
recorded (no camera frames in the board bag), so depth, stereo and depth_fusion changes are out
of scope. `/depth_scan`, the camera's per-frame clearing, is not in these bags: the camera layer
clears only by the tree's clears (the replay passes it through when a bag has it). The loop is
open: the recorded plan and motion, no planner or controller run. The clears are reconstructed;
a pipeline that failed with nothing running to cancel leaves no trace.

## Simulation

A kinematic simulator for Nav2's behaviour, on the Mac: `ros/sim.sh`. It runs **our** Nav2 —
`nav.launch.py side:=all` included whole (`ros/sim/sim_nav.launch.py`), `ros/params/nav2_params.yaml`
with the `PEPIN_LOCALIZER=rtabmap` overlay, the goal server with its saved planner pick, the run
recorder — in the board's own image, against a simulated cart in the room RTAB-Map saved. Goals go
through the goal server's socket with the client `ros/goto.sh` uses (`pepin.goal_link`), and every
goal is taped like a drive.

```bash
ros/sim.sh up                 # ~10 s to "Managed nodes are active"; the cart starts at home
ros/sim.sh goal printer       # or X Y [YAW_DEG]; prints goto's lines, then a SCORE line
ros/sim.sh scenario tour.yaml # ros/sim/scenarios: place, furnish, drive each leg, score it
ros/sim.sh stuck --repeat 3   # the shelf pocket (shelf_pocket.yaml)
ros/sim.sh place bookshelf    # teleport, standing still; both costmaps are emptied
ros/sim.sh up --rate 2        # the world's clock at 2x the wall, every node on use_sim_time
ros/sim.sh up --nav-first 20  # Nav2 up 20 s before anything publishes /map or TF
ros/sim.sh down               # gentle stop, logs kept in logs/containers
ros/sim.sh map                # re-export ros/sim/worlds/flat from ros/maps/rtabmap.db
```

**What it models** (`pepin.sim`, `ros/sim/sim_world.py`): RTAB-Map's saved grid (`Admin.opt_map`)
latched on `/map`, the places resolved on the saved graph on `/places`, the placement word,
`map -> odom` as identity (the sim's pose is the truth), `odom -> base_link` and `/odom` at 20 Hz;
the base as a unicycle at `config/base.json`'s limits (0.30 m/s, 1.0 rad/s) behind the base
bridge's 0.5 s deadman, refused any step that puts more of the hull into the grid or a box
(a contact, counted); `/scan` at 10 Hz raycast (exact DDA) from the grid and the boxes through the
LD19's own mount (`config/lidar.json`: upside down, 87.5 deg), its 455 bins, the posts' masked
sectors and the hull-box scan filter. Scenario boxes are furniture the grid does not hold: only
the lidar sees them. A score is the world's truth: time, recoveries, the hull's path against the
straight line, contacts, the arrival error. Isolation: the three containers share one network
namespace with no interface but loopback, no port published, `ROS_DOMAIN_ID` 42.

**What it does not model**: wheel slip and odometry drift, localisation error (no RTAB-Map, no
`map -> odom` steps), lidar noise and dropouts (dark or glossy surfaces), the camera (no depth
marks, no phantoms, no camera grid) and the ToF sensors, WiFi and zenoh stalls, the board's CPU
(Nav2 runs on the Mac at 10 Hz), light. The lidar sees the grid: where RTAB-Map's grid holds
something the LD19 looks under or past from where the cart stands, the simulated lidar sees a
wall (at the printer park the real scan runs 2 m through a block the grid has at 0.46 m;
`scratch/ros_sim/sim_vs_tape.py`), which makes the controller stricter there than on the robot.
The board's image carries Nav2 1.3.13, the laptop's 1.3.12; the sim runs the planner in the former.

**Measured 2026-09-27** (today's params: Hybrid + FollowPathShim; scores in
`ros/sim/run/maps/rec/sim_scores.jsonl`, the analyses in `scratch/ros_sim/`):

| scenario | sim (wall clock) | the robot |
| --- | --- | --- |
| tour: home -> printer | 113.7 s, 58 recoveries (twice: 115.9 s, 58) — the plan hugs the table block, RPP "collision ahead" 474x | 15-16 s (0485, 0491) |
| tour: printer -> bookshelf | 15.2 s, 5 recoveries | 12 s, 5 (0492) |
| tour: bookshelf -> home | 15.1 s, 5 recoveries | 28 s (0490); 133 s and 257 s (0493, 0498) |
| shelf_park: the 0492 arrival, grid only, -> home (2 rounds) | 25.2 s / 10 rec., 20.6 s / 8 | 28 s / 9 (0490) |
| shelf_pocket: + the sides at 0.41 / 0.46 m (3 rounds) | 99.8 s / 46 rec. / 10 contacts, 33.0 s / 21, 70.5 s / 40 | 133 s / 82, 257 s / 218 (0493, 0498) |
| corridor (control, 3 rounds) | 5.4 s and 11.1 s (median) | — |

**Faster than the wall clock**: the world keeps any rate (`clock x` in its report line), Nav2 does
not — it paces its controller, costmap and tree loops on the wall clock. On the corridor (sim
seconds, median of 3): x1 5.1 / 15.4 s (the wall clock's 5.4 / 11.1 within its spread), x2 6.1 /
18.8 s, x4 9.3 / 27.4 s with longer paths. Timing claims at x1; x2 screens pass/fail at twice the
speed; x4 is a different robot. Past x5 the velocity smoother's 10 Hz (wall) would also cross the
0.5 s deadman between two commands.

**Start order** (`--nav-first S`, journal 2026-09-25's "planner_server never activated"): with the
world 30 s late Nav2 waits and is active 1 s after the first TF, and drives; 70 s late, the
local costmap gives up on `odom` at 63 s ("Failed to activate local_costmap because transform
from base_link to odom did not become available before timeout"), the lifecycle manager aborts
the bring-up and never retries — the world arrived 6 s later and Nav2 stayed down until a restart.
**Cancel** (`ros/sim.sh cancel`, the goal server's socket): the goal ends CANCELED and the cart,
at 0.30 m/s, stands still at the first read 0.43 s after the cancel returned.

## What runs on the board

Four A53 cores and 1.5 GB. Everything the board is allowed to run is declared once, with a
budget, in [`config/board_manifest.json`](../config/board_manifest.json); `ros/board.sh census`
compares that file against a live `ps` and `ros/board.sh manifest` prints the registry itself.
Two rules hold this together:

- **Nothing is added to the board without a manifest entry with a measured budget.** The entry
  says what the process is, which of the three reasons of CLAUDE.md rule 20 puts it here
  (real-time, survives a WiFi loss, wired to the board's own pins), which systemd unit or launch
  file owns it, and what it may cost in CPU and resident memory. A feature that only works while
  the laptop is alive belongs on the laptop.
- **The census runs after every deploy.** `ros/sync.sh` ends with it; it never fails the deploy,
  it only prints. Run it by hand whenever the board feels slow.

```bash
ros/board.sh census          # the table and the verdict; exit 1 when red
ros/board.sh census --json   # the same as data
ros/board.sh manifest        # the registry: what we run there and why (touches no host)
```

A census is one `ps` over the multiplexed ssh — no `ros2` CLI (one `ros2 node list` costs
seconds of CPU on this board), no `docker exec`, nothing restarted. The verdict is red when
anything is **OVER** its budget, **MISSING** (expected always, not running), **FORBIDDEN**
(declared `expected: false` and running anyway) or **UNLISTED** (a process above 1 % CPU that no
entry claims — usually a `ros2 topic hz` left behind, which is what takes this board to load 12).
Entries marked `sometimes` (the per-drive recorder, the reaper, the SLAM-only and split-only
nodes) are **IDLE** when absent, never missing. The census is also a health probe (`board
budget`) in `scripts/health_check.py` and the tray.

Two things to know about the numbers before reading a table: ps's `%CPU` is the process's
average over its whole **life**, not an instant sample — a census taken a minute after a restart
shows start-up cost, and the report says so — and it is a percentage of **one** core, so 400 %
is the whole board.

| process | what it is | why on the board | budget | owner |
| --- | --- | --- | --- | --- |
| `nav2_container` | Nav2 in one process: map server, planner, controller, behaviours, tree, smoother (nice 5) | real-time, wifi-loss | 120 % / 152 MB | `pepin-ros.service` -> `nav.launch.py` |
| `lidar_container` | LD19 driver, hull filter, its static mount, the driver's lifecycle manager (nice -10, respawned) | real-time, hardware-attached | 22 % / 70 MB (to measure) | `pepin-ros.service` -> `robot.launch.py` |
| `base_container` | base bridge (wheels, IMU, gyro-bias tracker), the IMU's static mount (nice -10, respawned) | real-time, hardware-attached | 15 % / 60 MB (to measure) | `pepin-ros.service` -> `robot.launch.py` |
| `run_recorder` | every drive on disk: scans, odometry, pose, commands, camera | wifi-loss | 24 % / 93 MB | `pepin-ros.service` -> `nav.launch.py` |
| `tof_bridge` | the three VL53L1X ranges as ROS `Range` for the contact layer | real-time, hardware-attached | 20 % / 102 MB | `pepin-ros.service` -> `robot.launch.py` |
| `neck_state` | the neck's encoders, and `base_link -> camera_link` behind its flag | hardware-attached | 16 % / 99 MB | `pepin-ros.service` -> `robot.launch.py` |
| `base_server` | wheels, odometry and the deadman next to the UART (TCP 3336) | real-time, wifi-loss, hardware-attached | 16 % / 27 MB | `pepin-base.service` |
| `ekf_node` | wheels + gyro + the camera's odometry + the lidar's scan-to-scan odometry fused in the plane, owns `odom -> base_link`; runs on whichever of them are alive, the IMU included or not | real-time, wifi-loss | 14 % / 42 MB | `pepin-ros.service` -> `robot.launch.py` |
| `laser_odometry` | each LD19 scan matched against the one before it (rf2o, no map) as `/odom_laser` for the filter | real-time, wifi-loss | 30 % / 105 MB **(estimate, to measure)** | `pepin-ros.service` -> `robot.launch.py` |
| `tof_server` | the ToF sensors on I2C as a TCP stream (3335) | real-time, hardware-attached | 13 % / 22 MB | `pepin-tof.service` |
| `goal_server` | goals on a socket, with the places book of the map in use | wifi-loss | 10 % / 113 MB | `pepin-ros.service` -> `nav.launch.py` |
| `ros2_launch` | the launch process that started and respawns the ROS nodes | wifi-loss | 10 % / 105 MB | `pepin-ros.service` ExecStart |
| `ser2net` | servo bus and lidar as TCP ports 3333/3334 | hardware-attached | 6 % / 5 MB | `ser2net.service` |
| `ustreamer` | the overview camera as MJPEG on 8080 — frames copied, never decoded | hardware-attached | 3 % / 19 MB | `pepin-camera.service` |
| `docker` | dockerd, containerd and one supervisor per container | wifi-loss | 3 % / 225 MB | `docker.service`, `containerd.service` |
| `session_logger` | the per-drive jsonl recorder (`sometimes`) | wifi-loss | 25 % / 90 MB | `ros/goto.sh`, `ros/teleop.sh` |
| `bag_record` | `ros2 bag record` writing one run's MCAP bag, `PEPIN_RECORDER=bag` only (`sometimes`, an estimate) | wifi-loss | 30 % / 120 MB | `pepin_bringup.bag_recorder` (a subprocess per run) |
| `bag_recorder` | the node that starts and stops it and subscribes to nothing, `PEPIN_RECORDER=bag` only (`sometimes`, an estimate) | wifi-loss | 8 % / 60 MB | `pepin-ros.service` -> `nav.launch.py` |
| `link_watch` | stops the cart when the laptop half goes away, `side=board` only (`sometimes`) | real-time, wifi-loss | 15 % / 90 MB | `pepin-ros.service` -> `nav.launch.py` |
| `reap_ros2_cli` | kills ros2 CLI tools older than 90 s, once a minute (`sometimes`) | real-time | 5 % / 10 MB | `pepin-reap.timer` |
| `foxglove_bridge` | **not expected**: the websocket is being removed, the laptop reads the board through zenoh | - | 0 % | was a component of `sensors_container` |

### Laser odometry on the board

The board keeps odometry and no map localiser (2026-09-22), so the witness that the room really
moves past the cart has to reach the filter as a measurement of its own: `rf2o_laser_odometry`
matches each LD19 scan against the one before it — scan to scan, no map, no graph — and publishes
`/odom_laser` at 10 Hz, which the EKF fuses as `odom3`, **twist only** (`vx`, `vyaw`;
`ros/params/ekf.yaml` says why a pose would fight the wheels). It publishes no transform: the
filter owns `odom -> base_link`. `ros/feature.sh laser_odom on|off` is the switch, on by default,
and off is the stack of the day before — the wheels, the gyro and the camera untouched. There is
no Jazzy apt package, so the node is built into the board image from a pinned commit of the
MAPIRlab ROS 2 port with a three-hunk patch (`ros/patches/rf2o-base-twist.patch`): upstream
publishes the twist in the *laser's* frame, an empty covariance, and two `INFO` lines per scan.
Its budget is an **estimate** until the first census after it lands, and its numbers are sized to
be quiet: the drive that measures them pushes the cart forward by hand (`/odom_laser`'s `vx` must
be positive and near `/odom`'s) and turns it left (`vyaw` positive and near the gyro's), because
the LD19 hangs upside down and yawed and a sign error here is the one failure that would matter.

The entries that must always run promise **327 % of 400 %** and 1377 MB of the 1.5 GB. That is
the budget, not the measurement: the same stack idling on 2026-09-14 measured 235 % and 850 MB.
The gap is the +50 % headroom every entry carries, and it is the reason a new process needs a
number before it needs a launch line — which is exactly what it caught on 2026-09-22: with the
tracker still counted as always-on, adding laser odometry promised **427 %** of four cores and
1498 MB of 1.5 GB. Four A53 cores do not hold a map localiser *and* the laser odometry that
replaces what it gave the EKF. The tracker is `sometimes` from that day (a mode, not the
default); absent it is IDLE rather than MISSING, and running it is measured as before.

## Building the image

There are two ways to produce the board's image, `pepin-ros` (tagged `:latest` and `:zenoh`,
which is what `ros/run.sh` asks for). Both read the same `ros/Dockerfile`; they differ only in
which CPU compiles it.

| | `ros/build-image.sh` (laptop) | `ros/build.sh` (board) |
|---|---|---|
| where | `docker buildx build --platform linux/arm64` on the Mac — native, no emulation | `docker build` on the Orange Pi |
| cold build | 4-5 min (262 s measured, rf2o's layer 25 s of it at `-j18`) | 16-30 min (rf2o alone compiles at `-j1`) |
| rebuild after a `pepin_bringup` change | seconds (the layers below are cached) | minutes |
| needs the robot | no | yes, and the stack is stopped for the whole build |
| gets the image to the board | `--ship`: `docker save` piped into the board's `docker load` | it is already there |

Use the laptop path by default. Keep the board path for the case where the laptop cannot build
(no Docker, no disk) or where only the board can reach the network the build needs.

```bash
ros/build-image.sh            # build only: linux/arm64 image in the laptop's Docker, board untouched
ros/build-image.sh --ship     # build, then load it on the board (the stack must be stopped first)
ros/build-image.sh --ship-only  # ship the image already built here
```

The image is about 1.0 GB on the wire and travels uncompressed: Docker Desktop's containerd
image store already saves compressed layers, so a compressor takes 0.7 % off the tarball and
costs the board a decompression (`PEPIN_SHIP_COMPRESS=zstd` turns one on for a daemon whose
image store writes plain tars).

Neither step restarts the robot: `docker load` under a driving cart is refused unless `--force`,
and the restart (`ros/restart.sh board`) stays a separate, deliberate command. The image carries
only what is baked in — code, params and maps are mounted from `/root/pepin-ros` and travel with
`ros/sync.sh` as before.

The rf2o layer's `-j1` is the board's RAM limit, not the package's: it is the `RF2O_JOBS` build
argument, default 1 so an on-board `docker build` is safe without being told, and
`ros/build-image.sh` passes the laptop's core count. The rest of the caching is Docker's own —
an unchanged layer is reused, so a rebuild after a change to `pepin_bringup` does not recompile
rf2o. `PEPIN_BUILD_CACHE=<dir>` additionally exports the build cache to a directory (this needs
a `docker-container` builder, which the script creates as `pepin-arm64`).

## The XFeat image

`pepin-laptop:xfeat` is the laptop image with RTAB-Map able to run `rtabmap_frame`'s
`visual_features` default: XFeat keypoints matched by LighterGlue in the visual registration.
It is `pepin-laptop:zenoh` plus RTAB-Map 0.22.1 and the rtabmap_ros packages that link it,
rebuilt from their pinned upstream commits with Python (the apt build has none, and
`Vis/FeatureType 15` / `Vis/CorNNType 6` are compiled out of it), XFeat and LighterGlue at a pinned
commit with their weights inside, and the two adapters RTAB-Map loads by path in `/opt/xfeat`
(`ros/xfeat/`). Everything runs on the CPU; a registration costs 0.67-0.70 s inside RTAB-Map.

**Build.** `ros/laptop-build.sh xfeat` — about an hour (`ros/Dockerfile.xfeat`,
`ros/xfeat/build_rtabmap.sh`). The Docker VM also runs the live stack, and on 2026-09-24 nine
parallel compiles filled its 16 GB and the OOM killer took the stack's processes instead of the
compilers. So the build runs 2 jobs per phase (`PEPIN_CORE_JOBS`, `PEPIN_ROS_JOBS`) and is cancelled
once more than 75 % of the VM's memory is in use (`PEPIN_BUILD_MAX_USED_PCT`); `PEPIN_XFEAT_BASE`
names another base. It fails, rather than producing a different RTAB-Map, when the rebuilt core
lacks an optional library the apt one had (GTSAM, g2o, libpointmatcher and octomap decide defaults
the launch table does not name, such as `Optimizer/Strategy` and `Icp/Strategy`), or when a second
core is left anywhere under `/opt/ros/jazzy`.

**Run.** `ros/laptop.sh vslam` (and so `ros/restart.sh laptop`) starts `pepin-vslam` on
`pepin-laptop:xfeat` whenever that image exists and was built on the current laptop image. Its
layers must begin with that image's layers, so a laptop image rebuilt since is not traded for an
older one. Otherwise the usual image runs, `visual_features` falls back to ORB, and the script says
so on stderr. Restart check 2.12 fails a camera half without the adapters.

| setting | image | visual registration |
|---|---|---|
| unset | `pepin-laptop:xfeat` if built on the current base, else the base with a stderr line | xfeat, else ORB |
| `PEPIN_XFEAT=0` | the base (`pepin-laptop:zenoh`), the rollback | ORB, and check 2.12 passes |
| `PEPIN_XFEAT=1` | `pepin-laptop:xfeat`, or the command refuses | xfeat |
| `PEPIN_IMAGE=<tag>` | that tag, whatever the above | xfeat if the tag carries `/opt/xfeat` |

Inside the running node the switch is live (`visual_features orb`, `pnp_reproj_px`), with no
restart. The image tags kept for a rollback are `pepin-laptop:zenoh` (stock RTAB-Map) and
`pepin-laptop:xfeat-refblas`. `xfeat-refblas` is the xfeat image from before the adapters imported
torch with `RTLD_DEEPBIND`: there torch's matrix products bind to the reference BLAS RTAB-Map loads
first, and a registration costs 1.70 s instead of 0.67 s (`PEPIN_IMAGE=pepin-laptop:xfeat-refblas
ros/laptop.sh vslam`).

**Known broken in this image.** The GUI parts were not rebuilt, because this build leaves out Qt:
`rtabmap_viz`, `librtabmap_rviz_plugins.so`, and the `rtabmap` / `rtabmap-databaseViewer`
desktop tools. They still link apt's `librtabmap_gui` 0.22.1, which was compiled against the core's
old class layout (Python support adds a member to `rtabmap::Rtabmap`), so they must not be run
from this image. Nothing in `vslam.launch.py` starts them; Foxglove is the window. Open a database
with the tools of `pepin-laptop:zenoh`.

## The model services

Docker on macOS has no GPU, so a learned model in a container runs on the Docker VM's CPU. The
models run on the laptop instead, as two host processes — two failure domains, so one crashing or
hanging never takes the other with it — which the containers reach as
`http://host.docker.internal:<port>` (bound to 127.0.0.1, nothing open on the LAN):

| job | process | port | models (device from `config/models.json`) |
|---|---|---|---|
| `depth` | `pepin.depth_service` | 8790 | the depth network, RAFT-Stereo (as `ros/depth_host.sh`) |
| `localization` | `pepin.localization_service` | 8791 | XFeat (`/xfeat`, cpu), LighterGlue (`/match`, cpu), BoQ-DINOv2 place descriptors (`/place`, mps) |

Both are launchd jobs run from the repository's uv environment, loaded ON DEMAND: their plists
live in `~/Library/Application Support/pepin/launchd` (`PEPIN_LAUNCHD_DIR`), not in
`~/Library/LaunchAgents`, so nothing loads them at login — together they hold about 2 GB of
models. While a job is loaded launchd starts it and restarts it when it dies (KeepAlive,
`ThrottleInterval` 30 s, `ProcessType Standard`); `ros/laptop.sh vslam` starts both (as it started
the depth host before) and `ros/laptop.sh stop` stops them.

| command | what it does |
|---|---|
| `ros/models.sh install [depth\|localization\|all]` | write the job's plist and start it; waits for `/health` |
| `ros/models.sh start\|stop [...]` | load the job and wait for `/health` / unload it: the process ends and stays down until the next start |
| `ros/models.sh restart [...]` | kill it and let launchd start it again |
| `ros/models.sh status [...]` | launchd's word and each `/health` as one line: every model's tag (its weights' hash), device, requests and ms |
| `ros/models.sh installed depth\|localization` | exit 0 when the job is installed (what `laptop.sh` and `depth_host.sh` ask) |
| `ros/models.sh logs [depth\|localization]` | follow `logs/models_<job>.out` (moved to `.out.1` past 10 MB at a start) |
| `ros/models.sh plist [...]` | print the plist `install` would write |
| `ros/models.sh fetch-xfeat` | clone XFeat at the commit the image pins into `models/accelerated_features` |
| `ros/models.sh uninstall [...]` | stop and remove |

The localization job takes an XFeat checkout only at the commit `ros/Dockerfile.xfeat` pins
(`ARG XFEAT_SHA`, read from its git HEAD or a copy's `COMMIT` file), because `auto` registration
mixes the service's features with the image's local fallback (`PEPIN_XFEAT_UNPINNED=1` takes
another anyway). The service checks its port before it loads a model, so a job whose port is taken
exits in a second instead of reloading torch every restart. `ros/depth_host.sh` keeps working for
a depth host not installed here, and hands an installed one to `ros/models.sh start|stop depth`.
A model that fails to build does not take its process down: its endpoint answers 500 with the
reason and `/health` names it (`failed: ...`); restart check 2.13 reads it. A place vector that is
not a finite unit vector is a 500 too (one NaN aborts RTAB-Map). A client whose kept-alive
connection the service dropped (a restart, a sleep) is sent once more on a fresh connection before
anything counts as down.

**Measured through the service** (2026-09-24, 800x600 pictures, beside the live stack, whose depth
host keeps the GPU 80 % busy with RAFT-Stereo; `scratch/models/endpoint_bench.py`,
`load_probe.py`): XFeat 46 ms median, 54 ms p90 on the CPU against 41-81 ms and up to 241 ms p90 on
MPS — so the CPU — and 1-2 ms for a picture seen before (its LRU: RTAB-Map re-extracts the same
stored node pictures); LighterGlue 109-129 ms on the CPU against 518 ms on MPS; BoQ-DINOv2 57 ms on
MPS against 167 ms on the CPU. The container's round trip adds about 2 ms for a picture and 11 ms
for a LighterGlue pair. A registration inside RTAB-Map (the replay of run 0466, words at 0.05):
454 ms through the service against 1879 ms in RTAB-Map's own process on the VM's CPU, the update
492 ms against 1900 ms median; against the NIGHT branch's in-process adapters under the same
conditions, back to back (`scratch/models/backend_control.sh`, `.py`), 571 ms against 1896 and
2326 ms in the night runs before and after it, and 2428 ms for this branch's local fallback —
within that drift, so the fallback is the night's path, and the 0.67 s of 2026-09-23 was a quieter
laptop. One run, 18 localised updates each: a speed, not a recognition result. While RTAB-Map drove it (2.3 XFeat, 1.1 LighterGlue and 0.4 place
requests a second) the service used a core at its p90 and nothing at its median. **This laptop
swaps** (15 of 15 GB of swap in use on 2026-09-24): an idle model's pages go first, and a GPU
model's first answer after 2 s idle takes 200-460 ms, 1.5 s after minutes
(`scratch/models/gpu_idle_probe.py`); the clients' timeouts are sized for it and a single slow
answer does not back a client off.

**RTAB-Map's XFeat adapters** (`ros/xfeat`) are the service's clients: `rtabmap_frame`'s
`registration_backend` (`service`, `local`, `auto`) is written to `/tmp/pepin/registration.json`
in the container, which the adapters read on every call; `auto` falls back to the old in-process
computation (torch loaded on its first use) and leaves a dead service alone for 10 s at a time.
`ros/laptop.sh vslam` mounts the checkout's two adapter files over the xfeat image's, so an
adapter change needs a vslam restart and no rebuild (`PEPIN_ADAPTERS_MOUNT=0`: the image's).

**Place descriptors.** RTAB-Map can find which node a picture is by the dot product of learned
global descriptors instead of its ORB words (`rtabmap_frame`'s `place_recognition`), and it
compares two nodes' descriptors only when both carry one: a node with one against a node without
ABORTS it (`Signature.cpp:252`). So every node carries exactly one:

1. `sensor_pack` attaches one to every snapshot (`global_descriptor`, `auto`: only on an RTAB-Map
   built with the patch below — the unpatched core aborts even under the words, because mapping's
   rehearsal compares nodes too: on the replay of 0466 mapping it died on the third update with
   descriptors and ran 17 clean without, and the patched core ran the same 17 with descriptors
   clean, `scratch/models/unpatched_mapping_control.sh`): the service's `/place` vector for a camera snapshot, the null
   descriptor (zeros, which RTAB-Map scores as neither like nor unlike anything) for a lidar-only
   one, a picture the service did not describe in `place_timeout_s`, or any failure on the way
   (a worker that raised is never a worker that stopped);
2. `ros/tools/place_backfill.py ros/maps/rtabmap.db` writes one into every node the database
   already holds (the service up, the vslam stopped; a timestamped backup first, one transaction,
   a re-run replaces, a vector that is not a finite unit vector refused), and `--check` prints the
   census (of a copy, when a container holds the file);
3. `vslam.launch.py` takes that census itself on EVERY start, of the file RTAB-Map is given and
   before RTAB-Map opens it, and hands it to `rtabmap_frame` as `PEPIN_PLACE_CENSUS`;
   `rtabmap_frame` sends the descriptor likelihood only when the census, the snapshots and the
   RTAB-Map build all say nothing can abort it, and only while the recent camera snapshots were
   described (`descriptor_null_share`: a null query scores every node alike, so a service that is
   down hands the places back to the words); the words otherwise, with the reason in its report
   line. `place_recognition` stays `words` by default until a drive has shown the descriptor.
4. With descriptors on board the launch raises `Mem/RehearsalSimilarity` from 0.30 to 0.92: mapping's
   rehearsal compares consecutive nodes by descriptor whenever both carry one, on a scale where
   no two BoQ vectors score under ~0.57 and a null one scores 0.5; 0.92 calls as many of 121
   consecutive daylight pairs the same place as the words did at 0.30
   (`scratch/models/rehearsal_calibration.py`).

A known limit: RTAB-Map scores places only for a frame with ORB words (`Rtabmap.cpp:1971`), so a
frame too dark for a single GFTT corner is not recognised by descriptor either; the descriptor
replaces the words' score, not the words.

RTAB-Map 0.22.1 itself drops a node's descriptor the first time it registers against it (it
reloads the node's data from the database without the descriptor table, `Memory.cpp:2896-2909`)
and then aborts on the next descriptor comparison: `ros/patches/rtabmap-keep-global-descriptors.patch`
keeps them, and `ros/laptop-build.sh xfeat` applies it (`ros/xfeat/patch_rtabmap.sh`, a layer of
its own after the long builds: minutes, not the hour) and leaves the marker
`/opt/rtabmap_patches/keep-global-descriptors` that `rtabmap_frame` requires before it sends the
descriptor likelihood. The descriptor's scores are dense, so it travels with
`Rtabmap/VirtualPlaceLikelihoodRatio 1` (RTAB-Map's z-score normalisation).

**Measured in RTAB-Map itself** (2026-09-24, the evening runs replayed against the backfilled
daylight database, xfeat, 2 px, proximity on, a lidar-only snapshot every fourth frame;
`scratch/models/matrix_report.py`), camera updates localised and, where the lidar holds the truth,
judged right within 0.30 m / 10 deg:

| place recognition | run 0457 | run 0460 | camera-only 0466 | highest hypothesis |
|---|---|---|---|---|
| words, `LoopThr` 0.11 (stock) | 0 of 23 | 0 of 17 | 0 of 55 | 0.05-0.09 |
| words, `LoopThr` 0.05 (`visual_confirm aggressive`) | 9 (9 of 9 right) | 3 (0 of 1) | 53 | 0.05-0.09 |
| descriptor, ratio 1, `LoopThr` 0.11 | 16 (15 of 15) | 14 (11 of 11) | 54 | median 0.42-0.88 |
| descriptor, ratio 1, `LoopThr` 0.05 | 17 (16 of 16) | 14 (9 of 11) | 54 | median 0.42-0.88 |
| descriptor, ratio 0 | 0 | 0 | 0 | 0.01 |

With the descriptor in force `visual_confirm rtabmap` is the pairing: its hypotheses reach 0.11
by themselves, and aggressive's 0.05 let two wrong ones through on 0460. The unpatched core
aborted on the first update after a registration, exactly as read in the source.

## Build and run (on the board)

```bash
# from the laptop: copy ros/ to the board and build the image there (15-30 min the first time)
rsync -a --delete ros/ root@pepin.local:/root/pepin-ros/
ssh root@pepin.local 'cd /root/pepin-ros && docker build -t pepin-ros .'
# on the board: sensors + bridges (no Foxglove bridge here: the laptop serves it)
ssh root@pepin.local '/root/pepin-ros/run.sh ros2 launch pepin_bringup robot.launch.py'
# on the board, second terminal: navigation on a saved map
ssh root@pepin.local '/root/pepin-ros/run.sh ros2 launch pepin_bringup nav.launch.py map:=/maps/lap3.yaml'
```

Laptop: install Foxglove (`brew install --cask foxglove`), and let `ros/foxglove.sh` do the rest.

| command | what it does |
|---|---|
| `ros/foxglove.sh check` | one `PASS`/`FAIL` line each: `pepin-vslam` is up, port 8765 accepts connections, the websocket handshake succeeds, `serverInfo` and the channel count arrive, every topic the layout draws is advertised, and how many times the bridge has died since the container started |
| `ros/foxglove.sh reopen` | tells the running desktop app to reconnect (`foxglove://open?ds=foxglove-websocket&ds.url=…`), waiting for the port first; prints the link instead when the app is not running |
| `ros/foxglove.sh url` | prints that link |

It is wired in, so nobody has to remember it: `ros/restart.sh laptop|both` runs `check` as check
2.9 and `reopen` as its last act, and `ros/laptop.sh vslam` reopens the app after the container is
recreated (`PEPIN_FOXGLOVE_REOPEN=0` leaves the app alone and prints the link).

**Why a reconnect is needed at all.** A Foxglove client is bound to ONE bridge process: channel
ids are that process's own numbering and start again at 1 when it restarts. Recreating the
container (`ros/laptop.sh vslam`) or restarting the launch kills the bridge, and the app keeps
showing the dead socket's panels — empty — until a client re-attaches. The second, milder
emptiness is a channel *withdrawn* while the socket lives: the bridge advertises a topic while a
publisher exists and removes the channel when the last one goes, so every repair of the board's
routes takes `/global_costmap/costmap`, `/amcl_path`, `/tof/*` and friends out of the panel and
puts them back under a new id a second later. No bridge parameter can hold those open.

The bridge is the LAPTOP's (`ros/laptop.sh vslam`), which sees the board's topics through the
zenoh bridge; the board's own bridge is off since 2026-09-14 (it cost a second serialisation of
every topic on four A53 cores) and comes back with `robot.launch.py foxglove:=true`, on
`ws://pepin.local:8765`. Layouts live in `ros/foxglove/` (`pepin_slam.json` is the driving one);
send a goal with the "Publish" panel on `/goal_pose` (`geometry_msgs/PoseStamped`, frame `map`).

## At boot

`board/pepin-ros.service` starts the sensors container (`robot.launch.py`) after the base and ToF
servers; Nav2 (`nav.launch.py`) is started on demand inside it. Build or rebuild the image with
`ros/build-image.sh --ship` on the laptop, or `ros/build.sh` on the board itself (see
**Building the image**).

## Bring-up checklist (in this order, each step visible in Foxglove)

1. `robot.launch.py` alone: `/scan` at ~10 Hz (the hull box filter's output), `/odom` at 20 Hz, TF `odom -> base_link -> laser`.
   Push the cart forward by hand: `/odom` x grows. Turn it left: theta grows.
2. Laser orientation: a wall in front of the cart must draw at +x in `base_link`. Our lidar
   is mounted upside down and the LD19 counts angles clockwise; the static transform (roll pi,
   yaw -87.5 degrees) is read from `config/lidar.json` by the board's launch and by the
   laptop's camera node alike — there is no launch argument for it. If the scan comes out
   mirrored left/right, fix `roll_deg` in that file and `ros/sync.sh --restart`.
3. `nav.launch.py`: AMCL converges on the map after a few metres of teleop (or set the initial
   pose from Foxglove); then a goal.
4. Memory: `free -m` on the board while navigating; the container must stay under ~700 MB.

## Redundancy demo

Both sensors see the room, and either one alone can feed the costmaps, switched while the robot
runs. `ros/sensor.sh` is one command per sensor: what writes into the costmaps (`lidar_layer`,
`camera_layer`, `contact_layer`, on the local **and** the global costmap). Localisation is
RTAB-Map's either way (its snapshots carry whatever sensor is alive: `sensor_pack`'s `sources`).
The board tracker's half of this switch — its `sources` flag — is on the tag
`alt/tracker-2026-09-22`.

| mode | command | costmap layers |
| --- | --- | --- |
| fused | `ros/sensor.sh lidar on` + `ros/sensor.sh camera on` | `lidar_layer`, `camera_layer`, `contact_layer` |
| lidar only | `ros/sensor.sh camera off` | `lidar_layer` |
| camera only | `ros/sensor.sh lidar off` (`--hard` to stop the driver) | `camera_layer`, `contact_layer` |

The camera is one sensor read twice from the same frames: `depth_scan` is the band 8 cm-1.3 m
above the floor (table tops, seats, a hand) and `contact_scan` is where the floor ends (chair
feet, a plinth) — so `camera on` moves two costmap layers at once.

**Inside the camera layer, a frame clears and the MODEL marks** (2026-09-21). `/depth_scan` is a
single frame, and a single stereo frame is an eyewitness of what is open, not evidence that
something is there: SGBM on the herringbone parquet lifts floor pixels to 0.15-0.24 m in small
flickering blobs, and the first stereo drive left the costmap with 100-300 lethal cells the lidar
never saw, 115 "collision ahead" a minute and 44 recoveries. So the layer's `depth_scan` source
now only clears, and a second source marks: `/depth_marks`, the fused volume's own surface — the
one `/fusion/surface` draws, at `depth_fusion`'s `min_weight` — sliced around the cart over the
whole turn (`pepin.volume_scan`, published at the rate the volume is integrated). `ros/flags.sh
set depth_fusion marks_source frame` relays the frame's own fan onto `/depth_marks` instead and
is the pre-2026-09-21 costmap, live, for an A/B. 

```bash
ros/sensor.sh status            # which layers are on, what each camera node last said
ros/sensor.sh camera off        # lidar only
ros/sensor.sh lidar off         # camera only: /scan still arrives, nothing reads it
ros/sensor.sh lidar off --hard  # camera only, for real: the driver is deactivated and /scan stops
ros/sensor.sh lidar on          # back, driver included
```

Every call is idempotent and prints only what it moved (`already so` when nothing differs). The
soft `lidar off` is an ignored scan; `--hard` is the absence of one —
`ros2 lifecycle set /ldlidar_node deactivate`, which sticks because the sensors lifecycle manager
runs with `bond_timeout: 0.0` and does not resurrect it. Only `ros/sensor.sh lidar on` brings it
back. `--hard` (and the reactivation) first asks `ros/tools/nav_goal_running.py` — one rclpy pass
over `/navigate_to_pose/_action/status` and `/navigate_through_poses/_action/status`, piped into
the board's python so nothing has to be deployed — and refuses while a goal is running, the check
`ros/tools/turn_full.py` makes before it turns the cart. An answer it could not get refuses too,
and that is why the check is a node rather than a `ros2 topic echo --once` under a `timeout`: an
action server that has had no goal since it started has nothing latched to hand over, so the echo
hangs exactly as it hangs when the CLI is too slow to discover anything, and both come back as
exit 124. Measured on the board on 2026-09-11 with Nav2 up and no goal yet sent: the status
publishers are matched 0.26 s into the pass and not one message ever arrives. A subscription can
tell "the server is up and silent" from "this pass saw nothing at all"; a `timeout` cannot, and
the loaded board that makes the query slow is the one state in which a goal really is running.
The whole pass costs ~7.6 s. Nothing here restarts a container or writes a velocity.

Expect it to be slow: a `ros2` CLI call is a Python node that must start and discover, about 10 s
on the board under load. The script therefore reads one whole `ros2 param dump` per costmap
rather than a `ros2 param get` per layer, writes only what differs, and takes freshness from the
nodes' own report lines in `docker logs` instead of subscribing to anything.

### What to watch

Open `ros/foxglove/pepin_nav.json` in Foxglove Studio:

- the 3D panel's `/scan`, `/depth_scan` (what the camera clears with) and `/depth_marks` (what it
  marks with: the volume's surface, yellow) beside `/local_costmap/costmap` and
  `/global_costmap/costmap` — switching a layer changes the grid within a costmap cycle, and the
  marks that remain tell you which sensor drew them;
- the Log panel (filtered to `controller_server`, `planner_server`, ...).

The report lines say the same in words, every 30 s, and `ros/sensor.sh status` prints them:

- `depth_stream` in the laptop's SLAM container: `depth: 3.1 frames/s published ...`
- `contact_scan` in the same container: `contact: 2.9 scans/s published ...`

## Frames and conventions

`base_link` sits between the drive-wheel contact points (our robot frame): x forward, y left.
Footprint (meters, `base_link`): front 0.0625, rear 0.30, half-width 0.275 — from `config/base.json`.

`odom -> base_link` is planar and stays planar (`ros/params/ekf.yaml`, `two_d_mode`): Nav2 and the
tracker want it that way. The few degrees the body leans over a slipper or a threshold are a
separate number, estimated once per node by `pepin.lean` from `/imu/data_raw` (the accelerometer
for the slow truth, the gyro for the fast part, and the gyro's own zero offset learned from the
disagreement gravity keeps voting for — the bridge averages that offset once at boot and never
again, and 0.05 deg/s of drift integrated bare reads as half a degree of slope that nothing else
in the line can see) and composed on base_link's side of that planar pose by
`pepin.frame_pose.FramePoser` behind each node's `imu_lean` flag. The flag means one thing in
every node that carries it: on, the lean follows the gyro as well as the accelerometer and is
applied wherever that node places a frame; off, the accelerometer alone, estimated and reported
but not applied. Roll is positive right side down, pitch positive nose down. Every node that
estimates it prints it in its report line (`lean +0.3/-1.8 deg q0.94 bias 0.05 deg/s`) whether
the flag is on or off, so it can be watched before it is believed. The `q` in that line is the
share of the lean gravity itself voted for, and it is also a gate: below `lean_min_quality` the
lean is treated as unknown and the measurement is placed level, because a gyro whose zero has
drifted reports a tip nobody made (0.2 deg/s of bias reads as 3 degrees on a level floor) and it
arrives with a quality near zero, while a real tip over a threshold keeps `q` at 1.00. Both the
zero of the roll and pitch and the gyro's starting bias come from `config/imu.json`'s `level`
block — what the chip read over 30 s at rest on a floor that is level (2026-09-12): -0.39 deg of
roll, +0.29 of pitch and [-0.001, -0.028, +0.074] deg/s of rate. They are subtracted, so a still
cart on level ground reports no lean and the filter does not spend the first minute of every run
learning an offset that was measured once. Without the block nothing is subtracted, which is what
every run before it did.
