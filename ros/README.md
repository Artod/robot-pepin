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
Foxglove Studio  <-- ws 8765 -->  docker: foxglove_bridge, ldlidar_node -> laser_filters box filter (/scan), base_bridge
                                          (/odom, tf), tof_bridge, Nav2 (amcl, costmaps,
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

## Iterate without rebuilding

`ros/sync.sh` rsyncs `ros/` and `src/pepin` to the board and restarts the sensors container;
`ros/nav.sh [MAP]` starts Nav2 inside it. The container mounts the code from the host (see
`run.sh`), so Python nodes, launch files, params, maps and tools change in ~20 s. One node
changes in seconds: `ros/sync.sh --no-restart && ros/thin.sh kick relocalizer` ends that process
with SIGINT and the launch respawns it from the synced sources (`ros/laptop.sh kick depth_fusion`
does the same in the laptop's containers; `ros/laptop.sh kick` and `ros/thin.sh kick` without a
name list what each can reach). The laptop's SLAM container is its own: `ros/laptop.sh vslam`
restarts it with the RTAB-Map database kept, `ros/laptop.sh vslam --fresh` deletes the database
first and starts an empty map (in SLAM mode the session starts empty anyway: see below). Only a Dockerfile change (apt packages, the C++
driver) needs `ros/build.sh`, which stops the container first and uses BuildKit's apt cache.

## The zenoh bridge

The board and the laptop are two ROS graphs joined by `zenoh-bridge-ros2dds` 1.7.0, a router on
each side over one TCP link (board `-l tcp/0.0.0.0:7447`, laptop `-e tcp/<board>:7447`). Each
bridge gets a one-way allow-list generated from `pepin.deployment.bridge_config(side, mode)` into
`ros/zenoh-bridge-<side>[-<mode>].json`, and a test keeps the files equal to the generator: a
topic allowed as a publisher on BOTH sides loops until nothing crosses.

**The admin space is network-wide.** Either bridge answers `@/*/ros2/route/**`,
`@/*/ros2/dds/**`, `@/*/ros2/node/**` and `@/*/ros2/config` for BOTH bridges — the two replies
were byte-for-byte the same size on 2026-09-13 — with the owner in the key (`@/<zid>/...`). So a
`topic/sub/depth_scan` on the laptop's admin is the BOARD's route seen from here, not a second
crossing, and `@/*/ros2/dds/**` is `ros2 topic info -v` for both machines from one curl. Read it
all in one command with `scratch/bridge_flow.py`.

**DDS does not cross the WiFi.** Measured 2026-09-13: the board's bridge saw 11 DDS participants,
the laptop's 10, and the two sets did not intersect. The laptop's containers are on a docker
bridge network inside Docker Desktop's VM and their RTPS discovery never reaches the LAN; the
board's (`--network host`) never reaches the Mac. Every crossing is the bridge's. `-d 7` and
`ros_localhost_only: false` are therefore left as they are; the hardening for the day a ROS node
runs natively on the Mac, or a second robot joins domain 7, is `ROS_AUTOMATIC_DISCOVERY_RANGE=
LOCALHOST` in the board's two containers (both are `--network host`, so they still see each
other) — not enabled, because nothing has ever been measured crossing.

**A route's QoS is a race, and the race is the recurring failure.** A route is keyed by topic
name alone and is created by whichever declaration arrives first — a local ROS endpoint, or the
far bridge's announcement of its own. Its DDS endpoint takes that declaration's QoS, and
`routes_mgr` never revises it when the other side shows up (1.7.0 `route_publisher.rs`: "those
are either the QoS announced by a remote bridge on a Reader discovery, either the QoS adapted
from a local discovered Writer"). Two sides that disagree therefore get an endpoint that matches
one of them and starves the other, which is what "the route exists on both admins, the publisher
is there, nothing arrives" has meant every time. It is not rare: in the bring-up of 2026-09-13
22:43 the laptop bridge created 47 routes and 24 of them came from the far bridge's
announcement, including every route re-created after the board's bridge restarted while the
laptop's nodes stayed up (`scratch/bridge_state_184752_laptop_bridge.log`).

So: **a bridged topic carries the same reliability and depth on both sides**, and the pairs that
must agree live in `pepin.deployment.BRIDGED_QOS`, which `node_kit.bridged_qos_profile` hands to
every subscription that reads one. The first entry is `/imu/data_raw`: the board's C++ bridge
writes it RELIABLE, KEEP_LAST 10, the three laptop nodes read it through `node_kit.LeanFeed`, and
while they asked for the sensor-data default (BEST_EFFORT, 5) the laptop saw 10-11 Hz of the
board's 48 whenever the announcement won the race. The DDS legs are inside one host — the
wireless hop is zenoh's, not DDS's — so RELIABLE there costs a memcpy, not a retransmission.

**The watch verifies flow, not route counts** (`pepin_bringup.bridge_watch`, flags `flow_watch`,
`flow_silence_s`, `bridge_restart`). It subscribes to every topic both bridges agree should
arrive on this side — the far side has a publisher with a real node behind it, some node here is
waiting — counts the messages, and prints one line a minute with the rate of each. It never
subscribes to a topic nobody here reads: that subscription would create the route, pin its QoS
to the watch's own, and pay for a topic to cross that nobody wants. When a topic that should
flow carries nothing for `flow_silence_s`, the repair ladder is:

1. **Restart the laptop's bridge container alone** (`POST /containers/pepin-zenoh/restart` on the
   mounted docker socket). Every route is re-created; `pepin-vslam` keeps running, so the fusion
   model and RTAB-Map's database survive. There is nothing gentler: the REST admin is read-only
   (`permissions { read: true, write: false }`), so 1.7.0 has no config reload and no way to drop
   one route.
2. **Restart this half** — the old action, kept: the watch exits with code 3, the launch shuts
   down, the container's restart policy brings it back with fresh subscriptions. This is also
   what still happens when the board's bridge changes its zenoh id, and when the docker socket is
   not mounted.

Deploying a change to the bridge: `ros/sync.sh`, then `ros/thin.sh on|vision|slam` (the board's
bridge unit restarts with its config), then `ros/laptop.sh start` — in that order, because
`laptop.sh` waits for the board's bridge to answer and settles it before the laptop's containers
start their subscriptions.

## Online SLAM

The robot is put somewhere it has never been, builds **one** map while it drives, and navigates
in it. RTAB-Map on the laptop is that map: the camera names the places (appearance-based loop
closure), the lidar gives the geometry, and both go into one graph and one occupancy grid — there
is no second map anywhere. The board keeps the reflexes and the wheels; it serves no saved map and
runs no scan-matching tracker, because there is nothing yet to match against.

Who owns what:

| | known map (`ros/thin.sh vision`) | online SLAM (`ros/thin.sh slam`) |
| --- | --- | --- |
| `/map` | the board's `map_server`, from a file | the laptop's RTAB-Map, growing |
| `map -> odom` | the board's `relocalizer` | the board's `slam_frame`, from the laptop |
| RTAB-Map's odometry | the tracker's pose (`map`) | the EKF's `odom` |
| RTAB-Map's map frame | `rtabmap`, beside the real one | `map` — it *is* the real one |
| its database | kept (`ros/maps/rtabmap.db`) | empty each session (`rtabmap_slam.db`) |
| what a goal is judged on | the tracker's fit ≥ 0.50 | `map -> base_link` younger than 1 s |

(In SLAM mode `/map` can also come from the fused volume — `ros/laptop.sh vslam --world-map`,
see [The world map](#the-world-map). Still exactly one publisher: the mode's owner and the
launch's `world_map` must both say so, and the launch is what sets `map_source` accordingly.)

`/tf` crosses the bridge board → laptop only (a topic allowed as a publisher on both sides loops
until nothing crosses at all), so the correction RTAB-Map computes travels the other way as a
message on `/map_odom` and becomes a transform on the board, where Nav2 and the behaviours look it
up. One publisher of that edge, in either mode.

### A session

```bash
ros/sync.sh                      # the board gets the new launch, the bridge config and slam_frame
ros/thin.sh slam                 # board: bridge on with the slam allow-list, Nav2 on, no map server
ros/laptop.sh                    # laptop bridge; reads the board's mode and records it (ros/.mode)
ros/laptop.sh vslam              # RTAB-Map as the SLAM (the recorded mode); --slam forces it
ros/teleop.sh                    # or drive by goal, below — the map grows as the cart moves
ros/map.sh save flat3_slam       # freeze the grid into ros/maps/flat3_slam.{yaml,pgm}
```

Goals work with no places book: `ros/go.sh -1.0 0.3 90` drives to map coordinates (recorded like
any other drive), and a click in Foxglove (Publish → `/goal_pose`, frame `map`) does the same
without a tape. There is no tracker here to ask "am I localised", so **the goal server takes the
cart's pose from `map -> base_link`** — the edge `slam_frame` broadcasts from RTAB-Map's
correction — and accepts a goal while that edge is younger than 1 second; older, or missing, the
goal is refused with which of the two it was, and no whole-map search is attempted (there is
nothing to search). `ros/go.sh where` says `"pose": "tf"` where that is what answered, and
`ros/go.sh mark` fills the session's own book (`/maps/slam.places.yaml` on the board) on the same
evidence, writing no `fit` at all rather than a `0.00`. The saved map's places are not offered:
they are coordinates in a frame this new map does not share. The old behaviour — only ever ask
the tracker — is `ros/flags.sh set goal_server tf_pose false`.

**A fresh edge is no evidence that the laptop is still there.** `slam_frame` re-broadcasts the
LAST correction at 10 Hz with a fresh stamp, so with the laptop shut down `map -> base_link` is
still 0.1 s old: the gate would pass, and Nav2 — whose costmaps read that same edge against a
0.3 s tolerance — would not abort either, so the cart would follow its plan by dead reckoning
across a map that stopped growing. The goal server therefore also listens to the correction
itself (`/map_odom`, published at 10 Hz whether or not the graph moved): a goal is **refused**
when it has been silent for 2 seconds or has never arrived, and a running drive is **cancelled**
when it falls silent under it — this mode's answer to the blind-drive watch, which has no fit to
read here. `ros/go.sh where` prints `correction_s`, the age of the last one, wherever one has
ever landed; the board logs the silence when it starts (`nothing on /map_odom for ... s`) and
keeps broadcasting the edge all the same, because Nav2 there must not lose its global frame to a
wireless hiccup. Off: `ros/flags.sh set goal_server correction_watch false`.

Two things this mode needs that are **not** the operator's to remember:

- **The camera's costmap layers stay off for driving.** They marked within 2 cm of the hull and
  stalled two drives on 2026-09-13 (the contact ring at 1.2–1.5 m, the depth band beside the
  hull); `ros/sensor.sh camera off` before a goal, and `ros/sensor.sh status` to see where they
  stand. In this mode `ros/sensor.sh` prints `tracker: none in slam mode` for its other half and
  switches the costmaps alone — that is the whole switch here, not a failure.
- **`fit_gate` is handled by the launch.** The fusion fuses only while `/localization_fit` is
  healthy, and in SLAM mode nobody publishes that topic — with the gate on, the first session
  fused 0 frames until it was switched off by hand. `vslam.launch.py` now passes
  `fit_gate:=false` in SLAM mode (and `map_source:=volume` where the volume owns `/map`). Both
  remain live flags: `ros/flags.sh set depth_fusion fit_gate true` puts the gate back.

Watch the map grow with the `ros/foxglove/pepin_slam.json` layout at `ws://localhost:8765`: `/map` under the fused surface, the graph's path, the head camera.

To go back to driving a saved map: `ros/thin.sh vision` (which leaves SLAM mode), then
`ros/mode.sh nav /maps/flat3_slam.yaml`.

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

### Not yet verified on the robot

- RTAB-Map's own `map -> odom` quality against the wheels-plus-gyro EKF: on a known map its
  "odometry" was the tracker's centimetre-accurate pose, and the graph rejected closures as
  inconsistent when it was fed raw odometry (2026-09-10). Whether ICP on the scans plus
  `RGBD/NeighborLinkRefining` absorbs that drift is the first thing to look at.
- Nav2 on a growing map: the global costmap's static layer resizes with every new `/map`, and the
  local costmap works in the `map` frame, which jumps at a loop closure. `track_unknown_space:
  false` means unmapped ground is planned through as free floor — which is what exploration needs
  and what a wrong correction would exploit.
- The correction's path (laptop `mapGraph` → `/map_odom` → the board's `slam_frame`) adds a
  wireless hop before the transform moves; the transform itself is re-stamped at 10 Hz on the
  board, so only the *value* is late, never the lookup — which is why the freshness of the
  lookup says nothing about the laptop, and the goal server watches the message instead.
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
- **Slices.** A horizontal band of the volume reads out as an occupancy grid: `lidar_slice()` at
  the lidar's plane (what the tracker matches and what goes out as `/map`), `camera_band_slice()`
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
- **The lidar may localise on the volume too.** With `lidar_map` on, the same lidar slice that
  would go out as `/map` also goes out as **`/map_lidar`**, a topic of its own that crosses to the
  board, and the tracker's `map_topic` flag points it there instead of at the served file —
  Nav2, the `map_server` and the owner rule above are not touched at all. The tracker adopts the
  first map on the topic it is asked for and no other, unless `map_refresh_s` says how often it
  may take a changed one: adopting rebuilds the matcher and the tracker and forgets the episode's
  evidence, and `/map_lidar` is republished every second (`pepin.mapping.MapChoice`). Two things
  to know before pointing it there: the volume must have been **seeded** from the served map (an
  unseeded live volume held 52 % of that map's walls; a seeded one IS it, cell for cell), and the
  two grids have different sizes, so the map id differs and the laptop's candidates and camera
  measurements are refused until that half moves too.
- **`/map` has exactly one owner.** Two launch decisions, both told to the node, decide whether
  it may publish: `pepin.deployment.map_owner` per bridge mode — the board's `map_server` in
  `split` and `vision`, the laptop in `slam` — and `world_map:=true`, which is what keeps
  RTAB-Map's grid on `/rtabmap/map` instead of remapping it onto `/map`. With both, and
  `map_source=volume`, the fusion node publishes the lidar slice as `/map` (transient local,
  `map_hz`); without either it refuses and names the reason in its report line, so setting
  `map_source volume` live in an ordinary SLAM run cannot put a second publisher on `/map` —
  which is the failure this whole table exists to prevent.
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
ros/flags.sh set depth_fusion lidar_map true     # the lidar slice goes out on /map_lidar too
ros/flags.sh set relocalizer map_topic map_lidar # ...and the board's tracker matches on it
ros/flags.sh set relocalizer map_refresh_s 60    # it may take a changed one once a minute
```

Offline, `WorldMap.export_pgm_yaml` writes the map_server pair every existing tool already reads
(`ros/mode.sh nav /maps/NAME.yaml`, `pepin.mapping.grid_from_pgm`), so a volume can be frozen into
a file exactly like `ros/map.sh save`.

**The next step, not built:** re-fusing after a loop closure. A TSDF cannot be un-integrated, so a
graph correction leaves the old geometry standing. The cure is to replay the frames at their
corrected poses, and the snapshot already carries the index for it — every integration's stamp,
sensor and pose — while the measurements themselves stay in the run tape, where they already live.

## The laptop's localizer: the whole-map watchdog, and the camera's own poses

Two jobs the board has no CPU for, in one node (`pepin_bringup.laptop_localizer`) that shares a
map, the board's belief and a report line: the whole-map search below, and the camera's scans
matched where they are produced — see **Redundancy demo** for that half and why it moved here on
2026-09-13.

### The whole-map watchdog

The board's tracker follows the cart in a 9 cm window around the odometry's prediction. When it
loses the world it searches the whole map for itself — FFT correlation over every shift at 40
headings, **3.7 s** of an A53 (2026-09-06) — but only after the fit has been poor for three
checks in a row, and never while the cart is moving: a teleport mid-drive is worse than a poor
fit. The laptop runs the very same search in **0.12 s** and has nothing else to do with it, so
the laptop's localizer asks the question once a second, healthy or not, and sends the
answer to the board as a *candidate*.

| | the board alone | with the watchdog |
| --- | --- | --- |
| when "where am I really?" is asked | after 3 poor checks, standing still | every second, always |
| what one answer costs | 3.7 s of the board's CPU | 0.12 s of the laptop's |
| what the answer is compared with | a fit against a threshold | the tracked pose, place against place |
| what moves the belief | two searches that agree | 3 candidates, from 3 different scans, that agree with each other and disagree with the tracker |

A candidate is one self-contained JSON message on `/localization/candidate` (laptop → board in
vision mode): the place, a 3x3 covariance read off the correlation peak's own shape, the fit
there, how alike the runner-up explained the scan (`ambiguity`), the stamp and the identity of
the revolution it was computed on, and the map's identity. The board first carries it from the
moment of that revolution to now over its own odometry (`pepin.watchdog.carried`) — the search
costs 0.12-0.25 s and the link a hop on top, and an uncarried answer is that quarter-second of
driving installed as the pose now, always backwards along the drive — and then judges it against
its own fresher pose (`pepin.watchdog.judge`) — **agree** (the everyday verdict), **disagree** (another place,
clearly better, and the map is sure of it), **unknown_map** (nothing on this map fits, or two
places fit alike), **nothing** — and `CandidateGate` turns three disagreements about one place
into a re-seed through the same door the board's own search uses. The seed is not the candidate
but the two weighed by their information, so a sure candidate against a lost tracker *is* the
candidate, and a bounded one barely moves a healthy tracker.

Measured offline on tape 0171 (`scratch/kidnap_recovery.py`, replayed through the very tracker
the node builds; the cart is carried 1.0 m / 40 deg at t0+20 s while the odometry and the scans
go on as they were):

| | back under 10 cm |
| --- | --- |
| the tracker's local window alone | never (0.69 m after 39 s) |
| the board's own fallback, as today | never — the cart is driving, so it may not search |
| the same, told the cart had stopped | never — 4 searches found the truth and none was applied |
| watchdog, acting on the first candidate | 0.8 s, 8 scans |
| **watchdog + the streak of 3 (shipped)** | **2.9 s, 28 scans** |

Why the board's own path fails here: a metre from the truth this flat still fits the map at
**0.53**, just under the `lost_fit` of 0.55 — so the tracker hardly calls itself lost, and every
time the fit reads over the threshold the watch drops the pending candidate before a second
search can confirm it (`pepin.watch.LostWatch.observe`). Its four searches all found the right
place (0.72–0.79 against the tracker's 0.50–0.60) and all four died as unconfirmed candidates. A
fit against a threshold cannot see a wrong place that fits; two places compared can.

Over the whole undisturbed tape the watchdog re-seeded **0 times** (with a streak of 1 as well
as 3) and never moved the pose by a millimetre. Searched against another flat's map, 9 of 12
candidates read `unknown_map` and the gate says "the map does not fit" — in the report line and
on `/localization/sources`, never as an automatic mode switch.

One scan is one opinion. The laptop never searches a revolution twice: a message repeating the
stamp in hand is dropped, and a revolution nobody has replaced within `watch_max_scan_age_s`
(counted from when it ARRIVED here, never from its stamp — the board's clock runs seconds ahead
of the Mac's) stops being searched at all. Its id travels with the candidate, and the board
refuses a candidate whose scan id is already in the run (`distinct_scans`): a frozen `/scan`
publishes the same answer once a second, and three of those are one scan's evidence, not three
seconds of it — the failure that rubber-stamped every candidate of the board's own two-search
rule on 2026-09-09.

Flags: `global_watch`, `watch_period_s` and `watch_max_scan_age_s` on the laptop's node,
`accept_candidates`, `candidate_streak`, `carry_candidates` and `distinct_scans` on the tracker.
All seven are live; with `global_watch` or `accept_candidates` off the stack is exactly what it
was before, the board's own slow search and nothing else.

### Not yet verified on the robot

- Nothing here has run on the robot: the numbers above are a replay of a recorded tape.
- The camera half of this node has not run on the robot either. Offline
  (`scratch/laptop_localizer_replay.py`) the split path costs the same as the fused tracker did
  at full rate; live, only the board's own report line can say whether its match time came back
  to the lidar-only 45 ms and 9-10 Hz.
- The link's own latency is modelled as 50 ms in the replay. The board carries every candidate
  over the odometry between its scan and now, so that number sets how far the carry reaches,
  not how wrong the seed is; a candidate older than the odometry history (5 s) is dropped as
  `stale`, and the carry itself has never been measured on the robot.
- 3 of 12 candidates computed against a *wrong* map still read `disagree` rather than
  `unknown_map`: the fit floor (0.45) and the ambiguity ceiling (0.90) are the two numbers that
  decide it, and they are tuned on one flat.
- A re-seed while a goal is running is refused (`_navigating`); a re-seed while the cart is
  merely driving is allowed, which no drive has tried yet.

## Feature flags

Every behaviour that can be switched is a live parameter of the node that owns it, declared
once in that node's `FLAGS` table (`pepin.flags`, declared to ROS by
`pepin_bringup.node_kit.Switches`) with its kind, default and four texts, checked by kind on
every change, and printed in the node's report line (CLAUDE.md rule 19). A change lives until
the node restarts; a default changes in the table. This section is generated by
`ros/tools/flags_doc.py` (a unit test keeps it current).

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
the camera's intrinsics, the fusion band) live in `config/*.json` and are read at start.

| node | flag | kind | default | live | description |
| --- | --- | --- | --- | --- | --- |
| `bridge_watch` | `flow_watch` | bool | on | yes | count the messages of every topic that should arrive on this side and repair a topic that carries nothing; off, this watch sees only the board bridge's identity and its route count, as before |
| `bridge_watch` | `flow_silence_s` | number 5..300 | 20.0 | yes | seconds a topic both bridges say should flow may carry nothing before it counts as a dead route |
| `bridge_watch` | `bridge_restart` | bool | on | yes | repair a dead route by restarting the laptop's bridge container alone (Docker Engine API over /var/run/docker.sock); off, the repair is the old one — this whole half restarts, which throws away the fusion model and RTAB-Map's working set |
| `camera_stream` | `scale` | number 0..1 | 0.5 | yes | the published picture as a fraction of the camera's own 1280x720, its optics scaled with it; a change takes the next frame |
| `camera_stream` | `undistort` | bool | off | yes | the published picture is rectified with the checkerboard calibration (config/camera.json's intrinsics) and its camera_info then says no distortion; a no-op while the camera is uncalibrated, since there is nothing to undo. Rectifying crops to the largest all-valid rectangle, so the field of view narrows |
| `camera_stream` | `static_camera_tf` | bool | on | at start | base_link -> camera_link is broadcast from here; it goes off (ros/laptop.sh vslam --neck) when the board's neck node publishes that edge live from the servo encoders (neck_state, flag neck_tf), because two publishers of one edge fight |
| `contact_scan` | `contact_scan` | bool | on | yes | the contact line is published; off, the node is a subscriber that costs nothing — the costmap's own contact_layer.enabled is the other end of the same demo switch, and either one alone takes the camera's floor line out |
| `contact_scan` | `shadow` | bool | on | yes | the last floor pixel on a face stands a band's width UP that face, so its ray lands past the foot: on, that width is taken back off the range (pepin.contact.band_shadow); off is the raw boundary ray |
| `contact_scan` | `imu_lean` | bool | on | yes | the floor plane leans with the gyro as well as the accelerometer (pepin.lean: the lean of a wheel climbing a threshold is followed within a sample instead of being gated away as a push); off, the accelerometer alone, as it always has been |
| `contact_scan` | `max_range` | number 0.1..10 | 2.0 | yes | metres past which a column is called clear instead of ended; the costmap's contact_layer.obstacle_max_range must match it |
| `depth_fusion` | `enabled` | bool | on | yes | frames are fused into the model; off, they are dropped |
| `depth_fusion` | `fit_gate` | bool | on | yes | frames are fused only while the tracker reports /localization_fit >= 0.50; off, every frame is fused |
| `depth_fusion` | `camera_map` | bool | on | yes | the volume's camera band goes out on /map_camera at map_hz whoever owns /map: beside a known map the lidar keeps the served file and the camera's scans are matched against this band of the volume, their own cross-section of the room; off, /map_camera is published only where the volume is the map (SLAM with world_map) |
| `depth_fusion` | `imu_lean` | bool | on | yes | the cart's lean (pepin.lean, from /imu/data_raw) is followed with the gyro as well as the accelerometer, and a frame is placed with the lean at its stamp composed on base_link before the planar odometry instead of as if the cart stood level; the lidar's scan follows the same switch — its beams are walked as the 3D rays the leaning body sends them along, and lean_gate_deg drops the scans taken too far from level |
| `depth_fusion` | `lean_gate_deg` | number 0..90 | 3.0 | yes | a scan taken while the cart leans more than this many degrees is not integrated into the map; only with imu_lean on, which is where the lean is known at all |
| `depth_fusion` | `lean_min_quality` | number 0..1 | 0.5 | yes | how much of the lean gravity must have voted for (pepin.lean's quality, printed beside the lean in this line) before a frame or a scan is placed by it: below it the lean is treated as unknown — the measurement is placed level and the scan gate admits it |
| `depth_fusion` | `self_heal` | bool | off | yes | a streak of 30 frames refused at the alignment bound empties the model, so it re-seeds from the next frame instead of staying frozen until a human resets it |
| `depth_fusion` | `align` | bool | on | yes | frame-to-model: a frame's lidar-height band is turned about the cart to fit the model before it is fused, and a frame whose best turn is the search's bound (+-4 deg) is refused |
| `depth_fusion` | `min_weight` | number 0..100 | 2.0 | yes | observations a voxel needs before it is shown in /fusion/surface (the debug cloud only: /map has map_min_weight) |
| `depth_fusion` | `map_min_weight` | number 0..20 | 2.0 | yes | observations a voxel needs before it speaks in /map. Its own flag, and capped at the lidar's own weight cap |
| `depth_fusion` | `camera_map_min_weight` | number 0..60 | 20.0 | yes | observations a voxel needs before it speaks in /map_camera — the band the camera's own scans are MATCHED against. Its own flag, far above map_min_weight: a picture may show what one frame saw, a reference may not |
| `depth_fusion` | `lidar_map` | bool | off | yes | the volume's lidar layer also goes out on /map_lidar, at map_hz, whoever owns /map: a topic of its own the board's tracker can be pointed at (the relocalizer's map_topic flag) while Nav2 and the map_server keep the /map they have |
| `depth_fusion` | `surface_hz` | number 0.1..10 | 1.0 | yes | how often /fusion/surface is published (the crossing search costs a fraction of a second) |
| `depth_fusion` | `band_half_z` | number 0.02..0.5 | 0.125 | yes | half the height band around the lidar's plane a frame is seated on, metres (config/fusion.json's band_half_z_m is the default); the band's centre is the plane the published base_link -> laser edge names, and both are printed in the report line |
| `depth_fusion` | `lidar_layer` | bool | on | yes | /scan is integrated into the volume at the lidar's plane (rays carve free space, returns mark a surface); off, the volume is the camera's alone, as it was |
| `depth_fusion` | `no_return_free` | bool | off | yes | a beam that came back with nothing carves free space out to the sensor's reach (an open door reads as open); off, it writes nothing at all |
| `depth_fusion` | `map_source` | choice: file, volume | file | yes | where /map comes from: the saved file another node serves, or the volume's own lidar layer published from here at map_hz. Only where the stack was launched with world_map:=true; anywhere else volume is refused, because another node is on /map |
| `depth_fusion` | `map_hz` | number 0.1..5 | 1.0 | yes | how often the volume's layer goes out as /map when map_source is volume |
| `depth_fusion` | `snapshot_s` | number 0..3600 | 60.0 | yes | how often the volume is written to world_path (0: only at shutdown) |
| `depth_fusion` | `resume_volume` | bool | on | at start | a volume snapshot at world_path is loaded at start, so a known room is a resumed volume; off, the volume starts empty and grows from the sensors |
| `depth_stream` | `edge_filter` | bool | on | yes | flying pixels at object edges are dropped from the published depth and the scan; the law's beam pairs skip them regardless |
| `depth_stream` | `lidar_anchor` | bool | on | yes | the lidar's returns pair with the network's depth and fit the law; off, the last law is held (the failure mode of a lidar that stops) — with no law yet nothing is published until it is back on |
| `depth_stream` | `floor_pairs` | bool | off | yes | the floor's pixels pair the network's depth with the plane's geometric depth, a second hoop for the law that needs no lidar |
| `depth_stream` | `wall_anchor` | bool | off | yes | the lidar's returns extruded up the image, where the network's depth stays continuous, pair the rows above the lidar's with the wall's depth — a third hoop |
| `depth_stream` | `parallax_anchor` | bool | off | yes | the corners this frame shares with the previous one, triangulated against the odometry's transform between the two stamps (pepin.parallax), pair the network's depth with a depth in metres the cart measured by moving — a hoop that needs no lidar and no assumed plane and that lands at every elevation the picture has |
| `depth_stream` | `affine_law` | bool | on | yes | the network's depth through 1 / z = a / D + b, fitted on the pooled pairs; off, the raw network's depth goes out unwithheld |
| `depth_stream` | `ray_law` | bool | off | yes | the law's scale follows the ray's angle off the optical axis, a / D + b fitted per elevation (pepin.elevation) instead of one pair of numbers for the whole picture; it needs wall_anchor on, because on the lidar's own beams a return's elevation and its 1 / z are the same variable (\|corr\| 1.000) and the angular fit is refused |
| `depth_stream` | `wall_correct` | bool | off | yes | after the law, the pixels the wall walk covered are set to the extruded plane's depth outright (the same walk as wall_anchor, applied instead of fitted) |
| `depth_stream` | `floor_anchor` | bool | on | yes | pixels within centimetres of the floor plane snap to it in the published image (the scan is built before it); the plane leans with the cart, from the IMU's up vector |
| `depth_stream` | `depth_backend` | choice: remote, local, auto | local (env PEPIN_DEPTH_BACKEND) | yes | where the network runs: local (the CPU model in this container), remote (the laptop's GPU service, ros/depth_host.sh), auto (the service while it answers, the CPU model while it does not) |
| `depth_stream` | `scale_ceiling` | number 0.5..20 | 5.0 | yes | the largest 1 / scale the law may be fitted to (the upper half of pepin.depth.A_BOUNDS); a law that lands on a bound prints AT BOUND |
| `depth_stream` | `imu_lean` | bool | on | yes | the cart's lean (pepin.lean, from /imu/data_raw) is followed with the gyro as well as the accelerometer and carried into the scan's carry and the camera's place in the map; off, the floor plane leans with the accelerometer alone, as it always has, and nothing else is leaned |
| `depth_stream` | `lean_min_quality` | number 0..1 | 0.5 | yes | how much of the lean gravity must have voted for (pepin.lean's quality, printed beside the lean in this line) before a frame is placed by it: below it the lean is treated as unknown and the frame is placed level |
| `goal_server` | `tf_pose` | bool | on | yes | where no tracker answers, the cart's pose is read from TF (map -> base_link) and a goal is judged by how fresh that edge is; off, only the tracker is ever asked |
| `goal_server` | `correction_watch` | bool | on | yes | where no tracker answers, the SLAM correction (/map_odom) must be arriving for a goal to start, and a drive is cut when it stops; off, the age of map -> base_link is the only evidence read |
| `laptop_localizer` | `global_watch` | bool | on | yes | run the whole-map search once every watch_period_s and publish what it finds on /localization/candidate; off, this half of the node is a subscriber that costs nothing and the board is back to searching for itself only once it is already lost |
| `laptop_localizer` | `watch_period_s` | number 0.2..60 | 1.0 | yes | seconds between searches |
| `laptop_localizer` | `watch_max_scan_age_s` | number 0.1..3600 | 1.0 | yes | how long a revolution may sit in hand and still be searched, counted from when it ARRIVED here |
| `laptop_localizer` | `camera_sources` | list of: depth, contact | depth,contact | yes | which camera scans are matched here and sent to the board as pose measurements on /localization/measurement: the depth band, the floor-contact line; empty, nothing is matched and the board tracks on the lidar alone |
| `laptop_localizer` | `camera_match_hz` | number 0.2..30 | 5.0 | yes | how often each camera source is matched and a measurement published |
| `laptop_localizer` | `camera_window_m` | number 0.01..1 | 0.09 | yes | half-width of the window a camera scan is matched in, metres, around the board's belief carried to that scan's moment |
| `laptop_localizer` | `camera_window_deg` | number 0.5..90 | 9.0 | yes | half-width of the same window in heading, degrees |
| `laptop_localizer` | `camera_min_fit` | number 0..1 | 0.25 | yes | a camera match whose fit is below this is not sent: it is counted as low fit and the board never hears about it |
| `laptop_localizer` | `covariance` | choice: peak, fit | peak | yes | how sure a camera measurement says it is: peak — the spread of that match's own score peak at the camera matcher's temperature (config/matcher.json); fit — the fit-scaled second moment of the whole surface, with the source's trust in it, that shipped before it. It is the number the board's information filter weighs the fan by |
| `laptop_localizer` | `explained_vote` | bool | on | yes | returns the map cannot explain (a person, a moved chair) do not score a camera match: the same vote the board's tracker takes on its own scans (relocalizer's explained_vote), taken here, on the grid the camera is matched against |
| `neck_state` | `neck_tf` | bool | on | yes | base_link -> camera_link is published live from the neck's encoders; the laptop's camera node must then run with ros/laptop.sh vslam --neck, or two nodes publish that edge |
| `relocalizer` | `rest_lock` | bool | on | yes | hold the pose while the cart stands still (wheels quiet 0.6 s and the gyro under 1.5 deg/s): a match's residual is blended in with a time constant instead of taken whole |
| `relocalizer` | `explained_vote` | bool | on | yes | returns the static map cannot explain (a person, a moved chair) do not score the match |
| `relocalizer` | `rest_tau_s` | number 0.1..60 | 6.0 | yes | the rest lock's time constant: seconds for a residual to die at rest |
| `relocalizer` | `rest_gain` | number 0..1 | 0.05 | yes | the rest lock's share per match when no match cadence is known |
| `relocalizer` | `sources` | list of: lidar, depth, contact, camera | lidar | yes | what corrects the pose: the lidar's revolution (/scan), matched here, and the camera (`camera`), whose scans the laptop matches and whose ANSWER arrives on /localization/measurement. The lidar drives the updates while it is fresh and the camera's word rides along, carried to its moment; a stale lidar hands the updates to the measurements. `depth` and `contact` name the camera's raw scans, which this node no longer subscribes to — enabling them changes nothing here |
| `relocalizer` | `measurement_max_age_s` | number 0.05..5 | 0.5 | yes | how old a pose measurement from the laptop may be, in seconds, at the moment of the update that would take it: past this it is dropped instead of carried |
| `relocalizer` | `remote_floor_xy_m` | number 0..1 | 0.08 | yes | the least position sigma, metres, a measurement from the laptop is fused with, whatever its own peak claims; 0 takes the claim as it comes |
| `relocalizer` | `remote_floor_yaw_deg` | number 0..90 | 5.0 | yes | the least heading sigma, degrees, a measurement from the laptop is fused with; 0 takes the claim |
| `relocalizer` | `fusion` | bool | on | yes | fuse every enabled source's word by its information — a match made here, a measurement made on the laptop; off: the widest source corrects alone and the others only report |
| `relocalizer` | `covariance` | choice: peak, fit | peak | yes | how sure a match says it is: peak — the spread of its own score peak at the matcher's calibrated temperature (config/matcher.json); fit — the fit-scaled second moment of the whole surface that shipped before it. Both the covariance the lidar's match is fused by and the one /tracker_pose carries |
| `relocalizer` | `self_check` | bool | on | yes | every source vouches for itself: its covariance is widened by how far its answers fall from where its OWN previous answer, carried over the odometry, said they would (pepin.selfcheck). A source four times out in ALL THREE directions loses sixteen times its weight; the factor is that over-claim averaged over the three, so a source out in fewer of them loses proportionally less (a camera fan bound along a wall, four times out in the two directions it measures, is widened 9.7x not 16x — scratch/selfcheck_audit.py). One that is honest, or better, is not touched. Per source, never across sources: no lidar pose enters the camera's number and no camera pose the lidar's |
| `relocalizer` | `local_fit` | bool | on | yes | /localization_fit carries only a fit a scan of THIS machine measured: with no scan here at all — the camera's measurements driving the tracker alone — it carries 0.0, the value it holds before the first match, and the camera's own fit rides /localization/sources per source; off, the remote fit is published there as the tracker's own |
| `relocalizer` | `toe_reach` | number 0..0.6 | 0.27 | yes | how far past the leg the lidar sees a standing person's toe reaches, metres: the term the dynamic rings are sized on (pepin.dynamic.berth_for). The default is computed from the lidar's mount (config/lidar.json) |
| `relocalizer` | `near_rings` | bool | on | yes | a return is ringed as soon as it clears the cart's own outline, and only the marks that would land on that outline are dropped; off, nothing within the ring plus the outline is ringed at all — the older rule, whose blind disc grows with the ring |
| `relocalizer` | `accept_candidates` | bool | on | yes | re-seed from the laptop watchdog's whole-map candidates (/localization/candidate, pepin.watchdog): a place that disagrees with the tracked pose candidate_streak times in a row, about the same place each time, is adopted through the path the board's own search uses |
| `relocalizer` | `candidate_streak` | integer 1..10 | 3 | yes | how many candidates in a row must disagree with the tracker and agree with each other before one of them re-seeds it: the price of a teleport, in seconds |
| `relocalizer` | `map_topic` | choice: map, map_lidar | map | yes | which map this tracker matches on: /map, whatever the stack's map owner publishes there (the served pgm in split and vision mode), or /map_lidar, the lidar layer of the laptop's fused volume (pepin_bringup.depth_fusion, flag lidar_map) |
| `relocalizer` | `map_refresh_s` | number 0..600 | 0.0 | yes | the least time between two adoptions of the map topic: a newer map on the topic in use is taken only after this many seconds AND only if its cells changed. 0 takes the first map and no other, which is what a served file has always done |
| `relocalizer` | `carry_candidates` | bool | on | yes | a candidate's pose is moved from the moment of its own scan to now over the odometry between the two stamps (pepin.watchdog.carried) before it is judged and fused, and one the odometry history no longer covers is dropped |
| `relocalizer` | `distinct_scans` | bool | on | yes | a streak is counted in scans, not in messages: a candidate whose scan id is already in the run is a second opinion that heard the first one's scan, counted as replay and not lengthening the streak |
| `rtabmap_frame` | `slam` | bool | off | at start | RTAB-Map is the map (online SLAM): its correction is map -> odom and goes to the board as a message on /map_odom, where pepin_bringup.slam_frame broadcasts it; off, the board's tracker owns map -> odom and this node broadcasts map -> rtabmap here |

### The flags one by one

#### `bridge_watch`

- **`flow_watch`** — bool, default on
  - *What:* count the messages of every topic that should arrive on this side and repair a topic that carries nothing; off, this watch sees only the board bridge's identity and its route count, as before
  - *Default:* on — a route count cannot see the failure that cost two evenings: /depth_scan (2026-09-12) and /imu/data_raw (2026-09-13) each had a route on both admins and a live publisher on the far side, and carried zero messages until a bridge was restarted by hand. The mechanism is read off the bridge's own source and its admin (pepin.deployment.BRIDGED_QOS): a route's DDS QoS is whatever the declaration that created it carried, and it is never revised. The subscriptions this flag adds are free — every one of them is on a topic some other node here already receives
  - *On when:* always on a split or vision stack: it is the only thing that can tell a dead route from a quiet one
  - *Off when:* while bisecting the bridge by hand, so the watch takes no action of its own
- **`flow_silence_s`** — number 5..300, default 20.0
  - *What:* seconds a topic both bridges say should flow may carry nothing before it counts as a dead route (5..300)
  - *Default:* 20.0 — the topics watched here are the periodic ones — /scan at 10 Hz, /tf at 40, /imu at 48, /localization/sources at 1 — so twenty seconds is twenty missed messages of the slowest of them and four polls of this watch, while still shorter than a goal. A topic published on a change only (/pepin/run_status) is never starved: nothing publishes it between changes, and the watch only counts what both bridges call live. default by design, unmeasured
  - *On when:* shorten it when a dead route must be caught inside a drive
  - *Off when:* lengthen it on a congested link, where a ten-second stall is the wireless hop and not the bridge
- **`bridge_restart`** — bool, default on
  - *What:* repair a dead route by restarting the laptop's bridge container alone (Docker Engine API over /var/run/docker.sock); off, the repair is the old one — this whole half restarts, which throws away the fusion model and RTAB-Map's working set
  - *Default:* on — the bridge offers nothing gentler: its REST admin is read-only in 1.7.0 (the running config prints permissions { read: true, write: false }), so there is no reload and no way to drop a single route. Restarting the container re-creates every route in a few seconds and leaves pepin-vslam alive. It falls back by itself when the docker socket is not mounted, and escalates to the whole half when the silence returns after a restart. default by design, unmeasured
  - *On when:* always: it is strictly less destructive than the fallback
  - *Off when:* when the laptop's bridge must not be touched — bisecting it by hand, or running without the docker socket mounted

#### `camera_stream`

- **`scale`** — number 0..1, default 0.5
  - *What:* the published picture as a fraction of the camera's own 1280x720, its optics scaled with it; a change takes the next frame (0..1)
  - *Default:* 0.5 — default by design, unmeasured: the half size was chosen when the stream was made reliable for RTAB-Map (2026-09-09) and has never been compared with the full one — no feature count, no loop closure, no bandwidth measured either way. What is measured is the rate: 8.9 fps over the bridge then, 11-11.5 fps in the report lines since. A full-size bgr8 frame is 2.7 MB of arithmetic (1280 x 720 x 3), and 640x360 is what the depth network resizes to anyway
  - *On when:* raise it towards 1.0 when place recognition or a calibration needs the detail and the bridge has the bandwidth to carry it
  - *Off when:* lower it when the bridge is the bottleneck: the optics are scaled with the picture, so nothing downstream has to be told
- **`undistort`** — bool, default off
  - *What:* the published picture is rectified with the checkerboard calibration (config/camera.json's intrinsics) and its camera_info then says no distortion; a no-op while the camera is uncalibrated, since there is nothing to undo. Rectifying crops to the largest all-valid rectangle, so the field of view narrows
  - *Default:* off — default by design, unmeasured: the camera is calibrated (45 views, rms 0.230 px, fx 724.1, fy 726.8, cx 652.0, cy 374.0, k1 -0.150, k2 -0.129, k3 +0.092, HFOV 82.9 deg, 2026-09-13), but the straightened picture has never been compared with the raw one on the robot, and nobody has measured what the crop costs in field of view
  - *On when:* when a consumer needs straight lines — a checkerboard, a marker, a recogniser that assumes a pinhole
  - *Off when:* wherever a consumer was measured in the raw picture's optics (the depth pipeline's law was fitted there), and wherever the field of view matters more than straight lines
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
- **`max_range`** — number 0.1..10, default 2.0
  - *What:* metres past which a column is called clear instead of ended; the costmap's contact_layer.obstacle_max_range must match it (0.1..10)
  - *Default:* 2.0 — where the floor stops being the floor, not where the optics run out: on 22 open-floor frames of run 0171 the network's floor sits at 1.01 of the geometric plane at 1.0-1.5 m, 0.96 at 1.5-2.0, 0.89 at 2.0-2.5 and 0.80 at 2.5-3.0 — and 0.89 of the plane is 13 cm of height, the width of the band itself, so past 2 m the floor leaves the band on its own and the column ends on nothing. The marks agree: 3 % false below 1.75 m, 44 % beyond it, and without the cap every mark past 2 m is false (scratch/contact_vs_lidar.py). Measured 2026-09-11 on the old geometry (lidar 0.20 m, camera 1.23 m at 26 deg, hfov 78), all three since corrected — the cap has not been re-measured
  - *On when:* raise it only after the floor ratio is re-measured on the corrected geometry, and raise the costmap's obstacle_max_range with it
  - *Off when:* lower it where the floor is patterned, wet or dark, which shortens the range the network's floor stays flat over

#### `depth_fusion`

- **`enabled`** — bool, default on
  - *What:* frames are fused into the model; off, they are dropped
  - *Default:* on — default by design, unmeasured: the kill switch, so the model can be stopped growing without stopping the node, the camera or the report line
  - *On when:* whenever the fused surface or the volume's map is wanted
  - *Off when:* to freeze the model where it stands — a snapshot to save, a picture to read, a run where the camera is carried by hand
- **`fit_gate`** — bool, default on
  - *What:* frames are fused only while the tracker reports /localization_fit >= 0.50; off, every frame is fused
  - *Default:* on — where a tracker speaks, and the off state is measured: in the first online-SLAM session, where nobody publishes /localization_fit and the gate had to come off, the fused floor came out rough — offset +4.7 cm, sd 5.5 cm, 34 % within 3 cm at a tilt of 0.61 deg — against sd 3.3 cm and 71 % within 3 cm in the known-map mode with a centimetre tracker pose. The 0.50 itself is the drive rung of the tracker's own ladder (pepin.watch: blind 0.30, drive 0.50, lost 0.55), inherited, not swept for fusion
  - *On when:* in the known-map modes (split, vision), where the board's tracker publishes the fit: it keeps a frame taken while the pose was wrong out of the model. The launch brings it up on there and off in SLAM mode; this is how to put it back on by hand
  - *Off when:* in SLAM mode, where RTAB-Map owns the pose and no tracker speaks — with the gate on nothing is ever fused there. vslam.launch.py passes fit_gate:=false in that mode, so nobody has to remember it at the start of a session
- **`camera_map`** — bool, default on
  - *What:* the volume's camera band goes out on /map_camera at map_hz whoever owns /map: beside a known map the lidar keeps the served file and the camera's scans are matched against this band of the volume, their own cross-section of the room; off, /map_camera is published only where the volume is the map (SLAM with world_map)
  - *Default:* on — the camera's fan is the whole height 0.15-1.3 m and the lidar's map is one plane at 0.38 m: held to that plane the fan scored fit 0.34 at the right pose and the fused pose moved 0.8-1.5 cm off the lidar's (2026-09-13, drive_bisect on runs 190024 and 190422 against 184701 and 185621). Nobody else publishes /map_camera, so the owner rule that protects /map has nothing to protect here
  - *On when:* always beside a known map: it is the only reference the camera's scans can honestly be matched against, and it costs one slice per map_hz
  - *Off when:* to reproduce the old behaviour, the camera matched against the lidar's /map; or where the band is known to be wrong (a volume painted at a wrong pose) until it is wiped
- **`imu_lean`** — bool, default on
  - *What:* the cart's lean (pepin.lean, from /imu/data_raw) is followed with the gyro as well as the accelerometer, and a frame is placed with the lean at its stamp composed on base_link before the planar odometry instead of as if the cart stood level; the lidar's scan follows the same switch — its beams are walked as the 3D rays the leaning body sends them along, and lean_gate_deg drops the scans taken too far from level
  - *Default:* on — on: the gyro's sign was verified by hand on 2026-09-13 (the cart tipped nose-down read pitch +10.5 deg, left-side-down read roll -9.4 deg, the fast path following at once with quality 0.9 while held; scratch/lean_tip_test.txt), and the gyro's zero offset is learned. Off, the floor anchor keeps the accelerometer-only lean, which by design ignores any tip shorter than 10 s
  - *On when:* after a hand tip through a known angle shows the reported lean following it the right way and returning to zero
  - *Off when:* wherever the lean in the report line disagrees with the cart's visible attitude; off, the lean is still estimated and reported, only not applied
- **`lean_gate_deg`** — number 0..90, default 3.0
  - *What:* a scan taken while the cart leans more than this many degrees is not integrated into the map; only with imu_lean on, which is where the lean is known at all (0..90)
  - *Default:* 3.0 — default by design, unmeasured: chosen, not fitted. The stake is arithmetic: a beam at 5 m lands r sin(lean) off the sensor's plane — 26 cm at this 3 degrees, 44 cm at 5 — so a tipped revolution is looking at another slice of the room. The one replay that exists (a synthetic 5 degree bump over run 0171, scratch/lidar_lean_effect.txt) has the gate refusing 21 of 81 revolutions and keeping fewer walls than simply walking the beams as 3D rays (66.7 % against 74.9 % at the top of the bump, 89.2 % against 92.0 % six seconds later): there, it cost more than it bought
  - *On when:* lower it where the map must stay clean and revolutions are plentiful
  - *Off when:* 90 admits every revolution again, as before the gate existed, and the report line's leaned_out count says what would have been dropped
- **`lean_min_quality`** — number 0..1, default 0.5
  - *What:* how much of the lean gravity must have voted for (pepin.lean's quality, printed beside the lean in this line) before a frame or a scan is placed by it: below it the lean is treated as unknown — the measurement is placed level and the scan gate admits it (0..1)
  - *Default:* 0.5 — chosen on a simulation, not on the robot: in scratch/lean_quality_floor_probe.py a 0.2 deg/s gyro bias reports 3.0 degrees of tip on a level floor at quality 0.02 or less, nothing past 0.13 degrees of it survives a floor of 0.5, and a real 6 degree threshold climb keeps quality 1.00 throughout — so the floor costs the feature nothing. The 0.2 deg/s is hypothetical: this chip's worst measured axis is 0.074 deg/s (config/imu.json's level block). A drifting gyro that reported 3 degrees would otherwise sit exactly on lean_gate_deg and refuse every revolution
  - *On when:* raise it towards 1.0 on a robot that only ever leans when something real pushes it
  - *Off when:* 0 believes every lean, as before the floor existed: an A/B of the gyro's own drift
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
- **`min_weight`** — number 0..100, default 2.0
  - *What:* observations a voxel needs before it is shown in /fusion/surface (the debug cloud only: /map has map_min_weight) (0..100)
  - *Default:* 2.0 — inherited from the map slice, where it was measured: at min_weight 2 the lidar slice holds 905 walls and at 6 it holds 817, the cells a single pass wrote falling out (scratch/worldmap_from_tape.txt). For the debug cloud itself nothing was measured; it is the same number so the picture and the map agree
  - *On when:* raise it to show only what several frames agree on
  - *Off when:* 0 shows every voxel ever touched, noise included — a look at what one pass sees; it changes nothing the cart drives on
- **`map_min_weight`** — number 0..20, default 2.0
  - *What:* observations a voxel needs before it speaks in /map. Its own flag, and capped at the lidar's own weight cap (0..20)
  - *Default:* 2.0 — the cap is measured: lidar cells saturate at LidarLaw.max_weight 20 while the volume's own cap is 60, so anything above 20 leaves the whole map unknown — a synthetic box at min_weight 21 published free 0, occupied 0, unknown 14400, with Nav2 and the tracker driving on that. The 2.0 is the slice's own measured maturity (905 walls at 2, 817 at 6)
  - *On when:* raise it towards 20 for a map that must be certain — a world the cart has driven more than once, saved to file
  - *Off when:* lower it towards 0 in a fresh room, where the cart must plan through what a single pass saw
- **`camera_map_min_weight`** — number 0..60, default 20.0
  - *What:* observations a voxel needs before it speaks in /map_camera — the band the camera's own scans are MATCHED against. Its own flag, far above map_min_weight: a picture may show what one frame saw, a reference may not (0..60)
  - *Default:* 20.0 — measured on the live volume (scratch/camera_band_weights.py on ros/maps/world_live.npz, 190066 frames): the band's occupied columns carry weight p25 20.0, median 29.3, and the band holds 21276 occupied cells at weight 2, 14285 at 20 (67 %), 9510 at 50 — so the tens are reachable, with two thirds of the walls surviving. The columns the lidar never wrote, the camera's own, fall from 148 to 26 over the same step. The number itself is the volume's weighting read at the frame rate: at weight_ref_m 2.0 m an observation weighs 1 and the stream runs 8-9 frames a second, so 20 is 20 frames, 2.5 s of watching one cell from 2 m (5.6 s from 3 m, 10 s at the 4 m range limit, and 0.6 s at the weight_cap 4.0, a metre and nearer). Below that a cell is one glance from one place — which is how camera-only localisation walked away in 20-33 cm and 15-23 deg steps on 2026-09-13, matching a band it had painted itself at the drifting pose. 20 is also the lidar's own cap (LidarLaw.max_weight): a saturated lidar cell still speaks in the band at exactly 20 and nothing the lidar wrote speaks above it (the lidar slice holds 3561 occupied cells at 20 and 0 at 30)
  - *On when:* raise it toward the volume's cap 60 for a room the cart has driven more than once: only walls integrated for many seconds from several places would remain
  - *Off when:* lower it to map_min_weight to reproduce the old behaviour — the matcher handed every cell two frames had touched. The report line says what the band costs: the share of its occupied cells this threshold keeps
- **`lidar_map`** — bool, default off
  - *What:* the volume's lidar layer also goes out on /map_lidar, at map_hz, whoever owns /map: a topic of its own the board's tracker can be pointed at (the relocalizer's map_topic flag) while Nav2 and the map_server keep the /map they have
  - *Default:* off — a SEEDED volume's layer is the served map itself: slice it and 18274 of the pgm's 18274 known cells agree, every wall of each inside one cell of the other, and the four tapes of 2026-09-13 replayed on the exported slice give live pose error medians 0.6/0.5/1.3/0.7 cm against the file's own 0.6/0.5/1.3/0.6 — the same map (scratch/volume_vs_pgm.py, scratch/drive_bisect.py --map). Today's UNSEEDED live volume is not: only 52.1 % of the saved map's walls lie within a cell of it and 25.6 % of its own walls lie within a cell of the saved map, which is why this ships off and why seed_map exists. The second cost is the map id: the volume's grid is 280x250 cells and the served map 239x215, so a tracker that adopts /map_lidar answers to another map id and the laptop's candidates and camera measurements — stamped with the id of /map (pepin_bringup.laptop_localizer) — are refused as evidence about another map until that half moves too
  - *On when:* with a volume seeded from the served map (seed_map), to point the board's tracker at the room as it is now instead of at the frozen file
  - *Off when:* the default, and mandatory for a volume nobody seeded: one slice per map_hz saved on the laptop, and nothing on the bridge the board does not read
- **`surface_hz`** — number 0.1..10, default 1.0
  - *What:* how often /fusion/surface is published (the crossing search costs a fraction of a second) (0.1..10)
  - *Default:* 1.0 — default by design, unmeasured; what is measured is the cost it protects — the surface build took 45 ms a second and stalled the node's executor until it was moved onto a snapshot taken outside the model lock
  - *On when:* raise it for a demo where the surface must follow the head, watching the stage timings in the report line
  - *Off when:* lower it towards 0.1 on a busy machine, or where the model matters and the picture does not
- **`band_half_z`** — number 0.02..0.5, default 0.125
  - *What:* half the height band around the lidar's plane a frame is seated on, metres (config/fusion.json's band_half_z_m is the default); the band's centre is the plane the published base_link -> laser edge names, and both are printed in the report line (0.02..0.5)
  - *Default:* 0.125 — default by design, unmeasured as a width: the centre the band sits on is measured, this half-width is not. The lidar's plane is 0.383 m by tape (2026-09-12), where beams and vertical walls read the network's scale 3 % apart against 18 % at the 0.200 m that had been assumed, and moving the band there took the fused band's distance to the lidar from 12.9 cm to 3.8-5.2 cm. The 0.125 m is the width the band has always had (0.10-0.35 m around the assumed plane) and has never been swept. For scale: the band is the best layer the camera has — median 9.2 cm against the beams, against 15.7/39.4/46.7 cm for the slices above it — and a 5 degree lean moves a beam's world height by up to 55.7 cm, wider than the band itself
  - *On when:* widen it when frames are refused for want of band points (the count is in the report line): a narrow band on a leaning cart has nothing to seat on
  - *Off when:* narrow it to keep only the rows the beams truly anchor, at the price of fewer points to align on
- **`lidar_layer`** — bool, default on
  - *What:* /scan is integrated into the volume at the lidar's plane (rays carve free space, returns mark a surface); off, the volume is the camera's alone, as it was
  - *Default:* on — replayed from tape 0171 into an empty volume the lidar layer reproduces the saved map's walls to a median 0.0 cm, p90 13.0 cm, 79.3 % within one cell, and where the volume says free the saved map agrees 91.3 % of the time — for 1 ms a scan (575 scans in 0.8 s). It is also protected from the camera: 0 of 13499 lidar cells were changed by depth, while the camera filled 922 cells the lidar never reached
  - *On when:* on wherever the volume is the map (map_source volume) or the surface must show what the lidar knows
  - *Off when:* to measure the camera alone — what the depth adds, and where it lies
- **`no_return_free`** — bool, default off
  - *What:* a beam that came back with nothing carves free space out to the sensor's reach (an open door reads as open); off, it writes nothing at all
  - *Default:* off — default by design, unmeasured: no false-carve rate was ever taken, and with the real /scan the branch is unreachable anyway — pepin.msgs.scan_arrays turns everything past range_max into NaN and config/lidar.json's max_range_m is that same 12.0 m, so a doorway carved nothing and stayed unknown. It stays off because a mirror, a black chair leg and anything nearer than the 0.05 m minimum all say the identical nothing, and carving them out to 12 m would rub out the wall behind them
  - *On when:* when a beam carries something that separates an open bearing from a mirror or a black surface — return quality, or the same emptiness confirmed from several viewpoints; nothing on this robot does today
  - *Off when:* leave it off: an open door stays unknown, which a planner may be told to cross (allow_unknown) rather than being told a lie
- **`map_source`** — choice: file, volume, default file
  - *What:* where /map comes from: the saved file another node serves, or the volume's own lidar layer published from here at map_hz. Only where the stack was launched with world_map:=true; anywhere else volume is refused, because another node is on /map (one of: file, volume)
  - *Default:* file — the other state has been seen to break a run: on 2026-09-10 RTAB-Map's own grid landed on /map beside the board's static map and fed the laptop's global costmap a second, growing map. Two publishers of one /map is the failure, so the deployment's map_owner and the launch's world_map:=true must both agree before volume is allowed. The volume itself is good enough — its walls sit within one cell of the saved map 79.3 % of the time
  - *On when:* volume where the laptop owns /map (ros/laptop.sh vslam --world-map) and the room is being mapped as it is driven
  - *Off when:* file wherever a map server or RTAB-Map already publishes /map, which is every other mode
- **`map_hz`** — number 0.1..5, default 1.0
  - *What:* how often the volume's layer goes out as /map when map_source is volume (0.1..5)
  - *Default:* 1.0 — default by design, unmeasured: 1 Hz is the cadence RTAB-Map's own map updates at, and /map is transient-local, so a subscriber that arrives late is served the last one regardless
  - *On when:* raise it when the map is built while driving and the costmap lags visibly behind the room
  - *Off when:* lower it on a busy laptop: every publication is a whole grid over the bridge
- **`snapshot_s`** — number 0..3600, default 60.0
  - *What:* how often the volume is written to world_path (0: only at shutdown) (0..3600)
  - *Default:* 60.0 — default by design, unmeasured: the write holds the model lock for about half a second on a grid of noise and less on a real one, which at the node's 9.0-9.5 fps is four or five frames dropped once a minute
  - *On when:* shorten it for a long mapping run nobody will be there to shut down cleanly
  - *Off when:* 0 writes only at shutdown — the setting for a demo where no frame may be dropped
- **`resume_volume`** — bool, default on, not live
  - *What:* a volume snapshot at world_path is loaded at start, so a known room is a resumed volume; off, the volume starts empty and grows from the sensors (not live: set at the next start)
  - *Default:* on — default by design, unmeasured. The one hard rule around it is a guard: a snapshot is resumed only onto the grid config/fusion.json describes (280x250x34 voxels of 5 cm from -19.5, -5.5, -0.15), so a changed grid starts empty instead of resuming into the wrong place
  - *On when:* on in the room the snapshot was taken in
  - *Off when:* off for a new room, after the map's origin moves, or to measure how fast the volume fills from nothing

#### `depth_stream`

- **`edge_filter`** — bool, default on
  - *What:* flying pixels at object edges are dropped from the published depth and the scan; the law's beam pairs skip them regardless
  - *Default:* on — the band probe found 21 % of the band's pixels more than 15 cm from any lidar return, with no bearing trend to blame the focal length or the mount yaw on — flying pixels; the 8 % step that drops them costs 1.2-1.6 ms a frame and 2 % of the pixels (scratch/pipeline_vs_truth.txt, scratch/band_frame_probe.py). The cause is measured, the benefit is not: the one A/B on the robot, one turn each way, showed no difference
  - *On when:* whenever the scan feeds a costmap: the pixels it drops are the ones that become an obstacle with nothing behind them
  - *Off when:* to see the raw band's tail in Foxglove, or when a thin real object (a chair leg at range) is missing from the scan and the 8 % step is the suspect
- **`lidar_anchor`** — bool, default on
  - *What:* the lidar's returns pair with the network's depth and fit the law; off, the last law is held (the failure mode of a lidar that stops) — with no law yet nothing is published until it is back on
  - *Default:* on — the beams are the only metric ruler on board. Without them a floor-only fit reads the lidar's own row 1.98-2.46x too far and the raw network 1.62-1.96x (scratch/pipeline_vs_truth.txt, scratch/lidar_height_check.txt), and the pairing costs 0.0-0.1 ms a frame. The one on/off turn on the robot is split: a held law was better above the band (9.6-19.3 cm against 16-46 cm) and worse at it (12.0 cm against 7-9 cm), and the band is the row the costmap drives on
  - *On when:* whenever the lidar spins — it is what makes the network's depth metric
  - *Off when:* to rehearse a lidar that dies mid-run (the law freezes, nothing else changes), or to compare the slices above the band, where the frozen law measured better
- **`floor_pairs`** — bool, default off
  - *What:* the floor's pixels pair the network's depth with the plane's geometric depth, a second hoop for the law that needs no lidar
  - *Default:* off — useless as measured — it takes the floor pixels the network itself drew and checks the network against them, and the median D/E over the pixels it selects is the median over every ray that meets the plane (0.86 against 0.89, 1.12 against 1.22, 1.76 against 1.69: scratch/horizon_law_eval.txt). What it does change is the law: the lidar's own row goes 1.010 -> 1.391 and the band 12.9/38.2 cm -> 24.1/48.7 on run 0171 (scratch/pipeline_vs_truth.txt), 3.8/20.6 -> 4.8/26.7 cm at the corrected mount (scratch/lidar_height_check.txt). A floor anchor needs a floor cue the network did not draw itself
  - *On when:* when the floor's depth comes from something independent of the network — a measured plane, a second sensor; nothing measured so far supports turning it on
  - *Off when:* in every run that drives: it buys nothing above 0.8 m either (65.2 cm against 48.2) and moves the row the costmap reads by 38 %
- **`wall_anchor`** — bool, default off
  - *What:* the lidar's returns extruded up the image, where the network's depth stays continuous, pair the rows above the lidar's with the wall's depth — a third hoop
  - *Default:* off — redundant with the lidar and worse where the cart drives. The wall extrusion agrees with the beams to 3 % once the mount height is right (lidar/wall 0.966 at the tape's 0.383 m against 0.816 at the assumed 0.200, scratch/lidar_height_check.txt), and switching it on pulls the lidar's own row 10 % near (1.010 -> 0.896) and stretches the band's tail 2.6x (p90 20.6 -> 53.6 cm) on run 0171 (scratch/pipeline_vs_truth.txt, scratch/lidar_height_check.txt). What it buys is the rows above: the 3D error at z 0.80-1.20 m 47.4 -> 31.5 cm
  - *On when:* on a robot with no lidar, where the extrusion is the only wall cue; or when what reads the depth is above 0.5 m (a manipulator's reach) and no costmap is reading the band
  - *Off when:* whenever the cart drives on the band: that is the row the costmap reads, and wall pairs move it 10 %
- **`parallax_anchor`** — bool, default off
  - *What:* the corners this frame shares with the previous one, triangulated against the odometry's transform between the two stamps (pepin.parallax), pair the network's depth with a depth in metres the cart measured by moving — a hoop that needs no lidar and no assumed plane and that lands at every elevation the picture has
  - *Default:* off — not measured live yet: on runs 0171 and 0165 it costs 3.5-3.8 ms a frame and yields 30-190 pairs on only 31-36 % of frames (nothing at all while the cart stands still or turns on the spot), and at those runs' 2.9 cm median baseline the triangulated depth is +25-37 % too far under 1.5 m (19-30 samples a run) while it sits within 3 % of the lidar from 1.5 to 3 m (scratch/parallax_vs_lidar.txt). The odometry's own +-25 % scale band multiplies near and far alike and cannot make a range-dependent bias; the cause is not known
  - *On when:* after a run at driving speed (0.3 s of gap = 10-15 cm of baseline instead of 3 cm) with the calibrated focal length (fx 724.1, HFOV 82.9 deg) either explains the near-field bias or clears it
  - *Off when:* wherever the cart stands, turns on the spot or faces blank walls: it yields nothing there and costs its 3.5 ms anyway
- **`affine_law`** — bool, default on
  - *What:* the network's depth through 1 / z = a / D + b, fitted on the pooled pairs; off, the raw network's depth goes out unwithheld
  - *Default:* on — the raw network is 1.6-2.0x too far — the lidar's own row reads 1.958 raw against 1.033 through the law, and the band against the beams 35.5/112.0 cm against 3.8/20.6 cm, 14 % -> 57 % of it within 5 cm (scratch/lidar_height_check.txt). The fit costs 0.4 ms
  - *On when:* always, to drive
  - *Off when:* as an A/B measure of the correction, at rest — never a way to drive: every published metre is then 1.6-2.0x long
- **`ray_law`** — bool, default off
  - *What:* the law's scale follows the ray's angle off the optical axis, a / D + b fitted per elevation (pepin.elevation) instead of one pair of numbers for the whole picture; it needs wall_anchor on, because on the lidar's own beams a return's elevation and its 1 / z are the same variable (|corr| 1.000) and the angular fit is refused
  - *Default:* off — for good, unless new data arrives: the network's error follows the world's elevation, not the ray's angle. Across three neck pitches (tapes 0235/0236/0237 at 11.1, 25.8, 40.9 deg) the confound gate refuses the fit at two of them, and the one law that could be fitted helps in sample and hurts at both other pitches (scratch/ray_law_pitch_eval.txt); carried between tapes and geometries no angular law beats plain scale out of sample (mean |ln ratio| 13.6 % for scale against 15.1-15.4 % for the ray laws, while the room's own elevation reaches 12.9 %: scratch/horizon_law_eval.txt). Fixing the camera's pose was worth 9 of those 22 points, the best angular term 0.7 more
  - *On when:* only on data that separates the ray's angle from the world's elevation — a pitch sweep with the calibrated lens whose fit the confound gate does not refuse
  - *Off when:* it ships off; the scale itself still moves 21-25 % over 30 deg of neck pitch, which asks for a refit, not for an angular law
- **`wall_correct`** — bool, default off
  - *What:* after the law, the pixels the wall walk covered are set to the extruded plane's depth outright (the same walk as wall_anchor, applied instead of fitted)
  - *Default:* off — default by design, unmeasured as a win — standalone it is a wash — run 0171 keeps the same law and the same lidar row (1.010, |·-1| q3 0.320) and the band reads 12.8/39.3 cm against today's 12.9/38.2, or 3.6/20.2 against 3.8/20.6 at the corrected mount, with the 3D error slightly better at every slice (scratch/pipeline_vs_truth.txt, scratch/lidar_height_check.txt). It is off because it is the wall walk and the wall walk is off; no number says it hurts
  - *On when:* with wall_anchor on, when what reads the depth is the wall above the beams and a plane is a better answer there than a fitted one
  - *Off when:* wherever the published depth must stay the network's own measurement rather than a plane drawn over it
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
- **`scale_ceiling`** — number 0.5..20, default 5.0
  - *What:* the largest 1 / scale the law may be fitted to (the upper half of pepin.depth.A_BOUNDS); a law that lands on a bound prints AT BOUND (0.5..20)
  - *Default:* 5.0 — at 3.0 the law was a clipped constant once the lidar's plane was measured at its true 0.383 m — the fit saturated at a 3.00 with b pinned at -0.200 and stopped being a fit (scratch/lidar_height_fix_report.txt). Opened to 5.0 the same run fits a 2.80 in a, and the COLMAP control at 0.50-0.80 m reads 1.14 [0.90..1.24] against the clipped law's 1.32 [1.08..1.43]; the clipped law's tighter band against the beams (3.8 against 5.2 cm median) is luck, not fit. b still sits on its own bound, -0.200: the next one to question
  - *On when:* raise it above 5.0 only when a law reports AT BOUND in a and the mount height and the lens behind that law have been checked first
  - *Off when:* set it back to 3.0 to reproduce the clipped law in the field, side by side, with no restart
- **`imu_lean`** — bool, default on
  - *What:* the cart's lean (pepin.lean, from /imu/data_raw) is followed with the gyro as well as the accelerometer and carried into the scan's carry and the camera's place in the map; off, the floor plane leans with the accelerometer alone, as it always has, and nothing else is leaned
  - *Default:* on — on: the gyro's sign was verified by hand on 2026-09-13 (the cart tipped nose-down read pitch +10.5 deg, left-side-down read roll -9.4 deg, the fast path following at once with quality 0.9 while held; scratch/lean_tip_test.txt), and the gyro's zero offset is learned. Off, the floor anchor keeps the accelerometer-only lean, which by design ignores any tip shorter than 10 s
  - *On when:* after a hand tip through a known angle shows the reported lean following it the right way and returning to zero
  - *Off when:* the moment the lean in the report line disagrees with the cart's visible attitude
- **`lean_min_quality`** — number 0..1, default 0.5
  - *What:* how much of the lean gravity must have voted for (pepin.lean's quality, printed beside the lean in this line) before a frame is placed by it: below it the lean is treated as unknown and the frame is placed level (0..1)
  - *Default:* 0.5 — chosen on a simulation, not on the robot: in scratch/lean_quality_floor_probe.py a 0.2 deg/s gyro bias reports 3.0 degrees of tip on a level floor at quality 0.02 or less, nothing past 0.13 degrees of it survives a floor of 0.5, and a real 6 degree threshold climb keeps quality 1.00 throughout — so the floor costs the feature nothing. The 0.2 deg/s is hypothetical: this chip's worst measured axis is 0.074 deg/s (config/imu.json's level block). The same floor is declared in depth_fusion, so the pose and the scan gate make one decision
  - *On when:* raise it towards 1.0 on a robot that only ever leans when something real pushes it
  - *Off when:* 0 believes every lean, as before the floor existed: an A/B of the gyro's own drift

#### `goal_server`

- **`tf_pose`** — bool, default on
  - *What:* where no tracker answers, the cart's pose is read from TF (map -> base_link) and a goal is judged by how fresh that edge is; off, only the tracker is ever asked
  - *Default:* on — off, this node refused every goal of the first online-SLAM session — 'the tracker is not up' (2026-09-13 14:05), the goals driven by publishing /goal_pose by hand, which is Nav2 without a run, a tape or a verdict. In SLAM mode there IS no tracker: RTAB-Map owns the pose and pepin_bringup.slam_frame re-broadcasts its correction as map -> odom at 10 Hz, so 1.0 s without a transform is ten missed broadcasts, not jitter. The known-map modes are untouched: there the tracker answers first and its fit decides, exactly as before
  - *On when:* on in online SLAM, and anywhere else the pose is owned by something that publishes map -> base_link instead of a fit
  - *Off when:* to have a stack without a tracker refuse goals outright again — the old behaviour, and the honest one where a fit is the only evidence trusted
- **`correction_watch`** — bool, default on
  - *What:* where no tracker answers, the SLAM correction (/map_odom) must be arriving for a goal to start, and a drive is cut when it stops; off, the age of map -> base_link is the only evidence read
  - *Default:* on — map -> base_link is no evidence that the SLAM half is alive: slam_frame re-broadcasts the LAST correction at 10 Hz with a fresh stamp, so with the laptop shut down the edge is still 0.1 s old, the gate passes, and Nav2 — whose costmaps read that same edge against a 0.3 s tolerance — does not abort either. The cart would drive a map that stopped growing, on dead reckoning, with nothing to notice. The correction is a 10 Hz pulse whatever the graph does (pepin_bringup.rtabmap_frame publishes between optimisations too), so 2.0 s of silence is twenty missed messages over the bridge, not a hiccup
  - *On when:* in online SLAM, where the pose is owned by a machine on the other side of the bridge
  - *Off when:* when this node cannot hear /map_odom in a stack that is otherwise healthy — 'ros/go.sh where' prints 'correction_s' where one has ever landed, and prints none at all in that case; the drive then rests on the transform alone, as it did before

#### `laptop_localizer`

- **`global_watch`** — bool, default on
  - *What:* run the whole-map search once every watch_period_s and publish what it finds on /localization/candidate; off, this half of the node is a subscriber that costs nothing and the board is back to searching for itself only once it is already lost
  - *Default:* on — measured on the kidnap tape (run 0171, where the odometry jumps 1 m and 40 deg while the scans do not): the tracker's own window never recovered — 0.69 m of error still there after 39 s — and the board's own slow search found the truth four times (fits 0.76/0.72/0.79/0.75 against the tracker's 0.50-0.60) and died unconfirmed every time, because at a metre off this flat still fits 0.53, just under the 0.55 that declares the cart lost. This search costs 125 ms median (p90 165, max 258) against the board's 3700 ms, and with the shipped streak of 3 it brought the cart back in 2.9 s with 0 false re-seeds over the undisturbed tape
  - *On when:* whenever the map is a known one and the laptop is up
  - *Off when:* in SLAM mode: the map is RTAB-Map's there and still being built, so a whole-map search searches a map that changes under it
- **`watch_period_s`** — number 0.2..60, default 1.0
  - *What:* seconds between searches (0.2..60)
  - *Default:* 1.0 — it follows from the measured cost and the streak: one search is 125 ms median, 165 ms p90, 258 ms max of one core on this machine, so 1 Hz is 12-26 % of a core, and one candidate a second is what makes the shipped streak of 3 cost 2.9 s of recovery
  - *On when:* shorten it when recovery must be faster than three seconds and the laptop has the core to spare
  - *Off when:* lengthen it on a busy laptop, or on a map large enough that a search costs more than the measured 0.26 s
- **`watch_max_scan_age_s`** — number 0.1..3600, default 1.0
  - *What:* how long a revolution may sit in hand and still be searched, counted from when it ARRIVED here (0.1..3600)
  - *Default:* 1.0 — default by design, unmeasured as a number; the rule behind it is measured. /scan arrives at 9.8-10.8 Hz, so a healthy revolution is about 0.1 s old and one second is ten missed ones. The age is taken on this machine's monotonic clock and never from the stamp, because the board's clock runs 2.3-2.8 s ahead of the Mac's: a stamp-based age would either never fire or switch the watch off for good
  - *On when:* raise it when the bridge is slow but honest and candidates are being dropped as stale
  - *Off when:* a huge value is the old behaviour, which searched whatever was held — including a frozen scan, publishing the same answer again as if it were news
- **`camera_sources`** — list of: depth, contact, default depth,contact
  - *What:* which camera scans are matched here and sent to the board as pose measurements on /localization/measurement: the depth band, the floor-contact line; empty, nothing is matched and the board tracks on the lidar alone (any of: depth, contact, comma-separated)
  - *Default:* depth,contact — both, because fused they are what stays within 0.7/1.6/5.7 cm of lidar-only over run 0171 while neither carries the map alone (the depth band alone loses it in 0.5 s, the contact line in 12 s: scratch/camera_only_localization.py). Matching them HERE is the day's verdict: on the board the same pair cost 147 ms a scan, 4.7 Hz and 50 cm p90 of live error (scratch/drive_bisect.py, runs 0238-0241), and on this machine a match is a few milliseconds of a core that has nothing else to do
  - *On when:* whenever the camera is meant to help the pose — parked bumper to furniture, a blocked or dead lidar
  - *Off when:* empty is the switch that takes the camera out of the tracker's pose without touching the costmap layers, and the state to leave it in while the camera's own numbers are in doubt
- **`camera_match_hz`** — number 0.2..30, default 5.0
  - *What:* how often each camera source is matched and a measurement published (0.2..30)
  - *Default:* 5.0 — the cadence the offline replay fused at and the cadence the camera delivers: the depth pipeline runs at 9-11 fps and the contact scan beside it, and the replay that cost 0.7 cm fused every frame. 5 Hz per source is half of what arrives — two matches a frame period, a few ms each here — and it is what the board's own update rate can absorb without a measurement ever waiting longer than its carry is honest
  - *On when:* raise it towards the camera's own rate when the pose must follow the camera closely and this machine is idle
  - *Off when:* lower it on a busy laptop: the board fuses whatever arrives, and a measurement that comes at 2 Hz is still carried honestly to the update that takes it
- **`camera_window_m`** — number 0.01..1, default 0.09
  - *What:* half-width of the window a camera scan is matched in, metres, around the board's belief carried to that scan's moment (0.01..1)
  - *Default:* 0.09 — the tracker's own window (the board's 0.09 m), which is what the offline replay matched the camera scans in for its 0.7 cm: the camera REFINES a pose that the lidar and the odometry already hold to centimetres, it does not search for one. Wider is not better here — a +-40 degree fan has look-alikes a hand's width away that a full revolution does not
  - *On when:* widen it where the board's belief is poor and the camera is expected to pull it back — a long blind stretch, a lidar that has been off
  - *Off when:* narrow it to make a camera match cheaper and safer still; below the odometry's own error over a fifth of a second it stops being able to correct anything
- **`camera_window_deg`** — number 0.5..90, default 9.0
  - *What:* half-width of the same window in heading, degrees (0.5..90)
  - *Default:* 9.0 — the tracker's own 9 degrees, the replay's settings: a fan's heading is the one thing it measures well, and the belief it starts from is never more than a degree or two out while the lidar is alive
  - *On when:* widen it after a stretch on odometry alone, where the heading is what drifts
  - *Off when:* narrow it where the cart turns little and every degree of search is cost
- **`camera_min_fit`** — number 0..1, default 0.25
  - *What:* a camera match whose fit is below this is not sent: it is counted as low fit and the board never hears about it (0..1)
  - *Default:* 0.25 — 0.25 is the fit at which the tracker itself calls a scan weak (pepin.localization's lost_below): below it the scan explains nothing and its pose is the window's tie-break, not a measurement. It refuses only that much — the camera's fans sit at 0.5-0.6 against the lidar's map on run 0171, and the worst live camera-only fits of 2026-09-13 were 0.35-0.46. The covariance already widens a poor match a hundredfold at the bound; this floor is for what is not a match at all
  - *On when:* raise it to send only matches the map really explains — a room the camera sees badly, a map that has moved on
  - *Off when:* lower it to let the board's own disagreement gate do all the judging, which is what it is there for
- **`covariance`** — choice: peak, fit, default peak
  - *What:* how sure a camera measurement says it is: peak — the spread of that match's own score peak at the camera matcher's temperature (config/matcher.json); fit — the fit-scaled second moment of the whole surface, with the source's trust in it, that shipped before it. It is the number the board's information filter weighs the fan by (one of: peak, fit)
  - *Default:* peak — the fan's covariance decides everything the camera is allowed to do to the pose, and the fit-scaled one was never held against an error. On the peak path the scale is calibrated (T = 0.016, mean NEES 2.99 over 10047 lidar matches of the four goto tapes of 2026-09-13: scratch/peak_temperature.py) and reads 0.9-1.2 cm at a good fit, so the number a fan sends means something. It does not by itself weigh the camera down: both covariances shrink about sixfold together, and on the real matcher's own lattices a +-40 deg fan 5 cm off the truth keeps its share of the across-wall information — 29.8 % on fit, 34.8 % here, pulling the fused pose 17.4 mm of the 5 cm against 15.9 (scratch/peak_skeptic_fuse.py). The camera's own temperature is PROVISIONAL, the lidar's number: no camera scan is on those tapes, and until an operator records /localization/measurement against /tracker_pose and runs scratch/peak_temperature.py --camera, the source's trust (0.5) keeps widening the fan on top of its peak
  - *On when:* on: the board weighs the camera by a spread that means something
  - *Off when:* fit is what every tape before 2026-09-13 was recorded with, for an A/B; and the switch to reach for if a calibrated fan ever misbehaves in the field. Flip it TOGETHER with relocalizer's flag of the same name: a laptop on peak against a board on fit hands the same fan 75 % of the across-wall information and 38.9 mm of a 5 cm pull instead of 34.8 % and 17.4 mm (scratch/peak_skeptic_fuse.py)
- **`explained_vote`** — bool, default on
  - *What:* returns the map cannot explain (a person, a moved chair) do not score a camera match: the same vote the board's tracker takes on its own scans (relocalizer's explained_vote), taken here, on the grid the camera is matched against
  - *Default:* on — it is the board's own switch and it followed the match here: until 2026-09-13 these two fans were matched inside Localizer.update_from, which builds the vote from the static mask whenever explained_vote is on, and moving the matching to this machine took the vote off them silently. Measured on the furnished room with a person standing in the fan (scratch/camera_vote_probe.py): with 10 to 18 of the 41 beams on his legs, he moves the measured pose by 2.2 cm median and 2.5 cm at worst without the vote, and by 0.3 cm with it — a systematic pull that grows with how much of the fan he fills, replaced by a slide of a few mm. The worst voted case is 3.8 cm, a thinned fan sliding inside its own plateau, and the fan's sigma there is 8-11 cm, so the fusion already discounts it. The fans' floors are on the roster (vote_min_points 20): a mask that would leave a fan too thin to fix a pose is dropped and the whole scan votes
  - *On when:* in a room with people and furniture that moves — the room this robot lives in
  - *Off when:* to measure what the vote costs or buys the camera (A/B against the board's lidar-only pose), or in an empty room where every return should count

#### `neck_state`

- **`neck_tf`** — bool, default on
  - *What:* base_link -> camera_link is published live from the neck's encoders; the laptop's camera node must then run with ros/laptop.sh vslam --neck, or two nodes publish that edge
  - *Default:* on — the encoders are honest and their signs are checked by hand: the tilt reads 26 -> 103 degrees as the head goes down and the pan 0 -> -124 degrees to the left (config/neck.json's tilt_sign +1, pan_sign -1), 50 reads of a still head gave the same ticks every time, a read costs 9.7 ms, and the tick scale solved from the level frames is 1.067 true degrees per commanded degree, so 360/4096 stands (scratch/neck_tilt_scale.txt). At the reference pose the live transform equals the static one, so turning it on moves nothing until the head does
  - *On when:* whenever the head moves at all: with it off a turned head is a camera the map places where it is not
  - *Off when:* when the laptop broadcasts the static edge instead (camera_stream's static_camera_tf), or when the neck bus is suspect and a frozen edge is better than a wrong one

#### `relocalizer`

- **`rest_lock`** — bool, default on
  - *What:* hold the pose while the cart stands still (wheels quiet 0.6 s and the gyro under 1.5 deg/s): a match's residual is blended in with a time constant instead of taken whole
  - *Default:* on — the best-measured switch in the tracker. On tape 0182 (6 s at rest, a full turn, 8 s at rest) the published pose's rest band goes from 8.34 deg with the lattice alone to 0.51 deg with the rest lock, and on the robot it reads 0.42 deg at sd 0.13; while driving (tape 0170) it took p90 |yaw rate - gyro| from 7.74 to 4.86 deg/s and the correction's sd from 0.45 to 0.29 deg. It costs +0.04-0.08 ms a scan. Flipped live in the demo: 9 deg to 0.3
  - *On when:* always: a still cart whose pose wanders is the first thing a watcher sees
  - *Off when:* to show what the raw match does (the demo's A/B), or where the cart is carried by hand and the wheels have no say in whether it stands still
- **`explained_vote`** — bool, default on
  - *What:* returns the static map cannot explain (a person, a moved chair) do not score the match
  - *Default:* on — alone it took the rest band from 10.5 deg to 2.38, and with the rest lock and the sub-cell refinement to 0.51 (tape 0173). The mask is dropped when fewer than half the returns are explained or fewer than 60 come back, because a vote taken only on what already fits is mildly self-confirming
  - *On when:* in a room with people and furniture that moves — the room this robot lives in
  - *Off when:* in an empty room where every return should count, or to measure what a crowd costs the match
- **`rest_tau_s`** — number 0.1..60, default 6.0
  - *What:* the rest lock's time constant: seconds for a residual to die at rest (0.1..60)
  - *Default:* 6.0 — 6 s against the 3 s first tried: at the node's roughly 1 Hz rest cadence 3 s let 2.5x more match noise through, and 6 s halves that while still converging a nudge in seconds — the rest band reads 0.37 deg at 6 s
  - *On when:* lengthen it for a cart that stands for minutes and must not drift at all
  - *Off when:* shorten it when a nudged cart has to take its new pose quickly — a demo where the cart is pushed by hand
- **`rest_gain`** — number 0..1, default 0.05
  - *What:* the rest lock's share per match when no match cadence is known (0..1)
  - *Default:* 0.05 — default by design, unmeasured on its own: 0.05 against the driving gain of 0.5 was a guess, kept after the robot run because the rest bands it is inside of (0.42-0.51 deg) came out right. It has never been swept
  - *On when:* raise it towards the driving gain when the rest lock is too slow to accept a real correction
  - *Off when:* 0 lets no match move the pose while the cart stands: a hard hold, and a way to see how far the odometry alone wanders
- **`sources`** — list of: lidar, depth, contact, camera, default lidar
  - *What:* what corrects the pose: the lidar's revolution (/scan), matched here, and the camera (`camera`), whose scans the laptop matches and whose ANSWER arrives on /localization/measurement. The lidar drives the updates while it is fresh and the camera's word rides along, carried to its moment; a stale lidar hands the updates to the measurements. `depth` and `contact` name the camera's raw scans, which this node no longer subscribes to — enabling them changes nothing here (any of: lidar, depth, contact, camera, comma-separated)
  - *Default:* lidar — the lidar alone, because the camera cannot carry the map by itself: replayed on run 0171 against flat3 the depth band alone loses the map in 0.5 s (122 cm, 124 deg) and the contact line alone in 12 s (80 cm, 28 deg) — the camera's 0.15-1.3 m band is a different cross-section of the room than the lidar's 0.2 m map, so a look-alike place scores fit 0.90 at its own match and 0.12 at the truth. Fused with the lidar and gated on disagreement, all three together stay within 0.7/1.6/5.7 cm and 0.21/0.56/1.9 deg of lidar-only and never lose the map (scratch/camera_only_localization.py). `camera` is that same fusion with the matching moved to the laptop: on this board the raw scans took the tracker to 147 ms and 4.7 Hz and the live pose 50 cm p90 off the lidar's truth (scratch/drive_bisect.py, runs 0238-0241), while a measurement costs a matrix inverse
  - *On when:* add `camera` where the lidar is blocked or blind — parked bumper to furniture, or a lidar that stopped: the fusion is measured and gated, and the board pays nothing for it
  - *Off when:* drop a source the moment /localization/sources shows it disagreeing with the others; the lidar alone is the safe state, and it is what the board falls back to by itself when the link dies. `depth`/`contact` stay on the roster because the library still matches those scans where there is CPU for it — an offline replay (scratch/camera_only_localization.py), another robot — not because this board will
- **`measurement_max_age_s`** — number 0.05..5, default 0.5
  - *What:* how old a pose measurement from the laptop may be, in seconds, at the moment of the update that would take it: past this it is dropped instead of carried (0.05..5)
  - *Default:* 0.5 — the number the day of 2026-09-13 asked for: the camera's word pulled the live pose 50 cm p90 off the truth while the board matched at 4.7 Hz with 147 ms per scan, and every one of those measurements was fused as if it spoke for the moment it was used at. On the new path a measurement is 0.1-0.3 s old when an update takes it (a camera frame at 5 Hz plus the link), so half a second is the slack around that, not a threshold anybody has hit; the failure it is against — a bridge that stalls and delivers a burst — is seconds
  - *On when:* raise it only to see what a stale measurement does; the carry over odometry is honest for as long as the odometry is
  - *Off when:* lower it towards the measurement's own age (0.3 s) where the cart drives fast and a carry over a tenth of a second is already a decimetre
- **`remote_floor_xy_m`** — number 0..1, default 0.08
  - *What:* the least position sigma, metres, a measurement from the laptop is fused with, whatever its own peak claims; 0 takes the claim as it comes (0..1)
  - *Default:* 0.08 — measured 2026-09-13 with the camera recorded but not fused (scratch/camera_error.py, tapes 221822 and 221909): against its own band of the volume the camera's word was 8.5-10.6 cm off the lidar's truth at the median and 12-14 cm at p90, on both legs and both sources (depth, contact). 8 cm is the median; the self-check still inflates a source that scatters beyond its claim on top of the floor
  - *On when:* always while the camera's covariance is a peak at a provisional temperature: the floor is what its measured error says the word is worth
  - *Off when:* 0, to fuse the laptop's claim untouched: only to measure what a calibrated camera temperature does on a tape
- **`remote_floor_yaw_deg`** — number 0..90, default 5.0
  - *What:* the least heading sigma, degrees, a measurement from the laptop is fused with; 0 takes the claim (0..90)
  - *Default:* 5.0 — the same tapes: the camera's heading was 1.6-4.9 deg off at the median and 7-9 deg at p90, while a fan on one wall claimed 1.06 deg. Fused on that claim (22:08, sources lidar,camera) the pose spun 14-22 cm and 33-40 deg per update and the lidar's +-9 deg window could not find the truth back. 5 deg is the median of the worse source
  - *On when:* always, for the reason above
  - *Off when:* 0, only on a tape, never on the cart
- **`fusion`** — bool, default on
  - *What:* fuse every enabled source's word by its information — a match made here, a measurement made on the laptop; off: the widest source corrects alone and the others only report
  - *Default:* on — with all three sources the fused pose stays within 0.7-5.7 cm of lidar-only and never loses the map. One defect was found and fixed on the way: an edge-bound lidar used to be out-voted by a blind fan's plateau, so the anchor's bound is now taken alone — a 12 cm slip at rest is carried by the second match instead of held for 2.5 s, and recovery while driving is 3.2 cm against lidar-only's 2.9
  - *On when:* whenever more than one source is enabled
  - *Off when:* to see which source is actually moving the pose: off, the others still report
- **`covariance`** — choice: peak, fit, default peak
  - *What:* how sure a match says it is: peak — the spread of its own score peak at the matcher's calibrated temperature (config/matcher.json); fit — the fit-scaled second moment of the whole surface that shipped before it. Both the covariance the lidar's match is fused by and the one /tracker_pose carries (one of: peak, fit)
  - *Default:* peak — the fit-scaled numbers were never held against an error: the published sigma was a straight line from the inlier fraction (5 cm at a perfect fit, 35 cm at none). The peak's is calibrated — scratch/peak_temperature.py over the four goto tapes of 2026-09-13 (10047 matches off the window's edge, replayed against the lidar-only trace) solves T = 0.016 for a mean NEES of 3.00 (2.99 measured), and at that temperature a fit >= 0.7 match predicts 0.9/1.2 cm and 0.47 deg against an actual 0.72/0.77 cm and 0.40 deg. That, and only that, is the reason for the default: the sigma a match reports is the error it makes. It is NOT a reason to expect the camera to weigh less — both covariances shrink about sixfold together, and on the real matcher's own lattices (scratch/peak_skeptic_fuse.py, the furnished room of the unit tests) a +-40 deg fan 5 cm off the truth pulls the fused pose 17.4 mm here against 15.9 mm on fit, its share of the across-wall information going UP, 29.8 % to 34.8 %. The 0.24 mm of tests/unit/test_fusion.py is a synthetic lidar made 13 times sharper than the fan, where a real revolution is 3.5 times sharper. The balance is uneven too: position comes out 2-3x conservative and the heading optimistic (variance of error/sigma x 0.45, y 0.34, yaw 1.77)
  - *On when:* on: the sigma a match reports is the error it makes, which is what an information filter needs to weigh the camera against the lidar
  - *Off when:* fit puts back the numbers every tape before 2026-09-13 was recorded with — for an A/B against them, or if a calibrated covariance ever misbehaves in the field. Flip it TOGETHER with laptop_localizer's flag of the same name: the board weighs the laptop's fan against its own match, this path is about sixfold sharper in variance, and a board on fit with a laptop on peak hands the same fan 75 % of the across-wall information and 38.9 mm of a 5 cm pull instead of 34.8 % and 17.4 mm (scratch/peak_skeptic_fuse.py)
- **`self_check`** — bool, default on
  - *What:* every source vouches for itself: its covariance is widened by how far its answers fall from where its OWN previous answer, carried over the odometry, said they would (pepin.selfcheck). A source four times out in ALL THREE directions loses sixteen times its weight; the factor is that over-claim averaged over the three, so a source out in fewer of them loses proportionally less (a camera fan bound along a wall, four times out in the two directions it measures, is widened 9.7x not 16x — scratch/selfcheck_audit.py). One that is honest, or better, is not touched. Per source, never across sources: no lidar pose enters the camera's number and no camera pose the lidar's
  - *Default:* on — 2026-09-13: the camera's measurements claimed 25 cm from a linear formula over the fit (pepin.fusion.sigma_from_fit: fit 0.34 -> 24.8 cm) while nobody had measured how far apart two of its own answers fall a tenth of a second apart. On that claim they took 20-45 % of the fused weight and pulled the board's pose 0.8-1.5 cm off the lidar's, whose real error at fit >= 0.7 is 0.5-0.6 cm median against the replay truth (tapes 20260913_190024/190422, scratch/drive_bisect.py). A covariance nobody measured is a claim; this makes every source pay for its weight with its own repeatability. The ratio is a chi-square of 3 dof averaged over the last 20 measurements, so 1.0 is an honest covariance and the factor is capped at 25. One number for the whole matrix: an over-claim in one direction of three arrives divided by three (a depth source jumping 24 cm at rest against a 4 cm claim is widened 6x, not 36x), so the check takes back the over-claim a source's whole covariance carries, never a single direction's. The prediction it judges against pays for the odometry that carried it (pepin.fusion.odometry_covariance: 2 mm + 2 % of the distance, 0.05 deg + 70 % of the turn). Without that term the peak covariance made the check accuse the lidar itself: replayed over tape 20260913_190024 (scratch/lidar_selfcheck_replay.py) the lidar's own ratio ran at a median of 1.97 and a p90 of 6.23 while the cart moved and it was widened on 172 of the 307 moving updates — a false inflation of the one measurement this robot trusts. With it the same replay reads 0.59 median / 0.97 p90 in motion and 0.30 / 0.44 at rest, inflated on 25 of 2271 updates by at most 1.9x (23 of those in motion, by at most 1.09x). The 70 % is measured, not chosen: that tape's odometry turned 604 deg against the lidar's 359 (scratch/tape_odometry_error.py), the per-carry error's RMS is 0.77 of the reported turn and 0.70 with the lidar's own noise taken out — the wheels' 40-60 % in-place slip on this carpet, which is the odometry the tracker is left holding when the IMU drops (that tape carries no ekf and no imu at all). With the gyro alive it is some 2.5x conservative: on tape 0240_20260913_204114, which does carry ekf and imu, the same measurement is 678 deg of odometry against 716 of lidar and a per-carry RMS of 0.28, and the check there goes from 48 of 895 updates widened (max 1.31x) to 18 (max 1.10x)
  - *On when:* whenever more than one source is fused — it is the only thing standing between the fusion and a source whose covariance is a formula rather than a measurement
  - *Off when:* to measure what the check is worth on a tape (the ratios are still measured and printed with it off, so the A/B is one parameter set apart), or if a source that is known good is ever inflated by a real correction the odometry could not predict — a push by hand, a wheel slipping while the flag says the step was trusted
- **`local_fit`** — bool, default on
  - *What:* /localization_fit carries only a fit a scan of THIS machine measured: with no scan here at all — the camera's measurements driving the tracker alone — it carries 0.0, the value it holds before the first match, and the camera's own fit rides /localization/sources per source; off, the remote fit is published there as the tracker's own
  - *Default:* on — the number is read as 'how well the cart's own scan sits on /map' by everything downstream, and a remote one is neither. The camera's fit is measured on the laptop against /map_camera when depth_fusion publishes it (pepin_bringup.laptop_localizer), and depth_fusion paints that very band only while /localization_fit >= 0.50: published there, the camera's fit would bless the painting of the grid it was itself measured against, a circle no drift can break out of. The replay measures what such a fit cannot see: camera-only (split-no-lidar) sits 1.1 cm from lidar-only at the median, 25.1 at p90 and 43.4 at worst over run 0171, while the fits those same matches reported were 0.41 and 0.62 (scratch/laptop_localizer_replay.txt). 0.0 and not NaN because every gate downstream compares with `<` and NaN passes them all silently (pepin.watch.reported_fit)
  - *On when:* always on a cart that has a lidar: a fit nothing here measured stops the goal server and the volume rather than vouching for a pose
  - *Off when:* to drive on the camera alone — a dead lidar, a lidar-less robot — where the laptop's fit is the only word there is; watch /localization/sources for the drift it cannot report
- **`toe_reach`** — number 0..0.6, default 0.27
  - *What:* how far past the leg the lidar sees a standing person's toe reaches, metres: the term the dynamic rings are sized on (pepin.dynamic.berth_for). The default is computed from the lidar's mount (config/lidar.json) (0..0.6)
  - *Default:* 0.27 — one measured number and three assumed ones: the mount is 0.383 m by tape, and the reach is 0.21 + (z - 0.07) tan 10 deg — a 28 cm shoe whose ankle sits 7 cm back, a shin leaning 10 degrees — typed anthropometry that has never been measured against a person in front of this cart. It matters in metres: a point planner's ring is 0.41 m at the 0.20 that stood here before and 0.48 m at this 0.27. The flag exists because the cart once ran over feet
  - *On when:* raise it for boots, or for a cart that must give more room: every dynamic ring widens by the same amount
  - *Off when:* lower it to compare berths in the field without a restart; 0 rings only what the beams themselves see
- **`near_rings`** — bool, default on
  - *What:* a return is ringed as soon as it clears the cart's own outline, and only the marks that would land on that outline are dropped; off, nothing within the ring plus the outline is ringed at all — the older rule, whose blind disc grows with the ring
  - *Default:* on — exact geometry, no field A/B of the two rules. The old rule blanks a disc of ring + 0.457 m (the cart's circumscribed radius plus one costmap cell), so at the ring today's reach asks for, 0.48 m, a person standing 0.90 m ahead would not be ringed at all — the very case the ring exists for. The new rule trims only the marks that land on the cart's own outline, which is what the run-0087 failure actually was: a mark on itself that refuses its every command
  - *On when:* wherever a person may come within a metre of the cart — the close approach this robot is built for
  - *Off when:* to reproduce the older rule side by side; remember its blind disc grows with the ring (7 cm of extra ring stopped a person at 0.90 m from being ringed at all)
- **`accept_candidates`** — bool, default on
  - *What:* re-seed from the laptop watchdog's whole-map candidates (/localization/candidate, pepin.watchdog): a place that disagrees with the tracked pose candidate_streak times in a row, about the same place each time, is adopted through the path the board's own search uses
  - *Default:* on — measured on the kidnap tape (run 0171: the odometry jumps 1 m and 40 deg while the scans do not) this is the difference between coming back in 2.9 s over 28 scans and never coming back — the tracker's own window still had 0.69 m of error after 39 s, and the board's own searches found the truth four times (fits 0.76/0.72/0.79/0.75 against the tracker's 0.50-0.60) and died unconfirmed each time, because at a metre off this flat still fits 0.53, just under the 0.55 that declares the cart lost. Over the undisturbed tape it re-seeded 0 times, and against another flat's map 9 of 12 candidates were called unknown_map
  - *On when:* whenever the laptop's watchdog runs and the map is the right one
  - *Off when:* where a teleport is more dangerous than being lost — under a live goal, or in a room the map does not cover: off, the candidates are still judged, counted and reported
- **`candidate_streak`** — integer 1..10, default 3
  - *What:* how many candidates in a row must disagree with the tracker and agree with each other before one of them re-seeds it: the price of a teleport, in seconds (1..10)
  - *Default:* 3 — deliberate conservatism above a measurement that was neutral: on the kidnap tape a streak of 1 recovered in 0.8 s (8 scans) and this streak of 3 in 2.9 s (28 scans), and both re-seeded 0 times over the undisturbed tape, where the pose never left the reference by more than 0.000 m. Nothing measured prefers 3; the argument is that a look-alike keeps looking alike, so one agreement is not proof
  - *On when:* raise it in a room of look-alike corners, where a wrong teleport costs more than three seconds of being lost
  - *Off when:* 1 is the fastest recovery measured (0.8 s) and on that tape just as safe — the value to try when a demo has to show the cart coming back
- **`map_topic`** — choice: map, map_lidar, default map
  - *What:* which map this tracker matches on: /map, whatever the stack's map owner publishes there (the served pgm in split and vision mode), or /map_lidar, the lidar layer of the laptop's fused volume (pepin_bringup.depth_fusion, flag lidar_map) (one of: map, map_lidar)
  - *Default:* map — map is the default because the volume is unmeasured on the moving robot and because of one known cost: the volume's grid is 280x250 cells and the served map 239x215, so a tracker on /map_lidar answers to another map id, and the laptop's candidates and camera measurements — stamped with the id of /map (pepin_bringup.laptop_localizer) — are refused as evidence about another map until that half moves too. Offline a SEEDED volume is the same map: its slice agrees with flat3_straight.pgm on all 18274 cells that map knows, and the four tapes of 2026-09-13 replayed on the exported slice give live error medians 0.6/0.5/1.3/0.7 cm against the file's own 0.6/0.5/1.3/0.6 (scratch/volume_vs_pgm.py, scratch/drive_bisect.py --map). An UNSEEDED volume is not: today's live snapshot holds 52.1 % of the saved map's walls
  - *On when:* map_lidar to drive on the room as it is now — the volume carries what the cart has seen since the file was frozen, and it hardens where the cart drives
  - *Off when:* map wherever the laptop's watchdog and camera measurements must be believed, and wherever the laptop may go away: /map is served on the board and the volume is not
- **`map_refresh_s`** — number 0..600, default 0.0
  - *What:* the least time between two adoptions of the map topic: a newer map on the topic in use is taken only after this many seconds AND only if its cells changed. 0 takes the first map and no other, which is what a served file has always done (0..600)
  - *Default:* 0.0 — the cost is the measured one: adopting a map rebuilds the correlative matcher, the static mask and the tracker and forgets the episode's candidates and measurements — the whole-map lattice alone is 15 s on these four A53 cores — while /map_lidar is republished at the fusion's map_hz, once a second. 0 is the old behaviour exactly: the served map arrives once, latched, and is adopted once
  - *On when:* 30-60 s with map_topic map_lidar in a room being mapped as it is driven: the tracker then follows the volume as it hardens, at one rebuild a minute
  - *Off when:* 0 for a frozen map, and any time a rebuild mid-drive would cost more than a stale map does
- **`carry_candidates`** — bool, default on
  - *What:* a candidate's pose is moved from the moment of its own scan to now over the odometry between the two stamps (pepin.watchdog.carried) before it is judged and fused, and one the odometry history no longer covers is dropped
  - *Default:* on — by argument from a measured latency, not by a measured gain: a whole-map search takes 0.12-0.25 s plus a wireless hop, so at 0.8 m/s an uncarried pose is installed about 20 cm backwards along the drive every time — a bias, not noise. On the kidnap tape the carry changes nothing measurable (2.9 s, 28 scans, 0 false re-seeds either way), because that cart was barely moving when it was lost
  - *On when:* whenever the cart may re-seed while driving
  - *Off when:* only to reproduce the old behaviour, where the pose the laptop measured a search and a hop ago is installed as the pose now
- **`distinct_scans`** — bool, default on
  - *What:* a streak is counted in scans, not in messages: a candidate whose scan id is already in the run is a second opinion that heard the first one's scan, counted as replay and not lengthening the streak
  - *Default:* on — the failure it answers is real: a frozen /scan on the laptop published the same search answer once a second and the board counted three of them as three seconds of evidence — the same replay that fooled the board's own two-search rule on 2026-09-09. On the kidnap tape it costs nothing (2.9 s, 28 scans unchanged). A sender that names no scan says id 0, and a repeated 0 reads as replay too
  - *On when:* wherever the candidates cross a bridge that can freeze — which is this robot's
  - *Off when:* only to reproduce the old counting, where one scan's answer repeated could re-seed the tracker

#### `rtabmap_frame`

- **`slam`** — bool, default off, not live
  - *What:* RTAB-Map is the map (online SLAM): its correction is map -> odom and goes to the board as a message on /map_odom, where pepin_bringup.slam_frame broadcasts it; off, the board's tracker owns map -> odom and this node broadcasts map -> rtabmap here (not live: set at the next start)
  - *Default:* off — default by design, unmeasured: this says which edge is published — a mode, not a tunable — and the two modes are two different graphs of frames, which is also why it is not live. What the mode is worth was measured in the first session: from an empty database a room came up as a 341x341 map over 21 and then 55 graph nodes, a 1 m goal with a 90 degree turn landed within 2.8 cm and home within 6.6 cm after about 4 m of driving, one loop-closure hypothesis was rejected by the scan check (5 % against the 10 % it needs) and none was accepted
  - *On when:* in an unknown room, launched as one mode end to end (ros/thin.sh slam on the board, ros/laptop.sh vslam --slam): set at start, never mid-run
  - *Off when:* in every known-map mode, where the board's tracker owns map -> odom: the two publishers must never both run

## Build and run (on the board)

```bash
# from the laptop: copy ros/ to the board and build the image (15-30 min the first time)
rsync -a --delete ros/ root@pepin.local:/root/pepin-ros/
ssh root@pepin.local 'cd /root/pepin-ros && docker build -t pepin-ros .'
# on the board: sensors + bridges + foxglove_bridge
ssh root@pepin.local '/root/pepin-ros/run.sh ros2 launch pepin_bringup robot.launch.py'
# on the board, second terminal: navigation on a saved map
ssh root@pepin.local '/root/pepin-ros/run.sh ros2 launch pepin_bringup nav.launch.py map:=/maps/lap3.yaml'
```

Laptop: install Foxglove Studio (`brew install --cask foxglove-studio`), open a connection
to `ws://pepin.local:8765`, add the 3D panel with `/map`, `/scan`, `/tf`, the costmaps and
`/plan`; send a goal with the "Publish" panel on `/goal_pose` (`geometry_msgs/PoseStamped`,
frame `map`).

## At boot

`board/pepin-ros.service` starts the sensors container (`robot.launch.py`) after the base and ToF
servers; Nav2 (`nav.launch.py`) is started on demand inside it. Build or rebuild the image with
`ros/build.sh` (syncs `ros/` and `src/pepin` to the board).

## Bring-up checklist (in this order, each step visible in Foxglove)

1. `robot.launch.py` alone: `/scan` at ~10 Hz (the hull box filter's output), `/odom` at 20 Hz, TF `odom -> base_link -> laser`.
   Push the cart forward by hand: `/odom` x grows. Turn it left: theta grows.
2. Laser orientation: a wall in front of the cart must draw at +x in `base_link`. Our lidar
   is mounted upside down and the LD19 counts angles clockwise; the static transform (roll pi,
   yaw -87.5 degrees) is read from `config/lidar.json` by the board's launch and by the
   laptop's camera node alike — there is no launch argument for it. If the scan comes out
   mirrored left/right, fix `roll_deg` in that file and `ros/sync.sh`.
3. `nav.launch.py`: AMCL converges on the map after a few metres of teleop (or set the initial
   pose from Foxglove); then a goal.
4. Memory: `free -m` on the board while navigating; the container must stay under ~700 MB.

## Redundancy demo

Both sensors see the room; either one alone is a mode the robot can be driven in, switched while
it runs. `ros/sensor.sh` is one command per sensor, because a sensor has two ends and they must
move together: what the tracker matches against the map (the relocalizer's `sources` flag) and
what writes into the costmaps (`lidar_layer`, `camera_layer`, `contact_layer`, on the local
**and** the global costmap). Half a switch — the layer off, the tracker still matching on it —
is not a mode, it is a bug that looks like one.

| mode | command | tracker sources | costmap layers |
| --- | --- | --- | --- |
| fused | `ros/sensor.sh lidar on` + `ros/sensor.sh camera on` | `lidar,camera` | `lidar_layer`, `camera_layer`, `contact_layer` |
| lidar only | `ros/sensor.sh camera off` | `lidar` | `lidar_layer` |
| camera only | `ros/sensor.sh lidar off` (`--hard` to stop the driver) | `camera` | `camera_layer`, `contact_layer` |

The tracker column needs the relocalizer's `sources` flag, which arrives with the fusion wiring;
the generated **Feature flags** table above is the authority on whether this build carries it
(`ros/flags.sh list relocalizer` says the same about the running node). Until it does, every
`ros/sensor.sh lidar|camera on|off` applies the costmap half, reports that the tracker's sources
are unchanged and exits 1 — so the costmap column is what holds, and the modes are half modes.

The camera is one sensor read twice from the same frames: `depth_scan` is the band 8 cm-1.3 m
above the floor (table tops, seats, a hand) and `contact_scan` is where the floor ends (chair
feet, a plinth) — so `camera on` moves two costmap layers at once. In the tracker it is ONE
source, `camera`, and the difference is where the matching happens. **The camera's scans are
matched on the laptop**, where the depth network already runs (`pepin_bringup.laptop_localizer`):
each scan is matched in a small window around the pose the board believes in, carried to that
scan's own moment, and what crosses the link is the pose it measured with its covariance
(`/localization/measurement`, `pepin.measurements`). The board carries that pose to its next
update and fuses it by information beside the lidar's match, which costs it a 3x3 inverse.

That is the day's verdict, measured on the robot (2026-09-13, `scratch/drive_bisect.py`, runs
0238-0241): with the camera's scans matched ON THE BOARD the tracker took 147 ms per revolution
instead of 45, kept only every second revolution (4.7 Hz), and the camera's word — measured on a
scan that was a fifth of a second old by the time it was used — pulled the live pose 50 cm p90
and 78 cm max off the lidar-only truth over one drive. The same fusion offline, at full rate,
costs 0.7 cm. The arithmetic was never the problem; the Orange Pi was.

**No WiFi, no camera measurements.** The board then tracks on the lidar exactly as before — the
same code path, one source instead of two — and `ros/sensor.sh status` shows `camera` stale in
the tracker's roster. That is the redundancy the split buys: the link may die, and the robot
keeps its pose; the lidar may die, and the camera's measurements drive the updates by themselves
(the board's own report line says `fused camera` and the watch goes off, because a +-40 degree
fan cannot say "lost").

```bash
ros/sensor.sh status            # what the tracker matches on, which layers are on, what each node last said
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

- the 3D panel's `/scan` and `/depth_scan` beside `/local_costmap/costmap` and
  `/global_costmap/costmap` — switching a layer changes the grid within a costmap cycle, and the
  marks that remain tell you which sensor drew them;
- the `scan-to-map fit` plot (`/localization_fit`, 0..1) — the tracker's own score of the scan it
  matched. This is where camera-only localisation fails visibly; with no scan of the board's own
  at all (the camera's measurements driving alone) the plot reads 0.0 by design, because a fit the
  laptop measured against the camera's own band of the volume is not this board's word about
  `/map` (`local_fit`) — the camera's fit is in `/localization/sources`, per source;
- the Log panel (filtered to `relocalizer`, `controller_server`, `planner_server`, ...).

The report lines say the same in words, every 30 s, and `ros/sensor.sh status` prints all three:

- `relocalizer` on the board: `tracker: ... flags: rest_lock=on explained_vote=on ...`, with
  `sources=lidar,camera` among those flags (`ros/sensor.sh status` prints the sources from this
  line, and says so plainly when the line has none), and `measurements N (received N, taken N)`
  with every reason a camera pose was refused — stale, uncovered by the odometry, from another
  map
- `laptop_localizer` in the laptop's SLAM container: `measurements: depth N at fit 0.62,
  contact N at fit 0.55 (against /map), ... rejected: no belief 0, stale belief 0, ...` beside
  the whole-map watchdog's own candidates, and `ms median/max` for both
- `depth_stream` in the same container: `depth: 3.1 frames/s published ...`
- `contact_scan` in the same container: `contact: 2.9 scans/s published ...`

### What the numbers already say

Camera-only localisation on the **lidar's** map does not work, and was measured before it was
demonstrated (`scratch/camera_only_localization.py`, run 0171 replayed offline against
`ros/maps/flat3_straight.yaml`; the same modes replayed through the new split path in
`scratch/laptop_localizer_replay.py`):

| tracker sources | error vs lidar-only (median / p90 / max) | verdict |
| --- | --- | --- |
| `depth` | 122 cm / 124 deg at the loss | loses the map after 0.5 s |
| `contact` | 80 cm / 28 deg at the loss | loses the map after 12 s |
| `depth,contact` | 29 / 52 / 90 cm | loses the map |
| `lidar,depth,contact` | 0.7 / 1.6 / 5.7 cm, 0.21 / 0.56 / 1.9 deg | never lost |

And the split path against the same stick, on the same tape
(`scratch/laptop_localizer_replay.py`, 2026-09-13: 5 Hz per camera source, a 50 ms uplink for
the board's pose and a 100 ms link for the measurement, the shipped 0.5 s age gate):

| where the camera is matched | error vs lidar-only (median / p90 / max) | matches per board update |
| --- | --- | --- |
| on the board (`lidar,depth,contact`) | 0.8 / 1.9 / 7.5 cm | 2.91 |
| **on the laptop (`lidar,camera`)** | **0.7 / 1.5 / 3.1 cm** | **1.00** |
| on the laptop, lidar off at t+25 s | 1.1 / 25.1 / 43.4 cm | 0.70 |

The accuracy is the fused tracker's — slightly better at the tail, because a measurement is
carried to the update that uses it instead of being matched from a scan that was carried — at
the lidar-only cost on the board. One camera match costs 12.7 ms median on the laptop, and the
link carried 348 measurements over the 45 s run with none refused as stale. Live on the board,
where it was the other way round, the recorded pose of run 0238 sits 2.5 / 50.5 / 78.7 cm from
the same truth; that is the number this split exists to remove, and only the robot can confirm
it did.

The camera's band is a different cross-section of the room than the lidar's own plane:
sofa cushions and table clutter fit "some wall" well (fit 0.90 at the wrong pose, 0.12 at the
true one), and parked bumper-to-furniture the camera sees nothing of the floor below ~1.2 m.
Camera-only localisation needs a camera-built map, not the lidar's slice — which is what
`/map_camera` is for: where the world volume is the map (`pepin_bringup.depth_fusion`,
`map_source=volume`), the camera's own band of it goes out on that topic and the laptop matches
the camera's scans against THAT instead of against the lidar's plane. So the honest demo is:
**the costmap half survives either sensor alone; the tracker half needs the lidar** — and fusing
the camera in costs 0.7 cm of median agreement, which is free *as long as nobody asks the board
to do the matching*. `ros/sensor.sh lidar off` prints that warning itself when it takes the last
lidar out of the tracker's sources.

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
