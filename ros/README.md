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

(`/map` can also come from the fused volume in either mode — see [The world map](#the-world-map);
it is still exactly one publisher, chosen by the mode.)

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

Goals work with no places book: `ros/go.sh -1.0 0.3 90` drives to map coordinates, and a click in
Foxglove (Publish → `/goal_pose`, frame `map`) does the same. The saved map's places are not
offered — they are coordinates in a frame this new map does not share — and the session's own book
(`/maps/slam.places.yaml` on the board) stays empty: `ros/go.sh mark` and `ros/go.sh where` read
the pose from the tracker's `/where_am_i`, and in SLAM mode there is no tracker, so both answer
without one. Mark places after the map is saved and the board is back on it.

Watch the map grow with the `ros/foxglove/pepin_slam.json` layout at `ws://localhost:8765`: `/map` under the fused surface, the graph's path, the head camera.

To go back to driving a saved map: `ros/thin.sh vision` (which leaves SLAM mode), then
`ros/mode.sh nav /maps/flat3_slam.yaml`.

## Camera calibration

The neck camera's optics were a guess: one field-of-view number (78 deg, fitted against the lidar
on 2026-09-10) standing in for four — `fx`, `fy`, `cx`, `cy` — and a lens that bends straight
lines and was not modelled at all. A checkerboard measures all of it in one sitting.

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
9x6 ..., rms 0.31 px, 70.6 deg wide` instead of `optics: nominal 78 deg field of view
(uncalibrated)`. The `undistort` flag publishes a rectified picture (`ros/flags.sh set
camera_stream undistort true`); it is off until the straightened picture has been measured against
the raw one on the robot. A calibration that turns out bad is switched off with one boolean —
`calibrated: false` — and the block stays in the file as history, the nominal 78 deg pinhole
coming back untouched.

**Re-fit the tilt afterwards.** `mount.pitch_deg` (26 deg) was not measured on its own: it came
out of the same lidar fit as the 78 deg field of view, and in that fit the two traded against
each other (78/26 with a 1.2 % residual, the nominal 70/28 with 3.0 %). Pinning `fx` with a
checkerboard therefore leaves the tilt fitted against a focal length that is no longer in use —
the run says so when it writes. Re-fit `mount.pitch_deg` with the measured `fx` held fixed
(`scratch/depth_fit_models.py`) before trusting what `depth_stream` projects; `camera_stream`
broadcasts the same number as the static `base_link -> camera_link` edge, so it moves the whole
camera in TF too.

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
  board, so only the *value* is late, never the lookup.
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
  layer with `seed_map:=/maps/flat3_straight.yaml` — and an unknown one an empty volume. There is
  no mode switch between them, and the map keeps growing either way.

```bash
ros/laptop.sh vslam --world-map            # /map comes from the volume instead of RTAB-Map's grid
ros/flags.sh set depth_fusion map_source file    # back to the old behaviour, live
ros/flags.sh set depth_fusion no_return_free true # beams with no return carve an open door
ros/flags.sh set depth_fusion lidar_layer false  # the volume goes back to being the camera's alone
ros/flags.sh set depth_fusion snapshot_s 30      # write ros/maps/world_live.npz twice a minute
```

Offline, `WorldMap.export_pgm_yaml` writes the map_server pair every existing tool already reads
(`ros/mode.sh nav /maps/NAME.yaml`, `pepin.mapping.grid_from_pgm`), so a volume can be frozen into
a file exactly like `ros/map.sh save`.

**The next step, not built:** re-fusing after a loop closure. A TSDF cannot be un-integrated, so a
graph correction leaves the old geometry standing. The cure is to replay the frames at their
corrected poses, and the snapshot already carries the index for it — every integration's stamp,
sensor and pose — while the measurements themselves stay in the run tape, where they already live.

## The whole-map watchdog

The board's tracker follows the cart in a 9 cm window around the odometry's prediction. When it
loses the world it searches the whole map for itself — FFT correlation over every shift at 40
headings, **3.7 s** of an A53 (2026-09-06) — but only after the fit has been poor for three
checks in a row, and never while the cart is moving: a teleport mid-drive is worse than a poor
fit. The laptop runs the very same search in **0.12 s** and has nothing else to do with it, so
`pepin_bringup.global_watch` asks the question once a second, healthy or not, and sends the
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
the revolution it was computed on, and the map's identity. The board judges it again against its
own fresher pose (`pepin.watchdog.judge`) — **agree** (the everyday verdict), **disagree** (another place,
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
`accept_candidates`, `candidate_streak` and `distinct_scans` on the tracker. All six are live; with `global_watch` or `accept_candidates` off the stack is exactly what it
was before, the board's own slow search and nothing else.

### Not yet verified on the robot

- Nothing here has run on the robot: the numbers above are a replay of a recorded tape.
- The link's own latency is modelled as 50 ms; a candidate that arrives late is judged against a
  pose that has moved on, which can only turn a `disagree` into a `nothing` (the streak breaks),
  never the other way round.
- 3 of 12 candidates computed against a *wrong* map still read `disagree` rather than
  `unknown_map`: the fit floor (0.45) and the ambiguity ceiling (0.90) are the two numbers that
  decide it, and they are tuned on one flat.
- A re-seed while a goal is running is refused (`_navigating`); a re-seed while the cart is
  merely driving is allowed, which no drive has tried yet.

## Feature flags

Every behaviour that can be switched is a live parameter of the node that owns it, declared
once in that node's `FLAGS` table (`pepin.flags`, declared to ROS by
`pepin_bringup.node_kit.Switches`) with its kind, default and description, checked by kind on
every change, and printed in the node's report line (CLAUDE.md rule 19). `ros/flags.sh list
[NODE]` shows them with their current values, `ros/flags.sh set NODE FLAG VALUE` changes one
live (a value the flag refuses is refused with the reason, before any host is touched),
`ros/flags.sh get NODE FLAG` reads one; the script execs into whichever container the node
runs in. A change lives until the node restarts; a default changes in the table. This table is
generated by `ros/tools/flags_doc.py` (a unit test keeps it current).

| node | flag | kind | default | live | description |
| --- | --- | --- | --- | --- | --- |
| `camera_stream` | `scale` | number 0..1 | 0.5 | yes | the published picture as a fraction of the camera's own 1280x720, its optics scaled with it: features for place recognition do not need 720p, and a reliable 2.7 MB frame nine times a second is a cost with no return; a change takes the next frame |
| `camera_stream` | `undistort` | bool | off | yes | the published picture is rectified with the checkerboard calibration (config/camera.json's intrinsics) and its camera_info then says no distortion; off by default until the straightened picture has been measured against the raw one, and a no-op while the camera is uncalibrated, since there is nothing to undo. Rectifying crops to the largest all-valid rectangle, so the field of view narrows a little |
| `camera_stream` | `static_camera_tf` | bool | on | at start | base_link -> camera_link is broadcast from here; it goes off (ros/laptop.sh vslam --neck) when the board's neck node publishes that edge live from the servo encoders (neck_state, flag neck_tf), because two publishers of one edge fight. Not live: a static transform cannot be withdrawn once sent |
| `contact_scan` | `contact_scan` | bool | on | yes | the contact line is published; off, the node is a subscriber that costs nothing — the costmap's own contact_layer.enabled is the other end of the same demo switch, and either one alone takes the camera's floor line out |
| `contact_scan` | `shadow` | bool | on | yes | the last floor pixel on a face stands a band's width UP that face, so its ray lands past the foot: on (measured), that width is taken back off the range (pepin.contact.band_shadow); off is the raw boundary ray |
| `contact_scan` | `max_range` | number 0.1..10 | 2.0 | yes | metres past which a column is called clear instead of ended; the default is where the floor is still the floor, not where the optics run out, and the costmap's contact_layer.obstacle_max_range must match it |
| `depth_fusion` | `enabled` | bool | on | yes | frames are fused into the model; off, they are dropped |
| `depth_fusion` | `fit_gate` | bool | on | yes | frames are fused only while the tracker reports /localization_fit >= 0.50; off, every frame is fused: SLAM mode, where RTAB-Map owns the pose and no tracker speaks |
| `depth_fusion` | `self_heal` | bool | on | yes | a streak of frames refused at the alignment bound empties the model, so it re-seeds from the next frame instead of staying frozen until a human resets it |
| `depth_fusion` | `align` | bool | on | yes | frame-to-model: a frame's lidar-height band is turned about the cart to fit the model before it is fused, and a frame whose best turn is the search's bound is refused |
| `depth_fusion` | `min_weight` | number 0..100 | 2.0 | yes | observations a voxel needs before it is shown in /fusion/surface (the debug cloud only: /map has map_min_weight) |
| `depth_fusion` | `map_min_weight` | number 0..20 | 2.0 | yes | observations a voxel needs before it speaks in /map. Its own flag, and capped at the lidar's own weight cap: a value above that leaves every cell of the map unknown, and the cart drives on this one |
| `depth_fusion` | `surface_hz` | number 0.1..10 | 1.0 | yes | how often /fusion/surface is published (the crossing search costs a fraction of a second) |
| `depth_fusion` | `band_half_z` | number 0.02..0.5 | 0.125 | yes | half the height band around the lidar's plane a frame is seated on, metres (config/fusion.json's band_half_z_m is the default); the band's centre is the plane the published base_link -> laser edge names, and both are printed in the report line |
| `depth_fusion` | `lidar_layer` | bool | on | yes | /scan is integrated into the volume at the lidar's plane (rays carve free space, returns mark a surface); off, the volume is the camera's alone, as it was |
| `depth_fusion` | `no_return_free` | bool | off | yes | a beam that came back with nothing carves free space out to the sensor's reach (an open door reads as open); off, it writes nothing at all, because a mirror, a black chair leg and anything closer than the minimum say the same nothing |
| `depth_fusion` | `map_source` | choice: file, volume | file | yes | where /map comes from: the saved file another node serves, or the volume's own lidar layer published from here at map_hz. Only where the stack was launched with world_map:=true; anywhere else volume is refused, because another node is on /map |
| `depth_fusion` | `map_hz` | number 0.1..5 | 1.0 | yes | how often the volume's layer goes out as /map when map_source is volume |
| `depth_fusion` | `snapshot_s` | number 0..3600 | 60.0 | yes | how often the volume is written to world_path (0: only at shutdown) |
| `depth_fusion` | `resume_volume` | bool | on | at start | a volume snapshot at world_path is loaded at start, so a known room is a resumed volume; off, the volume starts empty and grows from the sensors |
| `depth_stream` | `edge_filter` | bool | on | yes | flying pixels at object edges are dropped from the published depth and the scan; the law's beam pairs skip them regardless |
| `depth_stream` | `lidar_anchor` | bool | on | yes | the lidar fits the depth's law; off, the last law is held (the failure mode of a lidar that stops) — with no law yet nothing is published until it is back on |
| `depth_stream` | `floor_pairs` | bool | off | yes | the floor's pixels pair the network's depth with the plane's geometric depth, a second hoop for the law that needs no lidar; off by default: measured, it pulls the law off the lidar's row (1.4x too far there), where the costmap lives |
| `depth_stream` | `wall_anchor` | bool | off | yes | the lidar's returns extruded up the image while the network's depth stays continuous pair the rows above the lidar's with the wall's depth, a third hoop; off by default: measured, it puts the lidar's row 10 % too near while fixing the rows above |
| `depth_stream` | `parallax_anchor` | bool | off | yes | the corners this frame shares with the previous one, triangulated against the odometry's transform between the two stamps (pepin.parallax), pair the network's depth with a depth in metres measured by the cart's own movement — a hoop that needs no lidar and no assumed plane and that lands at every elevation the picture has; off by default: measured offline on runs 0171 and 0165 it costs 3-5 ms and gives 30-190 pairs where the cart really stepped, but at those runs' 2-3 cm baselines the depth is +25-37 % too far under 1.5 m (19-30 samples a run) and unbiased from 1.5 to 3 m — a range-dependent bias the odometry's own +-25 % scale band cannot explain, cause not yet known — and it yields nothing at all while the cart stands still or turns on the spot |
| `depth_stream` | `affine_law` | bool | on | yes | the network's depth through 1 / z = a / D + b, fitted on the pooled pairs; off, the raw network's depth goes out unwithheld (1.5-2x too far: an A/B measure of the correction, never a way to drive) |
| `depth_stream` | `ray_law` | bool | off | yes | the law's scale follows the ray's angle off the optical axis, a / D + b fitted per elevation (pepin.elevation) instead of one pair of numbers for the whole picture; off, the affine law's image stands. Needs wall_anchor on as well: the lidar's own beams put a return's elevation on a curve of its range, so on them alone the angular fit is refused and this stage is the affine law. A property of the camera and the network, so the neck may tilt without refitting; off by default until it is measured on the robot (scratch/ray_law_eval.txt: held out, it tightens the beams' scatter on three drive halves of four and moves the median 5-10 % near) |
| `depth_stream` | `wall_correct` | bool | off | yes | after the law, the pixels the wall walk covered are set to the extruded plane's depth outright; off by default (the same walk as wall_anchor, applied instead of fitted) |
| `depth_stream` | `floor_anchor` | bool | on | yes | pixels within centimetres of the floor plane snap to it in the published image (the scan is built before it); the plane leans with the cart, from the IMU's up vector |
| `depth_stream` | `depth_backend` | choice: remote, local, auto | local (env PEPIN_DEPTH_BACKEND) | yes | where the network runs: local (the CPU model in this container), remote (the laptop's GPU service, ros/depth_host.sh), auto (the service while it answers, the CPU model while it does not) |
| `depth_stream` | `scale_ceiling` | number 0.5..20 | 5.0 | yes | the largest 1 / scale the law may be fitted to (pepin.depth.A_BOUNDS' upper half): raising it from the 3.0 the fit used to saturate at is what stopped the law from being a clipped constant once the lidar's plane was measured. Set it back to 3.0 to compare the two laws in the field; a law that lands on a bound prints AT BOUND |
| `global_watch` | `global_watch` | bool | on | yes | run the whole-map search once every watch_period_s and publish what it finds on /localization/candidate; off, this node is a subscriber that costs nothing and the board is back to searching for itself only once it is already lost. Off in SLAM mode: the map is RTAB-Map's there and it is still being built |
| `global_watch` | `watch_period_s` | number 0.2..60 | 1.0 | yes | seconds between searches; one search costs 0.1-0.3 s of one core on this machine, and a candidate is worth the most while the tracker is still healthy |
| `global_watch` | `watch_max_scan_age_s` | number 0.1..3600 | 1.0 | yes | how long a revolution may sit in hand and still be searched, counted from when it ARRIVED here: a bridge that stops delivering leaves the newest scan frozen, and searching it again would publish the same answer as if it were news. A huge value is the old behaviour, which searched whatever was held |
| `neck_state` | `neck_tf` | bool | on | yes | base_link -> camera_link is published live from the neck's encoders; the laptop's camera node must then run with ros/laptop.sh vslam --neck, or two nodes publish that edge |
| `relocalizer` | `rest_lock` | bool | on | yes | hold the pose while the cart stands still (wheels and gyro agree): a match's residual is blended in with a time constant instead of taken whole |
| `relocalizer` | `explained_vote` | bool | on | yes | returns the static map cannot explain (a person, a moved chair) do not score the match |
| `relocalizer` | `rest_tau_s` | number 0.1..60 | 6.0 | yes | the rest lock's time constant: seconds for a residual to die at rest |
| `relocalizer` | `rest_gain` | number 0..1 | 0.05 | yes | the rest lock's share per match when no match cadence is known |
| `relocalizer` | `sources` | list of: lidar, depth, contact | lidar | yes | the scan sources matched against the map: the lidar's revolution (/scan), the camera's depth band (/depth_scan), the floor-contact line (/contact_scan); the lidar drives the updates while it is fresh and the others ride along, a stale lidar hands the updates to them. The fused modes are measured offline (scratch/camera_only_localization.py) and turned on live |
| `relocalizer` | `fusion` | bool | on | yes | fuse every enabled source's match by its information; off: the widest source corrects alone and the others only report |
| `relocalizer` | `toe_reach` | number 0..0.6 | 0.27 | yes | how far past the leg the lidar sees a standing person's toe reaches, metres: the term the dynamic rings are sized on (pepin.dynamic.berth_for). The default is computed from the lidar's mount (config/lidar.json); set it to compare berths in the field without a restart |
| `relocalizer` | `near_rings` | bool | on | yes | a return is ringed as soon as it clears the cart's own outline, and only the marks that would land on that outline are dropped; off, nothing within the ring plus the outline is ringed at all — the older rule, whose blind disc grows with the ring (a 7 cm wider ring stops ringing a person at 0.90 m) |
| `relocalizer` | `accept_candidates` | bool | on | yes | re-seed from the laptop watchdog's whole-map candidates (/localization/candidate, pepin.watchdog): a place that disagrees with the tracked pose candidate_streak times in a row, about the same place each time, is adopted through the path the board's own search uses. Off: the candidates are still judged, counted and reported, and only the board's own slow whole-map search can bring the cart back |
| `relocalizer` | `candidate_streak` | integer 1..10 | 3 | yes | how many candidates in a row must disagree with the tracker and agree with each other before one of them re-seeds it: the price of a teleport, in seconds |
| `relocalizer` | `distinct_scans` | bool | on | yes | a streak is counted in scans, not in messages: a candidate whose scan id is already in the run is a second opinion that heard the first one's scan, counted as replay and not lengthening the streak. Off is the old behaviour, where a frozen /scan on the laptop could re-seed the tracker on one scan's answer repeated |
| `rtabmap_frame` | `slam` | bool | off | at start | RTAB-Map is the map (online SLAM): its correction is map -> odom and goes to the board as a message on /map_odom, where pepin_bringup.slam_frame broadcasts it; off, the board's tracker owns map -> odom and this node broadcasts map -> rtabmap here. Not live: the two modes are two different edges, and a transform once sent stands |

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
| fused | `ros/sensor.sh lidar on` + `ros/sensor.sh camera on` | `lidar,depth,contact` | `lidar_layer`, `camera_layer`, `contact_layer` |
| lidar only | `ros/sensor.sh camera off` | `lidar` | `lidar_layer` |
| camera only | `ros/sensor.sh lidar off` (`--hard` to stop the driver) | `depth,contact` | `camera_layer`, `contact_layer` |

The tracker column needs the relocalizer's `sources` flag, which arrives with the fusion wiring;
the generated **Feature flags** table above is the authority on whether this build carries it
(`ros/flags.sh list relocalizer` says the same about the running node). Until it does, every
`ros/sensor.sh lidar|camera on|off` applies the costmap half, reports that the tracker's sources
are unchanged and exits 1 — so the costmap column is what holds, and the modes are half modes.

The camera is one sensor read twice from the same frames: `depth_scan` is the band 8 cm-1.3 m
above the floor (table tops, seats, a hand) and `contact_scan` is where the floor ends (chair
feet, a plinth) — so `camera on` moves two sources and two layers at once.

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
  matched. This is where camera-only localisation fails visibly;
- the Log panel (filtered to `relocalizer`, `controller_server`, `planner_server`, ...).

The report lines say the same in words, every 30 s, and `ros/sensor.sh status` prints all three:

- `relocalizer` on the board: `tracker: ... flags: rest_lock=on explained_vote=on ...`, with
  `sources=lidar,depth,contact` among those flags once the tracker carries them (`ros/sensor.sh
  status` prints the sources from this line, and says so plainly when the line has none)
- `depth_stream` in the laptop's SLAM container: `depth: 3.1 frames/s published ...`
- `contact_scan` in the same container: `contact: 2.9 scans/s published ...`

### What the numbers already say

Camera-only localisation on the **lidar's** map does not work, and was measured before it was
demonstrated (`scratch/camera_only_localization.py`, run 0171 replayed offline against
`ros/maps/flat3_straight.yaml`):

| tracker sources | error vs lidar-only (median / p90 / max) | verdict |
| --- | --- | --- |
| `depth` | 122 cm / 124 deg at the loss | loses the map after 0.5 s |
| `contact` | 80 cm / 28 deg at the loss | loses the map after 12 s |
| `depth,contact` | 29 / 52 / 90 cm | loses the map |
| `lidar,depth,contact` | 0.7 / 1.6 / 5.7 cm, 0.21 / 0.56 / 1.9 deg | never lost |

The camera's band is a different cross-section of the room than the lidar's own plane:
sofa cushions and table clutter fit "some wall" well (fit 0.90 at the wrong pose, 0.12 at the
true one), and parked bumper-to-furniture the camera sees nothing of the floor below ~1.2 m.
Camera-only localisation needs a camera-built map, not the lidar's slice. So the honest demo is:
**the costmap half survives either sensor alone; the tracker half needs the lidar** — and fusing
all three costs 0.7 cm of median agreement, which is free. `ros/sensor.sh lidar off` prints that
warning itself when it takes the last lidar out of the tracker's sources.

## Frames and conventions

`base_link` sits between the drive-wheel contact points (our robot frame): x forward, y left.
Footprint (meters, `base_link`): front 0.0625, rear 0.30, half-width 0.275 — from `config/base.json`.
