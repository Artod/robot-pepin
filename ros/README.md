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
| `camera_stream` | `static_camera_tf` | bool | on | at start | base_link -> camera_link is broadcast from here; it goes off (ros/laptop.sh vslam --neck) when the board's neck node publishes that edge live from the servo encoders (neck_state, flag neck_tf), because two publishers of one edge fight. Not live: a static transform cannot be withdrawn once sent |
| `contact_scan` | `contact_scan` | bool | on | yes | the contact line is published; off, the node is a subscriber that costs nothing — the costmap's own contact_layer.enabled is the other end of the same demo switch, and either one alone takes the camera's floor line out |
| `contact_scan` | `shadow` | bool | on | yes | the last floor pixel on a face stands a band's width UP that face, so its ray lands past the foot: on (measured), that width is taken back off the range (pepin.contact.band_shadow); off is the raw boundary ray |
| `contact_scan` | `max_range` | number 0.1..10 | 2.0 | yes | metres past which a column is called clear instead of ended; the default is where the floor is still the floor, not where the optics run out, and the costmap's contact_layer.obstacle_max_range must match it |
| `depth_fusion` | `enabled` | bool | on | yes | frames are fused into the model; off, they are dropped |
| `depth_fusion` | `fit_gate` | bool | on | yes | frames are fused only while the tracker reports /localization_fit >= 0.50; off, every frame is fused: SLAM mode, where RTAB-Map owns the pose and no tracker speaks |
| `depth_fusion` | `self_heal` | bool | on | yes | a streak of frames refused at the alignment bound empties the model, so it re-seeds from the next frame instead of staying frozen until a human resets it |
| `depth_fusion` | `align` | bool | on | yes | frame-to-model: a frame's lidar-height band is turned about the cart to fit the model before it is fused, and a frame whose best turn is the search's bound is refused |
| `depth_fusion` | `min_weight` | number 0..100 | 2.0 | yes | observations a voxel needs before it is shown in /fusion/surface |
| `depth_fusion` | `surface_hz` | number 0.1..10 | 1.0 | yes | how often /fusion/surface is published (the crossing search costs a fraction of a second) |
| `depth_stream` | `edge_filter` | bool | on | yes | flying pixels at object edges are dropped from the published depth and the scan; the law's beam pairs skip them regardless |
| `depth_stream` | `lidar_anchor` | bool | on | yes | the lidar fits the depth's law; off, the last law is held (the failure mode of a lidar that stops) — with no law yet nothing is published until it is back on |
| `depth_stream` | `floor_pairs` | bool | off | yes | the floor's pixels pair the network's depth with the plane's geometric depth, a second hoop for the law that needs no lidar; off by default: measured, it pulls the law off the lidar's row (1.4x too far there), where the costmap lives |
| `depth_stream` | `wall_anchor` | bool | off | yes | the lidar's returns extruded up the image while the network's depth stays continuous pair the rows above the lidar's with the wall's depth, a third hoop; off by default: measured, it puts the lidar's row 10 % too near while fixing the rows above |
| `depth_stream` | `affine_law` | bool | on | yes | the network's depth through 1 / z = a / D + b, fitted on the pooled pairs; off, the raw network's depth goes out unwithheld (1.5-2x too far: an A/B measure of the correction, never a way to drive) |
| `depth_stream` | `wall_correct` | bool | off | yes | after the law, the pixels the wall walk covered are set to the extruded plane's depth outright; off by default (the same walk as wall_anchor, applied instead of fitted) |
| `depth_stream` | `floor_anchor` | bool | on | yes | pixels within centimetres of the floor plane snap to it in the published image (the scan is built before it); the plane leans with the cart, from the IMU's up vector |
| `depth_stream` | `depth_backend` | choice: remote, local, auto | local (env PEPIN_DEPTH_BACKEND) | yes | where the network runs: local (the CPU model in this container), remote (the laptop's GPU service, ros/depth_host.sh), auto (the service while it answers, the CPU model while it does not) |
| `neck_state` | `neck_tf` | bool | on | yes | base_link -> camera_link is published live from the neck's encoders; the laptop's camera node must then run with ros/laptop.sh vslam --neck, or two nodes publish that edge |
| `relocalizer` | `rest_lock` | bool | on | yes | hold the pose while the cart stands still (wheels and gyro agree): a match's residual is blended in with a time constant instead of taken whole |
| `relocalizer` | `explained_vote` | bool | on | yes | returns the static map cannot explain (a person, a moved chair) do not score the match |
| `relocalizer` | `rest_tau_s` | number 0.1..60 | 6.0 | yes | the rest lock's time constant: seconds for a residual to die at rest |
| `relocalizer` | `rest_gain` | number 0..1 | 0.05 | yes | the rest lock's share per match when no match cadence is known |
| `relocalizer` | `sources` | list of: lidar, depth, contact | lidar | yes | the scan sources matched against the map: the lidar's revolution (/scan), the camera's depth band (/depth_scan), the floor-contact line (/contact_scan); the lidar drives the updates while it is fresh and the others ride along, a stale lidar hands the updates to them. The fused modes are measured offline (scratch/camera_only_localization.py) and turned on live |
| `relocalizer` | `fusion` | bool | on | yes | fuse every enabled source's match by its information; off: the widest source corrects alone and the others only report |
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

The camera's band is a different cross-section of the room than the lidar's plane 20 cm up:
sofa cushions and table clutter fit "some wall" well (fit 0.90 at the wrong pose, 0.12 at the
true one), and parked bumper-to-furniture the camera sees nothing of the floor below ~1.2 m.
Camera-only localisation needs a camera-built map, not the lidar's slice. So the honest demo is:
**the costmap half survives either sensor alone; the tracker half needs the lidar** — and fusing
all three costs 0.7 cm of median agreement, which is free. `ros/sensor.sh lidar off` prints that
warning itself when it takes the last lidar out of the tracker's sources.

## Frames and conventions

`base_link` sits between the drive-wheel contact points (our robot frame): x forward, y left.
Footprint (meters, `base_link`): front 0.0625, rear 0.30, half-width 0.275 — from `config/base.json`.
