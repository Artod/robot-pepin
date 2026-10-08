# Tools

Find the job in this file before writing a script for it. When a tool already owns the job, extend it with a flag or a subcommand rather than adding a sibling.

`/tools/X.py` runs inside a stack container: `docker exec <container> /pepin_entrypoint.sh python3 /tools/X.py` (`pepin-macnav` for Nav2, `pepin-vslam` for the camera half, `pepin-ros` on the board). `uv run ...` runs on the laptop from the repository root.

## By job

| Job | Tool | Run |
|---|---|---|
| Drive to a place or a pose; cancel; where is the robot | `ros/goto.sh` | `ros/goto.sh NAME \| X Y [YAW] \| cancel \| where` |
| Start, restart and prove the stack | `ros/laptop.sh`, `ros/restart.sh` | `ros/laptop.sh nav`, `ros/restart.sh both` |
| Put a code change on the robot | `ros/push.sh` | `ros/push.sh FILE...` |
| EKF replay, odometry scoring, lidar truth, lidar mount yaw | `ros/tools/odom_bench` | `uv run --with rosbags==0.11.5 python -m ros.tools.odom_bench run \| truth TAPE \| crab RUN...` |
| Visual odometry A/B on a recorded drive | `ros/vio_replay.sh`, `ros/tools/vio_score.py` | `ros/vio_replay.sh RUN --arm E` |
| Costmap parameters against recorded drives | `ros/replay.sh` | `ros/replay.sh 483-498 --set KEY=VALUE` |
| Bag to tape, pose topic to CSV, camera clip to bag | `bag_to_tape.py`, `bag_poses.py`, `ros/clip_to_bag.sh` | see the tables below |
| Stream latency and clock agreement | `ros/tools/stream_latency.py` | `python3 /tools/stream_latency.py [seconds]` |
| Why a goal failed (behaviour tree, Nav2 log) | `ros/tools/bt_watch.py`, `ros/watch.sh` | `ros/watch.sh [FILE]` |
| A node is alive but silent | `ros/tools/stack.sh` | `ros/tools/stack.sh NODE [CONTAINER]` |
| What the board runs and what it costs | `ros/board.sh` | `ros/board.sh census` |
| Feature flags and config knobs | `ros/flags.sh` | `ros/flags.sh list [NODE]` |
| Planner and controller trials without the robot | `ros/sim.sh` | `ros/sim.sh up`, `ros/sim.sh scenario FILE` |

## ros/tools

| Tool | Does | Run |
|---|---|---|
| `bag_poses.py` | One pose topic of a bag as CSV (stamp, x, y, yaw), the input of `vio_score.py` | `python3 /repo/ros/tools/bag_poses.py BAG TOPIC > arm.csv` |
| `bag_to_tape.py` | A drive's rosbag2 bag converted to the JSONL tape the analyses read | `python3 /tools/bag_to_tape.py /maps/rec/RUN` (pepin-vslam) |
| `bt_watch.py` | The behaviour tree's decisive transitions, one line each | `python3 -u /tools/bt_watch.py` (pepin-macnav) |
| `build_rmw_zenoh_fix.sh` | Builds the patched `librmw_zenoh_cpp.so` in a throwaway container | `ros/tools/build_rmw_zenoh_fix.sh` |
| `clip_to_bag.py` | A drive's camera clip as a stereo camera bag stamped by capture time | `ros/clip_to_bag.sh RUN` |
| `flags_doc.py` | Renders the nodes' flags and knobs into ros/README.md; checks it is current | `ros/tools/flags_doc.py [--check \| nodes \| where NODE]` |
| `foxglove_probe.py` | Read-only probe of the foxglove_bridge websocket | `python3 ros/tools/foxglove_probe.py [--port 8765]` |
| `gaze_preset.py` | The gaze arbiter's follow, glance and reverse knobs set as a named preset | `uv run python ros/tools/gaze_preset.py baseline \| follow \| show` |
| `goto_ros.py` | Nav2 side of `goto.sh`: cancel fallback, marks, places, RTAB-Map seed | `goto_ros.py cancel \| seed X Y [YAW] \| mark NAME \| places` |
| `head_calib.py` | Head camera-IMU calibration: exposure check, bag summary, Kalibr verdict, apply | `uv run python ros/tools/head_calib.py exposure \| summary \| meta \| report \| apply` |
| `head_static_tf.py` | The camera's static transforms for an offline replay | `python3 /repo/ros/tools/head_static_tf.py --ros-args -p use_sim_time:=true` |
| `install_rmw_zenoh_fix.sh` | Image build step: the patched `librmw_zenoh_cpp.so` over the packaged one | `ros/Dockerfile*` |
| `kalibr_detect.py` | Kalibr's AprilGrid detector on one rectified picture, one JSON line | called by `neck_dance.py --check` (pepin-kalibr) |
| `map_odom.py` | Prints `map -> odom` from TF: is anything correcting the pose | `python3 /tools/map_odom.py [seconds]` |
| `move.py` | Measured straight legs and turns without the planner, recorded | `ros/goto.sh move NAME f0.40 t90 ...` |
| `nav_goal_running.py` | Is a navigation goal running right now (one rclpy pass, for guards) | `docker exec -i CONTAINER /pepin_entrypoint.sh python3 - < ros/tools/nav_goal_running.py` |
| `nav_speed.py` | Nav2's speed parameters read or set (`ros/speed.sh` uses it) | `python3 /tools/nav_speed.py [--set X]` (pepin-macnav) |
| `neck_dance.py` | Head poses that excite the IMU in front of the AprilGrid | `uv run python ros/tools/neck_dance.py [--check \| --move]` |
| `npz_to_map.py` | A saved occupancy grid as a `map_server` pair (.pgm + .yaml) | `uv run python ros/tools/npz_to_map.py data/maps/NAME.npz --out ros/maps/NAME` |
| `odom_bench/` | Odometry bench on a frozen drive set: lidar truth, stamp offsets, covariance honesty, EKF replays, verdict | `uv run --with rosbags==0.11.5 python -m ros.tools.odom_bench run \| freeze \| crab \| truth` |
| `place_backfill.py` | A place descriptor for every node of an RTAB-Map database; census | `uv run python ros/tools/place_backfill.py [--check [--json]] DB` |
| `planner_check.py` | The global costmap is updating and one path is computed from here | `python3 /tools/planner_check.py` (pepin-macnav) |
| `session_logger.py` | Raw scans, odometry and pose to a JSONL session for offline SLAM | `python3 /tools/session_logger.py /maps/rec/NAME.jsonl [SECONDS]` |
| `stack.sh` | Every thread's stack of a stuck node, taken before any restart | `ros/tools/stack.sh NODE [CONTAINER]` |
| `stereo_calibrate.py` | Checkerboard calibration of the stereo head | `ros/calibrate.sh stereo` |
| `stereo_kalibr.py` | Stereo head calibration with Kalibr on the raw eyes | `ros/calib_stereo.sh` (`record \| bag \| report \| apply`) |
| `stream_latency.py` | Stamp-to-receipt latency of the key streams and clock agreement, one line | `python3 /tools/stream_latency.py [seconds]` |
| `tof_check.py` | Do the three ToF sensors reach the laptop: OK or FAIL per sensor | `python3 /tools/tof_check.py [seconds]` |
| `topic_rate.py` | A topic's rate over a few seconds, one line | `python3 /tools/topic_rate.py TOPIC [seconds] [latched]` |
| `turn_full.py` | One full turn in place, judged by the odometry, recorded | `ros/goto.sh round [NAME]` |
| `vio_config.py` | OpenVINS and Kalibr configuration generated from the repository's config | `uv run python ros/tools/vio_config.py [--kalibr-only]` |
| `vio_score.py` | Each VIO arm's fused odometry scored against the lidar truth | `uv run python ros/tools/vio_score.py runs/RUN... --arms A B E --baseline B` |
| `vo_params.py` | The live stereo VO parameters as a file for an offline replay | `python3 ros/tools/vo_params.py > vo.yaml` |

## ros/*.sh: running the stack

| Tool | Does | Run |
|---|---|---|
| `board.sh` | What the board runs against its manifest; restart one board node | `ros/board.sh census [--json] \| manifest \| kick NODE` |
| `build-image.sh` | Builds the board's sensor image on the laptop and ships it | `ros/build-image.sh [--ship \| --ship-only]` |
| `calib_record.sh` | Records the head camera-IMU calibration bag while the head moves | `ros/calib_record.sh --check \| --centre X Y --distance D` |
| `calib_run.sh` | Kalibr on a calibration recording, with its verdict | `ros/calib_run.sh BAG TAG_M [--against BAG2]` |
| `calib_stereo.sh` | Stereo calibration with Kalibr: record, then run | `ros/calib_stereo.sh record \| run REC TAG_M [--apply]` |
| `calibrate.sh` | Checkerboard calibration of the camera, mono or stereo | `ros/calibrate.sh [--print \| stereo \| --images DIR]` |
| `camera_grid.sh` | Costmaps on the camera grid or on the camera obstacle layer | `ros/camera_grid.sh status \| on \| off` |
| `clip.sh` | Copies the head camera's MJPEG stream into a file until TERM | `ros/clip.sh FILE URL` |
| `clip_to_bag.sh` | A drive's camera clip as a camera bag beside the drive's bag | `ros/clip_to_bag.sh RUN` |
| `depth_host.sh` | The depth network on the laptop GPU as a service | `ros/depth_host.sh start \| stereo \| stop \| status \| bench [N]` |
| `entrypoint.sh` | Container entrypoint: sources ROS once, then runs the command | `/pepin_entrypoint.sh CMD...` |
| `exposure.sh` | The head camera's exposure, live | `ros/exposure.sh show \| apply \| auto \| manual \| capped` |
| `feature.sh` | Persistent switches of the board's sensor stack | `ros/feature.sh imu \| ekf \| laser_odom \| tof \| head_imu \| board_bag on \| off` |
| `fetch.sh` | Copies every recording still on the board to the laptop | `ros/fetch.sh` |
| `flags.sh` | Every node's feature flags and knobs, wherever the node runs | `ros/flags.sh list \| flag \| get \| set NODE FLAG [VALUE] \| drift` |
| `foxglove.sh` | Foxglove bridge check and the desktop app's link | `ros/foxglove.sh check \| reopen \| url` |
| `gaze_gate.sh` | The gaze gate flag and knobs in every node that carries it | `ros/gaze_gate.sh [on \| off \| KNOB VALUE]` |
| `goto.sh` | Goals through Nav2 with feedback; places, seed, measured moves, planner and controller choice | `ros/goto.sh NAME \| X Y [YAW] \| cancel \| where \| mark \| places \| seed \| round \| move \| planner \| controller` |
| `laptop-build.sh` | Builds the laptop image and its variants | `ros/laptop-build.sh [xfeat \| vio \| gaze]` |
| `laptop.sh` | Runs the navigation and camera stacks on the laptop | `ros/laptop.sh nav [down \| logs] \| vslam \| vio \| kick NODE \| logs \| stop` |
| `lib.sh` | Shared helpers: one multiplexed ssh connection to the board | sourced by `ros/*.sh` |
| `map.sh` | Saves the map being built as a file | `ros/map.sh save NAME` |
| `models.sh` | The depth and localisation model services as launchd jobs | `ros/models.sh install \| start \| stop \| restart \| status \| logs [depth \| localization]` |
| `neck.sh` | The neck servos through the base server: read, move, hold | `ros/neck.sh read \| home \| goto PAN TILT \| hold PAN TILT \| motion \| registers` |
| `preflight.sh` | Ready for a goal: pose, lidar, planner, snapshots, ToF, Foxglove, one plan | `ros/preflight.sh [--no-plan]` |
| `push.sh` | Copies changed files to the robot and restarts only the nodes that import them | `ros/push.sh [--dry-run] FILE...` |
| `ready.sh` | After the cart is placed on its base: seed, reset, proof | `ros/ready.sh [X Y YAW]` |
| `reap_ros2_cli.sh` | Ends leftover `ros2` CLI processes in the board container (timer) | `ros/reap_ros2_cli.sh` |
| `replay.sh` | Recorded drives through Nav2's costmaps with candidate parameters (`ros/replay/`) | `ros/replay.sh 483-498 [--set K=V] [--params FILE] [--save FILE]` |
| `reset_world.sh` | Empties the voxel volume and both costmaps | `ros/reset_world.sh` |
| `restart.sh` | Restarts the stack and proves it works | `ros/restart.sh board \| laptop \| both [--deploy] [--dry-run]` |
| `run.sh` | Runs the ROS 2 container on the board | `ros/run.sh [CMD...]` |
| `sensor.sh` | Per-sensor costmap layers and sensor mutes, live | `ros/sensor.sh status \| lidar on\|off \| camera on\|off \| mute SENSOR` |
| `sim.sh` | Kinematic simulator for Nav2 in the saved room (`ros/sim/`) | `ros/sim.sh up \| goal NAME \| scenario FILE \| status \| down` |
| `speed.sh` | The cart's speed ceiling in every place that holds it | `ros/speed.sh [X]` |
| `stop.sh` | Stops the wheels and cancels every goal, confirmed by the wheels | `ros/stop.sh` |
| `sync.sh` | Copies the checkout's code to the board without an image rebuild | `ros/sync.sh [--restart]` |
| `teleop.sh` | Keyboard driving through ROS, optionally a recorded mapping run | `ros/teleop.sh [NAME]` |
| `time.sh` | One clock: the board's chrony and the laptop's time server | `ros/time.sh install \| offset \| status \| point \| server` |
| `vio_replay.sh` | One recorded drive, one VIO arm, the board's EKF replayed | `ros/vio_replay.sh RUN --arm A \| B \| E \| E0` |
| `watch.sh` | The navigation stack's own messages, live or from a saved log | `ros/watch.sh [FILE]` |
| `xfeat/build_rtabmap.sh` | RTAB-Map rebuilt from source with Python support | `build_rtabmap.sh core \| ros` (in `ros/Dockerfile.xfeat`) |
| `xfeat/patch_rtabmap.sh` | RTAB-Map core patches applied and the library swapped in | `ros/Dockerfile.xfeat` |

## board/*.sh

| Tool | Does | Run |
|---|---|---|
| `chrony.sh` | The board's half of the clock: chrony pointed at the laptop | `ros/time.sh install` (on the board: `chrony.sh install HOST \| source \| status \| uninstall`) |
| `tof_init.sh` | Unique I2C addresses for the three ToF sensors at boot | `tof-init.service` |
| `wifi_primary.sh` | Which WiFi radio carries the board's network identity | `wifi_primary.sh dongle \| onboard \| status \| confirm` (board, root) |
| `xvf_host_install.sh` | Installs the microphone array's host-control tools | `bash xvf_host_install.sh` (board, root) |

## python -m pepin.*: servers and CLIs

| Module | Does | Run |
|---|---|---|
| `pepin.allan` | Allan deviation of a parked IMU recording; the noise block for VIO and Kalibr | `uv run python -m pepin.allan parked.jsonl.gz [--plot allan.png]` |
| `pepin.audio_server` | Board server of the microphone array: hearing, speaking, direction (TCP 3338) | `python -m pepin.audio_server --port 3338` |
| `pepin.base_server` | Board server that owns the wheels: odometry and twist at 50 Hz (TCP 3336) | `python -m pepin.base_server --config config/base.json` |
| `pepin.board_bag` | The board's rolling raw-sensor bag under a size cap | `python3 -m pepin.board_bag --dir /maps/board_rec` |
| `pepin.camera_controls` | The head camera's exposure as config: show, apply, frame rate | `python -m pepin.camera_controls show \| apply \| rate` |
| `pepin.census` | Board processes against `config/board_manifest.json` | `ros/board.sh census` |
| `pepin.depth_service` | The depth network on the laptop GPU as an HTTP service | `ros/depth_host.sh start` |
| `pepin.face` | The mouth table from `config/face.json`; regenerates the firmware and simulator tables | `uv run python -m pepin.face --write` |
| `pepin.goal_link` | Goal server client: cancel, where, planner, goal | `python3 -m pepin.goal_link cancel \| where \| planner NAME` |
| `pepin.head_imu` | Head IMU tools: a stand-in head server and a recorder | `python -m pepin.head_imu fake \| record --seconds N --out FILE` |
| `pepin.head_link` | Head server client: status, expressions, events, IMU | `python -m pepin.head_link status \| express \| event \| clear \| show \| imu` |
| `pepin.head_server` | Board server that owns the head ESP32's serial port (TCP 3340) | `python -m pepin.head_server` |
| `pepin.i2c_recover` | Frees a locked I2C bus on the board without a power cycle | `python -m pepin.i2c_recover check \| recover` (board) |
| `pepin.localization_service` | The localisation models on the laptop GPU as one HTTP service | `ros/models.sh start localization` |
| `pepin.push` | Which running nodes a changed file needs restarted, or why a push refuses | `uv run python -m pepin.push plan FILE...` |
| `pepin.red_button` | The base's own stop, confirmed by the wheels | `python3 -m pepin.red_button [--host BOARD]` |
| `pepin.speed` | The cart's speed ceiling: check a value, read the file, read or set the base | `python -m pepin.speed check X \| file \| base --host H --port P` |
| `pepin.static_facts` | OK or FAIL on nodes that wait for static transforms, from their logs | `python3 -m pepin.static_facts < LOG` |
| `pepin.teleop` | Keyboard driving in a game-mode window | `uv run python -m pepin.teleop --game` |
| `pepin.tof_server` | Board server streaming the three ToF ranges as JSON lines | `python -m pepin.tof_server --port 3335` |
| `pepin.tools.mcp` | The robot's tool registry as an MCP server over stdio | `uv run python -m pepin.tools.mcp [--board HOST]` |
| `pepin.voice` | The voice pipeline's attach point, listening to the array | `uv run python -m pepin.voice --host HOST` |
| `pepin.xvf3800` | Microphone array USB control: firmware version, voice direction | `python -m pepin.xvf3800 version \| doa --watch 5` |

## scripts/: pre-ROS utilities

| Script | Does | Run |
|---|---|---|
| `arm_boxes.py` | Fits the SO-101 links' collision boxes to the upstream meshes | `uv run python scripts/arm_boxes.py [--check]` |
| `build_map.py` | Occupancy map from a recorded session, optionally scan-matched | `uv run python scripts/build_map.py data/sessions/S.jsonl [--match]` |
| `calibrate_camera.py` | Checkerboard calibration of the neck camera | `ros/calibrate.sh` |
| `calibrate_neck.py` | Neck and head servo calibration | `uv run python scripts/calibrate_neck.py --port PORT` |
| `dashboard.py` | Live rerun dashboard: scan, camera, ToF, board vitals, log | `uv run python scripts/dashboard.py [--map data/maps/M.npz]` |
| `extrinsics.py` | Library: the camera's yaw against the lidar from the two range fans | imported by `tests/unit/test_extrinsics.py` |
| `health_check.py` | Launch-readiness check of every subsystem: GO or NO GO | `uv run python scripts/health_check.py [--quick]` |
| `jog.py` | Gentle single-motor jog for bring-up | `uv run python scripts/jog.py wheel \| neck --port PORT --id N` |
| `setup_motor_id.py` | Assigns a bus ID to one Feetech servo | `uv run python scripts/setup_motor_id.py --port PORT --id N` |
| `timesync.py` | The board's clock offset from the laptop over NTP, one line | `ros/time.sh offset` |
| `voice.py` | Voice loop: the array, Gemini on the audio, the robot's tools, speech | `uv run python scripts/voice.py` |
| `voice_live.py` | Voice over the Gemini Live API with a local wake gate | `uv run python scripts/voice_live.py [--dry-run]` |
