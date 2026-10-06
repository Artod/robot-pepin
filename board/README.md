# Board: Orange Pi Zero 3 as the robot's sensor box

The board runs Armbian and does nothing clever: it bridges the servo bus to TCP, streams the ToF
ranges, serves the camera and the microphone array, runs the ROS sensor stack, and — the one
real-time job — owns the wheels. Navigation (Nav2, the goal server, the drive recorders) and
everything that consumes the camera run on the laptop.

## The ROS sensor stack

One container, `pepin-ros` (`pepin-ros.service`, `ros/run.sh`), runs `bringup.launch.py`, which
is `robot.launch.py` alone: the LD19 driver and its hull filter (`/scan`), the C++ base bridge
(`/odom`, `/imu/data_raw`, `/zupt`, `/cmd_vel` to the base server, and the neck's encoders as
`/neck/state` and `base_link -> camera_link`), the EKF (`odom -> base_link`), the lidar's
scan-to-scan odometry, and behind their switches the ToF bridge and the raw-sensor recorder. Its
router is `pepin-zrouter.service` (rmw_zenoh), on the same image.

| Switch (`/etc/default/pepin-ros`) | `ros/feature.sh` | Default |
| --- | --- | --- |
| `PEPIN_IMU` | `imu on\|off` | true |
| `PEPIN_EKF` | `ekf on\|off` | true |
| `PEPIN_LASER_ODOM` | `laser_odom on\|off` | true |
| `PEPIN_TOF` | `tof on\|off` | true |
| `PEPIN_BOARD_BAG` | `board_bag on\|off` | false |

Older lines in that file (`PEPIN_CPP_BRIDGE`, `PEPIN_NAV`, `PEPIN_SLAM_TOOLBOX`, `PEPIN_MAP`,
`PEPIN_SIDE`, `PEPIN_RECORDER`) are read by nothing; `ros/build-image.sh --ship` refuses a board
whose file still says `PEPIN_NAV=true` or `PEPIN_SLAM_TOOLBOX=true`.

The image is built on the Mac (the same arm64 architecture, minutes instead of the board's half
hour) and loaded here with `ros/build-image.sh --ship`, which also installs
`board/pepin-ros.service` ([ros/README.md](../ros/README.md), "Building the image"). Code,
`params/ekf.yaml` and `config/` travel with `ros/sync.sh`; no map goes to the board.

## Services

| Port | Service | Unit | What it does |
| --- | --- | --- | --- |
| 3333 | ser2net | `ser2net.service` | raw TCP to `/dev/servo-bus` (Feetech bus, 1 Mbit/s); `kickolduser` hands the port to the newest client |
| 3335 | `pepin.tof_server` | `pepin-tof.service` | JSON lines with the three VL53L1X ranges at 15 Hz; needs `tof-init.service` first |
| 3336 | `pepin.base_server` | `pepin-base.service` | owns the wheels: reads the encoders and applies twists at 50 Hz over loopback to :3333, deadman 0.5 s, a state line every tick; the neck commands (below) |
| 3338 | `pepin.audio_server` | `pepin-audio.service` | the microphone array: its voice as 20 ms PCM frames, the voice direction at 10 Hz, the laptop's speech out through its jack, `status`; the speech's loudness to the head's mouth (lip sync, on by default; `--no-lipsync` off) |
| 3340 | `pepin.head_server` | `pepin-head.service` | the head ESP32 on `/dev/pepin-head`: the mouth's expressions, speech levels and info screens; the head IMU's samples on the board's clock to subscribers (the base bridge); the brain lease (see "The head" below) |
| 8080 | ustreamer | `pepin-camera.service` | the head camera as MJPEG and `/snapshot`; which camera is `/etc/default/pepin-camera`, its exposure the active rig's `exposure` block of `config/camera.json`, set before ustreamer starts (`pepin.camera_controls`; `ros/exposure.sh` shows and tries the modes live) |

The lidar belongs to the ROS container; its ser2net port (3334) stays commented out in
`ser2net.yaml`, for bench work with the container stopped. The base server is the only client of
:3333 while it runs: bench tools that talk to the servo bus directly (`scripts/jog.py`,
`scripts/setup_motor_id.py`, `scripts/calibrate_neck.py`) need `systemctl stop pepin-base` before
and `systemctl start pepin-base` after.

## Files and where they go

| In the repo | On the board |
| --- | --- |
| `board/ser2net.yaml` | `/etc/ser2net.yaml` |
| `board/ser2net-stale-locks.conf` | `/etc/systemd/system/ser2net.service.d/stale-locks.conf` |
| `board/99-pepin-usb.rules` | `/etc/udev/rules.d/99-pepin-usb.rules` (then `udevadm control --reload`) |
| `board/tof_init.sh` | `/usr/local/bin/tof_init.sh` (executable) |
| `board/tof-init.service`, `pepin-tof.service`, `pepin-base.service`, `pepin-camera.service`, `pepin-audio.service`, `pepin-head.service`, `pepin-zrouter.service` | `/etc/systemd/system/` |
| `board/pepin-ros.service` | `/etc/systemd/system/` (installed by `ros/build-image.sh --ship`) |
| `board/pepin-reap.service`, `pepin-reap.timer` | `/etc/systemd/system/` (they run `ros/reap_ros2_cli.sh`, which `ros/sync.sh` puts in `/root/pepin-ros/`) |
| `board/wifi-runtime-pm-on.conf` | `/etc/systemd/system/wifi-powersave-off.service.d/runtime-pm-on.conf` |
| `board/i2c3-400k.dts` | `armbian-add-overlay board/i2c3-400k.dts`, then a reboot: the IMU and the ToF share that bus, and at the default 100 kHz the ToF held the IMU at 53 of its 100 Hz |
| `board/chrony.sh`, `board/chrony/` | installed by `ros/time.sh install` ([ros/README.md](../ros/README.md), "One clock") |
| `board/wifi_primary.sh` | run by hand, as root: `dongle\|onboard\|status\|confirm` picks which radio carries the board's DHCP identity, with a timed rollback (see its header) |
| `board/xvf_host_install.sh` | run once: Seeed's `xvf_host` tools into `/opt/xvf_host` |
| `src/pepin/` (the package, stdlib only on the board) | `/opt/pepin/pepin/` |
| `config/base.json`, `config/neck.json`, `config/camera.json` | `/opt/pepin/config/` (`neck.json`: the ids read with the wheels, the limits and the motion the neck commands obey; absent, those commands answer an error and the wheels do not care; `camera.json`: the camera's exposure, read by pepin-camera at its start) |
| `config/head.json`, `config/face.json` | `/opt/pepin/config/` (`head.json`: the head's port, its IMU's rate and ranges, the clock map; `face.json`: the mouth's expressions and the robot events, read by pepin-head and by pepin-audio's lip sync) |
| `ros/` | `/root/pepin-ros/` (`ros/sync.sh`) |

Deploy the host package and its configuration from the laptop:

```bash
rsync -a --delete --exclude '__pycache__' src/pepin/ root@pepin.local:/opt/pepin/pepin/
scp config/base.json config/neck.json config/camera.json root@pepin.local:/opt/pepin/config/
ssh root@pepin.local 'systemctl restart pepin-base pepin-tof pepin-audio'
```

## Microphone array

The reSpeaker XVF3800 USB array (Seeed, `2886:001a`) sits on the powered hub and the robot's
speaker on its 3.5 mm jack: the voice must leave through the array, or its echo canceller has no
reference. `pepin.audio_server` streams the array's processed channel (16 kHz mono, echo
cancelled, beamformed, noise suppressed), reads the voice direction over USB control and plays
the laptop's PCM; the framing is `src/pepin/audio_link.py`. Install from the laptop:

```bash
ssh root@pepin.local 'apt install -y alsa-utils libusb-1.0-0 usbutils dfu-util && /opt/pepin/bin/pip install pyusb==1.3.1 libusb-package==1.0.30.0'
scp board/99-pepin-usb.rules root@pepin.local:/etc/udev/rules.d/ && ssh root@pepin.local 'udevadm control --reload'
scp board/xvf_host_install.sh root@pepin.local:/tmp/ && ssh root@pepin.local 'bash /tmp/xvf_host_install.sh'
scp board/pepin-audio.service root@pepin.local:/etc/systemd/system/ && ssh root@pepin.local 'systemctl daemon-reload && systemctl enable --now pepin-audio'
```

The udev rule names the card `respeaker` (`plughw:CARD=respeaker,DEV=0`) when the array is
plugged in after it; without it the card is `Array`, which the server finds too.
`libusb-package` is only for Seeed's `xvf_host.py`. Firmware 2.0.10 or newer (`xvf_host
VERSION`; the direction froze before it), 2.1.1 current. Check: `{"cmd":"status"}` on :3338
answers whether capture is alive, frames per second, dropouts and the direction's age; the same
numbers go to the journal once a minute.

After every cold power-up so far (4 of 4) the array enumerates and answers control transfers,
yet its capture endpoint delivers nothing: each `arecord` ends 0.55 s after the open in `read
error: Input/output error`, for hours; plugged in after boot, or after the chip's own reboot
(`xvf_host REBOOT 1`), it streams. The server reboots the chip itself once, after two streams in
a row closed without a byte, and logs the chip's state (GPO/GPI levels with the mute button, PLL
lock, USB FIFO) before and after: `journalctl -u pepin-audio | grep -E "ARRAY|array state"`.
`--no-array-reboot` turns it off; the reboot resets every parameter to its default.

## The head

An ESP32 board with a 1.9" screen sits under the stereo camera: the screen is the robot's mouth,
and an MPU6050 glued to the camera body is the head IMU ([firmware/head_esp32](../firmware/head_esp32/README.md):
wiring, flashing, the protocol). It hangs on the expansion board's free USB port (its own bus,
beside the WiFi dongle's); udev names its CH340 `/dev/pepin-head`, and `pepin.head_server` is the
only thing that opens it. Install from the laptop (the ESP32 flashed first, on the laptop):

```bash
scp board/99-pepin-usb.rules root@pepin.local:/etc/udev/rules.d/ && ssh root@pepin.local 'udevadm control --reload && udevadm trigger'
rsync -a --delete --exclude '__pycache__' src/pepin/ root@pepin.local:/opt/pepin/pepin/
scp config/head.json config/face.json root@pepin.local:/opt/pepin/config/
scp board/pepin-head.service root@pepin.local:/etc/systemd/system/ && ssh root@pepin.local 'systemctl daemon-reload && systemctl enable --now pepin-head'
uv run python -m pepin.head_link status       # link up, the face's fps, the IMU's rate, the clock
```

| Line on :3340 | What happens |
| --- | --- |
| `{"cmd":"express","source":"voice","name":"thinking","hold_s":null}` | the source's expression (config/face.json's names); `hold_s` null stands until replaced, a number lapses back to what stood before |
| `{"cmd":"event","source":"goal","name":"arrived","end":true}` | a robot event from config/face.json's table; `end` first clears what the source held |
| `{"cmd":"clear","source":"voice"}` | the source shows nothing any more |
| `{"cmd":"mouth","level":0.4}` | the speech level, 0..1, straight to the ESP32 (the audio server's lip sync) |
| `{"cmd":"show","text":"Temps\|left: 41/70 C","seconds":8}` | an info screen (`items` for the structured form) |
| `{"cmd":"lease","name":"goal_server","seconds":6}` | a laptop process is here; once a lease was taken and all have lapsed, the face falls asleep |
| `{"cmd":"config","imu_rate_hz":500,"dlpf":3,"accel_fs":1,"gyro_fs":1,"brightness":120}` | the head IMU's sampling and the screen's brightness, until a restart (`config/head.json` for good) |
| `{"cmd":"subscribe","imu":true}` | this client gets `{"type":"imu_config","cfg":1,"rate_hz":200,"dlpf":3,"gyro_fs_dps":500,"accel_fs_g":4,"filter_delay_s":0.0048}` first (and on every config change), then one line per serial frame (every 20 ms): `{"type":"imu","cfg":1,"samples":[[t_mono_s, esp_us, gx, gy, gz, ax, ay, az], ...]}`, `t_mono_s` the sample's data-ready edge on the board's `time.monotonic` (the filter delay NOT taken off), `esp_us` the ESP32's micros unwrapped, rad/s and m/s^2 in the chip's axes |
| `{"cmd":"status"}` | the link, the ESP32's own status (fps, IMU rate and mode, I2C errors, drops), the clock map (offset, ppm, smallest ping round trip), what shows and who asked; also broadcast once a second |

The clock map: every pong and every IMU frame gives (the moment its frame was read here on
CLOCK_MONOTONIC_RAW, the ESP32's micros in it), less what is known of its way (the frame's own
bytes on the wire, an IMU sample's I2C read); the smallest difference per second of ESP time is
the lower envelope, a line through the last 120 s of it carries the offset and the crystal's
drift (RAW is never slewed by chrony, so the skew is the crystal's alone), half the smallest ping
round trip is taken off, and each line's samples are carried to `time.monotonic` by reading both
clocks back to back. The ESP32 rebooting (its micros going back, or a status naming another
config) makes the server send the config and the face again.

## The neck on the base server's port (:3336)

The two neck servos hang on the same bus as the wheels, so the base server is the only thing that
may talk to them, and they ride the wheels' tick. Their encoders are read in the SAME sync_read as
the wheels, after them and waited for only `motion.read_window_ms` (3 ms) past the wheels'
replies: a dead head costs the odometry that window, never a retry, and three silent reads take
it out of the read until it is asked again `motion.retry_s` (5 s) later. Both ticks go out in
every state line (`"pan_ticks"`, `"tilt_ticks"`) under the odometry's own stamp, and the base
bridge turns them into `/neck/state` and `base_link -> camera_link`. Every neck write is one
unacknowledged sync_write, at most one a tick — the Torque_Enable..Goal_Velocity block (torque,
ramp, goal, speed in one packet) when the head is energised or changes pace, else a Goal_Position
— so nothing a servo can fail to answer runs on the wheels' thread, and the head moves while the
cart drives. The operating mode is read once each time the pair (re)appears; a servo that is not
in position mode (`scripts/jog.py wheel` writes velocity mode into its EEPROM, and the head would
turn forever) is never given a goal. A pair in position mode then has its Maximum_Acceleration
(register 85) lifted to 254 and read back, in RAM (the EEPROM lock stays on, so every power-up
gets it again): the servos came with 50, 439 deg/s^2, and stored every larger Acceleration as
50, so the motion's ramp did nothing above it.

| Line | What happens |
| --- | --- |
| `{"cmd":"neck"}` | answers `{"type":"neck","pan_ticks":..,"tilt_ticks":..,"age_s":..,"read_ms":..}` from the tick's own read: no bus traffic of its own |
| `{"cmd":"neck_target","pan_rad":..,"tilt_rad":..,"speed_deg_s":..,"acc_deg_s2":..}` | the gaze arbiter's stream: hold the head at these joint angles (`pepin.neck`: pan positive left, tilt the pitch below level, 23.8 deg at home) for the lease, `motion.lease_s` (2 s); speed and ramp are optional ceilings under the motion's. No target for the lease: the head goes home at the full pace and is let go on arrival. Accepted silently, like a twist; refused as `{"type":"neck_target","error":".."}` (outside the reach, never clamped; a jog holds the head; the servos silent or not in position mode) |
| `{"cmd":"neck_goto","pan_ticks":N,"tilt_ticks":N,"hold":false}` | moves the head there at the motion's pace and answers `{"type":"neck_goto","pan_ticks":..,"tilt_ticks":..,"reached":bool,"ms":..,"hold":bool}` when it arrives or gives up (half again its travel time plus a second, 3 s at least). Either target may be `null` to hold that servo where it is |
| `{"cmd":"neck_home"}` | the same move to the reference pose of `config/neck.json` (`reference.pan_ticks` / `tilt_ticks`), the pose the camera mount was measured in |
| `{"cmd":"neck_jog","pan":-1\|0\|1,"tilt":-1\|0\|1,"slow":false}` | walks the head at a rate (52 deg/s, `slow` 8 deg/s) in those directions — pan +1 left, tilt +1 down, the signs of `pepin.neck.NeckAngles` — for as long as the lines keep coming: each one re-arms the jog's own 0.5 s deadman, after which the head stops where it is and the servos are released. Both zero: stop where it is, still held until the deadman. Accepted silently; refused as `{"type":"neck_jog","error":".."}` |
| `{"cmd":"neck_motion","max_speed_deg_s":..,"max_acc_deg_s2":..,"tilt_max_acc_deg_s2":..,"lease_s":..}` | the motion in force beside `config/neck.json`'s (`tilt_max_acc_deg_s2`: the tilt's ramp under `max_acc_deg_s2`); each key given is set until the server restarts (`ros/neck.sh motion`), a value outside its range refuses the whole line |
| `{"cmd":"ping"}` | the servo roster, ids 1–10 (refused while the wheels turn: a ping waits for its answer) |
| `{"cmd":"registers","servo":"neck","address":85,"size":1}` | `{"type":"registers","servo":..,"address":..,"values":[..]}`: raw bytes (at most 64) of a roster servo's control table, one acknowledged read (`ros/neck.sh registers`); refused as `"busy":true` while the wheels turn |

Who holds the head, one at a time: a jog is the operator's and takes it from a lease; a lease takes
it from a move, which answers `"error":"preempted by neck_target"`; a move is refused while a lease
or a jog holds it; a move and a jog refuse each other. None of them looks at the wheels. A target
or a move is never made smaller: outside the limits in `config/neck.json` (pan 257–3812, tilt
1814–2760 ticks) it is refused, a wrong number being a mistake. The jog's goal is clamped to the
limits instead (it is walking, not aiming), and it never winds the servo up: a goal more than
~10 deg ahead of the encoder waits for the head. A move's answer is broadcast to every client of
the port, because it may come seconds after the request. WiFi lost: a jog stops within 0.5 s and
lets go, a lease goes home within its 2 s.

## The I2C bus

`/dev/i2c-2` is TWI3 (`5002c00.i2c`, driver `mv64xxx_i2c`) on PH4 = SCL (header pin 5) and PH5 =
SDA (pin 3), at 400 kHz (`board/i2c3-400k.dts`). On it: the three VL53L1X, front 0x30 (always
awake: its XSHUT is PC9, which the kernel will not drive), left 0x31 (XSHUT PC5) and right 0x32
(XSHUT PC6), and the base MPU6050 at 0x68. Three programs use it: `tof_init.sh` (at boot and
before every pepin-tof start: XSHUT, `i2cdetect`, `i2ctransfer`), `pepin.tof_server` (the ranges
at 15 Hz) and the C++ base bridge in `pepin-ros` (the IMU at 100 Hz).

**Serialisation is the kernel's.** The I2C core holds the adapter's lock for a whole transfer, and
every register read of every client is ONE transfer (`I2C_RDWR`: the register, a repeated START,
the read — the VL53L1X library always, the bridge since 2026-10-05), so no client's transaction
can be cut by another's, on the wire or in the driver. There is no file lock on top: it would
order nothing the adapter lock does not, and a client stuck in a 2 s timeout blocks the others
either way. Start order does not matter either: the bridge probes the IMU until it answers
(five tries a second apart, then one a minute) and recovers it live.

**A locked bus** reads `mv64xxx: I2C bus locked, block: 1, time_left: 0` in dmesg every 2 s (a
1 s timeout and a 1 s abort per transfer) and `Connection timed out` in every client: a slave
lost its place mid-byte and drives SDA low, so the controller can never make a START. A driver
unbind/rebind resets the controller, not the slave (and the unbind waits for every open
`/dev/i2c-2`). The driver calls the I2C core's bus recovery at every lock, but on this SoC the
core has no pins to drive: its generic recovery needs `scl-gpios` and a `gpio` pinctrl state, and
the sunxi pinmux is strict (`sunxi_pmx_ops.strict`, no `function_is_gpio`, Linux 6.18), so a GPIO
request on a pin muxed to i2c3 is refused; no overlay enables it on this kernel. Episodes so far,
all on a parked cart with the stack running, none at a restart: 2026-09-25 08:23 (100 kHz),
2026-10-04 11:32 (begun by an arbitration loss reading 0x68; power cycle at 15:03), 2026-10-05
08:38 (ended at 14:35:52 by a pepin-tof restart: tof_init's XSHUT reset of 0x31/0x32).

Freeing it, cheapest first (as root on the board; `/opt/pepin/pepin` is not installed in the
`/opt/pepin` venv, so the working directory and `PYTHONPATH` are the units' own):

```bash
# SDA/SCL from the controller's line register (50 reads over ~50 ms); who has the bus open
cd /opt/pepin && PYTHONPATH=/opt/pepin /opt/pepin/bin/python -m pepin.i2c_recover check
systemctl stop pepin-tof
# nine clocks on SCL and a STOP (TWI_LCR, through /dev/mem)
cd /opt/pepin && PYTHONPATH=/opt/pepin /opt/pepin/bin/python -m pepin.i2c_recover recover
systemctl start pepin-tof    # tof_init: XSHUT-resets 0x31/0x32, re-addresses all three
```

`check` calls a line held only when it is low in every read; a live bus (the IMU at 100 Hz, three
ToF) reads `busy`, exit 0. `recover` refuses while anything holds `/dev/i2c-2` open (`--force` overrides); the bridge closes
it by itself once five IMU reads in a row have failed. A bus still held after both steps (SCL low,
or 0x30 or 0x68 hung beyond clocks) needs the power cycle. The bridge's minute probe finds the IMU
again with no restart.

## Setting up a fresh board

```bash
apt install ser2net i2c-tools gpiod ffmpeg v4l-utils ustreamer
python3 -m venv /opt/pepin && /opt/pepin/bin/pip install VL53L1X smbus2
# copy the files from the table above, then:
systemctl daemon-reload
systemctl enable --now ser2net tof-init pepin-tof pepin-base pepin-camera pepin-audio pepin-head pepin-reap.timer
# from the laptop: the image and pepin-ros.service, the code, the clock
ros/build-image.sh --ship
ros/sync.sh
ros/time.sh install
# on the board again: the router and the sensor stack, at every boot from now on
systemctl enable --now pepin-zrouter pepin-ros
```

Notes that cost an evening each:

- `tof-init.service` must not order itself after `multi-user.target` (it is wanted by it): that
  ordering cycle made systemd drop the ToF service at boot.
- `/var/lock` is on the SD card on this image, so a UUCP lock file survives a power cut and ser2net
  then refuses the serial port, once with its own pid in the file, reused across the reboot.
  `ser2net.yaml` disables the locks (`nouucplock`); the drop-in still sweeps both lock styles.
- The base server waits for the servo bus instead of exiting: the servos may be powered minutes
  after the board, and a crash loop is not a state.
- `tof_init.sh` drives XSHUT high explicitly (a released GPIO line keeps its last driven level) and
  verifies its work: 0x30, 0x31 and 0x32 must each answer with the VL53L1X model id, else the
  sequence runs again (`TOF_INIT_ATTEMPTS`, default 3). The outcome is one line in the journal and
  in `/var/log/tof_init.log`, because the stack's logging rotates the 20 MB journal within hours.
- The wifi power-save flag and the wifi chip's runtime power management are both switched off;
  latency spikes of 300–600 ms remain on this radio and are the reason the wheel loop lives on the
  board.

Check from the laptop: `uv run python scripts/health_check.py --quick`.

## Stray ros2 CLI tools

`ros/reap_ros2_cli.sh` (systemd timer `pepin-reap.timer`, every minute) kills `ros2
topic|param|run|node|service|...` processes older than 90 s inside `pepin-ros`: a `timeout` around
a CLI probe ends the tool but not always its DDS shutdown, and a dozen leftovers took the board to
load 12 (2026-09-13). Nodes are never touched. The unit runs the script from `/root/pepin-ros/`,
where `ros/sync.sh` mirrors `ros/` (a unit runs nothing from `/root/pepin-ros/` that `ros/` does
not hold: `tests/unit/test_deployment.py`). Operator rule: probe the board with `timeout -s KILL`
and one tool at a time.
