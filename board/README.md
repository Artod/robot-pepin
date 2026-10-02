# Board: Orange Pi Zero 3 as the robot's relay

The board runs Armbian and does nothing clever: it bridges the servo bus and the
lidar to TCP, streams the ToF ranges, serves the camera, and — the one real-time
job — owns the wheels. Everything else runs on the laptop.

## Services

| Port | Service | Unit | What it does |
| --- | --- | --- | --- |
| 3333 | ser2net | `ser2net.service` | raw TCP to `/dev/servo-bus` (Feetech bus, 1 Mbit/s); `kickolduser` hands the port to the newest client |
| 3334 | ser2net | `ser2net.service` | raw TCP to `/dev/lidar` (LD19, 230400 baud) |
| 3335 | `pepin.tof_server` | `pepin-tof.service` | JSON lines with the three VL53L1X ranges at 15 Hz; needs `tof-init.service` first |
| 3336 | `pepin.base_server` | `pepin-base.service` | owns the wheels: reads encoders and applies twists at 50 Hz over loopback to :3333, deadman 0.5 s, publishes odometry state at 20 Hz |
| 8080 | ustreamer | `pepin-camera.service` | MJPEG stream and `/snapshot` of the overview camera |

The base server is the only client of :3333 while it runs. Bench tools that
talk to the servo bus directly (`scripts/base_smoke.py`, `jog.py`,
`setup_motor_id.py`, `calibrate_neck.py`) must be run with it stopped:
`systemctl stop pepin-base` before, `systemctl start pepin-base` after.

## Files and where they go

| In the repo | On the board |
| --- | --- |
| `board/ser2net.yaml` | `/etc/ser2net.yaml` |
| `board/99-pepin-usb.rules` | `/etc/udev/rules.d/99-pepin-usb.rules` (then `udevadm control --reload`) |
| `board/tof_init.sh` | `/usr/local/bin/tof_init.sh` (executable) |
| `board/tof-init.service` | `/etc/systemd/system/tof-init.service` |
| `board/pepin-tof.service` | `/etc/systemd/system/pepin-tof.service` |
| `board/pepin-base.service` | `/etc/systemd/system/pepin-base.service` |
| `board/pepin-ros.service` | `/etc/systemd/system/pepin-ros.service` (branch ros2-nav2: the ROS container at boot) |
| `board/ser2net-stale-locks.conf` | `/etc/systemd/system/ser2net.service.d/stale-locks.conf` |
| `board/wifi-runtime-pm-on.conf` | `/etc/systemd/system/wifi-powersave-off.service.d/runtime-pm-on.conf` |
| `src/pepin/` (the package, stdlib only on the board) | `/opt/pepin/pepin/` |
| `config/base.json` | `/opt/pepin/config/base.json` |
| `config/neck.json` | `/opt/pepin/config/neck.json` (the ids the `neck` command reads and the limits the move commands obey; absent, those commands answer an error and the wheels do not care) |

Deploy the package and the configuration from the laptop:

```bash
rsync -a --delete --exclude '__pycache__' src/pepin/ root@pepin.local:/opt/pepin/pepin/
scp config/base.json config/neck.json root@pepin.local:/opt/pepin/config/
ssh root@pepin.local 'systemctl restart pepin-base pepin-tof'
```

## The neck on the base server's port (:3336)

The two neck servos hang on the same bus as the wheels, so the base server is the only thing
that may talk to them. Five JSON lines, from anywhere that can reach the port — `ros/neck.sh`
is the shell around the first three, the game-mode teleop (`pepin.teleop --game`) around the
jog:

| Line | What happens |
| --- | --- |
| `{"cmd":"neck"}` | answers `{"type":"neck","pan_ticks":..,"tilt_ticks":..,"age_s":..,"read_ms":..}`: the encoders, cached, at most twenty bus reads a second however many clients ask |
| `{"cmd":"neck_goto","pan_ticks":N,"tilt_ticks":N,"hold":false}` | moves the head there and answers `{"type":"neck_goto","pan_ticks":..,"tilt_ticks":..,"reached":bool,"ms":..,"hold":bool}` when it arrives or after 3 s. Either target may be `null` to leave that servo alone |
| `{"cmd":"neck_home"}` | the same move to the reference pose of `config/neck.json` (`reference.pan_ticks` / `tilt_ticks`), the pose the camera mount was measured in |
| `{"cmd":"neck_jog","pan":-1\|0\|1,"tilt":-1\|0\|1,"slow":false}` | walks the head at a rate (52 deg/s, `slow` 8 deg/s) in those directions — pan +1 left, tilt +1 down, the signs of `pepin.neck.NeckAngles` — for as long as the lines keep coming: each one re-arms the jog's own 0.5 s deadman, after which the head stops where it is and the servos are released. Both zero: stop where it is, still held until the deadman. Accepted silently, like a twist; refused as `{"type":"neck_jog","error":".."}` |
| `{"cmd":"ping"}` | the servo roster, ids 1–10 |

What the move side refuses, rather than doing something smaller: a target outside the limits in
`config/neck.json` (pan 257–3812, tilt 1814–2760 ticks — never clamped, a wrong number is a
mistake); a move while the wheels turn (a write to a silent servo costs the wheel loop 0.4 s, and
the deadman lives on that thread); a second move while one is under way; a servo that is not in
position mode, which `scripts/jog.py wheel` writes into a servo's EEPROM and which would turn the
head forever. The move itself runs one short bus transaction per 20 ms tick, so it never delays
the wheels; the servos are released again when the head arrives, unless `"hold":true` asked them
to keep the pose. Its answer is broadcast to every client of the port, not only to the one that
asked — it is born when the head stops, and that may be seconds after the request.

The jog obeys the same rules and differs where a rate differs from a target: its goal is clamped
to the limits (it is walking, not aiming, so the limit is where it stops), it is refused while
the wheels turn and ends the tick they start — the goal freezes at once, the torque comes off at
the first tick at rest, so a tick that drives the wheels never waits on a neck servo — a jog and
a move refuse each other, the servo is never wound up (a goal more than ~10 deg ahead of the
encoder waits for the head), a tick that stalled advances the goal by at most two ticks' worth,
and encoders that fail mid-jog end it with torque off and an error line. Its cost on the board
while a key is held, counted on the unit tests' fake bus: per 20 ms tick ONE `Goal_Position`
write carrying every axis that moved (one transaction for both axes, not one each), plus the
cached encoder read at most every 50 ms (the same read a move or a polling client costs). In
milliseconds, from what the base server measured on 2026-09-11 (a two-servo neck read 9.7 ms;
the wheel loop's own read and write p95 5.6 ms, so an unacknowledged write is 1–2 ms): a tick
with the read ≈ 12 ms, a tick without ≈ 2 ms, ≈ 6 ms on average — with one write per axis it
was ≈ 13 / 3 / 7. Nothing when idle. WiFi lost mid-jog: the head stops within 0.5 s and lets go.

## Setting up a fresh board

```bash
apt install ser2net i2c-tools gpiod ffmpeg v4l-utils ustreamer
python3 -m venv /opt/pepin && /opt/pepin/bin/pip install VL53L1X smbus2
# copy the files from the table above, then:
systemctl daemon-reload
systemctl enable --now ser2net tof-init pepin-tof pepin-base pepin-camera
```

Notes that cost an evening each:

- `tof-init.service` must not order itself after `multi-user.target` (it is wanted
  by it): that ordering cycle made systemd drop the ToF service at boot.
- `/var/lock` is on the SD card on this image, so a UUCP lock file survives a
  power cut and ser2net then refuses the serial port — once with its own pid in
  the file, reused across the reboot ("Port in use by pid 1003"), which no
  stale-lock check catches. `ser2net.yaml` disables the locks (`nouucplock`);
  the drop-in still sweeps both lock styles before ser2net starts.
- The base server waits for the servo bus instead of exiting: the servos may
  be powered minutes after the board, and a crash loop is not a state.
- `tof_init.sh` documents the pinctrl quirk: a released GPIO line keeps its last
  driven level, so XSHUT must be driven high explicitly.
- `tof_init.sh` verifies its work: 0x30, 0x31 and 0x32 must each answer with the
  VL53L1X model id, else the XSHUT sequence runs again (`TOF_INIT_ATTEMPTS`,
  default 3). The outcome is one line in the journal and in `/var/log/tof_init.log`,
  because the stack's own logging rotates the 20 MB journal within hours (the
  day all three read status 255, nothing of that morning's tof-init was left).
- The wifi power-save flag and the wifi chip's runtime power management are both
  switched off; latency spikes of 300–600 ms remain on this radio and are the
  reason the wheel loop lives on the board.

Check from the laptop: `uv run python scripts/health_check.py --quick`.

## The retired bridge units

The CycloneDDS transport ran two zenoh-bridge-ros2dds sidecars (`pepin-bridge`, and the
laptop's kick to it, `pepin-bridge-kick.path`). They are on the tag
`alt/cyclone-bridges-2026-09-20`; `ros/thin.sh` disables them on a board that still has them.

## Stray ros2 CLI tools

`board/reap_ros2_cli.sh` (systemd timer `pepin-reap.timer`, every minute) kills `ros2 topic|param|run|node|service|...` processes older than 90 s inside `pepin-ros` — a `timeout` around a CLI probe ends the tool but not always its DDS shutdown, and a dozen leftovers took the board to load 12 (2026-09-13). Nodes are never touched. Install: copy `board/pepin-reap.{service,timer}` to `/etc/systemd/system/`, the script to `/root/pepin-ros/board/`, `systemctl enable --now pepin-reap.timer`. Operator rule: probe the board with `timeout -s KILL` and one tool at a time.
