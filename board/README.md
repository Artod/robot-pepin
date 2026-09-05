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
| `board/ser2net-stale-locks.conf` | `/etc/systemd/system/ser2net.service.d/stale-locks.conf` |
| `board/wifi-runtime-pm-on.conf` | `/etc/systemd/system/wifi-powersave-off.service.d/runtime-pm-on.conf` |
| `src/pepin/` (the package, stdlib only on the board) | `/opt/pepin/pepin/` |
| `config/base.json` | `/opt/pepin/config/base.json` |

Deploy the package and the configuration from the laptop:

```bash
rsync -a --delete --exclude '__pycache__' src/pepin/ root@pepin.local:/opt/pepin/pepin/
scp config/base.json root@pepin.local:/opt/pepin/config/base.json
ssh root@pepin.local 'systemctl restart pepin-base pepin-tof'
```

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
- `/var/lock` is on the SD card on this image, so a UUCP lock file survives an
  unclean shutdown and ser2net then refuses the serial port; the drop-in clears
  stale locks before ser2net starts.
- `tof_init.sh` documents the pinctrl quirk: a released GPIO line keeps its last
  driven level, so XSHUT must be driven high explicitly.
- The wifi power-save flag and the wifi chip's runtime power management are both
  switched off; latency spikes of 300–600 ms remain on this radio and are the
  reason the wheel loop lives on the board.

Check from the laptop: `uv run python scripts/health_check.py --quick`.
