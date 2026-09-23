# Maps and recordings

`*.yaml` + `*.pgm` are the served maps (`PEPIN_MAP` in `/etc/default/pepin-ros` names the one the
board serves); `*.places.yaml` is that map's book of remembered places. `rec/` holds the drives
and is not tracked.

## Two clocks in `rec/`, and which is which

A drive leaves two recordings, written by two processes with two clocks:

| File | Written by | Clock |
| --- | --- | --- |
| `<stamp>_goto.jsonl`, `_goto.log`, `_goto_cam.mkv`, `_goto_board.log` | `ros/goto.sh`, on the laptop | the laptop's local time |
| `0249_<stamp>Z_<place>.jsonl`, `_cam.mjpeg` | `pepin_bringup.run_recorder`, in the board's container | **UTC**, marked by the `Z` |
| `0249_<stamp>Z_<place>/` (an MCAP bag) and the `.jsonl` made from it | `ros2 bag record` under `pepin_bringup.bag_recorder` with `PEPIN_RECORDER=bag`, converted on the laptop by `ros/tools/bag_to_tape.py` | **UTC** too: the same stem, from the same container clock |

The board's own shell is on local time; only the container it runs ROS in is on `Etc/UTC`, so
`time.strftime` there is four or five hours ahead of everything a person reads beside it. A tape
named `0248_20260913_220039_home` was taken for a 22:00 drive and was in fact the 18:00 one
(2026-09-13). Hence the `Z`: the letter is the whole warning.

Two ways out of the two clocks, both open:

* give the container a timezone (`TZ` in its environment, or the host's `/etc/localtime` mounted
  in) — `run_recorder` asks for UTC explicitly, so drop the `time.gmtime()` there as well and
  every name is local;
* or read the numbered tape's name out of the drive's own log instead of off the clock:
  `ros/goto.sh` prints `run 0249: taped /maps/rec/0249_…` into `<stamp>_goto.log` beside the
  goal, and `ros/go.sh` prints `=== run #0249 ===` with the fetched paths under it. Neither
  line needs a timezone.

## Which tape answers which question

The numbered tape is the one the replays and the reports are named by: it carries the costmap,
the EKF and the IMU records a stall has to be judged from, and the camera clip beside it. The
`_goto` session log carries the scans, the odometry, the tracked pose, the commands, the camera's
measurements and the tracker's account of each update, and it exists whether or not the recorder
answered. Both are opened on every drive through `ros/goto.sh` and `ros/go.sh`.
