# Maps and recordings

`*.yaml` + `*.pgm` are frozen maps (`ros/map.sh save NAME`); `*.places.yaml` is a map file's book
of remembered places. `PEPIN_MAP` (`ros/laptop.sh nav`) names the map whose book the goal server
reads; the live map is RTAB-Map's database, `rtabmap.db`, whose places ride its graph nodes.
`rec/` holds the drives and is not tracked.

## Two clocks in `rec/`, and which is which

| File | Written by | Clock |
| --- | --- | --- |
| `<stamp>_goto.log`, `_goto.nav2.log`, `_goto_cam.mkv` | `ros/goto.sh`, on the laptop | the laptop's local time |
| `0249_<stamp>Z_<place>.jsonl` | `pepin_bringup.run_recorder`, in the laptop's Nav2 container (`pepin-macnav`) | **UTC**, marked by the `Z` |
| `0249_<stamp>Z_<place>/` (an MCAP bag) and the `.jsonl` made from it | `ros2 bag record` under `pepin_bringup.bag_recorder` with `PEPIN_RECORDER=bag`, converted by `ros/tools/bag_to_tape.py` | **UTC** too: the same stem, from the same container clock |

The ROS containers run on `Etc/UTC` while a person reads local time beside them, so
`time.strftime` there is four or five hours ahead. A tape named `0248_20260913_220039_home` was
taken for a 22:00 drive and was in fact the 18:00 one (2026-09-13). Hence the `Z`: the letter is
the whole warning. The drive's own log needs no timezone: `ros/goto.sh` writes `run 0249: taped
/maps/rec/0249_…` into `<stamp>_goto.log`, and `numbered tape: ros/maps/rec/0249_…` when the
drive ends.

## Which file answers which question

The numbered tape is the one replays and reports are named by: it carries the scans, the
odometry, the EKF and the IMU, the ToF ranges, the plan, both costmaps and the commands a stall
has to be judged from. `<stamp>_goto.log` is the goal's own lines, `<stamp>_goto.nav2.log` Nav2's reasons and the
behaviour tree's transitions, and the `.mkv` the head camera.
