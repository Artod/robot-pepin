# Head IMU calibration

The head carries an MPU6050 glued to the stereo camera module, read by the head ESP32 and served by
`pepin.head_server` on TCP 3340. The base bridge publishes it as `/head/imu`; the visual-inertial
odometry (OpenVINS, `ros/laptop.sh vio`) and the mast-sway filter use it. Three things about it are
measured, each with its own procedure, and each lands in exactly one place:

| What | Where it lives | Procedure |
| --- | --- | --- |
| The IMU's noise (white noise, bias random walk) | `config/head_imu.json` `noise` | parked Allan recording, below |
| The camera-IMU extrinsics and time offset | `config/camera.json` `stereo.head_imu` | Kalibr, below |
| The sign of the sway correction | the bridge's `mast_sway` flag stays off until it passes | tilt-sway check, below |

Every generated file (OpenVINS's three YAMLs, Kalibr's `camchain.yaml`, `imu.yaml`, `april.yaml`)
comes from `ros/tools/vio_config.py`; nothing is edited by hand.

## 0. Before Kalibr: the nominal extrinsics

Photograph the glued chip and note which way its axes point relative to the camera: for each chip
axis (x, y, z, printed on the GY-521), the camera's optical axis it points along (optical x is the
picture's right, y its down, z the viewing direction, all for the upright rectified left eye). Then:

```bash
uv run python ros/tools/vio_config.py --nominal "x,-z,y" 0.0 0.03 0.04
```

`AXES` lists the optical axis of chip x, y and z in turn; the three numbers are the chip's position
in the left eye's optical frame (metres, tape). The files say `NOMINAL` until Kalibr replaces them.
The live relay composes through TF, so the same guess also goes into `config/camera.json` as the
`stereo.head_imu` block (camera_stream then publishes `camera_optical -> head_imu`):

```bash
uv run python ros/tools/vio_config.py --print-block --nominal "x,-z,y" 0.0 0.03 0.04
```
Sign check: pan the head left by hand (torque off) and see which `/head/imu` axis reads positive;
tilt down, the same. Write both into the `head_imu` block's `note` when it is created.

## 1. Noise: the parked Allan recording (2-3 h)

Cart parked, head at home, the stack running as it runs (the servos hold torque). Either source:

```bash
# head_server's raw lines (no ROS involved), from the laptop:
uv run python -m pepin.head_imu record --host 10.0.0.187 --seconds 10800 --out parked.jsonl.gz
# then
uv run python -m pepin.allan parked.jsonl.gz --plot allan.png --inflate 10
```

The tool prints each axis's white noise density, random walk and bias instability, refuses a
recording with more than 0.1 % gaps, and prints the `noise` block (worst axis, inflated 10x for the
VIO as OpenVINS advises). Put it into `config/head_imu.json` and regenerate the VIO's files. The
starting block (rtabmap's OdomOpenVINS defaults, about 5-6x the base IMU's measured white noise)
stays until then.

## 2. The target

Print one of the AprilGrids at **100 % scale** ("actual size", no fit-to-page):

| File | Paper | Tag size as drawn |
| --- | --- | --- |
| `docs/aprilgrid/aprilgrid_6x6_34mm_A3.pdf` | A3 | 34 mm |
| `docs/aprilgrid/aprilgrid_6x6_24mm_A4.pdf` | A4 (hand-held at 0.4-0.7 m) | 24 mm |

Both are Kalibr's own `kalibr_create_target_pdf --type apriltag --nx 6 --ny 6 --tspace 0.3`
output (tag family 36h11). Glue the print flat on foam board, then **measure one tag's black square
with a ruler**: printers scale. Kalibr's target file takes that number:

```bash
uv run python ros/tools/vio_config.py --kalibr-only --tag-size 0.0341   # metres, measured
```

## 3. The recording

Both procedures need the capture stamps and a short exposure, or the time offset is meaningless:

```bash
ros/flags.sh set camera_stream camera_stamp grab      # ustreamer's capture stamp, not its send time
ros/exposure.sh capped 8                               # on the board; ros/exposure.sh auto afterwards
ros2 bag record /camera/image /camera/right/image /head/imu    # 60-90 s per run
```

**Procedure D (first, twice), on the robot.** The grid on a chair 0.8-1.0 m ahead, its centre about
0.8 m up (the lens looks 23.8 deg down from 1.2 m), good light, nothing else moving in view. Then:

```bash
uv run python ros/tools/neck_dance.py                # the views, nothing moves
uv run python ros/tools/neck_dance.py --move         # the gaze arbiter: pans +-45, tilts 20 up..60 down
```

followed by four +-0.3 m shuttles at 0.1 m/s and four +-30 deg pivots at 0.3 rad/s (teleop or
`ros/goto.sh`). The two runs must agree within 0.5 deg of rotation, 5 mm of translation and 2 ms of
time offset.

**Procedure C (only if D's two runs disagree by more than 0.5 deg), hand-held.** The head off its
bracket with both USB leads attached, moved in front of the grid at 0.4-0.8 m: slow rotations about
each axis (+-30 deg, about one per second), then translations along each (+-20 cm), then
figure-eights; the grid mostly in view. The camera runs at 10 Hz, half Kalibr's "good" rate: move at
half the usual speed. Taking the head off risks shifting the taped module (then the eye block of
`config/camera.json` must be measured again), which is why D comes first.

## 4. Kalibr

Kalibr is ROS 1. Convert the bag and run it in Kalibr's Docker image (built for arm64 from
ethz-asl/kalibr's `Dockerfile_ros1_20_04` with `--platform linux/arm64`):

```bash
uv run --with rosbags rosbags-convert --src calib_d1 --dst calib_d1.bag      # ROS 2 -> ROS 1
docker run --rm -it -v "$PWD:/data" -v "$PWD/ros/maps/vio:/cfg:ro" kalibr \
    rosrun kalibr kalibr_calibrate_imu_camera --bag /data/calib_d1.bag \
        --cam /cfg/camchain.yaml --imu /cfg/imu.yaml --target /cfg/april.yaml
```

`camchain.yaml` describes the rectified, upright eyes exactly as `/camera/image` and
`/camera/right/image` carry them (the rectified pinhole of `config/stereo_calibration.json`, no
distortion, cam1 +baseline along cam0's x), so no new intrinsic calibration is needed. An optional
first run of `kalibr_calibrate_cameras` on the same recording must return near-zero distortion and a
61 mm baseline: a free check of the rectification.

Accept when the mean reprojection error is under 0.5 px, the gyro and accelerometer error plots look
white, and the two runs agree as above. With capture stamps expect a time shift of a few
milliseconds plus half the exposure.

## 5. Writing it down

Kalibr's result is `T_cam_imu` (a point in the IMU's axes into cam0's optical frame, the rectified
left eye) and `timeshift_cam_imu`. Store it as Kalibr gives it, relative to the eye (the IMU is
glued to the module and moves with the eyes when the module is re-taped):

```json
"head_imu": {
  "T_cam_imu": [[...], [...], [...], [0.0, 0.0, 0.0, 1.0]],
  "time_offset_s": 0.0041,
  "date": "YYYY-MM-DD",
  "method": "kalibr_calibrate_imu_camera, procedure D x2",
  "note": "..."
}
```

in `config/camera.json`'s `stereo` block, then:

```bash
uv run python ros/tools/vio_config.py          # OpenVINS's files into ros/maps/vio
ros/laptop.sh kick camera_stream               # publishes camera_optical -> head_imu
ros/laptop.sh vio                              # OpenVINS on the new files
```

`config/camera.json` refuses a block that is not a rigid transform (pepin.camera.head_imu_transform),
and `vio_config.py` inverts it for OpenVINS's `T_imu_cam` with a test on an asymmetric transform.

## 6. Grid-free checks

On the same recording: the time offset from the cross-correlation of the head gyro with the image's
own rotation rate (phase correlation, 0.216 deg/px) must agree with Kalibr's within 2 ms, and the
rotation axes (Kabsch on the two rate series) within 0.5 deg.

## 7. The sway correction's sign

The bridge's `mast_sway` flag composes the mast's sway into `base_link -> camera_link`; a wrong sign
doubles the error instead of removing it. Parked, with the extrinsics in place: hard tilts (the cap
lifted to 2232 deg/s^2 for the test only), the image's ring measured by phase correlation against
`/mast/state`'s pitch. Same sign and phase, at least 70 % of a 0.3-0.4 deg ring removed, lag at most
10 ms, theta at rest at most 0.02 deg rms over 60 s: only then `ros/flags.sh set base_bridge
mast_sway true`.
