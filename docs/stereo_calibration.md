# Stereo head calibration with Kalibr

The stereo module sends one side-by-side MJPEG frame: two 800x600 eyes from two global shutters
captured at the same instant. `config/stereo_calibration.json` holds what a calibration measured
about it: each eye's pinhole and distortion, plus the right eye's pose in the left eye's frame
(OpenCV's `x_right = R x_left + T`). `camera_stream` builds the rectifier from that file
(`pepin.stereo.Rectifier`), and every depth pixel, VO pose and RTAB-Map frame goes through it.

This procedure measures the file with Kalibr's `kalibr_calibrate_cameras` on the RAW eyes, before
rectification. The eyes are cut and turned upright exactly as `pepin.stereo.SideBySide` does it,
so the result replaces the file's K, D, R and T wholesale. The chessboard procedure
(`ros/calibrate.sh stereo`) is still there as the alternative.

| Step | Command | Output |
| --- | --- | --- |
| Record | `ros/calib_stereo.sh record [--seconds 90]` | `ros/maps/rec/stereo_<UTC>Z/stereo.mjpeg` + `meta.json` |
| Calibrate | `ros/calib_stereo.sh run REC TAG_M` | `REC/kalibr/stereo-camchain.yaml`, `-results-cam.txt`, `-report-cam.pdf`, the report |
| Write | `ros/calib_stereo.sh run REC TAG_M --apply` | `config/stereo_calibration.json` (and the carried `camera.json` blocks) |

## 1. The recording

`record` copies ustreamer's stream byte for byte (`?extra_headers=1`; the format of the drives'
camera clips, which `pepin.mjpeg.parts` reads back). It adds one more reader on the board's
ustreamer and touches nothing else. Every two seconds it prints the frame rate and a rough tag
count per eye (OpenCV's AprilTag detector finds fewer tags than Kalibr's; a 0 means that eye does
not see the board).

Stamps don't matter: both eyes of a pair are halves of one transport frame, and a camera
calibration never compares time across pairs. So unlike `ros/calib_record.sh`, `record` doesn't
need `camera_stamp grab`; it keeps the capture stamp when the stream carries one.

How to move the board (printed at the start):

- 0.25 to about 0.65 m from the lens for the 25 mm A4 print. Kalibr detects nothing below about
  20 px per tag, which is 0.66 m at f 530. The A3 print (34 mm) reaches about 0.9 m.
- Tilted ±30 deg about both axes, not only flat. Tilt is what pins the focal length, and the
  failed 10-04 run (one distance, small angles) could not initialise it.
- Over all parts of the picture: centre, the four corners, the edges. The distortion lives in
  the corners.
- Slowly, with a pause at each pose, for 60-90 s. Both eyes must see the whole grid most of the
  time. Whole-grid views are also what Kalibr's focal initialisation needs.

## 2. Kalibr

`run` does the following:

1. Takes the sharpest frame of every 1/hz window (default 4 Hz; the variance of the Laplacian of
   the worse eye), cuts it into the two upright eyes, converts to mono8, and writes a ROS 1 bag
   with `/cam0/image_raw` (left) and `/cam1/image_raw` (right), one stamp per pair.
2. Writes `april.yaml`: 6x6, `tagSize` = TAG_M (the black square measured with a ruler),
   `tagSpacing` 0.3.
3. Runs `kalibr_calibrate_cameras --models pinhole-radtan pinhole-radtan` in `pepin-kalibr`
   (prehensile/kalibr:arm64, pinned). If the focal initialisation fails, Kalibr reads a guess from
   stdin (`KALIBR_MANUAL_FOCAL_LENGTH_INIT`). The script supplies the current file's fx.
4. Prints the report against the current `config/stereo_calibration.json`:
   - per-eye RMS reprojection
   - fx, fy, cx, cy old → new
   - the radtan distortion
   - baseline and translation
   - cam1's rotation
   - the rectified pinhole
   - **how far the rectified left eye turns**
   - **what the current file reads at 0.6 / 1.25 / 2.5 m if Kalibr is right** (depth ratio and
     disparity offset; this is the prediction the A/B below checks)

   It ends ACCEPTED only when each eye's RMS reprojection is < 0.3 px and the baseline is within
   2 mm of the rig's nominal 63 mm (`camera.json` `rig.baseline_m_nominal`).

The mapping (`pepin.kalibr_stereo`, unit-tested in `tests/unit/test_kalibr_stereo.py`):

- **Distortion:** Kalibr's radtan `[k1 k2 r1 r2]` uses OpenCV's tangential convention, so the
  file gets OpenCV's plumb_bob `[k1 k2 p1 p2 0]`. The test projects through both formulas,
  including a negative control with p1 and p2 swapped. The current chessboard file uses OpenCV's
  14-term rational model; radtan's four terms are the price of Kalibr. If the corners fit badly,
  the report PDF shows it, and `pinhole-equi` would need `cv2.fisheye` in the rectifier.
- **Translation and rotation:** Kalibr's `T_cn_cnm1` of cam1 is cam1 ← cam0, which is OpenCV's
  `x_right = R x_left + T` with cam0 = left. It is copied, not inverted. The test triangulates a
  true head's points through the mapped file to their depth, and the inverse does not.

## 3. What `--apply` changes

- `config/stereo_calibration.json` is replaced. The file before is kept once as
  `config/stereo_calibration.pre-kalibr-<day>.json`, and the new `method`/`board` say what it
  was, from what, and what it replaced.
- **The rectified left eye's frame moves.** `stereoRectify` turns both eyes to a common
  orientation that depends on the bar and the lenses. `camera.json`'s `eye` block (camera_link ←
  that eye) and `head_imu.T_cam_imu` (imu → that eye) are expressed in it.
  `pepin.kalibr_stereo.rectified_frame_shift` measures the turn Q from the same raw left pixels
  under both models (Wahba over the rays within 35 deg of the axis). `--apply` carries both blocks
  through Q:
  - `R_link←optical` becomes `R_link←optical Q^T`
  - `T_cam_imu` becomes `[Q 0; 0 1] T_cam_imu`

  This keeps camera_link ← imu (the hand-eye the board runs) and every base_link geometry fitted
  today (the arm mount, the wall) where it was measured. Each block's note records the carry.
  `--no-carry` (in `stereo_kalibr.py apply`) skips it.

Consumers, in order. `--apply` prints these commands but does not run them:

| Consumer | What it reads | What to do |
| --- | --- | --- |
| camera_stream | the file, polled by mtime: rebuilds the rectifier by itself; `eye`/`head_imu` only at start | `ros/laptop.sh kick camera_stream` |
| depth_stream | camera_stream's P per frame; the stereo law saved in `ros/maps/depth_law_stereo.json` was fitted under the old rectification | move that file aside, then `ros/laptop.sh kick depth_stream` |
| VIO | `vio_config.py` writes the camchain from the rectified pinhole and `T_cam_imu` at every start | `ros/laptop.sh vio down && ros/laptop.sh vio`, at rest |
| RTAB-Map | P from `/camera/camera_info` | a map started before stays in the old rectification |
| the board | nothing (the hand-eye is unchanged by the carry) | nothing |

## 4. The A/B that proves the fix

Before (2026-10-04):

- **depth_stream's stereo law** (watching only; `1/z = a/D + b` against the lidar) reads
  `a 0.30 b +0.200 [a+b AT BOUND]`.
- **The floor**, from a capture of seven head poses with the cart parked (stereo depth moved into
  base_link through TF). The statistic is the median base_link z of the floor points (|z| < 0.35 m,
  outside the cart's footprint) in each horizontal ring:

  | Ring | Median floor z |
  | --- | --- |
  | 0.50-0.75 m | -0.152 m |
  | 0.75-1.00 m | -0.183 m |
  | 1.00-1.25 m | -0.208 m |
  | 1.25-1.50 m | -0.241 m |

  The floor reads low, more so with range. The depth is too long by about 12 % at 1.3 m and 20 %
  at 1.9 m, which is the signature of a constant disparity offset of about -2.6 px.
- **Kalibr's camera calibration on the rectified eyes of the 10-04 camera-IMU recording** (the
  wall grid at one distance, so its focal is weak; the focal fallback above let it initialise)
  finds:
  - the rectified eyes turned 0.36 deg about y
  - cx 2.5 px apart
  - the published rectification reading 3 / 4 / 7 % long at 0.6 / 1.25 / 2.5 m (a disparity
    offset of -1.3 to -0.8 px)

  That is the same sign, and about half the floor's offset.

After, on the same day and the same cart pose:

1. Before `--apply`, the report's own prediction: the current file read against Kalibr's model
   should show a ratio > 1 that grows with range, with a negative disparity offset. Within about
   1 px of the floor's -2.6 px, the stereo model explains the bias. Far less means the rest is
   elsewhere (the matcher, the eye block); stop and look before applying.
2. After `--apply` and the consumer restarts:
   - **The law:** depth_stream's report line after a few minutes of lidar pairs with a fresh law
     file. Expect `a` toward 1 and `b` toward 0, off its bound.
   - **The floor:** one capture of the same seven poses (the cart parked on the same spot) and
     the same ring statistic. Accept when the floor 1.0-1.5 m out is at z 0 ± 0.02 m in every
     pose and the rings no longer slope with range.
   - **Regression checks:** the arm's top at 0.6 m still agrees with the arm mount fit (1 mm
     before), and the head IMU gravity check stays at about 0.85 deg (the carry keeps the
     hand-eye).
3. If it fails: put `config/stereo_calibration.pre-kalibr-<day>.json` back, revert `camera.json`
   with git (the carried blocks), and kick camera_stream. The previous state is two files.

## 5. Tested and untested

- **Synthetic end to end:** Kalibr's own 25 mm grid
  ray-traced into a known radtan head, waved at 0.26-0.50 m and ±30 deg, packed upside down as the
  module sends it, run through the whole `run`.
  - K within 0.1 px, distortion within 1e-4
  - bar rotation 0.002 deg, translation 0.01 mm, baseline 62.699 against 62.704 mm
  - the rectified eye turns 0.002 deg from the truth's
  - depth ratio 1.0001 / 1.0005 at 0.6 / 2.5 m
  - 0.11 px reprojection, 60 s
- **The recorder** against a local fake ustreamer that replays that clip with ustreamer's headers:
  a byte-for-byte copy, grab stamps, tag counts.
- **Untested:** the live stream (frame size check, the WiFi with a second reader), a real
  hand-held recording through Kalibr (blur, auto exposure, radtan on this 94 deg lens at the
  corners), and the A/B itself.
