#!/bin/bash
# Measure the neck camera's optics with a printed checkerboard, in one command. The nominal
# 78 deg in config/camera.json is one number standing in for four (fx, fy, cx, cy) and a lens
# that bends straight lines; this replaces all of it and sets calibrated: true.
#   ros/calibrate.sh --print          the board to print (data/checkerboard.pdf, A4, exact scale)
#   ros/calibrate.sh                  calibrate: a window with the corners and where to hold next
#   ros/calibrate.sh --no-window      the same with no window: a text coverage report
#   ros/calibrate.sh --board 9x6 --square 0.024    another board (inner corners, square in metres)
#   ros/calibrate.sh --images DIR     re-fit from the frames a previous run saved under data/
#
# With a STEREO head the same board measures both lenses and the bar between them, and writes
# config/stereo_calibration.json instead (the same printed board, the same window, the same
# no-keys collection — hold it where the hint says and move it):
#   ros/calibrate.sh stereo           calibrate the stereo head: both eyes of every frame
#   ros/calibrate.sh stereo --no-window          the same over ssh
#   ros/calibrate.sh stereo --images DIR         re-fit a saved session under data/stereo_calib/
#   ros/calibrate.sh stereo --square 0.0245      the square really on the paper, metres
# It refuses to write a result whose stereo RMS, baseline or rectified epipolar error is bad,
# and says what to reshoot; the accepted pairs are saved first, so a bad run is re-fitted and
# not reshot.
# No keys to press: hold the board where the hint says, keep it still, and the shot is taken on
# its own countdown. About 25 well-spread views, then the fit is printed and written to
# config/camera.json — but only when the RMS reprojection error is under 0.5 px and the views
# actually covered the frame; a poor run refuses to write and says why. Print at 100 %, not
# "fit to page", and measure one square afterwards: the square size is the only length in the
# whole calibration, and every metre it yields is wrong by however wrong it is.
# The camera node must be restarted to publish the new optics: ros/laptop.sh kick camera_stream.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.." || exit 1
BOARD="${PEPIN_HOST:-10.0.0.187}"
if [ "${1:-}" = "stereo" ]; then
    shift
    uv run -q python ros/tools/stereo_calibrate.py --host "$BOARD" "$@"
    exit $?
fi
uv run -q python scripts/calibrate_camera.py --host "$BOARD" "$@"
status=$?
# --print leaves a PDF to send to a printer; on this laptop, show it.
if [ $status -eq 0 ] && [[ " $* " == *" --print "* || " $* " == *" --print" ]] && [ -f data/checkerboard.pdf ]; then
    command -v open >/dev/null && open data/checkerboard.pdf
fi
exit $status
