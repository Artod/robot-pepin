#!/bin/bash
# The head camera's exposure on the board, live (pepin.camera_controls through v4l2-ctl):
#   ros/exposure.sh                   the same as show
#   ros/exposure.sh show              the exposure controls as the camera reports them: the auto
#                                     menu, the time's range (100 us units), the gain's, now/default
#   ros/exposure.sh apply             what /opt/pepin/config/camera.json's exposure block says
#   ros/exposure.sh auto              the camera's own defaults (as it ran before the block existed)
#   ros/exposure.sh manual MS [GAIN]  a fixed exposure of MS milliseconds, and the gain if given
#   ros/exposure.sh capped MS         auto, never longer than MS (or one frame period: it says)
# The next frame has it and nothing restarts; pepin-camera applies the config again at its next
# start, so a mode worth keeping goes into config/camera.json (then copied to /opt/pepin/config/).
# A value the camera does not offer is refused with its range. PEPIN_HOST picks the board.
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh
usage() { echo "usage: ros/exposure.sh [show | apply | auto | manual MS [GAIN] | capped MS]"; exit 2; }
# The device is the camera service's own (/etc/default/pepin-camera), never a guess.
on_board() {
    ssh "root@$BOARD" "set -a; . /etc/default/pepin-camera; set +a;" \
        "PYTHONPATH=/opt/pepin /opt/pepin/bin/python -m pepin.camera_controls $*"
}
case "${1:-show}" in
    show) [ $# -le 1 ] || usage; on_board show ;;
    apply) [ $# -eq 1 ] || usage; on_board apply ;;
    auto) [ $# -eq 1 ] || usage; on_board apply --mode auto ;;
    manual)
        [ $# -eq 2 ] || [ $# -eq 3 ] || usage
        on_board apply --mode manual --exposure-ms "$2" ${3:+--gain "$3"} ;;
    capped) [ $# -eq 2 ] || usage; on_board apply --mode capped --exposure-ms "$2" ;;
    *) usage ;;
esac
