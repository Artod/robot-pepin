#!/usr/bin/env python3
"""Kalibr's own AprilGrid detector on one rectified picture: what kalibr_calibrate_imu_camera will
see of the grid. Runs INSIDE pepin-kalibr (Python 3.8, ROS Noetic, Kalibr's aslam_cv bindings);
ros/tools/neck_dance.py --check is its caller. Prints one JSON line:

    {"success": .., "corners": n, "of": 144, "tags": n, "tag_px": .., "corners_px": [[u, v], ..],
     "centre_cam": [x, y, z]}

``success`` is Kalibr's own verdict (at least 7 tags); ``centre_cam`` the grid's centre in the
camera's optical frame (metres, from Kalibr's target pose), present on success.

    docker run --rm --network none -v DIR:/k -v ros/tools:/tools:ro --entrypoint bash \\
        pepin-kalibr -c 'source /catkin_ws/devel/setup.bash && python3 /tools/kalibr_detect.py \\
        /k/left.png /k/camchain.yaml /k/april.yaml'
"""

from __future__ import annotations

import json
import sys
from typing import Any


def detect(image_path: str, camchain_path: str, april_path: str) -> dict[str, Any]:
    """Kalibr's detector, set up exactly as kalibr_calibrate_imu_camera sets it up."""
    import aslam_cameras_april as acv_april
    import aslam_cv as acv
    import cv2
    import kalibr_common as kc
    import numpy as np

    camera = kc.AslamCamera.fromParameters(
        kc.CameraChainParameters(camchain_path).getCameraParameters(0)
    )
    params = kc.CalibrationTargetParameters(april_path).getTargetParams()
    options = acv_april.AprilgridOptions()
    options.minTagsForValidObs = int(max(params["tagRows"], params["tagCols"]) + 1)
    grid = acv_april.GridCalibrationTargetAprilgrid(
        params["tagRows"], params["tagCols"], params["tagSize"], params["tagSpacing"], options
    )
    detector_options = acv.GridDetectorOptions()
    detector_options.filterCornerOutliers = True
    detector = acv.GridDetector(camera.geometry, grid, detector_options)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"no picture at {image_path}")
    success, obs = detector.findTarget(acv.Time(0.0), np.array(image))
    total = int(grid.size())
    out: dict[str, Any] = {
        "success": bool(success),
        "corners": 0,
        "of": total,
        "tags": 0,
        "tag_px": None,
    }
    if obs is None:
        return out
    idx = [int(i) for i in np.asarray(obs.getCornersIdx()).ravel()]
    pixels = np.asarray(obs.getCornersImageFrame(), dtype=float).reshape(-1, 2)
    out["corners"] = len(idx)
    out["corners_px"] = [[round(float(u), 1), round(float(v), 1)] for u, v in pixels]
    # Points run row by row, two per tag in each direction: a tag's side is the step from an
    # even column to the next one in the same row.
    cols = int(grid.cols())
    where = dict(zip(idx, pixels))  # noqa: B905 (Python 3.8 in the image: no strict=)
    sides = [
        float(np.linalg.norm(where[i + 1] - where[i]))
        for i in idx
        if i % 2 == 0 and (i % cols) + 1 < cols and i + 1 in where
    ]
    out["tag_px"] = round(float(np.median(sides)), 1) if sides else None
    # A tag is whole when all four of its corners were found: (row, col) // 2 names it.
    per_tag: dict[tuple[int, int], int] = {}
    for i in idx:
        key = ((i // cols) // 2, (i % cols) // 2)
        per_tag[key] = per_tag.get(key, 0) + 1
    out["tags"] = sum(1 for n in per_tag.values() if n == 4)
    if success:
        points = np.asarray([np.asarray(grid.point(i)).ravel() for i in range(total)], dtype=float)
        centre_t = np.append(points.mean(axis=0), 1.0)
        t_c_t = np.linalg.inv(np.asarray(obs.T_t_c().T(), dtype=float))
        out["centre_cam"] = [round(float(v), 4) for v in (t_c_t @ centre_t)[:3]]
    return out


def main(argv: list[str]) -> int:
    """Detect, print the JSON line."""
    if len(argv) != 4:
        print(__doc__)
        return 2
    print(json.dumps(detect(argv[1], argv[2], argv[3])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
