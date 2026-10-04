"""The SO-101 follower arm's URDF, vendored: the kinematic chain the arm's self-filter poses.

UPSTREAM TheRobotStudio/SO-ARM100 at 5f6d2b8 (2026-09-23), ``Simulation/SO101/so101_new_calib.urdf``
(last changed upstream in 385e8d7, 2025-07-02), Apache License 2.0 — see LICENSE beside this file.
Copied VERBATIM, no edits. "new calib": every joint's zero is the middle of its range, the
convention lerobot's so101_follower reads its encoders in.

The meshes are NOT vendored (13 STL files, 16 MB): :mod:`pepin.arm` needs only the joints, and
the links' collision boxes in config/arm.json were fitted to the meshes of the same commit by
``scripts/arm_boxes.py``, which fetches them.
"""

from pathlib import Path

URDF = Path(__file__).resolve().parent / "so101_new_calib.urdf"
UPSTREAM = "https://github.com/TheRobotStudio/SO-ARM100"
COMMIT = "5f6d2b876a53a4872e405b991dd925556c9e38a4"
