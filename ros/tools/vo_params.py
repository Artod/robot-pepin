#!/usr/bin/env python3
"""vslam.launch.py's VISUAL_ODOMETRY table as a parameters file for an offline stereo_odometry
(ros/vio_replay.sh's arm B): the live node's own numbers, read from the launch file, not copied.

    python3 ros/tools/vo_params.py > /tmp/vo.yaml
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

LAUNCH = Path(__file__).resolve().parents[1] / "pepin_bringup/launch/vslam.launch.py"
NODE = "stereo_odometry"


def table(path: Path = LAUNCH) -> dict[str, object]:
    """The VISUAL_ODOMETRY literal of the launch file, with the frame the launch adds."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "VISUAL_ODOMETRY" for t in node.targets
        ):
            values = ast.literal_eval(node.value)
            assert isinstance(values, dict)
            return {"frame_id": "base_link", **values}
    raise SystemExit(f"no VISUAL_ODOMETRY in {path}")


def as_yaml(values: dict[str, object], node: str = NODE) -> str:
    """A ROS 2 parameters file (YAML; JSON scalars are YAML scalars)."""
    lines = [f"{node}:", "  ros__parameters:"]
    lines += [f"    {json.dumps(k)}: {json.dumps(v)}" for k, v in values.items()]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Print the file."""
    sys.stdout.write(as_yaml(table()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
