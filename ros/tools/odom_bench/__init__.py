"""The odometry bench: every speed source's claimed covariance measured against one yardstick on a
frozen set of recorded drives, and the board's EKF replayed to judge a change (ros/README.md
"Odometry bench").

    uv run --with rosbags==0.11.5 python -m ros.tools.odom_bench run
"""

import sys
from pathlib import Path

_TOOLS = str(Path(__file__).resolve().parents[1])
if _TOOLS not in sys.path:
    sys.path.append(_TOOLS)  # vio_score, beside the package
