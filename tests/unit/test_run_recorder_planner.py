"""What the PLANNER saw, on the tape: the global costmap's encoding, its throttle, Nav2's own
action outcomes, and the pose graph's words beside the camera's.

The tapes of 2026-09-17 could not answer why two legs piled up 78 and 90 recoveries with no path,
because they carry /plan and the LOCAL costmap and the planner reads the GLOBAL one
(ros/maps/rec/20260917_192935_goto.log, ..._201425_goto.log; the cart's own footprint was clear in
all 1740 taped local grids, scratch/footprint_in_costmap.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import ros_stubs

ros_stubs.install()

from action_msgs.msg import GoalStatus, GoalStatusArray  # noqa: E402
from nav_msgs.msg import OccupancyGrid  # noqa: E402
from nav_msgs.msg import Path as PathMsg  # noqa: E402
from pepin_bringup.run_recorder import (  # noqa: E402
    FREE,
    INFLATED,
    LETHAL_BAND,
    RLE,
    UNKNOWN,
    RunRecorder,
    feasibility_classes,
)
from std_msgs.msg import Header, String  # noqa: E402

from pepin.mapcache import run_length_decode, run_length_encode  # noqa: E402


def grid(cells: list[int], width: int = 4) -> OccupancyGrid:
    """A costmap message carrying ``cells`` row-major."""
    msg = OccupancyGrid(header=Header())
    msg.info.resolution = 0.05
    msg.info.width, msg.info.height = width, max(1, len(cells) // width)
    msg.data = cells
    return msg


def opened(tmp_path: Path, **kwargs: Any) -> tuple[RunRecorder, Path]:
    """A recorder with a tape open, so every record is kept, and the file it writes."""
    from rclpy.node import Node

    rec = RunRecorder(Node("run_recorder"), tmp_path, **kwargs)
    return rec, rec.start("test")


@pytest.fixture
def recorder(tmp_path: Path) -> tuple[RunRecorder, Path]:
    """The ordinary case: every record kept, the planner's own picture on."""
    return opened(tmp_path)


def records(path: Path, topic: str) -> list[dict[str, Any]]:
    """Everything of one topic the tape holds so far."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if r.get("topic") == topic]


def test_the_encoding_is_lossless_and_flat() -> None:
    """A reader needs no library: value, count, value, count over the row-major cells."""
    assert run_length_encode([]) == []
    assert run_length_encode([5]) == [5, 1]
    assert run_length_encode([0, 0, 0, 7, 7, -1]) == [0, 3, 7, 2, -1, 1]
    for cells in ([], [1], [0, 0, 0], [-1, 0, 99, 99, 1, 1, 1, -1]):
        assert run_length_decode(run_length_encode(cells)) == cells


def test_only_the_four_values_that_decide_whether_the_cart_fits_are_kept() -> None:
    """Nav2's gradient between free and lethal is cost, not feasibility — and it is what makes a
    costmap incompressible (the local grids of 2026-09-17: 1.5-fold raw, 8.6-fold reduced)."""
    assert feasibility_classes([-1, 0, 1, 50, 98, 99, 100]) == [
        UNKNOWN,
        FREE,
        INFLATED,
        INFLATED,
        INFLATED,
        LETHAL_BAND,
        LETHAL_BAND,
    ]
    walls = [0] * 40 + [99, 100, 100, 99] + [-1] * 20
    assert len(run_length_encode(feasibility_classes(walls))) < len(walls) / 4


def test_the_planner_s_grid_is_taped_once_per_plan(recorder: tuple[RunRecorder, Path]) -> None:
    """The throttle is the plan counter and not a clock: a grid nobody planned on answers nothing,
    and a drive re-planning ten times a second must not write ten grids a second."""
    rec, tape = recorder
    cells = [-1, -1, 0, 0, 0, 0, 99, 100]
    rec._on_global_costmap(grid(cells))
    assert len(records(tape, "gcostmap")) == 1, "the first grid of a run is always worth having"
    rec._on_global_costmap(grid(cells))
    rec._on_global_costmap(grid(cells))
    assert len(records(tape, "gcostmap")) == 1, "no new plan, no new grid"
    rec._on_plan(PathMsg(header=Header(), poses=[]))
    rec._on_global_costmap(grid(cells))
    taped = records(tape, "gcostmap")
    assert len(taped) == 2 and taped[-1]["plan"] == 1
    assert taped[-1]["encoding"] == RLE and taped[-1]["classes"] == [
        UNKNOWN,
        FREE,
        INFLATED,
        LETHAL_BAND,
    ]
    assert run_length_decode(taped[-1]["data"]) == feasibility_classes(cells)
    assert taped[-1]["width"] == 4 and taped[-1]["resolution"] == 0.05


def test_nav2_s_own_outcome_is_on_the_tape_per_action(recorder: tuple[RunRecorder, Path]) -> None:
    """The one thing a tape could not say: whether a plan was ABORTED or never asked for."""
    rec, tape = recorder
    rec._on_action_status(
        "compute_path_to_pose", GoalStatusArray(status_list=[GoalStatus(status=6)])
    )
    rec._on_action_status(
        "follow_path", GoalStatusArray(status_list=[GoalStatus(status=2), GoalStatus(status=4)])
    )
    taped = records(tape, "nav")
    assert [(r["action"], r["status"]) for r in taped] == [
        ("compute_path_to_pose", [6]),
        ("follow_path", [2, 4]),
    ]


def test_the_graph_s_own_words_are_taped_like_the_camera_s(
    recorder: tuple[RunRecorder, Path],
) -> None:
    """Until 2026-09-18 only /localization/measurement was taped, so the age the board's gate
    charged the GRAPH — the one source a camera-only drive runs on — could not be read off a
    tape at all (scratch/word_age.py)."""
    rec, tape = recorder
    rec._on_measurement(String(data='{"source": "graph", "stamp": 100.0}'))
    taped = records(tape, "meas")
    assert len(taped) == 1 and json.loads(taped[0]["json"])["source"] == "graph"
    assert taped[0]["t"] > 0.0, "the arrival time, against which the stamp inside is the age"


def test_the_records_stop_with_the_flag(tmp_path: Path) -> None:
    """CLAUDE.md rule 19: the tape of before 2026-09-18 stays reachable without a restart."""
    rec, tape = opened(tmp_path, planner_records=lambda: False)
    rec._on_global_costmap(grid([0, 0, 99, 100]))
    rec._on_action_status("follow_path", GoalStatusArray(status_list=[GoalStatus(status=4)]))
    assert records(tape, "gcostmap") == [] and records(tape, "nav") == []
