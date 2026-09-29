"""ros/tools/planner_check.py, the planner's proof in ros/restart.sh: a latched costmap proves
nothing, a path in any of four directions proves the planner, "no path" everywhere is BOXED (no
restart helps), and every other failure is BROKEN (a restart may). Against the real Nav2 it ran in
the sim (scratch/start_race/sim_start_race.sh); here its decisions, with the ROS parts faked."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import ros_stubs

REPO = Path(__file__).resolve().parents[2]
PI = math.pi


class Done:
    """A finished future holding ``value``."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def done(self) -> bool:
        return True

    def result(self) -> Any:
        return self.value


class Planner:
    """The planner server: answers each goal from ``answers`` in turn, keeps the goals."""

    def __init__(self, answers: list[tuple[int, int]], server: bool = True) -> None:
        self.answers = answers  # (poses in the path, error code) per goal
        self.server = server
        self.goals: list[Any] = []

    def __call__(self, node: Any, action: Any, name: str) -> Planner:
        return self

    def wait_for_server(self, timeout_sec: float = 0.0) -> bool:
        return self.server

    def send_goal_async(self, goal: Any) -> Done:
        self.goals.append(goal)
        poses, code = self.answers[len(self.goals) - 1]
        answer = SimpleNamespace(path=SimpleNamespace(poses=[object()] * poses), error_code=code)
        handle = SimpleNamespace(
            accepted=True, get_result_async=lambda: Done(SimpleNamespace(result=answer))
        )
        return Done(handle)

    def destroy(self) -> None:
        pass


class Tf:
    """tf2_ros.Buffer with the cart at (1, 2) facing +y, or with no map frame at all."""

    def __init__(self, known: bool = True) -> None:
        self.known = known

    def can_transform(self, *args: Any) -> bool:
        return self.known

    def lookup_transform(self, *args: Any) -> Any:
        rotation = SimpleNamespace(x=0.0, y=0.0, z=math.sin(PI / 4), w=math.cos(PI / 4))
        translation = SimpleNamespace(x=1.0, y=2.0, z=0.0)
        return SimpleNamespace(
            transform=SimpleNamespace(translation=translation, rotation=rotation)
        )


def _goal_msg() -> Any:
    pose = SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0), orientation=None, header=None)
    return SimpleNamespace(pose=pose, header=SimpleNamespace(frame_id=""))


def run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    planner: Planner,
    grids: int = 2,
    updates: int = 0,
    tf: Tf | None = None,
) -> tuple[int, str, Planner]:
    """planner_check.main() with the costmap sending ``grids`` + ``updates`` on the first spin."""
    rclpy = ros_stubs.install()
    node = ros_stubs.Node("pepin_planner_check")
    sent = {"done": False}

    def spin_once(_node: Any, timeout_sec: float = 0.0) -> None:
        if sent["done"]:
            return
        sent["done"] = True
        for _ in range(grids):
            node.subs["/global_costmap/costmap"][1](
                SimpleNamespace(info=SimpleNamespace(width=141, height=216))
            )
        for _ in range(updates):
            node.subs["/global_costmap/costmap_updates"][1](object())

    monkeypatch.setattr(rclpy, "create_node", lambda name: node, raising=False)
    monkeypatch.setattr(rclpy, "spin_once", spin_once, raising=False)
    monkeypatch.setattr(rclpy, "shutdown", lambda: None, raising=False)
    action = SimpleNamespace(Goal=lambda: SimpleNamespace(goal=None, planner_id="", use_start=True))
    monkeypatch.setattr(sys.modules["nav2_msgs.action"], "ComputePathToPose", action, raising=False)
    spec = importlib.util.spec_from_file_location(
        "planner_check", REPO / "ros/tools/planner_check.py"
    )
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "planner_check", module)  # its dataclass looks itself up
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PoseStamped", _goal_msg)
    monkeypatch.setattr(module, "ActionClient", planner)
    monkeypatch.setattr(module, "Buffer", lambda: tf or Tf())
    monkeypatch.setattr(module, "TransformListener", lambda buffer, node: None)
    monkeypatch.setattr(
        sys, "argv", ["planner_check.py", "--costmap-s", "0.05", "--plan-s", "0.05"]
    )
    code = module.main()
    return code, capsys.readouterr().out.strip(), planner


def test_a_path_ahead_is_the_proof_and_the_goal_is_half_a_metre_in_front(
    monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    code, out, planner = run(monkeypatch, capsys, Planner([(12, 0)]))
    assert code == 0, out
    assert out.startswith("planner: OK — path of 12 poses to 0.50 m ahead (GridBased)"), out
    goal = planner.goals[0]
    assert goal.use_start is False and goal.planner_id == "GridBased"
    assert (goal.goal.pose.position.x, goal.goal.pose.position.y) == pytest.approx((1.0, 2.5))
    assert goal.goal.header.frame_id == "map"


def test_an_occupied_goal_ahead_is_tried_behind_and_to_the_sides(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, planner = run(monkeypatch, capsys, Planner([(0, 206), (0, 208), (7, 0)]))
    assert code == 0 and "0.50 m left" in out, out
    xs = [(g.goal.pose.position.x, g.goal.pose.position.y) for g in planner.goals]
    assert xs == [pytest.approx(p) for p in ((1.0, 2.5), (1.0, 1.5), (0.5, 2.0))]


def test_no_path_in_any_direction_is_boxed_not_broken(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, _ = run(monkeypatch, capsys, Planner([(0, 206), (0, 206), (0, 208), (0, 204)]))
    assert code == 3, out
    assert out == (
        "planner: BOXED — it answers, but no path 0.50 m around: ahead goal occupied,"
        " behind goal occupied, left no valid path, right goal outside the map"
    )


def test_a_planner_timeout_is_broken_at_once(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, planner = run(monkeypatch, capsys, Planner([(0, 207)]))
    assert code == 1 and "BROKEN — ahead: error code 207" in out, out
    assert len(planner.goals) == 1, "a timed-out planner is not asked three more times"


def test_a_latched_costmap_proves_nothing_but_one_update_does(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, planner = run(monkeypatch, capsys, Planner([(5, 0)]), grids=1)
    assert code == 1 and "sent only its latched copy" in out, out
    assert planner.goals == []
    code, out, _ = run(monkeypatch, capsys, Planner([(5, 0)]), grids=1, updates=1)
    assert code == 0, out


def test_no_pose_or_no_server_is_broken(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, _ = run(monkeypatch, capsys, Planner([(5, 0)]), tf=Tf(known=False))
    assert code == 1 and "no map -> base_link in TF" in out, out
    code, out, _ = run(monkeypatch, capsys, Planner([(5, 0)], server=False))
    assert code == 1 and "no /compute_path_to_pose action server" in out, out
