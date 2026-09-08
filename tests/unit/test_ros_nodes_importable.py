"""The ROS nodes must at least parse, and their pure logic must live where a test can reach it.

rclpy is not installed on the laptop, so a node cannot be imported here — which is exactly why
its decisions belong in `src/pepin`. This guard catches the syntax errors that used to reach the
robot, and holds the line on how much undecidable logic a node may carry.
"""

import ast
from pathlib import Path

ROS_PYTHON = sorted(
    (Path(__file__).resolve().parents[2] / "ros").rglob("*.py"),
)


def test_every_ros_python_file_parses() -> None:
    assert ROS_PYTHON, "no ROS python found: the glob is wrong"
    for path in ROS_PYTHON:
        ast.parse(path.read_text(), filename=str(path))


def test_the_tracker_node_keeps_its_decisions_out_of_itself() -> None:
    """The node wires ROS to `pepin`; when it grows its own branching, the logic moved back in."""
    node = next(p for p in ROS_PYTHON if p.name == "relocalizer.py")
    tree = ast.parse(node.read_text())
    branches = sum(isinstance(n, (ast.If, ast.While)) for n in ast.walk(tree))
    assert branches <= 45, f"{branches} branches in the node: extract the decision into src/pepin"
    imports = {
        alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("pepin")
        for alias in n.names
    }
    assert {"LostWatch", "Localizer"} <= imports, "the node must use the tested pieces, not copies"
