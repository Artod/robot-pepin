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


def _self_names(tree: ast.Module) -> dict[str, tuple[set[str], set[str]]]:
    """Per class in the file: the private ``self._x`` names it defines, and the ones it uses."""
    found: dict[str, tuple[set[str], set[str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        defined = {
            f.name for f in node.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        used: set[str] = set()
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Attribute):
                continue
            if not (isinstance(inner.value, ast.Name) and inner.value.id == "self"):
                continue
            if not inner.attr.startswith("_") or inner.attr.startswith("__"):
                continue  # only our own helpers; the rclpy base owns the public API
            (defined if isinstance(inner.ctx, ast.Store) else used).add(inner.attr)
        bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
        for base in bases & found.keys():
            defined |= found[base][0]
        found[node.name] = (defined, used)
    return found


def test_every_node_has_the_private_members_it_uses() -> None:
    """`self._open_run(...)` with no `def _open_run` reached the robot twice this week.

    rclpy cannot be imported here, so nothing else notices until the node dies on the board with
    an AttributeError mid-goal. This is that noticing, in a millisecond.
    """
    for path in ROS_PYTHON:
        for name, (defined, used) in _self_names(ast.parse(path.read_text())).items():
            missing = used - defined
            assert not missing, f"{path.name}:{name} uses {sorted(missing)} but never defines them"


def test_the_tracker_pairs_every_scan_with_the_pose_of_its_own_moment() -> None:
    """The tracker never waits for a transform inside a callback and never falls back to the
    newest pose: scans go through pepin.timeline's gate and are deskewed with its history. The
    fallback cost 1-2 degrees of false correction per scan in every pivot (runs 0080-0083)."""
    src = next(p for p in ROS_PYTHON if p.name == "relocalizer.py").read_text()
    assert "timeout=" not in src, "a transform wait inside a callback: the old fallback path"
    assert "ScanGate(" in src and "deskew(" in src and "OdomHistory(" in src
    assert "/odometry/filtered" in src
