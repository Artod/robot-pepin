"""The relocaliser's backoff: a robot that cannot find itself must not saturate the board."""

import ast
from pathlib import Path
from typing import Any

SOURCE = Path(__file__).resolve().parents[2] / "ros/pepin_bringup/pepin_bringup/relocalizer.py"


def _load() -> tuple[Any, float]:
    """Compile just ``backoff_wait`` and its cap out of the node's source, because importing
    the module would pull in rclpy, which lives only on the board."""
    wanted = [
        node
        for node in ast.parse(SOURCE.read_text()).body
        if (isinstance(node, ast.FunctionDef) and node.name == "backoff_wait")
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "MAX_BACKOFF_S" for t in node.targets)
        )
    ]
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["backoff_wait"], namespace["MAX_BACKOFF_S"]


backoff_wait, MAX_BACKOFF_S = _load()


def test_the_first_search_waits_one_cooldown() -> None:
    assert backoff_wait(8.0, 0) == 8.0


def test_every_failed_search_doubles_the_wait() -> None:
    assert [backoff_wait(2.0, failures) for failures in range(4)] == [2.0, 4.0, 8.0, 16.0]


def test_the_wait_stops_growing_at_the_cap() -> None:
    assert backoff_wait(4.0, 2) == 16.0  # still under the cap
    assert backoff_wait(8.0, 3) == MAX_BACKOFF_S  # 64 s clipped
    assert backoff_wait(8.0, 40) == MAX_BACKOFF_S  # and no overflow after an hour of failures
