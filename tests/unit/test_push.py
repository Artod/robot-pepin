"""ros/push.sh and its planner (pepin.push): a changed file reaches exactly the running nodes that
import it, through any chain of imports, on each half; what a kick cannot deliver is refused before
anything is touched; a dry run prints the plan and touches nothing."""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pepin.push import (
    FRESH_EACH_START,
    HELD,
    HOST_SERVICES,
    LAUNCH_HALVES,
    PARAMS_HALVES,
    PER_CALL,
    ImportGraph,
    Kick,
    imports_of,
    kickable,
    main,
    make_plan,
    repo_paths,
)

REPO = Path(__file__).resolve().parents[2]

# A checkout in miniature: two halves, one node per role, and a library whose modules reach the
# nodes through chains of every import form.
FAKE_KICK = """#!/bin/bash
KICKABLE="{nodes}"
printf '%s %s\\n' "$(basename "$0")" "$*" >> "$FAKE_LOG"
case "${{FAKE_KICK:-ok}}" in
    ok) echo "$2 kicked at 10:00:05.1, ready at 10:00:07.3 UTC, 2.2 s, pid 47 -> 836: $2 up" ;;
    missing) echo "no $2 process in pepin-ros"; exit 3 ;;
    stuck) echo "$2 not ready within 120 s: pid 836 (python3-7) has not printed it"; exit 4 ;;
esac
"""
TREE = {
    "ros/thin.sh": FAKE_KICK.format(nodes="rec goal"),
    "ros/laptop.sh": FAKE_KICK.format(nodes="fusion goal"),
    "ros/pepin_bringup/pepin_bringup/__init__.py": "",
    "ros/pepin_bringup/pepin_bringup/kit.py": "from pepin.a import (\n    x,\n    y,  # two\n)\n",
    "ros/pepin_bringup/pepin_bringup/rec.py": "from pepin_bringup.kit import spin\n",
    "ros/pepin_bringup/pepin_bringup/goal.py": "from pepin import b\n",
    "ros/pepin_bringup/pepin_bringup/fusion.py": "def main():\n    import pepin.c as c\n",
    "ros/pepin_bringup/pepin_bringup/base_bridge.py": "from pepin.a import x\n",
    "ros/pepin_bringup/pepin_bringup/teleop_keys.py": "from pepin.d import keys\n",
    "ros/pepin_bringup/launch/robot.launch.py": "from pepin.lc import z\n",
    "ros/pepin_bringup/launch/vslam.launch.py": "",
    "ros/params/ekf.yaml": "",
    "ros/params/nav2_params.yaml": "",
    "ros/tools/probe.py": "from pepin.a import x\n",
    "ros/foxglove/layout.json": "",
    "ros/run.sh": "",
    "src/pepin/__init__.py": "",
    "src/pepin/a.py": "from .b import y\n",
    "src/pepin/b.py": '"""The b module: import the map, then from there on."""\n',
    "src/pepin/c.py": "",
    "src/pepin/d.py": "",
    "src/pepin/lc.py": "",
    "src/pepin/base_server.py": "from pepin import b\n",
    "config/base.json": "{}",
    "board/pepin-ros.service": "",
    "README.md": "",
}


def _repo(root: Path) -> Path:
    for rel, text in TREE.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    for script in ("ros/thin.sh", "ros/laptop.sh"):
        (root / script).chmod(0o755)
    return root


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _repo(tmp_path / "repo")


# ---- the import graph --------------------------------------------------------------------------


def test_the_scanner_reads_every_import_form() -> None:
    known = {
        n: Path()
        for n in ("pkg", "pkg.a", "pkg.b", "pkg.sub", "pkg.sub.c", "pkg.sub.d", "pkg.e", "top")
    }
    source = "\n".join(
        [
            '"""Import the map from here on: prose, not an import."""',
            "import top, pkg.a as a  # a comment",
            "from pkg import (",
            "    b,  # a comment inside",
            "    e as ee,",
            ")",
            "from . import d",
            "from ..sub \\",
            "    import c",
            "def lazy():",
            "    from pkg.sub.c import thing",
        ]
    )
    got = imports_of(source, "pkg.sub.x", is_package=False, known=known)
    assert got == {"top", "pkg", "pkg.a", "pkg.b", "pkg.e", "pkg.sub", "pkg.sub.d", "pkg.sub.c"}
    assert imports_of("from . import c", "pkg.sub", is_package=True, known=known) == {
        "pkg",
        "pkg.sub.c",
    }


def test_a_module_reaches_every_node_that_imports_it_through_any_chain(repo: Path) -> None:
    plan = make_plan(repo, ["src/pepin/b.py"])
    assert not plan.refused
    assert plan.files == ["src/pepin/b.py"]
    assert plan.kicks == [
        Kick("board", "rec", ("pepin_bringup.rec", "pepin_bringup.kit", "pepin.a", "pepin.b")),
        Kick("board", "goal", ("pepin_bringup.goal", "pepin.b")),
        Kick("laptop", "goal", ("pepin_bringup.goal", "pepin.b")),
    ]
    assert [(h.held.name, h.chain) for h in plan.held] == [
        ("base_bridge", ("pepin_bringup.base_bridge", "pepin.a", "pepin.b"))
    ]
    assert plan.notes == [
        f"not reached: pepin.base_server imports it ({HOST_SERVICES['pepin.base_server']})"
    ]


def test_a_lazy_import_inside_a_function_counts(repo: Path) -> None:
    assert make_plan(repo, ["src/pepin/c.py"]).kicks == [
        Kick("laptop", "fusion", ("pepin_bringup.fusion", "pepin.c"))
    ]


def test_a_node_s_own_file_kicks_it_on_every_half_that_lists_it(repo: Path) -> None:
    plan = make_plan(repo, ["ros/pepin_bringup/pepin_bringup/goal.py"])
    assert [(k.half, k.node) for k in plan.kicks] == [("board", "goal"), ("laptop", "goal")]
    assert plan.board_path(plan.files[0]) == "/root/pepin-ros/pepin_bringup/pepin_bringup/goal.py"


def test_a_module_nothing_long_lived_imports_goes_to_the_board_with_a_note(repo: Path) -> None:
    plan = make_plan(
        repo, ["src/pepin/d.py", "ros/tools/probe.py", "README.md", "ros/foxglove/layout.json"]
    )
    assert not plan.refused and not plan.kicks
    assert plan.files == ["src/pepin/d.py", "ros/tools/probe.py"]
    assert plan.notes == [
        "ros/tools/probe.py: no long-lived process runs it; its next call uses it",
        "README.md: not on the robot, skipped",
        "ros/foxglove/layout.json: a Foxglove layout, the laptop's app reads it, skipped",
        "src/pepin/d.py: no running node imports it; its next start uses it",
    ]


def test_a_module_a_launch_file_imports_is_refused_with_that_launch_s_halves(repo: Path) -> None:
    plan = make_plan(repo, ["src/pepin/lc.py", "src/pepin/b.py"])
    assert plan.refused and not plan.kicks
    assert [(r.path, r.why, r.fix) for r in plan.refusals] == [
        (
            "src/pepin/lc.py",
            "imported by robot.launch.py: the launch process keeps it",
            "ros/restart.sh board --deploy",
        )
    ]


@pytest.mark.parametrize(
    ("path", "fix"),
    [
        ("ros/params/ekf.yaml", "ros/restart.sh board --deploy"),
        ("ros/params/nav2_params.yaml", "ros/restart.sh both --deploy"),
        ("ros/pepin_bringup/launch/vslam.launch.py", "ros/restart.sh laptop"),
        ("ros/pepin_bringup/launch/robot.launch.py", "ros/restart.sh board --deploy"),
        ("board/pepin-ros.service", "ros/restart.sh board after the install"),
        ("config/base.json", "ros/restart.sh both --deploy"),
        ("ros/run.sh", "ros/restart.sh board --deploy"),
        ("src/pepin/gone.py", "ros/sync.sh --restart (its --delete)"),
    ],
)
def test_what_is_read_once_at_start_is_refused_with_the_restart_that_delivers_it(
    repo: Path, path: str, fix: str
) -> None:
    plan = make_plan(repo, [path, "src/pepin/b.py"])
    assert [(r.path, r.fix) for r in plan.refusals] == [(path, fix)]
    assert plan.text().splitlines()[0] == "push REFUSED, nothing is touched:"
    assert main(["plan", "--repo", str(repo), str(repo / path)]) == 2


def test_a_file_outside_the_checkout_is_refused(repo: Path, tmp_path: Path) -> None:
    paths, outside = repo_paths(repo, [str(repo / "src/pepin/b.py"), str(tmp_path / "x.py")])
    assert paths == ["src/pepin/b.py"]
    assert [r.path for r in outside] == [str(tmp_path / "x.py")]


def test_the_dry_run_text_and_the_shell_lines_say_the_same_plan(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["plan", "--repo", str(repo), str(repo / "src/pepin/b.py")]) == 0
    assert capsys.readouterr().out == (
        "push plan: 1 file(s) to the board, 3 kick(s)\n"
        "  rsync src/pepin/b.py -> /root/pepin-ros/pepin_src/pepin/b.py\n"
        "kick on the board (a node not running there is skipped):\n"
        "  rec               pepin_bringup.rec > pepin_bringup.kit > pepin.a > pepin.b\n"
        "  goal              pepin_bringup.goal > pepin.b\n"
        "kick on the laptop (a node not running there is skipped):\n"
        "  goal              pepin_bringup.goal > pepin.b\n"
        "refused if running (the launch does not respawn it):\n"
        "  board pepin-ros base_bridge: pepin_bringup.base_bridge > pepin.a > pepin.b\n"
        "notes:\n"
        f"  not reached: pepin.base_server imports it ({HOST_SERVICES['pepin.base_server']})\n"
    )
    assert main(["plan", "--shell", "--repo", str(repo), str(repo / "src/pepin/b.py")]) == 0
    assert capsys.readouterr().out == (
        "file\tsrc/pepin/b.py\n"
        "kick\tboard\trec\n"
        "kick\tboard\tgoal\n"
        "kick\tlaptop\tgoal\n"
        "held\tboard\tpepin-ros\t-f\tpepin_bringup[./]base_bridge\tbase_bridge\t"
        "ros/restart.sh board --deploy\n"
    )


# ---- the real tree -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph() -> ImportGraph:
    return ImportGraph.scan(REPO)


def test_every_node_a_launch_starts_is_kickable_held_or_fresh_at_each_start() -> None:
    """A new node in a launch file that no kick script lists would be left running the old code
    by every push: it must be kickable on its half, held (refuses while it runs) or exec'd anew."""
    halves = kickable(REPO)
    for half, nodes in halves.items():
        for node in nodes:
            assert (REPO / f"ros/pepin_bringup/pepin_bringup/{node}.py").is_file(), (half, node)
    modules = {p.stem for p in (REPO / "ros/pepin_bringup/pepin_bringup").glob("*.py")}
    covered = (
        {n for nodes in halves.values() for n in nodes}
        | {h.name for h in HELD}
        | {m.removeprefix("pepin_bringup.") for m in (*FRESH_EACH_START, *PER_CALL)}
    )
    for launch in sorted((REPO / "ros/pepin_bringup/launch").glob("*.launch.py")):
        text = launch.read_text()
        started = set(re.findall(r'"-m",\s*"pepin_bringup\.(\w+)"', text))
        started |= set(re.findall(r"-m pepin_bringup\.(\w+)", text))
        started |= set(re.findall(r'executable="(\w+)"', text)) & modules
        assert started <= covered, (launch.name, started - covered)


def test_every_launch_and_params_file_says_which_halves_read_it() -> None:
    assert {p.name for p in (REPO / "ros/pepin_bringup/launch").glob("*.py")} == set(LAUNCH_HALVES)
    assert {p.name for p in (REPO / "ros/params").iterdir()} == set(PARAMS_HALVES)


def test_every_long_lived_host_process_of_ours_is_named() -> None:
    """``-m pepin.X`` in a unit or a script is either a service a push cannot reach (named in the
    plan's notes) or a per-call tool."""
    found: set[str] = set()
    for path in [*(REPO / "board").iterdir(), *(REPO / "ros").glob("*.sh")]:
        if path.is_file():
            found |= set(re.findall(r"-m (pepin\.\w+)", path.read_text(errors="replace")))
    assert found <= set(HOST_SERVICES) | set(PER_CALL), found - set(HOST_SERVICES) - set(PER_CALL)


def test_the_real_tree_kicks_what_imports_the_change(graph: ImportGraph) -> None:
    def kicks(path: str) -> list[tuple[str, str]]:
        return [(k.half, k.node) for k in make_plan(REPO, [path], graph).kicks]

    assert kicks("ros/pepin_bringup/pepin_bringup/depth_fusion.py") == [("laptop", "depth_fusion")]
    assert kicks("ros/pepin_bringup/pepin_bringup/tof_bridge.py") == [("board", "tof_bridge")]
    assert kicks("ros/pepin_bringup/pepin_bringup/goal_server.py") == [
        ("board", "goal_server"),
        ("laptop", "goal_server"),
    ]
    refused = make_plan(REPO, ["src/pepin/deployment.py"], graph).refusals
    assert [r.fix for r in refused] == ["ros/restart.sh both --deploy"]


@pytest.mark.slow  # ~0.3 s: ast over the whole tree
def test_the_line_scanner_finds_every_edge_ast_finds(graph: ImportGraph) -> None:
    """The scanner may add an edge (prose that reads like an import: one kick too many), never lose
    one (a stale node): every import ast sees in our sources is in the graph."""
    for name, path in graph.files.items():
        module = name.split(":", 1)[-1]
        seen: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                seen |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    package = module.split(".")
                    if path.name != "__init__.py":
                        package = package[:-1]
                    package = package[: len(package) - (node.level - 1)]
                    base = ".".join([*package, *([base] if base else [])])
                seen.add(base)
                seen |= {f"{base}.{a.name}" for a in node.names}
        ours = {s for s in seen if s in graph.files and s != module}
        assert ours <= graph.edges[name], (name, ours - graph.edges[name])


# ---- ros/push.sh against fakes -------------------------------------------------------------------

FAKE_TOOL = """#!/bin/bash
printf '%s %s\\n' "$(basename "$0")" "$*" >> "$FAKE_LOG"
case "$(basename "$0") $1" in
    "ssh "*) exit "${FAKE_SSH_RC:-1}" ;;
    "docker inspect") [ -n "${FAKE_MOUNT:-}" ] || exit 1; echo "$FAKE_MOUNT" ;;
    "docker exec") exit 1 ;;
esac
"""
# The planner the script calls (`uv run -q python -m pepin.push ...`), on this interpreter.
FAKE_UV = f"""#!/bin/bash
shift 3
exec {sys.executable} "$@"
"""


def _push(tmp_path: Path, *args: str, **env: str) -> tuple[int, str, str]:
    """ros/push.sh ARGS in the miniature checkout, ssh/rsync/docker faked; (exit, output, calls)."""
    root = _repo(tmp_path / "repo")
    for script in ("push.sh", "lib.sh"):
        shutil.copy(REPO / "ros" / script, root / "ros" / script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (
        ("ssh", FAKE_TOOL),
        ("rsync", FAKE_TOOL),
        ("docker", FAKE_TOOL),
        ("uv", FAKE_UV),
    ):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "log"
    log.write_text("")
    run = subprocess.run(
        ["bash", str(root / "ros/push.sh"), *args],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=root,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PYTHONPATH": str(REPO / "src"),
            "FAKE_LOG": str(log),
            **env,
        },
    )
    return run.returncode, run.stdout + run.stderr, log.read_text()


@pytest.mark.slow  # ~0.3 s each below: bash, the planner twice, the fakes
def test_a_dry_run_prints_the_plan_and_what_would_run_and_touches_nothing(tmp_path: Path) -> None:
    code, out, calls = _push(tmp_path, "--dry-run", "src/pepin/b.py")
    assert code == 0, out
    assert calls == "", "a dry run asks nobody anything"
    assert out.startswith("push plan: 1 file(s) to the board, 3 kick(s)\n")
    tail = out[out.index("would check: base_bridge") :]
    assert tail == (
        "would check: base_bridge in pepin-ros on the board (runs -> refused,"
        " ros/restart.sh board --deploy)\n"
        "would run: (cd src && rsync -a --relative pepin/b.py"
        " root@10.0.0.187:/root/pepin-ros/pepin_src/)\n"
        "would run: ros/thin.sh kick rec\n"
        "would run: ros/thin.sh kick goal\n"
        "would run: ros/laptop.sh kick goal\n"
        "dry run: nothing was touched\n"
    )
    assert "would check: pepin-vslam and pepin-laptop mount" in out
    same = _push(tmp_path / "env", "src/pepin/b.py", PEPIN_PUSH_DRY="1")
    assert same[0] == 0 and same[2] == "" and same[1].endswith("dry run: nothing was touched\n")


@pytest.mark.slow
def test_a_refused_push_touches_nothing(tmp_path: Path) -> None:
    code, out, calls = _push(tmp_path, "ros/params/ekf.yaml", "src/pepin/b.py")
    assert code == 2 and calls == ""
    assert "ros/params/ekf.yaml: parameters are read once at start" in out


@pytest.mark.slow
def test_a_push_rsyncs_exactly_its_files_and_kicks_exactly_its_nodes(tmp_path: Path) -> None:
    code, out, calls = _push(tmp_path, "src/pepin/b.py", "ros/pepin_bringup/pepin_bringup/goal.py")
    assert code == 0, out
    rsyncs = [c for c in calls.splitlines() if c.startswith("rsync")]
    assert rsyncs == [
        "rsync -a --relative pepin_bringup/pepin_bringup/goal.py root@10.0.0.187:/root/pepin-ros/",
        "rsync -a --relative pepin/b.py root@10.0.0.187:/root/pepin-ros/pepin_src/",
    ]
    assert "--delete" not in calls, "the board writes files of its own beside ours"
    kicks = sorted(c for c in calls.splitlines() if " kick " in c)
    assert kicks == ["laptop.sh kick goal", "thin.sh kick goal", "thin.sh kick rec"]
    assert (
        "board  rec kicked at 10:00:05.1, ready at 10:00:07.3 UTC, 2.2 s, pid 47 -> 836: rec up"
        in out
    )
    assert "push: 3 node(s) back, 0 not running, 0 failed" in out


@pytest.mark.slow
def test_a_running_held_process_refuses_before_the_rsync(tmp_path: Path) -> None:
    code, out, calls = _push(tmp_path, "src/pepin/b.py", FAKE_SSH_RC="0")
    assert code == 2, out
    assert "base_bridge runs in pepin-ros on the board and no kick restarts it" in out
    assert "rsync" not in calls and " kick " not in calls
    code, out, calls = _push(tmp_path / "unreachable", "src/pepin/b.py", FAKE_SSH_RC="255")
    assert code == 2 and "could not ask pepin-ros on the board" in out and "rsync" not in calls


@pytest.mark.slow
def test_a_node_not_running_is_skipped_and_a_stuck_one_fails_the_push(tmp_path: Path) -> None:
    code, out, _ = _push(tmp_path, "src/pepin/c.py", FAKE_KICK="missing")
    assert code == 0, out
    assert "laptop fusion skipped: no fusion process in pepin-ros" in out
    code, out, _ = _push(tmp_path / "stuck", "src/pepin/c.py", FAKE_KICK="stuck")
    assert code == 1 and "laptop fusion FAILED (exit 4)" in out


@pytest.mark.slow
def test_laptop_containers_on_another_checkout_refuse_the_push(tmp_path: Path) -> None:
    code, out, calls = _push(tmp_path, "src/pepin/c.py", FAKE_MOUNT=str(tmp_path))
    assert code == 2, out
    assert "mounts" in out and "not this checkout's src/pepin" in out
    assert "rsync" not in calls
