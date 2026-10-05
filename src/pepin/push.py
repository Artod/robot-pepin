"""What a changed file needs to reach the running robot: the nodes to kick, or why a push refuses.

``ros/push.sh FILE...`` asks this module for a plan and then carries it out: the files go to the
board by rsync (the laptop's containers mount the checkout), and every running node whose Python
imports a changed module — directly or through any chain of imports — is kicked by
``ros/board.sh kick`` (board) or ``ros/laptop.sh kick`` (laptop): SIGINT, the launch respawns it
from the new sources, and the kick waits for the NEW pid's own ready line.

What a kick cannot reach is refused before anything is touched: a launch file, a params file or a
systemd unit is read once at start (``ros/restart.sh``), a module a launch file imports lives in
the launch process itself, and an image layer needs a build. A long-lived process of ours that the
launch does not respawn (:data:`HELD`) refuses the push only where it runs; ``push.sh`` asks the
container before the rsync.

The import graph is read from the sources with a line scanner, not ``ast``: ~20 ms for the whole
tree instead of ~250 ms, and a docstring line that happens to read like an import can only add a
kick, never lose one.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

HALVES = ("board", "laptop")

# Where the board keeps the tree ros/sync.sh mirrors (ros/run.sh mounts it into pepin-ros).
BOARD_ROS_DIR = "/root/pepin-ros"
BOARD_LIB_DIR = f"{BOARD_ROS_DIR}/pepin_src"

# The scripts that own the kick of each half: their KICKABLE line is the list of nodes a kick
# reaches there, and the ready line each node prints lives beside it in the same script.
KICK_SCRIPTS = {"board": "ros/board.sh", "laptop": "ros/laptop.sh"}

# Each launch file and the halves that run it (the board runs sensors only: navigation is the
# laptop's).
LAUNCH_HALVES: dict[str, tuple[str, ...]] = {
    "bringup.launch.py": ("board",),
    "robot.launch.py": ("board",),
    "nav.launch.py": ("laptop",),
    "vslam.launch.py": ("laptop",),
    "vio.launch.py": ("laptop",),  # pepin-vio (ros/laptop.sh vio)
}

# Each params file and the halves whose processes read it at start.
PARAMS_HALVES: dict[str, tuple[str, ...]] = {
    "nav2_params.yaml": ("laptop",),
    "ekf.yaml": ("board",),
    "rosbag_qos.yaml": ("laptop",),
    "calib_qos.yaml": ("laptop",),  # ros/calib_record.sh's recorder in pepin-vslam
    "pepin_nav_to_pose.xml": ("laptop",),
}


@dataclass(frozen=True)
class Held:
    """A long-lived process of ours that no kick restarts: a push refuses while it runs."""

    name: str
    half: str
    container: str
    pgrep: tuple[str, str]  # the pgrep flag and pattern that find it inside ``container``
    roots: tuple[str, ...]  # the modules it runs (graph names)
    what: str


HELD: tuple[Held, ...] = (
    Held(
        "rtabmap",
        "laptop",
        "pepin-vslam",
        ("-x", "rtabmap"),
        ("xfeat:rtabmap_xfeat", "xfeat:rtabmap_lighterglue"),
        "RTAB-Map, which loads the XFeat adapters once at start",
    ),
)

# Our long-lived processes outside the ROS containers, which no push reaches.
HOST_SERVICES: dict[str, str] = {
    "pepin.base_server": "board: pepin-base.service runs it from /opt/pepin (board/README.md)",
    "pepin.tof_server": "board: pepin-tof.service runs it from /opt/pepin (board/README.md)",
    "pepin.audio_server": "board: pepin-audio.service runs it from /opt/pepin (board/README.md)",
    "pepin.head_server": "board: pepin-head.service runs it from /opt/pepin (board/README.md)",
    "pepin.depth_service": "laptop: ros/models.sh restart depth",
    "pepin.localization_service": "laptop: ros/models.sh restart localization",
}

# Modules started per call by a script or by hand (nothing long-lived holds them).
PER_CALL = (
    "pepin.camera_controls",  # pepin-camera's ExecStartPre and ros/exposure.sh, once each
    "pepin.census",
    "pepin.goal_link",
    "pepin.head_link",  # the head server's door by hand (bring-up, a bench)
    "pepin.i2c_recover",  # a locked i2c-2 freed by hand on the board (board/README.md)
    "pepin.push",
    "pepin.red_button",
    "pepin.speed",
    "pepin.teleop",
    "pepin_bringup.teleop_keys",
)


def restart_hint(halves: Iterable[str]) -> str:
    """The restart that deploys what a kick cannot, for these halves."""
    got = set(halves)
    if got >= {"board", "laptop"}:
        return "ros/restart.sh both --deploy"
    if got == {"laptop"}:
        return "ros/restart.sh laptop"
    return "ros/restart.sh board --deploy"


# ---- the import graph ------------------------------------------------------------------------

_FROM = re.compile(r"from\s+(\.*)([\w.]*)\s+import\s+(.*)", re.S)
_IMPORT = re.compile(r"import\s+(.*)", re.S)


def _statements(source: str) -> Iterable[str]:
    """Each import statement of a Python source, continuation lines joined, comments dropped."""
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        stmt = lines[i].split("#", 1)[0].strip()
        i += 1
        if not stmt.startswith(("from ", "import ")):
            continue
        while (stmt.endswith("\\") or stmt.count("(") > stmt.count(")")) and i < len(lines):
            stmt = stmt.rstrip("\\") + " " + lines[i].split("#", 1)[0].strip()
            i += 1
        yield stmt


def _names(clause: str) -> list[str]:
    """The imported names of ``a, b as c`` or ``(a, b)``, aliases dropped."""
    names = []
    for part in clause.replace("(", " ").replace(")", " ").split(","):
        words = part.split()
        if words:
            names.append(words[0])
    return names


def _with_parents(name: str) -> list[str]:
    """``a.b.c`` and every package above it: importing a module runs its packages first."""
    parts = name.split(".")
    return [".".join(parts[:n]) for n in range(1, len(parts) + 1)]


def imports_of(source: str, module: str, is_package: bool, known: Mapping[str, Path]) -> set[str]:
    """The known modules one source imports (with their parent packages), relative imports
    resolved against ``module``; names that are not modules of ours are dropped."""
    found: set[str] = set()
    for stmt in _statements(source):
        m = _FROM.match(stmt)
        if m:
            dots, base, clause = m.groups()
            if dots:
                package = module.split(".") if is_package else module.split(".")[:-1]
                package = package[: len(package) - (len(dots) - 1)]
                base = ".".join([*package, *([base] if base else [])])
            found.update(_with_parents(base))
            for name in _names(clause):
                found.update(_with_parents(f"{base}.{name}"))
            continue
        m = _IMPORT.match(stmt)
        if m:
            for name in _names(m.group(1)):
                found.update(_with_parents(name))
    return {name for name in found if name in known and name != module}


@dataclass
class ImportGraph:
    """Which of our modules each of our modules and launch files imports.

    Names: ``pepin.x`` (src/pepin), ``pepin_bringup.x`` (the ROS package), ``launch:<file>``
    (ros/pepin_bringup/launch) and ``xfeat:<stem>`` (the RTAB-Map adapters, ros/xfeat)."""

    files: dict[str, Path]
    edges: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def scan(cls, repo: Path) -> ImportGraph:
        """Read every module and launch file under ``repo`` once."""
        files: dict[str, Path] = {}
        packages: set[str] = set()
        for path in sorted((repo / "src/pepin").rglob("*.py")):
            parts = path.relative_to(repo / "src").with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
                packages.add(".".join(parts))
            files[".".join(parts)] = path
        ros = repo / "ros/pepin_bringup/pepin_bringup"
        for path in sorted(ros.glob("*.py")):
            name = "pepin_bringup" if path.stem == "__init__" else f"pepin_bringup.{path.stem}"
            files[name] = path
        packages.add("pepin_bringup")
        for path in sorted((repo / "ros/pepin_bringup/launch").glob("*.py")):
            files[f"launch:{path.name}"] = path
        for path in sorted((repo / "ros/xfeat").glob("*.py")):
            files[f"xfeat:{path.stem}"] = path
        graph = cls(files)
        for name, path in files.items():
            module = name.split(":", 1)[1] if ":" in name else name
            source = path.read_text(errors="replace")
            graph.edges[name] = imports_of(source, module, name in packages, files)
        return graph

    def chain(self, root: str, targets: set[str]) -> tuple[str, ...] | None:
        """The shortest import chain from ``root`` to any of ``targets`` (both ends included),
        or None when ``root`` never imports them."""
        if root not in self.files:
            return None
        parent: dict[str, str | None] = {root: None}
        todo = deque([root])
        while todo:
            name = todo.popleft()
            if name in targets:
                path = [name]
                while (up := parent[path[-1]]) is not None:
                    path.append(up)
                return tuple(reversed(path))
            for nxt in sorted(self.edges.get(name, ())):
                if nxt not in parent:
                    parent[nxt] = name
                    todo.append(nxt)
        return None


def kickable(repo: Path) -> dict[str, tuple[str, ...]]:
    """The nodes a kick reaches on each half: the KICKABLE line of that half's kick script."""
    out: dict[str, tuple[str, ...]] = {}
    for half, script in KICK_SCRIPTS.items():
        m = re.search(r'^KICKABLE="([^"]*)"', (repo / script).read_text(), re.M)
        if m is None:
            raise ValueError(f"{script} has no KICKABLE line")
        out[half] = tuple(m.group(1).split())
    return out


# ---- the plan --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Refusal:
    """A file a push cannot deliver, why, and what delivers it instead."""

    path: str
    why: str
    fix: str


@dataclass(frozen=True)
class Kick:
    """One node to kick on one half, and the import chain that makes it stale."""

    half: str
    node: str
    chain: tuple[str, ...]


@dataclass(frozen=True)
class HeldHit:
    """A held process that imports a changed module: the push refuses if it runs."""

    held: Held
    chain: tuple[str, ...]


@dataclass
class Plan:
    """Everything a push of these files does, or the reasons it refuses."""

    files: list[str] = field(default_factory=list)  # repo paths that go to the board
    kicks: list[Kick] = field(default_factory=list)
    held: list[HeldHit] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        """True when any file cannot be pushed: nothing at all is done then."""
        return bool(self.refusals)

    def board_path(self, path: str) -> str:
        """Where the board keeps a repo file (ros/sync.sh's layout)."""
        if path.startswith("src/pepin/"):
            return f"{BOARD_LIB_DIR}/{path.removeprefix('src/')}"
        return f"{BOARD_ROS_DIR}/{path.removeprefix('ros/')}"

    def text(self) -> str:
        """The plan as a person reads it (ros/push.sh prints it, --dry-run stops after it)."""
        if self.refusals:
            out = ["push REFUSED, nothing is touched:"]
            out += [f"  {r.path}: {r.why}; this needs {r.fix}" for r in self.refusals]
            out += [f"  {note}" for note in self.notes]
            return "\n".join(out)
        out = [f"push plan: {len(self.files)} file(s) to the board, {len(self.kicks)} kick(s)"]
        for path in self.files:
            out.append(f"  rsync {path} -> {self.board_path(path)}")
        for half in HALVES:
            kicks = [k for k in self.kicks if k.half == half]
            if kicks:
                out.append(f"kick on the {half} (a node not running there is skipped):")
                out += [f"  {k.node:<17} {' > '.join(k.chain)}" for k in kicks]
        if self.held:
            out.append("refused if running (the launch does not respawn it):")
            for h in self.held:
                out.append(
                    f"  {h.held.half} {h.held.container} {h.held.name}: {' > '.join(h.chain)}"
                )
        if self.notes:
            out.append("notes:")
            out += [f"  {note}" for note in self.notes]
        return "\n".join(out)

    def shell(self) -> str:
        """The plan as tab-separated lines for ros/push.sh: ``file``, ``kick``, ``held``,
        ``refuse``."""
        out = [f"file\t{path}" for path in self.files]
        out += [f"kick\t{k.half}\t{k.node}" for k in self.kicks]
        for h in self.held:
            fix = restart_hint([h.held.half])
            flag, pattern = h.held.pgrep
            out.append(
                f"held\t{h.held.half}\t{h.held.container}\t{flag}\t{pattern}\t{h.held.name}\t{fix}"
            )
        out += [f"refuse\t{r.path}\t{r.why}; this needs {r.fix}" for r in self.refusals]
        return "\n".join(out)


def _module_of(path: str) -> str | None:
    """The graph name of a Python file a kick can deliver, or None."""
    if path.startswith("src/pepin/") and path.endswith(".py"):
        parts = Path(path).relative_to("src").with_suffix("").parts
        return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    if path.startswith("ros/pepin_bringup/pepin_bringup/") and path.endswith(".py"):
        stem = Path(path).stem
        return "pepin_bringup" if stem == "__init__" else f"pepin_bringup.{stem}"
    return None


_IMAGE = re.compile(
    r"^ros/(Dockerfile[^/]*|pepin_base_cpp/.*|patches/.*|xfeat/.*\.sh|"
    r"pepin_bringup/(setup\.py|setup\.cfg|package\.xml|resource/.*))$"
)


def _classify(path: str) -> Refusal | str | None:
    """A file that is not a Python module: its refusal, "board" when it goes to the board and
    nothing runs it continuously, a skip note when it is not on the robot at all, or None for a
    module the graph decides."""
    name = Path(path).name
    if _module_of(path) is not None:
        return None
    if path.startswith("ros/pepin_bringup/launch/"):
        halves = LAUNCH_HALVES.get(name, HALVES)
        return Refusal(path, "a launch file is read once at start", restart_hint(halves))
    if path.startswith("ros/params/"):
        halves = PARAMS_HALVES.get(name, HALVES)
        return Refusal(path, "parameters are read once at start", restart_hint(halves))
    if path.startswith("board/"):
        return Refusal(
            path,
            "a systemd unit or board script, installed by hand (board/README.md)",
            "ros/restart.sh board after the install",
        )
    if path.startswith("config/"):
        return Refusal(
            path, "config is read at start by the launches and services", restart_hint(HALVES)
        )
    if path == "ros/run.sh":
        return Refusal(path, "the board container's own docker run", restart_hint(["board"]))
    if path == "ros/entrypoint.sh":
        return Refusal(path, "every container's entrypoint", restart_hint(HALVES))
    if _IMAGE.match(path):
        return Refusal(
            path, "baked into an image", "an image build (ros/build-image.sh, ros/laptop-build.sh)"
        )
    if path.startswith("ros/xfeat/"):
        return Refusal(path, "RTAB-Map loads its adapters once at start", "ros/laptop.sh vslam")
    if path.startswith("ros/zenoh"):  # ros/zenoh/router.json5, ros/zenoh-bridge-*.json
        return Refusal(
            path,
            "transport configuration",
            "ros/sync.sh and the routers' or bridges' restart (ros/README.md)",
        )
    if path.startswith("ros/maps/"):
        return "skip: maps stay on the laptop, whose containers mount them"
    if path.startswith("ros/foxglove/"):
        return "skip: a Foxglove layout, the laptop's app reads it"
    if path.startswith(("ros/", "src/pepin/")):
        return "board"
    return "skip: not on the robot"


def make_plan(repo: Path, paths: Sequence[str], graph: ImportGraph | None = None) -> Plan:
    """The plan for pushing ``paths`` (repo-relative) from the checkout at ``repo``."""
    plan = Plan()
    changed: dict[str, str] = {}  # graph name -> path
    for path in paths:
        if not (repo / path).is_file():
            plan.refusals.append(
                Refusal(path, "no such file (a deletion)", "ros/sync.sh --restart (its --delete)")
            )
            continue
        verdict = _classify(path)
        if isinstance(verdict, Refusal):
            plan.refusals.append(verdict)
        elif verdict == "board":
            plan.files.append(path)
            plan.notes.append(f"{path}: no long-lived process runs it; its next call uses it")
        elif verdict is not None:
            plan.notes.append(f"{path}: {verdict.removeprefix('skip: ')}, skipped")
        else:
            plan.files.append(path)
            changed[_module_of(path) or ""] = path
    if not changed:
        return plan
    graph = graph or ImportGraph.scan(repo)
    targets = set(changed)
    by_launch: dict[str, list[str]] = {}  # changed path -> the launch files that import it
    for name in sorted(n for n in graph.files if n.startswith("launch:")):
        for module in sorted(targets):
            if graph.chain(name, {module}) is not None:
                by_launch.setdefault(changed[module], []).append(name.removeprefix("launch:"))
    if by_launch:
        for path, launches in by_launch.items():
            halves = {h for launch in launches for h in LAUNCH_HALVES.get(launch, HALVES)}
            plan.refusals.append(
                Refusal(
                    path,
                    f"imported by {', '.join(launches)}: the launch process keeps it",
                    restart_hint(halves),
                )
            )
        return plan
    for half, nodes in kickable(repo).items():
        for node in nodes:
            chain = graph.chain(f"pepin_bringup.{node}", targets)
            if chain is not None:
                plan.kicks.append(Kick(half, node, chain))
    for held in HELD:
        for root in held.roots:
            chain = graph.chain(root, targets)
            if chain is not None:
                plan.held.append(HeldHit(held, chain))
                break
    for module, where in HOST_SERVICES.items():
        chain = graph.chain(module, targets)
        if chain is not None:
            plan.notes.append(f"not reached: {module} imports it ({where})")
    reached = {k.chain[-1] for k in plan.kicks} | {h.chain[-1] for h in plan.held}
    for module, path in changed.items():
        if module not in reached:
            plan.notes.append(f"{path}: no running node imports it; its next start uses it")
    return plan


def repo_paths(repo: Path, args: Sequence[str]) -> tuple[list[str], list[Refusal]]:
    """The arguments as repo-relative paths; one outside this checkout is refused."""
    root = Path(os.path.realpath(repo))
    paths: list[str] = []
    outside: list[Refusal] = []
    for arg in args:
        full = Path(os.path.realpath(arg))
        try:
            paths.append(full.relative_to(root).as_posix())
        except ValueError:
            outside.append(Refusal(arg, f"outside this checkout ({root})", "a push from there"))
    return paths, outside


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m pepin.push plan [--shell] [--repo DIR] FILE...``: 0 pushable, 2 refused."""
    parser = argparse.ArgumentParser(prog="python -m pepin.push")
    parser.add_argument("command", choices=["plan"])
    parser.add_argument("--shell", action="store_true", help="tab-separated lines for push.sh")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("files", nargs="+")
    args = parser.parse_args(argv)
    paths, outside = repo_paths(args.repo, args.files)
    plan = make_plan(args.repo, paths)
    plan.refusals[:0] = outside
    print(plan.shell() if args.shell else plan.text())
    return 2 if plan.refused else 0


if __name__ == "__main__":
    sys.exit(main())
