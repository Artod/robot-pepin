"""ros/tools/flags_doc.py: the nodes' flag tables read from the sources, the README section
they render (kept current here, the way the zenoh bridge configs are), and the answers
ros/flags.sh asks it for."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pepin.flags import Flag, FlagSet

REPO = Path(__file__).resolve().parents[2]


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("flags_doc", REPO / "ros/tools/flags_doc.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("flags_doc", module)
    spec.loader.exec_module(module)
    return module


DOC = _tool()


def test_every_node_with_a_table_is_found_and_only_those() -> None:
    assert DOC.has_table(REPO / "ros/pepin_bringup/pepin_bringup/depth_fusion.py")
    assert not DOC.has_table(REPO / "ros/pepin_bringup/pepin_bringup/ghost_wait.py")
    tables = DOC.tables()
    assert set(tables) >= {"depth_stream", "depth_fusion", "relocalizer", "neck_state"}
    assert all(isinstance(t, FlagSet) and len(t) for t in tables.values())
    assert list(tables) == sorted(tables), "file order: the README reads the same every time"


def test_the_readme_s_feature_flags_section_is_current() -> None:
    """Regenerate with ros/tools/flags_doc.py when a node's table changes."""
    text = DOC.README.read_text()
    assert DOC.readme_with(text, DOC.section()) == text, (
        "ros/README.md: the Feature flags section is stale; run ros/tools/flags_doc.py"
    )
    assert text.count(DOC.HEADING) == 1
    assert text.index(DOC.HEADING) < text.index("## Build and run")
    assert DOC.HOW_TO_READ.splitlines()[0] in text, "how to read a flag, above the table"
    for node, flags in DOC.tables().items():
        assert f"#### `{node}`" in text, node
        for flag in flags:
            assert f"| `{node}` | `{flag.name}` |" in text, (node, flag.name)
            assert flag.markdown() in text, f"{node}/{flag.name}: the paragraph under the table"


def test_the_table_is_the_one_liner_and_the_paragraphs_come_under_it_by_node() -> None:
    """The short answer for scanning, the long one for deciding: one row per flag in a table
    over every node, then every flag whole, grouped by the node that owns it."""
    body = DOC.section()
    table, details = body.index("| node | flag |"), body.index(DOC.DETAILS)
    assert body.index(DOC.HOW_TO_READ.splitlines()[0]) < table < details
    for node, flags in DOC.tables().items():
        assert body.index(f"#### `{node}`") > table, node
        for flag in flags:
            assert f"  - *Default:* {flag.render(flag.default)} — " in body, (node, flag.name)
    assert body.endswith("\n") and "\n\n\n" not in body


def test_the_section_replaces_its_predecessor_or_is_inserted_before_the_build_notes() -> None:
    body = "## Feature flags\n\nnew\n"
    stale = "# Title\n\n## Feature flags\n\nold table\n\n## Build and run\n\nsteps\n"
    assert (
        DOC.readme_with(stale, body)
        == "# Title\n\n## Feature flags\n\nnew\n\n## Build and run\n\nsteps\n"
    )
    fresh = "# Title\n\n## Layout\n\nx\n\n## Build and run\n\nsteps\n"
    assert DOC.readme_with(fresh, body) == (
        "# Title\n\n## Layout\n\nx\n\n## Feature flags\n\nnew\n\n## Build and run\n\nsteps\n"
    )
    assert DOC.readme_with("# Title\n\n## Feature flags\n\nold\n", body) == "# Title\n\n" + body
    assert DOC.readme_with("# Title\n", body) == "# Title\n\n" + body
    assert DOC.readme_with(DOC.readme_with(fresh, body), body) == DOC.readme_with(fresh, body)


def test_a_parameter_dump_gives_the_current_values_and_nothing_else_does() -> None:
    dump = "/depth_fusion:\n  ros__parameters:\n    align: false\n    min_weight: 2.0\n    x: a\n"
    assert DOC.current_values(dump) == {"align": False, "min_weight": 2.0, "x": "a"}
    assert DOC.current_values("") == {} and DOC.current_values("- 1\n- 2\n") == {}
    assert DOC.current_values("/a:\n  b: 1\n") == {} and DOC.current_values(": [\n") == {}


def test_the_listing_shows_kind_current_value_and_description_per_flag() -> None:
    flags = FlagSet(
        Flag("align", True, description="frame-to-model"),
        Flag("backend", "local", choices=("local", "remote"), description="where"),
        Flag("gain", 0.5, range=(0.0, 1.0)),
    )
    dump = "/n:\n  ros__parameters:\n    align: false\n    backend: gpu\n"
    lines = DOC.listing("n", flags, dump).splitlines()
    assert lines[0] == "n/align    bool                   off    frame-to-model"
    assert lines[1] == "n/backend  choice: local, remote  'gpu'  where", "an alien value, shown raw"
    assert lines[2] == "n/gain     number 0..1            unset"
    down = DOC.listing("n", flags, "").splitlines()
    assert down[0].split() == ["n/align", "bool", "?", "frame-to-model"]
    assert down[-1] == "(n did not answer a parameter dump: is it up?)"


def test_a_value_is_written_the_way_ros2_param_set_reads_it_back() -> None:
    assert DOC.yaml_literal(Flag("a", True), False) == "false"
    assert DOC.yaml_literal(Flag("n", 8), 4) == "4"
    assert DOC.yaml_literal(Flag("w", 2.0), 3.0) == "3.0", "a double keeps its point"
    assert DOC.yaml_literal(Flag("b", "local", choices=("local", "remote")), "remote") == "remote"
    sources = Flag("s", ("a",), choices=("a", "b", "true", "1"))
    assert DOC.yaml_literal(sources, ("a", "b")) == "a,b"
    assert DOC.yaml_literal(sources, ()) == "''"
    assert DOC.yaml_literal(sources, ("true",)) == "!!str true"
    assert DOC.yaml_literal(sources, ("1",)) == "!!str 1"


def _main(argv: list[str], capsys: Any, stdin: str | None = None) -> tuple[int, str, str]:
    if stdin is not None:
        sys.stdin = type("In", (), {"read": lambda self: stdin, "isatty": lambda self: False})()
    try:
        code = DOC.main(argv)
    except SystemExit as exc:
        code = int(exc.code or 0)
    finally:
        sys.stdin = sys.__stdin__
    out, err = capsys.readouterr()
    return code, out, err


def test_the_verbs_flags_sh_asks_for(capsys: Any) -> None:
    code, out, _ = _main(["nodes"], capsys)
    assert code == 0 and "depth_stream" in out.split() and "ghost_wait" not in out.split()
    assert _main(["where", "depth_fusion"], capsys)[1].strip() == "laptop pepin-vslam"
    assert _main(["where", "/relocalizer"], capsys)[1].strip() == "board pepin-ros"
    code, out, err = _main(["where", "nope"], capsys)
    assert code == 2 and out == "" and err.startswith("nope: no node with a flags table")
    code, out, _ = _main(["flag", "neck_state", "neck_tf"], capsys)
    assert code == 0 and out.startswith("neck_state/neck_tf: bool, default on\n")
    assert [line.split(":")[0] for line in out.splitlines() if line[:1].isupper()] == [
        "What",
        "Default",
        "On when",
        "Off when",
    ], "the whole entry, not just the kind"
    assert max(len(line) for line in out.splitlines()) <= 100, "it is read in a terminal"
    code, _, err = _main(["flag", "neck_state", "gpu"], capsys)
    assert code == 2 and err.startswith("neck_state: no flag gpu; the flags are neck_tf")
    assert _main(["value", "depth_fusion", "min_weight", "3"], capsys)[1] == "3.0\n"
    assert _main(["value", "depth_stream", "depth_backend", "auto"], capsys)[1] == "auto\n"
    code, _, err = _main(["value", "depth_stream", "depth_backend", "gpu"], capsys)
    assert code == 2 and err.strip() == "depth_backend: 'gpu' is not one of remote, local, auto"
    code, _, err = _main(["value", "depth_stream", "threads", "4"], capsys)
    assert code == 2 and "no flag threads" in err, "a startup parameter is not a flag"
    dump = "/depth_fusion:\n  ros__parameters:\n    align: false\n"
    code, out, _ = _main(["list", "depth_fusion"], capsys, stdin=dump)
    assert code == 0 and out.startswith("depth_fusion/enabled") and "  unset  " in out
    assert _main(["frob"], capsys)[0] == 2
    code, out, _ = _main(["--check"], capsys)
    assert code == 0 and out == ""


def test_the_nodes_of_one_side_and_the_flags_that_are_not_their_default(capsys: Any) -> None:
    """What ros/restart.sh asks after a restart: the nodes of the half it restarted, and, per
    node, only the flags that are NOT what the table declares — a restart puts every flag back
    to its default, so anything listed is a switch someone set on purpose."""
    board, laptop = (
        _main(["nodes", "board"], capsys)[1].split(),
        _main(["nodes", "laptop"], capsys)[1].split(),
    )
    assert "relocalizer" in board and "depth_fusion" not in board
    assert "depth_fusion" in laptop and "relocalizer" not in laptop
    assert sorted(board + laptop) == sorted(_main(["nodes"], capsys)[1].split())
    code, _, err = _main(["nodes", "orbit"], capsys)
    assert code == 2 and err.startswith("orbit: no such side")

    dump = "/depth_fusion:\n  ros__parameters:\n    align: false\n    min_weight: 2.0\n"
    code, out, _ = _main(["drift", "depth_fusion"], capsys, stdin=dump)
    assert code == 0 and out == "depth_fusion/align off (default on)\n"
    code, out, _ = _main(["drift", "depth_fusion"], capsys, stdin="")
    assert code == 1 and out.strip() == "depth_fusion: no answer to a parameter dump (is it up?)"


@pytest.mark.parametrize("verb", ["where", "flag", "value"])
def test_a_verb_with_the_wrong_number_of_words_prints_the_usage(verb: str, capsys: Any) -> None:
    code, out, _ = _main([verb], capsys)
    assert code == 2 and "ros/tools/flags_doc.py" in out
