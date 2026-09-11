"""The feature-flag table: what a flag accepts, what it refuses and why, how a table reads back
for the report line and the README, and how a node's table is read without importing the node."""

from __future__ import annotations

from pathlib import Path

import pytest

from pepin.flags import COLUMNS, Flag, FlagSet, load_table, markdown_table


def _table() -> FlagSet:
    return FlagSet(
        Flag("align", True, description="frame-to-model before fusing"),
        Flag("backend", "remote", choices=("remote", "local", "auto"), env="PEPIN_BACKEND"),
        Flag("min_weight", 2.0, range=(0.0, 100.0), description="observations a voxel needs"),
        Flag("threads", 8, range=(1, 32)),
        Flag("sources", ("lidar",), choices=("lidar", "depth", "tof"), description="what marks"),
        Flag("url", "a", choices=("a", "b"), live=False),
    )


# ---- one flag ------------------------------------------------------------------------------
def test_a_flag_s_kind_is_told_by_its_default_and_its_keywords() -> None:
    assert Flag("align", True).kind == "bool"
    assert Flag("backend", "remote", choices=("remote", "local")).kind == "choice"
    assert Flag("min_weight", 2.0).kind == "number" and not Flag("min_weight", 2.0).integer
    assert Flag("threads", 8).kind == "number" and Flag("threads", 8).integer
    sources = Flag("sources", ["lidar"], choices=("lidar", "depth"))
    assert sources.kind == "list" and sources.default == ("lidar",)
    assert [f.wire_type for f in _table()] == [
        "bool",
        "string",
        "double",
        "integer",
        "string",
        "string",
    ]


@pytest.mark.parametrize(
    ("flag", "reason"),
    [
        (lambda: Flag("x", None), "a bool, a choice"),
        (lambda: Flag("x", "remote"), "needs its choices"),
        (lambda: Flag("x", ("a",)), "needs its choices"),
        (lambda: Flag("x", "gpu", choices=("remote", "local")), "not one of remote, local"),
        (lambda: Flag("x", ("a", "z"), choices=("a", "b")), "'z' is not one of a, b"),
        (lambda: Flag("x", True, choices=("a",)), "choices belong to"),
        (lambda: Flag("x", True, range=(0, 1)), "range belongs to a number"),
        (lambda: Flag("x", 5, range=(0, 4)), "5 is outside 0..4"),
        (lambda: Flag("x", 5, range=(4, 0)), "range 4..0 is empty"),
        (lambda: Flag("x", "a", choices=("a", "a")), "choices repeat"),
        (lambda: Flag("bad name", True), "identifier"),
    ],
)
def test_a_table_that_cannot_be_right_is_refused_at_import(flag: object, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        flag()  # type: ignore[operator]


def test_a_bool_takes_a_bool_or_its_words_and_nothing_else() -> None:
    align = Flag("align", True)
    assert align.parse(False) is False and align.parse("off") is False
    assert align.parse("True") is True and align.parse(" ON ") is True and align.parse("1")
    for bad in (1, 0, "maybe", None, 2.0):
        with pytest.raises(ValueError, match=r"align: .* is not on or off"):
            align.parse(bad)
    assert align.render(True) == "on" and align.render(False) == "off"


def test_a_number_keeps_its_kind_and_its_range() -> None:
    weight = Flag("min_weight", 2.0, range=(0.0, 100.0))
    assert weight.parse(3) == 3.0 and isinstance(weight.parse(3), float)
    assert weight.parse("4.5") == 4.5
    with pytest.raises(ValueError, match=r"min_weight: 101\.0 is outside 0\.\.100"):
        weight.parse(101)
    threads = Flag("threads", 8, range=(1, 32))
    assert threads.parse(4.0) == 4 and isinstance(threads.parse(4.0), int)
    assert threads.parse("16") == 16
    with pytest.raises(ValueError, match=r"threads: 2\.5 is not a whole number"):
        threads.parse(2.5)
    with pytest.raises(ValueError, match="threads: 'many' is not a number"):
        threads.parse("many")
    with pytest.raises(ValueError, match="threads: True is not a number"):
        threads.parse(True)  # a bool is an int to Python, never to a flag
    with pytest.raises(ValueError, match=r"outside 1\.\.32"):
        threads.parse(0)
    assert Flag("gain", 0.5).parse(1e9) == 1e9, "no range, no bound"
    assert weight.render(2.0) == "2.0" and threads.render(8) == "8"
    assert weight.kind_text() == "number 0..100" and threads.kind_text() == "integer 1..32"
    assert Flag("gain", 0.5).kind_text() == "number"


def test_a_choice_is_one_of_its_words() -> None:
    backend = Flag("backend", "remote", choices=("remote", "local", "auto"))
    assert backend.parse("auto") == "auto"
    with pytest.raises(ValueError, match="backend: 'gpu' is not one of remote, local, auto"):
        backend.parse("gpu")
    with pytest.raises(ValueError, match="backend: 1 is not one of"):
        backend.parse(1)
    assert backend.kind_text() == "choice: remote, local, auto"
    assert backend.help() == "(one of: remote, local, auto)"


def test_a_list_is_some_of_its_words_in_one_string_or_a_sequence() -> None:
    sources = Flag("sources", ("lidar",), choices=("lidar", "depth", "tof"))
    assert sources.parse("lidar,depth") == ("lidar", "depth")
    assert sources.parse(" depth , lidar,depth ") == ("depth", "lidar"), "trimmed, once each"
    assert sources.parse(["tof"]) == ("tof",) and sources.parse("") == ()
    with pytest.raises(ValueError, match="sources: 'sonar' is not one of lidar, depth, tof"):
        sources.parse("lidar,sonar")
    with pytest.raises(ValueError, match="sources: 3 is not a list of"):
        sources.parse(3)
    assert sources.render(("lidar", "depth")) == "lidar,depth" and sources.render(()) == "(none)"
    assert sources.wire(("lidar", "depth")) == "lidar,depth" and sources.wire_type == "string"
    assert sources.kind_text() == "list of: lidar, depth, tof"


def test_the_help_says_what_a_setter_must_know() -> None:
    assert Flag("align", True, description="frame-to-model").help() == "frame-to-model"
    assert (
        Flag("w", 2.0, range=(0.0, 9.0), description="weight", env="PEPIN_W", live=False).help()
        == "weight (0..9) (PEPIN_W overrides the default at start) (not live: set at the next"
        " start)"
    )
    assert Flag("s", ("a",), choices=("a", "b")).help() == "(any of: a, b, comma-separated)"


# ---- the table -----------------------------------------------------------------------------
def test_a_table_reads_its_values_and_changes_them_with_the_flag_s_own_check() -> None:
    flags = _table()
    assert flags.names == ("align", "backend", "min_weight", "threads", "sources", "url")
    assert len(flags) == 6 and "align" in flags and "gpu" not in flags
    assert flags["align"] is True and flags.on("align") and flags["sources"] == ("lidar",)
    assert flags.set("align", "off") is True and flags["align"] is False
    assert flags.set("sources", "depth,tof") == ("lidar",) and flags.on("sources")
    assert flags.set("min_weight", 3) == 2.0 and flags["min_weight"] == 3.0
    with pytest.raises(ValueError, match="backend: 'gpu' is not one of remote, local, auto"):
        flags.set("backend", "gpu")
    assert flags["backend"] == "remote", "a refused set changes nothing"
    with pytest.raises(ValueError, match="gpu: not a flag here; the flags are align, backend"):
        flags.set("gpu", True)
    with pytest.raises(ValueError, match="not a flag here"):
        flags["gpu"]
    assert flags.as_dict() == {
        "align": False,
        "backend": "remote",
        "min_weight": 3.0,
        "threads": 8,
        "sources": ("depth", "tof"),
        "url": "a",
    }
    assert flags.state() == "align=off backend=remote min_weight=3.0 threads=8 sources=depth,tof"
    assert flags.state(live_only=False).endswith(" sources=depth,tof url=a")
    with pytest.raises(ValueError, match="align: declared twice"):
        FlagSet(Flag("align", True), Flag("align", False))


def test_a_copy_starts_at_the_defaults_and_the_original_keeps_them() -> None:
    table = _table()
    mine = FlagSet(*table)
    mine.set("align", False)
    assert table["align"] is True and mine["align"] is False
    assert FlagSet(*table).flag("backend") is table.flag("backend")


def test_the_environment_overrides_a_default_at_start_and_a_bad_value_names_the_variable() -> None:
    flags = _table()
    assert flags.defaults({})["backend"] == "remote"
    assert flags.defaults(None)["backend"] == "remote"
    assert flags.defaults({"PEPIN_BACKEND": "auto", "OTHER": "x"})["backend"] == "auto"
    assert flags.defaults({"PEPIN_BACKEND": "auto"})["align"] is True
    with pytest.raises(ValueError, match="PEPIN_BACKEND: backend: 'gpu' is not one of"):
        flags.defaults({"PEPIN_BACKEND": "gpu"})
    assert flags["backend"] == "remote", "defaults() reads; it does not set"


def test_the_table_renders_as_markdown_for_the_readme() -> None:
    flags = FlagSet(
        Flag("align", True, description="frame-to-model | before fusing"),
        Flag("backend", "remote", choices=("remote", "auto"), env="PEPIN_BACKEND", live=False),
    )
    assert flags.describe() == (
        "| flag | kind | default | live | description |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `align` | bool | on | yes | frame-to-model \\| before fusing |\n"
        "| `backend` | choice: remote, auto | remote (env PEPIN_BACKEND) | at start |  |"
    )
    assert markdown_table(("a", "b"), [["1", "2"]]) == "| a | b |\n| --- | --- |\n| 1 | 2 |"
    assert flags.rows()[0] == ["`align`", "bool", "on", "yes", "frame-to-model | before fusing"]
    assert COLUMNS == ("flag", "kind", "default", "live", "description")


# ---- a node's table without the node ---------------------------------------------------------
def _module(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "node.py"
    path.write_text(body)
    return path


def test_a_node_s_table_is_read_from_its_source_with_its_constants_and_pepin_imports(
    tmp_path: Path,
) -> None:
    path = _module(
        tmp_path,
        "import rclpy  # not importable here\n"
        "from pepin.depth_service import MODES\n"
        "from pepin.flags import Flag, FlagSet\n"
        "TOPIC = '/scan'  # a literal the table may use\n"
        "RANGE = (0.0, 9.0)\n"
        "COMPUTED = len(TOPIC)  # not a literal: not usable\n"
        "FLAGS = FlagSet(\n"
        "    Flag('align', True, description=TOPIC),\n"
        "    Flag('backend', 'local', choices=MODES),\n"
        "    Flag('w', 2.0, range=RANGE),\n"
        ")\n"
        "OTHER = FlagSet(Flag('later', False))\n",
    )
    flags = load_table(path)
    assert flags.names == ("align", "backend", "w")
    assert flags.flag("backend").choices == ("remote", "local", "auto")
    assert flags.flag("align").description == "/scan" and flags.flag("w").range == (0.0, 9.0)
    assert load_table(path, "OTHER").names == ("later",)


def test_a_table_that_needs_the_node_running_is_refused(tmp_path: Path) -> None:
    path = _module(
        tmp_path,
        "import os\nfrom pepin.flags import Flag, FlagSet\n"
        "FLAGS = FlagSet(Flag('x', os.environ.get('X', 'a'), choices=('a',)))\n",
    )
    with pytest.raises(ValueError, match="FLAGS uses os, which is not a literal"):
        load_table(path)
    with pytest.raises(ValueError, match="no FLAGS = FlagSet"):
        load_table(_module(tmp_path, "X = 1\n"))
    with pytest.raises(ValueError, match="FLAGS is a int, not a FlagSet"):
        load_table(_module(tmp_path, "FLAGS = 1\n"))
