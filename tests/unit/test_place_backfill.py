"""ros/tools/place_backfill.py on a synthetic database in RTAB-Map's own schema: every node ends
with exactly one descriptor (its picture's, or the null one), re-runs replace, a failure writes
nothing, the census tells every way the invariant can break, and a held database is refused."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from pepin.global_descriptor import NULL_TAG, TYPE_PY_DESCRIPTOR, PlaceDescriptor

TOOL = Path(__file__).resolve().parents[2] / "ros" / "tools" / "place_backfill.py"
DIM = 6
TAG = "boq_dinov2@feedbeef"

# The three tables the tool reads and writes, as rtabmap/corelib/src/resources/DatabaseSchema.sql.in
# (0.22.1) declares them — the columns it does not touch included, so an INSERT that forgot one
# NOT NULL column would fail here as it would on the real file.
SCHEMA = """
CREATE TABLE Node (id INTEGER NOT NULL, map_id INTEGER NOT NULL, weight INTEGER, stamp FLOAT,
    pose BLOB, ground_truth_pose BLOB, velocity BLOB, label TEXT, gps BLOB, env_sensors BLOB,
    time_enter DATE, PRIMARY KEY (id));
CREATE TABLE Data (id INTEGER NOT NULL, image BLOB, depth BLOB, depth_confidence BLOB,
    calibration BLOB, scan BLOB, scan_info BLOB, ground_cells BLOB, obstacle_cells BLOB,
    empty_cells BLOB, cell_size FLOAT, view_point_x FLOAT, view_point_y FLOAT, view_point_z FLOAT,
    user_data BLOB, time_enter DATE, PRIMARY KEY (id));
CREATE TABLE GlobalDescriptor (node_id INTEGER NOT NULL, type INTEGER NOT NULL, info BLOB,
    data BLOB NOT NULL, FOREIGN KEY (node_id) REFERENCES Node(id));
CREATE INDEX IDX_GlobalDescriptor_node_id on GlobalDescriptor (node_id);
"""


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("place_backfill", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jpeg(value: int) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.full((12, 16, 3), value, dtype=np.uint8))
    assert ok
    return bytes(buf.tobytes())


@pytest.fixture
def database(tmp_path: Path) -> Path:
    """Five nodes: 1-3 with a picture, 4 lidar-only (no image), 5 an intermediate node."""
    path = tmp_path / "rtabmap.db"
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    for node_id, weight in ((1, 0), (2, 3), (3, 0), (4, 0), (5, -1)):
        db.execute("INSERT INTO Node(id, map_id, weight) VALUES (?, 0, ?)", (node_id, weight))
    for node_id in (1, 2, 3):
        db.execute("INSERT INTO Data(id, image) VALUES (?, ?)", (node_id, _jpeg(40 * node_id)))
    for node_id in (4, 5):
        db.execute("INSERT INTO Data(id, scan) VALUES (?, ?)", (node_id, b"scan"))
    db.commit()
    db.close()
    return path


def describe(rgb: Any) -> tuple[np.ndarray, str]:
    """A unit vector that says which picture it saw (its mean grey level)."""
    v = np.zeros(DIM, dtype=np.float32)
    v[int(rgb.mean()) % DIM] = 1.0
    return v, TAG


def _rows(path: Path) -> dict[int, list[PlaceDescriptor]]:
    db = sqlite3.connect(path)
    out: dict[int, list[PlaceDescriptor]] = {}
    for node_id, kind, info, data in db.execute(
        "SELECT node_id, type, info, data FROM GlobalDescriptor"
    ):
        assert kind == TYPE_PY_DESCRIPTOR
        out.setdefault(node_id, []).append(PlaceDescriptor.from_blobs(info, data))
    db.close()
    return out


def test_every_node_ends_with_exactly_one_descriptor(database: Path) -> None:
    tool = _tool()
    assert tool.census(database).refusal() == "5 of 5 nodes carry no descriptor"
    counts = tool.backfill(database, describe, DIM, progress=lambda _line: None)
    assert counts == {"described": 3, "null: no picture": 2}
    rows = _rows(database)
    assert sorted(rows) == [1, 2, 3, 4, 5] and all(len(v) == 1 for v in rows.values())
    assert [rows[i][0].tag for i in (1, 2, 3)] == [TAG] * 3
    assert rows[4][0].tag == rows[5][0].tag == NULL_TAG and not rows[4][0].vector.any()
    assert all(d.dim == DIM for v in rows.values() for d in v)
    census = tool.census(database)
    assert census.refusal(TAG, DIM) is None
    assert census.text() == f"5 nodes: 3 {TAG}/{DIM}, 2 null/{DIM}"


def test_a_re_run_replaces_and_a_failure_writes_nothing(database: Path) -> None:
    tool = _tool()
    tool.backfill(database, describe, DIM, progress=lambda _line: None)
    tool.backfill(database, describe, DIM, progress=lambda _line: None)
    assert all(len(v) == 1 for v in _rows(database).values()), "replaced, not added"
    before = _rows(database)
    calls = {"n": 0}

    def fails_on_the_third(rgb: Any) -> tuple[np.ndarray, str] | None:
        calls["n"] += 1
        return None if calls["n"] == 3 else describe(rgb)

    with pytest.raises(RuntimeError, match="node 3"):
        tool.backfill(database, fails_on_the_third, DIM, progress=lambda _line: None)
    after = _rows(database)
    assert {k: [d.tag for d in v] for k, v in after.items()} == {
        k: [d.tag for d in v] for k, v in before.items()
    }, "rolled back whole"
    with pytest.raises(RuntimeError, match="not 6"):
        tool.backfill(database, lambda _rgb: (np.ones(3, np.float32), TAG), DIM, lambda _l: None)


def test_a_vector_rtabmap_would_abort_on_is_never_stored(database: Path) -> None:
    """One NaN in the database fails compareTo's UASSERT_MSG(dotProd >= 0) against that node at
    every comparison for ever: the backfill stops and writes nothing."""
    tool = _tool()
    tool.backfill(database, describe, DIM, progress=lambda _line: None)
    before = {k: [d.tag for d in v] for k, v in _rows(database).items()}
    for bad in (np.full(DIM, np.nan, np.float32), np.full(DIM, 3.0, np.float32)):
        with pytest.raises(RuntimeError, match="no finite unit vector"):
            tool.backfill(database, lambda _rgb, b=bad: (b, TAG), DIM, lambda _l: None)
    assert {k: [d.tag for d in v] for k, v in _rows(database).items()} == before


def test_check_of_a_held_database_reads_a_copy(
    database: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """RTAB-Map sets no sqlite busy timeout: a reader's lock during its commit aborts it. The
    census of a database a container holds is taken of a copy, and says so."""
    tool = _tool()
    tool.backfill(database, describe, DIM, progress=lambda _line: None)
    monkeypatch.setattr(tool, "holders", lambda _path: ["pepin-vslam"])
    opened: list[str] = []
    real = tool.database_census
    monkeypatch.setattr(
        tool, "database_census", lambda path: opened.append(str(path)) or real(path)
    )
    assert tool.main(["--check", "--json", str(database)]) == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["nodes"] == 5 and "read from a copy: pepin-vslam hold it" in err
    assert opened and opened[0] != str(database), "never the held file itself"


def test_the_census_sees_every_way_the_invariant_breaks(database: Path) -> None:
    tool = _tool()
    tool.backfill(database, describe, DIM, progress=lambda _line: None)
    db = sqlite3.connect(database)
    info, data = PlaceDescriptor(TAG, np.full(4, 0.5, np.float32)).blobs()
    db.execute("INSERT INTO GlobalDescriptor VALUES (2, 1, ?, ?)", (info, data))
    db.execute("DELETE FROM GlobalDescriptor WHERE node_id = 4")
    db.commit()
    db.close()
    census = tool.census(database)
    assert census.missing == 1 and census.repeated == 1
    assert census.refusal() == "1 of 5 nodes carry no descriptor"
    assert tool.census(database.with_name("none.db")).nodes == 0, "no database: an empty room"


def test_check_prints_the_census_and_json_for_the_container(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _tool()
    assert tool.main(["--check", "--json", str(database)]) == 0
    out, err = capsys.readouterr()
    assert json.loads(out) == {"nodes": 5, "missing": 5, "repeated": 0, "kinds": []}
    assert "5 of 5 nodes carry no descriptor" in err


def test_a_held_or_mid_transaction_database_is_refused(
    database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _tool()
    monkeypatch.setattr(tool, "holders", lambda _path: [])
    assert tool.refusal(database) is None
    monkeypatch.setattr(tool, "holders", lambda _path: ["pepin-vslam"])
    assert "pepin-vslam" in str(tool.refusal(database))
    monkeypatch.setattr(tool, "holders", lambda _path: [])
    Path(f"{database}-journal").write_bytes(b"x")
    assert "mid-transaction" in str(tool.refusal(database))
    assert "not a database" in str(tool.refusal(database.with_name("none.db")))


def test_the_backup_is_a_copy_beside_it_named_by_the_time(database: Path) -> None:
    tool = _tool()
    copy = tool.backup(database, clock=lambda: 1758000000.0)
    assert copy.parent == database.parent and copy.name.startswith("rtabmap.db.backup-2025")
    assert copy.read_bytes() == database.read_bytes()
