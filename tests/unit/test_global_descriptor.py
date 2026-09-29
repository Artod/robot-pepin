"""RTAB-Map's global descriptor in its own bytes, and the invariant every node's descriptor keeps.

The bytes are pinned by LITERALS read in rtabmap/corelib/src/Compression.cpp (compressData: zlib,
then rows, cols and the OpenCV type as native int32) and GlobalDescriptorExtractor.h (type 1 is
the PyDescriptor, the only type Signature::compareTo reads): a test that re-derived them with the
module's own expressions would agree with the module instead of with RTAB-Map."""

from __future__ import annotations

import struct
import zlib
from typing import Any

import numpy as np
import pytest

from pepin.global_descriptor import (
    CV_8UC1,
    CV_32FC1,
    NULL_TAG,
    TYPE_PY_DESCRIPTOR,
    Census,
    PlaceDescriptor,
    compress_mat,
    database_census,
    descriptors_attached,
    global_descriptor_setting,
    mat_shape,
    normalised,
    null_vector,
    to_ros,
    uncompress_mat,
    unit_or_none,
)


def test_the_bytes_are_zlib_then_rows_cols_and_the_opencv_type() -> None:
    vector = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    blob = compress_mat(vector.reshape(1, 3), CV_32FC1)
    assert blob[-12:] == struct.pack("<iii", 1, 3, 5)  # CV_32FC1 is 5
    assert zlib.decompress(blob[:-12]) == vector.tobytes()
    assert mat_shape(blob) == (1, 3, 5)
    back = uncompress_mat(blob)
    assert back.dtype == np.float32 and back.shape == (1, 3) and np.array_equal(back[0], vector)
    text = compress_mat(np.frombuffer(b"boq", dtype=np.uint8), CV_8UC1)
    assert text[-12:] == struct.pack("<iii", 1, 3, 0)  # CV_8UC1 is 0, a 1 x N row
    assert TYPE_PY_DESCRIPTOR == 1


def test_a_descriptor_travels_as_its_tag_and_its_row_vector() -> None:
    d = PlaceDescriptor("boq_dinov2@d72ee0ce", normalised(np.arange(1.0, 5.0)))
    info, data = d.blobs()
    back = PlaceDescriptor.from_blobs(info, data)
    assert back.tag == d.tag and back.dim == 4 and np.allclose(back.vector, d.vector)
    assert abs(float(np.linalg.norm(back.vector)) - 1.0) < 1e-6
    assert mat_shape(data) == (1, 4, CV_32FC1)


def test_the_null_descriptor_is_zeros_and_scores_one_half_against_anything() -> None:
    """Signature::compareTo scores (a . b + 1) / 2: the zero vector is 0.5 against every vector
    INCLUDING another null — a node with no picture is neither like nor unlike anything — where a
    unit null would score 1.0 against every other null."""
    null = PlaceDescriptor.null(8)
    assert null.is_null and null.tag == NULL_TAG and null.dim == 8
    assert not np.any(null.vector) and null.vector.dtype == np.float32
    rng = np.random.default_rng(1)
    for _ in range(5):
        v = normalised(rng.standard_normal(8))
        assert (float(null.vector @ v) + 1.0) / 2.0 == 0.5
    assert (float(null.vector @ null_vector(8)) + 1.0) / 2.0 == 0.5
    assert np.array_equal(normalised(np.zeros(3)), np.zeros(3))


def test_the_ros_message_carries_type_one_and_the_two_blobs() -> None:
    class Message:
        def __init__(self, **fields: object) -> None:
            self.__dict__.update(fields)

    d = PlaceDescriptor("t", np.array([0.6, 0.8], dtype=np.float32))
    msg = to_ros(d, Message, header="h")
    assert msg.type == 1 and msg.header == "h"  # type: ignore[attr-defined]
    assert (msg.info, msg.data) == d.blobs()  # type: ignore[attr-defined]


def test_what_a_blob_that_is_not_one_is_refused_as() -> None:
    with pytest.raises(ValueError):
        mat_shape(b"short")
    with pytest.raises(ValueError):
        uncompress_mat(zlib.compress(b"abc") + struct.pack("<iii", 1, 3, 6))  # CV_64F
    with pytest.raises(ValueError):
        compress_mat(np.zeros((0, 3)), CV_32FC1)


def test_the_census_refuses_every_way_the_invariant_breaks() -> None:
    good = Census(169, 0, 0, {("boq@1", 12288): 147, (NULL_TAG, 12288): 22})
    assert good.refusal() is None and good.refusal("boq@1", 12288) is None
    assert good.text() == "169 nodes: 147 boq@1/12288, 22 null/12288"
    assert "22 of 169 nodes carry no descriptor" in str(
        Census(169, 22, 0, {("boq@1", 12288): 147}).refusal()
    )
    assert "more than one" in str(Census(3, 0, 1, {("boq@1", 4): 3}).refusal())
    assert "lengths" in str(Census(2, 0, 0, {("a", 4): 1, ("b", 8): 1}).refusal())
    assert "the snapshots' 16384" in str(good.refusal(dim=16384))
    assert "described by ['boq@1']" in str(good.refusal(tag="boq@2"))
    empty = Census(0, 0, 0, {})
    assert empty.refusal("boq@1", 12288) is None, "an empty database takes any descriptor"


def test_the_census_travels_as_one_json_line_and_a_bad_one_is_none() -> None:
    census = Census(169, 0, 0, {("boq@1", 12288): 147, (NULL_TAG, 12288): 22})
    assert Census.from_json(census.to_json()) == census
    assert Census.from_json("") is None and Census.from_json('{"nodes": 1}') is None


def test_the_descriptor_likelihood_is_sent_only_when_nothing_can_abort_rtabmap() -> None:
    from pepin.global_descriptor import (
        DESCRIPTOR_PARAMETERS,
        WORDS_PARAMETERS,
        SnapshotPlace,
        recognition_parameters,
    )

    good = Census(3, 0, 0, {("boq@1", 8): 2, (NULL_TAG, 8): 1})
    place = SnapshotPlace(True, "service", 8, "boq@1")
    assert recognition_parameters("descriptor", place, good, True) == (
        DESCRIPTOR_PARAMETERS,
        "descriptor (3 nodes: 2 boq@1/8, 1 null/8)",
    )
    assert DESCRIPTOR_PARAMETERS == {
        "Kp/TfIdfLikelihoodUsed": "false",
        "Rtabmap/VirtualPlaceLikelihoodRatio": "1",
    }
    for args, why in (
        ((place, good, False), "rtabmap-keep-global-descriptors.patch"),
        ((None, good, True), "sensor_pack has not said"),
        ((SnapshotPlace(False, "service", 0), good, True), "carry no descriptor"),
        ((place, None, True), "no census"),
        ((SnapshotPlace(True, "service", 16), good, True), "the snapshots' 16"),
    ):
        wanted, said = recognition_parameters("descriptor", *args)  # type: ignore[arg-type]
        assert wanted == WORDS_PARAMETERS and why in said, said
    assert recognition_parameters("words", place, good, True) == (WORDS_PARAMETERS, "words")


# ---- nothing but a unit vector or the null one ------------------------------------------------
def test_a_vector_that_is_not_a_finite_unit_vector_never_becomes_a_descriptor() -> None:
    """compareTo asserts (a . b + 1) / 2 >= 0 (Signature.cpp:263): a NaN fails it (NaN compares
    false) and aborts RTAB-Map, and one stored in the database fails it against every node."""
    nan = np.array([np.nan, 1.0, 0.0], dtype=np.float32)
    inf = np.array([np.inf, 0.0, 0.0], dtype=np.float32)
    for bad in (nan, inf, np.array([2.0, 0.0]), np.zeros(3), np.zeros(0)):
        assert unit_or_none(bad) is None, bad
        with pytest.raises(ValueError, match="not a finite unit vector"):
            PlaceDescriptor("boq@1", bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="null descriptor"):
        PlaceDescriptor(NULL_TAG, np.array([0.6, 0.8], dtype=np.float32))
    assert unit_or_none(normalised(nan)) is None, "normalising does not rescue a NaN"
    near = np.array([0.6, 0.8], dtype=np.float32) * np.float32(1.0005)  # within 1e-3
    d = PlaceDescriptor("boq@1", near)
    assert abs(float(np.linalg.norm(d.vector.astype(np.float64))) - 1.0) < 1e-6, "made unit"
    assert d.vector.dtype == np.float32 and d.vector.flags["C_CONTIGUOUS"]


# ---- whether the snapshots carry descriptors at all --------------------------------------------
def test_global_descriptor_auto_follows_the_core_and_on_and_off_do_not() -> None:
    """auto attaches exactly on an RTAB-Map that keeps a node's descriptor across the reload of
    its data: the unpatched core would drop one and abort at the next comparison, even under the
    words (mapping's rehearsal compares nodes too)."""
    assert descriptors_attached("auto", True) and not descriptors_attached("auto", False)
    assert descriptors_attached("on", False) and not descriptors_attached("off", True)
    assert global_descriptor_setting({}) == "auto"
    assert global_descriptor_setting({"PEPIN_GLOBAL_DESCRIPTOR": "off"}) == "off"
    for bad in ("0", "true", "Auto"):
        with pytest.raises(ValueError, match="PEPIN_GLOBAL_DESCRIPTOR"):
            global_descriptor_setting({"PEPIN_GLOBAL_DESCRIPTOR": bad})


# ---- the descriptor only while it can see -------------------------------------------------------
def test_the_descriptor_falls_back_to_the_words_while_the_service_is_not_describing() -> None:
    """A null query scores every node 0.5, adjustLikelihood's z-scores are all zero and no
    hypothesis forms: descriptor mode would go blind where the words still recognise."""
    from pepin.global_descriptor import (
        DESCRIPTOR_PARAMETERS,
        WORDS_PARAMETERS,
        SnapshotPlace,
        recognition_parameters,
    )

    good = Census(3, 0, 0, {("boq@1", 8): 2, (NULL_TAG, 8): 1})
    healthy = SnapshotPlace(True, "service", 8, "boq@1", recent=10, recent_null=5)
    assert recognition_parameters("descriptor", healthy, good, True)[0] == DESCRIPTOR_PARAMETERS
    sick = SnapshotPlace(True, "service", 8, "boq@1", recent=10, recent_null=6)
    wanted, why = recognition_parameters("descriptor", sick, good, True)
    assert wanted == WORDS_PARAMETERS and "6 of the last 10 camera snapshots" in why
    assert recognition_parameters("descriptor", sick, good, True, 1.0)[0] == DESCRIPTOR_PARAMETERS
    off = SnapshotPlace(True, "null", 8, "boq@1")
    wanted, why = recognition_parameters("descriptor", off, good, True)
    assert wanted == WORDS_PARAMETERS and "place_descriptor is off" in why
    fresh = SnapshotPlace(True, "service", 8, "")  # no camera snapshot yet: nothing to judge
    assert recognition_parameters("descriptor", fresh, good, True)[0] == DESCRIPTOR_PARAMETERS


def test_the_snapshots_state_travels_with_its_counts_and_an_older_line_still_reads() -> None:
    from pepin.global_descriptor import SnapshotPlace

    state = SnapshotPlace(True, "service", 8, "boq@1", recent=7, recent_null=2)
    assert SnapshotPlace.from_json(state.to_json()) == state
    old = '{"attached": true, "dim": 8, "source": "service", "tag": "boq@1"}'
    assert SnapshotPlace.from_json(old) == SnapshotPlace(True, "service", 8, "boq@1", 0, 0)
    assert SnapshotPlace.from_json("[]") is None and SnapshotPlace.from_json("x") is None


# ---- the census of a database -----------------------------------------------------------------
def _database(path: Any, rows: list[tuple[int, int, bytes | None, bytes]], nodes: int) -> None:
    """A database in RTAB-Map's schema, as far as a census reads it."""
    import sqlite3

    db = sqlite3.connect(str(path))
    db.execute("CREATE TABLE Node (id INTEGER PRIMARY KEY)")
    db.execute(
        "CREATE TABLE GlobalDescriptor (node_id INTEGER NOT NULL, type INTEGER NOT NULL,"
        " info BLOB, data BLOB NOT NULL)"
    )
    db.executemany("INSERT INTO Node (id) VALUES (?)", [(i,) for i in range(1, nodes + 1)])
    db.executemany("INSERT INTO GlobalDescriptor VALUES (?, ?, ?, ?)", rows)
    db.commit()
    db.close()


def test_the_census_reads_each_length_from_the_trailer_and_names_what_it_cannot_read(
    tmp_path: Any,
) -> None:
    good = PlaceDescriptor("boq@1", normalised(np.arange(1.0, 9.0)))
    null = PlaceDescriptor.null(8)
    rows = [(1, 1, *good.blobs()), (2, 1, *null.blobs()), (3, 1, b"junk", b"no")]
    _database(tmp_path / "db", rows, nodes=4)
    census = database_census(tmp_path / "db")
    assert census.nodes == 4 and census.missing == 1 and census.repeated == 0
    assert census.kinds == {("boq@1", 8): 1, (NULL_TAG, 8): 1, ("unreadable", 0): 1}
    assert census.refusal() is not None
    assert database_census(tmp_path / "absent") == Census(0, 0, 0, {})
