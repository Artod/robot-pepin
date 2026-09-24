"""RTAB-Map's global descriptor in its own bytes, and the invariant every node's descriptor keeps.

The bytes are pinned by LITERALS read in rtabmap/corelib/src/Compression.cpp (compressData: zlib,
then rows, cols and the OpenCV type as native int32) and GlobalDescriptorExtractor.h (type 1 is
the PyDescriptor, the only type Signature::compareTo reads): a test that re-derived them with the
module's own expressions would agree with the module instead of with RTAB-Map."""

from __future__ import annotations

import struct
import zlib

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
    mat_shape,
    normalised,
    null_vector,
    to_ros,
    uncompress_mat,
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

    d = PlaceDescriptor("t", np.ones(2, dtype=np.float32))
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
