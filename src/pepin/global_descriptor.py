"""RTAB-Map's global descriptor of a node: one place vector, in RTAB-Map's own bytes.

WHAT RTAB-MAP DOES WITH IT (0.22.1, read in the sources). With ``Kp/TfIdfLikelihoodUsed`` false
``Memory::computeLikelihood`` scores every node in working memory by ``Signature::compareTo``
(Memory.cpp:1905-1940), which, when both signatures carry a descriptor of type 1, is the dot
product of the two vectors rescaled from [-1, 1] to [0, 1] — ``(a . b + 1) / 2``, "we assume
normalized vectors!" (Signature.cpp:250-288) — and falls back to the words' pair ratio only when
neither carries one. ``Memory::rehearsal`` calls the same comparison on every new node while
MAPPING (Memory.cpp:3784-3839), whatever the likelihood setting.

THE INVARIANT, and why it is the whole design: ``compareTo`` begins with
``UASSERT(this->globalDescriptors().size() == s.globalDescriptors().size())`` (Signature.cpp:252),
so a node carrying a descriptor compared with one carrying none ABORTS RTAB-Map, and so do two
vectors of different lengths (``cv::Mat::dot``). So either every node carries exactly one
descriptor of one length, or none does. A snapshot with no picture — a lidar-only one, or a
picture whose descriptor could not be computed in time — gets :func:`null_vector`.

THE NULL DESCRIPTOR IS THE ZERO VECTOR, not a unit vector, and that was read in the source rather
than chosen by taste: ``compareTo`` scores it ``(0 + 1) / 2 = 0.5`` against EVERY vector,
including another null — exactly "orthogonal", the middle of the scale, a node that is neither like
nor unlike anything — and ``UASSERT_MSG(dotProd >= 0)`` holds. A fixed unit vector would score
1.0 against every OTHER null: a camera snapshot whose descriptor failed would then find every
lidar-only node its perfect match, and after ``Rtabmap::adjustLikelihood`` (Rtabmap.cpp:5303-5372)
the Bayes filter would propose one of them. Nothing in RTAB-Map normalises a descriptor or asserts
its norm.

THE BYTES are ``rtabmap::compressData`` (corelib/src/Compression.cpp:208-237): the matrix's raw
bytes through zlib's ``compress``, then three native int32 — rows, cols and the OpenCV type — which
is what ``rtabmap_conversions::globalDescriptorFromROS`` uncompresses from
``rtabmap_msgs/GlobalDescriptor.data`` and ``.info`` (MsgConversion.cpp:643-646 at the pinned
0.22.1-jazzy) and what the database's ``GlobalDescriptor`` table holds in its ``info`` and ``data``
blobs (DBDriverSqlite3.cpp:3556-3610 reads, :6811-6860 writes). The vector is a 1 x D ``CV_32FC1``;
the info is the model's tag as a 1 x N ``CV_8UC1`` string — ``null`` for the null descriptor —
which is how a database census tells which weights described each node.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# rtabmap::GlobalDescriptorExtractor::kPyDescriptor (GlobalDescriptorExtractor.h:42-44): the one
# type Signature::compareTo reads.
TYPE_PY_DESCRIPTOR = 1
# OpenCV's type codes as cv::Mat::type() returns them: depth + (channels - 1) * 8.
CV_8UC1 = 0
CV_32FC1 = 5
NULL_TAG = "null"
_TRAILER = struct.Struct("<iii")  # rows, cols, type (native int32; little-endian on both ends)
_DTYPES = {CV_8UC1: np.uint8, CV_32FC1: np.float32}

Vector = npt.NDArray[np.float32]


def compress_mat(array: npt.ArrayLike, cv_type: int) -> bytes:
    """``rtabmap::compressData`` of a 2-D matrix: zlib of its bytes, then rows, cols, type."""
    matrix = np.ascontiguousarray(np.asarray(array, dtype=_DTYPES[cv_type]))
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError(f"a descriptor matrix is 2-D and not empty, not {matrix.shape}")
    rows, cols = matrix.shape
    return zlib.compress(matrix.tobytes()) + _TRAILER.pack(rows, cols, cv_type)


def mat_shape(blob: bytes) -> tuple[int, int, int]:
    """(rows, cols, type) of a compressed matrix, read from its trailer without decompressing."""
    if len(blob) < _TRAILER.size:
        raise ValueError(f"{len(blob)} bytes is no compressed matrix")
    rows, cols, cv_type = _TRAILER.unpack(blob[-_TRAILER.size :])
    return int(rows), int(cols), int(cv_type)


def uncompress_mat(blob: bytes) -> npt.NDArray[np.generic]:
    """``rtabmap::uncompressData``: the matrix back, rows x cols of its type."""
    rows, cols, cv_type = mat_shape(blob)
    if cv_type not in _DTYPES:
        raise ValueError(f"OpenCV type {cv_type} is not one a descriptor uses")
    raw = zlib.decompressobj().decompress(blob[: -_TRAILER.size])
    matrix = np.frombuffer(raw, dtype=_DTYPES[cv_type])
    if matrix.size != rows * cols:
        raise ValueError(f"{matrix.size} values is not {rows}x{cols}")
    return matrix.reshape(rows, cols)


def null_vector(dim: int) -> Vector:
    """The descriptor of a node with no picture: ``dim`` zeros (see the module's docstring)."""
    return np.zeros(int(dim), dtype=np.float32)


def normalised(vector: npt.ArrayLike) -> Vector:
    """``vector`` as float32 with unit length (the zero vector stays zero)."""
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    return v if norm == 0.0 else np.ascontiguousarray(v / norm, dtype=np.float32)


@dataclass(frozen=True)
class PlaceDescriptor:
    """One node's descriptor: which model said it (``tag``, or ``null``) and the vector."""

    tag: str
    vector: Vector

    @property
    def dim(self) -> int:
        """The vector's length, which must be the same on every node."""
        return int(self.vector.size)

    @property
    def is_null(self) -> bool:
        """Whether this is the null descriptor of a node with no picture."""
        return self.tag == NULL_TAG

    @classmethod
    def null(cls, dim: int) -> PlaceDescriptor:
        """The null descriptor of length ``dim``."""
        return cls(NULL_TAG, null_vector(dim))

    def blobs(self) -> tuple[bytes, bytes]:
        """(info, data) as RTAB-Map stores and ships them (``compressData`` both)."""
        info = compress_mat(np.frombuffer(self.tag.encode(), dtype=np.uint8), CV_8UC1)
        return info, compress_mat(self.vector.reshape(1, -1), CV_32FC1)

    @classmethod
    def from_blobs(cls, info: bytes | None, data: bytes) -> PlaceDescriptor:
        """The descriptor RTAB-Map stored as (info, data); an empty info reads as no tag."""
        tag = uncompress_mat(info).tobytes().decode(errors="replace") if info else ""
        return cls(tag, np.asarray(uncompress_mat(data), dtype=np.float32).reshape(-1))


def to_ros(descriptor: PlaceDescriptor, message_type: type, header: object) -> object:
    """An ``rtabmap_msgs/GlobalDescriptor`` of ``message_type`` carrying ``descriptor``."""
    info, data = descriptor.blobs()
    return message_type(header=header, type=TYPE_PY_DESCRIPTOR, info=info, data=data)


@dataclass(frozen=True)
class Census:
    """What a database's nodes carry: the node count, how many carry no descriptor, how many
    more than one, and the (tag, dim) of the rest, counted."""

    nodes: int
    missing: int
    repeated: int
    kinds: Mapping[tuple[str, int], int]

    @property
    def dims(self) -> set[int]:
        """Every descriptor length present."""
        return {dim for _tag, dim in self.kinds}

    @property
    def tags(self) -> set[str]:
        """Every model tag present, the null descriptor's left out."""
        return {tag for tag, _dim in self.kinds if tag != NULL_TAG}

    def refusal(self, tag: str | None = None, dim: int | None = None) -> str | None:
        """Why RTAB-Map may NOT compare these nodes by descriptor, or ``None`` when it may:
        every node carries exactly one, all of one length — and, when given, that length is
        ``dim`` and the one model tag is ``tag`` (a vector from other weights is no measure of
        likeness to this one's)."""
        if self.missing:
            return f"{self.missing} of {self.nodes} nodes carry no descriptor"
        if self.repeated:
            return f"{self.repeated} of {self.nodes} nodes carry more than one descriptor"
        if len(self.dims) > 1:
            return f"descriptors of {len(self.dims)} lengths ({sorted(self.dims)})"
        if dim is not None and self.dims and self.dims != {dim}:
            return f"the nodes' descriptors are {sorted(self.dims)[0]} long, the snapshots' {dim}"
        if tag is not None and self.tags and self.tags != {tag}:
            return f"the nodes were described by {sorted(self.tags)}, the snapshots by {tag}"
        return None

    def text(self) -> str:
        """``169 nodes: 147 boq_dinov2@1a2b3c4d/12288, 22 null/12288``, and what is missing."""
        kinds = ", ".join(
            f"{count} {tag}/{dim}" for (tag, dim), count in sorted(self.kinds.items())
        )
        extra = ""
        if self.missing:
            extra += f", {self.missing} without"
        if self.repeated:
            extra += f", {self.repeated} with more than one"
        return f"{self.nodes} nodes: {kinds or 'no descriptors'}{extra}"
