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

A VECTOR IS A UNIT VECTOR OR THE NULL ONE, and nothing else ever leaves this module:
``compareTo`` also asserts ``(a . b + 1) / 2 >= 0`` (Signature.cpp:263), which a NaN fails (NaN
compares false) and a vector longer than one can fail, and a NaN stored in the database would fail
it against every node for ever. :class:`PlaceDescriptor` refuses anything but a finite vector
within :data:`UNIT_TOLERANCE` of unit length (made exactly unit) or the zero vector under the null
tag; :func:`unit_or_none` is the same test for a vector that may be replaced by the null one.

THE BYTES are ``rtabmap::compressData`` (corelib/src/Compression.cpp:208-237): the matrix's raw
bytes through zlib's ``compress``, then three native int32 — rows, cols and the OpenCV type — which
is what ``rtabmap_conversions::globalDescriptorFromROS`` uncompresses from
``rtabmap_msgs/GlobalDescriptor.data`` and ``.info`` (MsgConversion.cpp:643-646 at the pinned
0.22.1-jazzy) and what the database's ``GlobalDescriptor`` table holds in its ``info`` and ``data``
blobs (DBDriverSqlite3.cpp:3556-3610 reads, :6811-6860 writes). The vector is a 1 x D ``CV_32FC1``;
the info is the model's tag as a 1 x N ``CV_8UC1`` string — ``null`` for the null descriptor —
which is how a database census tells which weights described each node.

A KNOWN LIMIT: RTAB-Map computes the likelihood only for a frame with ORB words
(Rtabmap.cpp:1971 asks ``!signature->isBadSignature()``, and Signature.cpp:341 calls a node with
no words bad), so a picture too dark for GFTT/ORB to find a corner gets no place recognition by
descriptor either, and the words are still extracted for every frame. The descriptor replaces the
words' SCORE, not the words themselves.
"""

from __future__ import annotations

import json
import struct
import zlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

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
# How far from unit length a place vector may be and still count as one (then made exactly unit):
# BoQ's float32 L2 normalisation lands within 1e-6; anything further is a model that did not
# normalise, or worse.
UNIT_TOLERANCE = 1e-3

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
    """``vector`` as float32 with unit length (the zero vector stays zero; a vector with a NaN or
    an infinity comes back with them, which :func:`unit_or_none` then refuses)."""
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v.astype(np.float64)))
    return v if norm == 0.0 else np.ascontiguousarray(v / norm, dtype=np.float32)


def unit_or_none(vector: npt.ArrayLike) -> Vector | None:
    """``vector`` as a float32 row of exactly unit length when it is finite, not empty and within
    :data:`UNIT_TOLERANCE` of unit length; ``None`` otherwise (it must not reach RTAB-Map)."""
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    if v.size == 0 or not bool(np.all(np.isfinite(v))):
        return None
    norm = float(np.linalg.norm(v.astype(np.float64)))
    if abs(norm - 1.0) > UNIT_TOLERANCE:
        return None
    return np.ascontiguousarray(v / np.float32(norm), dtype=np.float32)


@dataclass(frozen=True)
class PlaceDescriptor:
    """One node's descriptor: which model said it (``tag``, or ``null``) and the vector — a unit
    vector under a model's tag, the zero vector under the null tag, and nothing else (a
    :class:`ValueError` names what is wrong)."""

    tag: str
    vector: Vector

    def __post_init__(self) -> None:
        if self.tag == NULL_TAG:
            v = np.ascontiguousarray(np.asarray(self.vector, dtype=np.float32).reshape(-1))
            if v.size == 0 or np.any(v != 0.0):
                raise ValueError("the null descriptor is a non-empty zero vector")
        else:
            unit = unit_or_none(self.vector)
            if unit is None:
                raise ValueError(f"{self.tag}: not a finite unit vector (RTAB-Map would abort)")
            v = unit
        object.__setattr__(self, "vector", v)

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

    def to_json(self) -> str:
        """The census as one JSON line (``PEPIN_PLACE_CENSUS``, ``place_backfill.py --json``)."""
        kinds = [[tag, dim, count] for (tag, dim), count in sorted(self.kinds.items())]
        return json.dumps(
            {
                "nodes": self.nodes,
                "missing": self.missing,
                "repeated": self.repeated,
                "kinds": kinds,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> Census | None:
        """A census from :meth:`to_json`'s line, or ``None`` when it is not one."""
        try:
            data = json.loads(text)
            kinds = {(str(tag), int(dim)): int(count) for tag, dim, count in data["kinds"]}
            return cls(int(data["nodes"]), int(data["missing"]), int(data["repeated"]), kinds)
        except (TypeError, ValueError, KeyError):
            return None


# THE DATABASE'S CENSUS AT A START, as rtabmap_frame reads it: :meth:`Census.to_json`'s line in this
# variable, set by vslam.launch.py from :func:`database_census` of the very file RTAB-Map is given,
# before RTAB-Map opens it (a reader beside a writing RTAB-Map could make its commit fail and abort
# it) — on EVERY start of the launch, so a container restarted after the file was swapped counts
# again. Fixed for that RTAB-Map's life, as the database's nodes are: every node it adds after the
# census comes from sensor_pack with its one descriptor, and RTAB-Map is not respawned.
CENSUS_ENV = "PEPIN_PLACE_CENSUS"
UNREADABLE = "unreadable"  # the tag a census gives a descriptor whose bytes it cannot read


def database_census(path: str | Path) -> Census:
    """What an RTAB-Map database's nodes carry: every ``Node`` id against its ``GlobalDescriptor``
    rows, each row's (tag, length) read from its info and from the LAST 12 BYTES of its data (the
    trailer, cut by sqlite's ``substr``: a 48 KB vector never crosses into Python, though sqlite
    still reads it from disk — 169 nodes 16 ms, 2000 synthetic ones 92 MB 31 ms warm,
    scratch/models/census_timing.py). Every row counts,
    whatever its type — compareTo's size check does (Signature.cpp:252) — and a row of another
    type is tagged ``type N: ...``. Read-only; a database that is not there is zero nodes (an
    empty room takes any descriptor). NEVER beside a running RTAB-Map on the same file: it sets no
    sqlite busy timeout, so a reader's lock during its commit fails its assertion and aborts it —
    the vslam launch takes this census before RTAB-Map starts, and ros/tools/place_backfill.py
    --check reads a copy of a database a container holds. ``sqlite3.Error`` when it is no
    database."""
    import sqlite3

    path = Path(path)
    if not path.exists():
        return Census(0, 0, 0, {})
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        ids = [int(r[0]) for r in db.execute("SELECT id FROM Node")]
        rows: dict[int, list[tuple[str, int]]] = {}
        table = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='GlobalDescriptor'"
        ).fetchone()
        if table is not None:
            for node_id, kind, info, trailer in db.execute(
                "SELECT node_id, type, info, substr(data, -12) FROM GlobalDescriptor"
            ):
                try:
                    tag = uncompress_mat(info).tobytes().decode(errors="replace") if info else ""
                    rows_, cols, _cv = mat_shape(bytes(trailer or b""))
                    length = rows_ * cols
                except (ValueError, zlib.error):
                    tag, length = UNREADABLE, 0
                if int(kind) != TYPE_PY_DESCRIPTOR:
                    tag = f"type {int(kind)}: {tag}"
                rows.setdefault(int(node_id), []).append((tag, length))
    finally:
        db.close()
    kinds: dict[tuple[str, int], int] = {}
    missing = repeated = 0
    for node_id in ids:
        found = rows.get(node_id, [])
        if not found:
            missing += 1
        elif len(found) > 1:
            repeated += 1
        else:
            kinds[found[0]] = kinds.get(found[0], 0) + 1
    return Census(len(ids), missing, repeated, kinds)


# ---------------------------------------------------------------- what the snapshots carry
PLACE_TOPIC = "/sensor_pack/place"
PLACE_WORDS, PLACE_DESCRIPTOR = "words", "descriptor"
TFIDF = "Kp/TfIdfLikelihoodUsed"
# HOW THE LIKELIHOODS BECOME A HYPOTHESIS (Rtabmap::adjustLikelihood, Rtabmap.cpp:5303-5372). With
# the ratio 0 (RTAB-Map's default) a node above mean + stddev scores (value - stddev) / mean and
# the "no place" hypothesis mean / stddev + 1. The words' TF-IDF scores are sparse (most nodes
# share no word, score 0 and are left out of the mean), so that works; a descriptor scores EVERY
# node, all of them in (cos + 1) / 2 ~ 0.5-0.75, so the mean is large, the deviation small, and
# "no place" wins every time. The ratio 1 scores (value - mean) / stddev and "no place"
# stddev / (max - mean) + 1 — a z-score, which is what a dense similarity needs. Measured
# 2026-09-24 on the replay of the evening runs (scratch/models/matrix_report.py): ratio 0, the
# highest hypothesis 0.009-0.011 and not one of 95 camera updates localised; ratio 1, medians
# 0.42 / 0.58-0.63 / 0.88 on runs 0457 / 0460 / 0466 and 16 of 23, 14 of 17, 54 of 55 localised.
VIRTUAL_PLACE_RATIO = "Rtabmap/VirtualPlaceLikelihoodRatio"
WORDS_PARAMETERS = {TFIDF: "true", VIRTUAL_PLACE_RATIO: "0"}
DESCRIPTOR_PARAMETERS = {TFIDF: "false", VIRTUAL_PLACE_RATIO: "1"}
# RTAB-Map 0.22.1 DROPS a node's descriptor the first time it registers against it: when the node's
# picture or scan is not in memory, Memory::computeTransform replaces the node's whole sensor data
# with the Data table's (Memory.cpp:2896-2909, getNodeData -> DBDriver::loadNodeDataQuery, which
# never reads the GlobalDescriptor table), and the next compareTo against that node aborts
# (Signature.cpp:252). Measured 2026-09-24: the replay's first rejected loop closure on node 353,
# then "Signature.cpp:252::compareTo() Condition ... not met!" on the next update. Every database
# node is such a node (they are loaded without their data), and so is every node of this session
# once it is saved (Memory::saveLocationData clears its data, Memory.cpp:2763) — which mapping's
# rehearsal can make the LAST node (rehearsalMerge, Memory.cpp:3958) and compare with the next
# one, whatever the likelihood. So NOTHING may carry a descriptor on RTAB-Map built without
# ros/patches/rtabmap-keep-global-descriptors.patch; ros/xfeat/patch_rtabmap.sh leaves this marker
# when the image carries it (the patch's file name without its ``rtabmap-``).
KEEPS_DESCRIPTORS_MARKER = "/opt/rtabmap_patches/keep-global-descriptors"


def rtabmap_keeps_descriptors(marker: str = KEEPS_DESCRIPTORS_MARKER) -> bool:
    """Whether the RTAB-Map in this container was built with the patch that keeps a node's
    descriptor across the reload of its data (its marker file exists)."""
    import os

    return os.path.exists(marker)


# WHETHER THE SNAPSHOTS CARRY DESCRIPTORS AT ALL: pepin_bringup.sensor_pack's global_descriptor
# (not live), which the vslam launch reads too, to set the rehearsal threshold that fits them.
# ``auto`` attaches exactly when this RTAB-Map keeps them (:func:`rtabmap_keeps_descriptors`),
# ``on`` always (a control against an unpatched core — it can abort RTAB-Map), ``off`` never.
GLOBAL_DESCRIPTOR_ENV = "PEPIN_GLOBAL_DESCRIPTOR"
ATTACH_AUTO, ATTACH_ON, ATTACH_OFF = "auto", "on", "off"
ATTACH_CHOICES = (ATTACH_AUTO, ATTACH_ON, ATTACH_OFF)


def global_descriptor_setting(environ: Mapping[str, str]) -> str:
    """The global_descriptor setting a node started in ``environ`` holds, read exactly as the
    flag reads its variable (one of :data:`ATTACH_CHOICES`, :data:`ATTACH_AUTO` when unset);
    :class:`ValueError` for any other word, as the node itself would refuse it."""
    value = environ.get(GLOBAL_DESCRIPTOR_ENV, ATTACH_AUTO)
    if value not in ATTACH_CHOICES:
        raise ValueError(
            f"{GLOBAL_DESCRIPTOR_ENV}: {value!r} is not one of {', '.join(ATTACH_CHOICES)}"
        )
    return value


def descriptors_attached(setting: str, rtabmap_keeps: bool) -> bool:
    """Whether the snapshots carry one descriptor each under global_descriptor ``setting``."""
    return setting == ATTACH_ON or (setting == ATTACH_AUTO and rtabmap_keeps)


# HOW MANY OF THE LAST CAMERA SNAPSHOTS a SnapshotPlace counts, and the share of them that may have
# carried the null descriptor before the descriptor likelihood is taken away: a query node whose
# descriptor is null scores 0.5 against EVERY node, Rtabmap::adjustLikelihood's z-scores are then
# all zero, and no hypothesis forms — the words would still have recognised the place. Ten at one
# snapshot a second is ten seconds: a service that went down costs at most five blind updates.
RECENT_CAMERA_SNAPSHOTS = 10
MAX_NULL_SHARE = 0.5


@dataclass(frozen=True)
class SnapshotPlace:
    """What pepin_bringup.sensor_pack attaches to its snapshots, said latched on
    :data:`PLACE_TOPIC` for rtabmap_frame: whether every snapshot carries one descriptor at all
    (``attached``), where a camera snapshot's comes from (``service``, or ``null`` when that is
    switched off), the length every one of them has, the tag of the last one the service
    described (empty until it has), and of the last ``recent`` camera snapshots how many went out
    with the null descriptor (``recent_null``: the service did not answer, answered late, or
    answered something that is not a unit vector)."""

    attached: bool
    source: str
    dim: int
    tag: str = ""
    recent: int = 0
    recent_null: int = 0

    def to_json(self) -> str:
        """One JSON object for a std_msgs/String."""
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> SnapshotPlace | None:
        """The state from :meth:`to_json`'s text, or ``None`` when it is not one."""
        try:
            data = json.loads(text)
            return cls(
                bool(data["attached"]),
                str(data["source"]),
                int(data["dim"]),
                str(data["tag"]),
                int(data.get("recent", 0)),
                int(data.get("recent_null", 0)),
            )
        except (TypeError, ValueError, KeyError, AttributeError):
            return None


def recognition_parameters(
    asked: str,
    snapshots: SnapshotPlace | None,
    census: Census | None,
    rtabmap_keeps: bool,
    max_null_share: float = MAX_NULL_SHARE,
) -> tuple[dict[str, str], str]:
    """The likelihood RTAB-Map should score places by, as the parameters it reads
    (:data:`WORDS_PARAMETERS`: the words' TF-IDF; :data:`DESCRIPTOR_PARAMETERS`: the descriptors'
    dot product, as z-scores), and the phrase that says why. ``descriptor`` is honoured only when
    comparing by descriptor cannot abort RTAB-Map — this RTAB-Map keeps a node's descriptor when
    it reloads the node's data (``rtabmap_keeps``, the patch's marker), the snapshots carry one
    each (``snapshots``), and the census of the database taken before RTAB-Map opened it says
    every node carries exactly one of the same length (and, once the service has answered, from
    the same weights) — and only while it can SEE: the camera snapshots' descriptors come from
    the service (not switched off) and no more than ``max_null_share`` of the recent ones were
    the null descriptor. Anything less is the words, and the phrase names what is missing."""
    words = dict(WORDS_PARAMETERS)
    if asked != PLACE_DESCRIPTOR:
        return words, PLACE_WORDS
    if not rtabmap_keeps:
        return words, (
            "words (descriptor asked; this RTAB-Map drops a node's descriptor when it reloads the"
            " node's data for a registration and would abort at the next comparison:"
            " ros/patches/rtabmap-keep-global-descriptors.patch, ros/laptop-build.sh xfeat)"
        )
    if snapshots is None:
        return words, "words (descriptor asked; sensor_pack has not said what its snapshots carry)"
    if not snapshots.attached:
        return words, "words (descriptor asked; the snapshots carry no descriptor)"
    if census is None:
        return words, "words (descriptor asked; no census of the database at this start)"
    refusal = census.refusal(snapshots.tag or None, snapshots.dim)
    if refusal is not None:
        return words, f"words (descriptor asked; {refusal}: ros/tools/place_backfill.py)"
    if snapshots.source != "service":
        return words, (
            "words (descriptor asked; sensor_pack's place_descriptor is off, so every snapshot"
            " carries the null descriptor and no place would score above another)"
        )
    if snapshots.recent and snapshots.recent_null > max_null_share * snapshots.recent:
        return words, (
            f"words (descriptor asked; {snapshots.recent_null} of the last {snapshots.recent}"
            " camera snapshots carried the null descriptor: the localisation service is not"
            " describing them — ros/models.sh status localization)"
        )
    return dict(DESCRIPTOR_PARAMETERS), f"descriptor ({census.text()})"
