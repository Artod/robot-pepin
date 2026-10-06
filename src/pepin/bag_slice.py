"""A time window of MCAP files cut into one rosbag2 bag, the messages copied as bytes.

The ring (:mod:`pepin.ring`) keeps a drive's topics in one-minute files; a goal's bag is the
part of them between two log times, written as ``<out>/<out.name>_0.mcap`` with the
``metadata.yaml`` rosbag2 (Jazzy, version 9) writes, so ``rosbag2_py``, ``ros2 bag play``,
``ros/tools/bag_to_tape.py`` and ``ros/vio_replay.sh`` read it as a bag ``ros2 bag record``
made. ``ros2 bag`` has no cut of its own (``convert``'s time range needs closed bags with their
metadata, and the ring's newest file is neither), so this uses the ``mcap`` library: pure
Python, nothing deserialised, the schemas, channels and their QoS metadata carried over.

The ring's newest file is still being written: it has no summary, its last record may be half
on disk. It is read record by record up to the last whole one (:func:`read_messages`).

Two things a fresh ``ros2 bag record`` hears at its start that a window cut from the middle of a
recording does not hold, and both are written at the window's start: the latched messages the
caller keeps (``carry``, :class:`pepin.ring.LatchedStore`'s ``/tf_static``) and, for each of
``carry_last``, that topic's last message before the window in the files read.
"""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROSBAG2_VERSION = 9  # Jazzy's metadata.yaml
ROS_DISTRO = "jazzy"
CHUNK_BYTES = 768 * 1024  # rosbag2's own MCAP chunk
PROFILE = "ros2"


@dataclass(frozen=True)
class Carry:
    """A message written at the window's start: what a fresh subscription hears of a latched
    topic."""

    topic: str
    data: bytes


@dataclass
class Sliced:
    """What a cut wrote: the bag, its messages and bytes, its span, and how far the files went."""

    path: Path
    messages: int = 0
    bytes: int = 0
    start_ns: int | None = None
    end_ns: int | None = None
    first_ns: int | None = None  # the window's first message of the ring's own (not carried)
    newest_ns: int = 0  # the newest log time in the files read: the ring's coverage
    files: int = 0
    open_files: int = 0  # files read without a summary (the one being written)
    per_topic: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _Topic:
    """A topic as the source files describe it."""

    name: str
    schema_name: str
    schema_encoding: str
    schema_data: bytes
    message_encoding: str
    metadata: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _Message:
    topic: _Topic
    log_time: int
    publish_time: int
    sequence: int
    data: bytes


def _topic(schema: Any, channel: Any) -> _Topic:
    return _Topic(
        channel.topic,
        schema.name if schema is not None else "",
        schema.encoding if schema is not None else "",
        bytes(schema.data) if schema is not None else b"",
        channel.message_encoding,
        tuple(sorted(channel.metadata.items())),
    )


def _summary(handle: Any) -> Any:
    """A closed file's summary, or None for one still being written (no footer)."""
    from mcap.reader import make_reader

    try:
        return make_reader(handle).get_summary()
    except Exception:  # no footer, a torn footer: the stream read below decides what is there
        return None


def read_messages(path: Path, topics: dict[str, _Topic] | None = None) -> Iterator[_Message]:
    """Every message of one file in file order; a file without a summary is read record by record
    up to its last whole record. ``topics`` collects every channel the file declares, messages
    or not (a latched topic's carried message needs its schema)."""
    from mcap.reader import make_reader
    from mcap.records import Channel, Message, Schema
    from mcap.stream_reader import StreamReader

    known = topics if topics is not None else {}
    with path.open("rb") as handle:
        summary = _summary(handle)
        if summary is not None:
            for channel in summary.channels.values():
                schema = summary.schemas.get(channel.schema_id)
                known.setdefault(channel.topic, _topic(schema, channel))
            handle.seek(0)
            for schema, channel, message in make_reader(handle).iter_messages(log_time_order=False):
                yield _Message(
                    known.get(channel.topic) or _topic(schema, channel),
                    message.log_time,
                    message.publish_time,
                    message.sequence,
                    bytes(message.data),
                )
            return
        handle.seek(0)
        schemas: dict[int, Any] = {}
        channels: dict[int, _Topic] = {}
        records = StreamReader(handle, emit_chunks=False).records
        while True:
            try:
                record = next(records)
            except StopIteration:
                return
            except Exception:  # the torn end of an open file: struct.error, EndOfFile, a length
                return  # read from half a header (RecordLengthLimitExceeded)
            if isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Channel):
                described = _topic(schemas.get(record.schema_id), record)
                channels[record.id] = described
                known.setdefault(record.topic, described)
            elif isinstance(record, Message) and record.channel_id in channels:
                yield _Message(
                    channels[record.channel_id],
                    record.log_time,
                    record.publish_time,
                    record.sequence,
                    bytes(record.data),
                )


def is_open(path: Path) -> bool:
    """True for a file with no summary: the ring's file being written, or one never closed."""
    with path.open("rb") as handle:
        return _summary(handle) is None


class _Out:
    """The cut's MCAP writer: one schema and one channel per topic, registered on first use."""

    def __init__(self, handle: Any) -> None:
        from mcap.writer import CompressionType, Writer

        self._writer = Writer(handle, chunk_size=CHUNK_BYTES, compression=CompressionType.NONE)
        self._writer.start(profile=PROFILE, library="pepin.bag_slice")
        self._schemas: dict[tuple[str, str, bytes], int] = {}
        self._channels: dict[str, int] = {}
        self.topics: dict[str, _Topic] = {}

    def write(self, topic: _Topic, log_time: int, publish_time: int, seq: int, data: bytes) -> None:
        channel = self._channels.get(topic.name)
        if channel is None:
            key = (topic.schema_name, topic.schema_encoding, topic.schema_data)
            schema = self._schemas.get(key)
            if schema is None and topic.schema_name:
                schema = self._writer.register_schema(*key)
                self._schemas[key] = schema
            channel = self._writer.register_channel(
                topic.name, topic.message_encoding, schema or 0, dict(topic.metadata)
            )
            self._channels[topic.name] = channel
            self.topics[topic.name] = topic
        self._writer.add_message(channel, log_time, data, publish_time, seq)

    def finish(self) -> None:
        self._writer.finish()  # type: ignore[no-untyped-call]  # mcap 1.5 leaves it unannotated


def cut(
    files: Sequence[Path],
    start_ns: int,
    end_ns: int,
    out: Path,
    *,
    carry: Sequence[Carry] = (),
    carry_last: Collection[str] = (),
) -> Sliced:
    """Write the messages of ``files`` logged in ``[start_ns, end_ns]`` as the bag ``out``; the
    carried messages first, at ``start_ns``. ``out`` is written whole beside itself
    (``<out>.part``) and renamed, so a reader that finds ``out`` finds it finished."""
    if end_ns < start_ns:
        raise ValueError(f"the window ends ({end_ns}) before it starts ({start_ns})")
    part = out.with_name(out.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    file_name = f"{out.name}_0.mcap"
    result = Sliced(out)
    topics: dict[str, _Topic] = {}
    last_before: dict[str, _Message] = {}
    carried = False

    def write_carried() -> None:
        """The carried messages, once, before the window's first message (or at the end)."""
        nonlocal carried
        carried = True
        for item in carry:
            described = topics.get(item.topic)
            if described is not None:
                _record(result, writer, described, start_ns, start_ns, 0, item.data)
        for name in carry_last:
            if name in last_before:
                m = last_before[name]
                _record(result, writer, m.topic, start_ns, m.publish_time, m.sequence, m.data)

    with (part / file_name).open("wb") as handle:
        writer = _Out(handle)
        # Streamed in file order (rosbag2 writes in arrival order, its reader orders by the
        # chunk index): a 30-minute run never sits in memory.
        for path in files:
            result.files += 1
            result.open_files += is_open(path)
            for m in read_messages(path, topics):
                result.newest_ns = max(result.newest_ns, m.log_time)
                if m.log_time < start_ns:
                    if m.topic.name in carry_last and not carried:
                        last_before[m.topic.name] = m
                elif m.log_time <= end_ns:
                    if not carried:
                        write_carried()
                    if result.first_ns is None:
                        result.first_ns = m.log_time
                    _record(result, writer, m.topic, m.log_time, m.publish_time, m.sequence, m.data)
        if not carried:
            write_carried()
        writer.finish()
    (part / "metadata.yaml").write_text(metadata_yaml(result, writer.topics, file_name))
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(out)
    part.rename(out)
    return result


def _record(
    result: Sliced, writer: _Out, topic: _Topic, log_time: int, publish: int, seq: int, data: bytes
) -> None:
    writer.write(topic, log_time, publish, seq, data)
    result.messages += 1
    result.bytes += len(data)
    result.per_topic[topic.name] = result.per_topic.get(topic.name, 0) + 1
    result.start_ns = log_time if result.start_ns is None else min(result.start_ns, log_time)
    result.end_ns = log_time if result.end_ns is None else max(result.end_ns, log_time)


def metadata_yaml(result: Sliced, topics: dict[str, _Topic], file_name: str) -> str:
    """The bag's ``metadata.yaml`` as rosbag2 Jazzy writes it (version 9): the QoS profiles and
    type hashes from each channel's own metadata, as ``ros2 bag record`` put them there."""
    start = result.start_ns or 0
    duration = (result.end_ns or start) - start
    entries = []
    for name, topic in topics.items():
        meta = dict(topic.metadata)
        qos = yaml.safe_load(meta.get("offered_qos_profiles", "") or "[]") or []
        entries.append(
            {
                "topic_metadata": {
                    "name": name,
                    "type": topic.schema_name,
                    "serialization_format": topic.message_encoding,
                    "offered_qos_profiles": qos,
                    "type_description_hash": meta.get("topic_type_hash", ""),
                },
                "message_count": result.per_topic.get(name, 0),
            }
        )
    info = {
        "version": ROSBAG2_VERSION,
        "storage_identifier": "mcap",
        "duration": {"nanoseconds": duration},
        "starting_time": {"nanoseconds_since_epoch": start},
        "message_count": result.messages,
        "topics_with_message_count": entries,
        "compression_format": "",
        "compression_mode": "",
        "relative_file_paths": [file_name],
        "files": [
            {
                "path": file_name,
                "starting_time": {"nanoseconds_since_epoch": start},
                "duration": {"nanoseconds": duration},
                "message_count": result.messages,
            }
        ],
        "custom_data": None,
        "ros_distro": ROS_DISTRO,
    }
    return str(
        yaml.safe_dump(
            {"rosbag2_bagfile_information": info}, sort_keys=False, default_flow_style=False
        )
    )
