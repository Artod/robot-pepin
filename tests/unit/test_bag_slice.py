"""pepin.bag_slice on tiny synthetic MCAP files shaped like the ring's: rosbag2's ros2msg schemas,
cdr channels with their QoS metadata, closed files with a summary and a newest one torn mid-chunk
with no footer. The cut's window, its carried latched messages, and the metadata.yaml rosbag2
reads are checked by reading the cut back with the same library."""

from __future__ import annotations

from pathlib import Path

import yaml
from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer

from pepin.bag_slice import Carry, cut, is_open, read_messages

S = 1_000_000_000
TF_QOS = "- history: keep_last\n  depth: 100\n  durability: transient_local\n"
IMU_QOS = "- history: keep_last\n  depth: 10\n  durability: volatile\n"


def ring_file(path: Path, messages: list[tuple[str, int, bytes]], torn: bool = False) -> Path:
    """One ring file of ``(topic, log time s*1e9, data)``; ``torn`` cuts the footer and the
    second half of its last chunks off, as a file still being written looks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        writer = Writer(handle, chunk_size=200, compression=CompressionType.NONE)
        writer.start(profile="ros2", library="test")
        imu = writer.register_schema("sensor_msgs/msg/Imu", "ros2msg", b"float64 x")
        tf = writer.register_schema("tf2_msgs/msg/TFMessage", "ros2msg", b"string frame")
        channels = {
            "/imu": writer.register_channel(
                "/imu", "cdr", imu, {"offered_qos_profiles": IMU_QOS, "topic_type_hash": "H1"}
            ),
            "/tf_static": writer.register_channel(
                "/tf_static", "cdr", tf, {"offered_qos_profiles": TF_QOS, "topic_type_hash": "H2"}
            ),
            "/costmap": writer.register_channel(
                "/costmap", "cdr", imu, {"offered_qos_profiles": TF_QOS, "topic_type_hash": "H3"}
            ),
        }
        for seq, (topic, t, data) in enumerate(messages):
            writer.add_message(channels[topic], t, data, t, seq)
        writer.finish()  # type: ignore[no-untyped-call]
    if torn:
        whole = path.read_bytes()
        path.write_bytes(whole[: int(len(whole) * 0.55)])
    return path


def imu(t0: int, t1: int) -> list[tuple[str, int, bytes]]:
    """/imu at 10 Hz from t0 to t1 seconds (exclusive), its data the tenth of a second."""
    return [("/imu", t * S // 10, f"{t}".encode()) for t in range(t0 * 10, t1 * 10)]


def test_a_window_is_cut_across_closed_files_and_the_torn_one(tmp_path: Path) -> None:
    ring = tmp_path / "ring" / "r"
    a = ring_file(ring / "r_0.mcap", [("/tf_static", 0, b"statics"), *imu(0, 10)])
    b = ring_file(ring / "r_1.mcap", [("/costmap", 12 * S, b"grid12"), *imu(10, 20)])
    c = ring_file(ring / "r_2.mcap", imu(20, 30), torn=True)
    assert not is_open(a) and not is_open(b) and is_open(c)
    readable = [m.log_time for m in read_messages(c)]
    assert readable and readable[0] == 20 * S and readable[-1] < 29 * S  # whole chunks only

    out = tmp_path / "rec" / "0001_20261005_200000Z_home"
    sliced = cut(
        [a, b, c],
        15 * S,
        22 * S,
        out,
        carry=[Carry("/tf_static", b"statics")],
        carry_last=("/costmap",),
    )
    assert sorted(p.name for p in out.iterdir()) == [f"{out.name}_0.mcap", "metadata.yaml"]
    assert not out.with_name(out.name + ".part").exists()
    with (out / f"{out.name}_0.mcap").open("rb") as handle:
        got = [(ch.topic, m.log_time, m.data) for _, ch, m in make_reader(handle).iter_messages()]
    # the carried statics and the last grid before the window, at the window's start
    assert ("/tf_static", 15 * S, b"statics") in got
    assert ("/costmap", 15 * S, b"grid12") in got
    imu_times = [t for topic, t, _ in got if topic == "/imu"]
    assert imu_times[0] == 15 * S and imu_times[-1] == 22 * S
    assert len(imu_times) == 71  # 15.0 .. 22.0 at 10 Hz, both ends kept
    assert sliced.first_ns == 15 * S and sliced.files == 3 and sliced.open_files == 1
    assert sliced.newest_ns >= 22 * S
    assert sliced.per_topic == {"/tf_static": 1, "/costmap": 1, "/imu": 71}


def test_the_metadata_is_rosbag2s_version_9_with_each_channels_own_qos(tmp_path: Path) -> None:
    a = ring_file(tmp_path / "ring/r/r_0.mcap", [("/tf_static", 0, b"s"), *imu(0, 5)])
    out = tmp_path / "rec" / "0002_x_home"
    cut([a], 1 * S, 3 * S, out, carry=[Carry("/tf_static", b"s")])
    info = yaml.safe_load((out / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    assert info["version"] == 9 and info["storage_identifier"] == "mcap"
    assert info["relative_file_paths"] == ["0002_x_home_0.mcap"]
    assert info["starting_time"]["nanoseconds_since_epoch"] == 1 * S
    assert info["duration"]["nanoseconds"] == 2 * S
    assert info["message_count"] == 22 and info["files"][0]["message_count"] == 22
    topics = {t["topic_metadata"]["name"]: t for t in info["topics_with_message_count"]}
    assert set(topics) == {"/tf_static", "/imu"}, "only what the bag holds"
    tf = topics["/tf_static"]["topic_metadata"]
    assert tf["type"] == "tf2_msgs/msg/TFMessage" and tf["serialization_format"] == "cdr"
    assert tf["offered_qos_profiles"][0]["durability"] == "transient_local"
    assert tf["type_description_hash"] == "H2"
    assert info["ros_distro"] == "jazzy"


def test_a_window_the_ring_does_not_reach_yet_is_cut_short_and_says_so(tmp_path: Path) -> None:
    a = ring_file(tmp_path / "ring/r/r_0.mcap", imu(0, 5))
    sliced = cut([a], 3 * S, 9 * S, tmp_path / "rec" / "0003_x_home")
    assert sliced.newest_ns < 9 * S
    assert sliced.end_ns is not None and sliced.end_ns < 5 * S


def test_a_cut_replaces_a_stale_one_whole(tmp_path: Path) -> None:
    a = ring_file(tmp_path / "ring/r/r_0.mcap", imu(0, 5))
    out = tmp_path / "rec" / "0004_x_home"
    out.mkdir(parents=True)
    (out / "junk").write_text("from before")
    cut([a], 1 * S, 2 * S, out)
    assert not (out / "junk").exists()
