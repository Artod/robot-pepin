"""The snapshot node under the ROS stubs: the three kinds of snapshot, and every field convention.

rclpy and rtabmap_msgs are faked (``ros_stubs``), TF is the stub's buffer, and the node is built
and driven here as on the laptop. Every expectation is a LITERAL — the topic's name, the format
number, the byte offsets of the cloud's fields, which image holds the depth — because the whole
value of this node is that it agrees with rtabmap_conversions, and a test that re-derived the
answer with the node's own expression would agree with the node instead.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import ros_stubs

ros_stubs.install()

from pepin_bringup.sensor_pack import (  # noqa: E402
    CAMERA_INFO_TOPIC,
    DEPTH_TOPIC,
    FLAGS,
    IMAGE_TOPIC,
    LASER_SCAN_FORMAT_XY,
    SCAN_TOPIC,
    SENSOR_DATA_TOPIC,
    STATE_TOPIC,
    TF_WAIT_S,
    XY_POINT_STEP,
    SensorPack,
    depth_is_a_factor,
    xy_cloud,
)

from pepin.snapshot import LIVE_PERIODS, PAIR_PERIODS, SnapshotState  # noqa: E402

# The board's clock, at an epoch second whose nanoseconds a double cannot hold: at 1.758e9 the
# step of a float is 238 ns, so a stamp that survives this round trip bit for bit was carried as a
# message and never through seconds as a number.
BOARD_SEC = 1_758_000_000
BOARD_NS = 123_456_789
WIDTH, HEIGHT = 8, 4  # a picture small enough to compare by hand
BEAMS = 8


def _stamp(offset_s: float) -> Any:
    """A board stamp ``offset_s`` after the reference moment, to the nanosecond."""
    total = BOARD_NS + round(offset_s * 1e9)
    return ros_stubs.Time(sec=BOARD_SEC + total // 1_000_000_000, nanosec=total % 1_000_000_000)


def _image(offset_s: float) -> Any:
    return ros_stubs.Image(
        header=ros_stubs.Header(stamp=_stamp(offset_s), frame_id="camera_optical"),
        height=HEIGHT,
        width=WIDTH,
        encoding="bgr8",
        step=WIDTH * 3,
        data=bytes(range(WIDTH * HEIGHT * 3 % 256)) + bytes(WIDTH * HEIGHT * 3),
    )


def _depth(offset_s: float) -> Any:
    return ros_stubs.Image(
        header=ros_stubs.Header(stamp=_stamp(offset_s), frame_id="camera_optical"),
        height=HEIGHT,
        width=WIDTH,
        encoding="32FC1",
        step=WIDTH * 4,
        data=struct.pack(f"<{WIDTH * HEIGHT}f", *[1.5] * (WIDTH * HEIGHT)),
    )


def _info() -> Any:
    return ros_stubs.CameraInfo(
        header=ros_stubs.Header(stamp=_stamp(0.0), frame_id="camera_optical"),
        height=HEIGHT,
        width=WIDTH,
        k=[4.0, 0.0, 4.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0],
    )


def _scan(offset_s: float, first_range: float = 1.0) -> Any:
    """Eight beams over a quarter turn, the first one at ``first_range`` so a test can tell one
    revolution from another, one of them infinite and one NaN so the dropping is visible."""
    ranges = [first_range, 2.0, float("inf"), 3.0, float("nan"), 4.0, 0.01, 5.0]
    return ros_stubs.LaserScan(
        header=ros_stubs.Header(stamp=_stamp(offset_s), frame_id="laser"),
        angle_min=0.0,
        angle_max=0.7,
        angle_increment=0.1,
        range_min=0.05,
        range_max=6.0,
        ranges=ranges,
    )


def _edge() -> Any:
    return ros_stubs.TransformStamped(
        header=ros_stubs.Header(stamp=_stamp(0.0)),
        transform=ros_stubs.Transform(
            translation=ros_stubs.Vector3(x=0.05, y=0.0, z=1.2),
            rotation=ros_stubs.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


Build = Callable[..., SensorPack]


@pytest.fixture
def build() -> Iterator[Build]:
    """A snapshot node over the stubs: both local transforms in TF, the optics delivered, and
    whatever flag overrides a launch would pass. Every node built is closed when the test ends."""
    made: list[SensorPack] = []

    def make(optics: bool = True, tf: bool = True, **params: Any) -> SensorPack:
        with ros_stubs.parameters(**params):
            node = SensorPack()
        made.append(node)
        if tf:
            node._tf.buffer.transforms[("base_link", "camera_optical")] = _edge()
            node._tf.buffer.transforms[("base_link", "laser")] = _edge()
        if optics:
            node.subs[CAMERA_INFO_TOPIC][1](_info())
        return node

    yield make
    for node in made:
        node.close()


def frame(node: SensorPack, offset_s: float) -> None:
    """One camera frame in: the picture and the depth that carries its stamp."""
    node.subs[IMAGE_TOPIC][1](_image(offset_s))
    node.subs[DEPTH_TOPIC][1](_depth(offset_s))


def scan(node: SensorPack, offset_s: float, first_range: float = 1.0) -> None:
    node.subs[SCAN_TOPIC][1](_scan(offset_s, first_range))


def sent(node: SensorPack) -> list[Any]:
    return node.pubs[SENSOR_DATA_TOPIC].sent


def _warm(node: SensorPack) -> None:
    """Both sources measured, with the camera's stamps lagging the lidar's the way they do on the
    wire. Ends with a full snapshot driven by the camera at 0.25 s, the nearest revolution 0.05 s
    before it; the two snapshots before it are the warm-up, where one source had one message and
    therefore no measured period yet."""
    scan(node, 0.00)
    scan(node, 0.10)
    frame(node, 0.05)
    frame(node, 0.15)
    scan(node, 0.20)
    scan(node, 0.32)
    frame(node, 0.25)


# ---- the topics and the three kinds ---------------------------------------------------------
def test_the_node_reads_four_topics_and_writes_one(build: Build) -> None:
    node = build()
    assert set(node.subs) == {"/camera/image", "/camera/depth", "/camera/camera_info", "/scan"}
    assert set(node.pubs) == {"/rtabmap/sensor_data", "/sensor_pack/state"}
    assert (IMAGE_TOPIC, DEPTH_TOPIC, CAMERA_INFO_TOPIC, SCAN_TOPIC) == (
        "/camera/image",
        "/camera/depth",
        "/camera/camera_info",
        "/scan",
    )
    assert SENSOR_DATA_TOPIC == "/rtabmap/sensor_data", (
        "rtabmap's own 'sensor_data' in its namespace"
    )
    assert STATE_TOPIC == "/sensor_pack/state", "what the snapshots carry, for rtabmap_frame"
    assert node.pubs[SENSOR_DATA_TOPIC].qos.reliability == "reliable"
    assert node.pubs[SENSOR_DATA_TOPIC].msg_type is ros_stubs.SensorData
    assert node.pubs[STATE_TOPIC].msg_type is ros_stubs.String
    assert node.pubs[STATE_TOPIC].qos.rest["durability"] == "transient_local", (
        "latched: a reader that starts later must not wait for the next change"
    )


# ---- what the snapshots carry ----------------------------------------------------------------
def state(node: SensorPack) -> SnapshotState:
    """The last state this node published on :data:`STATE_TOPIC`, parsed back."""
    heard = SnapshotState.from_json(node.pubs[STATE_TOPIC].sent[-1].data)
    assert heard is not None, "the state topic must carry a state"
    return heard


def test_the_state_says_which_sources_the_snapshots_carry(build: Build) -> None:
    """The one fact only this node can answer, and the one RTAB-Map's registration follows: a
    LIDAR-only stack says ``carrying lidar``, and its own refresh is the packer's liveness window
    (five of the source's measured periods) — the hold a reader must give a change."""
    node = build(pack_hz=15.0)
    scan(node, 0.00)
    scan(node, 0.20)
    scan(node, 0.40)
    carried = state(node)
    assert carried.carrying == ("lidar",) and carried.carries("lidar")
    assert not carried.carries("camera"), "no picture has arrived: the snapshots carry no camera"
    assert carried.kind == "lidar-only"
    assert carried.refresh_s == pytest.approx(0.2 * LIVE_PERIODS, abs=1e-6)
    assert SnapshotState.from_json(carried.to_json()) == carried, "it survives the wire"

    frame(node, 0.30)
    frame(node, 0.50)
    scan(node, 0.60)
    assert state(node).carrying == ("camera", "lidar"), "roster order, not arrival order"
    assert state(node).kind == "full"


def test_a_muted_source_leaves_the_state_at_once(build: Build) -> None:
    """``sources`` is the live A/B, so the state must follow it without waiting for a timeout: a
    camera-only roster carries the camera even while the lidar is still delivering."""
    node = build(pack_hz=15.0)
    _warm(node)
    assert state(node).carrying == ("camera", "lidar")
    node._switches.set("sources", ("camera",))
    frame(node, 0.35)
    assert state(node).carrying == ("camera",)
    assert state(node).kind == "camera-only"


def test_the_state_is_republished_every_second_whatever_happens(build: Build) -> None:
    """A reader must be able to tell a state that has stopped moving from one that has not
    changed, so the timer publishes whatever the packing path did."""
    node = build(pack_hz=15.0)
    _warm(node)
    sent_before = len(node.pubs[STATE_TOPIC].sent)
    periods = [period for period, _callback in node.timers]
    assert 1.0 in periods, "the state's own heartbeat"
    for period, callback in node.timers:
        if period == 1.0:
            callback()
    assert len(node.pubs[STATE_TOPIC].sent) == sent_before + 1


def test_both_sensors_make_a_full_snapshot_in_rtabmap_s_own_field_conventions(
    build: Build,
) -> None:
    """The one message that has to be right, checked field by field against what
    rtabmap_conversions::sensorDataFromROS reads: the picture in ``left``, the DEPTH in ``right``,
    ``right_camera_info`` empty so the message is not read as stereo, one camera info with one
    local transform beside it, and the scan as a two-field cloud declared as kXY."""
    node = build(pack_hz=15.0)
    _warm(node)
    msg = sent(node)[-1]

    assert msg.left.encoding == "bgr8" and msg.left.width == WIDTH
    assert msg.right.encoding == "32FC1", "right is the DEPTH for an RGB-D SensorData"
    assert msg.right_camera_info == [], "non-empty here would make rtabmap read it as stereo"
    assert len(msg.left_camera_info) == 1 and len(msg.local_transform) == 1
    assert msg.left_camera_info[0].k == [4.0, 0.0, 4.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0]
    assert msg.local_transform[0].translation.z == 1.2, "base_link <- camera_optical"
    assert msg.header.frame_id == "base_link"

    assert msg.laser_scan_format == 1, "rtabmap::LaserScan::kXY, asserted against the cloud"
    assert LASER_SCAN_FORMAT_XY == 1
    assert msg.laser_scan_max_pts == BEAMS, "the beam count the angles imply"
    assert msg.laser_scan_max_range == 6.0
    assert msg.laser_scan_local_transform.translation.z == 1.2, "base_link <- laser"
    cloud = msg.laser_scan
    assert [(f.name, f.offset, f.datatype, f.count) for f in cloud.fields] == [
        ("x", 0, 7, 1),
        ("y", 4, 7, 1),
    ], "x and y float32 and NO z: a z field would make rtabmap infer kXYZ and abort on 1"
    assert (cloud.point_step, cloud.height) == (8, 1) and XY_POINT_STEP == 8
    assert cloud.width == 5, "8 beams less the infinite, the NaN and the one under range_min"
    assert cloud.row_step == 40 and len(cloud.data) == 40
    assert cloud.header.frame_id == "laser", "the returns are in the LASER's frame"
    x0, y0 = struct.unpack_from("<ff", bytes(cloud.data), 0)
    assert (round(x0, 6), round(y0, 6)) == (1.0, 0.0), "the first beam at angle 0, range 1 m"


def test_the_stamp_is_the_board_s_own_and_never_this_laptop_s(build: Build) -> None:
    """The header's stamp is the driver member's own message stamp, to the nanosecond: RTAB-Map
    looks the odometry up at exactly this moment, and 123456789 ns does not survive a trip through
    seconds as a float (a double's step at this epoch second is 238 ns)."""
    node = build(pack_hz=15.0)
    node.clock.seconds = 42.0  # a laptop clock nowhere near the board's
    _warm(node)
    msg = sent(node)[-1]
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (BOARD_SEC, 373_456_789)
    assert msg.header.stamp == msg.left.header.stamp == msg.right.header.stamp
    # And the scan carries its OWN moment, so the pairing offset is in the message itself.
    assert msg.laser_scan.header.stamp == _stamp(0.20)
    assert (msg.laser_scan.header.stamp.sec, msg.laser_scan.header.stamp.nanosec) == (
        BOARD_SEC,
        323_456_789,
    )


def test_the_camera_alone_makes_a_camera_only_snapshot_with_no_restart(build: Build) -> None:
    """The lidar muted: within five of its own periods it is out of the choice and out of the
    snapshot, and the frames keep making nodes."""
    node = build(pack_hz=15.0)
    _warm(node)
    for k in range(6, 16):  # 0.60 s .. 1.50 s of camera, no scan at all
        frame(node, 0.1 * k)
    msg = sent(node)[-1]
    assert msg.left.encoding == "bgr8" and msg.right.encoding == "32FC1"
    assert msg.laser_scan.width == 0 and len(msg.laser_scan.data) == 0
    assert msg.laser_scan_max_pts == 0 and msg.laser_scan_format == 0
    node._report()
    line = node.logger.texts("info")[-1]
    assert "camera-only" in line and "silent lidar" in line
    assert "lidar silent" in line


def test_the_lidar_alone_makes_a_lidar_only_snapshot_with_no_restart(build: Build) -> None:
    """The camera muted — the failure that used to starve RTAB-Map altogether, because the
    synchroniser needed all three topics for one moment. Now the scan alone is a snapshot."""
    node = build(pack_hz=15.0)
    _warm(node)
    for k in range(4, 20):  # 0.40 s .. 1.90 s of scans, no frame at all
        scan(node, 0.1 * k)
    msg = sent(node)[-1]
    assert msg.left.width == 0 and msg.left.encoding == "", "no picture in this node"
    assert msg.right.width == 0
    assert msg.left_camera_info == [] and msg.local_transform == []
    assert msg.laser_scan.width == 5 and msg.laser_scan_format == 1
    assert msg.header.stamp == msg.laser_scan.header.stamp, "the scan drove it"
    node._report()
    line = node.logger.texts("info")[-1]
    assert "lidar-only" in line and "silent camera" in line


def test_nothing_is_published_while_neither_sensor_is_delivering(build: Build) -> None:
    """No snapshot at all rather than the last one for ever, and the report line asks the two
    questions an operator should then ask."""
    node = build()
    assert sent(node) == []
    node.subs[CAMERA_INFO_TOPIC][1](_info())  # optics alone are not a measurement of a moment
    assert sent(node) == []
    scan(node, 0.0)
    assert sent(node) == [], "one revolution is no measured period"
    node._report()
    line = node.logger.texts("warning")[-1]
    assert "no snapshot in this window" in line and "/scan" in line and "/camera/depth" in line


# ---- the pairing rule ------------------------------------------------------------------------
def test_the_scan_in_a_snapshot_is_the_revolution_nearest_its_moment(build: Build) -> None:
    """Not the newest revolution the lidar has delivered: the nearest to the moment the snapshot
    is stamped with, which is what keeps the two members inside half a period of each other."""
    node = build(pack_hz=15.0)
    scan(node, 0.00, first_range=1.0)
    scan(node, 0.10, first_range=2.0)
    frame(node, 0.05)
    frame(node, 0.15)
    scan(node, 0.20, first_range=3.0)
    scan(node, 0.32, first_range=4.0)
    frame(node, 0.25)
    msg = sent(node)[-1]
    x0, _y0 = struct.unpack_from("<ff", bytes(msg.laser_scan.data), 0)
    assert round(x0, 6) == 3.0, "the 0.20 s revolution, 0.05 s away, not the 0.32 s one"
    assert msg.laser_scan.header.stamp == _stamp(0.20)


def test_a_picture_with_no_depth_at_its_stamp_is_no_camera_member(build: Build) -> None:
    """Both raw images or neither: rtabmap only sets the RGB-D image when left AND right are
    there, so a picture whose depth never came is counted and dropped rather than sent alone."""
    node = build(pack_hz=15.0)
    for k in range(6):
        node.subs[IMAGE_TOPIC][1](_image(0.1 * k))  # pictures only: the network is behind
    scan(node, 0.60)
    scan(node, 0.70)
    msg = sent(node)[-1]
    assert msg.left.width == 0, "no camera member at all"
    assert msg.laser_scan.width == 5
    node._report()
    line = node.logger.texts("info")[-1]
    assert "6 images or depths with no partner at their stamp" in line


def test_the_optics_are_a_lens_not_a_moment_but_must_describe_the_picture(build: Build) -> None:
    """A CameraInfo is matched by SIZE and not by stamp — it describes the lens, and RTAB-Map
    reads only its matrices — and the size it must match is THE PICTURE's: rtabmap takes the model
    verbatim and asserts it against the RGB image (util3d.cpp:497), never against the depth."""
    node = build(optics=False, pack_hz=15.0)
    _warm(node)
    assert sent(node)[-1].left.width == 0, "no optics: no camera member"
    node._report()
    assert "frames whose camera_info does not describe the picture" in node.logger.texts("info")[-1]

    wrong = _info()
    wrong.width, wrong.height = WIDTH * 2, HEIGHT
    node.subs[CAMERA_INFO_TOPIC][1](wrong)
    for k in range(4, 8):
        frame(node, 0.1 * k)
        scan(node, 0.1 * k + 0.05)
    assert sent(node)[-1].left.width == 0, "a camera_info of another size describes another camera"

    node.subs[CAMERA_INFO_TOPIC][1](_info())  # stamped at 0.0, ages behind these frames
    for k in range(8, 12):
        frame(node, 0.1 * k)
        scan(node, 0.1 * k + 0.05)
    assert sent(node)[-1].left.width == WIDTH, "the right size is enough; the stamp is not read"


def test_a_depth_bigger_than_the_picture_is_refused_here_because_rtabmap_aborts_on_it() -> None:
    """RTAB-Map's RGB-D contract, which is a FATAL assertion on its side (Memory.cpp:4579,
    util3d.cpp:324): the depth is the picture's size divided by a whole number, or nothing."""
    picture = ros_stubs.Image(width=640, height=360)
    assert depth_is_a_factor(picture, ros_stubs.Image(width=640, height=360)), "equal is the case"
    assert depth_is_a_factor(picture, ros_stubs.Image(width=320, height=180)), "half is legal"
    assert depth_is_a_factor(picture, ros_stubs.Image(width=160, height=90)), "a quarter is legal"
    assert not depth_is_a_factor(picture, ros_stubs.Image(width=1280, height=720)), "never bigger"
    assert not depth_is_a_factor(picture, ros_stubs.Image(width=640, height=180)), (
        "one axis halved: rtabmap turns the ratio into ONE decimation"
    )
    assert not depth_is_a_factor(picture, ros_stubs.Image(width=300, height=180)), "not a factor"
    assert not depth_is_a_factor(picture, ros_stubs.Image(width=0, height=0)), "no image at all"


def test_a_depth_of_another_size_costs_the_camera_member_and_says_why(build: Build) -> None:
    node = build(pack_hz=15.0)

    def bad(offset_s: float) -> None:
        """A frame whose depth is bigger than its picture — what a scale change on camera_stream
        would produce, and what rtabmap aborts the whole process on."""
        node.subs[IMAGE_TOPIC][1](_image(offset_s))
        wide = _depth(offset_s)
        wide.width, wide.height = WIDTH * 2, HEIGHT * 2
        node.subs[DEPTH_TOPIC][1](wide)

    scan(node, 0.00)
    scan(node, 0.10)
    bad(0.05)
    bad(0.15)
    scan(node, 0.20)
    scan(node, 0.32)
    bad(0.25)
    assert sent(node)[-1].left.width == 0, "no camera member from a depth rtabmap would abort on"
    assert "ABORTS otherwise (Memory.cpp:4579)" in node.logger.texts("error")[-1]
    node._report()
    assert (
        "depths that are not the picture's size over a whole number"
        in node.logger.texts("info")[-1]
    )


# ---- the transform a member waits for --------------------------------------------------------
def test_tf_is_never_waited_for_on_this_thread(build: Build) -> None:
    """Every lookup here happens inside a subscription callback, so the timeout must be zero: a
    fifth of a second spent blocking is a fifth of a second in which no sensor is read."""
    node = build(pack_hz=15.0)
    _warm(node)
    assert TF_WAIT_S == 0.0
    timeouts = {call[3].nanoseconds for call in node._tf.buffer.calls}
    assert timeouts == {0}, "a blocking wait here stalls the executor"


def test_a_moment_whose_transform_is_late_is_put_back_and_packed_when_it_lands(
    build: Build,
) -> None:
    """The neck's edge arrives tens of milliseconds behind the frame it belongs to. The moment is
    not spent on a lidar-only snapshot: it goes back to the packer and is offered again, and when
    the edge lands the FULL snapshot goes out for that same moment."""
    node = build(tf=False, pack_hz=15.0)
    node._tf.buffer.transforms[("base_link", "laser")] = _edge()
    _warm(node)
    assert not any(msg.left.width for msg in sent(node)), "no camera member while the edge is out"
    node._report()
    assert "moments put back waiting for the camera's transform" in node.logger.texts("info")[-1]

    node._tf.buffer.transforms[("base_link", "camera_optical")] = _edge()
    scan(node, 0.34)  # any arrival re-offers the moment the packer was given back
    msg = sent(node)[-1]
    assert msg.left.width == WIDTH and msg.laser_scan.width == 5, "the full moment, not a remnant"


def test_a_moment_past_its_pairing_patience_goes_out_with_what_could_be_placed(
    build: Build,
) -> None:
    """The wait is BOUNDED by the member's own pairing patience, and past it the snapshot goes out
    with the scan alone — the behaviour of before the retry, which a dead neck must still get."""
    node = build(tf=False, pack_hz=15.0)
    node._tf.buffer.transforms[("base_link", "laser")] = _edge()
    _warm(node)
    patience = node._packer.cadence("camera").patience_s
    assert patience == pytest.approx(0.1 * PAIR_PERIODS), "1.5 of the camera's measured period"
    scan(node, 0.25 + patience + 0.01)  # the moment at 0.25 has now left its own patience
    msg = sent(node)[-1]
    assert msg.left.width == 0 and msg.laser_scan.width == 5
    node._report()
    line = node.logger.texts("info")[-1]
    assert "frames TF could not place" in line and "tf: " in line


def test_the_retry_is_a_flag_and_off_it_drops_the_member_at_once(build: Build) -> None:
    """CLAUDE.md rule 19: the old behaviour stays reachable without a restart."""
    node = build(tf=False, pack_hz=15.0, tf_retry=False)
    node._tf.buffer.transforms[("base_link", "laser")] = _edge()
    _warm(node)
    msg = sent(node)[-1]
    assert msg.left.width == 0 and msg.laser_scan.width == 5, "dropped at once, as before"
    node._report()
    assert "moments put back" not in node.logger.texts("info")[-1]


# ---- the flags -------------------------------------------------------------------------------
def test_the_flags_are_the_five_the_report_line_prints(build: Build) -> None:
    node = build()
    assert [flag.name for flag in FLAGS] == [
        "sensor_pack",
        "sources",
        "pack_hz",
        "pair_periods",
        "tf_retry",
    ]
    assert FLAGS["tf_retry"] is True
    assert FLAGS["sensor_pack"] is True
    assert FLAGS["sources"] == ("camera", "lidar")
    assert FLAGS["pack_hz"] == 1.0, "Rtabmap/DetectionRate in vslam.launch.py's table"
    assert FLAGS["pair_periods"] == PAIR_PERIODS == 1.5
    assert LIVE_PERIODS == 5.0, "liveness is not a flag: it is five of the source's own periods"
    node._report()
    assert (
        "flags: sensor_pack=on sources=camera,lidar pack_hz=1.0 pair_periods=1.5"
        in node.logger.texts("info")[-1]
    )


def test_camera_only_is_one_flag_and_no_longer_a_parameter_table(build: Build) -> None:
    """``camera_only:=true`` in the launch is ``sources:=camera`` here: the scan is still
    delivered and still measured, and simply enters no snapshot."""
    node = build(sources="camera", pack_hz=15.0)
    _warm(node)
    msg = sent(node)[-1]
    assert msg.left.width == WIDTH and msg.laser_scan.width == 0
    node._report()
    line = node.logger.texts("info")[-1]
    assert "camera-only" in line and "lidar off" in line
    assert "of 4 scans" in line, "the muted source is still counted as delivering"


def test_the_sources_flag_moves_live_and_an_empty_one_is_refused(build: Build) -> None:
    node = build(pack_hz=15.0)
    _warm(node)
    assert node.set_parameters([ros_stubs.Parameter("sources", value="lidar")])[0].successful
    for k in range(4, 8):
        frame(node, 0.1 * k)
        scan(node, 0.1 * k + 0.05)
    assert sent(node)[-1].left.width == 0, "the camera is out of the snapshots now"
    refused = node.set_parameters([ros_stubs.Parameter("sources", value="")])[0]
    assert not refused.successful and "at least one sensor" in refused.reason
    assert node._switches["sources"] == ("lidar",), "a refused change leaves the flag alone"


def test_the_switch_off_publishes_nothing_and_says_how_much_it_swallowed(build: Build) -> None:
    node = build(sensor_pack=False, pack_hz=15.0)
    _warm(node)
    assert sent(node) == []
    node._report()
    line = node.logger.texts("info")[-1]
    assert "arrivals with the switch off" in line and "sensor_pack=off" in line


# ---- the cloud on its own --------------------------------------------------------------------
def test_the_cloud_drops_what_the_old_path_s_projection_dropped() -> None:
    """laser_geometry kept the returns inside [range_min, range_max] and finite; so does this, and
    the bytes are x then y, little-endian, eight to a point."""
    cloud = xy_cloud(_scan(0.0))
    assert cloud.width == 5, "the infinite beam, the NaN one and the 0.01 m under range_min"
    assert cloud.is_dense is True and cloud.is_bigendian is False
    values = struct.unpack(f"<{cloud.width * 2}f", bytes(cloud.data))
    assert [round(v, 4) for v in values[:4]] == [1.0, 0.0, 1.99, 0.1997]


def test_a_scan_that_describes_no_revolution_is_refused_with_rtabmap_s_own_reason() -> None:
    """rtabmap refuses a LaserScan whose angle_increment is 0 or whose range_min is above its
    range_max (MsgConversion.cpp:2602-2613). This refuses it one hop earlier, so every return does
    not land on one ray in silence."""
    node = None
    with ros_stubs.parameters(pack_hz=15.0):
        node = SensorPack()
    try:
        node._tf.buffer.transforms[("base_link", "laser")] = _edge()
        broken = _scan(0.0)
        broken.angle_increment = 0.0
        node.subs[SCAN_TOPIC][1](broken)
        node.subs[SCAN_TOPIC][1](_scan(0.1))
        assert sent(node) == [], "the first scan was refused, so the second is the only one there"
        node._report()
        line = node.logger.texts("info")[-1]
        assert "1 scans describing no revolution" in line
        assert "angle_increment 0.0" in node.logger.texts("error")[-1]
    finally:
        node.close()
