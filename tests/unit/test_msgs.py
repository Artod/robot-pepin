"""The message codecs of the ROS nodes, round-tripped over fakes of the message classes."""

from __future__ import annotations

import math

import numpy as np
import pytest
import ros_stubs

ros_stubs.install()

from pepin_bringup import msgs  # noqa: E402

from pepin.mounts import Mount  # noqa: E402
from pepin.tsdf import RigidPose  # noqa: E402


def test_a_stamp_goes_to_seconds_and_back_to_the_nanosecond() -> None:
    """An epoch stamp keeps every nanosecond the double holds: scaling the whole value by 1e9
    would round it to the nearest 238 ns and the pairing of a depth frame with its picture,
    which is by exact stamp, would miss."""
    stamp = msgs.stamp_from_seconds(1789101071.9502695)
    assert (stamp.sec, stamp.nanosec) == (1789101071, 950269461)
    assert msgs.stamp_seconds(stamp) == 1789101071.9502695
    assert msgs.stamp_seconds(ros_stubs.Time(sec=2, nanosec=500_000_000)) == 2.5
    zero = msgs.stamp_from_seconds(0.9999999999)
    assert (zero.sec, zero.nanosec) == (1, 0)  # rounded, never truncated into a stray nanosecond
    assert msgs.header(stamp, "map").frame_id == "map"


def test_a_quaternion_s_angles_come_back_out() -> None:
    """A transform built from roll/pitch/yaw reads back as the same angles, and the yaw alone
    for the planar readers."""
    t = msgs.transform_from_rpy("a", "b", (1.0, 2.0, 3.0), (0.3, -0.2, 1.1), ros_stubs.Time())
    assert (t.header.frame_id, t.child_frame_id) == ("a", "b")
    x, y, z, roll, pitch, yaw = msgs.rpy_from_transform(t)
    assert (x, y, z) == (1.0, 2.0, 3.0)
    assert (roll, pitch, yaw) == pytest.approx((0.3, -0.2, 1.1))
    assert msgs.yaw_of(t.transform.rotation) == pytest.approx(1.1)
    upright = msgs.transform_from_rpy("a", "b", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None)
    assert msgs.rpy_of(upright.transform.rotation) == (0.0, 0.0, 0.0)


def test_the_laser_mount_reads_as_planar_and_upside_down() -> None:
    """The LD19 hangs (roll pi): the planar stack gets (x, y, yaw, mirrored) from the static
    transform, the way the tracker has read it since it stopped hard-coding the mount."""
    laser = Mount(x_m=0.005, z_m=0.2, roll_deg=180.0, yaw_deg=-87.5)
    t = msgs.transform_from_mount("base_link", "laser", laser, ros_stubs.Time())
    x, y, yaw, mirrored = msgs.planar_mount(t)
    assert (x, y) == (0.005, 0.0) and math.degrees(yaw) == pytest.approx(-87.5) and mirrored
    level = msgs.transform_from_mount("base_link", "laser", Mount(yaw_deg=10.0), None)
    assert msgs.planar_mount(level)[3] is False


def test_a_rigid_pose_survives_the_trip_through_a_transform() -> None:
    c, s = math.cos(0.7), math.sin(0.7)
    pose = RigidPose(
        np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.array([1.0, -2.0, 0.5])
    )
    t = msgs.transform_from_pose("map", "rtabmap", pose, ros_stubs.Time(sec=5))
    assert t.header.stamp.sec == 5 and (t.header.frame_id, t.child_frame_id) == ("map", "rtabmap")
    back = msgs.pose_from_transform(t)
    assert np.allclose(back.rotation, pose.rotation) and np.allclose(
        back.translation, pose.translation
    )
    bare = msgs.pose_from_transform(t.transform)  # a Transform without its header reads the same
    assert np.allclose(bare.rotation, pose.rotation)


def test_a_planar_pose_carries_its_sigmas_in_the_covariance() -> None:
    msg = msgs.pose_with_covariance(1.0, 2.0, math.pi / 2, 0.05, math.radians(5.0), None, "map")
    assert (msg.pose.pose.position.x, msg.pose.pose.position.y) == (1.0, 2.0)
    assert msgs.yaw_of(msg.pose.pose.orientation) == pytest.approx(math.pi / 2)
    cov = msg.pose.covariance
    assert cov[0] == cov[7] == pytest.approx(0.0025) and cov[35] == pytest.approx(
        math.radians(5.0) ** 2
    )
    assert sum(cov) == pytest.approx(cov[0] + cov[7] + cov[35]) and msg.header.frame_id == "map"


def test_a_depth_image_is_float_metres_and_a_picture_is_bytes() -> None:
    depth = np.array([[1.5, np.nan], [0.25, 4.0]], dtype=np.float32)
    msg = msgs.image_from_array(depth, "32FC1", ros_stubs.Time(sec=1), "camera_optical")
    assert (msg.height, msg.width, msg.encoding, msg.step) == (2, 2, "32FC1", 8)
    assert msg.header.frame_id == "camera_optical" and len(msg.data) == 16
    back = msgs.array_from_image(msg)
    assert back is not None and back.dtype == np.float32 and back.shape == (2, 2)
    assert back[0, 0] == 1.5 and np.isnan(back[0, 1]) and back[1, 1] == 4.0
    bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    picture = msgs.image_from_array(bgr, "bgr8", None, "camera_optical")
    assert (picture.step, picture.height, picture.width) == (6, 1, 2)
    rgb = msgs.array_from_image(picture)
    assert rgb is not None and rgb.tolist() == [[[3, 2, 1], [6, 5, 4]]]  # decoded as RGB
    mono = msgs.array_from_image(
        msgs.image_from_array(np.array([[7, 8]], dtype=np.uint8), "mono8", None, "")
    )
    assert mono is not None and mono.tolist() == [[7, 8]]
    picture.encoding = "yuv422"
    assert msgs.array_from_image(picture) is None
    with pytest.raises(ValueError, match="bgr8 wants 3"):
        msgs.image_from_array(depth, "bgr8", None, "")


def test_a_scan_keeps_its_nans_and_reads_back_as_bearings() -> None:
    """NaN is 'saw nothing' (a costmap neither marks nor clears), inf is 'nothing in range';
    reading a scan back, both and anything outside the limits are NaN, like pepin.lidar."""
    ranges = [1.0, math.nan, math.inf, 0.02, 7.0]
    scan = msgs.scan_from_ranges(ranges, -0.5, 0.25, ros_stubs.Time(), "base_link", 0.1, 6.0)
    assert scan.angle_min == -0.5 and scan.angle_max == pytest.approx(0.5)
    assert scan.ranges[0] == 1.0 and math.isnan(scan.ranges[1]) and scan.ranges[2] == math.inf
    assert (scan.range_min, scan.range_max, scan.header.frame_id) == (0.1, 6.0, "base_link")
    angles, back = msgs.scan_arrays(scan)
    assert angles.tolist() == pytest.approx([-0.5, -0.25, 0.0, 0.25, 0.5])
    assert back[0] == 1.0 and np.isnan(back[1:]).all()
    empty = msgs.scan_from_ranges([], 0.0, 0.1, None, "x", 0.0, 1.0)
    assert empty.angle_max == 0.0 and empty.ranges == []


def test_a_cloud_is_packed_the_way_foxglove_colours_it() -> None:
    """With colours: the 32-byte PCL layout, rgb packed into a float at offset 16; without:
    plain xyz, 12 bytes a point (what create_cloud_xyz32 made)."""
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    colours = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    cloud = msgs.cloud_from_points(points, colours, ros_stubs.Time(sec=3), "map")
    assert (cloud.height, cloud.width, cloud.point_step, cloud.row_step) == (1, 2, 32, 64)
    assert [(f.name, f.offset) for f in cloud.fields] == [("x", 0), ("y", 4), ("z", 8), ("rgb", 16)]
    assert cloud.fields[3].datatype == ros_stubs.PointField.FLOAT32 and cloud.is_dense
    raw = np.frombuffer(cloud.data, dtype=np.float32).reshape(2, 8)
    assert raw[:, :3].tolist() == points.tolist()
    packed = raw[:, 4].view(np.uint32)
    assert packed.tolist() == [0xFF0000, 0x0000FF]
    plain = msgs.cloud_from_points(points, None, None, "map")
    assert plain.point_step == 12 and len(plain.data) == 24 and len(plain.fields) == 3
    assert np.frombuffer(plain.data, dtype=np.float32).reshape(2, 3).tolist() == points.tolist()
    none = msgs.cloud_from_points(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), None, "map")
    assert none.width == 0 and none.data == b""


def test_an_imu_reading_becomes_two_vectors() -> None:
    imu = ros_stubs.Imu(
        linear_acceleration=ros_stubs.Vector3(x=0.1, y=9.8, z=0.2),
        angular_velocity=ros_stubs.Vector3(z=0.5),
    )
    accel, gyro = msgs.imu_arrays(imu)
    assert accel.tolist() == [0.1, 9.8, 0.2] and gyro.tolist() == [0.0, 0.0, 0.5]
