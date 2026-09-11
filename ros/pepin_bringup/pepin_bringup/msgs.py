"""ROS messages in and out of arrays: the conversions the nodes used to carry each for itself.

Pure functions over the message objects (``sensor_msgs``, ``geometry_msgs``), no node and no
clock: a stamp is seconds or a ``builtin_interfaces/Time``, a picture is a numpy array, a
placement is a :class:`pepin.tsdf.RigidPose` or a :class:`pepin.mounts.Mount`. What is packed
here is what the other side unpacks — an ``Image`` of ``32FC1`` metres, a ``LaserScan`` with
NaN where a bearing saw nothing, a ``PointCloud2`` in the layout Foxglove colours — so the two
ends of a topic agree by construction, and a test can check both against tiny fakes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from sensor_msgs.msg import Image, LaserScan, PointCloud2, PointField
from std_msgs.msg import Header

from pepin.camera import quaternion_from_rpy
from pepin.depth import decode_rgb, quaternion_from_matrix, rotation_matrix
from pepin.mounts import Mount
from pepin.tsdf import RigidPose

Array = npt.NDArray[np.float64]

# sensor_msgs/Image encodings the nodes speak, with the numpy dtype and channel count of each.
ENCODINGS: dict[str, tuple[type[np.generic], int]] = {
    "32FC1": (np.float32, 1),
    "16UC1": (np.uint16, 1),
    "mono8": (np.uint8, 1),
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
}
UPSIDE_DOWN_TOLERANCE = 0.2  # radians of roll away from pi that still count as hanging


# ---- time and headers ----------------------------------------------------------------------
def stamp_seconds(stamp: Any) -> float:
    """A ``builtin_interfaces/Time`` as seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def stamp_from_seconds(seconds: float) -> Any:
    """Seconds as a ``builtin_interfaces/Time``, rounded to the nanosecond. The whole seconds
    come off first: an epoch stamp (1.8e9 s) times 1e9 is a float step of 238 ns, and the
    fractional part alone keeps every nanosecond the double holds."""
    whole = math.floor(seconds)
    sec, nanosec = divmod(whole * 1_000_000_000 + round((seconds - whole) * 1e9), 1_000_000_000)
    return TimeMsg(sec=int(sec), nanosec=int(nanosec))


def header(stamp: Any, frame_id: str) -> Any:
    """A ``std_msgs/Header`` from a stamp message and a frame."""
    return Header(stamp=stamp, frame_id=frame_id)


# ---- rotations -----------------------------------------------------------------------------
def yaw_of(q: Any) -> float:
    """Yaw of a ``geometry_msgs/Quaternion`` (a planar robot)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def rpy_of(q: Any) -> tuple[float, float, float]:
    """Roll, pitch, yaw (radians) of a ``geometry_msgs/Quaternion``: the fixed-axis angles
    :func:`pepin.camera.quaternion_from_rpy` composes."""
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    return roll, pitch, yaw_of(q)


def _set_quaternion(q: Any, xyzw: tuple[float, float, float, float]) -> None:
    q.x, q.y, q.z, q.w = (float(v) for v in xyzw)


def _set_vector(v: Any, xyz: Any) -> None:
    v.x, v.y, v.z = (float(c) for c in xyz)


# ---- transforms ----------------------------------------------------------------------------
def transform_from_rpy(
    parent: str,
    child: str,
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
    stamp: Any,
) -> Any:
    """``parent -> child`` as a ``geometry_msgs/TransformStamped`` from metres and radians."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id, t.child_frame_id = parent, child
    _set_vector(t.transform.translation, xyz)
    _set_quaternion(t.transform.rotation, quaternion_from_rpy(*rpy))
    return t


def transform_from_mount(parent: str, child: str, mount: Mount, stamp: Any) -> Any:
    """A static ``parent -> child`` from a :class:`pepin.mounts.Mount`."""
    x, y, z, roll, pitch, yaw = mount.transform()
    return transform_from_rpy(parent, child, (x, y, z), (roll, pitch, yaw), stamp)


def transform_from_pose(parent: str, child: str, pose: RigidPose, stamp: Any) -> Any:
    """``parent -> child`` from a rotation matrix and a translation (``parent <- child``)."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id, t.child_frame_id = parent, child
    _set_vector(t.transform.translation, pose.translation)
    _set_quaternion(t.transform.rotation, quaternion_from_matrix(pose.rotation))
    return t


def pose_from_transform(msg: Any) -> RigidPose:
    """A ``TransformStamped`` (or a bare ``Transform``) as rotation matrix and translation."""
    transform = getattr(msg, "transform", msg)
    q, v = transform.rotation, transform.translation
    return RigidPose(rotation_matrix(q.x, q.y, q.z, q.w), np.array([v.x, v.y, v.z]))


def rpy_from_transform(msg: Any) -> tuple[float, float, float, float, float, float]:
    """A ``TransformStamped`` (or ``Transform``) as ``(x, y, z, roll, pitch, yaw)``."""
    transform = getattr(msg, "transform", msg)
    v = transform.translation
    return (v.x, v.y, v.z, *rpy_of(transform.rotation))


def planar_mount(msg: Any) -> tuple[float, float, float, bool]:
    """A static sensor frame for the planar stack (``pepin.timeline``): ``(x, y, yaw,
    upside_down)``, upside down when the roll is within :data:`UPSIDE_DOWN_TOLERANCE` of pi."""
    x, y, _z, roll, _pitch, yaw = rpy_from_transform(msg)
    return x, y, yaw, abs(abs(roll) - math.pi) < UPSIDE_DOWN_TOLERANCE


# ---- poses ---------------------------------------------------------------------------------
def pose_with_covariance(
    x: float,
    y: float,
    yaw: float,
    sigma_xy_m: float,
    sigma_yaw_rad: float,
    stamp: Any,
    frame_id: str,
) -> Any:
    """A planar pose as ``geometry_msgs/PoseWithCovarianceStamped``: isotropic position
    sigma, a yaw sigma, everything else zero."""
    msg = PoseWithCovarianceStamped()
    msg.header.stamp, msg.header.frame_id = stamp, frame_id
    msg.pose.pose.position.x, msg.pose.pose.position.y = float(x), float(y)
    _set_quaternion(msg.pose.pose.orientation, (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)))
    cov = [0.0] * 36
    cov[0] = cov[7] = sigma_xy_m**2
    cov[35] = sigma_yaw_rad**2
    msg.pose.covariance = cov
    return msg


# ---- images --------------------------------------------------------------------------------
def image_from_array(array: Any, encoding: str, stamp: Any, frame_id: str) -> Any:
    """A numpy picture as ``sensor_msgs/Image``: ``32FC1`` metres from a float image,
    ``bgr8``/``rgb8`` from an (h, w, 3) byte image, ``mono8`` / ``16UC1`` from one channel."""
    dtype, channels = ENCODINGS[encoding]
    source = np.asarray(array)  # the shape is checked before the cast: casting NaN to a byte warns
    if source.ndim != (3 if channels > 1 else 2) or (channels > 1 and source.shape[2] != channels):
        raise ValueError(f"{encoding} wants {channels} channel(s), not an array of {source.shape}")
    pixels = np.ascontiguousarray(source, dtype=dtype)
    msg = Image()
    msg.header.stamp, msg.header.frame_id = stamp, frame_id
    msg.height, msg.width = int(pixels.shape[0]), int(pixels.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = msg.width * channels * pixels.dtype.itemsize
    msg.data = pixels.tobytes()
    return msg


def array_from_image(msg: Any) -> Any | None:
    """A ``sensor_msgs/Image`` as numpy: ``32FC1`` -> (h, w) float32, ``16UC1`` / ``mono8`` ->
    (h, w) of the pixel type, ``bgr8`` / ``rgb8`` -> (h, w, 3) uint8 RGB; ``None`` for an
    encoding not spoken here."""
    if msg.encoding in ("bgr8", "rgb8"):
        return decode_rgb(bytes(msg.data), msg.height, msg.width, msg.encoding)
    if msg.encoding not in ENCODINGS:
        return None
    dtype, _channels = ENCODINGS[msg.encoding]
    return np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, msg.width)


# ---- scans ---------------------------------------------------------------------------------
def scan_from_ranges(
    ranges: Any,
    angle_min: float,
    angle_increment: float,
    stamp: Any,
    frame_id: str,
    range_min: float,
    range_max: float,
) -> Any:
    """Ranges by bearing as a ``sensor_msgs/LaserScan``; NaN stays NaN (a bearing that saw
    nothing: a costmap neither marks nor clears it), +inf stays +inf (nothing within range)."""
    r = np.asarray(ranges, dtype=np.float64)
    scan = LaserScan()
    scan.header.stamp, scan.header.frame_id = stamp, frame_id
    scan.angle_min = float(angle_min)
    scan.angle_max = float(angle_min + angle_increment * max(r.size - 1, 0))
    scan.angle_increment = float(angle_increment)
    scan.range_min, scan.range_max = float(range_min), float(range_max)
    scan.ranges = [float(v) for v in r]
    return scan


def scan_arrays(msg: Any) -> tuple[Array, Array]:
    """A ``LaserScan`` as ``(angles, ranges)`` in radians and metres, one entry per bin;
    ranges that are not a return — not finite, under ``range_min``, over ``range_max`` — are
    NaN, the convention of :class:`pepin.lidar.LaserScan`."""
    r = np.asarray(msg.ranges, dtype=np.float64)
    angles = msg.angle_min + msg.angle_increment * np.arange(r.size)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
    return angles, np.where(valid, r, np.nan)


# ---- point clouds --------------------------------------------------------------------------
def cloud_from_points(points: Any, colours: Any | None, stamp: Any, frame_id: str) -> Any:
    """(n, 3) metres as a ``sensor_msgs/PointCloud2``: ``x y z`` float32 (12-byte points), or
    with (n, 3) uint8 RGB the 32-byte PCL layout (``rgb`` packed into a float at offset 16)
    that Foxglove and RViz colour."""
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    if colours is None:
        data = np.ascontiguousarray(xyz).tobytes()
        step = 12
    else:
        rgb = np.asarray(colours, dtype=np.uint32).reshape(-1, 3)
        cloud = np.zeros(
            xyz.shape[0],
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("pad", "<f4"),
                ("rgb", "<f4"),
                ("pad2", "<f4", 3),
            ],
        )
        cloud["x"], cloud["y"], cloud["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        cloud["rgb"] = ((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]).view(np.float32)
        fields.append(PointField(name="rgb", offset=16, datatype=PointField.FLOAT32, count=1))
        data = cloud.tobytes()
        step = 32
    msg = PointCloud2()
    msg.header.stamp, msg.header.frame_id = stamp, frame_id
    msg.height, msg.width = 1, int(xyz.shape[0])
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step, msg.row_step = step, step * int(xyz.shape[0])
    msg.is_dense = True
    msg.data = data
    return msg


# ---- IMU -----------------------------------------------------------------------------------
def imu_arrays(msg: Any) -> tuple[Array, Array]:
    """A ``sensor_msgs/Imu`` as ``(acceleration, angular_velocity)``, m/s^2 and rad/s."""
    a, w = msg.linear_acceleration, msg.angular_velocity
    return np.array([a.x, a.y, a.z], dtype=np.float64), np.array([w.x, w.y, w.z], dtype=np.float64)
