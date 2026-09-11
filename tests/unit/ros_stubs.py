"""Just enough of ROS for ``pepin_bringup.node_kit`` and ``pepin_bringup.msgs`` to import here.

rclpy is not installed on the laptop, so the ROS package cannot be imported by the unit tests —
except its two pure modules, whose ROS imports are message classes (fields with defaults) and a
handful of rclpy names. :func:`install` puts fakes of exactly those into ``sys.modules`` (once,
idempotent) and puts ``ros/pepin_bringup`` on the path, the way ``test_ros_bridge_protocol``
reaches ``pepin_bringup.protocol``. The fakes hold the fields the real messages have and nothing
else: a codec that wrote a field the message lacks would fail here as it would on the robot.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, ClassVar

REPO = Path(__file__).resolve().parents[2]


class Msg:
    """A message: keyword fields over the class defaults, like rosidl's constructors."""

    _fields: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs: Any) -> None:
        for name, default in self._fields.items():
            setattr(self, name, default() if callable(default) else default)
        for name, value in kwargs.items():
            if name not in self._fields:
                raise AttributeError(f"{type(self).__name__} has no field {name}")
            setattr(self, name, value)

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and vars(self) == vars(other)


def _msg(_name: str, /, **fields: Any) -> type[Msg]:
    """A message class with these fields; the name is positional so a field may be called
    ``name`` (``PointField`` has one)."""
    return type(_name, (Msg,), {"_fields": fields})


Time = _msg("Time", sec=0, nanosec=0)
Header = _msg("Header", stamp=Time, frame_id="")
Vector3 = _msg("Vector3", x=0.0, y=0.0, z=0.0)
Point = _msg("Point", x=0.0, y=0.0, z=0.0)
Quaternion = _msg("Quaternion", x=0.0, y=0.0, z=0.0, w=1.0)
Transform = _msg("Transform", translation=Vector3, rotation=Quaternion)
TransformStamped = _msg("TransformStamped", header=Header, child_frame_id="", transform=Transform)
Pose = _msg("Pose", position=Point, orientation=Quaternion)
PoseWithCovariance = _msg("PoseWithCovariance", pose=Pose, covariance=lambda: [0.0] * 36)
PoseWithCovarianceStamped = _msg(
    "PoseWithCovarianceStamped", header=Header, pose=PoseWithCovariance
)
Image = _msg(
    "Image", header=Header, height=0, width=0, encoding="", is_bigendian=0, step=0, data=b""
)
LaserScan = _msg(
    "LaserScan",
    header=Header,
    angle_min=0.0,
    angle_max=0.0,
    angle_increment=0.0,
    time_increment=0.0,
    scan_time=0.0,
    range_min=0.0,
    range_max=0.0,
    ranges=list,
    intensities=list,
)
PointField = _msg("PointField", name="", offset=0, datatype=0, count=0)
for _name, _value in (("INT8", 1), ("UINT8", 2), ("UINT32", 6), ("FLOAT32", 7), ("FLOAT64", 8)):
    setattr(PointField, _name, _value)
PointCloud2 = _msg(
    "PointCloud2",
    header=Header,
    height=0,
    width=0,
    fields=list,
    is_bigendian=False,
    point_step=0,
    row_step=0,
    data=b"",
    is_dense=False,
)
Imu = _msg(
    "Imu",
    header=Header,
    orientation=Quaternion,
    angular_velocity=Vector3,
    linear_acceleration=Vector3,
)
SetParametersResult = _msg("SetParametersResult", successful=False, reason="")
IntegerRange = _msg("IntegerRange", from_value=0, to_value=0, step=0)
FloatingPointRange = _msg("FloatingPointRange", from_value=0.0, to_value=0.0, step=0.0)
ParameterDescriptor = _msg(
    "ParameterDescriptor",
    name="",
    type=0,
    description="",
    additional_constraints="",
    read_only=False,
    dynamic_typing=False,
    floating_point_range=list,
    integer_range=list,
)


class ParameterType:
    """rcl_interfaces' constants: the wire types a descriptor names."""

    PARAMETER_NOT_SET = 0
    PARAMETER_BOOL = 1
    PARAMETER_INTEGER = 2
    PARAMETER_DOUBLE = 3
    PARAMETER_STRING = 4


class Parameter:
    """rclpy.parameter.Parameter: a name and a value (the type inferred, as rclpy does)."""

    def __init__(self, name: str, type_: Any = None, value: Any = None) -> None:
        self.name, self.type_, self.value = name, type_, value


class ExternalShutdownException(Exception):  # noqa: N818 — rclpy's own name
    """rclpy's: the context was shut down under the spin."""


class RclpyTime:
    """rclpy.time.Time: nanoseconds, from and to the message."""

    def __init__(self, *, nanoseconds: int = 0) -> None:
        self.nanoseconds = nanoseconds

    @classmethod
    def from_msg(cls, msg: Any) -> RclpyTime:
        return cls(nanoseconds=msg.sec * 1_000_000_000 + msg.nanosec)

    def to_msg(self) -> Any:
        return Time(sec=self.nanoseconds // 1_000_000_000, nanosec=self.nanoseconds % 1_000_000_000)


class Duration:
    """rclpy.duration.Duration: seconds in, nanoseconds kept."""

    def __init__(self, *, seconds: float = 0.0, nanoseconds: int = 0) -> None:
        self.nanoseconds = int(seconds * 1e9) + nanoseconds


class Buffer:
    """tf2_ros.Buffer: a test fills ``transforms`` by (target, source) or sets ``error``."""

    def __init__(self) -> None:
        self.transforms: dict[tuple[str, str], Any] = {}
        self.error: Exception | None = None
        self.calls: list[tuple[Any, ...]] = []

    def lookup_transform(self, target: str, source: str, time: Any, timeout: Any = None) -> Any:
        self.calls.append((target, source, time, timeout))
        if self.error is not None:
            raise self.error
        return self.transforms[(target, source)]

    def lookup_transform_full(
        self,
        target: str,
        target_time: Any,
        source: str,
        source_time: Any,
        fixed: str,
        timeout: Any = None,
    ) -> Any:
        self.calls.append((target, target_time, source, source_time, fixed, timeout))
        if self.error is not None:
            raise self.error
        return self.transforms[(target, source)]


class TransformListener:
    """tf2_ros.TransformListener with ``spin_thread``: the executor and thread it would own."""

    def __init__(self, buffer: Any, node: Any, *, spin_thread: bool = False) -> None:
        self.buffer, self.node = buffer, node
        if spin_thread:
            self.executor = _Executor()
            self.dedicated_listener_thread = _Thread()


class _Executor:
    def __init__(self) -> None:
        self.shut_down = False

    def shutdown(self) -> None:
        self.shut_down = True


class _Thread:
    def __init__(self) -> None:
        self.joined = False

    def join(self, timeout: float | None = None) -> None:
        self.joined = True


class Rclpy(types.ModuleType):
    """The ``rclpy`` module a test drives: ``spin`` runs whatever ``on_spin`` is set to."""

    def __init__(self) -> None:
        super().__init__("rclpy")
        self.log: list[str] = []
        self.on_spin: Any = None

    def init(self, args: Any = None) -> None:
        self.log.append("init")

    def spin(self, node: Any) -> None:
        self.log.append("spin")
        if self.on_spin is not None:
            self.on_spin()

    def try_shutdown(self) -> None:
        self.log.append("try_shutdown")

    def ok(self) -> bool:
        return True


def _module(name: str, **names: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in names.items():
        setattr(module, key, value)
    return module


def install() -> Any:
    """Put the fakes into ``sys.modules`` and ``ros/pepin_bringup`` on the path; returns the
    fake ``rclpy`` module (its ``on_spin`` and ``log`` drive :func:`spin_main` tests)."""
    if "rclpy" in sys.modules and isinstance(sys.modules["rclpy"], Rclpy):
        return sys.modules["rclpy"]
    rclpy = Rclpy()
    modules = {
        "rclpy": rclpy,
        "rclpy.time": _module("rclpy.time", Time=RclpyTime),
        "rclpy.duration": _module("rclpy.duration", Duration=Duration),
        "rclpy.executors": _module(
            "rclpy.executors", ExternalShutdownException=ExternalShutdownException
        ),
        "rclpy.parameter": _module("rclpy.parameter", Parameter=Parameter),
        "rcl_interfaces": _module("rcl_interfaces"),
        "rcl_interfaces.msg": _module(
            "rcl_interfaces.msg",
            SetParametersResult=SetParametersResult,
            ParameterDescriptor=ParameterDescriptor,
            ParameterType=ParameterType,
            IntegerRange=IntegerRange,
            FloatingPointRange=FloatingPointRange,
        ),
        "builtin_interfaces": _module("builtin_interfaces"),
        "builtin_interfaces.msg": _module("builtin_interfaces.msg", Time=Time),
        "std_msgs": _module("std_msgs"),
        "std_msgs.msg": _module("std_msgs.msg", Header=Header),
        "geometry_msgs": _module("geometry_msgs"),
        "geometry_msgs.msg": _module(
            "geometry_msgs.msg",
            Transform=Transform,
            TransformStamped=TransformStamped,
            PoseWithCovarianceStamped=PoseWithCovarianceStamped,
            Quaternion=Quaternion,
            Vector3=Vector3,
        ),
        "sensor_msgs": _module("sensor_msgs"),
        "sensor_msgs.msg": _module(
            "sensor_msgs.msg",
            Image=Image,
            LaserScan=LaserScan,
            PointCloud2=PointCloud2,
            PointField=PointField,
            Imu=Imu,
        ),
        "tf2_ros": _module("tf2_ros", Buffer=Buffer, TransformListener=TransformListener),
    }
    sys.modules.update(modules)
    package = str(REPO / "ros" / "pepin_bringup")
    if package not in sys.path:
        sys.path.insert(0, package)
    return rclpy
