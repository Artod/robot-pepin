"""Just enough of ROS for the ``pepin_bringup`` modules to import and run here.

rclpy is not installed on the laptop, so the ROS package cannot be imported by the unit tests —
except its pure modules, whose ROS imports are message classes (fields with defaults) and a
handful of rclpy names. :func:`install` puts fakes of exactly those into ``sys.modules`` (once,
idempotent) and puts ``ros/pepin_bringup`` on the path, the way ``test_ros_bridge_protocol``
reaches ``pepin_bringup.protocol``. The fakes hold the fields the real messages have and nothing
else: a codec that wrote a field the message lacks would fail here as it would on the robot.

Beside the messages there is enough of :class:`Node` — parameters with the overrides of
:func:`parameters`, publishers that keep what they were handed, timers, a clock, a logger — for
a whole node to be built in a test and driven through :func:`pepin_bringup.node_kit.spin_main`.
"""

from __future__ import annotations

import contextlib
import sys
import types
from collections.abc import Iterator
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
CameraInfo = _msg(
    "CameraInfo",
    header=Header,
    height=0,
    width=0,
    distortion_model="",
    d=list,
    k=lambda: [0.0] * 9,
    r=lambda: [0.0] * 9,
    p=lambda: [0.0] * 12,
)
Twist = _msg("Twist", linear=Vector3, angular=Vector3)
TwistWithCovariance = _msg("TwistWithCovariance", twist=Twist, covariance=lambda: [0.0] * 36)
Odometry = _msg(
    "Odometry",
    header=Header,
    child_frame_id="",
    pose=PoseWithCovariance,
    twist=TwistWithCovariance,
)
PoseStamped = _msg("PoseStamped", header=Header, pose=Pose)
PoseArray = _msg("PoseArray", header=Header, poses=list)
Path_ = _msg("Path", header=Header, poses=list)
MapMetaData = _msg("MapMetaData", resolution=0.0, width=0, height=0, origin=Pose)
OccupancyGrid = _msg("OccupancyGrid", header=Header, info=MapMetaData, data=list)
Bool = _msg("Bool", data=False)
Float32 = _msg("Float32", data=0.0)
String = _msg("String", data="")
TFMessage = _msg("TFMessage", transforms=list)
GoalStatus = _msg("GoalStatus", status=0)
for _name, _value in (("STATUS_ACCEPTED", 1), ("STATUS_EXECUTING", 2), ("STATUS_SUCCEEDED", 4)):
    setattr(GoalStatus, _name, _value)
GoalStatusArray = _msg("GoalStatusArray", status_list=list)
ParticleCloud = _msg("ParticleCloud", header=Header, particles=list)
SetParametersResult = _msg("SetParametersResult", successful=False, reason="")


class Trigger:
    """std_srvs/Trigger: the request has nothing, the response a flag and a message."""

    Request = _msg("Trigger_Request")
    Response = _msg("Trigger_Response", success=False, message="")


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
    """rclpy.parameter.Parameter: a name and a value (the type inferred, as rclpy does) — what
    a ``set`` carries and what :meth:`Node.declare_parameter` answers with."""

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

    def __add__(self, other: Any) -> RclpyTime:
        return RclpyTime(nanoseconds=self.nanoseconds + other.nanoseconds)


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

    def set_transform_static(self, transform: Any, authority: str) -> None:
        """A static transform kept by (parent, child), as a lookup of that pair finds it."""
        self.transforms[(transform.header.frame_id, transform.child_frame_id)] = transform

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


class StaticTransformBroadcaster:
    """tf2_ros': keeps every transform a node asked it to send, in order."""

    def __init__(self, node: Any) -> None:
        self.node = node
        self.sent: list[Any] = []

    def sendTransform(self, transforms: Any) -> None:  # noqa: N802 — tf2_ros' own name
        self.sent.extend(transforms if isinstance(transforms, list) else [transforms])


class TransformBroadcaster(StaticTransformBroadcaster):
    """tf2_ros': the dynamic one, kept the same way."""


class DurabilityPolicy:
    """rclpy.qos.DurabilityPolicy, the two the nodes here ask for."""

    TRANSIENT_LOCAL = "transient_local"
    VOLATILE = "volatile"


class QoSProfile:
    """rclpy.qos.QoSProfile: what a publisher was created with, for a test to check."""

    def __init__(self, *, depth: int = 10, reliability: Any = None, **rest: Any) -> None:
        self.depth, self.reliability, self.rest = depth, reliability, rest


class ReliabilityPolicy:
    """rclpy.qos.ReliabilityPolicy, the two the nodes here ask for."""

    RELIABLE = "reliable"
    BEST_EFFORT = "best_effort"


class Publisher:
    """A publisher that publishes into a list: ``sent`` is what went out on the topic."""

    def __init__(self, msg_type: Any, topic: str, qos: Any) -> None:
        self.msg_type, self.topic, self.qos = msg_type, topic, qos
        self.sent: list[Any] = []

    def publish(self, msg: Any) -> None:
        self.sent.append(msg)


class Clock:
    """rclpy's clock: ``now()`` is whatever ``seconds`` says, so a test owns the time."""

    def __init__(self, seconds: float = 0.0) -> None:
        self.seconds = seconds

    def now(self) -> RclpyTime:
        return RclpyTime(nanoseconds=round(self.seconds * 1e9))


class Logger:
    """rclpy's logger: every line, with its level, kept for the test to read."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def info(self, text: str, **rest: Any) -> None:
        self.lines.append(("info", text))

    def warning(self, text: str, **rest: Any) -> None:
        self.lines.append(("warning", text))

    def error(self, text: str, **rest: Any) -> None:
        self.lines.append(("error", text))

    def debug(self, text: str, **rest: Any) -> None:
        self.lines.append(("debug", text))

    def texts(self, level: str | None = None) -> list[str]:
        """The lines said so far, of one level or of all of them."""
        return [text for kind, text in self.lines if level is None or kind == level]


PARAMETERS: dict[str, Any] = {}  # what declare_parameter answers instead of the node's default


@contextlib.contextmanager
def parameters(**values: Any) -> Iterator[None]:
    """The parameter overrides a launch would pass on the command line, for the nodes built
    inside the block (``with parameters(board='127.0.0.1'): node = CameraStream()``)."""
    PARAMETERS.update(values)
    try:
        yield
    finally:
        for name in values:
            PARAMETERS.pop(name, None)


class Node:
    """rclpy.node.Node as the nodes here use it: parameters over :data:`PARAMETERS`, publishers
    and subscriptions kept by topic, timers kept by period, one clock and one logger."""

    def __init__(self, name: str) -> None:
        self.node_name = name
        self.declared: dict[str, Any] = {}
        self.descriptors: dict[str, Any] = {}
        self.pubs: dict[str, Publisher] = {}
        self.subs: dict[str, tuple[Any, Any]] = {}  # topic -> (message type, callback)
        self.services: dict[str, tuple[Any, Any]] = {}  # name -> (service type, callback)
        self.timers: list[tuple[float, Any]] = []
        self.parameter_callbacks: list[Any] = []
        self.clock = Clock()
        self.logger = Logger()
        self.destroyed = False

    def declare_parameter(self, name: str, default: Any, descriptor: Any = None) -> Parameter:
        """Declare a parameter as rclpy does — the descriptor kept for a test to read — and
        answer with the value the node reads back: the override of :func:`parameters`, or the
        default."""
        self.declared[name] = default
        if descriptor is not None:
            self.descriptors[name] = descriptor
        return Parameter(name, value=PARAMETERS.get(name, default))

    def add_on_set_parameters_callback(self, callback: Any) -> None:
        self.parameter_callbacks.append(callback)

    def set_parameters(self, params: list[Any]) -> list[Any]:
        """What ``ros2 param set`` does: every callback is asked, the first refusal is the
        answer."""
        return [callback(params) for callback in self.parameter_callbacks]

    def create_publisher(self, msg_type: Any, topic: str, qos: Any) -> Publisher:
        self.pubs[topic] = Publisher(msg_type, topic, qos)
        return self.pubs[topic]

    def create_subscription(self, msg_type: Any, topic: str, callback: Any, qos: Any) -> None:
        self.subs[topic] = (msg_type, callback)

    def create_service(self, srv_type: Any, name: str, callback: Any) -> None:
        self.services[name] = (srv_type, callback)

    def create_timer(self, period_s: float, callback: Any) -> None:
        self.timers.append((period_s, callback))

    def get_clock(self) -> Clock:
        return self.clock

    def get_logger(self) -> Logger:
        return self.logger

    def destroy_node(self) -> None:
        self.destroyed = True


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
        "rclpy.node": _module("rclpy.node", Node=Node),
        "rclpy.qos": _module(
            "rclpy.qos",
            QoSProfile=QoSProfile,
            ReliabilityPolicy=ReliabilityPolicy,
            DurabilityPolicy=DurabilityPolicy,
            qos_profile_sensor_data=QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT),
        ),
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
        "std_msgs.msg": _module(
            "std_msgs.msg", Header=Header, Bool=Bool, Float32=Float32, String=String
        ),
        "geometry_msgs": _module("geometry_msgs"),
        "geometry_msgs.msg": _module(
            "geometry_msgs.msg",
            Transform=Transform,
            TransformStamped=TransformStamped,
            PoseWithCovarianceStamped=PoseWithCovarianceStamped,
            PoseWithCovariance=PoseWithCovariance,
            Pose=Pose,
            PoseStamped=PoseStamped,
            PoseArray=PoseArray,
            Point=Point,
            Quaternion=Quaternion,
            Vector3=Vector3,
            Twist=Twist,
        ),
        "nav_msgs": _module("nav_msgs"),
        "nav_msgs.msg": _module(
            "nav_msgs.msg", OccupancyGrid=OccupancyGrid, Odometry=Odometry, Path=Path_
        ),
        "nav2_msgs": _module("nav2_msgs"),
        "nav2_msgs.msg": _module("nav2_msgs.msg", ParticleCloud=ParticleCloud),
        "action_msgs": _module("action_msgs"),
        "action_msgs.msg": _module(
            "action_msgs.msg", GoalStatus=GoalStatus, GoalStatusArray=GoalStatusArray
        ),
        "std_srvs": _module("std_srvs"),
        "std_srvs.srv": _module("std_srvs.srv", Trigger=Trigger),
        "tf2_msgs": _module("tf2_msgs"),
        "tf2_msgs.msg": _module("tf2_msgs.msg", TFMessage=TFMessage),
        "sensor_msgs": _module("sensor_msgs"),
        "sensor_msgs.msg": _module(
            "sensor_msgs.msg",
            CameraInfo=CameraInfo,
            Image=Image,
            LaserScan=LaserScan,
            PointCloud2=PointCloud2,
            PointField=PointField,
            Imu=Imu,
        ),
        "tf2_ros": _module(
            "tf2_ros",
            Buffer=Buffer,
            TransformListener=TransformListener,
            StaticTransformBroadcaster=StaticTransformBroadcaster,
            TransformBroadcaster=TransformBroadcaster,
        ),
    }
    sys.modules.update(modules)
    package = str(REPO / "ros" / "pepin_bringup")
    if package not in sys.path:
        sys.path.insert(0, package)
    return rclpy
