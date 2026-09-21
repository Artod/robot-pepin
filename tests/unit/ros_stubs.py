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
Service clients (:class:`Client`) and action clients (:class:`ActionClient`) are faked too, and
both start with nobody on the other end: a test says a service is ready, which is how the
absence of the tracker in SLAM mode is written down.
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
# One ToF cone, field for field as pepin_bringup.tof_bridge fills it and run_recorder reads it,
# with the two radiation types the message declares as class constants.
Range = _msg(
    "Range",
    header=Header,
    radiation_type=0,
    range=0.0,
    min_range=0.0,
    max_range=0.0,
    field_of_view=0.0,
)
for _name, _value in (("ULTRASOUND", 0), ("INFRARED", 1)):
    setattr(Range, _name, _value)
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
# rtabmap_msgs/SensorData as pepin_bringup.sensor_pack fills it, field for field as
# ``ros2 interface show rtabmap_msgs/msg/SensorData`` prints them in the laptop image
# (rtabmap_msgs 0.22.1): the two raw images, the two camera-info ARRAYS, the local-transform
# ARRAY beside them, and the laser scan with its own four fields. The fields this node never
# writes (the compressed halves, the features, the grid, the IMU, the GPS, the landmarks) are
# left out on purpose: a node that started writing one would fail here as it would on the robot.
SensorData = _msg(
    "SensorData",
    header=Header,
    left=Image,
    right=Image,
    left_camera_info=list,
    right_camera_info=list,
    local_transform=list,
    laser_scan=PointCloud2,
    laser_scan_max_pts=0,
    laser_scan_max_range=0.0,
    laser_scan_format=0,
    laser_scan_local_transform=Transform,
)
# rtabmap_msgs/MapGraph as the laptop's nodes read it, field for field as
# /opt/ros/jazzy/share/rtabmap_msgs/msg/MapGraph.msg lists them: the map -> odom transform, the node
# ids and their OPTIMISED poses as two parallel arrays, and the links with their information
# matrices. pepin_bringup.depth_fusion reads poses_id + poses (the room's own movement,
# pepin.graphbend) and pepin_bringup.rtabmap_frame reads map_to_odom in SLAM.
Link_ = _msg(
    "Link", from_id=0, to_id=0, type=0, transform=Transform, information=lambda: [0.0] * 36
)
MapGraph = _msg(
    "MapGraph",
    header=Header,
    map_to_odom=Transform,
    poses_id=list,
    poses=list,
    links=list,
)
# rtabmap_msgs/Info: the statistics table as two parallel arrays, and the node a closure or a
# proximity link MATCHED — the one thing that says an update localised at all.
Info = _msg(
    "Info",
    header=Header,
    ref_id=0,
    loop_closure_id=0,
    proximity_detection_id=0,
    landmark_id=0,
    stats_keys=list,
    stats_values=list,
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
TwistStamped = _msg("TwistStamped", header=Header, twist=Twist)
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


class Empty:
    """std_srvs/Empty: nothing in and nothing out, which is the whole of RTAB-Map's two set_mode
    services and of its ``update_parameters``."""

    Request = _msg("Empty_Request")
    Response = _msg("Empty_Response")


# rcl_interfaces/Parameter and ParameterValue as a node builds them for somebody ELSE's parameters
# (rclpy.parameter.Parameter above is a different thing: the local value a node declares).
ParameterMsg = _msg("Parameter", name="", value=None)
ParameterValueMsg = _msg(
    "ParameterValue", type=0, bool_value=False, integer_value=0, double_value=0.0, string_value=""
)


class SetParameters:
    """rcl_interfaces/SetParameters: a list of parameters in, one result each back."""

    Request = _msg("SetParameters_Request", parameters=list)
    Response = _msg("SetParameters_Response", results=list)


class SetLabel:
    """rtabmap_msgs/SetLabel, field for field as SetLabel.srv:3-4 lists them — and the response is
    EMPTY, so success and failure look the same to a caller (rtabmap_msgs/srv/SetLabel.srv). Node
    id 0 means "the last node", or beside a loaded database the node nearest the last localisation
    (rtabmap/core/Rtabmap.cpp's labelLocation)."""

    Request = _msg("SetLabel_Request", node_id=0, node_label="")
    Response = _msg("SetLabel_Response")


class ListLabels:
    """rtabmap_msgs/ListLabels: nothing in, the ids and their labels as two parallel arrays back
    (ListLabels.srv:4-5). The only way to learn WHICH node a set_label landed on."""

    Request = _msg("ListLabels_Request")
    Response = _msg("ListLabels_Response", ids=list, labels=list)


class RemoveLabel:
    """rtabmap_msgs/RemoveLabel: a label in (RemoveLabel.srv:2), nothing back."""

    Request = _msg("RemoveLabel_Request", label="")
    Response = _msg("RemoveLabel_Response")


DurationMsg = _msg("Duration", sec=0, nanosec=0)
Transition = _msg("Transition", id=0)
State = _msg("State", id=0, label="")


class ChangeState:
    """lifecycle_msgs/ChangeState: the transition to send, and whether it was taken."""

    Request = _msg("ChangeState_Request", transition=Transition)
    Response = _msg("ChangeState_Response", success=False)


class GetState:
    """lifecycle_msgs/GetState: the state a managed node is in."""

    Request = _msg("GetState_Request")
    Response = _msg("GetState_Response", current_state=State)


class ClearEntireCostmap:
    """nav2_msgs/ClearEntireCostmap: empty a costmap; nothing goes in and nothing comes back."""

    Request = _msg("ClearEntireCostmap_Request")
    Response = _msg("ClearEntireCostmap_Response")


class NavigateToPose:
    """nav2_msgs/NavigateToPose: the drive's goal — where to go, in the map frame."""

    Goal = _msg("NavigateToPose_Goal", pose=PoseStamped)


class Spin:
    """nav2_msgs/Spin: the turn in place and how long it may take."""

    Goal = _msg("Spin_Goal", target_yaw=0.0, time_allowance=DurationMsg)


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

    def __sub__(self, other: Any) -> RclpyTime:
        """rclpy's: a Duration off a Time is a Time (a stamp dated backwards)."""
        return RclpyTime(nanoseconds=self.nanoseconds - other.nanoseconds)


class Duration:
    """rclpy.duration.Duration: seconds in, nanoseconds kept."""

    def __init__(self, *, seconds: float = 0.0, nanoseconds: int = 0) -> None:
        self.nanoseconds = int(seconds * 1e9) + nanoseconds


class FilterSubscriber:
    """message_filters.Subscriber: an ordinary subscription that also feeds a synchronizer.

    It registers on the node like any other, so a test drives it through ``node.subs[topic]``
    exactly as it drives a plain one, and hands every message to whoever registered a callback.
    """

    def __init__(self, node: Any, msg_type: Any, topic: str, qos_profile: Any = None) -> None:
        self.topic = topic
        self.callbacks: list[Any] = []
        node.create_subscription(msg_type, topic, self._deliver, qos_profile or 10)

    def registerCallback(self, callback: Any) -> None:  # noqa: N802 (message_filters' own name)
        self.callbacks.append(callback)

    def _deliver(self, msg: Any) -> None:
        for callback in self.callbacks:
            callback(msg)


class TimeSynchronizer:
    """message_filters.TimeSynchronizer: messages of several subscribers paired by exact stamp.

    The real one keeps a queue per input and calls back with one message of each whose header
    stamps are identical. This keeps the newest of each and fires when every input has spoken
    for the same stamp — which is the only case the nodes here rely on (the depth copies the
    image's header, so a pair has one exact stamp)."""

    def __init__(self, subscribers: list[Any], queue_size: int) -> None:
        self.queue_size = queue_size
        self.callbacks: list[Any] = []
        self._held: list[dict[tuple[int, int], Any]] = [{} for _ in subscribers]
        for slot, subscriber in enumerate(subscribers):
            subscriber.registerCallback(lambda msg, slot=slot: self._offer(slot, msg))

    def registerCallback(self, callback: Any) -> None:  # noqa: N802 (message_filters' own name)
        self.callbacks.append(callback)

    def _offer(self, slot: int, msg: Any) -> None:
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._held[slot][stamp] = msg
        if not all(stamp in held for held in self._held):
            return
        paired = [held.pop(stamp) for held in self._held]
        for callback in self.callbacks:
            callback(*paired)


class Buffer:
    """tf2_ros.Buffer: a test fills ``transforms`` by (target, source) or sets ``error``.

    ``cache_time`` is kept the way the real one keeps it (a node that asks for a short history
    must not fail here), never enforced: these lookups have no history to trim.
    """

    def __init__(self, cache_time: Any = None) -> None:
        self.cache_time = cache_time
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
    """tf2_ros': keeps every transform a node asked it to send, in order — and in ``batches``,
    the calls themselves, because how many MESSAGES a node sends is a cost on the board."""

    def __init__(self, node: Any) -> None:
        self.node = node
        self.sent: list[Any] = []
        self.batches: list[list[Any]] = []

    def sendTransform(self, transforms: Any) -> None:  # noqa: N802 — tf2_ros' own name
        items = list(transforms) if isinstance(transforms, list) else [transforms]
        self.batches.append(items)
        self.sent.extend(items)


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


class Future:
    """rclpy's future, already finished: a callback fires the moment it is registered, which is
    what :meth:`GoalServer._wait` waits for."""

    def __init__(self, result: Any = None) -> None:
        self._result = result

    def add_done_callback(self, callback: Any) -> None:
        callback(self)

    def done(self) -> bool:
        return True

    def result(self) -> Any:
        return self._result


class Client:
    """A service client: ``ready`` says whether the service exists at all (an absent tracker is
    a client that never becomes ready), ``response`` is what a call answers with, every request
    is kept in ``calls`` and every ``wait_for_service`` timeout in ``waits`` — a wait on a
    service nobody serves is a second of the robot standing still, and tests say where those
    are paid."""

    def __init__(self, srv_type: Any, name: str) -> None:
        # ``srv_name``, as rclpy spells it: a stub that offered ``.name`` let a node log
        # ``client.name`` through every test and crash on its first live call (2026-09-18).
        self.srv_type, self.srv_name = srv_type, name
        self.ready = False
        self.response: Any = None
        self.calls: list[Any] = []
        self.waits: list[float] = []

    def wait_for_service(self, timeout_sec: float = 0.0) -> bool:
        self.waits.append(timeout_sec)
        return self.ready

    def service_is_ready(self) -> bool:
        return self.ready

    def call_async(self, request: Any) -> Future:
        self.calls.append(request)
        return Future(self.response)


class ActionClient:
    """rclpy.action's client as the nodes here use it: whether a server answers, and the goals
    it was sent (``handle`` is what ``send_goal_async`` hands back)."""

    def __init__(self, node: Any, action_type: Any, name: str) -> None:
        self.node, self.action_type, self.name = node, action_type, name
        self.server = False
        self.handle: Any = None
        self.goals: list[Any] = []

    def wait_for_server(self, timeout_sec: float = 0.0) -> bool:
        return self.server

    def send_goal_async(self, goal: Any, feedback_callback: Any = None) -> Future:
        self.goals.append(goal)
        return Future(self.handle)


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
        self.service_clients: dict[str, Client] = {}  # name -> the client this node created
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

    def get_parameter(self, name: str) -> Parameter:
        """rclpy's: the value the node holds now — what :func:`parameters` overrode, or what
        the node declared (a node that reads a parameter per message reads it through this)."""
        return Parameter(name, value=PARAMETERS.get(name, self.declared[name]))

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

    def create_client(self, srv_type: Any, name: str) -> Client:
        """A client of someone else's service; not ready until a test says the service is there."""
        self.service_clients[name] = Client(srv_type, name)
        return self.service_clients[name]

    def create_timer(self, period_s: float, callback: Any) -> None:
        self.timers.append((period_s, callback))

    def get_clock(self) -> Clock:
        return self.clock

    def get_logger(self) -> Logger:
        return self.logger

    def destroy_node(self) -> None:
        self.destroyed = True


class AsyncParameterClient:
    """rclpy.parameter_client.AsyncParameterClient as the nodes here use it: the name of the
    node whose parameters are being set, and every set that was fired at it (never waited on)."""

    def __init__(self, node: Any, remote: str) -> None:
        self.node = node
        self.remote = remote
        self.sets: list[list[Any]] = []

    def set_parameters(self, parameters: list[Any]) -> Any:
        """Remember the call and answer with a future nobody waits on."""
        self.sets.append(list(parameters))
        return None


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
        "rclpy.parameter_client": _module(
            "rclpy.parameter_client", AsyncParameterClient=AsyncParameterClient
        ),
        "rclpy.action": _module("rclpy.action", ActionClient=ActionClient),
        "rcl_interfaces": _module("rcl_interfaces"),
        "rcl_interfaces.msg": _module(
            "rcl_interfaces.msg",
            SetParametersResult=SetParametersResult,
            ParameterDescriptor=ParameterDescriptor,
            ParameterType=ParameterType,
            IntegerRange=IntegerRange,
            FloatingPointRange=FloatingPointRange,
            Parameter=ParameterMsg,
            ParameterValue=ParameterValueMsg,
        ),
        "rcl_interfaces.srv": _module("rcl_interfaces.srv", SetParameters=SetParameters),
        "builtin_interfaces": _module("builtin_interfaces"),
        "builtin_interfaces.msg": _module(
            "builtin_interfaces.msg", Time=Time, Duration=DurationMsg
        ),
        "lifecycle_msgs": _module("lifecycle_msgs"),
        "lifecycle_msgs.srv": _module(
            "lifecycle_msgs.srv", ChangeState=ChangeState, GetState=GetState
        ),
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
            TwistStamped=TwistStamped,
        ),
        "nav_msgs": _module("nav_msgs"),
        "nav_msgs.msg": _module(
            "nav_msgs.msg", OccupancyGrid=OccupancyGrid, Odometry=Odometry, Path=Path_
        ),
        "rtabmap_msgs": _module("rtabmap_msgs"),
        "rtabmap_msgs.msg": _module(
            "rtabmap_msgs.msg",
            SensorData=SensorData,
            MapGraph=MapGraph,
            Info=Info,
            Link=Link_,
        ),
        "rtabmap_msgs.srv": _module(
            "rtabmap_msgs.srv",
            SetLabel=SetLabel,
            ListLabels=ListLabels,
            RemoveLabel=RemoveLabel,
        ),
        "nav2_msgs": _module("nav2_msgs"),
        "nav2_msgs.msg": _module("nav2_msgs.msg", ParticleCloud=ParticleCloud),
        "nav2_msgs.action": _module("nav2_msgs.action", NavigateToPose=NavigateToPose, Spin=Spin),
        "nav2_msgs.srv": _module("nav2_msgs.srv", ClearEntireCostmap=ClearEntireCostmap),
        "action_msgs": _module("action_msgs"),
        "action_msgs.msg": _module(
            "action_msgs.msg", GoalStatus=GoalStatus, GoalStatusArray=GoalStatusArray
        ),
        "std_srvs": _module("std_srvs"),
        "std_srvs.srv": _module("std_srvs.srv", Trigger=Trigger, Empty=Empty),
        "tf2_msgs": _module("tf2_msgs"),
        "tf2_msgs.msg": _module("tf2_msgs.msg", TFMessage=TFMessage),
        "sensor_msgs": _module("sensor_msgs"),
        "sensor_msgs.msg": _module(
            "sensor_msgs.msg",
            CameraInfo=CameraInfo,
            Image=Image,
            LaserScan=LaserScan,
            PointCloud2=PointCloud2,
            Range=Range,
            PointField=PointField,
            Imu=Imu,
        ),
        "message_filters": _module(
            "message_filters", Subscriber=FilterSubscriber, TimeSynchronizer=TimeSynchronizer
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
