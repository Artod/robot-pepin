"""The frames OpenVINS sees: camera_stream's two eyes, without the pairs a fast head smeared.

OpenVINS tracks features from one frame to the next and never re-initialises by itself: on the
run4 replay of 2026-10-04 the fast pans (290 deg/s) threw its position 2.2-3.9 m off and the
saccade dance diverged it for good, while slow sweeps (20 deg/s) cost it 1-2 cm. So the frames to
keep from it are the FAST ones, and only those: the gaze arbiter's blind intervals
(:class:`pepin.gaze_gate.FrameGate`'s first rule, what depth and RTAB-Map drop by) cover every
head move, slow ones included, and fed through them OpenVINS lost its frames for whole 2-3 s
legs and diverged within seconds (the same replay: km off, 12-24 deg). This node judges a pair
by the head's own rate instead, with the same gate's rate rule: a pair whose exposure window
(``gate_exposure_s``, ``gate_stamp_end``, as in every gated node) holds a ``/head/imu`` sample
turning faster than ``head_rate_dps`` (the norm of the gyro) is held back. The two eyes are
paired by their one stamp (camera_stream stamps both alike) and an admitted pair goes out on
``/vio/image`` (the left eye in grey: OpenVINS converts it to mono8 anyway, and grey is a third
of the bytes) and ``/vio/right/image`` (already grey, passed through); ``vio.launch.py`` remaps
OpenVINS's two image topics onto these. Between two admitted pairs OpenVINS propagates on the
head IMU alone, which is what it is for.

UNTIL OPENVINS HAS INITIALISED every pair passes: its static initialisation reads the jerk of
the first motion off the frames taken during it (gated from the start, the replay never
initialised: 566 "no accel jerk detected" in 120 s). Its first ``/ov_msckf/poseimu`` turns the
gate on; :data:`INIT_STALE_S` without one (OpenVINS restarted, respawned or dead) turns it off.

With the ``rate_gate`` flag off every pair passes: a plain relay, the A/B arm.

NOT YET A CURE (the same replay, executor single): slow sweeps lose nothing to it (position
error p50/max 1.7/4.2 and 1.3/1.8 cm, as ungated), but the fast pans end 5.8 m off (ungated
3.9 m) and the fast tilts 60 m off (ungated 14 cm): propagating a saccade on the IMU alone and
re-acquiring after it is worse than tracking through the smear. So ``vio.launch.py`` runs
without it (``feed`` false) until something does better.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu

from pepin.camera import camera_rate, follow_period
from pepin.flags import Flag, FlagSet, load_knobs, with_knobs
from pepin.gaze_gate import EXPOSURE_PER_PERIOD, SPINNING, FrameGate
from pepin_bringup.msgs import stamp_seconds
from pepin_bringup.node_kit import Switches, Tally, spin_main
from pepin_bringup.stereo_frames import LEFT_IMAGE_TOPIC, RIGHT_IMAGE_TOPIC

VIO_LEFT_TOPIC = "/vio/image"
VIO_RIGHT_TOPIC = "/vio/right/image"
HEAD_IMU_TOPIC = "/head/imu"
OV_POSE_TOPIC = "/ov_msckf/poseimu"  # OpenVINS publishes it per update, from its init on
# Seconds without an OpenVINS pose after which it counts as uninitialised again: longer than any
# stretch of held-back pairs (a saccade lasts tenths of a second), shorter than OpenVINS's
# respawn and re-initialisation.
INIT_STALE_S = 10.0
REPORT_S = 30.0
# Halves kept while their other eye is on its way: a second of frames. camera_stream publishes
# both eyes back to back, so a half older than this lost its partner and is counted.
PENDING_KEPT = 10

FLAGS = FlagSet(
    Flag(
        "rate_gate",
        True,
        description="a pair whose exposure window (gate_exposure_s, gate_stamp_end) holds a"
        " /head/imu sample turning faster than head_rate_dps (the gyro's norm) is held back from"
        " OpenVINS, once OpenVINS has initialised; off, every pair passes (a plain relay)",
        why="the run4 replay through the zenoh router (2026-10-04): ungated, the fast pans"
        " threw OpenVINS 3.9 m off; gated by the arbiter's blind intervals (every move, slow"
        " sweeps included) it diverged within seconds",
        on_when="whenever the head moves at saccade speed while OpenVINS runs",
        off_when="to measure what the smeared frames do to OpenVINS (the A/B arm), or if the"
        " held-back pairs starve it (the report line counts them)",
    ),
)


class EyePairs:
    """The two eyes matched by their stamp (nanoseconds): :meth:`add` returns ``(left, right)``
    when the other half of that stamp is already here, and ``None`` until then. At most
    ``keep`` stamps wait; the oldest goes first and is counted in :attr:`unpaired`."""

    def __init__(self, keep: int = PENDING_KEPT) -> None:
        self._keep = keep
        self._pending: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.unpaired = 0

    def add(self, side: str, stamp_ns: int, msg: Any) -> tuple[Any, Any] | None:
        """One eye (``"left"`` or ``"right"``) of the frame stamped ``stamp_ns``."""
        halves = self._pending.setdefault(stamp_ns, {})
        halves[side] = msg
        if "left" in halves and "right" in halves:
            del self._pending[stamp_ns]
            return halves["left"], halves["right"]
        while len(self._pending) > self._keep:
            self._pending.popitem(last=False)
            self.unpaired += 1
        return None


def grey(msg: Any) -> Any:
    """``msg`` as a mono8 picture: a mono8 one as it is, a bgr8 one through OpenCV's BGR2GRAY
    (what OpenVINS's own cv_bridge conversion does), header and size kept."""
    if msg.encoding == "mono8":
        return msg
    if msg.encoding != "bgr8":
        raise ValueError(f"the left eye is bgr8 or mono8, not {msg.encoding}")
    rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    pixels = np.ascontiguousarray(rows[:, : msg.width * 3]).reshape(msg.height, msg.width, 3)
    out = Image()
    out.header = msg.header
    out.height, out.width = msg.height, msg.width
    out.encoding, out.is_bigendian, out.step = "mono8", 0, msg.width
    out.data = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY).tobytes()
    return out


class VioFeed(Node):
    """Republishes the eye pairs a fast head did not smear, for OpenVINS."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__("vio_feed")
        self._now = clock
        self._pose_at: float | None = None  # when OpenVINS's last pose arrived (``clock``)
        self._switches = Switches(
            self, with_knobs(FLAGS, load_knobs("vio_feed")), on_change=self._on_switch
        )
        # The gaze gate's rate rule alone: no /gaze/state is fed, so no blind interval exists.
        self._period_s = camera_rate().period_s  # gate_exposure_s 0 follows the camera's rate
        self._gate = FrameGate(
            exposure_s=follow_period(
                float(self._switches["gate_exposure_s"]), EXPOSURE_PER_PERIOD, self._period_s
            ),
            stamp_end=float(self._switches["gate_stamp_end"]) >= 0.5,
            yaw_dps=float(self._switches["head_rate_dps"]),
        )
        self._pairs = EyePairs()
        self._tally = Tally(("grey",))
        # camera_stream's publishers are reliable, five deep; OpenVINS's message_filters
        # subscribers are reliable, ten deep.
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._left_pub = self.create_publisher(Image, VIO_LEFT_TOPIC, reliable)
        self._right_pub = self.create_publisher(Image, VIO_RIGHT_TOPIC, reliable)
        self.create_subscription(Image, LEFT_IMAGE_TOPIC, self._on_left, reliable)
        self.create_subscription(Image, RIGHT_IMAGE_TOPIC, self._on_right, reliable)
        self.create_subscription(Imu, HEAD_IMU_TOPIC, self._on_imu, qos_profile_sensor_data)
        self.create_subscription(
            PoseWithCovarianceStamped,
            OV_POSE_TOPIC,
            self._on_pose,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.create_timer(REPORT_S, self._report)
        self.get_logger().info(
            f"vio feed up: {LEFT_IMAGE_TOPIC} + {RIGHT_IMAGE_TOPIC} -> {VIO_LEFT_TOPIC} +"
            f" {VIO_RIGHT_TOPIC}, without the pairs the head turned faster than head_rate_dps"
            f" through; flags: {self._switches.state()}"
        )

    def _on_switch(self, name: str, _old: object, new: object) -> None:
        """A knob changed: the gate takes it for the next pair."""
        if name == "gate_exposure_s":
            self._gate.exposure_s = follow_period(
                float(new),  # type: ignore[arg-type]
                EXPOSURE_PER_PERIOD,
                self._period_s,
            )
        elif name == "gate_stamp_end":
            self._gate.stamp_end = float(new) >= 0.5  # type: ignore[arg-type]
        elif name == "head_rate_dps":
            self._gate.yaw_dps = float(new)  # type: ignore[arg-type]

    def _on_imu(self, msg: Any) -> None:
        w = msg.angular_velocity
        self._gate.observe_yaw(stamp_seconds(msg.header.stamp), math.hypot(w.x, w.y, w.z))

    def _on_pose(self, _msg: Any) -> None:
        self._pose_at = self._now()

    def initialised(self) -> bool:
        """Whether OpenVINS has published a pose within :data:`INIT_STALE_S`."""
        return self._pose_at is not None and self._now() - self._pose_at < INIT_STALE_S

    def _on_left(self, msg: Any) -> None:
        self._take("left", msg)

    def _on_right(self, msg: Any) -> None:
        self._take("right", msg)

    def _take(self, side: str, msg: Any) -> None:
        stamp = msg.header.stamp
        pair = self._pairs.add(side, int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec), msg)
        if pair is None:
            return
        self._tally.count("in")
        if not self.initialised():
            self._tally.count("before_init")
        elif (
            self._switches.on("rate_gate")
            and self._gate.verdict(stamp_seconds(stamp), self._now()) == SPINNING
        ):
            self._tally.count("fast")
            return
        left, right = pair
        with self._tally.measure("grey"):
            left = grey(left)
        self._left_pub.publish(left)
        self._right_pub.publish(right)
        self._tally.count("out")

    def _report(self) -> None:
        """Every 30 s: pairs in and out, how many passed ungated before OpenVINS's init, how
        many a fast head held back, halves that never found their other eye, the grey
        conversion's milliseconds and the flags."""
        w = self._tally.take()
        init = "OpenVINS initialised" if self.initialised() else "OpenVINS NOT initialised"
        grey_ms = w.timing["grey"]
        conversion = (
            f", grey {grey_ms.median_ms:.1f}/{grey_ms.p95_ms:.1f} ms" if grey_ms.count else ""
        )
        self.get_logger().info(
            f"vio feed: {w.rate('in'):.1f} pairs/s in, {w.rate('out'):.1f} out; {init},"
            f" {w.counts['before_init']} passed ungated before it; {w.counts['fast']} of"
            f" {w.counts['in']} held back for a head faster than"
            f" {float(self._switches['head_rate_dps']):g} deg/s;"
            f" {self._pairs.unpaired} halves unpaired since the start{conversion};"
            f" flags: {self._switches.state()}"
        )


def main() -> None:
    spin_main(VioFeed)


if __name__ == "__main__":
    main()
