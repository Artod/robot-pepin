"""RTAB-Map's visual odometry on its way to the EKF: ``/vo/raw`` in, ``/vo`` out.

``rgbd_odometry`` (rtabmap_odom, started beside this node by vslam.launch.py) reads the neck
camera's picture, the network's metric depth and the camera's optics and publishes a pose per
frame on ``/vo/raw``. This node is the one thing between it and the board's EKF: it drops what
the filter must never see (a frame rtabmap lost, a jump no cart of this speed could have made —
:class:`pepin.visual_odometry.VoGate`), puts a documented constant covariance on what is left
(:func:`pepin.visual_odometry.planar_covariance`; rtabmap's own covariance is a registration's
verdict on a depth image whose SCALE is a network's, so it cannot answer for that scale's
error), and publishes ``/vo``, which crosses the bridge to the board where robot_localization
fuses x and y — differentially, as a velocity, so the visual odometry can never move the odom
frame — and no yaw at all, because the gyro owns heading (with it the EKF's turn error is ~5 %;
the wheels alone over-report turns 40-70 %, 2026-09-13).

The number that decides whether it may be fused at all is measured here: with the wheels
reporting zero the cart stands still, and every centimetre the visual odometry walks in that
minute is its own drift (:class:`pepin.visual_odometry.RestWatch`, fed from the board's
``/odom``). It is printed in every report line beside the rates and the drops.

The flags and knobs (:data:`FLAGS` and config/knobs.json, ``ros/flags.sh set visual_odometry <flag>
<value>``): ``vo_publish`` (whether the measured odometry leaves this laptop at all — off, the EKF
is exactly what it was before this node existed), ``vo_covariance`` (the constant or rtabmap's own),
``vo_sigma_m`` and ``vo_yaw_sigma_deg`` (the constant), ``vo_max_speed``, ``vo_max_turn``,
``vo_max_gap_s`` and ``vo_reset_radius_m`` (the gate's ceilings) and ``vo_publish_hz`` (how often it
is published). The published pose is the sum of the steps the gate admitted, never rtabmap's own: a
refused jump re-anchors the gate, and a filter that differences the stream it receives would
otherwise get the whole discontinuity in one frame time (2026-09-14, odom -> base_link 43 km out).

The gaze gate (``gaze_gate`` and the ``gate_*`` knobs, :mod:`pepin.gaze_gate`) withholds the
same way: a pose of a frame taken during a head saccade (``/gaze/state``) or a body yaw above
``gate_yaw_dps``, and the first pose after one, only re-anchor — the EKF coasts on the wheels and
the gyro across them.

THE INPUT is the ``vo_input`` parameter, read at start and passed by vslam.launch.py: ``stereo``
or ``depth`` read rtabmap's ``/vo/raw`` as above; ``vio`` reads OpenVINS's ``/ov_msckf/poseimu``
(the head IMU's pose in its own gravity frame, per image update, stamped camera time + t_d; it
runs in its own container, ros/laptop.sh vio) and composes base_link's pose through TF at the
pose's own stamp (``head_imu <- base_link``: camera_stream's static head_imu edge, the board's
neck chain; :func:`pepin.visual_odometry.compose_base_pose`). Then the same gate, track, cap and
EKF slot. Its covariance is the ``vio`` mode, a per-step model floored at ``vo_sigma_m``
(OpenVINS's own marginal covariance only grows: it is read as a health signal, re-inits counted);
it is marked lost on the three rules of :class:`pepin.visual_odometry.VioLost` (the wheels'
speed, motion under ``/zupt``, the feature count of ``/ov_msckf/points_msckf``), and a lost pose
only re-anchors. OpenVINS never resets itself: ``ros/laptop.sh vio kick`` at rest.
"""

from __future__ import annotations

import math
import time
from collections import Counter

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from pepin.depth import rotation_matrix
from pepin.flags import Flag, FlagSet, load_knobs, with_knobs
from pepin.gaze_gate import GATE_KNOBS, GAZE_GATE
from pepin.visual_odometry import (
    PublishCap,
    RestWatch,
    VioHealth,
    VioLost,
    VoGate,
    VoPose,
    VoTrack,
    compose_base_pose,
    homogeneous,
    is_lost,
    planar_covariance,
    scaled_covariance,
)
from pepin_bringup.gaze_feed import GazeFeed, gate_counts
from pepin_bringup.msgs import pose_from_transform, stamp_seconds, yaw_of
from pepin_bringup.node_kit import Switches, Tally, TfLookup, bridged_qos_profile, spin_main

RAW_TOPIC = "/vo/raw"  # rgbd_odometry's own output, on this laptop only
VO_TOPIC = "/vo"  # what crosses to the board's EKF
WHEELS_TOPIC = "/odom"  # the board's wheel odometry: the rest signal, over the bridge
REPORT_S = 30.0
# The vo_input parameter: what the relay reads. stereo/depth: rtabmap's /vo/raw (the launch
# starts stereo_odometry or rgbd_odometry beside it); vio: OpenVINS in its own container.
VO_INPUTS = ("stereo", "depth", "vio")
VIO_POSE_TOPIC = "/ov_msckf/poseimu"  # the IMU's pose in G per image update (not odomimu)
VIO_POINTS_TOPIC = "/ov_msckf/points_msckf"  # the features of the last update: the count
ZUPT_TOPIC = "/zupt"  # the board's zero-velocity update: published only while it says rest
IMU_FRAME = "head_imu"  # camera_stream's static camera_optical -> head_imu (config/camera.json)
BASE_FRAME = "base_link"
ODOM_FRAME = "odom"  # the frame rtabmap's /vo/raw carries (odom_frame_id), kept for the EKF
TF_TIMEOUT_S = 0.05  # how long a pose waits for the neck chain at its stamp

FLAGS = FlagSet(
    Flag(
        "vo_publish",
        True,
        description=f"the gated visual odometry leaves this laptop as {VO_TOPIC}, where the"
        " board's EKF fuses it as a third input beside the wheels and the gyro; off, the node"
        " still measures and reports and the EKF is exactly what it was without it",
        why="on since 2026-09-14 13:40: at full rate (vo_publish_hz 10) the board's EKF missed its"
        " 20 Hz period 0 times in 130 s at rest and twice in 3 min of driving, |vy| stayed under"
        " 0.0003 m/s at rest, the odometry runaway guard counted 0, and the live pose against the"
        " lidar truth was 0.9 / 2.1 / 4.2 cm — the same as without (tapes 0261/0262 vs 0265/0266)."
        " Before that it shipped off because the half that matters is unmeasured. AT REST it is"
        " measured and it"
        " passes: on this laptop's live topics, with the launch's own parameters, this node"
        " gated 9.4-9.7 poses/s and dropped none, and the drift over 60 s with the wheels"
        " reporting a hard zero was 0.2 cm and 0.0 deg (worst stretch 0.4 cm); the same camera"
        " read by scratch/vo_probe.py for 85 s gave 9.1 poses/s, 630 inlier features a frame,"
        " one lost frame (the first) and 0.48 cm / 0.075 deg — against the centimetre and"
        " half-degree a minute this source has to stay under (2026-09-14). IN"
        " MOTION nobody has compared it with anything, and the EKF's odom -> base_link is what"
        " every other measurement in the stack is carried over: the tracker's scans, the"
        " camera's measurements, the costmaps. The scale is why the caution is not ceremony —"
        " the translation rgbd_odometry reports is the depth image's, and that depth is a"
        " network's corrected by a law fitted against the lidar (0.94 to 1.98 across one"
        " afternoon, 2026-09-11)",
        on_when="after one drive compares odom -> base_link with it on and off over the same"
        " path (it is a live flag exactly so the two runs are a minute apart) and the visual"
        " odometry did not disagree with a lidar-measured distance by more than the wheels did",
        off_when="the moment odom -> base_link must be the wheels and the gyro alone: a dark"
        " room, a blank wall, a depth law that has not been fitted this session, or any drive"
        " whose odometry is the measurement",
    ),
    Flag(
        "vo_covariance",
        "dynamic",
        choices=("dynamic", "constant", "rtabmap", "vio"),
        description="whose covariance rides on the published pose: `dynamic`, the registration's"
        " own sigma and the depth scale's share of the step just taken added in quadrature"
        " (pepin.visual_odometry.scaled_covariance); the documented constant (vo_sigma_m,"
        " vo_yaw_sigma_deg); the one rtabmap's registration computed, untouched; or `vio`, a"
        " per-step model for the visual-inertial input (vio_step_fraction of the step, floored at"
        " vo_sigma_m = one wheel sample; vo_input vio pins it, since OpenVINS's own covariance is"
        " the MARGINAL of an unobservable global pose and only grows)",
        why="dynamic since 2026-09-16, because neither of the other two answers the right"
        " question. rtabmap's own number is a claim about the PICTURE: measured at rest on this"
        " robot (2026-09-14, scratch/vo_probe.py, 85 s) its registration claimed a position"
        " standard deviation of 3.8 mm at the median and 15.9 mm at p90, and 3.8 mm through"
        " robot_localization's differential conversion (2 * sigma^2 * dt) is a velocity sigma of"
        " 1.8 mm/s, 325x the wheels' certainty per sample — no longer a third opinion but the"
        " whole odometry (scratch/vo_weight.py). The error that matters grows with the step the"
        " cart took, through the depth's scale, which no registration can see; a constant ignores"
        " both. So the published sigma is the registration's own, floored, and the scale's share"
        " of the step (pepin.visual_odometry.SCALE_ERROR) in quadrature",
        on_when="never as such — it is a choice: 'constant' or 'rtabmap' while comparing on a"
        " tape, and with vo_publish off unless the point of the session is that comparison",
        off_when="'dynamic' is the shipping value; leave it there unless a session is about the"
        " covariance itself",
    ),
    # A pose of a frame taken while the head turned, or the body spun, and the first pose after
    # one, never reach the EKF: each becomes the anchor the next step is measured from, so the
    # pan rtabmap reads as a base yaw is never differenced into the filter (gaze.md 3.4).
    GAZE_GATE,
)


class VisualOdometry(Node):
    """Gates rtabmap's visual odometry, gives it a covariance and publishes it for the EKF."""

    def __init__(self) -> None:
        super().__init__("visual_odometry")
        # Read at start, before the switches (whose callback refuses anything that is not a flag).
        self._input = str(self.declare_parameter("vo_input", "stereo").value).strip()
        if self._input not in VO_INPUTS:
            raise ValueError(f"vo_input {self._input!r}: one of {', '.join(VO_INPUTS)}")
        self._switches = Switches(
            self, with_knobs(FLAGS, load_knobs("visual_odometry")), on_change=self._on_switch
        )
        if self._input == "vio" and self._switches["vo_covariance"] in ("dynamic", "rtabmap"):
            # OpenVINS's own covariance is the marginal of an unobservable pose (vio.md M3) and
            # there is no registration variance: the per-step model is the only honest one.
            self._switches.set("vo_covariance", "vio")
        self._gate = VoGate(
            max_speed_m_s=float(self._switches["vo_max_speed"]),
            max_turn_deg_s=float(self._switches["vo_max_turn"]),
            max_gap_s=float(self._switches["vo_max_gap_s"]),
            reset_radius_m=float(self._switches["vo_reset_radius_m"]),
        )
        self._track = VoTrack()
        self._cap = PublishCap(float(self._switches["vo_publish_hz"]))
        self._rest = RestWatch()
        self._tally = Tally()
        self._last_xy: tuple[float, float] | None = None  # the frame before this one, for its step
        self._drop: str | None = None  # the last reason, for the report line
        self._hold: str | None = None  # the last reason a pose was not published, for the same
        self._gaze = GazeFeed(
            self,
            exposure_s=float(self._switches["gate_exposure_s"]),
            settle_s=float(self._switches["gate_settle_s"]),
            yaw_dps=float(self._switches["gate_yaw_dps"]),
            stamp_end=float(self._switches["gate_stamp_end"]),
            sway_dps=float(self._switches["gate_sway_dps"]),
            sway_deg=float(self._switches["gate_sway_deg"]),
        )
        self._after_gate = False  # the last pose was gated: this one only anchors the next step
        # Both of these cross the bridge, so their QoS is not this node's to choose: it is
        # pinned on both sides in pepin.deployment.BRIDGED_QOS (reliable, ten deep — what the
        # board's EKF subscribes with and what base_bridge.cpp writes /odom with). /vo/raw never
        # leaves this laptop; rtabmap writes it RELIABLE.
        local = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(Odometry, VO_TOPIC, bridged_qos_profile(VO_TOPIC))
        self._vio_health = VioHealth()
        self._vio_lost = VioLost(
            lost_speed_m_s=float(self._switches["vio_lost_speed_m_s"]),
            lost_s=float(self._switches["vio_lost_s"]),
            min_features=int(self._switches["vio_min_features"]),
        )
        self._tf: TfLookup | None = None
        self._tf_failure: str | None = None
        self._last_vio: VoPose | None = None  # the previous composed pose, for the VIO's speed
        self._last_vio_xy: tuple[float, float] | None = None  # the last published, for its step
        source = RAW_TOPIC
        if self._input == "vio":
            source = VIO_POSE_TOPIC
            self._tf = TfLookup(self, on_failure=self._on_tf_failure)
            self.create_subscription(PoseWithCovarianceStamped, VIO_POSE_TOPIC, self._on_vio, local)
            self.create_subscription(PointCloud2, VIO_POINTS_TOPIC, self._on_points, local)
            self.create_subscription(
                Odometry,
                ZUPT_TOPIC,
                self._on_zupt,
                QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE),
            )
        else:
            self.create_subscription(Odometry, RAW_TOPIC, self._on_vo, local)
        self.create_subscription(
            Odometry, WHEELS_TOPIC, self._on_wheels, bridged_qos_profile(WHEELS_TOPIC)
        )
        self.create_timer(REPORT_S, self._report)
        self.get_logger().info(
            f"visual odometry up: vo_input {self._input}, {source} -> {VO_TOPIC} for the board's"
            f" EKF (x, y and yaw, differentially), rest drift measured against"
            f" {WHEELS_TOPIC}; flags: {self._switches.state()}"
        )

    def close(self) -> None:
        """Stop the TF listener's thread (vo_input vio) before the node is destroyed."""
        if self._tf is not None:
            self._tf.close()

    # ---- inputs ------------------------------------------------------------------------------
    def _on_switch(self, name: str, _old: object, new: object) -> None:
        """A flag changed: the three ceilings are the gate's, the rest are read where they act."""
        if name == "vo_max_speed":
            self._gate.max_speed_m_s = float(new)  # type: ignore[arg-type]
        elif name == "vo_max_turn":
            self._gate.max_turn_deg_s = float(new)  # type: ignore[arg-type]
        elif name == "vo_max_gap_s":
            self._gate.max_gap_s = float(new)  # type: ignore[arg-type]
        elif name == "vo_reset_radius_m":
            self._gate.reset_radius_m = float(new)  # type: ignore[arg-type]
        elif name == "vo_publish_hz":
            self._cap.hz = float(new)  # type: ignore[arg-type]
        elif name == "vo_covariance" and self._input == "vio" and new in ("dynamic", "rtabmap"):
            raise ValueError(
                f"vo_input vio has no registration variance and OpenVINS's own covariance is the"
                f" marginal of an unobservable pose: {new} is refused, vio or constant"
            )
        elif name == "vio_lost_speed_m_s":
            self._vio_lost.lost_speed_m_s = float(new)  # type: ignore[arg-type]
        elif name == "vio_lost_s":
            self._vio_lost.lost_s = float(new)  # type: ignore[arg-type]
        elif name == "vio_min_features":
            self._vio_lost.min_features = int(new)  # type: ignore[call-overload]
        elif name in GATE_KNOBS:
            self._gaze.set(name, float(new))  # type: ignore[arg-type]
        elif name == "gaze_gate":
            self._after_gate = False

    def _on_wheels(self, msg: Odometry) -> None:
        """The board's wheel odometry: only its twist is read, and only to know whether the cart
        is standing still (the drift at rest is what says this source may be fused at all).

        Timed by this node's own clock, not by the message's stamp: this one is the board's and
        the visual poses are the laptop's (:class:`pepin.visual_odometry.RestWatch`).
        """
        now = time.monotonic()
        self._rest.wheels(now, float(msg.twist.twist.linear.x), float(msg.twist.twist.angular.z))
        self._vio_lost.wheels(now, float(msg.twist.twist.linear.x))

    def _on_vo(self, msg: Odometry) -> None:
        """One pose from rgbd_odometry: gated, measured at rest, and published with the
        covariance the flags chose."""
        self._tally.count("in")
        pose = VoPose(
            stamp=stamp_seconds(msg.header.stamp),
            x=float(msg.pose.pose.position.x),
            y=float(msg.pose.pose.position.y),
            yaw=yaw_of(msg.pose.pose.orientation),
        )
        lost = is_lost(msg.pose.covariance)
        published = self._admit(pose, lost)
        if published is None:
            return
        _write_planar_pose(msg, published)
        mode = self._switches["vo_covariance"]
        if mode == "constant":
            msg.pose.covariance = planar_covariance(
                float(self._switches["vo_sigma_m"]), float(self._switches["vo_yaw_sigma_deg"])
            )
        elif mode == "dynamic":
            msg.pose.covariance = scaled_covariance(
                float(msg.pose.covariance[0]),
                self._step_since_last(msg),
                float(self._switches["vo_yaw_sigma_deg"]),
            )
        elif mode == "vio":
            msg.pose.covariance = self._vio_covariance(self._step_since_last(msg))
        self._pub.publish(msg)
        self._tally.count("out")

    def _admit(self, pose: VoPose, lost: bool) -> VoPose | None:
        """The shared path of every input: the gaze gate, the jump gate, the track, the rest
        watch, the publish flag and the cap. Returns the track's pose to publish, or ``None``."""
        if self._switches.on("gaze_gate") and self._gated(pose, lost):
            return None
        refused = self._gate.admit(pose, lost)
        if refused is not None:
            self._tally.count("dropped")
            self._drop = refused
            # The gate's anchor moved; the track's must move with it, or the jump the gate just
            # refused would reach the EKF inside the next pose that passes.
            self._track.anchor(self._gate.anchor)
            self._rest.restart()  # a drift measured across a re-initialised origin is not one
            return None
        published = self._track.advance(pose)
        self._rest.pose(pose, time.monotonic())
        if not self._switches.on("vo_publish"):
            self._tally.count("withheld")
            return None
        held = self._cap.refuse(published.stamp)
        if held is not None:
            self._tally.count("skipped")
            self._hold = held
            return None
        return published

    def _vio_covariance(self, step_m: float) -> list[float]:
        """The `vio` mode: vio_step_fraction of the step, floored at vo_sigma_m (vio.md M3)."""
        return scaled_covariance(
            0.0,
            step_m,
            float(self._switches["vo_yaw_sigma_deg"]),
            scale_error=float(self._switches["vio_step_fraction"]),
            floor_sigma_m=float(self._switches["vo_sigma_m"]),
        )

    # ---- the visual-inertial input -------------------------------------------------------------
    def _on_vio(self, msg: PoseWithCovarianceStamped) -> None:
        """One OpenVINS pose: health, base_link composed through TF at its stamp, lost rules,
        then the shared gate and track; published as an Odometry in rtabmap's frames."""
        self._tally.count("in")
        stamp = stamp_seconds(msg.header.stamp)
        if self._vio_health.observe(msg.pose.covariance):
            self._tally.count("vio_reinit")
            self._last_vio = None
            self._reanchor(None)  # a new gravity frame: nothing differences across it
        assert self._tf is not None
        transform = self._tf.transform(IMU_FRAME, BASE_FRAME, msg.header.stamp, TF_TIMEOUT_S)
        if transform is None:
            # Skipped, not re-anchored: the next composed pose's step from the last composed one
            # is still a step between two poses that each had their neck chain.
            self._tally.count("tf_miss")
            return
        t_i_b = pose_from_transform(transform)
        q, v = msg.pose.pose.orientation, msg.pose.pose.position
        t_g_i = homogeneous(rotation_matrix(q.x, q.y, q.z, q.w), (v.x, v.y, v.z))
        pose = compose_base_pose(t_g_i, homogeneous(t_i_b.rotation, t_i_b.translation), stamp)
        speed = self._vio_speed(pose)
        if self._vio_lost.check(time.monotonic(), speed) is not None:
            self._tally.count("vio_lost")
            self._reanchor(pose)
            return
        published = self._admit(pose, lost=False)
        if published is None:
            return
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = ODOM_FRAME
        out.child_frame_id = BASE_FRAME
        _write_planar_pose(out, published)
        here = (pose.x, pose.y)
        there, self._last_vio_xy = self._last_vio_xy, here
        step = 0.0 if there is None else math.hypot(here[0] - there[0], here[1] - there[1])
        if self._switches["vo_covariance"] == "constant":
            out.pose.covariance = planar_covariance(
                float(self._switches["vo_sigma_m"]), float(self._switches["vo_yaw_sigma_deg"])
            )
        else:
            out.pose.covariance = self._vio_covariance(step)
        self._pub.publish(out)
        self._tally.count("out")

    def _vio_speed(self, pose: VoPose) -> float:
        """The composed base speed since the previous composed pose (m/s); 0 for the first."""
        last, self._last_vio = self._last_vio, pose
        if last is None or pose.stamp <= last.stamp:
            return 0.0
        return math.hypot(pose.x - last.x, pose.y - last.y) / (pose.stamp - last.stamp)

    def _reanchor(self, pose: VoPose | None) -> None:
        """A pose nothing may be differenced across (a lost VIO, a re-init): the gate and the
        track start the next step from it, and the rest watch starts again."""
        if pose is not None:
            self._gate.reanchor(pose)
            self._track.anchor(self._gate.anchor)
        else:
            self._track.anchor(None)
        self._rest.restart()

    def _on_points(self, msg: PointCloud2) -> None:
        """The features of OpenVINS's last update: their count is lost rule (c)."""
        self._vio_lost.features(time.monotonic(), int(msg.width) * int(msg.height))

    def _on_zupt(self, _msg: Odometry) -> None:
        """The board says the cart is at rest (wheels, gyro and command witnessed)."""
        self._vio_lost.zupt(time.monotonic())

    def _on_tf_failure(self, kind: str, text: str) -> None:
        self._tf_failure = f"{kind}: {text}"

    def _gated(self, pose: VoPose, lost: bool) -> bool:
        """Whether this pose is withheld by the gaze gate: its frame was taken while the head
        turned or the body spun, or it is the first pose after such a frame (its step starts on
        one). It then only anchors the next step — the gate's and the track's alike, so nothing
        of the motion across it is differenced into the EKF — and the rest watch starts again. A
        lost pose anchors nothing, and the pose after it is withheld in its place."""
        verdict = self._gaze.verdict(pose.stamp)
        if verdict is None and not self._after_gate:
            return False
        self._tally.count(f"gaze_{verdict or 'after'}")
        self._after_gate = verdict is not None or lost
        if not lost:
            self._gate.reanchor(pose)
            self._track.anchor(self._gate.anchor)
        self._rest.restart()
        return True

    def _step_since_last(self, msg: Odometry) -> float:
        """How far the visual odometry says the cart moved since the frame before this one, in
        metres — the step whose metres carry the depth scale's error. The first frame is 0.0."""
        here = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        there = self._last_xy
        self._last_xy = here
        if there is None:
            return 0.0
        return float(math.hypot(here[0] - there[0], here[1] - there[1]))

    # ---- the report --------------------------------------------------------------------------
    def _report(self) -> None:
        """The window's rates, what was dropped and why, the drift at rest and the flags."""
        w = self._tally.take()
        c = w.counts
        drop = f" (last: {self._drop})" if self._drop else ""
        hold = f" (last: {self._hold})" if self._hold else ""
        x, y, _ = self._track.pose
        source = "OpenVINS" if self._input == "vio" else "rtabmap"
        self.get_logger().info(
            f"vo: {w.rate('in'):.1f} poses/s from {source}, {w.rate('out'):.1f} published,"
            f" {c['dropped']} dropped{drop}, {c['skipped']} skipped{hold},"
            f" {c['withheld']} withheld from the EKF; track at ({x:.2f}, {y:.2f});"
            f" {self._rest.report()}; {self._gate_text(c)};{self._vio_text(c)}"
            f" flags: {self._switches.state()}"
        )
        if c["in"] == 0 and self._input == "vio":
            self.get_logger().warning(
                f"no pose on {VIO_POSE_TOPIC} in this window: is OpenVINS up (ros/laptop.sh vio),"
                " is /head/imu flowing (the board's head_imu:=true) and has it initialised (a"
                " head pan or a roll-off triggers it)?"
            )
        elif c["in"] == 0:
            self.get_logger().warning(
                f"no pose on {RAW_TOPIC} in this window: is rgbd_odometry running (vslam.launch.py"
                " vo:=true) and is /camera/depth alive (the depth law needs the lidar)?"
            )

    def _vio_text(self, counts: Counter[str]) -> str:
        """The visual-inertial input's health for the report line; nothing for rtabmap's."""
        if self._input != "vio":
            return ""
        tf = f" (last: {self._tf_failure})" if counts["tf_miss"] and self._tf_failure else ""
        return (
            f" vio: reinit {counts['vio_reinit']} ({self._vio_health.reinits} since the start),"
            f" {counts['vio_lost']} poses withheld as lost, {self._vio_lost.report()},"
            f" tf_miss {counts['tf_miss']}{tf}; kick at rest: ros/laptop.sh vio kick;"
        )

    def _gate_text(self, counts: Counter[str]) -> str:
        """The poses the gaze gate withheld this window (and the ones after them), the head."""
        if not self._switches.on("gaze_gate"):
            return "gaze gate off"
        return (
            f"gaze gate: {gate_counts(counts, counts['in'])} withheld, {counts['gaze_after']}"
            f" after them; {self._gaze.text()}"
        )


def _write_planar_pose(msg: Odometry, pose: VoPose) -> None:
    """Put a planar pose into an ``Odometry`` message in place: x, y and yaw, the floor at z=0."""
    msg.pose.pose.position.x = pose.x
    msg.pose.pose.position.y = pose.y
    msg.pose.pose.position.z = 0.0
    q = msg.pose.pose.orientation
    q.x, q.y, q.z, q.w = 0.0, 0.0, math.sin(pose.yaw / 2.0), math.cos(pose.yaw / 2.0)


def main() -> None:
    spin_main(VisualOdometry)


if __name__ == "__main__":
    main()
