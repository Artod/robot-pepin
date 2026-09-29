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
"""

from __future__ import annotations

import math
import time

from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from pepin.flags import Flag, FlagSet, load_knobs, with_knobs
from pepin.visual_odometry import (
    PublishCap,
    RestWatch,
    VoGate,
    VoPose,
    VoTrack,
    is_lost,
    planar_covariance,
    scaled_covariance,
)
from pepin_bringup.msgs import stamp_seconds, yaw_of
from pepin_bringup.node_kit import Switches, Tally, bridged_qos_profile, spin_main

RAW_TOPIC = "/vo/raw"  # rgbd_odometry's own output, on this laptop only
VO_TOPIC = "/vo"  # what crosses to the board's EKF
WHEELS_TOPIC = "/odom"  # the board's wheel odometry: the rest signal, over the bridge
REPORT_S = 30.0

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
        choices=("dynamic", "constant", "rtabmap"),
        description="whose covariance rides on the published pose: `dynamic`, the registration's"
        " own sigma and the depth scale's share of the step just taken added in quadrature"
        " (pepin.visual_odometry.scaled_covariance); the documented constant (vo_sigma_m,"
        " vo_yaw_sigma_deg); or the one rtabmap's registration computed, untouched",
        why="the constant, because rtabmap's own number answers the wrong question. Measured at"
        " rest on this robot (2026-09-14, scratch/vo_probe.py, 85 s): its registration claimed a"
        " position standard deviation of 3.8 mm at the median and 15.9 mm at p90 — an honest"
        " spread of the feature matches, and a claim about the PICTURE. The error that matters"
        " is the scale of the depth those features sit on, and that scale is a network's law"
        " fitted against the lidar (0.94 to 1.98 across one afternoon, 2026-09-11), which no"
        " registration can see. So the topic carries a constant a person can argue with, and"
        " rtabmap's own is one flag away for the session that wants to compare them — one flag"
        " away and worth reading twice before it is turned on with vo_publish: 3.8 mm through"
        " robot_localization's differential conversion (2 * sigma^2 * dt) is a velocity sigma of"
        " 1.8 mm/s, 325x the wheels' certainty per sample, which is no longer a third opinion"
        " but the whole odometry (scratch/vo_weight.py)",
        on_when="never as such — it is a choice: 'rtabmap' while comparing the two on a tape,"
        " and with vo_publish off unless the point of the session is that comparison",
        off_when="'constant' is the shipping value; leave it there unless a session is about the"
        " covariance itself",
    ),
)


class VisualOdometry(Node):
    """Gates rtabmap's visual odometry, gives it a covariance and publishes it for the EKF."""

    def __init__(self) -> None:
        super().__init__("visual_odometry")
        self._switches = Switches(
            self, with_knobs(FLAGS, load_knobs("visual_odometry")), on_change=self._on_switch
        )
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
        # Both of these cross the bridge, so their QoS is not this node's to choose: it is
        # pinned on both sides in pepin.deployment.BRIDGED_QOS (reliable, ten deep — what the
        # board's EKF subscribes with and what base_bridge.cpp writes /odom with). /vo/raw never
        # leaves this laptop; rtabmap writes it RELIABLE.
        local = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(Odometry, VO_TOPIC, bridged_qos_profile(VO_TOPIC))
        self.create_subscription(Odometry, RAW_TOPIC, self._on_vo, local)
        self.create_subscription(
            Odometry, WHEELS_TOPIC, self._on_wheels, bridged_qos_profile(WHEELS_TOPIC)
        )
        self.create_timer(REPORT_S, self._report)
        self.get_logger().info(
            f"visual odometry up: {RAW_TOPIC} -> {VO_TOPIC} for the board's EKF (x and y,"
            f" differentially; no yaw — the gyro owns heading), rest drift measured against"
            f" {WHEELS_TOPIC}; flags: {self._switches.state()}"
        )

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

    def _on_wheels(self, msg: Odometry) -> None:
        """The board's wheel odometry: only its twist is read, and only to know whether the cart
        is standing still (the drift at rest is what says this source may be fused at all).

        Timed by this node's own clock, not by the message's stamp: this one is the board's and
        the visual poses are the laptop's (:class:`pepin.visual_odometry.RestWatch`).
        """
        self._rest.wheels(
            time.monotonic(),
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.angular.z),
        )

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
        refused = self._gate.admit(pose, is_lost(msg.pose.covariance))
        if refused is not None:
            self._tally.count("dropped")
            self._drop = refused
            # The gate's anchor moved; the track's must move with it, or the jump the gate just
            # refused would reach the EKF inside the next pose that passes.
            self._track.anchor(self._gate.anchor)
            self._rest.restart()  # a drift measured across a re-initialised origin is not one
            return
        published = self._track.advance(pose)
        self._rest.pose(pose, time.monotonic())
        if not self._switches.on("vo_publish"):
            self._tally.count("withheld")
            return
        held = self._cap.refuse(published.stamp)
        if held is not None:
            self._tally.count("skipped")
            self._hold = held
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
        self._pub.publish(msg)
        self._tally.count("out")

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
        self.get_logger().info(
            f"vo: {w.rate('in'):.1f} poses/s from rtabmap, {w.rate('out'):.1f} published,"
            f" {c['dropped']} dropped{drop}, {c['skipped']} skipped{hold},"
            f" {c['withheld']} withheld from the EKF; track at ({x:.2f}, {y:.2f});"
            f" {self._rest.report()}; flags: {self._switches.state()}"
        )
        if c["in"] == 0:
            self.get_logger().warning(
                f"no pose on {RAW_TOPIC} in this window: is rgbd_odometry running (vslam.launch.py"
                " vo:=true) and is /camera/depth alive (the depth law needs the lidar)?"
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
