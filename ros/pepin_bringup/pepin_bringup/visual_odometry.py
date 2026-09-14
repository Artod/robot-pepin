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

The flags (:data:`FLAGS`, ``ros/flags.sh set visual_odometry <flag> <value>``): ``vo_publish``
(whether the measured odometry leaves this laptop at all — off, the EKF is exactly what it was
before this node existed), ``vo_covariance`` (the constant or rtabmap's own), ``vo_sigma_m`` and
``vo_yaw_sigma_deg`` (the constant), ``vo_max_speed`` and ``vo_max_turn`` (the gate's ceilings).
"""

from __future__ import annotations

import time

from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from pepin.flags import Flag, FlagSet
from pepin.visual_odometry import RestWatch, VoGate, VoPose, is_lost, planar_covariance
from pepin_bringup.msgs import stamp_seconds, yaw_of
from pepin_bringup.node_kit import Switches, Tally, spin_main

RAW_TOPIC = "/vo/raw"  # rgbd_odometry's own output, on this laptop only
VO_TOPIC = "/vo"  # what crosses to the board's EKF
WHEELS_TOPIC = "/odom"  # the board's wheel odometry: the rest signal, over the bridge
REPORT_S = 30.0

FLAGS = FlagSet(
    Flag(
        "vo_publish",
        False,
        description=f"the gated visual odometry leaves this laptop as {VO_TOPIC}, where the"
        " board's EKF fuses it as a third input beside the wheels and the gyro; off, the node"
        " still measures and reports and the EKF is exactly what it was without it",
        why="off because the half that matters is unmeasured. AT REST it is measured and it"
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
        "constant",
        choices=("constant", "rtabmap"),
        description="whose covariance rides on the published pose: the documented constant"
        " (vo_sigma_m, vo_yaw_sigma_deg) or the one rtabmap's registration computed",
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
    Flag(
        "vo_sigma_m",
        0.07,
        range=(0.001, 1.0),
        description="the constant position sigma of one visual-odometry pose, in metres; the EKF"
        " differences two of them into a velocity and the covariance rides along — as"
        " (this pose's + the previous pose's) TIMES the gap, so what the filter actually weighs"
        " is a velocity variance of 2 * sigma^2 * dt",
        why="7 cm is not a claim about the registration — it is the sigma at which the wheels"
        " stay dominant once robot_localization has done its arithmetic, which is the shape this"
        " source was designed to have. That arithmetic is not the obvious one: the differential"
        " path multiplies the summed pose covariance BY the gap (ros_filter.cpp 3249-3257,"
        " jazzy-devel) instead of dividing by its square, so at 9.4 poses/s a 2 cm pose sigma"
        " becomes a velocity sigma of 0.9 cm/s — against the wheels' own 3.2 cm/s"
        " (pepin_base_cpp/protocol.hpp, 0.001 m^2/s^2 on vx) that is 11.8x their certainty per"
        " sample and 5.5x their information per second, i.e. the camera would BE the odometry"
        " (scratch/vo_weight.py). At 7 cm the same conversion gives 3.2 cm/s: one camera sample"
        " is worth one wheel sample and, at 9.4 Hz against 20 Hz, the camera carries 45 % of the"
        " wheels' information — a third opinion that can pull the filter when a wheel slips, on"
        " top of a distance the wheels are honest about to 3 % (2026-09-06). Tighten it only"
        " against a drive where a lidar-measured distance says who was right",
        on_when="not a switch: raise it when the visual odometry argues with the wheels on a"
        " drive where the wheels were right, lower it when it was right and was not heard",
        off_when="not a switch",
    ),
    Flag(
        "vo_yaw_sigma_deg",
        5.0,
        range=(0.1, 180.0),
        description="the constant yaw sigma of one visual-odometry pose, in degrees; the board's"
        " EKF does not fuse yaw from this source at all, so it is carried for whoever reads the"
        " message rather than for the filter",
        why="unmeasured on purpose, because nothing fuses it: the gyro owns heading — with it"
        " the EKF's turn error is ~5 % against the wheels' 40-70 % (2026-09-13) — and"
        " ros/params/ekf.yaml fuses no"
        " yaw from this topic. 5 degrees is a deliberately weak claim so that a future consumer"
        " of /vo cannot mistake this for a heading source",
        on_when="not a switch",
        off_when="not a switch",
    ),
    Flag(
        "vo_max_speed",
        1.0,
        range=(0.05, 10.0),
        description="a step between two visual-odometry poses faster than this, in m/s, is"
        " dropped: rtabmap restarting its tracking moves the pose without moving the cart",
        why="1.0 m/s is over three times the fastest this cart can go — the base's own cap is"
        " 0.30 m/s (pepin.deployment's BASE_MAX_LINEAR_M_S) and the C++ bridge clamps /cmd_vel"
        " at 0.25 — and 26 times the largest step this source took at rest, where 85 s of poses"
        " were at most 4.2 mm apart over ~0.11 s, a median of 1.0 mm (2026-09-14,"
        " scratch/vo_probe.py). So it cannot refuse a real motion and still refuses the"
        " metre-scale jump a re-initialised visual odometry publishes — which, differenced into"
        " a velocity, is the one thing that could move the odom frame",
        on_when="not a switch: lower it towards 0.4 m/s on a tape where the camera argued with"
        " the wheels about the speed itself",
        off_when="not a switch",
    ),
    Flag(
        "vo_max_turn",
        180.0,
        range=(5.0, 720.0),
        description="a turn between two visual-odometry poses faster than this, in deg/s, is"
        " dropped, for the same reason as vo_max_speed",
        why="180 deg/s is three times the base's own angular cap of 1.0 rad/s = 57 deg/s"
        " (pepin.deployment's BASE_MAX_ANGULAR_RAD_S) and some two thousand times what this"
        " source turned at rest (0.075 deg over 85 s, 2026-09-14, scratch/vo_probe.py): it"
        " catches a tracking restart and nothing a cart could do",
        on_when="not a switch",
        off_when="not a switch",
    ),
)


class VisualOdometry(Node):
    """Gates rtabmap's visual odometry, gives it a covariance and publishes it for the EKF."""

    def __init__(self) -> None:
        super().__init__("visual_odometry")
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
        self._gate = VoGate(
            max_speed_m_s=float(self._switches["vo_max_speed"]),
            max_turn_deg_s=float(self._switches["vo_max_turn"]),
        )
        self._rest = RestWatch()
        self._tally = Tally()
        self._drop: str | None = None  # the last reason, for the report line
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(Odometry, VO_TOPIC, reliable)
        self.create_subscription(Odometry, RAW_TOPIC, self._on_vo, reliable)
        self.create_subscription(Odometry, WHEELS_TOPIC, self._on_wheels, reliable)
        self.create_timer(REPORT_S, self._report)
        self.get_logger().info(
            f"visual odometry up: {RAW_TOPIC} -> {VO_TOPIC} for the board's EKF (x and y,"
            f" differentially; no yaw — the gyro owns heading), rest drift measured against"
            f" {WHEELS_TOPIC}; flags: {self._switches.state()}"
        )

    # ---- inputs ------------------------------------------------------------------------------
    def _on_switch(self, name: str, _old: object, new: object) -> None:
        """A flag changed: the two ceilings are the gate's, the rest are read where they act."""
        if name == "vo_max_speed":
            self._gate.max_speed_m_s = float(new)  # type: ignore[arg-type]
        elif name == "vo_max_turn":
            self._gate.max_turn_deg_s = float(new)  # type: ignore[arg-type]

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
            self._rest.restart()  # a drift measured across a re-initialised origin is not one
            return
        self._rest.pose(pose, time.monotonic())
        if not self._switches.on("vo_publish"):
            self._tally.count("withheld")
            return
        if self._switches["vo_covariance"] == "constant":
            msg.pose.covariance = planar_covariance(
                float(self._switches["vo_sigma_m"]), float(self._switches["vo_yaw_sigma_deg"])
            )
        self._pub.publish(msg)
        self._tally.count("out")

    # ---- the report --------------------------------------------------------------------------
    def _report(self) -> None:
        """The window's rates, what was dropped and why, the drift at rest and the flags."""
        w = self._tally.take()
        c = w.counts
        drop = f" (last: {self._drop})" if self._drop else ""
        self.get_logger().info(
            f"vo: {w.rate('in'):.1f} poses/s from rtabmap, {w.rate('out'):.1f} published,"
            f" {c['dropped']} dropped{drop}, {c['withheld']} withheld from the EKF;"
            f" {self._rest.report()}; flags: {self._switches.state()}"
        )
        if c["in"] == 0:
            self.get_logger().warning(
                f"no pose on {RAW_TOPIC} in this window: is rgbd_odometry running (vslam.launch.py"
                " vo:=true) and is /camera/depth alive (the depth law needs the lidar)?"
            )


def main() -> None:
    spin_main(VisualOdometry)


if __name__ == "__main__":
    main()
