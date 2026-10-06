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
is published; ``vio_publish_hz`` under vo_input vio). The published pose is the sum of the steps
the gate admitted, never rtabmap's own: a refused jump re-anchors the gate, and a filter that
differences the stream it receives would otherwise get the whole discontinuity in one frame time
(2026-09-14, odom -> base_link 43 km out).

The gaze gate (``gaze_gate`` and the ``gate_*`` knobs, :mod:`pepin.gaze_gate`) withholds the
same way: a pose of a frame taken during a head saccade (``/gaze/state``) or a body yaw above
``gate_yaw_dps``, and the first pose after one, only re-anchor — the EKF coasts on the wheels and
the gyro across them.

THE INPUT is the ``vo_input`` flag, passed at start by vslam.launch.py and live between the
launched rtabmap input and ``vio`` (``ros/flags.sh set visual_odometry vo_input vio``; back with
``stereo``): ``stereo`` or ``depth`` read rtabmap's ``/vo/raw`` as above; ``vio`` reads OpenVINS's
``/ov_msckf/poseimu`` (the head IMU's pose in its own gravity frame, per image update, stamped
camera time + t_d; it runs in its own container, ros/laptop.sh vio) and composes base_link's pose
through TF at the pose's own stamp (``head_imu <- base_link``: camera_stream's static head_imu
edge, the board's neck chain; :func:`pepin.visual_odometry.compose_base_pose`). Then the same
gate, track, cap and EKF slot. A switch restarts the gate and the track's anchor; the published
track carries on from where it stood, so the EKF differences no jump across it. Its covariance is
the ``vio`` mode, a per-step model floored at ``vo_sigma_m`` (OpenVINS's own marginal covariance
only grows: it is read as a health signal, re-inits counted); under ``vio`` the ``dynamic`` and
``rtabmap`` choices read as ``vio``.

THE OUTPUT under vio is the ``vo_output`` flag: ``pose`` (the default) is the track above on
``/vo`` with the per-step covariance, which the EKF differences (odom1); ``twist`` sends the same
admitted step as a BODY VELOCITY on ``/vo_twist`` (vx, vy, vyaw: the SE(2) logarithm of the step
between two composed base poses, so the neck's own motion stays subtracted, over their stamps'
difference; :func:`pepin.visual_odometry.se2_twist`), weighed by OpenVINS's OWN velocity
covariance from ``/ov_msckf/odomimu`` at the pose's stamp, turned into base_link's axes
(:func:`pepin.visual_odometry.base_twist_covariance`), no floor and no constant; ``/vo`` then
carries the track with :data:`pepin.visual_odometry.WEIGHTLESS_VARIANCE` so the filter keeps
differencing it without weight, and nothing is counted twice. A sample whose covariance is
missing or not a covariance (non-finite, not positive definite) is withheld and counted. The
covariance's median sigmas are in the report line under either output, so the switch is
measured before it is made. Under rtabmap's inputs the output is ``pose``.

THE TWIST'S SOURCE is ``vio_twist_source``: ``step`` (above, one per camera frame) or ``imu``,
odomimu's own velocity state and bias-corrected gyro carried into base_link through the neck chain
(:func:`pepin.visual_odometry.imu_base_twist`) at ``vio_publish_hz`` (50, the wheels' rate), each
sample processed :data:`pepin.visual_odometry.VIO_IMU_LAG_S` behind the newest so its TF is
already there, judged by the last poseimu's verdict and by the gaze gate at its own stamp. Both
sources carry ``vio_sigma_scale`` on vx and vy and ``vio_yaw_sigma_scale`` on vyaw (1.0: no
fudge until drives measure one). LOST SPLITS THE TWIST: while the lost rules hold, vx and vy are
withheld (``WEIGHTLESS_VARIANCE`` on both, 0 as the value: robot_localization's twist0 fuses them
with a gain of ~1e-8) and the yaw rate keeps flowing -- the gyro needs no light -- with its
variance grown by the bias walk since the last visual update,
``sigma_yaw(t)^2 = (vio_yaw_sigma_scale * sigma_reported)^2 + gyro_random_walk^2 * t``
(:class:`pepin.visual_odometry.YawOnly`, the walk from config/head_imu.json), unless the mast
sways beyond ``gate_sway_dps`` / ``gate_sway_deg`` or the gaze gate holds the stamp (withheld and
counted). The guard's divergence still withholds everything.

Every composed sample first passes the plausibility guard (:class:`pepin.visual_odometry.VioGuard`:
a base velocity over ``vio_max_speed_m_s`` is not sent, nor, while ``vio_guard`` is on, one
``vio_wheel_diff_m_s`` from the wheels'), then the three lost rules of
:class:`pepin.visual_odometry.VioLost` (the wheels' speed over time, motion under ``/zupt``, the
features of ``/ov_msckf/points_msckf`` + ``points_slam`` short for ``vio_lost_s`` while the base
moves, never at rest); a refused sample only re-anchors. OpenVINS never resets itself: after
``vio_restart_rejects`` guard rejections in a row with the wheels at rest for 2 s this node calls
``/vio/restart`` (pepin_bringup.vio_keeper in pepin-vio), logged once; by hand it is
``ros/laptop.sh vio kick``, at rest. The report line adds the samples in, out and refused, the
stamp-to-receipt latency, whether the board's EKF subscribes ``/vo``, and the EKF's own odom ->
base_link (once the VIO input has been on: the TF listener).
"""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter
from typing import Any

from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger

from pepin.depth import rotation_matrix
from pepin.flags import Flag, FlagSet, load_knobs, with_knobs
from pepin.gaze_gate import GATE_KNOBS, GAZE_GATE, SWAYING
from pepin.head_imu import HeadImuConfig
from pepin.mounts import load_camera_mounts
from pepin.visual_odometry import (
    ImuQueue,
    ImuState,
    NeckBaseline,
    PublishCap,
    RestWatch,
    TwistCovariances,
    VioGuard,
    VioHealth,
    VioLost,
    VoGate,
    VoPose,
    VoTrack,
    YawOnly,
    base_twist_covariance,
    body_velocity,
    compose_base_pose,
    homogeneous,
    imu_base_twist,
    is_lost,
    planar_covariance,
    scaled_covariance,
    scaled_twist_covariance,
    se2_twist,
    weightless_covariance,
    yaw_only_covariance,
)
from pepin_bringup.gaze_feed import GazeFeed, gate_counts
from pepin_bringup.msgs import (
    camera_edges,
    pose_from_transform,
    stamp_from_seconds,
    stamp_seconds,
    yaw_of,
)
from pepin_bringup.node_kit import Switches, Tally, TfLookup, bridged_qos_profile, spin_main

RAW_TOPIC = "/vo/raw"  # rgbd_odometry's own output, on this laptop only
VO_TOPIC = "/vo"  # what crosses to the board's EKF
VO_TWIST_TOPIC = "/vo_twist"  # vo_output twist: the EKF's twist0 (pepin.deployment.VO_TWIST_TOPIC)
VO_OUTPUTS = ("pose", "twist")
# Where vo_output twist takes its velocity: `step`, the composed poseimu step (one per camera
# frame); `imu`, odomimu's velocity state and gyro at vio_publish_hz through the neck.
VIO_TWIST_SOURCES = ("step", "imu")
WHEELS_TOPIC = "/odom"  # the board's wheel odometry: the rest signal, over the bridge
REPORT_S = 30.0
# The vo_input parameter: what the relay reads. stereo/depth: rtabmap's /vo/raw (the launch
# starts stereo_odometry or rgbd_odometry beside it); vio: OpenVINS in its own container.
VO_INPUTS = ("stereo", "depth", "vio")
VIO_POSE_TOPIC = "/ov_msckf/poseimu"  # the IMU's pose in G per image update (not odomimu)
# OpenVINS's fast-propagated state per IMU sample (published only while subscribed, from 1 s after
# init): read for its TWIST covariance alone, the IMU's velocity and angular rate in its own axes.
# Its pose is an IMU-only prediction since the last update, not poseimu's filtered estimate at the
# camera time, so poseimu stays the composition's input and the health signal.
VIO_ODOM_TOPIC = "/ov_msckf/odomimu"
# The features of OpenVINS's last update, summed for lost rule (c): the MSCKF ones (tracks that
# ended, none at rest) and the SLAM ones in its state (kept at rest); both come from one call per
# update (ROS2Visualizer::publish_features).
VIO_POINTS_TOPIC = "/ov_msckf/points_msckf"
VIO_SLAM_POINTS_TOPIC = "/ov_msckf/points_slam"
ZUPT_TOPIC = "/zupt"  # the board's zero-velocity update: published only while it says rest
IMU_FRAME = "head_imu"  # camera_stream's static camera_optical -> head_imu (config/camera.json)
BASE_FRAME = "base_link"
ODOM_FRAME = "odom"  # the frame rtabmap's /vo/raw carries (odom_frame_id), kept for the EKF
TF_TIMEOUT_S = 0.05  # how long a pose waits for the neck chain at its stamp
VIO_RESTART_SERVICE = "/vio/restart"  # pepin_bringup.vio_keeper, in pepin-vio beside OpenVINS
EKF_NODE = "ekf_filter_node"  # the board's robot_localization (robot.launch.py), /vo's reader

FLAGS = FlagSet(
    Flag(
        "vo_input",
        "stereo",
        choices=VO_INPUTS,
        description="what the relay reads: `stereo` or `depth`, rtabmap's /vo/raw from the"
        " odometry node vslam.launch.py started (its own vo_input argument); `vio`, OpenVINS's"
        " /ov_msckf/poseimu (ros/laptop.sh vio) composed into base_link through the neck's TF,"
        " every sample past the plausibility guard (vio_max_speed_m_s, vio_wheel_diff_m_s) and"
        " OpenVINS restarted at rest after vio_restart_rejects rejections in a row. Live between"
        " the launched rtabmap input and vio: the gate and the track's anchor restart, the"
        " published track carries on",
        why="stereo since 2026-10-02 (the night deploy). vio is measured head-only on the parked"
        " cart (2026-10-04, scratch/vio_day/live1, live2): orientation inside slow head moves"
        " 0.8-1.6 deg, translation error 1.0-1.4 cm p50, rest drift 0.15 cm/min and 0.064"
        " deg/min; the first fast pan diverged it (7.7 m, then km) and it never re-initialised,"
        " which the guard and the restart exist for. Never in motion, never into the EKF before"
        " this flag",
        on_when="a static test or a drive whose point is the VIO: ros/laptop.sh vio up and"
        " initialised (one head pan), the guard's line in the report reading 0 rejected at rest",
        off_when="stereo whenever odom -> base_link must be today's: a drive that measures"
        " something else, a VIO whose guard keeps rejecting, OpenVINS down",
    ),
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
    Flag(
        "vo_output",
        "pose",
        choices=VO_OUTPUTS,
        description=f"how the VIO reaches the board's EKF: `pose`, the admitted track on {VO_TOPIC}"
        " with the vo_covariance model, differenced by the EKF (odom1); `twist`, the same admitted"
        f" step as a body velocity (vx, vy, vyaw) on {VO_TWIST_TOPIC} (twist0) with OpenVINS's own"
        " velocity covariance from /ov_msckf/odomimu at the pose's stamp turned into base_link's"
        f" axes, no floor, while {VO_TOPIC} carries the track weightless (1e6). A sample whose"
        " covariance is missing or not one is withheld and counted. Live; vo_input vio only (under"
        " rtabmap's inputs it reads as `pose`)",
        why="pose until the drives measure twist. Under pose the VIO moved the EKF <= 1.7 cm over"
        " 3.5 m on drives 304/305 (2026-10-05, scratch/vio_ab_1005/moved_1005.py): the 7 cm floor"
        " differenced at 10 Hz is a velocity variance of 2 * 0.07^2 * 0.1 = 9.8e-4, one wheel"
        " sample, beside the wheels at 20 Hz, rf2o and the gyro, so the source barely counts."
        " OpenVINS's VELOCITY covariance is observable and is its own answer; its pose covariance"
        " in G grows without bound and is never a weight",
        on_when="a drive whose point is the VIO's weight in the EKF: vo_input vio, OpenVINS"
        " initialised, the report line's twist sigmas and `cov missing 0` read first under pose",
        off_when="pose whenever odom -> base_link must be today's, OpenVINS's velocity sigmas read"
        " implausibly small, or the twist drives do not beat pose against the lidar truth",
    ),
    Flag(
        "vio_twist_source",
        "step",
        choices=VIO_TWIST_SOURCES,
        description="vo_output twist only: where the body velocity comes from. `step`, the SE(2)"
        " log of two composed poseimu poses (one per camera frame, 9.5-11.6 Hz, the neck"
        " subtracted by the composition); `imu`, every odomimu sample vio_publish_hz takes (50 Hz):"
        " OpenVINS's IMU velocity STATE and bias-corrected gyro carried into base_link through the"
        " neck chain (its rate differenced from TF over the step,"
        " pepin.visual_odometry.imu_base_twist), its covariance odomimu's own with the lever arm,"
        " processed 0.1 s behind the newest sample so the TF lookup never waits. Under lost both"
        " send the yaw rate alone",
        why="step until drives measure imu: per sample imu is the noisier of the two. Offline on"
        " drives 0329/0330 through the same functions (scratch/vio_relay_night/"
        "imu_twist_offline.py), where the live relay sent a twist, moving, head still: imu at 47"
        " twists/s, vx vs the wheels z p50 1.58/1.45, p90 3.7/4.1 (the live step: 1.20/1.24,"
        " 2.9/3.9); sideways |vy| p50 2.2/1.7 cm/s (step 1.0/0.7); the yaw rate vs the base gyro"
        " 1.1/1.0 deg/s, z 1.2/1.1 (step 1.3/0.7 deg/s); at rest vx z p50 0.6/1.1 (step"
        " 0.12/0.03). In the dark (the live relay's lost stretches, the gaze gate replayed) the"
        " yaw rate it would now send alone reads against the base gyro z p50 0.48/0.29, p90"
        " 1.24/0.85 under step, 1.41/1.36 and 6.2/4.2 under imu (dark_yaw_offline.py)."
        " odomimu's velocity follows its own poseimu track in the median (angle"
        " -1.5/+0.1 deg, speed ratio 0.97/1.00) with a p10-p90 spread of -29..+13 deg over 0.1 s"
        " (odomimu_direction.py). Differencing odomimu's POSES instead is out: every update moves"
        " them (|dp/dt - v| p99 2.2-2.5 m/s at 20 ms, odomimu_steps.py)",
        on_when="imu for a drive whose point is the VIO's vote at the wheels' rate, the report"
        " line's sideways numbers read first",
        off_when="step whenever the twist must be the measured one",
    ),
    Flag(
        "vio_guard",
        False,
        description="the plausibility guard's wheel rule: a composed base velocity farther than"
        " vio_wheel_diff_m_s from the wheels' is refused; off, the guard judges the speed alone"
        " (vio_max_speed_m_s, always on), so the VIO may disagree with slipping wheels",
        why="off since 2026-10-05 (Artem): a VIO with weight is worth most exactly when it"
        " disagrees with the wheels, which this rule refused. On from the guard's landing"
        " (2026-10-04) until then: 15 of drives 306/307's 16 refusals fell inside or right"
        " after a >= 45 deg head swing, OpenVINS's own velocity errors, which the velocity"
        " covariance now carries instead",
        on_when="the default while the VIO's velocity is unproven in motion",
        off_when="a drive that tests the VIO against wheel slip (carpet, a stall), with vo_output"
        " twist; VioLost's wheel rule (vio_lost_speed_m_s for vio_lost_s) still catches a sustained"
        " divergence",
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
        self._switches = Switches(
            self, with_knobs(FLAGS, load_knobs("visual_odometry")), on_change=self._on_switch
        )
        # The rtabmap input the launch started an odometry node for (none under vio): the one
        # a live switch away from vio may go back to.
        self._launched = str(self._switches["vo_input"])
        self._gate = self._new_gate()
        self._track = VoTrack()
        self._cap = PublishCap(self._publish_hz())
        self._rest = RestWatch()
        self._tally = Tally()
        self._last_window: Any = None  # the report's window, for the twist sigmas
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
        self._twist_pub = self.create_publisher(
            TwistWithCovarianceStamped, VO_TWIST_TOPIC, bridged_qos_profile(VO_TWIST_TOPIC)
        )
        self._covariances = TwistCovariances()
        # vio_twist_source imu: odomimu samples waiting for their neck chain, the neck at the
        # last one taken, and the lost episode's clock (the yaw rate's bias walk)
        self._imu_queue = ImuQueue(float(self._switches["vio_publish_hz"]))
        self._neck = NeckBaseline()
        self._yaw_only = YawOnly(self._gyro_random_walk())
        self._vio_verdict: str | None = None  # the last poseimu's: ok, lost or rejected
        self._last_yaw_sigma: float | None = None  # rad/s, the last yaw-only twist's
        self._seeded: str | None = None  # the static camera edges put into the buffer
        self._vio_health = VioHealth()
        self._vio_lost = VioLost(
            lost_speed_m_s=float(self._switches["vio_lost_speed_m_s"]),
            lost_s=float(self._switches["vio_lost_s"]),
            min_features=int(self._switches["vio_min_features"]),
        )
        self._guard = VioGuard(
            max_speed_m_s=float(self._switches["vio_max_speed_m_s"]),
            wheel_diff_m_s=float(self._switches["vio_wheel_diff_m_s"]),
            restart_rejects=int(self._switches["vio_restart_rejects"]),
            wheel_rule=self._switches.on("vio_guard"),
        )
        self._tf: TfLookup | None = None
        self._tf_failure: str | None = None
        self._restart: Any = None  # the /vio/restart client, made with the VIO's subscriptions
        self._points: dict[str, int] = {}  # the latest feature count of each cloud (msckf, slam)
        self._last_vio: VoPose | None = None  # the previous composed pose, for the VIO's speed
        self._last_vio_xy: tuple[float, float] | None = None  # the last published, for its step
        self._local = local
        if self._launched != "vio":
            self.create_subscription(Odometry, RAW_TOPIC, self._on_vo, local)
        if self._input == "vio":
            self._vio_up()
        self.create_subscription(
            Odometry, WHEELS_TOPIC, self._on_wheels, bridged_qos_profile(WHEELS_TOPIC)
        )
        self.create_timer(REPORT_S, self._report)
        self.get_logger().info(
            f"visual odometry up: vo_input {self._input}, {self._source()} -> {VO_TOPIC} for the"
            f" board's EKF (x, y and yaw, differentially), rest drift measured against"
            f" {WHEELS_TOPIC}; flags: {self._switches.state()}"
        )

    @property
    def _input(self) -> str:
        """What the relay reads now: the live ``vo_input`` flag."""
        return str(self._switches["vo_input"])

    def _source(self) -> str:
        return VIO_POSE_TOPIC if self._input == "vio" else RAW_TOPIC

    def _publish_hz(self) -> float:
        """The cap's rate: vio_publish_hz under vio (the wheels' 50), vo_publish_hz otherwise."""
        name = "vio_publish_hz" if self._input == "vio" else "vo_publish_hz"
        return float(self._switches[name])

    def _gyro_random_walk(self) -> float | None:
        """The head IMU's gyro bias random walk (rad/s/sqrt(s)) from config/head_imu.json's noise
        block: the x10 Allan number OpenVINS itself runs on (1.02e-4; measured 1.02e-5), so the
        bias walk and the reported rate sigma (sigma_w^2/dt from the same block) are one model.
        ``None`` (unreadable) refuses the yaw rate under lost rather than guess one."""
        try:
            return float(HeadImuConfig.load().noise["gyro_random_walk"])
        except (OSError, KeyError, ValueError) as exc:
            self.get_logger().warning(
                f"config/head_imu.json's gyro_random_walk unreadable ({exc}): no yaw rate is"
                " sent while the VIO is lost"
            )
            return None

    def _new_gate(self) -> VoGate:
        return VoGate(
            max_speed_m_s=float(self._switches["vo_max_speed"]),
            max_turn_deg_s=float(self._switches["vo_max_turn"]),
            max_gap_s=float(self._switches["vo_max_gap_s"]),
            reset_radius_m=float(self._switches["vo_reset_radius_m"]),
        )

    def _vio_up(self) -> None:
        """The VIO input's TF listener, subscriptions and restart client, made the first time it
        is selected and kept (their callbacks read nothing while another input is)."""
        if self._tf is not None:
            return
        self._tf = TfLookup(self, on_failure=self._on_tf_failure)
        self._seed_camera_edges()
        local = self._local
        self.create_subscription(PoseWithCovarianceStamped, VIO_POSE_TOPIC, self._on_vio, local)
        self.create_subscription(Odometry, VIO_ODOM_TOPIC, self._on_vio_odom, local)
        self.create_subscription(
            PointCloud2, VIO_POINTS_TOPIC, lambda m: self._on_points("msckf", m), local
        )
        self.create_subscription(
            PointCloud2, VIO_SLAM_POINTS_TOPIC, lambda m: self._on_points("slam", m), local
        )
        self.create_subscription(
            Odometry,
            ZUPT_TOPIC,
            self._on_zupt,
            QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._restart = self.create_client(Trigger, VIO_RESTART_SERVICE)

    def _seed_camera_edges(self) -> None:
        """camera_stream's static camera edges (camera_link -> camera_optical -> head_imu) put
        into this node's TF buffer straight from config/camera.json, through the reader and the
        builder camera_stream broadcasts them with; the board's neck chain stays live TF.

        Without this the composition waited for /tf_static's replay to a late joiner, and under
        rmw_zenoh 0.2.10 that replay is held until the subscriber's history query finalises
        (zenoh-ext delivers a new source's samples only then, and rmw_zenoh sets that query's
        timeout to u64::MAX): on 2026-10-05 every laptop listener got camera_stream's edges
        114-125 s after its own start, this one at 18:25:40 for a switch at 18:23:34. The same
        file and the same numbers: the broadcast, when it lands, rewrites them unchanged."""
        assert self._tf is not None
        try:
            camera = load_camera_mounts()
        except (OSError, KeyError, ValueError) as exc:
            self.get_logger().warning(
                f"config/camera.json unreadable ({exc}): the camera edges wait for /tf_static"
            )
            return
        edges = camera_edges(camera, self.get_clock().now().to_msg())
        for edge in edges:
            self._tf.buffer.set_transform_static(edge, "config/camera.json")
        self._seeded = ", ".join(f"{e.header.frame_id} -> {e.child_frame_id}" for e in edges)
        if camera.imu is None:
            self.get_logger().warning(
                f"config/camera.json's rig has no head_imu block: nothing composes {IMU_FRAME}"
                f" into {BASE_FRAME}"
            )

    def close(self) -> None:
        """Stop the TF listener's thread (vo_input vio) before the node is destroyed."""
        if self._tf is not None:
            self._tf.close()

    # ---- inputs ------------------------------------------------------------------------------
    def _on_switch(self, name: str, old: object, new: object) -> None:
        """A flag changed: the three ceilings are the gate's, the rest are read where they act."""
        if name == "vo_input":
            self._switch_input(str(old), str(new))
        elif name == "vo_max_speed":
            self._gate.max_speed_m_s = float(new)  # type: ignore[arg-type]
        elif name == "vo_max_turn":
            self._gate.max_turn_deg_s = float(new)  # type: ignore[arg-type]
        elif name == "vo_max_gap_s":
            self._gate.max_gap_s = float(new)  # type: ignore[arg-type]
        elif name == "vo_reset_radius_m":
            self._gate.reset_radius_m = float(new)  # type: ignore[arg-type]
        elif name in ("vo_publish_hz", "vio_publish_hz"):
            self._cap.hz = self._publish_hz()
            self._imu_queue.rate.hz = float(self._switches["vio_publish_hz"])
        elif name in ("vio_twist_source", "vo_output"):
            self._twist_restart()
        elif name == "vio_max_speed_m_s":
            self._guard.max_speed_m_s = float(new)  # type: ignore[arg-type]
        elif name == "vio_wheel_diff_m_s":
            self._guard.wheel_diff_m_s = float(new)  # type: ignore[arg-type]
        elif name == "vio_restart_rejects":
            self._guard.restart_rejects = int(new)  # type: ignore[call-overload]
        elif name == "vio_guard":
            self._guard.wheel_rule = bool(new)
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

    def _switch_input(self, old: str, new: str) -> None:
        """A live ``vo_input`` change: vio, or back to the rtabmap input the launch started (no
        other rtabmap odometry runs). The gate starts afresh and the track's next step is measured
        from the new source's first pose; the published total stands where it stood."""
        if new != "vio" and new != self._launched:
            started = (
                "none (launched with vio)" if self._launched == "vio" else f"{self._launched}'s"
            )
            raise ValueError(
                f"vo_input {new}: the rtabmap odometry running is {started}; another needs a"
                " vslam restart (ros/laptop.sh vslam, or vslam --vo-depth)"
            )
        if new == "vio":
            self._vio_up()
        self._gate = self._new_gate()
        self._track.anchor(None)
        self._rest.restart()
        self._last_xy = None
        self._last_vio = None
        self._last_vio_xy = None
        self._after_gate = False
        self._cap.hz = self._publish_hz()
        self._twist_restart(full=True)
        self.get_logger().info(
            f"vo_input {old} -> {new}: {self._source()} -> {VO_TOPIC}, the gate restarted, the"
            f" track carries on from ({self._track.pose[0]:.2f}, {self._track.pose[1]:.2f})"
        )

    def _covariance_mode(self) -> str:
        """The covariance the published pose carries: the flag's, except that under vio the
        registration's modes read as ``vio`` — OpenVINS has no registration variance and its
        own covariance is the marginal of an unobservable pose (vio.md M3)."""
        mode = str(self._switches["vo_covariance"])
        if self._input == "vio" and mode in ("dynamic", "rtabmap"):
            return "vio"
        return mode

    def _on_wheels(self, msg: Odometry) -> None:
        """The board's wheel odometry: only its twist is read, and only to know whether the cart
        is standing still (the drift at rest is what says this source may be fused at all).

        Timed by this node's own clock, not by the message's stamp: this one is the board's and
        the visual poses are the laptop's (:class:`pepin.visual_odometry.RestWatch`).
        """
        now = time.monotonic()
        linear, yaw_rate = float(msg.twist.twist.linear.x), float(msg.twist.twist.angular.z)
        self._rest.wheels(now, linear, yaw_rate)
        self._vio_lost.wheels(now, linear)
        self._guard.wheels(now, linear, yaw_rate)

    def _on_vo(self, msg: Odometry) -> None:
        """One pose from rgbd_odometry: gated, measured at rest, and published with the
        covariance the flags chose. Nothing is read while the input is vio."""
        if self._input == "vio":
            return
        self._tally.count("in")
        stamp = stamp_seconds(msg.header.stamp)
        self._tally.sample("latency_ms", (time.time() - stamp) * 1000.0)
        pose = VoPose(
            stamp=stamp,
            x=float(msg.pose.pose.position.x),
            y=float(msg.pose.pose.position.y),
            yaw=yaw_of(msg.pose.pose.orientation),
        )
        lost = is_lost(msg.pose.covariance)
        published = self._admit(pose, lost)
        if published is None:
            return
        _write_planar_pose(msg, published)
        mode = self._covariance_mode()
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
        """One OpenVINS pose: health, base_link composed through TF at its stamp, the guard,
        the lost rules, then the shared gate and track; published as an Odometry in rtabmap's
        frames. Nothing is read while the input is rtabmap's."""
        if self._input != "vio":
            return
        self._tally.count("in")
        stamp = stamp_seconds(msg.header.stamp)
        self._tally.sample("latency_ms", (time.time() - stamp) * 1000.0)
        if self._vio_health.observe(msg.pose.covariance):
            self._tally.count("vio_reinit")
            self._last_vio = None
            self._reanchor(None)  # a new gravity frame: nothing differences across it
            self._twist_restart(full=True)
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
        last, self._last_vio = self._last_vio, pose
        velocity = None if last is None else body_velocity(last, pose)
        now = time.monotonic()
        if self._guard.check(now, velocity) is not None:
            self._tally.count("rejected")
            self._vio_verdict = "rejected"
            self._reanchor(pose)
            self._maybe_restart(now)
            return
        speed = 0.0 if velocity is None else math.hypot(*velocity)
        lost = self._vio_lost.check(now, speed)
        if lost is not None:
            self._tally.count("vio_lost")
            self._vio_verdict = "lost"
            self._yaw_only.lost(stamp, lost)
            self._reanchor(pose)
            if last is not None and self._twist_source() == "step":
                self._step_yaw_only(msg.header.stamp, last, pose, t_i_b.rotation.T)
            return
        self._vio_verdict = "ok"
        self._yaw_only.visual(stamp)
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
        # OpenVINS's velocity covariance at this stamp in base_link's axes (R_B_I = R_I_B^T):
        # the twist's weight, and the report's sigmas under either output.
        source = self._covariances.nearest(stamp)
        covariance = None if source is None else base_twist_covariance(source, t_i_b.rotation.T)
        if covariance is not None:
            self._tally.sample("sigma_vx", math.sqrt(covariance[0]))
            self._tally.sample("sigma_vy", math.sqrt(covariance[7]))
            self._tally.sample("sigma_vyaw", math.sqrt(covariance[35]))
        if self._output_mode() == "twist":
            if self._twist_source() == "step":  # under imu the odomimu path sends the twist
                step = self._track.last_step
                twist = None if step is None else se2_twist(*step)
                if twist is None:
                    self._tally.count("no_step")  # the first pose after an anchor: no velocity yet
                    return
                if covariance is None:
                    self._tally.count("cov_missing" if source is None else "cov_bad")
                    return
                self._publish_twist(msg.header.stamp, twist, self._scaled(covariance))
            out.pose.covariance = weightless_covariance()
        elif self._covariance_mode() == "constant":
            out.pose.covariance = planar_covariance(
                float(self._switches["vo_sigma_m"]), float(self._switches["vo_yaw_sigma_deg"])
            )
        else:
            step_m = 0.0 if there is None else math.hypot(here[0] - there[0], here[1] - there[1])
            out.pose.covariance = self._vio_covariance(step_m)
        self._pub.publish(out)
        self._tally.count("out")

    def _output_mode(self) -> str:
        """How the published sample weighs in the EKF: the ``vo_output`` flag under vio; rtabmap's
        inputs carry no velocity covariance, so they read as ``pose``."""
        return str(self._switches["vo_output"]) if self._input == "vio" else "pose"

    # ---- the twist: its source, the lost rule's split, the scales -----------------------------
    def _twist_source(self) -> str:
        """Where the twist's velocity comes from (``vio_twist_source``); ``none`` unless the
        output is twist."""
        if self._output_mode() != "twist":
            return "none"
        return str(self._switches["vio_twist_source"])

    def _scaled(self, covariance: list[float]) -> list[float]:
        """The twist covariance with vio_sigma_scale on vx and vy, vio_yaw_sigma_scale on vyaw."""
        return scaled_twist_covariance(
            covariance,
            float(self._switches["vio_sigma_scale"]),
            float(self._switches["vio_yaw_sigma_scale"]),
        )

    def _twist_restart(self, full: bool = False) -> None:
        """Nothing of the twist path differences across this: the IMU queue and the neck baseline
        start again; ``full`` (a re-init, an input switch: a new filter) also forgets the lost
        episode and the last verdict."""
        self._imu_queue.clear()
        self._neck.reset()
        if full:
            self._yaw_only.reset()
            self._vio_verdict = None

    def _step_yaw_only(self, stamp_msg: Any, last: VoPose, pose: VoPose, r_b_i: Any) -> None:
        """Lost under the step source: the composed step's yaw rate alone, weighed by odomimu's
        rate variance at the pose's stamp and the bias walk since the last visual update."""
        twist = se2_twist(last, pose)
        source = self._covariances.nearest(pose.stamp)
        covariance = None if source is None else base_twist_covariance(source, r_b_i)
        if twist is None:
            self._tally.count("no_step")
            return
        if covariance is None:
            self._tally.count("cov_missing" if source is None else "cov_bad")
            return
        self._publish_yaw_only(stamp_msg, (last.stamp, pose.stamp), twist[2], covariance)

    def _publish_yaw_only(
        self, stamp_msg: Any, judged: tuple[float, ...], yaw_rate: float, covariance: list[float]
    ) -> None:
        """The yaw rate alone while the VIO is lost (vx, vy weightless), unless a stamp in
        ``judged`` is gated -- the mast swaying beyond gate_sway_dps / gate_sway_deg (counted as
        mast) or the head blind / the body spinning (gaze) -- or it turns faster than
        vo_max_turn. Its variance: (vio_yaw_sigma_scale * sigma_reported)^2 + brw^2 * t."""
        if not self._switches.on("vo_publish"):
            self._tally.count("withheld")
            return
        if self._switches.on("gaze_gate"):
            for stamp in judged:
                verdict = self._gaze.verdict(stamp)
                if verdict is not None:
                    self._tally.count("yaw_mast" if verdict == SWAYING else "yaw_gaze")
                    return
        if abs(yaw_rate) > math.radians(float(self._switches["vo_max_turn"])):
            self._tally.count("yaw_fast")
            return
        bias = self._yaw_only.variance(judged[-1])
        if bias is None:
            self._tally.count("yaw_no_brw")
            return
        yaw_scale = float(self._switches["vio_yaw_sigma_scale"])
        weighed = yaw_only_covariance(scaled_twist_covariance(covariance, 1.0, yaw_scale), bias)
        self._last_yaw_sigma = math.sqrt(weighed[35])
        self._tally.sample("sigma_yaw_only", self._last_yaw_sigma)
        self._publish_twist(stamp_msg, (0.0, 0.0, yaw_rate), weighed)
        self._tally.count("yaw_out")

    def _imu_twist(self, sample: ImuState) -> None:
        """One odomimu sample the rate took (vio_twist_source imu): base_link's twist through the
        neck chain at its stamp (no wait: the sample is VIO_IMU_LAG_S old), judged by the last
        poseimu's verdict -- ok: the whole twist; lost: the yaw rate alone; rejected or none yet:
        nothing -- and by the gaze gate at its own stamp."""
        verdict = self._vio_verdict
        if verdict is None or verdict == "rejected":
            self._tally.count("imu_held")
            self._neck.reset()
            return
        if not self._switches.on("vo_publish"):
            self._tally.count("withheld")
            return
        assert self._tf is not None
        stamp_msg = stamp_from_seconds(sample.stamp)
        transform = self._tf.transform(IMU_FRAME, BASE_FRAME, stamp_msg, 0.0)
        if transform is None:
            self._tally.count("imu_tf_miss")
            self._neck.reset()
            return
        t_i_b = pose_from_transform(transform)
        r_b_i = t_i_b.rotation.T
        t_b_i = homogeneous(r_b_i, -r_b_i @ t_i_b.translation)
        step = self._neck.step(sample.stamp, t_b_i)
        twist = None if step is None else imu_base_twist(sample.velocity, sample.rate, t_b_i, *step)
        if twist is None:
            self._tally.count("no_step")
            return
        covariance = base_twist_covariance(sample.covariance, r_b_i, lever=t_b_i[:3, 3])
        if covariance is None:
            self._tally.count("cov_bad")
            return
        if verdict == "lost":
            self._publish_yaw_only(stamp_msg, (sample.stamp,), twist[2], covariance)
            return
        if self._switches.on("gaze_gate") and self._gaze.verdict(sample.stamp) is not None:
            self._tally.count("imu_gated")
            return
        too_fast = math.hypot(twist[0], twist[1]) > float(self._switches["vio_max_speed_m_s"])
        if too_fast or abs(twist[2]) > math.radians(float(self._switches["vo_max_turn"])):
            self._tally.count("imu_fast")
            return
        self._publish_twist(stamp_msg, twist, self._scaled(covariance))

    def _publish_twist(
        self, stamp: Any, twist: tuple[float, float, float], covariance: list[float]
    ) -> None:
        """One body velocity for the EKF's twist0: base_link's vx, vy and vyaw with OpenVINS's own
        covariance (the other axes weightless)."""
        msg = TwistWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = BASE_FRAME
        msg.twist.twist.linear.x, msg.twist.twist.linear.y = twist[0], twist[1]
        msg.twist.twist.angular.z = twist[2]
        msg.twist.covariance = covariance
        self._twist_pub.publish(msg)
        self._tally.count("twist_out")

    def _on_vio_odom(self, msg: Odometry) -> None:
        """OpenVINS's odomimu: its twist covariance kept by stamp for the pose of the same
        instant; under vo_output twist with vio_twist_source imu, also the sample itself, sent as
        a twist once it is VIO_IMU_LAG_S old (:meth:`_imu_twist`)."""
        if self._input != "vio":
            return
        stamp = stamp_seconds(msg.header.stamp)
        self._covariances.add(stamp, msg.twist.covariance)
        self._tally.count("odomimu")
        if self._twist_source() != "imu":
            return
        v, w = msg.twist.twist.linear, msg.twist.twist.angular
        self._imu_queue.add(
            ImuState(
                stamp,
                (float(v.x), float(v.y), float(v.z)),
                (float(w.x), float(w.y), float(w.z)),
                tuple(float(c) for c in msg.twist.covariance),
            )
        )
        for sample in self._imu_queue.due():
            self._imu_twist(sample)

    def _maybe_restart(self, now: float) -> None:
        """Restart OpenVINS when the guard says so (rejections in a row, the wheels at rest):
        ``/vio/restart`` signals it in its container and the launch respawns it. Logged once per
        restart; the VIO's state starts again from the first pose of the new gravity frame."""
        reason = self._guard.restart_due(now)
        if reason is None:
            return
        self._tally.count("vio_restart")
        client = self._restart
        if client is None or not client.service_is_ready():
            self.get_logger().warning(
                f"OpenVINS restart due ({reason}) but {VIO_RESTART_SERVICE} is not served"
                " (pepin_bringup.vio_keeper in pepin-vio): ros/laptop.sh vio kick, at rest"
            )
            return
        self.get_logger().warning(
            f"restarting OpenVINS through {VIO_RESTART_SERVICE}: {reason}; it initialises again"
            " at rest, on the next motion (a head pan)"
        )
        client.call_async(Trigger.Request()).add_done_callback(self._on_restarted)
        self._last_vio = None
        self._reanchor(None)

    def _on_restarted(self, future: Any) -> None:
        """The keeper's answer: said only when it could not signal OpenVINS."""
        response = future.result()
        if response is None or not response.success:
            text = "no answer" if response is None else response.message
            self.get_logger().warning(f"{VIO_RESTART_SERVICE} failed: {text}")

    def _reanchor(self, pose: VoPose | None) -> None:
        """A pose nothing may be differenced across (a lost VIO, a re-init): the gate and the
        track start the next step from it, and the rest watch starts again."""
        if pose is not None:
            self._gate.reanchor(pose)
            self._track.anchor(self._gate.anchor)
        else:
            self._track.anchor(None)
        self._rest.restart()

    def _on_points(self, kind: str, msg: PointCloud2) -> None:
        """One of OpenVINS's two feature clouds of its last update: the sum of the latest of
        each is lost rule (c)'s count."""
        self._points[kind] = int(msg.width) * int(msg.height)
        self._vio_lost.features(time.monotonic(), sum(self._points.values()))

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
        self._last_window = w
        c = w.counts
        drop = f" (last: {self._drop})" if self._drop else ""
        hold = f" (last: {self._hold})" if self._hold else ""
        x, y, _ = self._track.pose
        source = "OpenVINS" if self._input == "vio" else "rtabmap"
        latency = w.samples.get("latency_ms", [])
        late = f"{statistics.median(latency):.0f} ms" if latency else "n/a"
        self.get_logger().info(
            f"vo: {w.rate('in'):.1f} poses/s from {source} ({c['in']} in, {c['out']} out),"
            f" {w.rate('out'):.1f} published, stamp to receipt p50 {late}; {self._ekf_text()};"
            f" {c['dropped']} dropped{drop}, {c['skipped']} skipped{hold},"
            f" {c['withheld']} withheld from the EKF; track at ({x:.2f}, {y:.2f});"
            f"{self._odom_text()} {self._rest.report()}; {self._gate_text(c)};{self._vio_text(c)}"
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
            f" vio: {counts['rejected']} rejected by the guard, {self._guard.report()};"
            f" output {self._output_mode()}: {self._twist_text(counts)};"
            f" covariance {self._covariance_mode()}; reinit {counts['vio_reinit']}"
            f" ({self._vio_health.reinits} since the start), {counts['vio_lost']} poses withheld"
            f" as lost, {self._vio_lost.report()}, features {self._features_text()},"
            f" tf_miss {counts['tf_miss']}{tf}; kick at rest:"
            " ros/laptop.sh vio kick;"
        )

    def _twist_text(self, counts: Counter[str]) -> str:
        """OpenVINS's velocity sigmas in base_link (the window's medians), the twists sent and
        the samples withheld for want of a covariance, for the report line."""
        window = self._last_window
        sigma = [] if window is None else window.samples.get("sigma_vx", [])
        if sigma:
            vx = statistics.median(sigma) * 100.0
            vy = statistics.median(window.samples["sigma_vy"]) * 100.0
            vyaw = math.degrees(statistics.median(window.samples["sigma_vyaw"]))
            sigmas = f"sigma p50 vx {vx:.2f} vy {vy:.2f} cm/s vyaw {vyaw:.2f} deg/s ({len(sigma)})"
        else:
            sigmas = "no velocity covariance matched"
        seeded = self._seeded or "none"
        return (
            f"{sigmas}, {counts['twist_out']} twists sent, withheld: cov missing"
            f" {counts['cov_missing']}, cov bad {counts['cov_bad']}, no step {counts['no_step']}"
            f"{self._source_text(counts)}; {self._yaw_only_text(counts)};"
            f" odomimu {counts['odomimu']} in; static edges from config/camera.json: {seeded}"
        )

    def _source_text(self, counts: Counter[str]) -> str:
        """The twist's source, its rate and scales, and what the imu source withheld."""
        source = str(self._switches["vio_twist_source"])
        scales = (
            f"sigma scale x{float(self._switches['vio_sigma_scale']):g}"
            f" yaw x{float(self._switches['vio_yaw_sigma_scale']):g}"
        )
        if source != "imu":
            return f"; source step (per camera frame), {scales}"
        return (
            f"; source imu at {float(self._switches['vio_publish_hz']):g} Hz, {scales}, held"
            f" {counts['imu_held']} (no verdict / rejected), gated {counts['imu_gated']}, too fast"
            f" {counts['imu_fast']}, tf miss {counts['imu_tf_miss']}"
        )

    def _yaw_only_text(self, counts: Counter[str]) -> str:
        """The lost rule's split: ``yaw only since N s, sigma_yaw X deg/s`` while lost, the
        yaw-only twists sent, and what the mast and gaze gates withheld of them."""
        lost = self._yaw_only
        last = self._last_vio
        if lost.active:
            since = 0.0 if last is None else lost.age(last.stamp)
            sigma = self._last_yaw_sigma
            shown = "n/a" if sigma is None else f"{math.degrees(sigma):.2f}"
            state = f"yaw only since {since:.1f} s, sigma_yaw {shown} deg/s ({lost.reason})"
        else:
            state = "full twist"
        brw = "none: no yaw while lost" if lost.brw is None else f"{lost.brw:.2e} rad/s/sqrt(s)"
        return (
            f"{state}; yaw only {counts['yaw_out']} sent, withheld: mast {counts['yaw_mast']}, gaze"
            f" {counts['yaw_gaze']}, too fast {counts['yaw_fast']}, no brw {counts['yaw_no_brw']};"
            f" lost episodes {lost.episodes}, gyro bias walk {brw}"
        )

    def _features_text(self) -> str:
        """The last feature counts: ``57 (msckf 0 + slam 57)``, or ``none heard``."""
        if not self._points:
            return "none heard"
        parts = " + ".join(f"{k} {n}" for k, n in sorted(self._points.items()))
        return f"{sum(self._points.values())} ({parts})"

    def _ekf_text(self) -> str:
        """Whether the board's EKF subscribes /vo, from the graph (rmw_zenoh carries the remote
        node's name): the nearest thing to an ack the filter gives."""
        nodes = sorted({info.node_name for info in self.get_subscriptions_info_by_topic(VO_TOPIC)})
        ekf = "yes" if EKF_NODE in nodes else "NO"
        text = f"{VO_TOPIC} read by {EKF_NODE}: {ekf} ({len(nodes)} subscriber nodes)"
        if self._output_mode() != "twist":
            return text
        # Under twist /vo is weightless: an EKF without twist0 (ekf.yaml before 2026-10-05) gives
        # the VIO no weight at all, and this says so.
        readers = {info.node_name for info in self.get_subscriptions_info_by_topic(VO_TWIST_TOPIC)}
        return f"{text}, {VO_TWIST_TOPIC}: {'yes' if EKF_NODE in readers else 'NO'}"

    def _odom_text(self) -> str:
        """The EKF's own odom -> base_link now, for a drift read off two report lines; nothing
        before the TF listener exists (it comes with the VIO input)."""
        if self._tf is None:
            return ""
        pose = self._tf.pose(ODOM_FRAME, BASE_FRAME)
        if pose is None:
            return " EKF odom -> base_link not in TF;"
        yaw = math.degrees(math.atan2(float(pose.rotation[1, 0]), float(pose.rotation[0, 0])))
        tx, ty = float(pose.translation[0]), float(pose.translation[1])
        return f" EKF odom -> base_link ({tx:.3f}, {ty:.3f}, {yaw:.1f} deg);"

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
