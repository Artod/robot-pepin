"""OpenVINS's keeper in pepin-vio: it says when OpenVINS is lost and brings it back in its process.

OpenVINS has no failure detector and never re-initialises by itself once it diverges (2026-10-04:
the first fast head pan threw it metres off for good; 2026-10-05, drive 0330: no VIO after the dark
printer, even under the lamp at home). Our patch of its wrapper (ros/patches/openvins-reset.patch)
gives it ``/ov_msckf/reset`` (a new filter in the same process, seeded warm from a picture, the old
filter's biases, gravity and the velocity this node publishes) and ``/ov_msckf/health``, one line
per frame. This node is the policy around them (:mod:`pepin.vio_recover` without ROS):

- it publishes the VELOCITY SEED on ``/vio/seed_twist``: the board EKF's body twist
  (``/odometry/filtered``) carried to the head IMU through the neck's TF, in the IMU's own axes,
  with its covariance — only while that transform says the head is still, when the head and
  the cart are one rigid body;
- it judges ``/ov_msckf/health`` (``vio_watch``): a velocity that disagrees with that seed, a
  dark stretch or a runaway speed is lost
  (:class:`pepin.vio_recover.VioWatch`);
- it answers ``/vio/restart`` (std_srvs/Trigger), the relay's door (pepin_bringup.visual_odometry
  asks after a run of implausible samples), the same way;
- a lost OpenVINS is recovered as ``vio_recover`` says: ``reset`` (the default) calls
  ``/ov_msckf/reset`` after pushing the seed's knobs to OpenVINS's parameters; ``restart`` is the
  old way, the launch's SIGINT to the process (the launch respawns it in 2 s and it initialises at
  rest), also the fallback when the reset service does not answer (``vio_restart_fallback``);
  ``off`` only logs.

Each recovery is timed from the verdict to the new filter's first published frame, and is a moment
on the head's face (``vio_restart``). A report line every 30 s says what happened.
"""

from __future__ import annotations

import math
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

from pepin.face_events import VIO_RESTART, FaceSink
from pepin.flags import Flag, FlagSet, load_knobs, with_knobs
from pepin.mounts import load_camera_mounts
from pepin.vio_recover import (
    OPENVINS_FLAGS,
    OPENVINS_KNOBS,
    HeadStill,
    Health,
    Recoveries,
    VioWatch,
    health_values,
    imu_velocity,
    twist_block,
)
from pepin_bringup.msgs import camera_edges, stamp_seconds
from pepin_bringup.node_kit import Switches, TfLookup, spin_main

RESTART_SERVICE = "/vio/restart"  # the relay's door (pepin_bringup.visual_odometry)
RESET_SERVICE = "/ov_msckf/reset"  # openvins-reset.patch: a new filter in the same process
HEALTH_TOPIC = "/ov_msckf/health"
SEED_TOPIC = "/vio/seed_twist"
EKF_TOPIC = "/odometry/filtered"
OPENVINS_NODE = "/ov_msckf/run_subscribe_msckf"
PROCESS = "run_subscribe_msckf"  # OpenVINS's executable, matched on the command line
SIGNAL = ("pkill", "-INT", "-f", PROCESS)
IMU_FRAME = "head_imu"
BASE_FRAME = "base_link"
REPORT_S = 30.0
RESET_ANSWER_S = 2.0  # how long a reset call may stay unanswered before the fallback

FLAGS = FlagSet(
    Flag(
        "vio_recover",
        "reset",
        choices=("reset", "restart", "off"),
        description="how a lost OpenVINS (the keeper's own verdict, or the relay's on /vio/restart)"
        " is brought back: `reset`, a new filter in the same process (/ov_msckf/reset,"
        " openvins-reset.patch) seeded warm on the first frame with a picture; `restart`, the"
        " launch's SIGINT to the process (respawned in 2 s, initialises after ~1 s at rest);"
        " `off`, logged only",
        why="reset since 2026-10-06: drive 0330's replay came back in the light after the dark"
        " printer, where the live OpenVINS never did (journal 2026-10-06 vio-recover); a restart"
        " waits for rest, and a dynamic initialisation took 7.6-8.9 s on a slow cart",
        on_when="always; `restart` if a reset misbehaves (the fallback below covers a reset"
        " service that does not answer)",
        off_when="`off` to watch the verdicts without acting on them",
    ),
    Flag(
        "vio_watch",
        True,
        description="the keeper's own failure rules on /ov_msckf/health call for a recovery: a"
        " velocity disagreeing with the EKF's at the IMU (vio_disagree_m_s for vio_disagree_s), a"
        " dark stretch (vio_dark_tracks, vio_dark_s, swings excused by vio_swing_rad_s) and a"
        " runaway speed (vio_max_speed_m_s for vio_speed_s); off, only the relay's /vio/restart"
        " does",
        why="on: the relay's restart needs 20 implausible samples with the wheels at rest, and a"
        " dark stretch while driving produces none (0330: 150 poses withheld as lost, no restart)",
        on_when="always",
        off_when="a test of the relay's own restart path, or a detector that misfires",
    ),
    Flag(
        "vio_dark_at_rest",
        False,
        description="the dark rule also judges frames OpenVINS held with its zero-velocity update"
        " (the cart at rest): a still cart staring at a blank wall is then lost too",
        why="off: at rest OpenVINS's ZUPT holds the filter whatever the picture (the stand,"
        " 2026-10-06: every frame accepted, chi2 0.03-0.09 under 16.9, |v| 0.025 m/s), and a reset"
        " there only stops the VIO until the head moves",
        on_when="a stand test of the reset path without driving",
        off_when="always otherwise",
    ),
    Flag(
        "vio_restart_fallback",
        True,
        description="when /ov_msckf/reset is not served or does not answer within 2 s, the"
        " process is restarted instead (the launch's SIGINT)",
        why="on: an OpenVINS built without openvins-reset.patch, or a frozen one, still comes back:"
        " the respawn takes 2 s and the static initialisation 1.0-1.3 s at rest (the stand,"
        " 2026-10-06)",
        on_when="always",
        off_when="a test of the reset alone",
    ),
    Flag(
        "seed_warm",
        OPENVINS_FLAGS["seed_warm"],
        description="a reset's new filter is seeded warm (kept biases, gravity from the"
        " accelerometer, /vio/seed_twist's velocity) on the first frame with a picture; off, it"
        " waits for OpenVINS's own initialiser (static at rest). Pushed to OpenVINS's parameters",
        why="on: the static initialiser needs a second of stillness and the dynamic one 7.6-8.9 s"
        " of motion on this cart (journal 2026-10-05, VIO A/B)",
        on_when="always",
        off_when="a seed that misleads the filter (its line in OpenVINS's log names every input)",
    ),
    Flag(
        "seed_dyn_init",
        OPENVINS_FLAGS["seed_dyn_init"],
        description="a reset's new filter may also initialise dynamically (in motion) when the warm"
        " seed cannot (OpenVINS's init_dyn_use for the reset only; the config's is the cold"
        " start's). Pushed to OpenVINS's parameters",
        why="off: a dynamic initialisation took 7.6-8.9 s on this slow cart (journal 2026-10-05,"
        " VIO A/B) and is weak with so little acceleration; the warm seed needs a second at most",
        on_when="biases older than seed_bias_max_age_s are common (long dark stretches) and the"
        " cart rarely rests",
        off_when="always otherwise",
    ),
)


def face_client() -> FaceSink:
    """The head server's door for the recoveries' moments (``source`` vio), on the board."""
    from pepin.audio_link import board_host
    from pepin.head_link import HEAD_PORT, HeadClient

    return HeadClient(board_host(), HEAD_PORT, source="vio").start()


def _run(command: Sequence[str]) -> int:
    """Run ``command`` and answer its exit code (pkill: 0 signalled, 1 nothing matched)."""
    return subprocess.run(list(command), check=False, timeout=5).returncode


class VioKeeper(Node):
    """Judges OpenVINS's health, publishes its velocity seed and recovers it when lost."""

    def __init__(
        self,
        run: Callable[[Sequence[str]], int] = _run,
        face: FaceSink | None = None,
        tf: TfLookup | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``run`` runs a command (pkill), ``face`` takes the recoveries' moments (the board's
        head server by default), ``tf`` looks up the neck chain (a listener of this node's own
        by default), ``clock`` is the receiving clock."""
        super().__init__("vio_keeper")
        self._run = run
        self._face = face if face is not None else face_client()
        self._now = clock  # never self._clock: rclpy.Node keeps its ROS clock there
        self._switches = Switches(
            self, with_knobs(FLAGS, load_knobs("vio_keeper")), on_change=self._on_switch
        )
        s = self._switches
        self._watch = VioWatch(
            dark_tracks=int(s["vio_dark_tracks"]),
            disagree_m_s=float(s["vio_disagree_m_s"]),
            disagree_s=float(s["vio_disagree_s"]),
            dark_s=float(s["vio_dark_s"]),
            swing_rad_s=float(s["vio_swing_rad_s"]),
            max_speed_m_s=float(s["vio_max_speed_m_s"]),
            speed_s=float(s["vio_speed_s"]),
            grace_s=float(s["vio_grace_s"]),
            dark_at_rest=s.on("vio_dark_at_rest"),
        )
        self._head = HeadStill(float(s["head_still_s"]), float(s["head_still_deg"]))
        self._recoveries = Recoveries()
        self._counts: Counter[str] = Counter()
        self._health: Health | None = None
        self._last_reason: str | None = None
        self._last_seed: str | None = None
        self._reference: tuple[float, Any] | None = None  # (EKF stamp, v_I): the last seed sent
        self._tf_failure: str | None = None
        self._tf = tf if tf is not None else TfLookup(self, on_failure=self._on_tf_failure)
        self._seeded = self._seed_camera_edges()
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._seed_pub = self.create_publisher(TwistWithCovarianceStamped, SEED_TOPIC, reliable)
        self.create_subscription(DiagnosticStatus, HEALTH_TOPIC, self._on_health, reliable)
        self.create_subscription(Odometry, EKF_TOPIC, self._on_ekf, reliable)
        self.create_service(Trigger, RESTART_SERVICE, self._on_restart)
        self._reset = self.create_client(Trigger, RESET_SERVICE)
        from rclpy.parameter_client import AsyncParameterClient

        self._params = AsyncParameterClient(self, OPENVINS_NODE)
        self._push_openvins_knobs()
        self.create_timer(REPORT_S, self._report)
        self.get_logger().info(
            f"vio keeper up: {HEALTH_TOPIC} judged (vio_watch), {RESTART_SERVICE} answered, a lost"
            f" OpenVINS recovered by {s['vio_recover']} ({RESET_SERVICE}, fallback: {PROCESS}"
            f" SIGINT), the velocity seed on {SEED_TOPIC} from {EKF_TOPIC} through"
            f" {IMU_FRAME} <- {BASE_FRAME} while the head is still; static edges from"
            f" config/camera.json: {self._seeded or 'none'}; flags: {s.state()}"
        )

    # ---- inputs ------------------------------------------------------------------------------
    def _on_switch(self, name: str, old: object, new: object) -> None:
        """A flag or knob changed: the rules' numbers are read where they act; OpenVINS's own
        are pushed to it."""
        watch, head = self._watch, self._head
        if name == "vio_dark_tracks":
            watch.dark_tracks = int(new)  # type: ignore[call-overload]
        elif name == "vio_disagree_m_s":
            watch.disagree_m_s = float(new)  # type: ignore[arg-type]
        elif name == "vio_disagree_s":
            watch.disagree_s = float(new)  # type: ignore[arg-type]
        elif name == "vio_dark_s":
            watch.dark_s = float(new)  # type: ignore[arg-type]
        elif name == "vio_swing_rad_s":
            watch.swing_rad_s = float(new)  # type: ignore[arg-type]
        elif name == "vio_max_speed_m_s":
            watch.max_speed_m_s = float(new)  # type: ignore[arg-type]
        elif name == "vio_speed_s":
            watch.speed_s = float(new)  # type: ignore[arg-type]
        elif name == "vio_grace_s":
            watch.grace_s = float(new)  # type: ignore[arg-type]
        elif name == "vio_dark_at_rest":
            watch.dark_at_rest = bool(new)
        elif name == "head_still_s":
            head.still_s = float(new)  # type: ignore[arg-type]
        elif name == "head_still_deg":
            head.still_deg = float(new)  # type: ignore[arg-type]
        elif name in OPENVINS_KNOBS or name in OPENVINS_FLAGS:
            self._params.set_parameters([Parameter(name, value=new)])

    def _on_health(self, msg: DiagnosticStatus) -> None:
        """One frame of OpenVINS: the recovery it completes, the watch's verdict."""
        health = Health.parse(health_values(msg.values), str(msg.hardware_id))
        if health is None:
            self._counts["health_bad"] += 1
            return
        now = self._now()
        self._counts["health"] += 1
        self._counts["initialized" if health.initialized else "initialising"] += 1
        self._health = health
        done = self._recoveries.health(now, health)
        if done is not None and done.seconds is not None:
            self.get_logger().info(
                f"OpenVINS back {done.seconds:.2f} s after the reset ({done.reason}), poses in"
                f" {health.frame}: {done.seed or 'seeded'}"
            )
        if not self._switches.on("vio_watch"):
            return
        reason = self._watch.observe(health, self._reference)
        if reason is not None:
            self._counts["lost_" + reason.split(":", 1)[0]] += 1
            self._recover(reason)

    def _on_ekf(self, msg: Odometry) -> None:
        """The EKF's twist, carried to the head IMU while the head is still: the velocity seed."""
        now = self._now()
        transform = self._tf.pose(IMU_FRAME, BASE_FRAME)
        if transform is None:
            self._counts["seed_no_tf"] += 1
            return
        self._head.reading(now, transform.rotation)
        if not self._head.still(now):
            self._counts["seed_head_moving"] += 1
            return
        t = msg.twist.twist
        v_i, cov_i = imu_velocity(
            (t.linear.x, t.linear.y, t.angular.z),
            twist_block(msg.twist.covariance),
            transform.rotation,
            transform.translation,
            float(self._switches["seed_vz_sigma_m_s"]),
        )
        out = TwistWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = IMU_FRAME
        out.twist.twist.linear.x, out.twist.twist.linear.y, out.twist.twist.linear.z = (
            float(v) for v in v_i
        )
        covariance = [0.0] * 36
        for r in range(3):
            for c in range(3):
                covariance[6 * r + c] = float(cov_i[r, c])
        out.twist.covariance = covariance
        self._seed_pub.publish(out)
        self._counts["seed"] += 1
        self._reference = (stamp_seconds(msg.header.stamp), v_i)
        self._last_seed = (
            f"|v_I| {math.sqrt(float(v_i @ v_i)):.3f} m/s, sigma"
            f" {math.sqrt(max(float(cov_i.trace()), 0.0) / 3.0):.3f}"
        )

    def _on_restart(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """The relay's door: its verdict recovers OpenVINS as ``vio_recover`` says."""
        self._counts["lost_relay"] += 1
        response.success, response.message = self._recover("the relay's /vio/restart")
        return response

    def _on_tf_failure(self, kind: str, text: str) -> None:
        self._tf_failure = f"{kind}: {text}"

    # ---- recovery ----------------------------------------------------------------------------
    def _recover(self, reason: str) -> tuple[bool, str]:
        """Bring OpenVINS back: a reset in its process, or a restart of it; ``(done, text)``."""
        mode = str(self._switches["vio_recover"])
        now = self._now()
        self._last_reason = reason
        if mode == "off":
            text = f"OpenVINS lost ({reason}); vio_recover off: nothing done"
            self.get_logger().warning(text)
            return False, text
        epoch = self._health.epoch if self._health is not None else 0
        self._recoveries.asked(now, reason, epoch)
        self._face.event(VIO_RESTART)
        if mode == "restart":
            return self._restart_process(reason)
        if not self._reset.service_is_ready():
            self._counts["reset_unserved"] += 1
            if self._switches.on("vio_restart_fallback"):
                return self._restart_process(f"{reason}; {RESET_SERVICE} not served")
            text = f"OpenVINS lost ({reason}) but {RESET_SERVICE} is not served"
            self.get_logger().warning(text)
            return False, text
        self._counts["reset_asked"] += 1
        self.get_logger().warning(f"OpenVINS lost ({reason}): {RESET_SERVICE}")
        self._push_openvins_knobs(then=lambda: self._call_reset(reason))
        return True, f"reset asked ({reason})"

    def _call_reset(self, reason: str) -> None:
        """The reset call itself, its answer logged; unanswered for RESET_ANSWER_S, the fallback."""
        future = self._reset.call_async(Trigger.Request())
        asked = self._now()

        def done(f: Any) -> None:
            response = f.result()
            if response is None:
                return
            self._counts["reset_answered"] += 1
            self.get_logger().info(f"{RESET_SERVICE}: {response.message}")

        future.add_done_callback(done)

        def check() -> None:
            timer.cancel()
            if future.done():
                return
            self._counts["reset_unanswered"] += 1
            waited = self._now() - asked
            if self._switches.on("vio_restart_fallback"):
                self._restart_process(f"{reason}; {RESET_SERVICE} silent for {waited:.1f} s")

        timer = self.create_timer(RESET_ANSWER_S, check)

    def _restart_process(self, reason: str) -> tuple[bool, str]:
        """The launch's SIGINT to OpenVINS (respawned in 2 s, initialises at rest)."""
        code = self._run(SIGNAL)
        if code == 0:
            self._counts["restarted"] += 1
            text = (
                f"{PROCESS} signalled ({reason}): respawned in 2 s, it initialises at rest on the"
                " next motion"
            )
        else:
            text = f"no {PROCESS} process to signal (pkill exit {code}; {reason})"
        self.get_logger().warning(text)
        return code == 0, text

    def _push_openvins_knobs(self, then: Callable[[], None] | None = None) -> None:
        """OpenVINS's seed knobs and flag onto its parameters; ``then`` once they are set, or
        straight away when its parameter service is not up (OpenVINS keeps the launch's), and at
        the latest after RESET_ANSWER_S: a set that never answers delays a reset, never stops it."""
        s = self._switches
        params = [Parameter(n, value=s[n]) for n in (*OPENVINS_KNOBS, *OPENVINS_FLAGS)]
        ready = getattr(self._params, "services_are_ready", None)
        if then is not None and ready is not None and not ready():
            then()
            return
        future = self._params.set_parameters(params)
        if then is None:
            return
        if future is None or not hasattr(future, "add_done_callback"):
            then()
            return
        fired = [False]

        def once(*_args: Any) -> None:
            if not fired[0]:
                fired[0] = True
                then()

        def late() -> None:
            timer.cancel()
            once()

        future.add_done_callback(once)
        timer = self.create_timer(RESET_ANSWER_S, late)

    def _seed_camera_edges(self) -> str | None:
        """camera_stream's static camera edges put into this node's TF buffer from
        config/camera.json, as the relay does (a late /tf_static joiner waited ~2 min on
        2026-10-05); the neck chain stays live TF."""
        buffer = getattr(self._tf, "buffer", None)
        if buffer is None:
            return None
        try:
            camera = load_camera_mounts()
        except (OSError, KeyError, ValueError) as exc:
            self.get_logger().warning(
                f"config/camera.json unreadable ({exc}): the camera edges wait for /tf_static"
            )
            return None
        edges = camera_edges(camera, self.get_clock().now().to_msg())
        for edge in edges:
            buffer.set_transform_static(edge, "config/camera.json")
        return ", ".join(f"{e.header.frame_id} -> {e.child_frame_id}" for e in edges)

    # ---- the report --------------------------------------------------------------------------
    def _report(self) -> None:
        """The window's health, verdicts, recoveries and seeds, then the flags."""
        c, self._counts = self._counts, Counter()
        h = self._health
        state = "no health yet"
        if h is not None:
            state = (
                f"{'initialized' if h.initialized else 'initialising'} in {h.frame} (epoch"
                f" {h.epoch}), {h.tracked} tracked, {h.used} used, resets {h.resets} (warm"
                f" {h.warm_seeds}, standard {h.standard_inits})"
            )
        last = f" (last: {self._last_reason})" if self._last_reason else ""
        seed = f" (last: {self._last_seed})" if self._last_seed else ""
        tf = f" (tf: {self._tf_failure})" if c["seed_no_tf"] and self._tf_failure else ""
        self.get_logger().info(
            f"vio keeper: health {c['health'] / REPORT_S:.1f}/s ({c['initialized']} initialised,"
            f" {c['initialising']} initialising), {state}; lost: disagree {c['lost_disagree']},"
            f" dark {c['lost_dark']}, runaway"
            f" {c['lost_runaway']}, relay {c['lost_relay']}{last}; resets asked"
            f" {c['reset_asked']}, answered {c['reset_answered']}, unanswered"
            f" {c['reset_unanswered']}, unserved {c['reset_unserved']}, restarts"
            f" {c['restarted']}; {self._recoveries.report()}; seeds {c['seed'] / REPORT_S:.1f}/s"
            f"{seed}, withheld: head moving {c['seed_head_moving']}, no tf {c['seed_no_tf']}{tf};"
            f" flags: {self._switches.state()}"
        )

    def close(self) -> None:
        """Stop the TF listener's thread before the node is destroyed."""
        close = getattr(self._tf, "close", None)
        if close is not None:
            close()


def main() -> None:
    spin_main(VioKeeper)


if __name__ == "__main__":
    main()
