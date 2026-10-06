"""OpenVINS brought back by itself after it lost the picture, without restarting its process.

OpenVINS has no failure detector and no reset of its own: once a dark stretch or a blank wall has
left it propagating on the IMU alone, it diverges and stays diverged (drive 0330, 2026-10-05: no
VIO after the dark printer, even under the lamp at home). Our patch of its ROS 2 wrapper
(ros/patches/openvins-reset.patch) adds the mechanism: ``/ov_msckf/reset`` makes a new filter in
the same process, the way ModalAI's voxl-open-vins-server does its hard reset, and seeds it WARM
on the first frame that has a picture — the old filter's biases from a healthy moment, gravity
from the accelerometer, and a velocity from outside — so it publishes again about half a second
later instead of waiting for a second of stillness. ``/ov_msckf/health`` says per frame what it
sees. This module is the policy around it, with no ROS in it (pepin_bringup.vio_keeper is the
node):

- :func:`imu_velocity` carries the board EKF's body twist to the head IMU through the neck's
  transform: the velocity seed, with its covariance;
- :class:`HeadStill` says from that transform when the head is still relative to the cart,
  the only time that rigid-body carry holds;
- :class:`Health` reads one ``/ov_msckf/health`` message;
- :class:`VioWatch` is the failure detector: OpenVINS's velocity at the IMU disagreeing with
  the EKF's (the same seed) for ``disagree_s``, a dark stretch (almost no persistent tracks while
  the cart is not resting) for ``dark_s`` of still frames — a head swing is not dark, its frames
  neither count nor clear the run — and a runaway speed for ``speed_s``; nothing is judged in the
  grace after a reset or while the filter initialises;
- :class:`Recoveries` measures each recovery: from the verdict to the first initialised frame of
  the new filter.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DARK_S",
    "DARK_TRACKS",
    "DISAGREE_M_S",
    "DISAGREE_S",
    "GRACE_S",
    "HEAD_STILL_DEG",
    "HEAD_STILL_S",
    "MAX_SPEED_M_S",
    "OPENVINS_FLAGS",
    "OPENVINS_KNOBS",
    "SPEED_S",
    "SWING_RAD_S",
    "VZ_SIGMA_M_S",
    "HeadStill",
    "Health",
    "Recoveries",
    "Recovery",
    "VioWatch",
    "health_values",
    "imu_velocity",
    "twist_block",
]

# The knobs (config/knobs.json, vio_keeper) and the flag that are OpenVINS's own, read by the
# patched wrapper at the moment of use: vio.launch.py passes the file's defaults at OpenVINS's
# start, and pepin_bringup.vio_keeper pushes the live values at its start, on every change and
# before every reset (a respawned OpenVINS has the launch's values again).
OPENVINS_KNOBS = (
    "seed_min_features",
    "seed_track_frames",
    "seed_bias_max_age_s",
    "seed_bias_lag_s",
    "seed_accel_window_s",
    "seed_accel_allow_m_s2",
    "seed_gyro_max_rad_s",
    "seed_g_tol_m_s2",
    "seed_vel_max_age_s",
    "seed_sigma_q_rad",
    "seed_sigma_p_m",
    "seed_sigma_v_m_s",
    "seed_bias_inflation",
    "seed_sigma_bg_floor",
    "seed_sigma_ba_floor",
    "reset_frame_gap_s",
    "odomimu_delay_s",
)
# Their defaults: a warm seed after a reset, and no dynamic initialisation as its fallback (the
# config's init_dyn_use is the cold start's; a dynamic init took 7.6-8.9 s on this cart).
OPENVINS_FLAGS = {"seed_warm": True, "seed_dyn_init": False}

# The dark rule: fewer persistent tracks (the left eye's features seen in 3+ frames) than this on a
# frame with no swing and no zero-velocity update. Not the features the update USED: with the gaze
# moving that is 0-8 in good light (drives 0329/0330 replayed, scratch/vio_recover r4), while the
# tracker kept 20-45 persistent tracks even in the dimmest stretch.
DARK_TRACKS = 10
# How much of such frames make a verdict, seconds (frames of a head swing are not counted).
DARK_S = 1.0
# A frame whose largest bias-corrected gyro rate since the last one is over this was taken in a
# swing: the picture blurs and the tracks end, which is not the dark. Saccades reach 5 rad/s, the
# cart turns at 0.5 at most (drives 0329/0330, scratch/vio_recover/frames.py).
SWING_RAD_S = 1.0
# The disagreement rule: OpenVINS's velocity at the IMU farther than this from the EKF's (the
# keeper's seed: the board EKF's twist carried to the IMU, only while the head is still) for
# DISAGREE_S of frames with a reference. The replayed divergences ran 0.2 -> 0.5 m/s off within 1 s;
# healthy filters stayed within 0.02-0.17 of the wheels.
DISAGREE_M_S = 0.2
DISAGREE_S = 1.0
# A reference velocity farther than this from the frame's time says nothing about it, seconds.
REFERENCE_AGE_S = 0.3
# The runaway rule: OpenVINS's own IMU speed over this for SPEED_S. The cart's cap is 0.30 m/s and a
# head pan moves the IMU at a few cm/s; OpenVINS's 2026-10-04 divergence went to km.
MAX_SPEED_M_S = 1.0
SPEED_S = 0.3
# After a reset nothing is judged until the new filter is initialised and this long has passed
# (ModalAI's ok_state_grace_timeout_s 2.0): its first second has few SLAM features by
# construction.
GRACE_S = 2.0
# The head is still when its orientation in base_link turned no more than this over the last
# HEAD_STILL_S (degrees; the encoders' jitter is ~0.1 deg, the mast's sway up to ~1).
HEAD_STILL_S = 0.3
HEAD_STILL_DEG = 1.0
# The cart's vertical speed is not in the planar EKF: zero with this sigma (the mast sways).
VZ_SIGMA_M_S = 0.02
# An encoder reading older than this says nothing about the head now.
NECK_FRESH_S = 0.5

Array = NDArray[np.float64]


def twist_block(covariance: Iterable[float]) -> Array:
    """The (vx, vy, wz) block of a 6x6 row-major twist covariance (``nav_msgs/Odometry``'s)."""
    full = np.asarray(list(covariance), dtype=float).reshape(6, 6)
    index = [0, 1, 5]
    return np.asarray(full[np.ix_(index, index)])


def imu_velocity(
    twist: tuple[float, float, float],
    covariance: Array,
    r_i_b: Array,
    p_i_b: Array,
    vz_sigma_m_s: float = VZ_SIGMA_M_S,
) -> tuple[Array, Array]:
    """The head IMU's velocity in its own axes and its 3x3 covariance, from base_link's twist.

    ``twist`` is ``(vx, vy, wz)`` in base_link (the EKF's ``/odometry/filtered``, child frame
    base_link) and ``covariance`` its 3x3 block (:func:`twist_block`); ``r_i_b``, ``p_i_b`` are
    base_link's pose in the IMU's frame (TF ``head_imu <- base_link``: x_I = R_I_B x_B + p_I_B).
    With the head still on the neck the two are one rigid body, so the IMU's velocity is the base's
    plus the turn's lever arm, ``v_B + w x r_BI`` with ``r_BI = -R_I_B^T p_I_B`` (the IMU's origin
    in base_link), turned into the IMU's axes by ``R_I_B``. The covariance goes through the same
    linear map, plus the vertical speed's ``vz_sigma_m_s`` the planar EKF does not estimate."""
    r = np.asarray(r_i_b, dtype=float).reshape(3, 3)
    p = np.asarray(p_i_b, dtype=float).reshape(3)
    r_bi = -r.T @ p
    vx, vy, wz = (float(v) for v in twist)
    v_b = np.array([vx - wz * r_bi[1], vy + wz * r_bi[0], 0.0])
    jacobian = np.array([[1.0, 0.0, -r_bi[1]], [0.0, 1.0, r_bi[0]], [0.0, 0.0, 0.0]])
    cov_b = jacobian @ np.asarray(covariance, dtype=float).reshape(3, 3) @ jacobian.T
    cov_b[2, 2] += vz_sigma_m_s**2
    cov_i = r @ cov_b @ r.T
    return r @ v_b, 0.5 * (cov_i + cov_i.T)


class HeadStill:
    """Whether the head is still on the cart: the IMU's orientation in base_link (TF ``head_imu <-
    base_link``: the neck's encoders and the mast, the chain the velocity is carried through)
    turned by no more than ``still_deg`` over the last ``still_s``, with a fresh reading. Times
    are the receiving node's clock."""

    def __init__(self, still_s: float = HEAD_STILL_S, still_deg: float = HEAD_STILL_DEG) -> None:
        self.still_s = still_s  # live: head_still_s
        self.still_deg = still_deg  # live: head_still_deg
        self._readings: deque[tuple[float, Array]] = deque()

    def reading(self, now: float, rotation: Array) -> None:
        """One orientation of the head in base_link (any consistent 3x3, e.g. R_I_B)."""
        self._readings.append((now, np.asarray(rotation, dtype=float).reshape(3, 3)))
        keep = max(self.still_s, NECK_FRESH_S) + 1.0
        while self._readings and self._readings[0][0] < now - keep:
            self._readings.popleft()

    def still(self, now: float) -> bool:
        """True when the head has not turned over the window (False with no fresh reading)."""
        if not self._readings or now - self._readings[-1][0] > NECK_FRESH_S:
            return False
        if self._readings[0][0] > now - self.still_s + 1e-6:
            return False  # not watched for the whole window yet
        newest = self._readings[-1][1]
        window = [r for t, r in self._readings if t >= now - self.still_s]
        limit = math.radians(self.still_deg)
        return all(_angle(newest @ r.T) <= limit for r in window)


def _angle(rotation: Array) -> float:
    """The angle of a 3x3 rotation, radians."""
    cos = (float(np.trace(rotation)) - 1.0) / 2.0
    return math.acos(min(1.0, max(-1.0, cos)))


@dataclass(frozen=True)
class Health:
    """One ``/ov_msckf/health`` message (diagnostic_msgs/DiagnosticStatus, key/value strings): the
    frame's camera time, the reset epoch, whether the filter publishes, the tracker's features
    (all, and those seen in 3+ frames), the features its update used (MSCKF + SLAM), whether the
    frame was a zero-velocity update, the largest gyro rate since the last frame, its IMU speed,
    and the reset counters."""

    t: float
    epoch: int
    initialized: bool
    tracked: int
    persistent: int
    msckf: int
    slam: int
    zupt: bool
    gyro: float
    speed: float | None
    v_i: tuple[float, float, float] | None
    resets: int
    warm_seeds: int
    standard_inits: int
    seed: str
    frame: str

    @property
    def used(self) -> int:
        """Features the update used: MSCKF + SLAM."""
        return self.msckf + self.slam

    @classmethod
    def parse(cls, values: Mapping[str, str], frame: str = "") -> Health | None:
        """From the message's key/value pairs (and its ``hardware_id``, the poses' frame); ``None``
        when a required key is missing or not a number."""
        try:
            speed = values.get("speed")
            v_i = values.get("v_I")
            return cls(
                t=float(values["t"]),
                epoch=int(values["epoch"]),
                initialized=values["initialized"] == "1",
                tracked=int(values["tracked"]),
                persistent=int(values.get("persistent", values["tracked"])),
                msckf=int(values["msckf"]),
                slam=int(values["slam"]),
                zupt=values["zupt"] == "1",
                gyro=float(values["gyro"]),
                speed=None if speed is None else float(speed),
                v_i=None if v_i is None else _vector(v_i),
                resets=int(values.get("resets", values["epoch"])),
                warm_seeds=int(values.get("warm_seeds", "0")),
                standard_inits=int(values.get("standard_inits", "0")),
                seed=str(values.get("seed", "")),
                frame=frame,
            )
        except (KeyError, ValueError):
            return None


class VioWatch:
    """The failure detector on ``/ov_msckf/health``: ``observe`` answers the reason OpenVINS is
    lost, once per epoch (a reset starts the next), or ``None``.

    DISAGREE: the frame's velocity at the IMU (health ``v_I``) farther than ``disagree_m_s`` from
    the reference — the EKF's velocity carried to the IMU, given with its time, used within
    ``REFERENCE_AGE_S`` of the frame — adds the frame's interval to a run; ``disagree_s`` of run is
    a verdict. A frame that agrees clears the run; one without a fresh reference (the head moving:
    the keeper publishes no seed then) neither adds nor clears.
    DARK: a frame with fewer than ``dark_tracks`` persistent tracks adds its interval to a run;
    ``dark_s`` is a verdict. A frame with enough tracks, or held by OpenVINS's zero-velocity update
    (the cart at rest, dark or not; ``dark_at_rest`` judges those too), clears it; a frame taken in
    a swing (gyro over ``swing_rad_s``) neither adds nor clears: a saccade ends tracks for a few
    frames in the best light, and a dark stretch swung through still adds up between the swings.
    RUNAWAY: OpenVINS's IMU speed over ``max_speed_m_s`` for ``speed_s``.
    Nothing is judged while the filter is not initialised, nor for ``grace_s`` after the first
    initialised frame of an epoch. Times are the frames' own (camera clock)."""

    def __init__(
        self,
        dark_tracks: int = DARK_TRACKS,
        dark_s: float = DARK_S,
        swing_rad_s: float = SWING_RAD_S,
        disagree_m_s: float = DISAGREE_M_S,
        disagree_s: float = DISAGREE_S,
        max_speed_m_s: float = MAX_SPEED_M_S,
        speed_s: float = SPEED_S,
        grace_s: float = GRACE_S,
        dark_at_rest: bool = False,
    ) -> None:
        self.dark_tracks = dark_tracks  # live: vio_dark_tracks
        self.dark_s = dark_s  # live: vio_dark_s
        self.swing_rad_s = swing_rad_s  # live: vio_swing_rad_s
        self.disagree_m_s = disagree_m_s  # live: vio_disagree_m_s
        self.disagree_s = disagree_s  # live: vio_disagree_s
        self.max_speed_m_s = max_speed_m_s  # live: vio_max_speed_m_s
        self.speed_s = speed_s  # live: vio_speed_s
        self.grace_s = grace_s  # live: vio_grace_s
        self.dark_at_rest = dark_at_rest  # live: vio_dark_at_rest
        self.counts = {"disagree": 0, "dark": 0, "runaway": 0}
        self.last: str | None = None
        self._epoch: int | None = None
        self._live_since: float | None = None  # the first initialised frame of this epoch
        self._last_t: float | None = None
        self._runs = {"disagree": 0.0, "dark": 0.0, "runaway": 0.0}
        self._lost = False  # this epoch already had its verdict

    def observe(self, health: Health, reference: tuple[float, Array] | None = None) -> str | None:
        """One frame's health and the latest reference velocity ``(time, v_I)``: the verdict
        when this frame makes it, else ``None``."""
        if health.epoch != self._epoch:
            self._epoch = health.epoch
            self._lost = False
            self._restart()
        if not health.initialized:
            self._restart()
            return None
        if self._live_since is None:
            self._live_since = health.t
        dt = 0.0 if self._last_t is None else min(max(health.t - self._last_t, 0.0), 0.5)
        self._last_t = health.t
        if self._lost or health.t - self._live_since < self.grace_s - 1e-6:
            return None
        reason = (
            self._disagree_rule(health, reference, dt)
            or self._dark_rule(health, dt)
            or self._runaway_rule(health, dt)
        )
        if reason is not None:
            self._lost = True
            self.last = reason
        return reason

    def _restart(self) -> None:
        self._live_since = None
        self._last_t = None
        self._runs = dict.fromkeys(self._runs, 0.0)

    def _run(self, rule: str, dt: float, limit: float) -> float | None:
        """Adds ``dt`` to ``rule``'s run; the run when it reached ``limit`` (the rule counted)."""
        self._runs[rule] += dt
        if self._runs[rule] >= limit - 1e-6:
            self.counts[rule] += 1
            return self._runs[rule]
        return None

    def _disagree_rule(
        self, health: Health, reference: tuple[float, Array] | None, dt: float
    ) -> str | None:
        if (
            health.v_i is None
            or reference is None
            or abs(health.t - reference[0]) > REFERENCE_AGE_S
        ):
            return None
        diff = float(np.linalg.norm(np.asarray(health.v_i) - np.asarray(reference[1])))
        if diff <= self.disagree_m_s:
            self._runs["disagree"] = 0.0
            return None
        run = self._run("disagree", dt, self.disagree_s)
        if run is None:
            return None
        return (
            f"disagree: {diff:.2f} m/s from the EKF's velocity at the IMU (over"
            f" {self.disagree_m_s:.2f}) for {run:.1f} s"
        )

    def _dark_rule(self, health: Health, dt: float) -> str | None:
        if health.persistent >= self.dark_tracks or (health.zupt and not self.dark_at_rest):
            self._runs["dark"] = 0.0
            return None
        if health.gyro > self.swing_rad_s:
            return None
        run = self._run("dark", dt, self.dark_s)
        if run is None:
            return None
        return (
            f"dark: {health.persistent} persistent tracks (under {self.dark_tracks},"
            f" {health.tracked} tracked) for {run:.1f} s of still frames"
        )

    def _runaway_rule(self, health: Health, dt: float) -> str | None:
        if health.speed is None or health.speed <= self.max_speed_m_s:
            self._runs["runaway"] = 0.0
            return None
        run = self._run("runaway", dt, self.speed_s)
        if run is None:
            return None
        return f"runaway: {health.speed:.2f} m/s (over {self.max_speed_m_s:.2f}) for {run:.1f} s"


@dataclass(frozen=True)
class Recovery:
    """One recovery: why, when the reset was asked (the receiving node's clock), and when the new
    filter's first initialised frame arrived (None until then), with how it was seeded."""

    reason: str
    asked: float
    epoch_before: int
    back: float | None = None
    seed: str = ""

    @property
    def seconds(self) -> float | None:
        """From the verdict to the new filter publishing, seconds."""
        return None if self.back is None else self.back - self.asked


class Recoveries:
    """The recoveries asked and how long each took to publish again, for the report line."""

    def __init__(self, keep: int = 50) -> None:
        self.done: deque[Recovery] = deque(maxlen=keep)
        self.open: Recovery | None = None

    def asked(self, now: float, reason: str, epoch: int) -> None:
        """A reset was asked now; the epoch the filter was in."""
        self.open = Recovery(reason=reason, asked=now, epoch_before=epoch)

    def health(self, now: float, health: Health) -> Recovery | None:
        """A health message: the recovery it completes (a later epoch, initialised), else None."""
        rec = self.open
        if rec is None or health.epoch <= rec.epoch_before or not health.initialized:
            return None
        done = Recovery(rec.reason, rec.asked, rec.epoch_before, now, health.seed)
        self.done.append(done)
        self.open = None
        return done

    def report(self) -> str:
        """The recoveries' count and their times to publish (p50, max), the open one."""
        times = sorted(r.seconds for r in self.done if r.seconds is not None)
        text = f"{len(self.done)} recovered"
        if times:
            text += f" in p50 {times[len(times) // 2]:.1f} s, max {times[-1]:.1f} s"
        warm = sum(1 for r in self.done if r.seed.startswith("warm"))
        text += f" ({warm} warm)"
        if self.open is not None:
            text += f", 1 waiting since its reset ({self.open.reason})"
        return text


def _vector(text: str) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in text.split(","))
    return (x, y, z)


def health_values(pairs: Iterable[Any]) -> dict[str, str]:
    """A DiagnosticStatus's ``values`` (KeyValue messages) as a dict."""
    return {str(p.key): str(p.value) for p in pairs}
