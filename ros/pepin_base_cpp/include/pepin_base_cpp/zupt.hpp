// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The zero-velocity update of a parked cart: when the bridge may tell the EKF that the cart is not
// moving (/zupt, ekf.yaml's odom2), and how much that claim weighs — the C++ twin of pepin.zupt
// (src/pepin/zupt.py). Same vetoes in the same order, same verdict words, same covariance.
//
// WHY, MEASURED. Parked on its charger on 2026-09-24 the EKF's heading crept ~0.08 deg/min, about
// 5 deg an hour, while its yaw-rate sources said over 120 s: the gyro after the bias tracker
// -0.001 deg/min, the wheels 0, the camera's VO -0.57, rf2o +1.5
// (scratch/link_autopsy/rest_yaw_sources.py). ekf.yaml already fused a zero-velocity update, but
// its only publisher was the lidar tracker's slip watch, which does not start under
// PEPIN_LOCALIZER=rtabmap.
//
// THE PYTHON SIDE IS THE REFERENCE. tests/unit/test_zupt.py pins the contract this file
// reproduces (test_zupt_contract_the_cpp_bridge_mirrors is written for exactly that purpose);
// test/zupt_contract.cpp replays the same table against this header and compiles with one c++
// line, without ROS or ament. Anything changed here changes there first.

#ifndef PEPIN_BASE_CPP__ZUPT_HPP_
#define PEPIN_BASE_CPP__ZUPT_HPP_

#include <array>
#include <cmath>
#include <cstddef>

namespace pepin
{

// What the update claims on vx, vy and vyaw: a variance of 1e-6, a sigma of 1 mm/s and 1 mrad/s.
// Tight because the claim is true — ZuptGate only lets it out while three witnesses agree that the
// cart is standing still — and because a loose one is out-voted by the very source it exists to
// answer. Beside it the filter hears the gyro at 4e-4, rf2o at 2.5e-3 (vx 9e-4), the wheels at
// 1e-3 (vyaw 0.01) and the camera's differenced yaw at ~1.6e-3 (ros/params/ekf.yaml): 1e-6 is 400x
// under the tightest of them. The EKF's own structure sets the rest (scratch/zupt_variance_sim.py,
// the yaw half of ekf.yaml fed the sources' measured means at rest): its process noise regrows
// vyaw's uncertainty between updates, so an update tighter than ~1e-5 is already a reset of vyaw,
// and 1e-5, 1e-6 and 1e-8 cut the modelled rf2o pull alike (7.9x, 13.2x, 14.3x, median over
// phases, at 10 Hz) where 4e-4 — the slip watch's own yaw sigma — cuts it 1.3x.
// robot_localization raises any variance under 1e-9 to 1e-9 (ekf.cpp:145): 1e-6 is clear of that.
constexpr double kRestZuptVariance = 1e-6;
// ...and what it says about everything it does not claim (vz, vroll, vpitch; and the pose).
constexpr double kUnclaimedVariance = 1e6;
// |bias-corrected yaw rate| at or above this is a turn: 0.005 rad/s = 0.29 deg/s. The parked chip's
// per-sample noise is 0.036 deg/s and the largest |yaw - bias| of the 30 s at-rest trace is
// 0.139 deg/s (scratch/zupt_gyro_quiet_threshold.py over scratch/imu_level_gyro.csv, 2026-09-12)
// — 7.9 sigma and 2.1x the parked maximum — while a slow hand quarter turn in 10 s (9 deg/s) is
// 31x over it. The default of the bridge's `zupt_gyro_quiet_rad_s`.
constexpr double kGyroQuietRadS = 0.005;
// How often the update is published while the cart is at rest: the slip watch's 10 Hz. Past 1e-5
// the RATE matters more than the variance (scratch/zupt_variance_sim.py): at 10 Hz the update is
// phase-locked to rf2o's ~10 Hz scans, and the modelled pull it leaves ranges with the phase between
// them from -0.012 to +0.018 deg/min against +0.020 without it (median 13x less); at 50 Hz it is
// 12x less at every phase, and the zero-mean walk of the sources' noise shrinks with it. The
// default of the bridge's `zupt_rate_hz`, read at start.
constexpr double kZuptHz = 10.0;

/// Why a zero-velocity update is, or is not, published at one tick.
enum class ZuptVerdict
{
  kAtRest,       ///< every witness agrees: publish
  kCommanded,    ///< a non-zero /cmd_vel is younger than the command hold
  kNoRest,       ///< the wheels do not witness rest
  kSettling,     ///< they do, for less than the settle window
  kNoGyro,       ///< no fresh bias-corrected gyro sample
  kGyroTurning,  ///< the newest gyro sample, or one within the settle window, is a turn
};

/// The verdict in the words the report line prints — pepin.zupt.ZuptVerdict's values.
inline const char * describe(ZuptVerdict verdict)
{
  switch (verdict) {
    case ZuptVerdict::kAtRest: return "at rest";
    case ZuptVerdict::kCommanded: return "a command is live";
    case ZuptVerdict::kNoRest: return "the wheels do not witness rest";
    case ZuptVerdict::kSettling: return "settling after the last motion";
    case ZuptVerdict::kNoGyro: return "no gyro reading to judge by";
    case ZuptVerdict::kGyroTurning: return "the gyro reports a turn";
  }
  return "unknown";
}

/// Each witness's last word, as a time on the one monotonic clock the bridge's threads share;
/// 0 means never (or, for `still_since`, not at rest).
///
/// `still_since` is rest_witnessed()'s answer (gyro_bias.hpp) — the start of the wheels' rest
/// spell, already 0 when the witness is stale. `command_at` is the last non-zero /cmd_vel.
/// `gyro_at` is the last bias-corrected gyro sample, `gyro_turn_at` the last one gyro_turning()
/// called a turn.
struct RestEvidence
{
  double still_since = 0.0;
  double command_at = 0.0;
  double gyro_at = 0.0;
  double gyro_turn_at = 0.0;
};

/// True when a bias-corrected yaw rate (rad/s) is a turn: at or above `quiet_rad_s` in magnitude.
/// A NaN is a turn: a reading nobody can judge is not a quiet one.
inline bool gyro_turning(double yaw_rate, double quiet_rad_s)
{
  return !(std::abs(yaw_rate) < quiet_rad_s);
}

/// Whether the cart is CERTAINLY standing still at one moment — the rule behind /zupt.
///
/// Five vetoes, checked in this order, the first that holds being the verdict. A non-zero command
/// younger than `command_hold_s` (the bridge's own `cmd_timeout_s`: the age up to which it keeps
/// re-sending a command to the board) — the operator's intent is known before any wheel turns. No
/// rest witnessed by the wheels (RestWitness: they moved, a command is being applied, the stream
/// broke, or nobody is watching). Rest witnessed for less than `settle_s` (`imu_bias_s`: the window
/// the bias tracker waits for the chassis to stop rocking before it trusts a sample). No gyro
/// sample younger than `max_gap_s` (no IMU, no bias yet, or reads failing): without it a cart
/// turned by hand on still wheels cannot be told from a parked one, so there is no update. And a
/// gyro turn — the newest sample, or any within `settle_s`: a turn is motion like any other and the
/// settle window is served after it just as after a wheel's move.
///
/// Pure and clockless, like GyroBiasTracker: every time comes from the caller.
class ZuptGate
{
public:
  /// `settle_s`: rest needed before the update, and the hold after a gyro turn. `command_hold_s`:
  /// how long a non-zero command vetoes. `max_gap_s`: how old the newest gyro sample may be (the
  /// wheels' own staleness is judged before, in rest_witnessed()).
  explicit ZuptGate(double settle_s = 0.0, double command_hold_s = 0.0, double max_gap_s = 1.0)
  : settle_s_(settle_s), command_hold_s_(command_hold_s), max_gap_s_(max_gap_s) {}

  /// The verdict at monotonic time `now`; only ZuptVerdict::kAtRest publishes.
  ZuptVerdict judge(double now, const RestEvidence & evidence) const
  {
    if (evidence.command_at > 0.0 && now - evidence.command_at < command_hold_s_) {
      return ZuptVerdict::kCommanded;
    }
    if (evidence.still_since <= 0.0) {
      return ZuptVerdict::kNoRest;
    }
    if (now - evidence.still_since < settle_s_) {
      return ZuptVerdict::kSettling;
    }
    if (evidence.gyro_at <= 0.0 || now - evidence.gyro_at > max_gap_s_) {
      return ZuptVerdict::kNoGyro;
    }
    const double turn = evidence.gyro_turn_at;
    if (turn > 0.0 && (turn >= evidence.gyro_at || now - turn < settle_s_)) {
      return ZuptVerdict::kGyroTurning;
    }
    return ZuptVerdict::kAtRest;
  }

private:
  double settle_s_;
  double command_hold_s_;
  double max_gap_s_;
};

/// Row-major 6x6 twist covariance of the update: kRestZuptVariance on vx, vy and vyaw — the
/// indices ekf.yaml's odom2 fuses — and kUnclaimedVariance on the three it does not.
inline std::array<double, 36> rest_zupt_twist_covariance()
{
  const std::array<double, 6> diagonal = {
    kRestZuptVariance, kRestZuptVariance, kUnclaimedVariance,
    kUnclaimedVariance, kUnclaimedVariance, kRestZuptVariance};
  std::array<double, 36> matrix{};
  for (std::size_t i = 0; i < diagonal.size(); ++i) {
    matrix[i * 6 + i] = diagonal[i];
  }
  return matrix;
}

}  // namespace pepin

#endif  // PEPIN_BASE_CPP__ZUPT_HPP_
