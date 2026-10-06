// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// /odom's twist covariance as a law of the MEASURED wheel motion — the C++ twin of
// pepin.wheel_noise (src/pepin/wheel_noise.py), which is the reference: the same numbers from the
// same inputs, held by tests/unit/test_base_cpp_contracts.py through test/wheel_noise_contract.cpp.
//
// While the wheels move (|v| >= moving_m_s or |w| >= moving_rad_s) and for hold_s after their last
// moving sample, the 1 s sigma of their mean forward speed is v_per_yaw_rate |w| + v_floor_m_s and
// of their mean yaw rate yaw_per_yaw_rate |w| + yaw_floor_rad_s, and each sample of a rate_hz
// stream carries rate_hz sigma^2 on vx (index 0) and vyaw (index 35); every other entry, and the
// whole matrix at rest past the hold, is the constant one the caller passes in. Fit on drives
// 0329-0347 against the lidar truth (config/base.json `odometry_noise` has the numbers and CIs).
//
// Cost: two comparisons, four multiply-adds and a 36-double copy per state line.

#ifndef PEPIN_BASE_CPP__WHEEL_NOISE_HPP_
#define PEPIN_BASE_CPP__WHEEL_NOISE_HPP_

#include <array>
#include <cmath>

#include "pepin_base_cpp/twist_from_pose.hpp"

namespace pepin
{

/// The coefficients of the law; the defaults are the fit's (config/base.json).
struct WheelNoiseLaw
{
  double v_per_yaw_rate = 0.034;  // m/s of sigma_v per rad/s of |w|
  double v_floor_m_s = 0.026;
  double yaw_per_yaw_rate = 0.20;  // rad/s of sigma_w per rad/s of |w|
  double yaw_floor_rad_s = 0.038;
  double moving_m_s = 0.03;
  double moving_rad_s = 0.052;
  double hold_s = 2.0;  // the law stays on this long after the last moving sample
  double rate_hz = 50.0;  // the state lines' rate: one sample's share of a second
};

/// The measured twist is motion, not rest.
inline bool wheels_moving(const WheelNoiseLaw & law, const BodyTwist & measured)
{
  return std::fabs(measured.linear) >= law.moving_m_s ||
         std::fabs(measured.angular) >= law.moving_rad_s;
}

/// Whether the law applies to a sample: the wheels move, or moved within hold_s before it.
/// pepin.wheel_noise.WheelNoise's gate; one per stream, fed every sample in order.
class WheelNoiseGate
{
public:
  /// Feed one sample; true while the law applies to it. A stamp before the last motion (a board
  /// clock that went backwards) is rest, as in the Python twin.
  bool on(const WheelNoiseLaw & law, const BodyTwist & measured, double stamp_s)
  {
    if (wheels_moving(law, measured)) {
      primed_ = true;
      last_moving_s_ = stamp_s;
      return true;
    }
    const double since = stamp_s - last_moving_s_;
    return primed_ && since >= 0.0 && since <= law.hold_s;
  }

  /// Forget the last motion: the next standing sample is rest at once.
  void reset() {primed_ = false;}

private:
  bool primed_ = false;
  double last_moving_s_ = 0.0;
};

/// The twist covariance for one /odom sample: the law on vx and vyaw when `on`, else `constant`.
inline std::array<double, 36> wheel_twist_covariance(
  const WheelNoiseLaw & law, const BodyTwist & measured, const std::array<double, 36> & constant,
  bool on)
{
  std::array<double, 36> matrix = constant;
  if (!on) {
    return matrix;
  }
  const double turn = std::fabs(measured.angular);
  const double sigma_v = law.v_per_yaw_rate * turn + law.v_floor_m_s;
  const double sigma_w = law.yaw_per_yaw_rate * turn + law.yaw_floor_rad_s;
  matrix[0] = law.rate_hz * sigma_v * sigma_v;
  matrix[35] = law.rate_hz * sigma_w * sigma_w;
  return matrix;
}

}  // namespace pepin

#endif  // PEPIN_BASE_CPP__WHEEL_NOISE_HPP_
