// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The body twist the wheels MEASURED, differenced off two consecutive wheel poses — the C++
// twin of pepin.odometry.TwistFromPose (src/pepin/odometry.py:229). Same projection, same
// wrap, same dt guard, same zero on the first sample, so the two bridges publish the same
// number from the same state lines.
//
// THE PYTHON SIDE IS THE REFERENCE. tests/unit/test_odometry.py pins the contract this file
// reproduces (test_twist_from_pose_contract_the_cpp_bridge_mirrors is written for exactly
// that purpose); anything changed here changes there first.

#ifndef PEPIN_BASE_CPP__TWIST_FROM_POSE_HPP_
#define PEPIN_BASE_CPP__TWIST_FROM_POSE_HPP_

#include <cmath>

namespace pepin
{

/// A planar body twist: forward speed in m/s, yaw rate in rad/s counter-clockwise.
struct BodyTwist
{
  double linear = 0.0;
  double angular = 0.0;
};

/// Map any angle to the interval (-pi, pi] — pepin.odometry.wrap_angle, atan2 and all.
inline double wrap_angle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

/// Turns consecutive wheel poses into the body twist the wheels actually measured.
///
/// The base server reports ``v`` and ``w`` as the twist it was COMMANDED to apply, not one it
/// measured: snapshot() copies self.twist, whatever /cmd_vel last asked for
/// (src/pepin/base_server.py:466). Its x/y/theta, on the other hand, are integrated from the
/// wheel travel and are a measurement. So the honest wheel velocity is the difference of two
/// consecutive poses over their gap, which is what this returns — and what a filter may fuse
/// without closing a loop from its own command back into its own state estimate.
///
/// Feed every pose as it arrives; the first one primes and returns a zero twist.
class TwistFromPose
{
public:
  /// ``max_gap_s``: a longer silence re-primes instead of dividing by a stale gap.
  explicit TwistFromPose(double max_gap_s = 1.0)
  : max_gap_s_(max_gap_s) {}

  /// Forget the last pose; the next sample primes again and returns a zero twist.
  void reset() {primed_ = false;}

  /// The body twist between the previous pose and this one, in m/s and rad/s.
  ///
  /// Forward speed is the straight-line step signed by the direction the robot was facing (a
  /// differential cart cannot move sideways, so the step is forward or backward); the yaw rate
  /// is the wrapped heading step over the gap. Returns a zero twist on the first sample and
  /// after a gap longer than ``max_gap_s`` — a board clock that jumped backwards counts as one.
  BodyTwist update(double x, double y, double theta, double stamp_s)
  {
    const double previous_x = x_;
    const double previous_y = y_;
    const double previous_theta = theta_;
    const double dt = stamp_s - stamp_;
    const bool had_pose = primed_;
    x_ = x;
    y_ = y;
    theta_ = theta;
    stamp_ = stamp_s;
    primed_ = true;
    if (!had_pose || dt <= 0.0 || dt > max_gap_s_) {
      return BodyTwist{0.0, 0.0};
    }
    const double dx = x - previous_x;
    const double dy = y - previous_y;
    const double forward = dx * std::cos(previous_theta) + dy * std::sin(previous_theta);
    return BodyTwist{forward / dt, wrap_angle(theta - previous_theta) / dt};
  }

private:
  double max_gap_s_;
  bool primed_ = false;
  double x_ = 0.0;
  double y_ = 0.0;
  double theta_ = 0.0;
  double stamp_ = 0.0;
};

}  // namespace pepin

#endif  // PEPIN_BASE_CPP__TWIST_FROM_POSE_HPP_
