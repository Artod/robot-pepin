// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The mast's sway from the head gyro, pure: no ROS, no thread, no clock of its own. The twin of
// pepin.mast (vio.md section 5).
//
// THE PHYSICS. The neck column bends between the shelf top and the pan servo; after a hard tilt
// the camera rings at 5.3 Hz, 0.3-0.4 deg p-p for 0.5-1.0 s, and wheel jerks shake it too
// (2026-10-02). The neck's encoders cannot see it. The head gyro sees everything: the cart's own
// rates, the neck joints' rates, and the sway; the sway is what is left.
//
// THE FILTER, per head sample:
//   1. the head rate (bias removed) in base_link's axes: R(base <- camera_link) at the latest
//      neck line, times R(camera_link <- head_imu) (config/camera.json's head_imu, static);
//   2. minus the base gyro's YAW rate only (the body's rock is part of what the picture sees and
//      nothing else in TF carries it);
//   3. HELD while the neck moves (differentiating 12-bit encoders at 50 Hz is +-4.4 deg/s of
//      quantisation, more than the sway): theta frozen at 0, the output NaN. ARMED at the first
//      still sample, when the joint rates are exactly zero;
//   4. integrated with a leak, theta += omega dt - theta dt / tau, tau = 1 / (2 pi crossover):
//      the encoders are the low-frequency truth (the mast's mean deflection is zero), so this IS
//      the complementary filter with its low-pass input identically zero;
//   5. for `arm_window_s` after arming the published theta is theta minus its running mean over
//      one ring period: the mast is already deflected when the encoders settle, and an
//      integrator started at 0 tracks theta - theta0 exp(-t/tau), an error the ring outlives.
// composed INTO the one base_link -> camera_link edge as a rotation about the hinge point
// (compose_sway), so every consumer of that edge gets the corrected camera without a line changed.

#ifndef PEPIN_BASE_CPP__MAST_HPP_
#define PEPIN_BASE_CPP__MAST_HPP_

#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <utility>

namespace pepin
{

constexpr double kMastPi = 3.14159265358979323846;

/// The filter's tunables (config/head_imu.json's mast block, parameters of the bridge).
struct MastSettings
{
  double crossover_hz = 0.5;   ///< the leak: tau = 1 / (2 pi crossover_hz), 0.32 s
  double arm_window_s = 0.5;   ///< after arming, theta minus its running mean over one ring
  double ring_hz = 5.3;        ///< the running mean's length: one period of the ring
  double max_dt_s = 0.05;      ///< a sample gap longer than this re-arms (a stalled link)
};

/// One output: the sway angles (roll, pitch, yaw about base_link's axes, rad) and rates
/// (rad/s), or `held` (both NaN) while the neck moves or nothing vouches for them.
struct MastOutput
{
  std::array<double, 3> theta{};
  std::array<double, 3> omega{};
  bool held = true;
};

using Matrix3 = std::array<double, 9>;  ///< row-major

/// The rotation yaw * pitch * roll (ROS's rpy convention, pepin.camera.quaternion_from_rpy).
inline Matrix3 rotation_from_rpy(double roll, double pitch, double yaw)
{
  const double cr = std::cos(roll), sr = std::sin(roll);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  const double cy = std::cos(yaw), sy = std::sin(yaw);
  return {
    cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
    -sp, cp * sr, cp * cr};
}

/// a * b.
inline Matrix3 multiply(const Matrix3 & a, const Matrix3 & b)
{
  Matrix3 out{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k) {
        sum += a[i * 3 + k] * b[k * 3 + j];
      }
      out[i * 3 + j] = sum;
    }
  }
  return out;
}

/// m * v.
inline std::array<double, 3> rotate(const Matrix3 & m, const std::array<double, 3> & v)
{
  return {
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

/// x, y, z, w of a rotation matrix (Shepperd's method: stable for every rotation).
inline std::array<double, 4> quaternion_from_matrix(const Matrix3 & m)
{
  const double trace = m[0] + m[4] + m[8];
  double x, y, z, w;
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    w = 0.25 * s;
    x = (m[7] - m[5]) / s;
    y = (m[2] - m[6]) / s;
    z = (m[3] - m[1]) / s;
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const double s = std::sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0;
    w = (m[7] - m[5]) / s;
    x = 0.25 * s;
    y = (m[1] + m[3]) / s;
    z = (m[2] + m[6]) / s;
  } else if (m[4] > m[8]) {
    const double s = std::sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0;
    w = (m[2] - m[6]) / s;
    x = (m[1] + m[3]) / s;
    y = 0.25 * s;
    z = (m[5] + m[7]) / s;
  } else {
    const double s = std::sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0;
    w = (m[3] - m[1]) / s;
    x = (m[2] + m[6]) / s;
    y = (m[5] + m[7]) / s;
    z = 0.25 * s;
  }
  return {x, y, z, w};
}

/// The head's angular rate in base_link's axes: `head_rate` in the chip's axes, through the
/// static R(camera_link <- head_imu) and the neck's R(base <- camera_link) = Rz(pan) Ry(pitch).
inline std::array<double, 3> head_rate_in_base(
  const std::array<double, 3> & head_rate, const Matrix3 & camera_from_imu, double pan_rad,
  double pitch_rad)
{
  return rotate(multiply(rotation_from_rpy(0.0, pitch_rad, pan_rad), camera_from_imu), head_rate);
}

/// The neck's base_link -> camera_link with the sway composed in as a rotation about the hinge
/// point H: T(H) R(theta) T(-H) T_neck. Returns the translation and the quaternion (x, y, z, w).
/// theta = 0 is exactly the neck's own edge.
inline std::pair<std::array<double, 3>, std::array<double, 4>> compose_sway(
  const std::array<double, 3> & theta, const std::array<double, 3> & hinge,
  const std::array<double, 3> & neck_xyz, double neck_roll, double neck_pitch, double neck_yaw)
{
  const Matrix3 sway = rotation_from_rpy(theta[0], theta[1], theta[2]);
  const std::array<double, 3> lever{
    neck_xyz[0] - hinge[0], neck_xyz[1] - hinge[1], neck_xyz[2] - hinge[2]};
  const auto turned = rotate(sway, lever);
  const std::array<double, 3> xyz{
    hinge[0] + turned[0], hinge[1] + turned[1], hinge[2] + turned[2]};
  const Matrix3 rotation = multiply(sway, rotation_from_rpy(neck_roll, neck_pitch, neck_yaw));
  return {xyz, quaternion_from_matrix(rotation)};
}

/// The leaky integrator with its hold, its arming and its arming-window mean (the header's 3-5).
class MastFilter
{
public:
  explicit MastFilter(MastSettings settings = MastSettings())
  : settings_(settings) {}

  /// The neck moved, the head link was silent or nothing vouches: theta back to 0, held.
  void hold()
  {
    held_ = true;
    theta_ = {0.0, 0.0, 0.0};
    window_.clear();
    sum_ = {0.0, 0.0, 0.0};
    has_last_ = false;
  }

  /// Whether the output is held right now.
  bool held() const {return held_;}

  /// One sample at `t` (s): the head's rate in base_link's axes (head_rate_in_base, bias
  /// removed) and the base gyro's yaw rate (rad/s, bias removed). Arms on the first sample after
  /// a hold; a gap longer than max_dt_s re-arms.
  MastOutput update(double t, const std::array<double, 3> & head_rate_base, double base_yaw_rate)
  {
    const std::array<double, 3> omega{
      head_rate_base[0], head_rate_base[1], head_rate_base[2] - base_yaw_rate};
    for (double value : omega) {
      if (!std::isfinite(value) || !std::isfinite(t)) {
        hold();
        return held_output();
      }
    }
    const bool gap = has_last_ && (t <= last_t_ || t - last_t_ > settings_.max_dt_s);
    if (held_ || !has_last_ || gap) {  // arm: from zero, with a fresh arming window
      hold();
      held_ = false;
      armed_at_ = t;
      last_t_ = t;
      has_last_ = true;
      remember(t);
      return output(t, omega);
    }
    const double dt = t - last_t_;
    last_t_ = t;
    const double tau = 1.0 / (2.0 * kMastPi * settings_.crossover_hz);
    for (int axis = 0; axis < 3; ++axis) {
      theta_[axis] += omega[axis] * dt - theta_[axis] * dt / tau;
    }
    remember(t);
    return output(t, omega);
  }

  /// The integrator's own theta (no arming correction), for tests.
  const std::array<double, 3> & raw_theta() const {return theta_;}

private:
  MastOutput held_output() const
  {
    MastOutput out;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    out.theta = {nan, nan, nan};
    out.omega = {nan, nan, nan};
    out.held = true;
    return out;
  }

  void remember(double t)
  {
    window_.emplace_back(t, theta_);
    for (int axis = 0; axis < 3; ++axis) {
      sum_[axis] += theta_[axis];
    }
    const double period = settings_.ring_hz > 0.0 ? 1.0 / settings_.ring_hz : 0.0;
    while (!window_.empty() && t - window_.front().first >= period) {
      for (int axis = 0; axis < 3; ++axis) {
        sum_[axis] -= window_.front().second[axis];
      }
      window_.pop_front();
    }
  }

  MastOutput output(double t, const std::array<double, 3> & omega) const
  {
    MastOutput out;
    out.held = false;
    out.omega = omega;
    const bool arming = t - armed_at_ < settings_.arm_window_s && !window_.empty();
    for (int axis = 0; axis < 3; ++axis) {
      const double mean = arming ? sum_[axis] / static_cast<double>(window_.size()) : 0.0;
      out.theta[axis] = theta_[axis] - mean;
    }
    return out;
  }

  MastSettings settings_;
  bool held_ = true;
  bool has_last_ = false;
  double last_t_ = 0.0;
  double armed_at_ = 0.0;
  std::array<double, 3> theta_{};
  std::deque<std::pair<double, std::array<double, 3>>> window_;
  std::array<double, 3> sum_{};
};

}  // namespace pepin

#endif  // PEPIN_BASE_CPP__MAST_HPP_
