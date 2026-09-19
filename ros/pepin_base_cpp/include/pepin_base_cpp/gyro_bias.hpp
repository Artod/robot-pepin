// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The MPU6050's zero, re-measured from every block of rest the wheels witness — the C++ twin of
// pepin.gyro.GyroBiasTracker (src/pepin/gyro.py:32). Same settle window, same block size, same
// replacement by a block mean, same "not known to be still" sentinel, so the two bridges subtract
// the same number from the same wheel evidence.
//
// THE PYTHON SIDE IS THE REFERENCE. tests/unit/test_gyro.py pins the contract this file
// reproduces (test_gyro_bias_contract_the_cpp_bridge_mirrors is written for exactly that
// purpose); anything changed here changes there first. test/gyro_bias_contract.cpp replays the
// same table against this header and can be compiled with one clang++ line (its own comment says
// which) without ROS or ament.

#ifndef PEPIN_BASE_CPP__GYRO_BIAS_HPP_
#define PEPIN_BASE_CPP__GYRO_BIAS_HPP_

#include <algorithm>
#include <cmath>
#include <limits>

namespace pepin
{

/// A gyro zero in rad/s, on the chip's own axes (the bias is subtracted before mounting).
struct GyroBias
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

/// Float slack on "within one tick": pepin.gyro.TICK_SLACK.
constexpr double kTickSlack = 1e-6;

/// Whether the wheels have stayed within ONE encoder tick of where they came to rest — the twin
/// of pepin.gyro.TickDither. A parked cart does not read zero: live on 2026-09-19 the right
/// encoder flipped by one tick on every state line (dr -9.587e-05, +9.587e-05, ... m), so "the
/// measured twist is exactly zero" was never true and the rest block never came. One tick is the
/// encoder's quantisation unit, not a tuned tolerance; a creep of one tick a line in one direction
/// leaves the band on its second line.
class TickDither
{
public:
  explicit TickDither(double tick_m)
  : tick_m_(tick_m) {}

  /// One state line's wheel travel; true while neither wheel has left the band. A line that
  /// leaves it re-anchors the band where the wheels are now.
  bool still(double d_left_m, double d_right_m)
  {
    left_m_ += d_left_m;
    right_m_ += d_right_m;
    const double band = tick_m_ * (1.0 + kTickSlack);
    if (std::abs(left_m_) <= band && std::abs(right_m_) <= band) {
      return true;
    }
    left_m_ = 0.0;
    right_m_ = 0.0;
    return false;
  }

private:
  double tick_m_;
  double left_m_ = 0.0;
  double right_m_ = 0.0;
};

/// The wheels' word on whether the cart is standing still, boiled down to one timestamp.
///
/// The twin of pepin.gyro.RestWitness (src/pepin/gyro.py:37). Fed one wheel state line at a time,
/// it answers with the time since which rest has been witnessed without a break — which is the
/// time of the last motion when there is none, and 0 when the question cannot be answered at all.
/// A consumer on another thread needs nothing but that number and rest_witnessed().
///
/// A line witnesses rest only when all three hold. The MEASURED wheel twist is exactly zero: one
/// encoder tick is pi * 0.125 m / 4096 = 9.6e-5 m of travel (config/base.json), so a cart that
/// moves at all — driven, pushed, or turned by hand with the torque off — reports a non-zero
/// twist. The base server is applying no twist: `moving` is its own word on a live non-zero
/// command, whoever sent it, which is the case of a command whose blocked wheels never tick. And
/// the line continues an unbroken stream: across a gap longer than `max_gap_s` TwistFromPose
/// re-primes and returns a zero twist that measured nothing, while the cart may have moved the
/// whole time.
class RestWitness
{
public:
  /// `max_gap_s`: the same gap after which the twist estimator re-primes (TwistFromPose).
  explicit RestWitness(double max_gap_s = 1.0)
  : max_gap_s_(max_gap_s) {}

  /// The monotonic time the current rest spell began; 0 when the cart is not known to be still.
  double since() const {return since_;}

  /// The monotonic time of the last line judged — how fresh the answer above is.
  double at() const {return at_;}

  /// Start the stream and the spell over: what comes next measured nothing about the past.
  ///
  /// Called wherever the twist estimator is re-primed — /odom muted, say. Whatever makes the next
  /// twist not a measurement makes the rest it would witness unknown, and a zero twist that
  /// measured nothing must not be read as a cart standing still.
  void forget()
  {
    stamp_s_ = 0.0;
    since_ = 0.0;
  }

  /// One wheel state line; returns the spell start to publish (see since()).
  ///
  /// Two clocks on purpose: `stamp_s` is the board's own clock, on which the gap between
  /// consecutive state lines is measured, and `at` is the local monotonic clock the gyro's samples
  /// are stamped with, on which the rest is counted.
  double judge(double stamp_s, double at, bool moving, bool twist_is_zero)
  {
    const double gap = stamp_s - stamp_s_;
    const bool unbroken = stamp_s_ > 0.0 && gap > 0.0 && gap <= max_gap_s_;
    stamp_s_ = stamp_s;
    at_ = at;
    const bool still = unbroken && twist_is_zero && !moving;
    if (!still || since_ <= 0.0) {
      since_ = at;
    }
    return since_;
  }

private:
  double max_gap_s_;
  double stamp_s_ = 0.0;
  double at_ = 0.0;
  double since_ = 0.0;
};

/// The rest spell a reader on another thread may still believe in, or 0 if nobody is watching.
///
/// `since` and `at` are RestWitness's two numbers as they were last published (two atomics in the
/// bridge, so the 50 Hz gyro loop never waits on the wheels' reader thread). A witness older than
/// `max_gap_s` is no witness: the link may be down or /odom muted, and a cart nobody is watching
/// is not at rest, however still the last thing anyone saw was.
inline double rest_witnessed(double since, double at, double now, double max_gap_s)
{
  return (since <= 0.0 || now - at > max_gap_s) ? 0.0 : since;
}

/// The gyro's zero, replaced by the mean of every finished block of rest.
///
/// WHY, MEASURED. This node used to estimate the bias once, over the first `imu_bias_s` after
/// start, and subtract that number forever. The chip's bias moves with temperature: parked hours
/// after boot with the wheels blocked, /odom (the wheels) read 0.00 deg/min of yaw while
/// /odometry/filtered — the EKF, whose only yaw-rate source is this gyro (ros/params/ekf.yaml,
/// imu0 index 11) — read +0.19, -0.54 and +0.67 deg/min in three measurements of one night
/// (scratch/odom_drift_at_rest.py, 2026-09-19), and RTAB-Map, which builds its graph on that
/// odometry, turned its map +27 deg in 40 min and +88 deg in four hours under a cart that never
/// moved. On the level floor the chip's raw yaw axis reads -0.028 deg/s, which is -1.7 deg/min if
/// integrated (scratch/imu_level_*.csv, 2026-09-12).
///
/// THE CURE. While the cart stands still the gyro's reading IS its bias, and the wheels know when
/// it stands still — their twist is differenced off consecutive encoder poses (twist_from_pose.hpp)
/// and one tick is pi * 0.125 m / 4096 = 9.6e-5 m of travel, so a cart that moves at all reports a
/// non-zero twist. A REST BLOCK is `block_s * rate_hz` samples taken inside one unbroken spell of
/// witnessed rest that began at least `block_s` before the first of them (the chassis settling).
/// A finished block REPLACES the bias with its mean: one block, one mean, no gain and no time
/// constant to tune.
///
/// HOW QUIET THE MEAN IS, measured on the 30 s at-rest tape (scratch/gyro_block_mean_noise.py):
/// per-sample noise 0.036 deg/s, and the mean of a 2.0 s block scatters by 0.30 deg/min (1 sigma)
/// — so the cure does not reach zero, it turns a one-way creep into a zero-mean random walk:
/// 40 min of 2 s blocks accumulate ~0.35 deg against the +27 deg measured. A longer `imu_bias_s`
/// shrinks the error in force at any one moment, which is what a DRIVE inherits (0.21 deg/min at
/// 5 s, 0.02 at 10); the parked walk stays ~0.35 deg either way, the sqrt trade.
///
/// Pure and clockless: every time comes from the caller, no thread and no lock lives here. One
/// thread owns an instance (the IMU thread); the wheels' word reaches it through atomics.
class GyroBiasTracker
{
public:
  /// `block_s` seconds of rest per block, at `rate_hz` samples a second.
  ///
  /// Both are the node's existing parameters (`imu_bias_s`, `imu_rate_hz`): the block is the
  /// length the boot calibration already asked the operator to hold still for, and the settle
  /// window before it is the same length again. `block_s` <= 0 is "no calibration" — the zero bias
  /// is ready at once, as before. The default constructs that case, for a member declared before
  /// the parameters are read.
  explicit GyroBiasTracker(double block_s = 0.0, double rate_hz = 0.0)
  : settle_s_(block_s),
    samples_(block_s > 0.0 ? std::max(1L, std::lround(block_s * rate_hz)) : 0L),
    ready_(samples_ == 0) {}

  /// The zero to subtract from the chip's readings, rad/s.
  GyroBias bias() const {return bias_;}

  /// True once a bias exists; before that the caller publishes nothing.
  bool ready() const {return ready_;}

  /// How many rest blocks have been averaged into a bias since the node started.
  long blocks() const {return blocks_;}

  /// How many samples one block holds; 0 when no calibration was asked for.
  long block_samples() const {return samples_;}

  /// Seconds since the last block finished; infinity while none has.
  double age_s(double t) const
  {
    return blocks_ == 0 ? std::numeric_limits<double>::infinity() : t - at_s_;
  }

  /// One gyro sample at monotonic time `t`; true when a block just replaced the bias.
  ///
  /// `still_since` is the wheels' word: the monotonic time since which rest has been witnessed
  /// without a break, or <= 0 when the cart is not known to be standing still (it is moving, a
  /// command is live, or nobody is watching — the link is down or /odom is muted). It is compared
  /// for exact equality, not nearness: it is the identity of a rest spell, not a measurement, and
  /// the caller repeats the same value for every sample of one spell.
  ///
  /// A sample is ignored while the chassis settles (less than `block_s` since the spell began) and
  /// a spell that ends throws away whatever it had accumulated — so a block is never a mean across
  /// a move.
  bool update(double t, double gyro_x, double gyro_y, double gyro_z, double still_since)
  {
    if (samples_ == 0) {
      return false;
    }
    if (still_since <= 0.0 || t - still_since < settle_s_) {
      forget();
      spell_ = still_since;
      return false;
    }
    if (still_since != spell_) {
      forget();
      spell_ = still_since;
    }
    sum_x_ += gyro_x;
    sum_y_ += gyro_y;
    sum_z_ += gyro_z;
    ++count_;
    if (count_ < samples_) {
      return false;
    }
    const double count = static_cast<double>(count_);
    bias_ = GyroBias{sum_x_ / count, sum_y_ / count, sum_z_ / count};
    ++blocks_;
    at_s_ = t;
    ready_ = true;
    forget();
    return true;
  }

private:
  /// Throw away the block being accumulated; the bias already taken is untouched.
  void forget()
  {
    sum_x_ = 0.0;
    sum_y_ = 0.0;
    sum_z_ = 0.0;
    count_ = 0;
  }

  double settle_s_;
  long samples_;
  GyroBias bias_{};
  bool ready_;
  long blocks_ = 0;
  double at_s_ = 0.0;
  double spell_ = 0.0;
  double sum_x_ = 0.0;
  double sum_y_ = 0.0;
  double sum_z_ = 0.0;
  long count_ = 0;
};

}  // namespace pepin

#endif  // PEPIN_BASE_CPP__GYRO_BIAS_HPP_
