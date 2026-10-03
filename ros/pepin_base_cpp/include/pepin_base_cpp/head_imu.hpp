// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The head IMU's samples as head_server hands them over (TCP 3340), pure: no ROS, no JSON, no
// clock of its own. The twin of pepin.head_imu; head_line.hpp parses the JSON into these.
//
// THE WIRE (vio.md section 2; pepin.head_server, board/README.md "The head"): a client sends
// {"cmd":"subscribe","imu":true} and gets first, and again at every change of the chip's setup,
//   {"type":"imu_config","cfg":1,"rate_hz":200,"dlpf":3,"gyro_fs_dps":500,"accel_fs_g":4,
//    "filter_delay_s":0.0048}
// then one line per serial frame (every ~20 ms),
//   {"type":"imu","cfg":1,"samples":[[t_mono_s, esp_us, gx, gy, gz, ax, ay, az], ...]}
// where `t_mono_s` is the sample's data-ready edge on the board's CLOCK_MONOTONIC (head_server's
// lower-envelope clock map; the filter delay NOT taken off), `esp_us` the ESP32's micros
// unwrapped, gyro rad/s and accel m/s^2 in the chip's axes, biases not removed. The bridge dates
// a sample at t_mono_s less its config's filter_delay_s (and head_imu_filter_delay_s on top, 0 by
// default), carried onto the ROS clock exactly as the neck's encoder stamp. A batch whose config
// it has not been told is refused and counted: a default is a refusal.

#ifndef PEPIN_BASE_CPP__HEAD_IMU_HPP_
#define PEPIN_BASE_CPP__HEAD_IMU_HPP_

#include <cmath>
#include <cstddef>
#include <optional>

namespace pepin
{

/// One `imu_config` line: what the samples of config `cfg` were taken at.
struct HeadConfig
{
  int cfg = -1;
  double rate_hz = 0.0;         ///< the chip's output rate
  double filter_delay_s = 0.0;  ///< the gyro's group delay at its DLPF: when before t it happened
};

/// One head IMU sample in SI and the chip's axes, with both clocks.
struct HeadSample
{
  double t_mono_s = 0.0;  ///< board CLOCK_MONOTONIC seconds at the data-ready edge
  double esp_us = 0.0;    ///< the ESP32's micros, unwrapped
  double gyro[3] = {0.0, 0.0, 0.0};   ///< rad/s
  double accel[3] = {0.0, 0.0, 0.0};  ///< m/s^2
  int cfg = 0;
};

/// Whether a config can date and count samples: a positive finite rate, a finite delay >= 0.
inline bool config_valid(const HeadConfig & config)
{
  return config.rate_hz > 0.0 && std::isfinite(config.rate_hz) &&
         std::isfinite(config.filter_delay_s) && config.filter_delay_s >= 0.0;
}

/// One wire row [t_mono_s, esp_us, gx, gy, gz, ax, ay, az] as a sample; nothing when a value is
/// not finite (a malformed row is refused, never guessed at).
inline std::optional<HeadSample> sample_from_row(const double row[8], int cfg)
{
  for (std::size_t i = 0; i < 8; ++i) {
    if (!std::isfinite(row[i])) {
      return std::nullopt;
    }
  }
  HeadSample sample;
  sample.t_mono_s = row[0];
  sample.esp_us = row[1];
  for (std::size_t axis = 0; axis < 3; ++axis) {
    sample.gyro[axis] = row[2 + axis];
    sample.accel[axis] = row[5 + axis];
  }
  sample.cfg = cfg;
  return sample;
}

/// How a sample is dated on arrival: its age on the board's monotonic clock, or nothing when
/// that age is not one a sample on this machine can have (in the future, or older than
/// `max_age_s`) and the sample must be dated on arrival instead (counted by the caller).
inline std::optional<double> head_sample_age(double now_mono_s, double t_mono_s, double max_age_s)
{
  const double age = now_mono_s - t_mono_s;
  if (!std::isfinite(age) || age < 0.0 || age > max_age_s) {
    return std::nullopt;
  }
  return age;
}

/// The samples received per second and the gaps longer than 1.5 sample periods, by the samples'
/// own clock: the minute line's "200.1 Hz received, 0 gaps > 1.5 periods".
class HeadRate
{
public:
  /// One sample at `t_mono_s` from a chip at `rate_hz`; a sample not newer than the last is
  /// counted apart (out of order) and leaves the span alone.
  void add(double t_mono_s, double rate_hz)
  {
    ++samples_;
    if (has_last_) {
      const double gap = t_mono_s - last_;
      if (gap <= 0.0) {
        ++out_of_order_;
        return;
      }
      if (rate_hz > 0.0 && gap > 1.5 / rate_hz) {
        ++gaps_;
        if (gap > longest_gap_s_) {
          longest_gap_s_ = gap;
        }
      }
    } else {
      first_ = t_mono_s;
    }
    last_ = t_mono_s;
    has_last_ = true;
  }

  /// Samples per second over the span seen so far (0 with fewer than two).
  double rate_hz() const
  {
    const double span = last_ - first_;
    return (samples_ > 1 && span > 0.0) ? static_cast<double>(samples_ - 1) / span : 0.0;
  }

  long samples() const {return samples_;}
  long gaps() const {return gaps_;}
  long out_of_order() const {return out_of_order_;}
  double longest_gap_s() const {return longest_gap_s_;}

  /// Start a new window (the minute line takes one and resets).
  void reset() {*this = HeadRate();}

private:
  long samples_ = 0;
  long gaps_ = 0;
  long out_of_order_ = 0;
  double longest_gap_s_ = 0.0;
  double first_ = 0.0;
  double last_ = 0.0;
  bool has_last_ = false;
};

/// The publish cap: every n-th sample, n = ceil(chip rate / cap), so a 1 kHz chip capped at
/// 200 Hz keeps every fifth sample and a 200 Hz chip keeps them all; a cap of 0 publishes every
/// sample. Counted per sample, not timed: the chip's own rate decides, deterministically.
class HeadDecimator
{
public:
  explicit HeadDecimator(double cap_hz = 0.0)
  : cap_hz_(cap_hz) {}

  /// The keep-one-in-n of a chip at `rate_hz` under this cap.
  long every(double rate_hz) const
  {
    if (cap_hz_ <= 0.0 || rate_hz <= cap_hz_) {
      return 1;
    }
    return static_cast<long>(std::ceil(rate_hz / cap_hz_ - 1e-9));
  }

  /// Whether the next sample of a chip at `rate_hz` goes out (a rate change restarts the count).
  bool due(double rate_hz)
  {
    const long n = every(rate_hz);
    if (n != n_) {
      n_ = n;
      count_ = 0;
    }
    const bool keep = count_ % n_ == 0;
    ++count_;
    return keep;
  }

private:
  double cap_hz_;
  long n_ = 1;
  long count_ = 0;
};

}  // namespace pepin

#endif  // PEPIN_BASE_CPP__HEAD_IMU_HPP_
