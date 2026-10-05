// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// When the base bridge asks the MPU6050 whether it is there, and when it gives a running chip up.
// No ROS, no I/O, no clock of its own: the IMU thread calls it with the board's monotonic seconds
// and the outcome of each open and each read, and asks it what to do next.
//
// Why it exists: the bridge probed WHO_AM_I once at start, and a probe that failed left it on wheel
// odometry until the next restart. On 2026-10-05 that probe ran on a locked i2c-2 (a slave held SDA
// low from 08:38:16 until 14:35:52, every transfer timing out after 2 s), and after the bus came
// back the IMU stayed off; on 2026-10-04 the same at the 14:57 restart.
//
// THE SCHEDULE. A round of `start_attempts` probes, `start_interval_s` apart, at start and again
// whenever a live chip is lost; after a round that found nothing, one probe every `reprobe_s`
// until one answers. A live chip whose reads fail `lost_after` times in a row is LOST: the bridge
// closes the bus (so nothing holds /dev/i2c-2 open while it is recovered: an unbind waits for every
// open file) and a fresh round begins. A glitch costs a second; a locked bus one 2 s probe a
// minute, on the IMU thread alone.

#ifndef PEPIN_BASE_CPP__IMU_PROBE_HPP_
#define PEPIN_BASE_CPP__IMU_PROBE_HPP_

#include <algorithm>
#include <cstdio>
#include <string>

namespace pepin
{

/// Where the chip stands: being asked in a round, publishing, or given up on until the next probe.
enum class ImuPresence { kProbing, kLive, kAbsent };

/// The schedule's numbers; the defaults are the bridge's.
struct ImuProbeSettings
{
  int start_attempts = 5;        ///< probes in a round, at start and after a loss
  double start_interval_s = 1.0;  ///< between the probes of a round
  double reprobe_s = 60.0;        ///< between probes once a round has found nothing
  int lost_after = 5;             ///< consecutive failed reads that lose a live chip
};

/// What the report line prints: a copy the IMU thread hands over at every change.
struct ImuProbeState
{
  ImuPresence presence = ImuPresence::kProbing;
  double since = 0.0;          ///< when the current presence began
  double next_probe_at = 0.0;  ///< meaningful while not live
  int attempt = 0;             ///< probes made in the current round
  int start_attempts = 0;
  double reprobe_s = 0.0;
  long probes = 0;             ///< every open tried
  long failed_probes = 0;
  long lives = 0;              ///< times the chip went live: more than one is a recovery
  long losses = 0;
};

/// The schedule above; one owner thread.
class ImuProbe
{
public:
  /// A round starts at `now`: the first probe is due at once.
  explicit ImuProbe(ImuProbeSettings settings = {}, double now = 0.0)
  : settings_(settings)
  {
    settings_.start_attempts = std::max(1, settings_.start_attempts);
    settings_.lost_after = std::max(1, settings_.lost_after);
    state_.since = now;
    state_.next_probe_at = now;
    state_.start_attempts = settings_.start_attempts;
    state_.reprobe_s = settings_.reprobe_s;
  }

  /// Probing, live or absent.
  ImuPresence presence() const {return state_.presence;}

  /// True when the chip is not live and its next probe is due at `now`.
  bool due(double now) const
  {
    return state_.presence != ImuPresence::kLive && now >= state_.next_probe_at;
  }

  /// When the next probe is due (meaningful while not live).
  double next_probe_at() const {return state_.next_probe_at;}

  /// One open of the chip (WHO_AM_I and the configuration), its outcome at `now`.
  /// Returns true when the chip goes live.
  bool probed(bool ok, double now)
  {
    ++state_.probes;
    if (ok) {
      state_.presence = ImuPresence::kLive;
      state_.since = now;
      state_.attempt = 0;
      ++state_.lives;
      failures_ = 0;
      return true;
    }
    ++state_.failed_probes;
    ++state_.attempt;
    if (state_.presence == ImuPresence::kProbing && state_.attempt >= settings_.start_attempts) {
      state_.presence = ImuPresence::kAbsent;
      state_.since = now;
    }
    state_.next_probe_at = now +
      (state_.presence == ImuPresence::kProbing ? settings_.start_interval_s : settings_.reprobe_s);
    return false;
  }

  /// One read of a live chip, its outcome at `now`. Returns true at the moment the chip is lost
  /// (`lost_after` failures in a row): the caller closes the bus, and a fresh round is due at once.
  bool read(bool ok, double now)
  {
    if (state_.presence != ImuPresence::kLive) {
      return false;
    }
    if (ok) {
      failures_ = 0;
      return false;
    }
    if (++failures_ < settings_.lost_after) {
      return false;
    }
    failures_ = 0;
    ++state_.losses;
    state_.presence = ImuPresence::kProbing;
    state_.since = now;
    state_.attempt = 0;
    state_.next_probe_at = now;
    return true;
  }

  /// Consecutive failed reads of the live chip so far.
  int failures() const {return failures_;}

  /// A copy for the report line.
  const ImuProbeState & state() const {return state_;}

private:
  ImuProbeSettings settings_;
  ImuProbeState state_;
  int failures_ = 0;
};

/// The IMU as the report line prints it, from a state copy, the last failure and `now`:
///
/// ``imu: live 3600 s`` (``, live 2 times, lost 1, 7 probes failed`` once anything went wrong);
/// ``imu: probing, 2 of 5 failed, next in 0.6 s (last: ...)``;
/// ``imu: ABSENT 61 s, next probe in 59 s, 6 of 6 probes failed (last: ...); wheel odometry only``.
inline std::string describe_imu(const ImuProbeState & s, const std::string & last_error, double now)
{
  char line[384];
  const std::string last = last_error.empty() ? "" : " (last: " + last_error + ")";
  switch (s.presence) {
    case ImuPresence::kLive:
      if (s.lives <= 1 && s.losses == 0 && s.failed_probes == 0) {
        std::snprintf(line, sizeof(line), "imu: live %.0f s", now - s.since);
      } else {
        std::snprintf(
          line, sizeof(line), "imu: live %.0f s, live %ld time%s, lost %ld, %ld probe%s failed",
          now - s.since, s.lives, s.lives == 1 ? "" : "s", s.losses, s.failed_probes,
          s.failed_probes == 1 ? "" : "s");
      }
      return line;
    case ImuPresence::kProbing:
      std::snprintf(
        line, sizeof(line), "imu: probing%s, %d of %d failed, next in %.1f s%s",
        s.losses > 0 ? " after a loss" : "", s.attempt, s.start_attempts,
        std::max(0.0, s.next_probe_at - now), last.c_str());
      return line;
    case ImuPresence::kAbsent:
    default:
      std::snprintf(
        line, sizeof(line),
        "imu: ABSENT %.0f s, next probe in %.0f s (every %.0f s), %ld of %ld probes failed%s; "
        "wheel odometry only",
        now - s.since, std::max(0.0, s.next_probe_at - now), s.reprobe_s, s.failed_probes,
        s.probes, last.c_str());
      return line;
  }
}

}  // namespace pepin

#endif  // PEPIN_BASE_CPP__IMU_PROBE_HPP_
