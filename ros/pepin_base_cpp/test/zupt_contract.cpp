// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The contract of the zero-velocity update, replayed row for row against zupt.hpp, verdict words
// included. Like gyro_bias_contract.cpp it is a
// stand-alone main() with no ROS and no gtest, because the package has no ament test target. One
// command (no line continuation here: GCC's -Wcomment reads a backslash as one):
//
//     c++ -std=c++17 -Wall -Wextra -Wpedantic -O2 -I ros/pepin_base_cpp/include
//         ros/pepin_base_cpp/test/zupt_contract.cpp -o /tmp/zupt_contract && /tmp/zupt_contract
//
// It prints one line per row and exits non-zero on any disagreement;
// tests/unit/test_base_cpp_contracts.py compiles and runs it.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "pepin_base_cpp/zupt.hpp"

namespace
{

constexpr double kCommandHoldS = 0.5;  // cmd_timeout_s, zupt_cmd_hold_s by default
constexpr double kMaxGapS = 1.0;       // kStateGapMaxS

struct Row
{
  double now;
  double still_since;
  double command_at;
  double gyro_at;
  double gyro_turn_at;
  double settle_s;  // the live zupt_settle_s
  double hold_s;    // the live zupt_cmd_hold_s
  const char * verdict;
};

int failures = 0;

void check(bool ok, const char * what)
{
  if (!ok) {
    std::printf("FAIL  %s\n", what);
    ++failures;
  }
}

}  // namespace

int main()
{
  const Row rows[] = {
    // now    still  command gyro_at turn_at settle hold verdict
    {100.0, 40.0, 0.0, 99.98, 0.0, 2.0, 0.5, "at rest"},
    {100.0, 0.0, 0.0, 99.98, 0.0, 2.0, 0.5, "the wheels do not witness rest"},
    {100.0, 98.5, 0.0, 99.98, 0.0, 2.0, 0.5, "settling after the last motion"},
    {100.0, 98.0, 0.0, 99.98, 0.0, 2.0, 0.5, "at rest"},  // exactly the settle window: rest
    {100.0, 40.0, 99.9, 99.98, 0.0, 2.0, 0.5, "a command is live"},
    {100.0, 40.0, 99.5, 99.98, 0.0, 2.0, 0.5, "at rest"},  // exactly the hold: over
    {100.0, 40.0, 0.0, 0.0, 0.0, 2.0, 0.5, "no gyro reading to judge by"},
    {100.0, 40.0, 0.0, 98.9, 0.0, 2.0, 0.5, "no gyro reading to judge by"},
    {100.0, 40.0, 0.0, 99.0, 0.0, 2.0, 0.5, "at rest"},  // exactly max_gap_s old: still fresh
    // the gyro: its newest sample a turn, a turn inside the settle window, and one served
    {100.0, 40.0, 0.0, 99.98, 99.98, 2.0, 0.5, "the gyro reports a turn"},
    {100.0, 40.0, 0.0, 99.98, 98.5, 2.0, 0.5, "the gyro reports a turn"},
    {100.0, 40.0, 0.0, 99.98, 98.0, 2.0, 0.5, "at rest"},
    // several vetoes at once: the command first, the wheels before the gyro, stale before turning
    {100.0, 0.0, 99.9, 0.0, 99.98, 2.0, 0.5, "a command is live"},
    {100.0, 99.0, 0.0, 0.0, 99.98, 2.0, 0.5, "settling after the last motion"},
    {100.0, 40.0, 0.0, 98.0, 97.9, 2.0, 0.5, "no gyro reading to judge by"},
    // the live windows moved: a longer settle, a shorter one, a longer hold, no hold at all, and
    // the gyro's hold following the settle window
    {100.0, 97.0, 0.0, 99.98, 0.0, 5.0, 0.5, "settling after the last motion"},
    {100.0, 99.5, 0.0, 99.98, 0.0, 0.5, 0.5, "at rest"},
    {100.0, 40.0, 98.0, 99.98, 0.0, 2.0, 3.0, "a command is live"},
    {100.0, 40.0, 99.95, 99.98, 0.0, 2.0, 0.0, "at rest"},
    {100.0, 40.0, 0.0, 99.98, 97.0, 5.0, 0.5, "the gyro reports a turn"},
  };

  for (const Row & row : rows) {
    const pepin::ZuptGate gate(row.settle_s, row.hold_s, kMaxGapS);  // the bridge builds one a tick
    pepin::RestEvidence evidence;
    evidence.still_since = row.still_since;
    evidence.command_at = row.command_at;
    evidence.gyro_at = row.gyro_at;
    evidence.gyro_turn_at = row.gyro_turn_at;
    const char * verdict = pepin::describe(gate.judge(row.now, evidence));
    check(std::strcmp(verdict, row.verdict) == 0, row.verdict);
    std::printf(
      "still %5.2f command %5.2f gyro %5.2f turn %5.2f settle %.1f hold %.1f -> %s\n",
      row.still_since, row.command_at, row.gyro_at, row.gyro_turn_at, row.settle_s, row.hold_s,
      verdict);
  }

  // imu_bias_s 0 is no settle window; the newest sample turning must still veto.
  const pepin::ZuptGate no_settle(0.0, kCommandHoldS, kMaxGapS);
  pepin::RestEvidence turning{40.0, 0.0, 99.98, 99.98};
  check(
    no_settle.judge(100.0, turning) == pepin::ZuptVerdict::kGyroTurning,
    "the newest sample turning vetoes with no settle window");
  turning.gyro_turn_at = 99.96;
  check(
    no_settle.judge(100.0, turning) == pepin::ZuptVerdict::kAtRest,
    "and one sample of quiet after it is rest");

  // The quiet threshold, NaN included.
  check(!pepin::gyro_turning(0.0049, pepin::kGyroQuietRadS), "under the threshold is quiet");
  check(!pepin::gyro_turning(-0.0049, pepin::kGyroQuietRadS), "in both directions");
  check(pepin::gyro_turning(0.005, pepin::kGyroQuietRadS), "at the threshold is a turn");
  check(
    pepin::gyro_turning(std::numeric_limits<double>::quiet_NaN(), pepin::kGyroQuietRadS),
    "a NaN is a turn");

  // The covariance: vx, vy, vyaw at 1e-6, the rest of the diagonal 1e6, nothing off it.
  const auto matrix = pepin::rest_zupt_twist_covariance();
  const double diagonal[6] = {1e-6, 1e-6, 1e6, 1e6, 1e6, 1e-6};
  for (std::size_t row = 0; row < 6; ++row) {
    for (std::size_t col = 0; col < 6; ++col) {
      const double expected = row == col ? diagonal[row] : 0.0;
      check(matrix[row * 6 + col] == expected, "twist covariance entry");
    }
  }
  const auto live = pepin::rest_zupt_twist_covariance(1e-4, 4e-4);
  check(live[0] == 1e-4 && live[7] == 1e-4, "zupt_var_linear on vx and vy");
  check(live[35] == 4e-4 && live[14] == 1e6, "zupt_var_yaw on vyaw, the rest unclaimed");

  // The live settings' ranges: inclusive, NaN refused, an unknown name is not one of them.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const pepin::ZuptRange & range : pepin::kZuptRanges) {
    check(pepin::zupt_range(range.name) == &range, range.name);
    check(pepin::zupt_in_range(range, range.low), "the low end is inside");
    check(pepin::zupt_in_range(range, range.high), "the high end is inside");
    check(!pepin::zupt_in_range(range, nan), "a NaN is refused");
  }
  check(pepin::zupt_range("imu_bias_s") == nullptr, "not a zero-velocity setting");
  const pepin::ZuptRange & rate = *pepin::zupt_range("zupt_rate_hz");
  check(rate.low == 1.0 && rate.high == 100.0, "zupt_rate_hz: 1..100");
  check(pepin::zupt_in_range(rate, 50.0), "50 Hz is legal");
  check(!pepin::zupt_in_range(rate, 0.5) && !pepin::zupt_in_range(rate, 101.0), "0.5, 101 not");
  check(
    !pepin::zupt_in_range(*pepin::zupt_range("zupt_var_yaw"), 1e-10), "under the EKF's floor");
  const pepin::ZuptRange & settle = *pepin::zupt_range("zupt_settle_s");
  check(pepin::zupt_clamped(settle, 2.0) == 2.0, "a borrowed default inside stays");
  check(pepin::zupt_clamped(settle, -1.0) == 0.0, "imu_bias_s -1 becomes no window");
  check(pepin::zupt_clamped(settle, 600.0) == 60.0, "and 600 s the longest window");
  check(pepin::zupt_clamped(*pepin::zupt_range("zupt_cmd_hold_s"), nan) == 0.0, "NaN: low end");

  check(pepin::kRestZuptVariance == 1e-6, "kRestZuptVariance");
  check(pepin::kGyroQuietRadS == 0.005, "kGyroQuietRadS");
  check(pepin::kZuptHz == 10.0, "kZuptHz");

  std::printf(failures == 0 ? "\nthe contract holds\n" : "\n%d checks failed\n", failures);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
