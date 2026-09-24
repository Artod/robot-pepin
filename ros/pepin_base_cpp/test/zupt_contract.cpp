// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The contract of tests/unit/test_zupt.py::test_zupt_contract_the_cpp_bridge_mirrors, replayed row
// for row against zupt.hpp, verdict words included. Like gyro_bias_contract.cpp it is a
// stand-alone main() with no ROS and no gtest, because the package has no ament test target. One
// command (no line continuation here: GCC's -Wcomment reads a backslash as one):
//
//     c++ -std=c++17 -Wall -Wextra -Wpedantic -O2 -I ros/pepin_base_cpp/include
//         ros/pepin_base_cpp/test/zupt_contract.cpp -o /tmp/zupt_contract && /tmp/zupt_contract
//
// It prints one line per row and exits non-zero on any disagreement. Python is the reference: a
// row changes there first, then here.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "pepin_base_cpp/zupt.hpp"

namespace
{

constexpr double kSettleS = 2.0;       // imu_bias_s
constexpr double kCommandHoldS = 0.5;  // cmd_timeout_s
constexpr double kMaxGapS = 1.0;       // kStateGapMaxS

struct Row
{
  double now;
  double still_since;
  double command_at;
  double gyro_at;
  double gyro_turn_at;
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
    // now    still  command gyro_at turn_at verdict
    {100.0, 40.0, 0.0, 99.98, 0.0, "at rest"},
    {100.0, 0.0, 0.0, 99.98, 0.0, "the wheels do not witness rest"},
    {100.0, 98.5, 0.0, 99.98, 0.0, "settling after the last motion"},
    {100.0, 98.0, 0.0, 99.98, 0.0, "at rest"},  // exactly the settle window: rest
    {100.0, 40.0, 99.9, 99.98, 0.0, "a command is live"},
    {100.0, 40.0, 99.5, 99.98, 0.0, "at rest"},  // exactly the hold: over
    {100.0, 40.0, 0.0, 0.0, 0.0, "no gyro reading to judge by"},
    {100.0, 40.0, 0.0, 98.9, 0.0, "no gyro reading to judge by"},
    {100.0, 40.0, 0.0, 99.0, 0.0, "at rest"},  // exactly max_gap_s old: still fresh
    {100.0, 40.0, 0.0, 99.98, 99.98, "the gyro reports a turn"},  // the newest sample
    {100.0, 40.0, 0.0, 99.98, 98.5, "the gyro reports a turn"},  // inside the settle window
    {100.0, 40.0, 0.0, 99.98, 98.0, "at rest"},  // served
    {100.0, 0.0, 99.9, 0.0, 99.98, "a command is live"},  // every veto: the command first
    {100.0, 99.0, 0.0, 0.0, 99.98, "settling after the last motion"},  // wheels before gyro
    {100.0, 40.0, 0.0, 98.0, 97.9, "no gyro reading to judge by"},  // stale before turning
  };

  const pepin::ZuptGate gate(kSettleS, kCommandHoldS, kMaxGapS);
  for (const Row & row : rows) {
    pepin::RestEvidence evidence;
    evidence.still_since = row.still_since;
    evidence.command_at = row.command_at;
    evidence.gyro_at = row.gyro_at;
    evidence.gyro_turn_at = row.gyro_turn_at;
    const char * verdict = pepin::describe(gate.judge(row.now, evidence));
    check(std::strcmp(verdict, row.verdict) == 0, row.verdict);
    std::printf(
      "still %5.2f command %5.2f gyro %5.2f turn %5.2f -> %s\n", row.still_since, row.command_at,
      row.gyro_at, row.gyro_turn_at, verdict);
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
  check(pepin::kRestZuptVariance == 1e-6, "pepin.zupt.REST_ZUPT_VARIANCE");
  check(pepin::kGyroQuietRadS == 0.005, "pepin.zupt.GYRO_QUIET_RAD_S");
  check(pepin::kZuptHz == 10.0, "pepin.zupt.ZUPT_HZ");

  std::printf(failures == 0 ? "\nthe contract holds\n" : "\n%d checks failed\n", failures);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
