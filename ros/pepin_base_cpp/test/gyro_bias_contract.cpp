// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The contract of tests/unit/test_gyro.py::test_gyro_bias_contract_the_cpp_bridge_mirrors,
// replayed row for row against gyro_bias.hpp. The package has no ament test target and the board
// image is not built on a laptop, so this is a stand-alone main() with no ROS and no gtest:
//
//     c++ -std=c++17 -Wall -Wextra -Wpedantic -O2 -I ros/pepin_base_cpp/include \
//         ros/pepin_base_cpp/test/gyro_bias_contract.cpp -o /tmp/gyro_bias_contract && \
//         /tmp/gyro_bias_contract
//
// It prints one line per row and exits non-zero on the first disagreement. Python is the
// reference: a row changes there first, then here.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "pepin_base_cpp/gyro_bias.hpp"

namespace
{

constexpr double kBlockS = 2.0;   // imu_bias_s
constexpr double kRateHz = 2.0;   // a table small enough to write by hand: four samples a block
constexpr double kMaxGapS = 1.0;  // TwistFromPose's max_gap_s, which the witness borrows
constexpr double kTolerance = 1e-12;

struct Row
{
  double t;
  double reading;      // the same value on all three axes
  double still_since;  // the wheels' word; 0 = not known to be standing still
  bool finished;       // a block closed on this sample
  double bias;         // what the bias reads afterwards, on every axis
  long blocks;
};

struct WitnessRow
{
  double stamp_s;      // the board's clock on the state line
  double at;           // the local monotonic clock when it was judged
  bool moving;         // the board's own word: a non-zero twist is being applied
  bool twist_is_zero;  // the wheels' MEASURED twist, differenced off two poses
  double since;        // the rest spell the witness then publishes
};

int failures = 0;

void check(bool ok, const char * what, double t)
{
  if (!ok) {
    std::printf("FAIL  %s at t=%.2f\n", what, t);
    ++failures;
  }
}

}  // namespace

int main()
{
  const Row rows[] = {
    // t      reading  still_since  finished  bias   blocks
    {10.00, 0.90, 0.0, false, 0.00, 0},   // nobody watching: not rest, whatever the reading
    {10.50, 0.90, 10.5, false, 0.00, 0},  // the spell starts here; the chassis is settling
    {12.00, 0.90, 10.5, false, 0.00, 0},  // 1.5 s < 2.0 s: still settling
    {12.50, 0.10, 10.5, false, 0.00, 0},  // 2.0 s: the first sample of the block
    {13.00, 0.20, 10.5, false, 0.00, 0},
    {13.50, 0.30, 10.5, false, 0.00, 0},
    {14.00, 0.40, 10.5, true, 0.25, 1},   // four samples: the mean replaces the bias
    {14.50, 1.00, 10.5, false, 0.25, 1},  // the next block tiles on, no settle window again
    {15.00, 1.00, 10.5, false, 0.25, 1},
    {15.50, 1.00, 10.5, false, 0.25, 1},
    {16.00, 1.00, 10.5, true, 1.00, 2},   // replaced outright, not blended towards 0.25
    {16.50, 0.10, 10.5, false, 1.00, 2},  // a third block opens
    {17.00, 0.10, 10.5, false, 1.00, 2},
    {17.50, 9.00, 17.5, false, 1.00, 2},  // the cart moved: the two samples are thrown away
    {19.00, 0.10, 17.5, false, 1.00, 2},  // 1.5 s since the move: settling, not counting
    {19.50, 0.10, 17.5, false, 1.00, 2},  // 2.0 s: counting again, from one
    {20.00, 0.10, 17.5, false, 1.00, 2},
    {20.50, 0.10, 17.5, false, 1.00, 2},
    {21.00, 0.10, 17.5, true, 0.10, 3},   // and the block after the move closes here
  };

  pepin::GyroBiasTracker tracker(kBlockS, kRateHz);
  check(!tracker.ready(), "a fresh tracker must not be ready", 0.0);
  check(std::isinf(tracker.age_s(100.0)), "age is infinite before the first block", 0.0);
  check(tracker.block_samples() == 4, "2.0 s at 2 Hz is four samples a block", 0.0);

  for (const Row & row : rows) {
    const bool finished =
      tracker.update(row.t, row.reading, row.reading, row.reading, row.still_since);
    const pepin::GyroBias bias = tracker.bias();
    check(finished == row.finished, "block boundary", row.t);
    check(std::fabs(bias.x - row.bias) < kTolerance, "bias x", row.t);
    check(bias.y == bias.x && bias.z == bias.x, "the three axes are averaged alike", row.t);
    check(tracker.blocks() == row.blocks, "block count", row.t);
    std::printf(
      "t=%5.2f reading %+.2f still_since %5.1f -> %s bias %+.3f blocks %ld\n",
      row.t, row.reading, row.still_since, finished ? "BLOCK" : "     ", bias.x,
      tracker.blocks());
  }
  check(std::fabs(tracker.age_s(26.0) - 5.0) < kTolerance, "age since the last block", 26.0);
  check(tracker.ready(), "ready after three blocks", 26.0);

  // The wheels' witness: tests/unit/test_gyro.py's three veto tests and the silence one, in a
  // table. The board's clock runs at 20 Hz state lines; the local clock is what the gyro uses.
  const WitnessRow witness_rows[] = {
    // stamp_s   at    moving twist_is_zero  since
    {100.00, 10.00, false, true, 10.00},   // the first line can only start a spell
    {100.05, 10.05, false, true, 10.00},   // rest carries on
    {100.10, 10.10, false, false, 10.10},  // the wheels turned: motion, here
    {100.15, 10.15, false, true, 10.10},   // still again, spell dated from the motion
    {100.20, 10.20, true, true, 10.20},    // a live command on blocked wheels is not rest
    {100.25, 10.25, false, true, 10.20},   // the command gone: the spell dates from it, not now
    {103.25, 13.25, false, true, 13.25},   // 3 s of silence: the zero twist measured nothing
    {103.30, 13.30, false, true, 13.25},   // measuring again
    {99.00, 14.00, false, true, 14.00},    // the board's clock restarted: not a continuation
  };
  pepin::RestWitness witness(kMaxGapS);
  for (const WitnessRow & row : witness_rows) {
    const double since = witness.judge(row.stamp_s, row.at, row.moving, row.twist_is_zero);
    check(std::fabs(since - row.since) < kTolerance, "witnessed rest since", row.at);
    check(witness.at() == row.at, "the witness's own freshness", row.at);
    std::printf(
      "line stamp %6.2f at %5.2f moving %d twist_zero %d -> rest since %5.2f\n",
      row.stamp_s, row.at, row.moving ? 1 : 0, row.twist_is_zero ? 1 : 0, since);
  }
  witness.forget();
  check(witness.since() == 0.0, "a forgotten spell is not rest", 0.0);
  check(
    witness.judge(99.05, 14.05, false, true) == 14.05, "and the next line starts over", 14.05);

  // A witness nobody refreshed stops counting as one: the link is down, or /odom is muted.
  check(pepin::rest_witnessed(10.0, 10.5, 11.0, kMaxGapS) == 10.0, "judged 0.5 s ago", 11.0);
  check(pepin::rest_witnessed(10.0, 10.5, 11.6, kMaxGapS) == 0.0, "judged 1.1 s ago", 11.6);
  check(pepin::rest_witnessed(0.0, 10.5, 10.6, kMaxGapS) == 0.0, "never at rest", 10.6);

  // imu_bias_s 0 has always meant "do not calibrate": ready at once, nothing subtracted.
  pepin::GyroBiasTracker uncalibrated(0.0, 50.0);
  check(uncalibrated.ready(), "no calibration asked for is ready at once", 0.0);
  check(!uncalibrated.update(1.0, 7.0, 7.0, 7.0, 0.5), "and never takes a block", 1.0);
  check(uncalibrated.bias().x == 0.0, "and subtracts nothing", 1.0);
  check(uncalibrated.block_samples() == 0, "and has no block size", 1.0);

  std::printf(failures == 0 ? "\nthe contract holds\n" : "\n%d checks failed\n", failures);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
