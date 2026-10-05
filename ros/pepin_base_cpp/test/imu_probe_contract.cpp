// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The contract of the IMU's probe schedule, replayed event by event against imu_probe.hpp: a round
// at start, the minute probes after it, a live chip lost and found again, the report line's words.
// Like the other contracts it is a stand-alone main() with no ROS and no gtest. One command:
//
//     c++ -std=c++17 -Wall -Wextra -Wpedantic -O2 -I ros/pepin_base_cpp/include
//         ros/pepin_base_cpp/test/imu_probe_contract.cpp -o /tmp/imu_probe_contract && /tmp/imu_probe_contract
//
// It prints one line per event and exits non-zero on any disagreement;
// tests/unit/test_base_cpp_contracts.py compiles and runs it.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#include "pepin_base_cpp/imu_probe.hpp"

namespace
{

using pepin::ImuPresence;

enum class Kind { kProbe, kRead };

struct Row
{
  Kind kind;
  bool ok;
  double now;
  bool returned;          // probed(): went live; read(): lost
  ImuPresence presence;   // after the event
  double next_probe_at;   // after the event; NaN: not checked (live)
};

int failures = 0;

void check(bool ok, const std::string & what)
{
  if (!ok) {
    std::printf("FAIL  %s\n", what.c_str());
    ++failures;
  }
}

const char * name(ImuPresence p)
{
  switch (p) {
    case ImuPresence::kLive: return "live";
    case ImuPresence::kProbing: return "probing";
    default: return "absent";
  }
}

void expect_text(const std::string & got, const char * want)
{
  check(got == want, "describe: got '" + got + "', want '" + want + "'");
  std::printf("%s\n", got.c_str());
}

}  // namespace

int main()
{
  constexpr double kNo = std::numeric_limits<double>::quiet_NaN();
  pepin::ImuProbeSettings settings;  // 5 x 1 s, then every 60 s, lost after 5 failed reads
  pepin::ImuProbe probe(settings, 100.0);
  check(probe.due(100.0), "the first probe is due at once");
  check(probe.presence() == ImuPresence::kProbing, "a fresh schedule is probing");

  const Row rows[] = {
    // the start round: four failures 1 s apart (each took the bus's 2 s timeout), then the chip
    {Kind::kProbe, false, 102.0, false, ImuPresence::kProbing, 103.0},
    {Kind::kProbe, false, 105.0, false, ImuPresence::kProbing, 106.0},
    {Kind::kProbe, false, 108.0, false, ImuPresence::kProbing, 109.0},
    {Kind::kProbe, false, 111.0, false, ImuPresence::kProbing, 112.0},
    {Kind::kProbe, true, 112.1, true, ImuPresence::kLive, kNo},
    // reads: four failures in a row are not a loss, a success resets the count
    {Kind::kRead, false, 120.00, false, ImuPresence::kLive, kNo},
    {Kind::kRead, false, 120.01, false, ImuPresence::kLive, kNo},
    {Kind::kRead, false, 120.02, false, ImuPresence::kLive, kNo},
    {Kind::kRead, false, 120.03, false, ImuPresence::kLive, kNo},
    {Kind::kRead, true, 120.04, false, ImuPresence::kLive, kNo},
    {Kind::kRead, false, 130.0, false, ImuPresence::kLive, kNo},
    {Kind::kRead, false, 132.0, false, ImuPresence::kLive, kNo},
    {Kind::kRead, false, 134.0, false, ImuPresence::kLive, kNo},
    {Kind::kRead, false, 136.0, false, ImuPresence::kLive, kNo},
    // the fifth: lost, a fresh round due at once
    {Kind::kRead, false, 138.0, true, ImuPresence::kProbing, 138.0},
    // a read reported after the loss changes nothing
    {Kind::kRead, false, 138.5, false, ImuPresence::kProbing, 138.0},
    // the round finds nothing: absent after the fifth, then a probe a minute
    {Kind::kProbe, false, 140.0, false, ImuPresence::kProbing, 141.0},
    {Kind::kProbe, false, 143.0, false, ImuPresence::kProbing, 144.0},
    {Kind::kProbe, false, 146.0, false, ImuPresence::kProbing, 147.0},
    {Kind::kProbe, false, 149.0, false, ImuPresence::kProbing, 150.0},
    {Kind::kProbe, false, 152.0, false, ImuPresence::kAbsent, 212.0},
    {Kind::kProbe, false, 214.0, false, ImuPresence::kAbsent, 274.0},
    // the bus was recovered: the next minute probe finds the chip, no restart
    {Kind::kProbe, true, 274.2, true, ImuPresence::kLive, kNo},
  };

  int index = 0;
  for (const Row & row : rows) {
    ++index;
    const bool returned = row.kind == Kind::kProbe ? probe.probed(row.ok, row.now) :
      probe.read(row.ok, row.now);
    char what[160];
    std::snprintf(
      what, sizeof(what), "row %d (%s %s at %.2f)", index,
      row.kind == Kind::kProbe ? "probe" : "read", row.ok ? "ok" : "failed", row.now);
    check(returned == row.returned, std::string(what) + ": returned");
    check(probe.presence() == row.presence, std::string(what) + ": presence");
    if (!std::isnan(row.next_probe_at)) {
      check(
        std::fabs(probe.next_probe_at() - row.next_probe_at) < 1e-9,
        std::string(what) + ": next probe");
      check(!probe.due(row.next_probe_at - 0.01), std::string(what) + ": due early");
      check(probe.due(row.next_probe_at), std::string(what) + ": not due on time");
    } else {
      check(!probe.due(1e9), std::string(what) + ": a live chip is never due");
    }
    std::printf(
      "%-30s %d %-7s next %.2f\n", what, static_cast<int>(returned), name(probe.presence()),
      probe.next_probe_at());
  }

  const pepin::ImuProbeState & s = probe.state();
  check(s.probes == 12 && s.failed_probes == 10, "probes counted");
  check(s.lives == 2 && s.losses == 1, "lives and losses counted");

  // The report line's words, one per presence.
  pepin::ImuProbeState live;
  live.presence = ImuPresence::kLive;
  live.since = 10.0;
  live.lives = 1;
  expect_text(pepin::describe_imu(live, "", 3610.0), "imu: live 3600 s");
  expect_text(
    pepin::describe_imu(s, "", 334.2),
    "imu: live 60 s, live 2 times, lost 1, 10 probes failed");
  pepin::ImuProbeState probing;
  probing.attempt = 2;
  probing.start_attempts = 5;
  probing.next_probe_at = 10.6;
  expect_text(
    pepin::describe_imu(probing, "seek to 0x75 failed: Connection timed out", 10.0),
    "imu: probing, 2 of 5 failed, next in 0.6 s (last: seek to 0x75 failed: Connection timed "
    "out)");
  pepin::ImuProbeState absent;
  absent.presence = ImuPresence::kAbsent;
  absent.since = 100.0;
  absent.next_probe_at = 220.0;
  absent.reprobe_s = 60.0;
  absent.probes = 6;
  absent.failed_probes = 6;
  expect_text(
    pepin::describe_imu(absent, "WHO_AM_I unreadable", 161.0),
    "imu: ABSENT 61 s, next probe in 59 s (every 60 s), 6 of 6 probes failed (last: WHO_AM_I "
    "unreadable); wheel odometry only");

  // Nonsense settings are clamped: a round of at least one probe, a loss after at least one read.
  pepin::ImuProbe single({0, 1.0, 60.0, 0}, 0.0);
  single.probed(false, 1.0);
  check(single.presence() == ImuPresence::kAbsent, "a zero-probe round is one probe");
  single.probed(true, 61.0);
  check(single.read(false, 62.0), "lost_after 0 loses at the first failure");

  if (failures == 0) {
    std::printf("the contract holds\n");
    return 0;
  }
  std::printf("%d disagreement(s)\n", failures);
  return 1;
}
