// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The head IMU's pure half (head_imu.hpp), held to what the bridge relies on, and to its Python
// twin (pepin.head_imu) through --stdin. Stand-alone, no ROS, no JSON (head_line.hpp's parse is
// compiled with the bridge in the board image):
//
//     c++ -std=c++17 -Wall -Wextra -Wpedantic -O2 -I ros/pepin_base_cpp/include \
//         ros/pepin_base_cpp/test/head_imu_contract.cpp -o /tmp/head_imu_contract
//     /tmp/head_imu_contract            # self-checks: prints "the contract holds"
//     /tmp/head_imu_contract --stdin    # parity: "config cfg rate delay", then "row ..." lines

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "pepin_base_cpp/head_imu.hpp"

namespace
{

int failures = 0;

void check(bool ok, const char * what)
{
  if (!ok) {
    std::printf("FAILED: %s\n", what);
    ++failures;
  }
}

bool near(double a, double b, double tol = 1e-12) {return std::fabs(a - b) <= tol;}

int self_checks()
{
  const double row[8] = {100.0, 1e6, 0.01, -0.02, 0.03, 0.1, -0.2, 9.80665};
  const auto sample = pepin::sample_from_row(row, 1);
  check(sample.has_value(), "a finite row parses");
  check(near(sample->gyro[0], 0.01) && near(sample->gyro[2], 0.03), "gyro first, rad/s");
  check(near(sample->accel[1], -0.2) && near(sample->accel[2], 9.80665), "then accel, m/s^2");
  check(sample->t_mono_s == 100.0 && sample->esp_us == 1e6 && sample->cfg == 1, "both clocks");
  double broken[8] = {100.0, 1e6, 0.0, 0.0, NAN, 0.0, 0.0, 9.8};
  check(!pepin::sample_from_row(broken, 1).has_value(), "a NaN row is refused");
  check(pepin::config_valid({1, 200.0, 0.0048}), "a config at 200 Hz, 4.8 ms");
  check(!pepin::config_valid({1, 0.0, 0.0048}), "no rate, no config");
  check(!pepin::config_valid({1, 200.0, -0.001}), "a negative delay is no config");

  check(pepin::head_sample_age(10.0, 9.99, 0.5).has_value(), "a fresh sample has an age");
  check(near(*pepin::head_sample_age(10.0, 9.99, 0.5), 0.01, 1e-12), "and it is 10 ms");
  check(!pepin::head_sample_age(10.0, 10.01, 0.5).has_value(), "the future is dated on arrival");
  check(!pepin::head_sample_age(10.0, 9.0, 0.5).has_value(), "so is a stale one");

  pepin::HeadRate rate;
  for (int i = 0; i < 200; ++i) {
    rate.add(i * 0.005, 200.0);
  }
  rate.add(200 * 0.005 + 0.010, 200.0);  // one gap of 15 ms
  rate.add(200 * 0.005 + 0.005, 200.0);  // and one out of order
  check(rate.samples() == 202, "every sample counted");
  check(rate.gaps() == 1 && near(rate.longest_gap_s(), 0.015, 1e-9), "a 15 ms gap at 200 Hz");
  check(rate.out_of_order() == 1, "an out-of-order sample counted apart");

  pepin::HeadDecimator capped(200.0);
  check(capped.every(1000.0) == 5 && capped.every(200.0) == 1, "1 kHz keeps 1 in 5, 200 Hz all");
  check(capped.every(500.0) == 3, "500 Hz keeps 1 in 3: never above the cap");
  int kept = 0;
  for (int i = 0; i < 1000; ++i) {
    kept += capped.due(1000.0) ? 1 : 0;
  }
  check(kept == 200, "a second of a 1 kHz chip is 200 samples out");
  pepin::HeadDecimator open(0.0);
  check(open.every(1000.0) == 1, "a cap of 0 publishes every sample");

  if (failures == 0) {
    std::printf("the contract holds\n");
  }
  return failures == 0 ? 0 : 1;
}

/// Parity with pepin.head_imu: the same rows in, the same samples, rate and decisions out.
int parity()
{
  std::string line;
  pepin::HeadConfig config;
  pepin::HeadRate rate;
  pepin::HeadDecimator capped(200.0);
  while (std::getline(std::cin, line)) {
    std::istringstream words(line);
    std::string kind;
    words >> kind;
    if (kind == "config") {
      words >> config.cfg >> config.rate_hz >> config.filter_delay_s;
      continue;
    }
    double row[8];
    for (double & value : row) {
      std::string token;
      words >> token;
      value = std::strtod(token.c_str(), nullptr);
    }
    const auto sample = pepin::sample_from_row(row, config.cfg);
    if (!sample.has_value()) {
      std::printf("refused\n");
      continue;
    }
    rate.add(sample->t_mono_s, config.rate_hz);
    const bool due = capped.due(config.rate_hz);
    std::printf(
      "sample %.9f %.1f %.9f %.9f %.9f %.9f %.9f %.9f %d\n", sample->t_mono_s, sample->esp_us,
      sample->gyro[0], sample->gyro[1], sample->gyro[2], sample->accel[0], sample->accel[1],
      sample->accel[2], due ? 1 : 0);
  }
  std::printf(
    "rate %ld %ld %ld %.9f\n", rate.samples(), rate.gaps(), rate.out_of_order(),
    rate.longest_gap_s());
  return 0;
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc > 1 && std::strcmp(argv[1], "--stdin") == 0) {
    return parity();
  }
  return self_checks();
}
