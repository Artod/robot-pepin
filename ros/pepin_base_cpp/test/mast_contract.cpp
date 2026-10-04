// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The mast-sway filter (mast.hpp), held to what vio.md section 5 promises, and to its Python twin
// (pepin.mast) through --stdin. Stand-alone, no ROS:
//
//     c++ -std=c++17 -Wall -Wextra -Wpedantic -O2 -I ros/pepin_base_cpp/include \
//         ros/pepin_base_cpp/test/mast_contract.cpp -o /tmp/mast_contract
//     /tmp/mast_contract           # self-checks: prints "the contract holds"
//     /tmp/mast_contract --stdin   # parity: lines "t hx hy hz yaw" or "hold"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "pepin_base_cpp/mast.hpp"

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

constexpr double kDeg = 3.14159265358979323846 / 180.0;

/// vio.md's pass line on a synthetic ring: 5.3 Hz, 0.35 deg p-p, armed at a 0.17 deg deflection;
/// over the first full ring period after arming the residual (truth minus published pitch) is
/// at most 30 % of the ring's p-p.
void ring_is_removed()
{
  pepin::MastFilter filter;
  const double rate_hz = 200.0, f = 5.3, amplitude = 0.175 * kDeg;
  const double phase = std::asin(0.17 / 0.175);
  const double period = 1.0 / f;
  double raw_min = 1e9, raw_max = -1e9, res_min = 1e9, res_max = -1e9;
  for (int i = 0; i < 400; ++i) {
    const double t = 50.0 + i / rate_hz;
    const double local = t - 50.0;
    const double omega = 2.0 * pepin::kMastPi * f;
    const double truth = amplitude * std::sin(omega * local + phase);
    const double rate = amplitude * omega * std::cos(omega * local + phase);
    const auto out = filter.update(t, {0.0, rate, 0.0}, 0.0);
    if (local >= period && local < 2.0 * period) {
      raw_min = std::min(raw_min, truth);
      raw_max = std::max(raw_max, truth);
      res_min = std::min(res_min, truth - out.theta[1]);
      res_max = std::max(res_max, truth - out.theta[1]);
    }
  }
  const double removed = 1.0 - (res_max - res_min) / (raw_max - raw_min);
  std::printf("ring: %.0f %% removed over the first full period\n", removed * 100.0);
  check(removed >= 0.70, "at least 70 % of the ring removed from the first period");
}

void rest_held_and_base_yaw()
{
  pepin::MastFilter filter;
  for (int i = 0; i < 2000; ++i) {
    const auto out = filter.update(i / 200.0, {0.0, 0.0, 0.3}, 0.3);  // the cart turns: no sway
    check(!out.held, "armed");
    if (i == 1999) {
      check(std::fabs(out.theta[2]) < 1e-12, "a base turn the head shares is no sway");
    }
  }
  filter.hold();
  check(filter.held(), "held when told");
  const auto nan_in = filter.update(20.0, {NAN, 0.0, 0.0}, 0.0);
  check(nan_in.held && std::isnan(nan_in.theta[0]), "a NaN rate holds, and says so with NaN");
  const auto rearmed = filter.update(20.01, {0.0, 1.0, 0.0}, 0.0);
  check(!rearmed.held && rearmed.theta[1] == 0.0, "the first sample after a hold arms at zero");
  const auto gap = filter.update(21.0, {0.0, 1.0, 0.0}, 0.0);
  check(gap.theta[1] == 0.0, "a gap past max_dt_s re-arms at zero");
}

void composition_and_sign()
{
  const std::array<double, 3> hinge{-0.058, 0.0, 0.78};
  const std::array<double, 3> neck{0.03, 0.0, 1.20};
  const auto still = pepin::compose_sway({0.0, 0.0, 0.0}, hinge, neck, 0.0, 0.4154, 0.2);
  const auto q = pepin::quaternion_from_matrix(pepin::rotation_from_rpy(0.0, 0.4154, 0.2));
  check(
    std::fabs(still.first[0] - neck[0]) < 1e-12 && std::fabs(still.first[2] - neck[2]) < 1e-12,
    "theta 0 is the neck's own edge (translation)");
  check(
    std::fabs(still.second[0] - q[0]) < 1e-12 && std::fabs(still.second[3] - q[3]) < 1e-12,
    "theta 0 is the neck's own edge (rotation)");
  // A positive sway pitch (REP 103: about +y, nose down) tilts the camera DOWN and carries it
  // forward over the hinge.
  const auto swayed = pepin::compose_sway({0.0, 0.01, 0.0}, hinge, neck, 0.0, 0.0, 0.0);
  const auto r = pepin::rotation_from_rpy(0.0, 0.01, 0.0);
  check(r[6] < 0.0, "the camera's x axis points below the horizon after a positive pitch");
  check(swayed.first[0] > neck[0], "the lens moves forward about a hinge below it");
  const double lever = std::hypot(neck[0] - hinge[0], neck[2] - hinge[2]);
  const double moved = std::hypot(swayed.first[0] - neck[0], swayed.first[2] - neck[2]);
  check(std::fabs(moved - lever * 0.01) < 1e-4, "an arc of lever x theta");
  const auto in_base = pepin::head_rate_in_base(
    {0.0, 1.0, 0.0}, pepin::rotation_from_rpy(0.0, 0.0, 0.0), 90.0 * kDeg, 0.0);
  check(std::fabs(in_base[0] + 1.0) < 1e-12, "a head panned 90 deg left: its y rate is base -x");
}

int self_checks()
{
  ring_is_removed();
  rest_held_and_base_yaw();
  composition_and_sign();
  if (failures == 0) {
    std::printf("the contract holds\n");
  }
  return failures == 0 ? 0 : 1;
}

int parity()
{
  pepin::MastFilter filter;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "hold") {
      filter.hold();
      std::printf("held\n");
      continue;
    }
    std::istringstream words(line);
    double v[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (double & value : v) {  // strtod reads "nan", operator>> does not
      std::string token;
      words >> token;
      value = std::strtod(token.c_str(), nullptr);
    }
    const auto out = filter.update(v[0], {v[1], v[2], v[3]}, v[4]);
    std::printf(
      "%d %.12e %.12e %.12e %.12e %.12e %.12e\n", out.held ? 1 : 0, out.theta[0], out.theta[1],
      out.theta[2], out.omega[0], out.omega[1], out.omega[2]);
  }
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
