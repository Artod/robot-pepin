// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The wheel noise law of wheel_noise.hpp against its Python twin, pepin.wheel_noise. Like the
// other contracts a stand-alone main() with no ROS and no gtest. It checks the fit's defaults on
// three motions, then reads a law and measured twists from stdin and answers each with the
// covariance's vx and vyaw entries; tests/unit/test_base_cpp_contracts.py writes the repo's
// config/base.json law in and holds every answer to pepin.wheel_noise's. Stdin,
// whitespace-separated:
//
//     v_per_yaw_rate v_floor yaw_per_yaw_rate yaw_floor moving_m_s moving_rad_s rate_hz
//     v w          (one measured twist per line, to the end)
//
// Out, per twist: var_vx var_vyaw moving (0/1), then "the contract holds". One command:
//
//     c++ -std=c++17 -Wall -Wextra -Wpedantic -O2 -I ros/pepin_base_cpp/include
//         ros/pepin_base_cpp/test/wheel_noise_contract.cpp -o /tmp/wheel_noise_contract

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

#include "pepin_base_cpp/wheel_noise.hpp"

namespace
{

int failures = 0;

void check(bool ok, const char * what)
{
  if (!ok) {
    std::printf("FAIL  %s\n", what);
    ++failures;
  }
}

bool near(double a, double b) {return std::fabs(a - b) <= 1e-12 + 1e-9 * std::fabs(b);}

/// protocol.hpp's odometry_twist_covariance(), spelled out: that header needs nlohmann/json,
/// which a laptop compiling this contract need not have.
std::array<double, 36> constant_covariance()
{
  std::array<double, 36> matrix{};
  const double variances[6] = {0.001, 0.001, 0.001, 0.001, 0.001, 0.01};
  for (std::size_t i = 0; i < 6; ++i) {
    matrix[i * 6 + i] = variances[i];
  }
  return matrix;
}

/// The fit's defaults: rest keeps the constant, a straight at 0.3 m/s gets the floors, a pivot at
/// 1 rad/s the floors plus the turn's share; the other entries never move.
void defaults_table()
{
  const pepin::WheelNoiseLaw law;
  const auto constant = constant_covariance();
  const auto rest = pepin::wheel_twist_covariance(law, {0.0, 0.0}, constant);
  check(rest == constant, "rest: the constant covariance, untouched");
  const auto creep = pepin::wheel_twist_covariance(law, {0.02, 0.03}, constant);
  check(creep == constant, "2 cm/s and 0.03 rad/s: still rest");
  const auto straight = pepin::wheel_twist_covariance(law, {0.3, 0.0}, constant);
  check(near(straight[0], 50.0 * 0.026 * 0.026), "straight 0.3 m/s: vx 50 * 0.026^2");
  check(near(straight[35], 50.0 * 0.038 * 0.038), "straight 0.3 m/s: vyaw 50 * 0.038^2");
  const auto pivot = pepin::wheel_twist_covariance(law, {0.0, -1.0}, constant);
  check(near(pivot[0], 50.0 * 0.060 * 0.060), "pivot 1 rad/s: vx 50 * (0.034 + 0.026)^2");
  check(near(pivot[35], 50.0 * 0.238 * 0.238), "pivot 1 rad/s: vyaw 50 * (0.20 + 0.038)^2");
  for (std::size_t i = 1; i < 35; ++i) {
    check(pivot[i] == constant[i], "only vx and vyaw follow the law");
  }
}

}  // namespace

int main()
{
  defaults_table();
  pepin::WheelNoiseLaw law;
  if (std::scanf(
      "%lf %lf %lf %lf %lf %lf %lf", &law.v_per_yaw_rate, &law.v_floor_m_s,
      &law.yaw_per_yaw_rate, &law.yaw_floor_rad_s, &law.moving_m_s, &law.moving_rad_s,
      &law.rate_hz) == 7)
  {
    const auto constant = constant_covariance();
    double v = 0.0;
    double w = 0.0;
    while (std::scanf("%lf %lf", &v, &w) == 2) {
      const auto m = pepin::wheel_twist_covariance(law, {v, w}, constant);
      std::printf("%.17g %.17g %d\n", m[0], m[35], pepin::wheels_moving(law, {v, w}) ? 1 : 0);
    }
  }
  if (failures == 0) {
    std::printf("the contract holds\n");
  }
  return failures == 0 ? 0 : 1;
}
