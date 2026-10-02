// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The neck model of neck.hpp against its Python twin, pepin.neck. Like the other contracts a
// stand-alone main() with no ROS and no gtest. It checks the rate cap's own table, then reads a
// model and readings from stdin and answers each reading with the joint angles and the camera's
// pose; tests/unit/test_base_cpp_contracts.py writes the repo's config/neck.json (and variants)
// in and holds every answer to pepin.neck's. Stdin, whitespace-separated:
//
//     ref_pan ref_tilt pan_sign tilt_sign x y z pitch_rad tpx tpz ctx ctz
//     pan_ticks tilt_ticks          (one reading per line, to the end)
//
// Out, per reading: pan_rad pitch_rad x y z qx qy qz qw, then "the contract holds" when every
// row of the rate cap's table agreed. One command (no line continuation: GCC's -Wcomment):
//
//     c++ -std=c++17 -Wall -Wextra -Wpedantic -O2 -I ros/pepin_base_cpp/include
//         ros/pepin_base_cpp/test/neck_contract.cpp -o /tmp/neck_contract

#include <cstdio>

#include "pepin_base_cpp/neck.hpp"

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

/// The rate cap: every line at the line rate (lines a little late or early), a lower cap
/// averaging out to exactly its rate, a gap starting a fresh grid.
void grid_table()
{
  pepin::NeckGrid every(50.0);
  int published = 0;
  for (int i = 0; i < 500; ++i) {
    published += every.due(1000.0 + i * 0.0201) ? 1 : 0;
  }
  check(published == 500, "50 Hz on 50 Hz lines, each a little late: every line");
  pepin::NeckGrid early(50.0);
  const double stamps[] = {1000.0, 1000.0195, 1000.0391, 1000.0602};
  published = 0;
  for (double stamp : stamps) {
    published += early.due(stamp) ? 1 : 0;
  }
  check(published == 4, "a line a little early still counts");
  pepin::NeckGrid twenty(20.0);
  published = 0;
  for (int i = 0; i < 500; ++i) {
    published += twenty.due(1000.0 + i * 0.02) ? 1 : 0;
  }
  check(published == 200, "20 Hz on 50 Hz lines: 200 lines in 10 s");
  pepin::NeckGrid gap(50.0);
  check(gap.due(1000.0) && gap.due(1000.02) && gap.due(1000.5), "a gap: published at once");
  check(gap.due(1000.52) && !gap.due(1000.525), "then a fresh grid, no burst");
  pepin::NeckGrid uncapped(0.0);
  check(uncapped.due(1.0) && uncapped.due(1.0) && uncapped.due(1.001), "no cap: every line");
}

}  // namespace

int main()
{
  grid_table();
  pepin::NeckModel model;
  const int read = std::scanf(
    "%d %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf", &model.reference_pan_ticks,
    &model.reference_tilt_ticks, &model.pan_sign, &model.tilt_sign, &model.mount_x_m,
    &model.mount_y_m, &model.mount_z_m, &model.mount_pitch_rad, &model.tilt_from_pan_x_m,
    &model.tilt_from_pan_z_m, &model.camera_from_tilt_x_m, &model.camera_from_tilt_z_m);
  if (read == 12) {
    int pan_ticks = 0;
    int tilt_ticks = 0;
    while (std::scanf("%d %d", &pan_ticks, &tilt_ticks) == 2) {
      const auto angles = pepin::joint_angles(model, pan_ticks, tilt_ticks);
      const auto pose = pepin::camera_pose(model, angles);
      const auto q = pepin::quaternion_from_rpy(pose.roll, pose.pitch, pose.yaw);
      std::printf(
        "%.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n", angles.pan_rad,
        angles.pitch_rad, pose.x, pose.y, pose.z, q[0], q[1], q[2], q[3]);
    }
  }
  if (failures != 0) {
    std::printf("%d disagreements\n", failures);
    return 1;
  }
  std::printf("the contract holds\n");
  return 0;
}
