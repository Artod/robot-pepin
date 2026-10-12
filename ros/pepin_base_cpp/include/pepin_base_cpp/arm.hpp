// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The SO-101 arm's joints as the base server's state lines carry them, into the form its readers
// take. pepin.arm_driver puts `"arm": {"deg": [six angles], ...}` in every line whose read heard
// the arm (every tick while it is energised, `idle_read_hz` while it rests, never on a board
// without config/arm.json): DEGREES, in pepin.expression.trajectory.JOINTS order, counted as
// lerobot's calibration counts them (the middle of each servo's EEPROM range is 0, the gripper's
// closed end -10), which is the URDF's convention. The bridge publishes them as /arm/joint_states
// (sensor_msgs/JointState: the URDF's joint names, RADIANS), what pepin.arm reads for
// depth_fusion's arm_filter (config/arm.json `joints.topic`). Pure, no ROS and no JSON:
// test/arm_contract.cpp holds it.

#ifndef PEPIN_BASE_CPP__ARM_HPP_
#define PEPIN_BASE_CPP__ARM_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace pepin
{

constexpr std::size_t kArmJoints = 6;

/// The URDF's joint names in the wire's order (pepin.expression.trajectory.JOINTS).
inline constexpr std::array<const char *, kArmJoints> kArmJointNames{
  "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"};

constexpr double kArmRadPerDeg = 3.14159265358979323846 / 180.0;

/// A line's `arm.deg` as six angles in degrees; nothing unless exactly six finite numbers came.
inline std::optional<std::array<double, kArmJoints>> arm_degrees(const std::vector<double> & values)
{
  if (values.size() != kArmJoints) {
    return std::nullopt;
  }
  std::array<double, kArmJoints> degrees{};
  for (std::size_t i = 0; i < kArmJoints; ++i) {
    if (!std::isfinite(values[i])) {
      return std::nullopt;
    }
    degrees[i] = values[i];
  }
  return degrees;
}

/// The angles as /arm/joint_states' positions: radians, the same order.
inline std::array<double, kArmJoints> arm_radians(const std::array<double, kArmJoints> & degrees)
{
  std::array<double, kArmJoints> radians{};
  for (std::size_t i = 0; i < kArmJoints; ++i) {
    radians[i] = degrees[i] * kArmRadPerDeg;
  }
  return radians;
}

}  // namespace pepin

#endif  // PEPIN_BASE_CPP__ARM_HPP_
