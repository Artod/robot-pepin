// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The neck as numbers, the C++ twin of pepin.neck: two encoder readings (pan, tilt) into the
// joint angles and into base_link -> camera_link. Pure, no ROS: the base bridge feeds it the
// ticks of every state line and publishes what it answers; test/neck_contract.cpp holds it to
// the Python model row for row (tests/unit/test_base_cpp_contracts.py).
//
// The model is anchored at the REFERENCE pose: the camera's measured mount (config/neck.json's
// reference x/y/z/pitch, config/camera.json's mount of the same day) and the ticks the two
// servos read in it. Then
//
//     pan   = pan_sign  * (pan_ticks  - reference_pan)  * 2 pi / 4096          (+ left)
//     pitch = mount_pitch + tilt_sign * (tilt_ticks - reference_tilt) * 2 pi / 4096   (+ down)
//
// and the camera hangs on a chain: base_link -> pan pivot -> Rz(pan) -> tilt pivot -> Ry(pitch)
// -> lens, with the two lever arms of the pivot block; the pan pivot is derived so the reference
// ticks land the camera exactly on the measured mount. Unread reference ticks (negative here,
// null in the file) measure nothing: every reading answers the static mount.

#ifndef PEPIN_BASE_CPP__NECK_HPP_
#define PEPIN_BASE_CPP__NECK_HPP_

#include <array>
#include <cmath>

namespace pepin
{

constexpr int kNeckTicksPerTurn = 4096;
constexpr double kNeckRadPerTick = 2.0 * 3.14159265358979323846 / kNeckTicksPerTurn;

/// config/neck.json's geometry: the reference pose and ticks, the signs, the lever arms (metres).
struct NeckModel
{
  int reference_pan_ticks = -1;   ///< negative: unread, the model answers the static mount
  int reference_tilt_ticks = -1;
  int pan_sign = 1;               ///< which way a rising tick count turns the camera
  int tilt_sign = 1;
  double mount_x_m = 0.0;         ///< the camera's measured pose at the reference, base_link
  double mount_y_m = 0.0;
  double mount_z_m = 0.0;
  double mount_pitch_rad = 0.0;   ///< positive down
  double tilt_from_pan_x_m = 0.0; ///< pan axis -> tilt axis, in the pan frame (x fwd, z up)
  double tilt_from_pan_z_m = 0.0;
  double camera_from_tilt_x_m = 0.0;  ///< tilt axis -> lens, in the tilt frame
  double camera_from_tilt_z_m = 0.0;

  /// Both reference ticks have been read: only then do the encoders measure anything.
  bool known() const {return reference_pan_ticks >= 0 && reference_tilt_ticks >= 0;}
};

/// The neck's joint angles, radians: pan positive left, pitch positive down (REP 103).
struct NeckAngles
{
  double pan_rad;
  double pitch_rad;
};

/// base_link -> camera_link as a translation (metres) and roll/pitch/yaw (radians).
struct NeckPose
{
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

/// Signed encoder travel from `reference` to `ticks` the short way round: a 12-bit reading never
/// means more than half a revolution away (Python's modulo, never negative, is spelled out here).
inline int ticks_from(int reference, int ticks)
{
  int travel = (ticks - reference + kNeckTicksPerTurn / 2) % kNeckTicksPerTurn;
  if (travel < 0) {
    travel += kNeckTicksPerTurn;
  }
  return travel - kNeckTicksPerTurn / 2;
}

/// Encoder ticks of both servos into the joint angles; the static mount while the reference is
/// unread.
inline NeckAngles joint_angles(const NeckModel & model, int pan_ticks, int tilt_ticks)
{
  if (!model.known()) {
    return {0.0, model.mount_pitch_rad};
  }
  return {
    model.pan_sign * ticks_from(model.reference_pan_ticks, pan_ticks) * kNeckRadPerTick,
    model.mount_pitch_rad +
    model.tilt_sign * ticks_from(model.reference_tilt_ticks, tilt_ticks) * kNeckRadPerTick};
}

/// A lever (x forward, z up) after a pitch about y; a downward pitch dips its tip.
inline std::array<double, 2> pitched(double x, double z, double pitch)
{
  return {x * std::cos(pitch) + z * std::sin(pitch), -x * std::sin(pitch) + z * std::cos(pitch)};
}

/// base_link -> camera_link at these joint angles (pepin.neck.camera_pose).
inline NeckPose camera_pose(const NeckModel & model, const NeckAngles & angles)
{
  const auto at_reference =
    pitched(model.camera_from_tilt_x_m, model.camera_from_tilt_z_m, model.mount_pitch_rad);
  const double pivot_x = model.mount_x_m - model.tilt_from_pan_x_m - at_reference[0];
  const double pivot_y = model.mount_y_m;
  const double pivot_z = model.mount_z_m - model.tilt_from_pan_z_m - at_reference[1];
  const auto lens =
    pitched(model.camera_from_tilt_x_m, model.camera_from_tilt_z_m, angles.pitch_rad);
  const double forward = model.tilt_from_pan_x_m + lens[0];
  const double up = model.tilt_from_pan_z_m + lens[1];
  return {
    pivot_x + std::cos(angles.pan_rad) * forward,
    pivot_y + std::sin(angles.pan_rad) * forward,
    pivot_z + up,
    0.0,
    angles.pitch_rad,
    angles.pan_rad};
}

/// x, y, z, w of the rotation yaw * pitch * roll (pepin.camera.quaternion_from_rpy).
inline std::array<double, 4> quaternion_from_rpy(double roll, double pitch, double yaw)
{
  const double cr = std::cos(roll / 2.0);
  const double sr = std::sin(roll / 2.0);
  const double cp = std::cos(pitch / 2.0);
  const double sp = std::sin(pitch / 2.0);
  const double cy = std::cos(yaw / 2.0);
  const double sy = std::sin(yaw / 2.0);
  return {
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
    cr * cp * cy + sr * sp * sy};
}

/// The neck's publishing rate cap on the state lines' own clock (pepin.base_server.PublishGrid):
/// a line is due within half a 50 Hz line of its grid point, so a cap at the line rate takes
/// every line and a lower one averages out to exactly its rate; a gap starts a fresh grid.
class NeckGrid
{
public:
  /// `hz` is the cap; zero or less publishes every line.
  explicit NeckGrid(double hz)
  : every_s_(hz > 0.0 ? 1.0 / hz : 0.0) {}

  /// Whether the line stamped `stamp_s` is published; true moves the grid on by one period.
  bool due(double stamp_s)
  {
    if (stamp_s + kSlackS < next_s_) {
      return false;
    }
    next_s_ += every_s_;
    if (next_s_ <= stamp_s) {
      next_s_ = stamp_s + every_s_;
    }
    return true;
  }

private:
  static constexpr double kSlackS = 0.01;  // half the 20 ms between state lines
  double every_s_;
  double next_s_ = 0.0;
};

}  // namespace pepin

#endif  // PEPIN_BASE_CPP__NECK_HPP_
