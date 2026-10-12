// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// The arm's joints (arm.hpp) as the bridge publishes them, held to the wire pepin.arm_driver
// writes and to what pepin.arm reads, through --stdin. Stand-alone, no ROS. Where
// nlohmann/json.hpp is on the include path (the board image, the laptop containers) the state
// line's parse in protocol.hpp is checked too; the host's tests compile it without. One command
// (no line continuation: GCC's -Wcomment):
//
//     c++ -std=c++17 -Wall -Wextra -Wpedantic -O2 -I ros/pepin_base_cpp/include
//         ros/pepin_base_cpp/test/arm_contract.cpp -o /tmp/arm_contract
//     /tmp/arm_contract           # self-checks: prints "the contract holds"
//     /tmp/arm_contract --stdin   # parity: "names ..." first, then per row of angles in degrees
//                                 # (any count) "rad r1 .. r6" or "refused"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "pepin_base_cpp/arm.hpp"

#if __has_include(<nlohmann/json.hpp>)
#include "pepin_base_cpp/protocol.hpp"
#define PEPIN_ARM_CONTRACT_PARSE 1
#endif

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

constexpr double kPi = 3.14159265358979323846;

void self_checks()
{
  const char * urdf[] = {
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"};
  bool names = pepin::kArmJointNames.size() == 6;
  for (std::size_t i = 0; names && i < 6; ++i) {
    names = std::strcmp(pepin::kArmJointNames[i], urdf[i]) == 0;
  }
  check(names, "the URDF's six joint names, in the wire's order");

  const auto degrees = pepin::arm_degrees({90.0, -90.0, 180.0, 0.0, 45.0, -10.0});
  check(degrees.has_value(), "six finite angles are a sample");
  const auto radians = pepin::arm_radians(*degrees);
  check(near(radians[0], kPi / 2) && near(radians[1], -kPi / 2), "degrees in, radians out");
  check(near(radians[2], kPi) && radians[3] == 0.0 && near(radians[4], kPi / 4), "every joint");
  check(near(radians[5], -10.0 * kPi / 180.0), "the gripper too, its -10 deg closed end kept");

  check(!pepin::arm_degrees({}).has_value(), "no angles, no sample");
  check(!pepin::arm_degrees({1, 2, 3, 4, 5}).has_value(), "five angles are no sample");
  check(!pepin::arm_degrees({1, 2, 3, 4, 5, 6, 7}).has_value(), "nor are seven");
  check(!pepin::arm_degrees({1, 2, NAN, 4, 5, 6}).has_value(), "a NaN costs the sample");
  check(!pepin::arm_degrees({1, 2, 3, 4, 5, INFINITY}).has_value(), "so does an infinity");
}

#ifdef PEPIN_ARM_CONTRACT_PARSE
/// One state line as the base server writes it, with `arm` (raw JSON, empty: none) appended.
std::optional<pepin::BaseState> line(const std::string & arm)
{
  std::string text =
    R"({"type":"state","t":1000.0,"x":0.0,"y":0.0,"theta":0.0,"dl":0.0,"dr":0.0,"v":0.0,)"
    R"("w":0.0,"moving":false,"armed":false,"deadman":false,"bus_ok":true,"bus_p95_ms":0.0)";
  text += arm.empty() ? "}" : ",\"arm\":" + arm + "}";
  const auto message = pepin::detail::decode(text);
  return message.has_value() ? pepin::parse_state(*message) : std::nullopt;
}

void parse_checks()
{
  const auto with = line(
    R"({"deg":[-34.3,-102.6,96.8,76.6,-54.1,-0.4],"torque":false,"phase":"off","id":"",)"
    R"("progress_s":0.0,"error":""})");
  check(with.has_value() && with->arm_read, "a line with the arm carries its angles");
  check(
    with.has_value() && with->arm_deg[0] == -34.3 && with->arm_deg[1] == -102.6 &&
    with->arm_deg[5] == -0.4, "in the wire's order, degrees as written");
  const auto without = line("");
  check(without.has_value() && !without->arm_read, "a line without the arm: no sample");
  const char * broken[] = {
    R"({"deg":[1,2,3,4,5]})", R"({"deg":[1,2,3,4,5,6,7]})", R"({"deg":[1,2,3,4,5,"x"]})",
    R"({"deg":[1,2,3,4,5,6,"x"]})", R"({"deg":"1,2,3,4,5,6"})", R"({"torque":true})",
    R"([1,2,3,4,5,6])", "null"};
  for (const char * arm : broken) {
    const auto state = line(arm);
    check(state.has_value(), "a broken arm field never costs the line");
    check(state.has_value() && !state->arm_read, "and gives no sample");
  }
}
#endif

/// Parity: the names, then each row of degrees as the radians published or "refused".
int parity()
{
  std::printf("names");
  for (const char * name : pepin::kArmJointNames) {
    std::printf(" %s", name);
  }
  std::printf("\n");
  std::string text;
  while (std::getline(std::cin, text)) {
    std::vector<double> values;
    const char * cursor = text.c_str();
    char * end = nullptr;
    for (double value = std::strtod(cursor, &end); end != cursor;
      value = std::strtod(cursor, &end))
    {
      values.push_back(value);
      cursor = end;
    }
    const auto degrees = pepin::arm_degrees(values);
    if (!degrees.has_value()) {
      std::printf("refused\n");
      continue;
    }
    std::printf("rad");
    for (double radians : pepin::arm_radians(*degrees)) {
      std::printf(" %.17g", radians);
    }
    std::printf("\n");
  }
  return 0;
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc > 1 && std::strcmp(argv[1], "--stdin") == 0) {
    return parity();
  }
  self_checks();
#ifdef PEPIN_ARM_CONTRACT_PARSE
  parse_checks();
  std::printf("the state line's parse checked (nlohmann/json.hpp found)\n");
#else
  std::printf("the state line's parse not checked here (no nlohmann/json.hpp)\n");
#endif
  if (failures != 0) {
    std::printf("%d disagreements\n", failures);
    return 1;
  }
  std::printf("the contract holds\n");
  return 0;
}
