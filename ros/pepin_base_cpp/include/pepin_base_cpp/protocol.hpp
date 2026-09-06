// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// Wire format of the board's base server, with no ROS in sight — the C++ twin of
// pepin_bringup/protocol.py. Parse a line, encode a command, nothing else, so the
// same fixture (test/protocol_samples.json) can check both sides.
//
// Wire format:
//
//   base -> us   {"type":"state","t":..,"x":..,"y":..,"theta":..,"dl":..,"dr":..,
//                 "v":..,"w":..,"moving":bool,"armed":bool,"deadman":bool,
//                 "bus_ok":bool,"bus_p95_ms":..}
//   base -> us   {"type":"pong",...}                     answer to a ping; ignored here
//   us -> base   {"cmd":"twist","v":<m/s>,"w":<rad/s>}   drive; re-arms the deadman
//   us -> base   {"cmd":"stop"}                          stop the wheels now
//
// Conventions: x forward, y left, theta counter-clockwise, SI units.

#ifndef PEPIN_BASE_CPP__PROTOCOL_HPP_
#define PEPIN_BASE_CPP__PROTOCOL_HPP_

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pepin
{

/// One state line from the base server: where the wheels think they are, and how they feel.
struct BaseState
{
  double stamp_s;    ///< board clock (its time.monotonic) when the line was made
  double x;          ///< wheel odometry integrated on the board, odometry frame, metres
  double y;
  double theta;      ///< radians, counter-clockwise from x
  double d_left_m;   ///< left wheel travel since the previous state line
  double d_right_m;
  double v;          ///< twist currently applied, m/s forward
  double w;          ///< rad/s counter-clockwise
  bool moving;       ///< a non-zero twist is being applied
  bool armed;        ///< torque on (the wheels resist being pushed)
  bool deadman;      ///< the board stopped the wheels because commands stopped arriving
  bool bus_ok;       ///< the servos answered on the last tick
  double bus_p95_ms; ///< board-local servo round trip, 95th percentile
};

namespace detail
{

/// Python's float() over a JSON value: numbers, bools and numeric strings; false if it refuses.
inline bool as_double(const nlohmann::json & value, double & out)
{
  if (value.is_number()) {
    out = value.get<double>();
    return true;
  }
  if (value.is_boolean()) {
    out = value.get<bool>() ? 1.0 : 0.0;
    return true;
  }
  if (value.is_string()) {  // the board never sends one, but float("1.5") is what Python does
    const std::string & text = value.get_ref<const std::string &>();
    try {
      std::size_t used = 0;
      out = std::stod(text, &used);
      return used == text.size();
    } catch (const std::exception &) {
      return false;
    }
  }
  return false;
}

/// Python's bool() over a JSON value: emptiness and zero are false, everything else is true.
inline bool as_bool(const nlohmann::json & value)
{
  if (value.is_boolean()) {
    return value.get<bool>();
  }
  if (value.is_number()) {
    return value.get<double>() != 0.0;
  }
  if (value.is_string()) {
    return !value.get_ref<const std::string &>().empty();
  }
  return !value.is_null() && !value.empty();
}

/// Read one required numeric field; false if it is missing or not a number.
inline bool number_field(const nlohmann::json & message, const char * key, double & out)
{
  const auto found = message.find(key);
  return found != message.end() && as_double(*found, out);
}

/// Read one required boolean field; false if it is missing (any type is truthy or not).
inline bool bool_field(const nlohmann::json & message, const char * key, bool & out)
{
  const auto found = message.find(key);
  if (found == message.end()) {
    return false;
  }
  out = as_bool(*found);
  return true;
}

/// One raw line into a JSON object, or nothing if it is not one.
inline std::optional<nlohmann::json> decode(std::string_view line)
{
  const auto first = line.find_first_not_of(" \t\r\n\f\v");
  if (first == std::string_view::npos) {
    return std::nullopt;  // a blank line costs that line and nothing else
  }
  auto message = nlohmann::json::parse(line.begin(), line.end(), nullptr, false);
  if (message.is_discarded() || !message.is_object()) {
    return std::nullopt;
  }
  return message;
}

}  // namespace detail

/// A `state` line as a BaseState; nothing for any other or malformed message.
inline std::optional<BaseState> parse_state(const nlohmann::json & message)
{
  if (!message.is_object()) {
    return std::nullopt;
  }
  const auto type = message.find("type");
  if (type == message.end() || !type->is_string() ||
    type->get_ref<const std::string &>() != "state")
  {
    return std::nullopt;
  }
  BaseState state{};
  const bool numbers =
    detail::number_field(message, "t", state.stamp_s) &&
    detail::number_field(message, "x", state.x) &&
    detail::number_field(message, "y", state.y) &&
    detail::number_field(message, "theta", state.theta) &&
    detail::number_field(message, "dl", state.d_left_m) &&
    detail::number_field(message, "dr", state.d_right_m) &&
    detail::number_field(message, "v", state.v) &&
    detail::number_field(message, "w", state.w);
  const bool flags =
    detail::bool_field(message, "moving", state.moving) &&
    detail::bool_field(message, "armed", state.armed) &&
    detail::bool_field(message, "deadman", state.deadman) &&
    detail::bool_field(message, "bus_ok", state.bus_ok);
  if (!numbers || !flags) {
    return std::nullopt;  // a truncated line is dropped, never raised
  }
  const auto p95 = message.find("bus_p95_ms");  // optional: older boards do not send it
  if (p95 == message.end()) {
    state.bus_p95_ms = 0.0;
  } else if (!detail::as_double(*p95, state.bus_p95_ms)) {
    return std::nullopt;
  }
  return state;
}

/// One `twist` command line: `v` m/s forward, `w` rad/s counter-clockwise, newline included.
inline std::string encode_twist(double v, double w)
{
  const nlohmann::ordered_json message{{"cmd", "twist"}, {"v", v}, {"w", w}};
  return message.dump() + "\n";
}

/// One `stop` command line: the board cuts the wheels on receipt.
inline std::string encode_stop()
{
  const nlohmann::ordered_json message{{"cmd", "stop"}};
  return message.dump() + "\n";
}

/// Row-major 6x6 covariance with `variances` on the diagonal and zeros elsewhere.
inline std::array<double, 36> diagonal(const std::array<double, 6> & variances)
{
  std::array<double, 36> matrix{};
  for (std::size_t i = 0; i < variances.size(); ++i) {
    matrix[i * 6 + i] = variances[i];
  }
  return matrix;
}

// One odometry sample, the diff_drive_controller defaults: wheel odometry is precise per tick
// and hopeless over a long run, and the drift is a consumer's problem (a filter, or SLAM
// correcting odom->map). z/roll/pitch get the same small number: the robot cannot leave the floor.
/// Row-major 6x6 pose covariance for a nav_msgs/Odometry from wheel odometry.
inline std::array<double, 36> odometry_pose_covariance()
{
  return diagonal({0.001, 0.001, 0.001, 0.001, 0.001, 0.01});
}

/// Row-major 6x6 twist covariance for a nav_msgs/Odometry from wheel odometry.
inline std::array<double, 36> odometry_twist_covariance()
{
  return diagonal({0.001, 0.001, 0.001, 0.001, 0.001, 0.01});
}

/// Reassembles JSON objects out of arbitrary TCP chunks.
///
/// TCP hands out bytes, not lines: one recv can hold three messages and half of a
/// fourth. Feed it what arrives and take back whole objects. A line that is not a
/// JSON object costs that line and nothing else, and a stream that never sends a
/// newline cannot grow the buffer past `max_line_bytes`.
class LineReader
{
public:
  /// Prepare an empty reader that drops any line longer than `max_line_bytes`.
  explicit LineReader(std::size_t max_line_bytes = 1u << 16)
  : max_line_bytes_(max_line_bytes) {}

  /// Add received bytes; return every complete JSON object they finished, in order.
  std::vector<nlohmann::json> feed(const char * data, std::size_t size)
  {
    buffer_.append(data, size);
    std::vector<nlohmann::json> messages;
    std::size_t start = 0;
    for (std::size_t end = buffer_.find('\n', start); end != std::string::npos;
      end = buffer_.find('\n', start))
    {
      auto message = detail::decode(std::string_view(buffer_).substr(start, end - start));
      start = end + 1;
      if (message.has_value()) {
        messages.push_back(std::move(*message));
      }
    }
    buffer_.erase(0, start);
    if (buffer_.size() > max_line_bytes_) {
      buffer_.clear();  // no newline in sight: the sender is not talking our language
    }
    return messages;
  }

  /// Convenience overload for tests and fixtures.
  std::vector<nlohmann::json> feed(std::string_view chunk)
  {
    return feed(chunk.data(), chunk.size());
  }

private:
  std::size_t max_line_bytes_;
  std::string buffer_;
};

}  // namespace pepin

#endif  // PEPIN_BASE_CPP__PROTOCOL_HPP_
