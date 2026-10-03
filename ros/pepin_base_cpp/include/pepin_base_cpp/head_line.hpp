// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// head_server's JSON lines (TCP 3340) into head_imu.hpp's samples: the one place the bridge reads
// that wire. The twin of pepin.head_imu's parse_*; the subscribe request the bridge sends on
// every (re)connection is kSubscribeLine.

#ifndef PEPIN_BASE_CPP__HEAD_LINE_HPP_
#define PEPIN_BASE_CPP__HEAD_LINE_HPP_

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

#include "pepin_base_cpp/head_imu.hpp"

namespace pepin
{

/// What a client sends head_server to receive the IMU stream.
inline const char * const kSubscribeLine = "{\"cmd\":\"subscribe\",\"imu\":true}\n";

/// One `imu` line: its config id, the samples that parsed, and how many rows did not.
struct HeadBatch
{
  int cfg = -1;
  std::vector<HeadSample> samples;
  long refused = 0;
};

/// head_server's clock map as its status line reports it (the minute line's spread and skew).
struct HeadClock
{
  bool ready = false;
  double spread_ms = 0.0;
  double esp_fast_ppm = 0.0;
  double min_rtt_ms = -1.0;  ///< -1 while no ping has come back
};

/// A JSON number as a double; nothing for anything else (a boolean is not a number here).
inline std::optional<double> json_number(const nlohmann::json & value)
{
  if (!value.is_number()) {
    return std::nullopt;
  }
  return value.get<double>();
}

/// `message[key]` as a number, or nothing when absent or not one.
inline std::optional<double> json_field(const nlohmann::json & message, const char * key)
{
  return message.contains(key) ? json_number(message[key]) : std::nullopt;
}

/// The type of a line, or "" for anything that is not an object with a string `type`.
inline std::string line_type(const nlohmann::json & message)
{
  if (!message.is_object() || !message.contains("type") || !message["type"].is_string()) {
    return std::string();
  }
  return message["type"].get<std::string>();
}

/// An `imu_config` line; nothing for any other line or one that cannot date samples.
inline std::optional<HeadConfig> parse_head_config(const nlohmann::json & message)
{
  if (line_type(message) != "imu_config") {
    return std::nullopt;
  }
  const auto cfg = json_field(message, "cfg");
  const auto rate = json_field(message, "rate_hz");
  const auto delay = json_field(message, "filter_delay_s");
  if (!cfg || !rate || !delay) {
    return std::nullopt;
  }
  HeadConfig config{static_cast<int>(*cfg), *rate, *delay};
  if (!config_valid(config)) {
    return std::nullopt;
  }
  return config;
}

/// An `imu` line as a batch; nothing for any other line. A row that is not eight finite numbers
/// is refused and counted.
inline std::optional<HeadBatch> parse_head_imu(const nlohmann::json & message)
{
  if (line_type(message) != "imu" || !message.contains("samples") ||
    !message["samples"].is_array())
  {
    return std::nullopt;
  }
  HeadBatch batch;
  const auto cfg = json_field(message, "cfg");
  batch.cfg = cfg ? static_cast<int>(*cfg) : -1;
  for (const auto & row : message["samples"]) {
    double values[8];
    bool ok = row.is_array() && row.size() == 8;
    for (std::size_t i = 0; ok && i < 8; ++i) {
      const auto value = json_number(row[i]);
      ok = value.has_value();
      values[i] = value.value_or(0.0);
    }
    const auto sample = ok ? sample_from_row(values, batch.cfg) : std::nullopt;
    if (sample.has_value()) {
      batch.samples.push_back(*sample);
    } else {
      ++batch.refused;
    }
  }
  return batch;
}

/// A `status` line's clock map; nothing for any other line.
inline std::optional<HeadClock> parse_head_status(const nlohmann::json & message)
{
  if (line_type(message) != "status") {
    return std::nullopt;
  }
  HeadClock clock;
  if (!message.contains("clock") || !message["clock"].is_object()) {
    return clock;
  }
  const auto & map = message["clock"];
  clock.ready = map.contains("ready") && map["ready"].is_boolean() && map["ready"].get<bool>();
  clock.spread_ms = json_field(map, "spread_ms").value_or(0.0);
  clock.esp_fast_ppm = json_field(map, "esp_fast_ppm").value_or(0.0);
  clock.min_rtt_ms = json_field(map, "min_rtt_ms").value_or(-1.0);
  return clock;
}

}  // namespace pepin

#endif  // PEPIN_BASE_CPP__HEAD_LINE_HPP_
