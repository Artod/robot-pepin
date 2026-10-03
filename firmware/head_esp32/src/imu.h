// The head's MPU6050: sampled at a configured rate on its own core-0 task, each sample stamped
// with micros() at its data-ready edge (or at the poll that found it), handed out in batches.
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace head {

struct ImuConfig {
  uint8_t id;        // echoed in every sample: which scale and rate it was taken at
  uint16_t rate_hz;  // the chip's output rate: 1000 / (1 + SMPLRT_DIV), 4..1000
  uint8_t dlpf;      // DLPF_CFG 1..6 (0 and 7 switch the gyro to 8 kHz: refused, 3 used)
  uint8_t accel_fs;  // 0..3: +-2/4/8/16 g
  uint8_t gyro_fs;   // 0..3: +-250/500/1000/2000 deg/s
};

// Config id 0: what the firmware boots with until the host sends 'C'. The head pans at up to
// ~290 deg/s, past the +-250 range, so the gyro starts at +-500; the accel at +-4 g for the
// mast's jerks.
constexpr ImuConfig kImuDefaults = {0, 1000, 3, 1, 1};

struct ImuSample {
  uint32_t t_us;  // micros() at the data-ready edge (polled: at the poll, less half its period)
  int16_t a[3];   // raw counts, the chip's axes
  int16_t g[3];
  uint8_t cfg;
};

// 3: the chip answers but no data-ready edge comes (the INT wire): nothing is streamed.
enum ImuState : uint8_t { kImuAbsent = 0, kImuInterrupt = 1, kImuPolled = 2, kImuNoInterrupt = 3 };

// Polling at twice the rate when the INT wire is missing: off. Polled stamps are a poll period
// off and a timer beats against the chip's own oscillator; a missing wire is reported instead.
#ifndef HEAD_IMU_POLL_FALLBACK
#define HEAD_IMU_POLL_FALLBACK 0
#endif

struct ImuStats {
  uint32_t samples;     // sent to the sink since boot
  uint32_t last_stamp;  // micros of the newest sample sent: the chip's rate in ESP32 time
  uint32_t i2c_errors;  // failed transactions
  uint32_t gaps;        // stamps more than 1.5 periods apart
  uint32_t missed;      // data-ready edges inside those gaps
  uint32_t duplicates;  // polled reads equal to the previous one (HEAD_IMU_POLL_FALLBACK only)
  uint8_t who_am_i;     // 0x68 a genuine MPU6050; this robot's spare answers 0x72
  uint8_t state;        // ImuState
  uint8_t config_id;    // the config in force
};

// Called on the IMU task with a batch of consecutive samples (20 ms of them, at most 20).
using ImuSink = void (*)(const ImuSample* samples, size_t n);

// Start the task (core 0); it finds the chip, retries every second while it is absent.
void imu_start(ImuSink sink);

// A new configuration, applied by the task at its next sample; thread-safe.
void imu_configure(const ImuConfig& config);

ImuStats imu_stats();

}  // namespace head
