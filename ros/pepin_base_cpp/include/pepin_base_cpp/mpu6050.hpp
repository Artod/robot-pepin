// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// An MPU6050 (GY-521) on a Linux i2c-dev bus, in SI units. No ROS, no threads, no
// exceptions: open a bus, read fourteen registers, get a sample; every failure comes
// back as false/nullopt plus a message the caller can log.
//
// Why it exists: wheel odometry over-reports a turn in place by 10-25% on carpet, so
// the yaw rate has to come from a gyro. The board's RAM cannot afford another Python
// process, so the reader lives inside the C++ base bridge and this header is the part
// of it that knows about the chip.
//
// The chip shares /dev/i2c-2 with three VL53L1X ranging sensors (0x30-0x32); it answers
// at 0x68 and each transaction here is a self-contained write-then-read, so a second
// process talking to the ToFs between our transfers is harmless.
//
// Configuration written by init(): gyro +-500 dps (65.5 LSB/dps), accel +-4 g
// (8192 LSB/g), DLPF ~44 Hz (which also fixes the internal sample rate at 1 kHz, from
// which SMPLRT_DIV divides down), clock from the PLL with the X gyro as reference.
//
// THE OUTPUT RATE IS NOT THE READ RATE. A read returns whatever sample the output registers hold,
// and SMPLRT_DIV sets how often they are refreshed, so a chip that updates at our own read rate
// hands us a sample 0..one period old (at 100 Hz: 0-10 ms, and 4 % of the bridge's reads were the
// previous read's sample again, 2026-10-02). The bridge asks for the chip's whole 1 kHz (divider
// 0, `imu_output_rate_hz`): the sample read is at most 1 ms old. The register map says the divider
// only decimates, but this clone's gyro noise falls with it, as averaging over its window would:
// yaw-rate std 1.15 mrad/s at divider 0 and 1, 1.02 at 4, 0.91 at 9, accel unchanged (live A/B on
// the parked cart, 2026-10-02). A quarter more gyro noise is the price of the fresh sample.
//
// THE FILTER DELAYS THE SAMPLE. The DLPF is a fixed group delay the stamp must subtract (the
// bridge's `imu_filter_delay_s`, config/imu.json's timing block). MPU-6050 register map rev 4.2,
// CONFIG (0x1A), by DLPF_CFG (the gyro's output is 8 kHz at 0 and 1 kHz otherwise):
//   DLPF_CFG      0        1       2       3       4       5        6
//   gyro BW       256 Hz   188 Hz  98 Hz   42 Hz   20 Hz   10 Hz    5 Hz
//   gyro delay    0.98 ms  1.9 ms  2.8 ms  4.8 ms  8.3 ms  13.4 ms  18.6 ms
//   accel delay   0 ms     2.0 ms  3.0 ms  4.9 ms  8.5 ms  13.8 ms  19.0 ms
// One stamp carries both; the gyro's delay is the one subtracted (the EKF fuses its yaw rate).
// The chip on this cart answers WHO_AM_I 0x72, a clone; if it follows the MPU-6500's table
// instead, DLPF_CFG 3 is 41 Hz / 5.9 ms.

#ifndef PEPIN_BASE_CPP__MPU6050_HPP_
#define PEPIN_BASE_CPP__MPU6050_HPP_

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <thread>

namespace pepin
{

/// One conversion of the chip, in SI units: x forward, y left, z up on the breakout.
struct ImuSample
{
  double accel_x = 0.0;       ///< m/s^2, gravity included (the chip measures proper acceleration)
  double accel_y = 0.0;
  double accel_z = 0.0;
  double gyro_x = 0.0;        ///< rad/s, right-handed about the same axes
  double gyro_y = 0.0;
  double gyro_z = 0.0;        ///< yaw rate: the number the EKF is here for
  double temperature_c = 0.0; ///< die temperature, useful only as a drift witness
};

/// The MPU6050 registers this driver touches. Names are the datasheet's.
namespace mpu6050_register
{
constexpr std::uint8_t kSampleRateDivider = 0x19;   ///< SMPLRT_DIV
constexpr std::uint8_t kConfig = 0x1A;              ///< CONFIG: the DLPF
constexpr std::uint8_t kGyroConfig = 0x1B;          ///< GYRO_CONFIG: full-scale range
constexpr std::uint8_t kAccelConfig = 0x1C;         ///< ACCEL_CONFIG: full-scale range
constexpr std::uint8_t kAccelXOutHigh = 0x3B;       ///< first of the 14 measurement bytes
constexpr std::uint8_t kPowerManagement1 = 0x6B;    ///< PWR_MGMT_1: sleep bit and clock source
constexpr std::uint8_t kWhoAmI = 0x75;              ///< WHO_AM_I: 0x68, or a clone's own value
}  // namespace mpu6050_register

/// DLPF_CFG written to CONFIG: ~42 Hz gyro bandwidth, 4.8 ms group delay (the table above).
constexpr std::uint8_t kDlpfConfig = 0x03;

/// Blocking reader for one MPU6050 on an i2c-dev bus; not thread-safe, one owner thread.
class Mpu6050
{
public:
  Mpu6050() = default;
  Mpu6050(const Mpu6050 &) = delete;
  Mpu6050 & operator=(const Mpu6050 &) = delete;

  /// Close the bus.
  ~Mpu6050() {close_device();}

  /// Open `device`, claim `address`, identify the chip and configure its output for
  /// `output_rate_hz` (how often the registers we read are refreshed, not how often we read).
  ///
  /// False (with `error` set) if any of that fails; the caller keeps working without a gyro.
  bool open_device(
    const std::string & device, int address, double output_rate_hz, std::string & error)
  {
    close_device();
    fd_ = ::open(device.c_str(), O_RDWR);
    if (fd_ < 0) {
      error = device + ": " + std::strerror(errno);
      return false;
    }
    if (::ioctl(fd_, I2C_SLAVE, static_cast<unsigned long>(address)) < 0) {
      error = "address " + hex_byte(static_cast<std::uint8_t>(address)) + " not claimed: " +
        std::strerror(errno);
      close_device();
      return false;
    }
    return identify(error) && configure(output_rate_hz, error);
  }

  /// One 14-byte burst from ACCEL_XOUT_H as accel, temperature and gyro; nullopt on a bus error.
  std::optional<ImuSample> read_sample(std::string & error)
  {
    std::uint8_t raw[14];
    if (!read_block(mpu6050_register::kAccelXOutHigh, raw, sizeof(raw), error)) {
      return std::nullopt;
    }
    ImuSample sample;
    sample.accel_x = word(raw, 0) * kAccelScale;
    sample.accel_y = word(raw, 2) * kAccelScale;
    sample.accel_z = word(raw, 4) * kAccelScale;
    sample.temperature_c = word(raw, 6) / 340.0 + 36.53;  // the datasheet's own formula
    sample.gyro_x = word(raw, 8) * kGyroScale;
    sample.gyro_y = word(raw, 10) * kGyroScale;
    sample.gyro_z = word(raw, 12) * kGyroScale;
    return sample;
  }

  /// The WHO_AM_I byte read at open time: 0x68 for an MPU6050, 0x70/0x71 for the clones.
  std::uint8_t who_am_i() const {return who_am_i_;}

  /// SMPLRT_DIV as written at open time: the chip's output rate is 1 kHz / (1 + this).
  std::uint8_t output_divider() const {return divider_;}

  /// Whether the bus is open.
  bool is_open() const {return fd_ >= 0;}

  /// Close the bus; safe to call twice.
  void close_device()
  {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

private:
  static constexpr double kGravity = 9.80665;              ///< m/s^2 per g
  static constexpr double kAccelLsbPerG = 8192.0;          ///< +-4 g
  static constexpr double kGyroLsbPerDps = 65.5;           ///< +-500 dps
  static constexpr double kDegToRad = 0.017453292519943295;
  static constexpr double kAccelScale = kGravity / kAccelLsbPerG;
  static constexpr double kGyroScale = kDegToRad / kGyroLsbPerDps;
  static constexpr double kInternalRateHz = 1000.0;        ///< with the DLPF on, per the datasheet

  /// Read WHO_AM_I and accept the MPU6050 and its pin-compatible clones.
  bool identify(std::string & error)
  {
    std::uint8_t value = 0;
    if (!read_block(mpu6050_register::kWhoAmI, &value, 1, error)) {
      error = "WHO_AM_I unreadable: " + error;
      close_device();
      return false;
    }
    who_am_i_ = value;
    // 0x68 MPU6050, 0x70 MPU6500, 0x71 MPU9250, 0x72 the GY-521 clone on this cart (2026-09-07)
    // — the same register map for what we use.
    if (value != 0x68 && value != 0x70 && value != 0x71 && value != 0x72) {
      error = "WHO_AM_I is " + hex_byte(value) + ", not an MPU6050";
      close_device();
      return false;
    }
    return true;
  }

  /// Wake the chip and write the output rate, the filter and both full-scale ranges.
  bool configure(double output_rate_hz, std::string & error)
  {
    // PLL with the X gyro as the reference: steadier than the internal oscillator.
    if (!write_register(mpu6050_register::kPowerManagement1, 0x01, error)) {
      close_device();
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));  // the PLL needs a moment
    divider_ = sample_rate_divider(output_rate_hz);
    const bool written =
      write_register(mpu6050_register::kSampleRateDivider, divider_, error) &&
      write_register(mpu6050_register::kConfig, kDlpfConfig, error) &&  // DLPF ~42 Hz
      write_register(mpu6050_register::kGyroConfig, 0x08, error) &&    // +-500 dps
      write_register(mpu6050_register::kAccelConfig, 0x08, error);     // +-4 g
    if (!written) {
      close_device();
      return false;
    }
    return true;
  }

  /// SMPLRT_DIV for `rate_hz` off the 1 kHz internal rate, clamped to the register's range.
  static std::uint8_t sample_rate_divider(double rate_hz)
  {
    if (!(rate_hz > 0.0)) {
      return 19;  // 50 Hz: a nonsense rate must not divide by zero
    }
    const double divider = kInternalRateHz / rate_hz - 1.0;
    if (divider <= 0.0) {
      return 0;
    }
    if (divider >= 255.0) {
      return 255;
    }
    return static_cast<std::uint8_t>(divider + 0.5);
  }

  /// Write one register; false with `error` on a bus failure.
  bool write_register(std::uint8_t reg, std::uint8_t value, std::string & error)
  {
    const std::uint8_t frame[2] = {reg, value};
    if (!write_all(frame, sizeof(frame))) {
      error = "write to " + hex_byte(reg) + " failed: " + std::strerror(errno);
      return false;
    }
    return true;
  }

  /// Set the register pointer, then read `size` bytes from it; false with `error` on failure.
  bool read_block(std::uint8_t reg, std::uint8_t * out, std::size_t size, std::string & error)
  {
    if (fd_ < 0) {
      error = "bus not open";
      return false;
    }
    if (!write_all(&reg, 1)) {
      error = "seek to " + hex_byte(reg) + " failed: " + std::strerror(errno);
      return false;
    }
    ssize_t received = 0;
    do {
      received = ::read(fd_, out, size);
    } while (received < 0 && errno == EINTR);
    if (received != static_cast<ssize_t>(size)) {
      error = "read of " + std::to_string(size) + " bytes at " + hex_byte(reg) + " failed: " +
        (received < 0 ? std::strerror(errno) : "short transfer");
      return false;
    }
    return true;
  }

  /// Push a whole i2c-dev frame out, retrying only an interrupted call.
  bool write_all(const std::uint8_t * data, std::size_t size)
  {
    if (fd_ < 0) {
      errno = EBADF;
      return false;
    }
    ssize_t written = 0;
    do {
      written = ::write(fd_, data, size);
    } while (written < 0 && errno == EINTR);
    return written == static_cast<ssize_t>(size);
  }

  /// Two measurement bytes as the chip sends them: big-endian, signed.
  static double word(const std::uint8_t * raw, std::size_t offset)
  {
    const std::uint16_t bits =
      static_cast<std::uint16_t>(raw[offset] << 8 | raw[offset + 1]);
    return static_cast<std::int16_t>(bits);
  }

  /// A byte as 0x.. for log lines.
  static std::string hex_byte(std::uint8_t value)
  {
    static const char * digits = "0123456789abcdef";
    return std::string("0x") + digits[value >> 4] + digits[value & 0x0F];
  }

  int fd_ = -1;
  std::uint8_t who_am_i_ = 0;
  std::uint8_t divider_ = 0;
};

}  // namespace pepin

#endif  // PEPIN_BASE_CPP__MPU6050_HPP_
