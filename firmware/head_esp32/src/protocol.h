// The head's serial protocol (the head contract), both ways over the CH340 at 921600 baud:
//
//   0xA5 | type (1) | length (2, LE) | payload (length) | CRC-8 (poly 0x07, init 0, over type,
//   length and payload)
//
// ESP32 -> host                                   host -> ESP32
//   'I' IMU samples: n x 17 bytes                   'E' expression: id u8, intensity u8, ms u16
//       micros u32, ax ay az gx gy gz i16, cfg u8   'M' mouth openness u8 (0..255)
//   'S' status once a second (struct Status)        'T' info screen (parse_info)
//   'P' pong: ping id u32, micros at receipt u32    'Q' ping: id u32
//                                                   'C' config: id u8, IMU rate u16 (Hz),
//                                                       DLPF u8, accel FS u8, gyro FS u8,
//                                                       brightness u8
// Everything little-endian. src/pepin/head_link.py is the host's side of the same bytes. Pure
// C++ (no Arduino): the native tests share it.
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace head {

constexpr uint8_t kSync = 0xA5;
constexpr size_t kMaxPayload = 1024;
constexpr size_t kOverhead = 5;  // sync, type, length (2), CRC
constexpr uint8_t kFirmwareVersion = 1;

enum Type : uint8_t {
  kImu = 'I',
  kStatus = 'S',
  kPong = 'P',
  kExpression = 'E',
  kMouth = 'M',
  kInfo = 'T',
  kPing = 'Q',
  kConfig = 'C',
};

constexpr size_t kImuSampleBytes = 17;
constexpr size_t kStatusBytes = 36;
constexpr size_t kConfigBytes = 7;

uint8_t crc8(const uint8_t* data, size_t n, uint8_t crc = 0);

// One frame into `out` (room for n + kOverhead bytes); its size.
size_t encode(uint8_t type, const uint8_t* payload, size_t n, uint8_t* out);

// Frames out of a byte stream; a bad CRC or an impossible length costs that frame only.
class Parser {
 public:
  // One byte in; true when it completed a valid frame (type(), payload(), size() until the next).
  bool feed(uint8_t b);
  uint8_t type() const { return type_; }
  const uint8_t* payload() const { return buf_; }
  size_t size() const { return len_; }
  uint32_t crc_errors = 0;
  uint32_t length_errors = 0;
  uint32_t frames = 0;

 private:
  enum State : uint8_t { kWaitSync, kType, kLen0, kLen1, kBody, kCrc };
  State state_ = kWaitSync;
  uint8_t type_ = 0;
  size_t len_ = 0;
  size_t got_ = 0;
  uint8_t buf_[kMaxPayload];
};

// The status line's fields, in wire order.
struct Status {
  uint32_t micros;
  uint16_t fps_x10;       // face frames per second x 10
  uint16_t imu_rate_hz;   // IMU samples sent in the last second
  uint32_t i2c_errors;
  uint32_t dropped;       // IMU samples that did not fit the serial buffer, or were missed
  uint32_t rx_errors;     // host frames with a bad CRC or length
  uint32_t rx_frames;     // host frames accepted
  uint8_t expression;     // the expression shown
  uint8_t mode;           // 0 face, 1 info screen, 2 asleep (the host is silent)
  uint8_t imu_state;      // 0 absent, 1 data-ready interrupt, 2 polled
  uint8_t who_am_i;
  uint8_t config_id;
  uint8_t version;
  uint16_t free_heap_kb;
  uint32_t duplicates;    // polled reads that repeated the previous sample (dropped)
};

size_t pack_status(const Status& s, uint8_t* out);  // kStatusBytes

// An info screen: a few lines of text, key/value rows and bars, shown for duration_ms.
constexpr int kInfoItemsMax = 8;
constexpr int kInfoTextMax = 48;  // bytes of UTF-8, the terminating zero included

enum InfoKind : uint8_t { kText = 0, kKeyValue = 1, kBar = 2 };

struct InfoItem {
  uint8_t kind;
  uint8_t frac;  // a bar's fill, 0..255
  char key[kInfoTextMax];
  char value[kInfoTextMax];
};

struct InfoScreen {
  uint16_t duration_ms;  // 0: back to the face now
  uint8_t n;
  InfoItem items[kInfoItemsMax];
};

// 'T': duration u16, count u8, then per item: kind u8 and
//   text  len u8 + UTF-8;   key/value  len u8 + key, len u8 + value;
//   bar   len u8 + label, frac u8, len u8 + value text.
// False for a malformed payload (out is then unspecified).
bool parse_info(const uint8_t* p, size_t n, InfoScreen* out);

inline uint16_t get_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
inline uint32_t get_u32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline uint8_t* put_u16(uint8_t* p, uint16_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  return p + 2;
}
inline uint8_t* put_u32(uint8_t* p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i));
  return p + 4;
}

}  // namespace head
