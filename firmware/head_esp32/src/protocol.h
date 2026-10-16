// The head's serial protocol (the head contract), both ways over the CH340 at 921600 baud:
//
//   0xA5 | type (1) | length (2, LE) | payload (length) | CRC-8 (poly 0x07, init 0, over type,
//   length and payload)
//
// ESP32 -> host                                   host -> ESP32
//   'I' IMU samples: n x 17 bytes                   'E' expression: id u8, intensity u8, ms u16
//       micros u32, ax ay az gx gy gz i16, cfg u8       (the mouth and the eyes' default look)
//   'S' status once a second (struct Status)        'M' mouth openness u8 (0..255)
//   'P' pong: ping id u32, micros at receipt u32    'T' info screen (parse_info)
//                                                   'Q' ping: id u32
//                                                   'C' config: id u8, IMU rate u16 (Hz),
//                                                       DLPF u8, accel FS u8, gyro FS u8,
//                                                       brightness u8[, face style u8: 0 pepin,
//                                                       1 rose, 2 manga, 3 anime, 4 kawaii,
//                                                       5 comics[, eyes brightness u8: 0 off
//                                                       .. 255, 64 the default[, eye set u8:
//                                                       0 follow the face style, 1 classic,
//                                                       2 kawaii]]]; a shorter 'C' leaves the
//                                                       later fields as they are
//                                                   'Y' eyes override, 16 or 19 bytes (below)
//
// 'Y' (eyes::decode_override): mode u8 (0 release: the eyes back to the expression's look, the
// rest of the frame but blink ignored; 1 standing; 2 timed by ms), ms u16, brow_pos i8 (-100
// toward the nose .. 100 outward, 0 the top), brow_raise i8 (-100 low .. 100 high), brow_width
// u8, lid_top u8 (0 open .. 254 closed), lid_bottom u8, gaze_x i8 (-100 the robot's left ..
// 100 its right), gaze_y i8 (-100 down .. 100 up), iris r g b u8, glow u8, blink u8 (0 none,
// 1 both eyes, 2 wink left, 3 wink right: acts in every mode), side u8 (0 both eyes, 1 the
// robot's left only, 2 its right only); then, in the 19-byte form, effect u8 (the inner
// ring's effect, eye_table.h's Effect: 0 none = back to the look's own, 1 sparkle, 2 flash,
// 3 spin, 4 pulse, 5 flicker, 6 doze, 7 tear; 0xFF keep) and effect_ms u16 (its life; 0: a one-shot
// effect's own length, a standing one until the next effect byte or a release). A field the
// host leaves to the expression: 0x80 in an i8 field, 0xFF in a u8 field (so a u8 field's
// range is 0..254, 254 = 1.0), 0 0 0 for the iris; a 16-byte 'Y' keeps the effect. A 'Y'
// replaces the override on the eyes it names whole (fields it keeps go back to the
// expression's); the override tweens in and out over the table's override_ms; a timed one
// lapses back to the expression's look; a release (mode 0) ends the host's effect too. Sides
// are the robot's own.
//
// Everything little-endian. src/pepin/head_link.py is the host's side of the same bytes. Pure
// C++ (no Arduino): the native tests share it.
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace head {

constexpr uint8_t kSync = 0xA5;
constexpr size_t kMaxPayload = 1024;
constexpr size_t kOverhead = 5;  // sync, type, length (2), CRC
// 4: the shape engine draws any shapes style (kawaii 4, comics 5 in 'C'); the eyes ('Y', 'C'
// bytes 9-10, the status's eye fields)
constexpr uint8_t kFirmwareVersion = 4;

enum Type : uint8_t {
  kImu = 'I',
  kStatus = 'S',
  kPong = 'P',
  kExpression = 'E',
  kMouth = 'M',
  kInfo = 'T',
  kPing = 'Q',
  kConfig = 'C',
  kEyes = 'Y',
};

constexpr size_t kImuSampleBytes = 17;
constexpr size_t kStatusBytes = 42;
constexpr size_t kConfigBytes = 7;        // without the face's style (an older host)
constexpr size_t kConfigBytesStyled = 8;  // with it
constexpr size_t kConfigBytesEyes = 9;    // with the eyes' brightness
constexpr size_t kConfigBytesEyeSet = 10;  // with the eye look set
constexpr size_t kEyesBytes = 16;         // a 'Y' payload without the effect
constexpr size_t kEyesBytesEffect = 19;   // with effect u8 and effect_ms u16

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
  uint16_t imu_rate_x10;  // the chip's output rate measured in ESP32 micros, 0.1 Hz units
  uint32_t i2c_errors;
  uint32_t dropped;       // IMU samples that did not fit the serial buffer
  uint32_t rx_errors;     // host frames with a bad CRC or length
  uint32_t rx_frames;     // host frames accepted
  uint8_t expression;     // the expression shown
  uint8_t mode;           // 0 face, 1 info screen, 2 asleep (the host is silent)
  uint8_t imu_state;      // 0 absent, 1 data-ready interrupt, 2 polled
  uint8_t who_am_i;
  uint8_t config_id;
  uint8_t version;
  uint16_t free_heap_kb;
  uint32_t gaps;          // IMU stamps more than 1.5 periods apart
  uint16_t eyes_ma;       // the eyes' current estimate after the cap, mA
  uint8_t eyes_brightness;
  uint8_t eyes_mode;      // 0 the expression's look, 1 override standing, 2 timed, 3 self-test
  uint8_t eye_layout;     // eye_table.h's layout id (the chain's order and angles)
  uint8_t eye_set;        // the look set in force: 0 classic, 1 kawaii (eye_table.h EyeSet)
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
