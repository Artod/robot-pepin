#include "protocol.h"

#include <string.h>

namespace head {

uint8_t crc8(const uint8_t* data, size_t n, uint8_t crc) {
  for (size_t i = 0; i < n; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
  }
  return crc;
}

size_t encode(uint8_t type, const uint8_t* payload, size_t n, uint8_t* out) {
  out[0] = kSync;
  out[1] = type;
  out[2] = (uint8_t)n;
  out[3] = (uint8_t)(n >> 8);
  if (n) memcpy(out + 4, payload, n);
  out[4 + n] = crc8(out + 1, n + 3);
  return n + kOverhead;
}

bool Parser::feed(uint8_t b) {
  switch (state_) {
    case kWaitSync:
      if (b == kSync) state_ = kType;
      return false;
    case kType:
      type_ = b;
      state_ = kLen0;
      return false;
    case kLen0:
      len_ = b;
      state_ = kLen1;
      return false;
    case kLen1:
      len_ |= (size_t)b << 8;
      if (len_ > kMaxPayload) {
        ++length_errors;
        state_ = b == kSync ? kType : kWaitSync;
        return false;
      }
      got_ = 0;
      state_ = len_ ? kBody : kCrc;
      return false;
    case kBody:
      buf_[got_++] = b;
      if (got_ == len_) state_ = kCrc;
      return false;
    case kCrc: {
      uint8_t head[3] = {type_, (uint8_t)len_, (uint8_t)(len_ >> 8)};
      const uint8_t crc = crc8(buf_, len_, crc8(head, 3));
      state_ = kWaitSync;
      if (crc != b) {
        ++crc_errors;
        return false;
      }
      ++frames;
      return true;
    }
  }
  return false;
}

size_t pack_status(const Status& s, uint8_t* out) {
  uint8_t* p = out;
  p = put_u32(p, s.micros);
  p = put_u16(p, s.fps_x10);
  p = put_u16(p, s.imu_rate_hz);
  p = put_u32(p, s.i2c_errors);
  p = put_u32(p, s.dropped);
  p = put_u32(p, s.rx_errors);
  p = put_u32(p, s.rx_frames);
  *p++ = s.expression;
  *p++ = s.mode;
  *p++ = s.imu_state;
  *p++ = s.who_am_i;
  *p++ = s.config_id;
  *p++ = s.version;
  p = put_u16(p, s.free_heap_kb);
  p = put_u32(p, s.duplicates);
  return (size_t)(p - out);
}

namespace {

// `len` bytes of UTF-8 into `dst`, cut on a character boundary to fit, zero-terminated.
void copy_text(char* dst, const uint8_t* src, size_t len) {
  size_t n = len < (size_t)kInfoTextMax - 1 ? len : (size_t)kInfoTextMax - 1;
  while (n > 0 && n < len && (src[n] & 0xC0) == 0x80) --n;  // not inside a character
  memcpy(dst, src, n);
  dst[n] = 0;
}

// A length-prefixed string at p[*at]; false when it runs past the payload.
bool take_text(const uint8_t* p, size_t n, size_t* at, char* dst) {
  if (*at >= n) return false;
  const size_t len = p[(*at)++];
  if (*at + len > n) return false;
  copy_text(dst, p + *at, len);
  *at += len;
  return true;
}

}  // namespace

bool parse_info(const uint8_t* p, size_t n, InfoScreen* out) {
  if (n < 3) return false;
  out->duration_ms = get_u16(p);
  const int count = p[2];
  size_t at = 3;
  out->n = 0;
  for (int i = 0; i < count; ++i) {
    if (at >= n) return false;
    InfoItem item;
    item.kind = p[at++];
    item.frac = 0;
    item.key[0] = item.value[0] = 0;
    if (item.kind == kText) {
      if (!take_text(p, n, &at, item.value)) return false;
    } else if (item.kind == kKeyValue) {
      if (!take_text(p, n, &at, item.key) || !take_text(p, n, &at, item.value)) return false;
    } else if (item.kind == kBar) {
      if (!take_text(p, n, &at, item.key) || at >= n) return false;
      item.frac = p[at++];
      if (!take_text(p, n, &at, item.value)) return false;
    } else {
      return false;
    }
    if (out->n < kInfoItemsMax) out->items[out->n++] = item;
  }
  return at == n;
}

}  // namespace head
