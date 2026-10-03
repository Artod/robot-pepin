// The firmware's face model and renderer on this computer, for tests/unit/test_face_parity.py:
// reads a script on stdin and writes what the head would draw, so the simulator (sim/dump.js,
// the same script) can be compared pixel by pixel.
//
//   R p0 .. p15 phase    render this shape: 320x170 RGB565 little-endian on stdout, drawn as
//                        two byte-swapped strips (as the firmware does) and swapped back
//   E id intensity ms t  the model: setExpression at t ms
//   M level t            the model: setMouth at t ms
//   S t                  the model: update to t ms, then print its shape as one text line
//   F type hex           protocol.cpp: the frame of that type around that payload, as hex
//   T hex                protocol.cpp: a 'T' payload parsed, one line per item (or "bad")
//
// Build: clang++ -std=c++17 -O1 -I../src dump.cpp ../src/face_model.cpp ../src/face_render.cpp
//        ../src/protocol.cpp
#include <stdio.h>
#include <string.h>

#include "face_model.h"
#include "face_render.h"
#include "protocol.h"

static size_t unhex(const char* text, uint8_t* out) {
  size_t n = 0;
  if (strcmp(text, "-") == 0) return 0;  // an empty payload
  for (const char* p = text; p[0] && p[1]; p += 2) {
    unsigned v;
    sscanf(p, "%2x", &v);
    out[n++] = (uint8_t)v;
  }
  return n;
}

static char g_hex[2 * (head::kMaxPayload + head::kOverhead) + 2];
static uint8_t g_bytes[head::kMaxPayload + head::kOverhead];
static uint8_t g_frame[head::kMaxPayload + head::kOverhead];
static head::InfoScreen g_info;

using namespace face;

static Layout g_layout;
static uint16_t g_strip_a[kScreenWidth * (kScreenHeight / 2)];
static uint16_t g_strip_b[kScreenWidth * (kScreenHeight - kScreenHeight / 2)];

int main() {
  FaceModel model;
  char op[4];
  while (scanf("%3s", op) == 1) {
    if (strcmp(op, "R") == 0) {
      Shape s;
      for (int i = 0; i < kParamCount; ++i) scanf("%f", &s.p[i]);
      scanf("%f", &s.wave_phase);
      layout(s, &g_layout);
      const int top = kScreenHeight / 2;
      draw(g_layout, Strip{g_strip_a, 0, top, true});
      draw(g_layout, Strip{g_strip_b, top, kScreenHeight - top, true});
      for (int i = 0; i < kScreenWidth * top; ++i) {
        const uint16_t v = g_strip_a[i];
        const uint8_t le[2] = {(uint8_t)(v >> 8), (uint8_t)(v & 0xFF)};  // swapped back
        fwrite(le, 1, 2, stdout);
      }
      for (int i = 0; i < kScreenWidth * (kScreenHeight - top); ++i) {
        const uint16_t v = g_strip_b[i];
        const uint8_t le[2] = {(uint8_t)(v >> 8), (uint8_t)(v & 0xFF)};
        fwrite(le, 1, 2, stdout);
      }
    } else if (strcmp(op, "E") == 0) {
      int id;
      float intensity, ms;
      unsigned t;
      scanf("%d %f %f %u", &id, &intensity, &ms, &t);
      model.setExpression(id, intensity, ms, t);
    } else if (strcmp(op, "M") == 0) {
      float level;
      unsigned t;
      scanf("%f %u", &level, &t);
      model.setMouth(level, t);
    } else if (strcmp(op, "F") == 0) {
      int type;
      scanf("%d %s", &type, g_hex);
      const size_t n = unhex(g_hex, g_bytes);
      const size_t size = head::encode((uint8_t)type, g_bytes, n, g_frame);
      head::Parser parser;
      bool parsed = false;
      for (size_t i = 0; i < size; ++i) parsed = parser.feed(g_frame[i]);
      for (size_t i = 0; i < size; ++i) printf("%02x", g_frame[i]);
      printf(" %s\n", parsed && parser.size() == n ? "ok" : "unparsed");
    } else if (strcmp(op, "T") == 0) {
      scanf("%s", g_hex);
      const size_t n = unhex(g_hex, g_bytes);
      if (!head::parse_info(g_bytes, n, &g_info)) {
        printf("bad\n");
        continue;
      }
      printf("%u %u\n", g_info.duration_ms, g_info.n);
      for (int i = 0; i < g_info.n; ++i) {
        const head::InfoItem& it = g_info.items[i];
        printf("%u|%u|%s|%s\n", it.kind, it.frac, it.key, it.value);
      }
    } else if (strcmp(op, "S") == 0) {
      unsigned t;
      scanf("%u", &t);
      model.update(t);
      const Shape s = model.shape();
      for (int i = 0; i < kParamCount; ++i) printf("%.5f ", s.p[i]);
      printf("%.5f\n", s.wave_phase);
    }
  }
  return 0;
}
