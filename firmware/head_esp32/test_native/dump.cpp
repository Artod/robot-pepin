// The firmware's face model and renderer on this computer, for tests/unit/test_face_parity.py:
// reads a script on stdin and writes what the head would draw, so the simulator (sim/dump.js,
// the same script) can be compared pixel by pixel.
//
//   R p.. phase heart z c..  render this shape (kParamCount parameters, the wave's phase, the
//                        heart's and the "z"'s flight or -1, kElementCount RGB565 colours) in
//                        the style set by the last Y: 320x170 RGB565 little-endian on stdout,
//                        drawn as two byte-swapped strips (as the firmware does) and swapped back
//   Y style              the style (a Style index) for the R lines and the model from here on
//                        (the model: as a 'C' frame at t = 0 would set it)
//   E id intensity ms t  the model: setExpression at t ms
//   M level t            the model: setMouth at t ms
//   C style t            the model: setStyle at t ms (the 'C' frame's style byte)
//   S t                  the model: update to t ms, then print its shape as one text line
//                        (parameters, phase, heart, z, colours, the style, the expression
//                        shown, the one asked for)
//   D t                  the model: update to t ms, then draw the frame the head would show
//                        (the parametric renderer, or the shape engine in a shapes style):
//                        320x170 RGB565 little-endian on stdout, as R writes it
//   F type hex           protocol.cpp: the frame of that type around that payload, as hex
//   T hex                protocol.cpp: a 'T' payload parsed, one line per item (or "bad")
//
// Build: clang++ -std=c++17 -O1 -I../src dump.cpp ../src/face_model.cpp ../src/face_render.cpp
//        ../src/face_anime.cpp ../src/protocol.cpp
#include <stdio.h>
#include <string.h>

#include "face_anime.h"
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
static AnimeModel g_anime;
static uint16_t g_strip_a[kScreenWidth * (kScreenHeight / 2)];
static uint16_t g_strip_b[kScreenWidth * (kScreenHeight - kScreenHeight / 2)];

// The two strips to stdout, swapped back to little-endian RGB565.
static void write_strips() {
  const int top = kScreenHeight / 2;
  for (int i = 0; i < kScreenWidth * top; ++i) {
    const uint16_t v = g_strip_a[i];
    const uint8_t le[2] = {(uint8_t)(v >> 8), (uint8_t)(v & 0xFF)};
    fwrite(le, 1, 2, stdout);
  }
  for (int i = 0; i < kScreenWidth * (kScreenHeight - top); ++i) {
    const uint16_t v = g_strip_b[i];
    const uint8_t le[2] = {(uint8_t)(v >> 8), (uint8_t)(v & 0xFF)};
    fwrite(le, 1, 2, stdout);
  }
}

int main() {
  FaceModel model;
  int style = kStyleDefault;
  char op[4];
  while (scanf("%3s", op) == 1) {
    if (strcmp(op, "R") == 0) {
      Shape s;
      for (int i = 0; i < kParamCount; ++i) scanf("%f", &s.p[i]);
      scanf("%f %f %f", &s.wave_phase, &s.heart, &s.z);
      for (int e = 0; e < kElementCount; ++e) {
        unsigned c;
        scanf("%u", &c);
        s.color[e] = (uint16_t)c;
      }
      s.style = (uint8_t)style;
      layout(s, &g_layout);
      const int top = kScreenHeight / 2;
      draw(g_layout, Strip{g_strip_a, 0, top, true});
      draw(g_layout, Strip{g_strip_b, top, kScreenHeight - top, true});
      write_strips();
    } else if (strcmp(op, "D") == 0) {
      unsigned t;
      scanf("%u", &t);
      model.update(t);
      const int top = kScreenHeight / 2;
      if (shape_table_of(model.style()) != nullptr) {
        g_anime.follow(model, t);
        draw_anime(g_anime.frame(), Strip{g_strip_a, 0, top, true});
        draw_anime(g_anime.frame(), Strip{g_strip_b, top, kScreenHeight - top, true});
      } else {
        layout(model.shape(), &g_layout);
        draw(g_layout, Strip{g_strip_a, 0, top, true});
        draw(g_layout, Strip{g_strip_b, top, kScreenHeight - top, true});
      }
      write_strips();
    } else if (strcmp(op, "Y") == 0) {
      scanf("%d", &style);
      model.setStyle(style, 0);
    } else if (strcmp(op, "C") == 0) {
      int s;
      unsigned t;
      scanf("%d %u", &s, &t);
      model.setStyle(s, t);
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
      printf("%.5f %.5f %.5f", s.wave_phase, s.heart, s.z);
      for (int e = 0; e < kElementCount; ++e) printf(" %u", (unsigned)s.color[e]);
      printf(" %d %d %d\n", (int)s.style, model.shown(), model.expression());
    }
  }
  return 0;
}
