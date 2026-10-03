// The firmware's face model and renderer on this computer, for tests/unit/test_face_parity.py:
// reads a script on stdin and writes what the head would draw, so the simulator (sim/dump.js,
// the same script) can be compared pixel by pixel.
//
//   R p0 .. p15 phase    render this shape: 320x170 RGB565 little-endian on stdout, drawn as
//                        two byte-swapped strips (as the firmware does) and swapped back
//   E id intensity ms t  the model: setExpression at t ms
//   M level t            the model: setMouth at t ms
//   S t                  the model: update to t ms, then print its shape as one text line
//
// Build: clang++ -std=c++17 -O1 -I../src dump.cpp ../src/face_model.cpp ../src/face_render.cpp
#include <stdio.h>
#include <string.h>

#include "face_model.h"
#include "face_render.h"

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
