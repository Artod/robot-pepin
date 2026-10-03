#include "face_render.h"

#include <math.h>

namespace face {

namespace {

constexpr float kTwoPi = 6.28318530718f;
constexpr float kBiteGapPx = 1.0f;  // half the dark line between the two rows of teeth
// The lip curves are sampled every half pixel; a disc of the stroke's radius is swept along
// them, so ends are round and steep parts as thick as flat ones.
constexpr int kSamplesMax = 2 * kScreenWidth + 3;
constexpr int kDiscMax = 2 * 8 + 2;  // half-pixel offsets within the largest radius (7 px)

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// A triangle wave in phase with sin(2 pi ph): 0 at 0, 1 at 1/4, 0 at 1/2, -1 at 3/4.
float triangle(float ph) {
  const float f = ph - floorf(ph);
  if (f < 0.25f) return 4.0f * f;
  if (f < 0.75f) return 2.0f - 4.0f * f;
  return 4.0f * f - 4.0f;
}

struct Geometry {
  float cx, cy, half_w, r, open_px, upper, pexp, smile_px, asym_px, wave_px, wave_n, sharp, phase;

  // The mouth's line at u (-1 left corner .. 1 right corner, as seen).
  float mid(float u) const {
    const float u2 = u * u;
    const float taper = 1.0f - u2 * u2 * u2;
    const float ph = (u + 1.0f) * 0.5f * wave_n - phase;
    const float wave = (1.0f - sharp) * sinf(kTwoPi * ph) + sharp * triangle(ph);
    return cy - smile_px * u2 - asym_px * u + wave_px * taper * wave;
  }

  // The opening's profile: 1 in the middle, 0 at the corners.
  float profile(float u) const {
    const float a = fabsf(u);
    if (a >= 1.0f) return 0.0f;
    return powf(1.0f - powf(a, pexp), 1.0f / pexp);
  }
};

// The curves, sampled: x, the upper lip's line and the lower lip's line.
struct Samples {
  int n;
  float x[kSamplesMax];
  float top[kSamplesMax];
  float bot[kSamplesMax];
};

Samples g_samples;  // static: 7.7 KB that a task's stack need not carry

inline uint16_t load(const Strip& s, int i) {
  const uint16_t v = s.px[i];
  return s.swapped ? (uint16_t)((v >> 8) | (v << 8)) : v;
}

inline void store(const Strip& s, int i, uint16_t c) {
  s.px[i] = s.swapped ? (uint16_t)((c >> 8) | (c << 8)) : c;
}

// `c` over the pixel at weight w/32.
inline uint16_t mix(uint16_t dst, uint16_t c, int w) {
  const int r = (((dst >> 11) & 31) * (32 - w) + ((c >> 11) & 31) * w) >> 5;
  const int g = (((dst >> 5) & 63) * (32 - w) + ((c >> 5) & 63) * w) >> 5;
  const int b = ((dst & 31) * (32 - w) + (c & 31) * w) >> 5;
  return (uint16_t)((r << 11) | (g << 5) | b);
}

void blend(const Strip& s, int x, int y, float coverage, uint16_t c) {
  const int w = (int)(coverage * 32.0f + 0.5f);
  if (w <= 0) return;
  const int i = (y - s.y0) * kScreenWidth + x;
  store(s, i, w >= 32 ? c : mix(load(s, i), c, w));
}

// [ya, yb) of column x in colour c, its end pixels blended by how much of them it covers.
void span(const Strip& s, int x, float ya, float yb, uint16_t c) {
  ya = fmaxf(ya, (float)s.y0);
  yb = fminf(yb, (float)(s.y0 + s.rows));
  if (yb <= ya) return;
  const int ia = (int)floorf(ya);
  const int ib = (int)floorf(yb);
  if (ia == ib) {
    blend(s, x, ia, yb - ya, c);
    return;
  }
  blend(s, x, ia, (float)(ia + 1) - ya, c);
  for (int y = ia + 1; y < ib; ++y) store(s, (y - s.y0) * kScreenWidth + x, c);
  if (yb > (float)ib && ib < s.y0 + s.rows) blend(s, x, ib, yb - (float)ib, c);
}

}  // namespace

void layout(const Shape& shape, Layout* out) {
  const float* p = shape.p;
  const float w = (float)kScreenWidth;
  const float h = (float)kScreenHeight;
  Geometry g;
  g.half_w = 0.5f * w * clampf(p[kWidth], 0.02f, 1.0f);
  g.r = 0.5f * (kThickMinPx + (kThickMaxPx - kThickMinPx) * clampf(p[kThick], 0.0f, 1.0f));
  g.r = clampf(g.r, 0.5f, 0.5f * (float)(kDiscMax - 2));
  float room = 0.5f * w - g.half_w - g.r - 2.0f;  // how far the mouth may move and stay whole
  if (room < 0.0f) room = 0.0f;
  g.cx = 0.5f * w + clampf(p[kShiftX] * kShiftXPx, -room, room);
  g.cy = 0.5f * h + p[kShiftY] * kShiftYPx;
  g.open_px = clampf(p[kOpen], 0.0f, 1.0f) * kMaxOpenPx;
  g.upper = clampf(p[kUpper], 0.0f, 1.0f);
  g.pexp = clampf(p[kCornerExp], 1.0f, 12.0f);
  g.smile_px = p[kSmile] * kSmilePx;
  g.asym_px = p[kAsym] * kAsymPx;
  g.wave_px = clampf(p[kWave], 0.0f, 1.0f) * kWavePx;
  g.wave_n = p[kWaveN];
  g.sharp = clampf(p[kWaveSharp], 0.0f, 1.0f);
  g.phase = shape.wave_phase;
  const float teeth = clampf(p[kTeeth], 0.0f, 1.0f);

  // The curves every half pixel, and exactly at both corners.
  Samples& sm = g_samples;
  const float left = g.cx - g.half_w;
  const float right = g.cx + g.half_w;
  sm.n = 0;
  const int k0 = (int)ceilf(left * 2.0f);
  const int k1 = (int)floorf(right * 2.0f);
  for (int k = k0 - 1; k <= k1 + 1 && sm.n < kSamplesMax; ++k) {
    float xs = 0.5f * (float)k;
    if (k == k0 - 1) xs = left;
    if (k == k1 + 1) xs = right;
    const float u = clampf((xs - g.cx) / g.half_w, -1.0f, 1.0f);
    const float m = g.mid(u);
    const float o = g.open_px * g.profile(u);
    sm.x[sm.n] = xs;
    sm.top[sm.n] = m - o * g.upper;
    sm.bot[sm.n] = m + o * (1.0f - g.upper);
    ++sm.n;
  }

  int x0 = (int)floorf(left - g.r);
  int x1 = (int)ceilf(right + g.r);
  x0 = x0 < 0 ? 0 : x0;
  x1 = x1 > kScreenWidth - 1 ? kScreenWidth - 1 : x1;
  out->x_first = x0;
  out->x_last = x1;
  const float r2 = g.r * g.r;
  // A regular sample sits a whole number of half pixels from a column's centre: its disc's
  // half-height comes from this table; the two corners' samples are computed.
  float disc[kDiscMax];
  const int mmax = (int)(2.0f * g.r);
  for (int m = 0; m <= mmax; ++m) disc[m] = sqrtf(fmaxf(r2 - 0.25f * (float)(m * m), 0.0f));
  int first = 0;  // the first sample that may still be within reach of a column
  for (int x = 0; x < kScreenWidth; ++x) {
    Column& c = out->col[x];
    c.active = false;
    c.tooth_gap = false;
    c.teeth_up = c.teeth_lo = 0.0f;
    if (x < x0 || x > x1) continue;
    const float xc = (float)x + 0.5f;
    float ot = 1e9f, it = -1e9f, ob = -1e9f, ib = 1e9f;
    while (first < sm.n && sm.x[first] < xc - g.r) ++first;
    for (int i = first; i < sm.n && sm.x[i] <= xc + g.r; ++i) {
      const float dx = sm.x[i] - xc;
      float hh;
      if (i == 0 || i == sm.n - 1) {
        hh = sqrtf(fmaxf(r2 - dx * dx, 0.0f));
      } else {
        const int m = (int)(fabsf(dx) * 2.0f + 0.5f);
        hh = m <= mmax ? disc[m] : 0.0f;
      }
      ot = fminf(ot, sm.top[i] - hh);
      it = fmaxf(it, sm.top[i] + hh);
      ob = fmaxf(ob, sm.bot[i] + hh);
      ib = fminf(ib, sm.bot[i] - hh);
      c.active = true;
    }
    if (!c.active) continue;
    c.outer_top = ot;
    c.outer_bot = ob;
    c.inner_top = it;
    c.inner_bot = ib;
    const float cavity = ib - it;
    if (cavity > 0.0f && teeth > 0.0f) {
      float up = clampf(2.0f * teeth, 0.0f, 1.0f) * 0.5f * cavity;
      float lo = clampf(2.0f * teeth - 1.0f, 0.0f, 1.0f) * 0.5f * cavity;
      if (lo > 0.0f) {
        up = fmaxf(0.0f, fminf(up, 0.5f * cavity - kBiteGapPx));
        lo = fmaxf(0.0f, fminf(lo, 0.5f * cavity - kBiteGapPx));
      }
      c.teeth_up = up;
      c.teeth_lo = lo;
      const float m = fmodf(fabsf(xc - g.cx) + 0.5f * kToothPx, kToothPx);
      c.tooth_gap = m < kToothGapPx;
    }
  }
}

void draw(const Layout& l, Strip s) {
  const uint16_t bg = s.swapped ? (uint16_t)((kColorBackground >> 8) | (kColorBackground << 8))
                                : kColorBackground;
  const int n = s.rows * kScreenWidth;
  for (int i = 0; i < n; ++i) s.px[i] = bg;
  for (int x = l.x_first; x <= l.x_last; ++x) {
    const Column& c = l.col[x];
    if (!c.active) continue;
    span(s, x, c.outer_top, c.outer_bot, kColorLip);
    if (c.inner_bot <= c.inner_top) continue;
    span(s, x, c.inner_top, c.inner_bot, kColorCavity);
    if (c.tooth_gap) continue;
    if (c.teeth_up > 0.0f) span(s, x, c.inner_top, c.inner_top + c.teeth_up, kColorTeeth);
    if (c.teeth_lo > 0.0f) span(s, x, c.inner_bot - c.teeth_lo, c.inner_bot, kColorTeeth);
  }
}

}  // namespace face
