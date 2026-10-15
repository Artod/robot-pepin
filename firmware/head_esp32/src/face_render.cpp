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
constexpr float kBowHalf = 0.45f;    // Cupid's bow: its dip spans |u| < this
constexpr float kDimpleArc = 0.95f;  // a fold's half angle at dimples 1, radians (54 deg)
constexpr int kDimplePoints = 7;
constexpr int kHookPoints = 5;        // a corner hook's arc, from the line's heading to 95 deg
constexpr float kHookArc = 1.658f;    // radians
constexpr float kBulbDip = 0.6f;      // the dumbbell's middle, this much thinner than its ends
constexpr int kStrokeMax = 16;       // a stroke's points
// The pucker's lobes, sampled every 30 deg from the top round the right to the bottom
constexpr int kLobeSteps = 6;
constexpr float kLobeCos[kLobeSteps + 1] = {0.0f, 0.5f, 0.8660254f, 1.0f, 0.8660254f, 0.5f, 0.0f};
constexpr float kLobeSin[kLobeSteps + 1] = {-1.0f, -0.8660254f, -0.5f, 0.0f,
                                            0.5f,  0.8660254f,  1.0f};
// The heart in units of its size: two lobes of radius 0.5 at (+-0.45, 0), the tip at (0, 1.2),
// the sides tangent to the lobes at (+-0.8126, 0.3443); an inset moves the sides in along their
// normal (0.725, 0.689) and the tip up by 1.45 (1/sin of its half angle).
constexpr float kLobeX = 0.45f;
constexpr float kLobeR = 0.5f;
constexpr float kTipY = 1.2f;
constexpr float kSideX = 0.8126f;
constexpr float kSideY = 0.3443f;
constexpr float kSideNx = 0.725f;
constexpr float kSideNy = 0.689f;
constexpr float kTipInset = 1.45f;
constexpr float kHeartHeight = kTipY + kLobeR;  // 1.7 of its size
constexpr float kDroolStemPx = 1.2f;            // the drool's stem's radius
constexpr float kDroolDropPx = 3.6f;            // its drop's radius at 1

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
  float bow_px;
  float bulb;  // the dumbbell profile's share

  // The mouth's line at u (-1 left corner .. 1 right corner, as seen).
  float mid(float u) const {
    const float u2 = u * u;
    const float line = cy - smile_px * u2 - asym_px * u;
    if (wave_px == 0.0f) return line;  // no sinf where there is no wave (the ESP32's is slow)
    const float taper = 1.0f - u2 * u2 * u2;
    const float ph = (u + 1.0f) * 0.5f * wave_n - phase;
    const float wave = (1.0f - sharp) * sinf(kTwoPi * ph) + sharp * triangle(ph);
    return line + wave_px * taper * wave;
  }

  // The opening's profile: 1 in the middle, 0 at the corners. An ellipse (exponent 2, most
  // expressions) is a square root; anything else costs two powf. With bulb, a dumbbell takes
  // over: round ends and a thinner middle.
  float profile(float u) const {
    const float a = fabsf(u);
    if (a >= 1.0f) return 0.0f;
    float p;
    if (pexp == 2.0f) {
      p = sqrtf(1.0f - a * a);
    } else {
      p = powf(1.0f - powf(a, pexp), 1.0f / pexp);
    }
    if (bulb <= 0.0f) return p;
    // the ends: discs whose radius is the half opening, as a share of the half width
    const float rho = fminf(0.5f, open_px / (2.0f * half_w));
    float bone;
    if (a > 1.0f - rho) {
      const float d = (a - (1.0f - rho)) / rho;
      bone = sqrtf(fmaxf(1.0f - d * d, 0.0f));
    } else {
      const float t = a / (1.0f - rho);
      bone = 1.0f - kBulbDip * (1.0f - t * t);
    }
    return p + (bone - p) * bulb;
  }

  // Cupid's bow: how far the upper lip dips at u, 1 at the middle, 0 from |u| = kBowHalf.
  static float bow(float u) {
    const float a = u / kBowHalf;
    if (a <= -1.0f || a >= 1.0f) return 0.0f;
    const float k = 1.0f - a * a;
    return k * k;
  }

  // The lower lip's line at u (the mouth's line when closed).
  float lower(float u) const {
    const float o = open_px > 0.0f ? open_px * profile(u) : 0.0f;
    return mid(u) + o * (1.0f - upper);
  }

  // The line's outward unit direction at a corner (u = -1 or 1; the wave tapers out there).
  void tangent(float u, float* tx, float* ty) const {
    const float slope = (-2.0f * smile_px * u - asym_px) / half_w;
    float x = u, y = u * slope;
    const float n = sqrtf(x * x + y * y);
    *tx = x / n;
    *ty = y / n;
  }
};

// The curves, sampled: x, the upper lip's line and the lower lip's line.
struct Samples {
  int n;
  float x[kSamplesMax];
  float top[kSamplesMax];
  float bot[kSamplesMax];
  float open[kSamplesMax];  // the opening there (the fit to the screen scales it)
};

Samples g_samples;  // static: 7.7 KB that a task's stack need not carry

void push(Layout* out, int x, float top, float bot, uint8_t element, uint8_t alpha) {
  if (x < 0 || x >= kScreenWidth || !(bot > top) || alpha == 0 || out->n_deco >= kDecoMax) return;
  Deco& d = out->deco[out->n_deco++];
  d.x = (int16_t)x;
  d.top = top;
  d.bot = bot;
  d.element = element;
  d.alpha = alpha;
}

uint8_t opacity(float v) { return (uint8_t)(clampf(v, 0.0f, 1.0f) * 32.0f + 0.5f); }

// The y-interval of column xc inside a capsule (the segment a-b swept by a disc of radius r):
// the end discs' and the band's, joined (a capsule is convex); false when the column misses it.
bool capsule(float ax, float ay, float bx, float by, float r, float xc, float* top, float* bot) {
  float lo = 1e9f, hi = -1e9f;
  const float ex[2] = {ax, bx};
  const float ey[2] = {ay, by};
  for (int k = 0; k < 2; ++k) {
    const float d = xc - ex[k];
    if (fabsf(d) <= r) {
      const float h = sqrtf(fmaxf(r * r - d * d, 0.0f));
      lo = fminf(lo, ey[k] - h);
      hi = fmaxf(hi, ey[k] + h);
    }
  }
  const float dx = bx - ax;
  const float dy = by - ay;
  const float len2 = dx * dx + dy * dy;
  if (len2 > 0.0f) {
    const float len = sqrtf(len2);
    const float ux = xc - ax;
    float ya = -1e9f, yb = 1e9f;
    // along the segment: 0 <= ux dx + (y - ay) dy <= len2
    const float along = ux * dx;
    if (fabsf(dy) > 1e-6f) {
      float y0 = ay + (0.0f - along) / dy;
      float y1 = ay + (len2 - along) / dy;
      if (y0 > y1) {
        const float t = y0;
        y0 = y1;
        y1 = t;
      }
      ya = fmaxf(ya, y0);
      yb = fminf(yb, y1);
    } else if (along < 0.0f || along > len2) {
      yb = ya;
    }
    // across it: |ux dy - (y - ay) dx| <= r len
    const float across = ux * dy;
    if (fabsf(dx) > 1e-6f) {
      float y0 = ay + (across - r * len) / dx;
      float y1 = ay + (across + r * len) / dx;
      if (y0 > y1) {
        const float t = y0;
        y0 = y1;
        y1 = t;
      }
      ya = fmaxf(ya, y0);
      yb = fminf(yb, y1);
    } else if (fabsf(across) > r * len) {
      yb = ya;
    }
    if (yb > ya) {
      lo = fminf(lo, ya);
      hi = fmaxf(hi, yb);
    }
  }
  if (!(hi > lo)) return false;
  *top = lo;
  *bot = hi;
  return true;
}

// A polyline of n points (at most kStrokeMax) swept by a disc of radius r: in each column the
// intervals of its segments, sorted and merged, each pushed as a span.
void stroke(const float* xs, const float* ys, int n, float r, uint8_t element, uint8_t alpha,
            Layout* out) {
  if (n < 2) return;
  float x_lo = xs[0], x_hi = xs[0];
  for (int i = 1; i < n; ++i) {
    x_lo = fminf(x_lo, xs[i]);
    x_hi = fmaxf(x_hi, xs[i]);
  }
  int x0 = (int)floorf(x_lo - r);
  int x1 = (int)ceilf(x_hi + r);
  x0 = x0 < 0 ? 0 : x0;
  x1 = x1 > kScreenWidth - 1 ? kScreenWidth - 1 : x1;
  for (int x = x0; x <= x1; ++x) {
    const float xc = (float)x + 0.5f;
    float lo[kStrokeMax], hi[kStrokeMax];
    int m = 0;
    for (int i = 0; i + 1 < n; ++i) {
      float a, b;
      if (!capsule(xs[i], ys[i], xs[i + 1], ys[i + 1], r, xc, &a, &b)) continue;
      int j = m++;
      while (j > 0 && lo[j - 1] > a) {  // insertion by the top
        lo[j] = lo[j - 1];
        hi[j] = hi[j - 1];
        --j;
      }
      lo[j] = a;
      hi[j] = b;
    }
    if (m == 0) continue;
    float a = lo[0], b = hi[0];
    for (int j = 1; j < m; ++j) {
      if (lo[j] <= b) {
        b = fmaxf(b, hi[j]);
      } else {
        push(out, x, a, b, element, alpha);
        a = lo[j];
        b = hi[j];
      }
    }
    push(out, x, a, b, element, alpha);
  }
}

// The blush: a soft pink ellipse on each side, three bright strokes across it (///).
void cheeks(float amount, Layout* out) {
  if (amount <= 0.0f) return;
  const uint8_t fill = opacity(0.45f * amount);
  const uint8_t line = opacity(amount);
  const float cy = 0.5f * (float)kScreenHeight - kCheekYPx;
  for (int side = -1; side <= 1; side += 2) {
    const float cx = 0.5f * (float)kScreenWidth + (float)side * kCheekXPx;
    const int x0 = (int)floorf(cx - kCheekRxPx);
    const int x1 = (int)ceilf(cx + kCheekRxPx);
    for (int x = x0; x <= x1; ++x) {
      const float d = ((float)x + 0.5f - cx) / kCheekRxPx;
      if (d <= -1.0f || d >= 1.0f) continue;
      const float h = kCheekRyPx * sqrtf(1.0f - d * d);
      push(out, x, cy - h, cy + h, kElCheek, fill);
    }
    for (int k = -1; k <= 1; ++k) {
      const float sx = cx + (float)k * 0.42f * kCheekRxPx;
      const float xs[2] = {sx - 0.22f * kCheekRxPx, sx + 0.22f * kCheekRxPx};
      const float ys[2] = {cy + 0.6f * kCheekRyPx, cy - 0.6f * kCheekRyPx};
      stroke(xs, ys, 2, 1.4f, kElCheek, line, out);
    }
  }
}

// The tongue out over the lower lip: a band with a round tip from `where` along the mouth, the
// tip leaning sideways by `lean` of its length; in each column it starts under the upper lip (the
// cavity's upper part when the mouth is open, the line's middle when it is closed), and a faint
// crease runs down its middle. Shortened to keep its tip on the screen. A style with an outline
// draws the band in the lip's colour first and the tongue inset within it; a style with a shine
// adds a lighter patch high on it (the plump anime tongue).
void tongue(const Geometry& g, const StyleSpec& st, float amount, float where, float lean,
            Layout* out) {
  if (amount <= 0.0f) return;
  const float line = st.deco_line_px;
  const float u = clampf(where, -1.0f, 1.0f);
  const float tx = g.cx + u * g.half_w;
  const float root = g.lower(u);
  const float len = fminf(amount * kTonguePx, (float)kScreenHeight - kMarginPx - root);
  if (len <= 0.0f) return;
  const float wr = fminf(0.5f * st.tongue_w_px, len);  // the band's half width, the tip's radius
  const float reach = len - wr;                    // the tip's centre below the root
  const float bx = tx + lean * reach;
  const float by = root + reach;
  const int x0 = (int)floorf(fminf(tx, bx) - wr);
  const int x1 = (int)ceilf(fmaxf(tx, bx) + wr);
  for (int pass = line > 0.0f ? 0 : 1; pass < 2; ++pass) {
    const float rr = pass == 0 || line <= 0.0f ? wr : fmaxf(wr - line, 0.5f);
    const uint8_t element = pass == 0 ? kElLip : kElTongue;
    for (int x = x0; x <= x1; ++x) {
      float top, bot;
      if (!capsule(tx, root, bx, by, rr, (float)x + 0.5f, &top, &bot)) continue;
      float under = root;  // under the upper lip
      if (x >= out->x_first && x <= out->x_last && out->col[x].active) {
        const Column& c = out->col[x];
        under = c.inner_bot > c.inner_top ? c.inner_top + 0.4f * (c.inner_bot - c.inner_top)
                                          : 0.5f * (c.outer_top + c.outer_bot);
      }
      push(out, x, fmaxf(top, under), bot, element, 32);
    }
  }
  if (st.tongue_shine > 0.0f && reach > 0.0f) {  // the highlight: an ellipse high on the tongue
    const float sr = st.tongue_shine * (wr - line);
    const float sx = tx + lean * 0.35f * reach - 0.25f * wr;
    const float sy = root + 0.4f * reach;
    const int sx0 = (int)floorf(sx - sr);
    const int sx1 = (int)ceilf(sx + sr);
    for (int x = sx0; x <= sx1; ++x) {
      const float d = ((float)x + 0.5f - sx) / sr;
      if (d <= -1.0f || d >= 1.0f) continue;
      const float h = 0.6f * sr * sqrtf(1.0f - d * d);
      push(out, x, fmaxf(sy - h, root), sy + h, kElShade, 32);
    }
  }
  const float y0 = root + g.r + 1.0f;
  const float y1 = root + 0.6f * len;
  if (y1 > y0) {
    const float xs[2] = {tx + lean * (y0 - root), tx + lean * (y1 - root)};
    const float ys[2] = {y0, y1};
    stroke(xs, ys, 2, 0.8f, kElCavity, 10, out);
  }
}

// The tongue patch inside an open mouth: a lighter dome rising from the bottom of the opening,
// `amount` of the opening's height at the dome's peak (at kStyles' patch_x along the mouth),
// its top one upward curve from corner to corner (the guides' tongue line); a style with an
// outline rims the curve in the lip's colour.
void patch(const Geometry& g, const StyleSpec& st, float amount, Layout* out) {
  if (amount <= 0.0f || g.open_px <= 0.0f) return;
  const float span = 1.0f + fabsf(st.patch_x);
  const float rise = amount * g.open_px;
  for (int x = out->x_first; x <= out->x_last; ++x) {
    const Column& c = out->col[x];
    if (!c.active || c.inner_bot <= c.inner_top) continue;
    const float u = ((float)x + 0.5f - g.cx) / g.half_w;
    const float d = (u - st.patch_x) / span;
    const float dome = sqrtf(fmaxf(1.0f - d * d, 0.0f));
    const float floor_y = c.inner_top + c.teeth_up + (c.teeth_up > 0.0f ? st.teeth_line_px : 0.0f);
    const float top = fmaxf(c.inner_bot - rise * dome, floor_y);
    push(out, x, top, c.inner_bot - c.teeth_lo, kElTongue, 32);
    if (st.deco_line_px > 0.0f) {
      push(out, x, fmaxf(top - st.deco_line_px, floor_y), top, kElLip, 32);
    }
  }
}

// The fangs: two pointed teeth hanging from the upper lip (its inner edge when the mouth is
// open, the teeth band's lower edge when there is one, the middle of the line when it is
// closed) at `at` of the half width either side; `side_of` leaves one out (-1 the left one
// only, as seen; 1 the right one only).
void fangs(const Geometry& g, const StyleSpec& st, float amount, float at, float side_of,
           Layout* out) {
  if (amount <= 0.0f) return;
  for (int side = -1; side <= 1; side += 2) {
    const float a = amount * fminf(1.0f, 1.0f + (float)side * side_of);
    if (a <= 0.0f) continue;
    const float len = a * kFangPx;
    const float fx = g.cx + (float)side * at * g.half_w;
    const int x0 = (int)floorf(fx - kFangWPx);
    const int x1 = (int)ceilf(fx + kFangWPx);
    for (int x = x0; x <= x1; ++x) {
      if (x < out->x_first || x > out->x_last) continue;
      const Column& c = out->col[x];
      if (!c.active) continue;
      const float d = fabsf((float)x + 0.5f - fx) / kFangWPx;
      if (d >= 1.0f) continue;
      const bool open = c.inner_bot > c.inner_top;
      float root = open ? c.inner_top : 0.5f * (c.outer_top + c.outer_bot);
      if (open && c.teeth_up > 0.0f && !c.tooth_gap) root += c.teeth_up + st.teeth_line_px;
      push(out, x, root - 1.0f, root + len * (1.0f - d), kElTeeth, 32);
    }
  }
}

// The cheek folds: an arc round each corner, centred on it and clear of the line's end,
// concave towards the mouth: (‿).
void dimples(const Geometry& g, float amount, float left, float right, Layout* out) {
  if (amount <= 0.0f) return;
  const float half = amount * kDimpleArc;
  const float r = fmaxf(1.0f, 0.8f * g.r);
  const float rad = kDimplePx + g.r;
  for (int side = -1; side <= 1; side += 2) {
    const float u = (float)side;
    const float ex = side < 0 ? left : right;
    const float ey = g.mid(u);
    float tx, ty;
    g.tangent(u, &tx, &ty);
    float xs[kDimplePoints], ys[kDimplePoints];
    for (int k = 0; k < kDimplePoints; ++k) {
      const float phi = -half + 2.0f * half * (float)k / (float)(kDimplePoints - 1);
      const float c = cosf(phi), s = sinf(phi);
      xs[k] = ex + rad * (c * tx - s * ty);
      ys[k] = ey + rad * (c * ty + s * tx);
    }
    stroke(xs, ys, kDimplePoints, r, kElLip, 32, out);
  }
}

// The corner hooks: at each corner the line goes on along its heading and curls down (amount
// > 0) or up (< 0) through 95 deg, an arc of kHookPx * |amount| radius in the line's stroke.
void hooks(const Geometry& g, float amount, float left, float right, Layout* out) {
  if (amount == 0.0f) return;
  const float rad = kHookPx * fabsf(amount);
  for (int side = -1; side <= 1; side += 2) {
    const float u = (float)side;
    const float ex = side < 0 ? left : right;
    const float ey = g.mid(u);
    float tx, ty;
    g.tangent(u, &tx, &ty);
    float nx = -ty, ny = tx;  // the normal, pointing down the screen
    if (ny < 0.0f) {
      nx = -nx;
      ny = -ny;
    }
    if (amount < 0.0f) {
      nx = -nx;
      ny = -ny;
    }
    const float ccx = ex + rad * nx;  // the arc's centre
    const float ccy = ey + rad * ny;
    float xs[kHookPoints], ys[kHookPoints];
    for (int k = 0; k < kHookPoints; ++k) {
      const float th = kHookArc * (float)k / (float)(kHookPoints - 1);
      const float c = cosf(th), s = sinf(th);
      xs[k] = ccx - rad * c * nx + rad * s * tx;
      ys[k] = ccy - rad * c * ny + rad * s * ty;
    }
    stroke(xs, ys, kHookPoints, g.r, kElLip, 32, out);
  }
}

// A rounded block: columns [cx - hw, cx + hw], rows [top, bot), corners of radius rc.
void block(float cx, float top, float bot, float hw, float rc, uint8_t element, Layout* out) {
  const int x0 = (int)floorf(cx - hw);
  const int x1 = (int)ceilf(cx + hw);
  for (int x = x0; x <= x1; ++x) {
    const float d = fabsf((float)x + 0.5f - cx);
    if (d >= hw) continue;
    float inset = 0.0f;
    if (d > hw - rc) {
      const float e = (d - (hw - rc)) / rc;
      inset = rc * (1.0f - sqrtf(fmaxf(1.0f - e * e, 0.0f)));
    }
    push(out, x, top + inset, bot - inset, element, 32);
  }
}

// The moustache: a short rounded block above the upper lip, a rim in the line's colour round
// it (a black moustache on the black screen), bristles cut into its lower edge.
void moustache(const Geometry& g, float amount, Layout* out) {
  if (amount <= 0.0f) return;
  const float h = amount * kMoustacheHPx;
  const float hw = 0.5f * kMoustacheWPx;
  const float rim = kMoustacheRimPx;
  const float lip = g.mid(0.0f) - g.open_px * g.upper - g.r;  // the upper lip's top
  const float bot = lip - kMoustacheGapPx - rim;
  const float top = bot - h;
  const float rc = fminf(3.0f, 0.5f * h);
  if (rim > 0.0f) block(g.cx, top - rim, bot + rim, hw + rim, rc + rim, kElLip, out);
  block(g.cx, top, bot, hw, rc, kElMoustache, out);
  const int x0 = (int)floorf(g.cx - hw);
  const int x1 = (int)ceilf(g.cx + hw);
  for (int x = x0; x <= x1; ++x) {
    const float d = fabsf((float)x + 0.5f - g.cx);
    if (d >= hw - 2.0f || fmodf(d + 2.5f, 5.0f) >= 1.0f) continue;
    push(out, x, bot - 0.35f * h, bot + rim + 1.0f, kDecoBackground, 32);
  }
}

// The kiss: the lips pursed into a "3", two lobes bulging right with a point between them, as
// tall as kPuckerPx at 1 (the mouth's line shrinks into it as it grows). Its stroke is
// `line_share` of a lobe's radius, or the mouth's own line when that is 0.
void pucker(const Geometry& g, float amount, float width, float line_share, Layout* out) {
  if (amount <= 0.0f) return;
  // a lobe's radius; a mouth narrower than 0.2 scales the "3" down (a small "ε")
  const float rl = 0.25f * kPuckerPx * amount * fminf(1.0f, width / 0.2f);
  const float cx = g.cx - 0.3f * rl;
  const float cy = g.mid(0.0f);
  float xs[2 * kLobeSteps + 1], ys[2 * kLobeSteps + 1];
  for (int k = 0; k <= kLobeSteps; ++k) {
    xs[k] = cx + rl * kLobeCos[k];
    ys[k] = cy - rl + rl * kLobeSin[k];
    xs[kLobeSteps + k] = cx + rl * kLobeCos[k];
    ys[kLobeSteps + k] = cy + rl + rl * kLobeSin[k];
  }
  xs[kLobeSteps] = cx - 0.25f * rl;  // the point between the lobes
  ys[kLobeSteps] = cy;
  const float r = line_share > 0.0f ? fmaxf(1.0f, line_share * rl) : g.r;
  stroke(xs, ys, 2 * kLobeSteps + 1, r, kElLip, 32, out);
}

// A heart of size s (its lobes' centres at (x0 +- 0.45 s, y0)), shrunk inwards by `inset`
// pixels, as one span a column.
void heart_shape(float x0, float y0, float s, float inset, uint8_t element, uint8_t alpha,
                 Layout* out) {
  const float lr = kLobeR * s - inset;
  if (lr <= 0.0f) return;
  const float sx = kSideX * s - kSideNx * inset;
  const float sy = y0 + kSideY * s - kSideNy * inset;
  const float px[3] = {x0 - sx, x0 + sx, x0};
  const float py[3] = {sy, sy, y0 + kTipY * s - kTipInset * inset};
  const int xa = (int)floorf(x0 - kLobeX * s - lr);
  const int xb = (int)ceilf(x0 + kLobeX * s + lr);
  for (int x = xa; x <= xb; ++x) {
    const float xc = (float)x + 0.5f;
    float lo = 1e9f, hi = -1e9f;
    for (int side = -1; side <= 1; side += 2) {
      const float d = xc - (x0 + (float)side * kLobeX * s);
      if (fabsf(d) < lr) {
        const float h = sqrtf(lr * lr - d * d);
        lo = fminf(lo, y0 - h);
        hi = fmaxf(hi, y0 + h);
      }
    }
    for (int e = 0; e < 3; ++e) {  // the triangle's edges
      const int f = (e + 1) % 3;
      const float xa_ = fminf(px[e], px[f]), xb_ = fmaxf(px[e], px[f]);
      if (xc < xa_ || xc > xb_ || px[e] == px[f]) continue;
      const float y = py[e] + (xc - px[e]) * (py[f] - py[e]) / (px[f] - px[e]);
      lo = fminf(lo, y);
      hi = fmaxf(hi, y);
    }
    push(out, x, lo, hi, element, alpha);
  }
}

// The heart-shaped lips: a filled heart kHeartLipsPx * amount tall on the mouth's line, its
// outline the line's stroke in the lip's colour, the fill inside it.
void heart_lips(const Geometry& g, float amount, Layout* out) {
  if (amount <= 0.0f) return;
  const float s = amount * kHeartLipsPx / kHeartHeight;
  const float y0 = g.mid(0.0f) - 0.35f * s;
  heart_shape(g.cx, y0, s, 0.0f, kElLip, 32, out);
  heart_shape(g.cx, y0, s, 2.0f * g.r, kElCavity, 32, out);
}

// The drool: a thin stem down from the lower lip at kDroolAt, a round drop at its end, in the
// shade's colour; a style with an outline rims both in the lip's colour.
void drool(const Geometry& g, float amount, float line, Layout* out) {
  if (amount <= 0.0f) return;
  const float u = kDroolAt;
  const float x = g.cx + u * g.half_w;
  const float root = g.lower(u) + 0.5f * g.r;
  const float len = fminf(amount * kDroolPx, (float)kScreenHeight - kMarginPx - root);
  if (len <= 0.0f) return;
  const float dr = fminf(kDroolDropPx * amount, 0.5f * len);  // the drop's radius
  const float stem = kDroolStemPx;
  const float dy = root + len - dr;  // the drop's centre
  for (int pass = line > 0.0f ? 0 : 1; pass < 2; ++pass) {
    const float grow = pass == 0 ? 0.0f : -line;
    const uint8_t element = pass == 0 ? kElLip : kElShade;
    const float rs = fmaxf(stem + grow, 0.4f);
    const float rd = fmaxf(dr + grow, 0.5f);
    const int x0 = (int)floorf(x - rd);
    const int x1 = (int)ceilf(x + rd);
    for (int xx = x0; xx <= x1; ++xx) {
      const float xc = (float)xx + 0.5f;
      float lo = 1e9f, hi = -1e9f;
      float top, bot;
      if (capsule(x, root, x, dy, rs, xc, &top, &bot)) {
        lo = fminf(lo, top);
        hi = fmaxf(hi, bot);
      }
      const float d = xc - x;
      if (fabsf(d) < rd) {
        const float h = sqrtf(rd * rd - d * d);
        lo = fminf(lo, dy - h);
        hi = fmaxf(hi, dy + h);
      }
      if (hi > lo) push(out, xx, lo, hi, element, 32);
    }
  }
}

// The sleeper's "z" at flight k (0..1): a glyph of three strokes from beside the mouth's right
// edge up and out, swaying, growing as it rises (the small z, the big Z), never past the
// screen's margin; it fades over the last 40 %.
void z_glyph(const Geometry& g, float k, float edge, Layout* out) {
  if (k < 0.0f) return;
  const float s = kZGlyphPx * (0.55f + 0.45f * k);
  const float rise = 1.0f - (1.0f - k) * (1.0f - k);
  float x0 = edge + 6.0f + 0.5f * s + 10.0f * k + 4.0f * sinf(kTwoPi * 1.2f * k);
  float y0 = g.mid(1.0f) - 0.5f * s - 6.0f - kZGlyphRisePx * rise;
  x0 = fminf(x0, (float)kScreenWidth - kMarginPx - 0.5f * s - g.r);  // on the screen
  y0 = fmaxf(y0, kMarginPx + 0.5f * s + g.r);
  const uint8_t alpha = opacity(k < 0.6f ? 1.0f : 1.0f - (k - 0.6f) / 0.4f);
  const float h = 0.5f * s;
  const float xs[4] = {x0 - h, x0 + h, x0 - h, x0 + h};
  const float ys[4] = {y0 - h, y0 - h, y0 + h, y0 + h};
  stroke(xs, ys, 4, fmaxf(1.0f, 0.7f * g.r), kElLip, alpha, out);
}

// The kiss's heart at flight k (0..1): from beside the mouth's right edge up and out, swaying,
// never past the screen's margin; it grows in at the start and fades over the last 45 %.
void heart(const Geometry& g, float k, float edge, Layout* out) {
  if (k < 0.0f) return;
  const float s = kHeartPx * (0.55f + 0.45f * ease(k * 5.0f));
  const float rise = 1.0f - (1.0f - k) * (1.0f - k);
  float x0 = edge + 4.0f + 0.6f * kHeartPx + 12.0f * k + 5.0f * sinf(kTwoPi * 1.5f * k);
  float y0 = g.mid(1.0f) - 0.6f * kHeartPx - kHeartRisePx * rise;
  x0 = fminf(x0, (float)kScreenWidth - kMarginPx - (kLobeX + kLobeR) * s);  // on the screen
  y0 = fmaxf(y0, kMarginPx + kLobeR * s);
  const uint8_t alpha = opacity(k < 0.55f ? 1.0f : 1.0f - (k - 0.55f) / 0.45f);
  heart_shape(x0, y0, s, 0.0f, kElHeart, alpha, out);
}

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

void blend(const Strip& s, int x, int y, float coverage, uint16_t c, int alpha) {
  const int w = (int)(coverage * (float)alpha + 0.5f);
  if (w <= 0) return;
  const int i = (y - s.y0) * kScreenWidth + x;
  store(s, i, w >= 32 ? c : mix(load(s, i), c, w));
}

// [ya, yb) of column x in colour c at an opacity of alpha/32, its end pixels blended by how
// much of them it covers.
void span(const Strip& s, int x, float ya, float yb, uint16_t c, int alpha = 32) {
  ya = fmaxf(ya, (float)s.y0);
  yb = fminf(yb, (float)(s.y0 + s.rows));
  if (yb <= ya) return;
  const int ia = (int)floorf(ya);
  const int ib = (int)floorf(yb);
  if (ia == ib) {
    blend(s, x, ia, yb - ya, c, alpha);
    return;
  }
  blend(s, x, ia, (float)(ia + 1) - ya, c, alpha);
  if (alpha >= 32) {
    for (int y = ia + 1; y < ib; ++y) store(s, (y - s.y0) * kScreenWidth + x, c);
  } else {
    for (int y = ia + 1; y < ib; ++y) {
      const int i = (y - s.y0) * kScreenWidth + x;
      store(s, i, mix(load(s, i), c, alpha));
    }
  }
  if (yb > (float)ib && ib < s.y0 + s.rows) blend(s, x, ib, yb - (float)ib, c, alpha);
}

}  // namespace

void layout(const Shape& shape, Layout* out) {
  const float* p = shape.p;
  const int style = shape.style < kStyleCount ? shape.style : kStyleDefault;
  const StyleSpec& st = kStyles[style];
  const float w = (float)kScreenWidth;
  const float h = (float)kScreenHeight;
  Geometry g;
  g.half_w = 0.5f * w * clampf(p[kWidth], 0.02f, 1.0f);
  const float pucker_k = clampf(p[kPucker], 0.0f, 1.0f);  // the line shrinks into the pucker
  const float lips_k = clampf(p[kHeartLips], 0.0f, 1.0f);  // and into the heart
  const float shrink = (1.0f - pucker_k) * (1.0f - lips_k);
  if (shrink < 1.0f) g.half_w = fmaxf(g.half_w * shrink, 1.0f);
  g.r = 0.5f * (st.thick_min_px + (st.thick_max_px - st.thick_min_px) * clampf(p[kThick], 0.0f, 1.0f));
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
  g.bow_px = clampf(p[kBow], 0.0f, 1.0f) * kBowPx;
  g.bulb = clampf(p[kBulb], 0.0f, 1.0f);
  const float teeth = clampf(p[kTeeth], 0.0f, 1.0f);

  // The curves every half pixel, and exactly at both corners.
  Samples& sm = g_samples;
  const float left = g.cx - g.half_w;
  const float right = g.cx + g.half_w;
  sm.n = 0;
  const int k0 = (int)ceilf(left * 2.0f);
  const int k1 = (int)floorf(right * 2.0f);
  for (int k = k0 - 1; k <= k1 + 1 && sm.n < kSamplesMax && shrink > 0.0f; ++k) {
    float xs = 0.5f * (float)k;
    if (k == k0 - 1) xs = left;
    if (k == k1 + 1) xs = right;
    const float u = clampf((xs - g.cx) / g.half_w, -1.0f, 1.0f);
    const float m = g.mid(u);
    const float o = g.open_px > 0.0f ? g.open_px * g.profile(u) : 0.0f;
    sm.x[sm.n] = xs;
    sm.top[sm.n] = m - o * g.upper;
    if (g.bow_px > 0.0f) sm.top[sm.n] += g.bow_px * Geometry::bow(u);
    sm.bot[sm.n] = m + o * (1.0f - g.upper);
    sm.open[sm.n] = o;
    ++sm.n;
  }

  // The mouth kept on the screen with a margin: an opening that would reach past it (speech on
  // a wide-open smile) is scaled down to fit; the line itself stays where it is.
  float fit = 1.0f;
  const float y_lo = kMarginPx + g.r;
  const float y_hi = h - kMarginPx - g.r;
  for (int i = 0; i < sm.n; ++i) {
    const float below = sm.open[i] * (1.0f - g.upper);
    const float above = sm.open[i] * g.upper;
    if (sm.bot[i] > y_hi && below > 0.0f) fit = fminf(fit, (y_hi - (sm.bot[i] - below)) / below);
    if (sm.top[i] < y_lo && above > 0.0f) fit = fminf(fit, (sm.top[i] + above - y_lo) / above);
  }
  if (fit < 1.0f) {
    fit = fmaxf(fit, 0.0f);
    for (int i = 0; i < sm.n; ++i) {
      const float cut = sm.open[i] * (1.0f - fit);
      sm.top[i] += cut * g.upper;
      sm.bot[i] -= cut * (1.0f - g.upper);
    }
    g.open_px *= fit;  // what the decorations take the opening to be
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
      if (fabsf(xc - g.cx) > st.teeth_span * g.half_w) up = lo = 0.0f;  // the band's reach
      c.teeth_up = up;
      c.teeth_lo = lo;
      const float m = fmodf(fabsf(xc - g.cx) + 0.5f * kToothPx, kToothPx);
      c.tooth_gap = m < st.tooth_gap_px;
    }
  }

  for (int e = 0; e < kElementCount; ++e) out->color[e] = shape.color[e];
  out->background = st.background;
  out->style = (uint8_t)style;
  out->n_deco = 0;
  cheeks(clampf(p[kCheeks], 0.0f, 1.0f), out);
  out->n_under = out->n_deco;
  patch(g, st, clampf(p[kPatch], 0.0f, 1.0f), out);
  tongue(g, st, clampf(p[kTongue], 0.0f, 1.0f), p[kTongueX], p[kTongueLean], out);
  drool(g, clampf(p[kDrool], 0.0f, 1.0f), st.deco_line_px, out);
  fangs(g, st, clampf(p[kFangs], 0.0f, 1.0f), clampf(p[kFangAt], 0.0f, 1.0f),
        clampf(p[kFangSide], -1.0f, 1.0f), out);
  dimples(g, clampf(p[kDimples], 0.0f, 1.0f), left, right, out);
  hooks(g, clampf(p[kHooks], -1.0f, 1.0f) * shrink, left, right, out);
  moustache(g, clampf(p[kMoustache], 0.0f, 1.0f), out);
  pucker(g, pucker_k, clampf(p[kWidth], 0.02f, 1.0f), st.pucker_line, out);
  heart_lips(g, lips_k, out);
  const float lobe = 0.25f * kPuckerPx * pucker_k;  // the pucker's reach: 1.2 lobes right
  const float heart_edge = g.cx + 0.95f * lips_k * kHeartLipsPx / kHeartHeight;  // the heart's
  heart(g, shape.heart, fmaxf(fmaxf(right, g.cx + 1.2f * lobe), heart_edge), out);
  z_glyph(g, shape.z, fmaxf(right, g.cx + 1.2f * lobe), out);
}

void draw(const Layout& l, Strip s) {
  const StyleSpec& st = kStyles[l.style < kStyleCount ? l.style : kStyleDefault];
  const uint16_t bg = s.swapped ? (uint16_t)((l.background >> 8) | (l.background << 8))
                                : l.background;
  const int n = s.rows * kScreenWidth;
  for (int i = 0; i < n; ++i) s.px[i] = bg;
  for (int i = 0; i < l.n_under; ++i) {
    const Deco& d = l.deco[i];
    span(s, d.x, d.top, d.bot, l.color[d.element], d.alpha);
  }
  const uint16_t lip = l.color[kElLip];
  const uint16_t cavity = l.color[kElCavity];
  const uint16_t teeth = l.color[kElTeeth];
  const uint16_t shade = l.color[kElShade];
  for (int x = l.x_first; x <= l.x_last; ++x) {
    const Column& c = l.col[x];
    if (!c.active) continue;
    span(s, x, c.outer_top, c.outer_bot, lip);
    if (c.inner_bot <= c.inner_top) continue;
    if (st.shade > 0.0f) {  // the shade band at the bottom of the opening
      const float split = c.inner_bot - st.shade * (c.inner_bot - c.inner_top);
      span(s, x, c.inner_top, split, cavity);
      span(s, x, split, c.inner_bot, shade);
    } else {
      span(s, x, c.inner_top, c.inner_bot, cavity);
    }
    if (c.tooth_gap) continue;
    if (c.teeth_up > 0.0f) {
      const float edge = c.inner_top + c.teeth_up;
      span(s, x, c.inner_top, edge, teeth);
      if (st.teeth_line_px > 0.0f) span(s, x, edge, fminf(edge + st.teeth_line_px, c.inner_bot), lip);
    }
    if (c.teeth_lo > 0.0f) {
      const float edge = c.inner_bot - c.teeth_lo;
      span(s, x, edge, c.inner_bot, teeth);
      if (st.teeth_line_px > 0.0f) span(s, x, fmaxf(edge - st.teeth_line_px, c.inner_top), edge, lip);
    }
  }
  for (int i = l.n_under; i < l.n_deco; ++i) {
    const Deco& d = l.deco[i];
    const uint16_t c = d.element == kDecoBackground ? l.background : l.color[d.element];
    span(s, d.x, d.top, d.bot, c, d.alpha);
  }
}

}  // namespace face
