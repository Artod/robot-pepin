#include "face_anime.h"

#include <math.h>
#include <string.h>

namespace face {

namespace {

constexpr float kTwoPi = 6.28318530718f;
constexpr float kPi = 3.14159265359f;
constexpr float kUnit = 1.0f / (float)kAnimeUnit;  // a table vertex's 1/16 px as pixels
constexpr float kAbsent = 0.004f;      // a coverage under this draws nothing
constexpr float kJoinCos = 0.906f;     // a turn sharper than 25 deg gets a round join
constexpr float kReverseCos = -0.9f;   // a turn back on itself gets a round cap
constexpr float kArcStep = 0.5236f;    // the joins' arcs, 30 deg a step
constexpr float kMitreMax = 2.0f;      // a mitre's reach, in half widths
constexpr int kArcMax = 8;             // points of one arc
constexpr int kRingMax = 64;           // a slot polygon's vertices at most
constexpr float kDroolStemPx = 1.2f;   // the drool's stem's radius
constexpr float kDroolRim = 1.2f;      // its outline's reach beyond the drop
constexpr float kZPx = 9.0f;           // the z's size at the start of its flight
constexpr float kZRisePx = 42.0f;      // how far it floats up
constexpr float kZLinePx = 2.2f;
// The heart in units of its size (face_render.cpp's): two lobes of radius 0.5 at (+-0.45, 0),
// the tip at (0, 1.2), the sides tangent to the lobes at (+-0.8126, 0.3443).
constexpr float kLobeX = 0.45f;
constexpr float kLobeR = 0.5f;
constexpr float kTipY = 1.2f;
constexpr float kSideX = 0.8126f;
constexpr float kSideY = 0.3443f;
constexpr int kLobeArc = 7;  // points along each lobe's arc
constexpr float kLobeFrom = -2.689f;  // the cleft, on the right lobe (-154 deg)
constexpr float kLobeSpan = 3.448f;   // round to the side's tangent (43.5 deg)
constexpr float kLobeLeftFrom = 2.383f;  // the left lobe: its tangent (136.5 deg) round to the cleft

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
int16_t q16(float v) { return (int16_t)floorf(v * (float)kAnimeUnit + 0.5f); }
uint8_t q8(float v) { return (uint8_t)floorf(clampf(v, 0.0f, 1.0f) * 255.0f + 0.5f); }

// Where polygon k of slot s sits in a ShapeSet: its first vertex and its polygon index, and
// the way back.
struct SlotMap {
  int vertex[kAnimeSlotCount][10];
  int poly[kAnimeSlotCount][10];
  uint8_t slot_of[kAnimeSlotPolys];
  uint8_t index_of[kAnimeSlotPolys];
  int count;
};

SlotMap g_map;
bool g_map_built = false;
ShapeSet g_target;  // the sets a retarget builds (the loop task's stack is small)
ShapeSet g_neutral;

void build_map() {
  int v = 0, p = 0;
  for (int s = 0; s < kAnimeSlotCount; ++s) {
    for (int k = 0; k < kAnimeSlotMax[s]; ++k) {
      g_map.vertex[s][k] = v;
      g_map.poly[s][k] = p;
      g_map.slot_of[p] = (uint8_t)s;
      g_map.index_of[p] = (uint8_t)k;
      v += kAnimeSlotN[s];
      ++p;
    }
  }
  g_map.count = p;
  g_map_built = true;
}

// A keyframe of a style's table as a set: its polygons in place, every other one absent.
void load(const ShapeTable& t, int frame, ShapeSet* out) {
  for (int p = 0; p < g_map.count; ++p) {
    out->a[p] = 0;
    out->w[p] = 0;
    out->flags[p] = 0;
  }
  if (frame < 0 || frame >= (int)t.frame_count) return;
  const AnimeKey& kf = t.keyframes[frame];
  for (int k = 0; k < kf.count; ++k) {
    const AnimePoly& poly = t.polys[kf.first + k];
    const int s = poly.slot;
    const int v = g_map.vertex[s][poly.index];
    const int p = g_map.poly[s][poly.index];
    const int n = kAnimeSlotN[s];
    for (int i = 0; i < n; ++i) {
      out->x[v + i] = t.vertices[2 * (poly.first + i)];
      out->y[v + i] = t.vertices[2 * (poly.first + i) + 1];
    }
    out->w[p] = (int16_t)(poly.width * (kAnimeUnit / kAnimeWidthUnit));
    out->a[p] = q8(kAnimeSlotAlpha[s]);
    out->flags[p] = poly.flags;
  }
}

// A polygon's vertices moved onto the nearest points of the set's outline (line[0]): where a
// layer the keyframe lacks sits while it fades, so a mouth opens out of its line.
void snap_to_outline(const ShapeSet& ref, int p, ShapeSet* out) {
  const int s = g_map.slot_of[p], k = g_map.index_of[p];
  const int n = kAnimeSlotN[s];
  const int v = g_map.vertex[s][k];
  const int lv = g_map.vertex[kSlotLine][0];
  const int ln = kAnimeSlotN[kSlotLine];
  for (int i = 0; i < n; ++i) {
    const int32_t px = ref.x[v + i], py = ref.y[v + i];
    int32_t best = 0x7FFFFFFF;
    int16_t bx = out->x[lv], by = out->y[lv];
    for (int j = 0; j < ln; ++j) {
      const int32_t dx = out->x[lv + j] - px, dy = out->y[lv + j] - py;
      const int32_t d = dx * dx + dy * dy;
      if (d < best) {
        best = d;
        bx = out->x[lv + j];
        by = out->y[lv + j];
      }
    }
    out->x[v + i] = bx;
    out->y[v + i] = by;
  }
}

// Every polygon `set` lacks that `ref` has: ref's, snapped onto set's outline, invisible; so
// a blend of the two is defined for every polygon either one has.
void collapse(ShapeSet* set, const ShapeSet& ref) {
  for (int p = 0; p < g_map.count; ++p) {
    if (set->a[p] > 0 || ref.a[p] == 0) continue;
    snap_to_outline(ref, p, set);
    set->w[p] = ref.w[p];
    set->a[p] = 0;
    set->flags[p] = ref.flags[p];
  }
}

int16_t lerp16(int16_t a, int16_t b, float t) {
  return (int16_t)floorf((float)a + (float)(b - a) * t + 0.5f);
}

void blend(const ShapeSet& a, const ShapeSet& b, float t, ShapeSet* out) {
  for (int p = 0; p < g_map.count; ++p) {
    out->flags[p] = b.flags[p];
    if (a.a[p] == 0 && b.a[p] == 0) {
      out->a[p] = 0;
      out->w[p] = 0;
      continue;
    }
    const int s = g_map.slot_of[p];
    const int v = g_map.vertex[s][g_map.index_of[p]];
    const int n = kAnimeSlotN[s];
    for (int i = 0; i < n; ++i) {
      out->x[v + i] = lerp16(a.x[v + i], b.x[v + i], t);
      out->y[v + i] = lerp16(a.y[v + i], b.y[v + i], t);
    }
    out->w[p] = lerp16(a.w[p], b.w[p], t);
    out->a[p] = (uint8_t)floorf((float)a.a[p] + (float)((int)b.a[p] - (int)a.a[p]) * t + 0.5f);
  }
}

// Into g_target: keyframe `frame` of the table at `intensity` (1: itself; under it, part way
// from the neutral rest drawing).
void target(const ShapeTable& t, int frame, float intensity) {
  if (intensity >= 1.0f) {
    load(t, frame, &g_target);
    return;
  }
  load(t, frame, &g_neutral);  // the target first, then blended over the neutral drawing
  load(t, kNeutral * kAnimeKeys, &g_target);
  collapse(&g_neutral, g_target);
  collapse(&g_target, g_neutral);
  blend(g_target, g_neutral, intensity, &g_target);
}

// -- the frame's polygons ------------------------------------------------------------------

void push_vertex(AnimeFrame* f, float x, float y) {
  if (f->n_vertices >= kAnimeFrameVertices) return;
  // kept just inside the screen: the rasterizer's columns are 0..kScreenWidth - 1
  x = clampf(x, 1.0f, (float)kScreenWidth - 2.0f);
  y = clampf(y, -8.0f, (float)kScreenHeight + 8.0f);
  f->vx[f->n_vertices] = q16(x);
  f->vy[f->n_vertices] = q16(y);
  ++f->n_vertices;
}

// A polygon of the vertices pushed since `first`.
void end_poly(AnimeFrame* f, int first) {
  const int n = f->n_vertices - first;
  if (n < 3 || f->n_polys >= kAnimeFramePolys) {
    f->n_vertices = first;
    return;
  }
  f->poly_first[f->n_polys] = (uint16_t)first;
  f->poly_n[f->n_polys] = (uint16_t)n;
  ++f->n_polys;
}

bool begin_layer(AnimeFrame* f, uint16_t color, float alpha) {
  if (f->n_layers >= kAnimeLayers) return false;
  f->layer_first[f->n_layers] = (uint8_t)f->n_polys;
  f->layer_n[f->n_layers] = 0;
  f->layer_color[f->n_layers] = color;
  f->layer_alpha[f->n_layers] = (uint8_t)floorf(clampf(alpha, 0.0f, 1.0f) * 32.0f + 0.5f);
  return true;
}

void end_layer(AnimeFrame* f) {
  const int n = f->n_polys - f->layer_first[f->n_layers];
  if (n <= 0 || f->layer_alpha[f->n_layers] == 0) {
    f->n_polys = f->layer_first[f->n_layers];
    return;
  }
  f->layer_n[f->n_layers] = (uint8_t)n;
  ++f->n_layers;
}

void push_polygon(AnimeFrame* f, const float* xs, const float* ys, int n) {
  const int first = f->n_vertices;
  for (int i = 0; i < n; ++i) push_vertex(f, xs[i], ys[i]);
  end_poly(f, first);
}

// One side of a stroke: the ring's centreline offset by `half` to its left (side 1) or right
// (side -1), a mitre at gentle turns, an arc round sharp ones and a half circle where the line
// turns back on itself (the caps of a line drawn there and back). The right side comes out
// reversed, so the two sides fill as a band under non-zero winding.
void offset_side(AnimeFrame* f, const float* xs, const float* ys, int n, float half, int side) {
  // the segments' directions, the degenerate ones skipped
  float dx[kRingMax], dy[kRingMax];
  int idx[kRingMax];
  int m = 0;
  for (int i = 0; i < n && m < kRingMax; ++i) {
    const int j = (i + 1) % n;
    const float ex = xs[j] - xs[i], ey = ys[j] - ys[i];
    const float len = sqrtf(ex * ex + ey * ey);
    if (len < 1e-3f) continue;
    dx[m] = ex / len;
    dy[m] = ey / len;
    idx[m] = i;
    ++m;
  }
  if (m < 2) return;
  const int first = f->n_vertices;
  const float sd = (float)side;
  for (int k = 0; k < m; ++k) {
    const int kp = (k + m - 1) % m;
    const float ax = dx[kp], ay = dy[kp];  // the segment arriving at this vertex
    const float bx = dx[k], by = dy[k];    // the one leaving it
    const float px = xs[idx[k]], py = ys[idx[k]];
    // the left normal of a direction (y down): (dy, -dx); the side flips it
    const float nax = sd * ay, nay = -sd * ax;
    const float nbx = sd * by, nby = -sd * bx;
    const float dot = ax * bx + ay * by;
    const float cross = ax * by - ay * bx;
    if (dot > kJoinCos) {  // a mitre: the point at `half` from both segments
      const float mx = nax + nbx, my = nay + nby;
      const float m2 = mx * mx + my * my;
      float scale = m2 > 1e-6f ? 2.0f / m2 : 1.0f;
      if (scale > kMitreMax) scale = kMitreMax;
      push_vertex(f, px + half * mx * scale, py + half * my * scale);
      continue;
    }
    // an arc from na to nb: the short way round, or the cap's half circle through the
    // arriving direction when the line turns back
    const float angle = dot < kReverseCos ? sd * kPi : atan2f(cross, dot);
    const float a0 = atan2f(nay, nax);
    int steps = (int)(fabsf(angle) / kArcStep + 0.999f);
    if (steps < 1) steps = 1;
    if (steps > kArcMax) steps = kArcMax;
    for (int s = 0; s <= steps; ++s) {
      const float a = a0 + angle * (float)s / (float)steps;
      push_vertex(f, px + half * cosf(a), py + half * sinf(a));
    }
  }
  if (side < 0) {  // reversed: the band's inner boundary runs against the outer
    for (int i = first, j = f->n_vertices - 1; i < j; ++i, --j) {
      const int16_t tx = f->vx[i], ty = f->vy[i];
      f->vx[i] = f->vx[j];
      f->vy[i] = f->vy[j];
      f->vx[j] = tx;
      f->vy[j] = ty;
    }
  }
  end_poly(f, first);
}

void push_stroke(AnimeFrame* f, const float* xs, const float* ys, int n, float width) {
  if (width <= 0.1f) return;
  offset_side(f, xs, ys, n, 0.5f * width, 1);
  offset_side(f, xs, ys, n, 0.5f * width, -1);
}

// A capsule (the segment a-b swept by a disc of radius r) as a polygon.
void push_capsule(AnimeFrame* f, float ax, float ay, float bx, float by, float r) {
  float dx = bx - ax, dy = by - ay;
  const float len = sqrtf(dx * dx + dy * dy);
  if (len < 1e-3f) {
    dx = 0.0f;
    dy = 1.0f;
  } else {
    dx /= len;
    dy /= len;
  }
  const float base = atan2f(-dx, dy);  // the left normal's angle
  const int first = f->n_vertices;
  for (int s = 0; s <= 6; ++s) {  // round b: from the left normal over the far end to the right
    const float a = base + kPi * (float)s / 6.0f;
    push_vertex(f, bx + r * cosf(a), by + r * sinf(a));
  }
  for (int s = 0; s <= 6; ++s) {  // and round a back to the left
    const float a = base + kPi + kPi * (float)s / 6.0f;
    push_vertex(f, ax + r * cosf(a), ay + r * sinf(a));
  }
  end_poly(f, first);
}

void push_circle(AnimeFrame* f, float cx, float cy, float r) {
  const int first = f->n_vertices;
  for (int s = 0; s < 12; ++s) {
    const float a = kTwoPi * (float)s / 12.0f;
    push_vertex(f, cx + r * cosf(a), cy + r * sinf(a));
  }
  end_poly(f, first);
}

// A heart of size s (its lobes' centres at (x0 +- 0.45 s, y0)) as one polygon: the right lobe
// from the cleft over its top to the side's tangent, the tip, the left lobe from its tangent
// over its top back to the cleft.
void push_heart(AnimeFrame* f, float x0, float y0, float s) {
  const int first = f->n_vertices;
  for (int k = 0; k <= kLobeArc; ++k) {
    const float a = kLobeFrom + kLobeSpan * (float)k / (float)kLobeArc;
    push_vertex(f, x0 + kLobeX * s + kLobeR * s * cosf(a), y0 + kLobeR * s * sinf(a));
  }
  push_vertex(f, x0 + kSideX * s, y0 + kSideY * s);
  push_vertex(f, x0, y0 + kTipY * s);
  push_vertex(f, x0 - kSideX * s, y0 + kSideY * s);
  for (int k = 0; k <= kLobeArc; ++k) {
    const float a = kLobeLeftFrom + kLobeSpan * (float)k / (float)kLobeArc;
    push_vertex(f, x0 - kLobeX * s + kLobeR * s * cosf(a), y0 + kLobeR * s * sinf(a));
  }
  end_poly(f, first);
}

// -- the rasterizer ------------------------------------------------------------------------

struct Edge {
  float x0, y0, y1;  // from (x0, y0) down to y1
  float dxdy;
  float dir;
  int16_t next;  // the next edge starting on the same row
};

constexpr int kEdgeMax = 512;  // a layer's edges within a strip
Edge g_edges[kEdgeMax];
int16_t g_row_head[kScreenHeight];
int16_t g_active[kEdgeMax];
float g_acc[kScreenWidth + 2];

inline uint16_t load_px(const Strip& s, int i) {
  const uint16_t v = s.px[i];
  return s.swapped ? (uint16_t)((v >> 8) | (v << 8)) : v;
}

inline void store_px(const Strip& s, int i, uint16_t c) {
  s.px[i] = s.swapped ? (uint16_t)((c >> 8) | (c << 8)) : c;
}

inline uint16_t mix(uint16_t dst, uint16_t c, int w) {
  const int r = (((dst >> 11) & 31) * (32 - w) + ((c >> 11) & 31) * w) >> 5;
  const int g = (((dst >> 5) & 63) * (32 - w) + ((c >> 5) & 63) * w) >> 5;
  const int b = ((dst & 31) * (32 - w) + (c & 31) * w) >> 5;
  return (uint16_t)((r << 11) | (g << 5) | b);
}

// The signed area an edge's piece within row `y` adds to the row's accumulator (the
// accumulation rasterizer: the coverage at a pixel is the running sum along the row).
void accumulate(const Edge& e, int y, float* acc, int* xmin, int* xmax) {
  const float ya = fmaxf((float)y, e.y0);
  const float yb = fminf((float)(y + 1), e.y1);
  const float dy = yb - ya;
  if (dy <= 0.0f) return;
  const float x = e.x0 + e.dxdy * (ya - e.y0);
  const float xnext = x + e.dxdy * dy;
  const float d = dy * e.dir;
  float x0 = x, x1 = xnext;
  if (x1 < x0) {
    x0 = xnext;
    x1 = x;
  }
  const float x0floor = floorf(x0);
  const int x0i = (int)x0floor;
  const float x1ceil = ceilf(x1);
  const int x1i = (int)x1ceil;
  if (x0i < *xmin) *xmin = x0i;
  if (x1i + 1 > *xmax) *xmax = x1i + 1;
  if (x1i <= x0i + 1) {
    const float xmf = 0.5f * (x + xnext) - x0floor;
    acc[x0i] += d - d * xmf;
    acc[x0i + 1] += d * xmf;
    return;
  }
  const float s = 1.0f / (x1 - x0);
  const float x0f = x0 - x0floor;
  const float a0 = 0.5f * s * (1.0f - x0f) * (1.0f - x0f);
  const float x1f = x1 - x1ceil + 1.0f;
  const float am = 0.5f * s * x1f * x1f;
  acc[x0i] += d * a0;
  if (x1i == x0i + 2) {
    acc[x0i + 1] += d * (1.0f - a0 - am);
  } else {
    const float a1 = s * (1.5f - x0f);
    acc[x0i + 1] += d * (a1 - a0);
    for (int xi = x0i + 2; xi < x1i - 1; ++xi) acc[xi] += d * s;
    const float a2 = a1 + (float)(x1i - x0i - 3) * s;
    acc[x1i - 1] += d * (1.0f - a2 - am);
  }
  acc[x1i] += d * am;
}

// One layer of the frame into the strip: its edges bucketed by their first row, the rows
// swept with the active ones, each row's coverage blended in at the layer's opacity.
void draw_layer(const AnimeFrame& f, int layer, const Strip& s) {
  const float top = (float)s.y0;
  const float bottom = (float)(s.y0 + s.rows);
  for (int r = 0; r < s.rows; ++r) g_row_head[r] = -1;
  int n_edges = 0;
  int row_lo = s.rows, row_hi = -1;
  const int p0 = f.layer_first[layer];
  const int p1 = p0 + f.layer_n[layer];
  for (int p = p0; p < p1; ++p) {
    const int first = f.poly_first[p];
    const int n = f.poly_n[p];
    for (int i = 0; i < n; ++i) {
      const int j = i + 1 < n ? i + 1 : 0;
      float ax = (float)f.vx[first + i] * kUnit, ay = (float)f.vy[first + i] * kUnit;
      float bx = (float)f.vx[first + j] * kUnit, by = (float)f.vy[first + j] * kUnit;
      if (ay == by) continue;
      float dir = 1.0f;
      if (ay > by) {
        const float tx = ax, ty = ay;
        ax = bx;
        ay = by;
        bx = tx;
        by = ty;
        dir = -1.0f;
      }
      if (by <= top || ay >= bottom || n_edges >= kEdgeMax) continue;
      Edge& e = g_edges[n_edges];
      e.x0 = ax;
      e.y0 = ay;
      e.y1 = by;
      e.dxdy = (bx - ax) / (by - ay);
      e.dir = dir;
      int row = (int)floorf(ay) - s.y0;
      if (row < 0) row = 0;
      e.next = g_row_head[row];
      g_row_head[row] = (int16_t)n_edges;
      if (row < row_lo) row_lo = row;
      int last = (int)ceilf(by) - s.y0 - 1;
      if (last > s.rows - 1) last = s.rows - 1;
      if (last > row_hi) row_hi = last;
      ++n_edges;
    }
  }
  if (n_edges == 0) return;
  const uint16_t color = f.layer_color[layer];
  const int alpha = f.layer_alpha[layer];
  int n_active = 0;
  for (int r = row_lo; r <= row_hi; ++r) {
    const int y = s.y0 + r;
    // the edges leaving and the ones arriving
    int m = 0;
    for (int k = 0; k < n_active; ++k) {
      if (g_edges[g_active[k]].y1 > (float)y) g_active[m++] = g_active[k];
    }
    n_active = m;
    for (int e = g_row_head[r]; e >= 0; e = g_edges[e].next) {
      if (n_active < kEdgeMax) g_active[n_active++] = (int16_t)e;
    }
    if (n_active == 0) continue;
    int xmin = kScreenWidth, xmax = 0;
    for (int k = 0; k < n_active; ++k) accumulate(g_edges[g_active[k]], y, g_acc, &xmin, &xmax);
    if (xmin < 0) xmin = 0;
    if (xmax > kScreenWidth) xmax = kScreenWidth;
    float sum = 0.0f;
    const int row = r * kScreenWidth;
    for (int x = xmin; x < xmax; ++x) {
      sum += g_acc[x];
      g_acc[x] = 0.0f;
      float cov = fabsf(sum);
      if (cov < kAbsent) continue;
      if (cov > 1.0f) cov = 1.0f;
      const int w = (int)(cov * (float)alpha + 0.5f);
      if (w <= 0) continue;
      const int i = row + x;
      store_px(s, i, w >= 32 ? color : mix(load_px(s, i), color, w));
    }
    g_acc[xmax] = 0.0f;
  }
}

}  // namespace

const ShapeTable* shape_table_of(int style) {
  for (int i = 0; i < kShapeTableCount; ++i) {
    if (kShapeTables[i]->style == style) return kShapeTables[i];
  }
  return nullptr;
}

int AnimeModel::talkKey(int style, float level) {
  // the lip flap's cut, as the parametric model takes a style with flap_shapes 3
  const float shapes = kStyles[style].flap_shapes;
  if (shapes < 2.0f) return 0;
  const float q = floorf(level * (shapes - 1.0f) + 0.5f);
  return q <= 0.0f ? 0 : (q >= shapes - 1.0f ? kAnimeKeys - 1 : 1);
}

AnimeModel::AnimeModel() {
  if (!g_map_built) build_map();
  memset(&from_, 0, sizeof from_);
  memset(&to_, 0, sizeof to_);
  memset(&cur_, 0, sizeof cur_);
  memset(&gest_, 0, sizeof gest_);
  memset(&frame_, 0, sizeof frame_);
}

// The tween from the shape on the screen to g_target over `duration_ms`.
void AnimeModel::retarget(float duration_ms, uint32_t now_ms) {
  from_ = cur_;
  to_ = g_target;
  collapse(&to_, from_);
  collapse(&from_, to_);
  if (gesture_toward_) {
    collapse(&to_, gest_);
    collapse(&gest_, to_);
  }
  t0_ms_ = now_ms;
  dur_ms_ = duration_ms > 0.0f ? duration_ms : 0.0f;
  if (dur_ms_ == 0.0f) cur_ = to_;
}

void AnimeModel::follow(const FaceModel& model, uint32_t now_ms) {
  const ShapeTable* t = shape_table_of(model.style());
  if (t == nullptr) return;  // a parametric style: the other renderer's frame
  const Shape ps = model.shape();
  const int key = talkKey(t->style, model.level());
  const int frame = model.shown() * kAnimeKeys + key;
  if (!primed_ || t != table_) {
    // the first frame, or the style arrived: from another shapes style its drawing morphs
    // over the model's transition (a gesture's target comes from the new table), from a
    // parametric style (its shape unknown here) the drawing is shown at once
    const bool morph = primed_ && table_ != nullptr;
    table_ = t;
    target(*t, frame, model.intensity());
    if (morph) {
      if (gesture_toward_) load(*t, t->gestures[gesture_].toward, &gest_);
      retarget(model.transitionMs(), model.transitionStartMs());
    } else {
      cur_ = g_target;
      retarget(0.0f, now_ms);
    }
    primed_ = true;
    starts_ = model.starts();
    key_ = key;
  } else if (model.starts() != starts_) {  // an expression transition began
    target(*t, frame, model.intensity());
    retarget(model.transitionMs(), model.transitionStartMs());  // on the model's clock
    starts_ = model.starts();
    key_ = key;
  } else if (key != key_) {  // the talk cut
    target(*t, frame, model.intensity());
    retarget(t->talk_tween_ms, now_ms);
    key_ = key;
  }
  const float elapsed = (float)(int32_t)(now_ms - t0_ms_);
  const float e = dur_ms_ > 0.0f ? ease(elapsed / dur_ms_) : 1.0f;
  blend(from_, to_, e, &cur_);

  // the idle gesture's target, materialized when it starts
  const int gesture = model.idleGesture();
  if (gesture != gesture_) {
    gesture_ = gesture;
    gesture_toward_ = gesture >= 0 && t->gestures[gesture].toward >= 0;
    if (gesture_toward_) {
      load(*t, t->gestures[gesture].toward, &gest_);
      collapse(&gest_, cur_);
      collapse(&from_, gest_);
      collapse(&to_, gest_);
      blend(from_, to_, e, &cur_);
    }
  }
  build(model, ps, now_ms);
}

// The frame's polygons from cur_: the gesture, the breath, the drift, the wave and the wag
// applied to each slot's polygons in draw order, then the drool, the heart and the z.
void AnimeModel::build(const FaceModel& model, const Shape& ps, uint32_t now_ms) {
  const ShapeTable& t = *table_;
  const int gesture = gesture_;
  const float amp = model.idleAmp();  // signed: a mirror gesture's side
  const float mag = fabsf(amp);
  const float toward = gesture >= 0 && gesture_toward_ && mag > 0.0f
                           ? mag * t.gestures[gesture].amount
                           : 0.0f;
  // the mouth's centre and half width: its outline's box
  const int lv = g_map.vertex[kSlotLine][0];
  const int ln = kAnimeSlotN[kSlotLine];
  float x0 = (float)cur_.x[lv], x1 = x0, y0 = (float)cur_.y[lv], y1 = y0;
  for (int i = 1; i < ln; ++i) {
    x0 = fminf(x0, (float)cur_.x[lv + i]);
    x1 = fmaxf(x1, (float)cur_.x[lv + i]);
    y0 = fminf(y0, (float)cur_.y[lv + i]);
    y1 = fmaxf(y1, (float)cur_.y[lv + i]);
  }
  x0 *= kUnit;
  x1 *= kUnit;
  y0 *= kUnit;
  y1 *= kUnit;
  const float cx = 0.5f * (x0 + x1), cy = 0.5f * (y0 + y1);
  const float hw = fmaxf(0.5f * (x1 - x0), 1.0f);
  // the gesture's deformation and the breath, about the mouth's centre
  float sx = 1.0f, sy = 1.0f, dx = 0.0f, dy = 0.0f, tilt = 0.0f, curl = 0.0f;
  if (gesture >= 0 && mag > 0.0f) {
    const AnimeGesture& g = t.gestures[gesture];
    sx += (g.sx - 1.0f) * mag;
    sy += (g.sy - 1.0f) * mag;
    dx += g.dx * amp;
    dy += g.dy * mag;
    tilt += g.tilt * amp;
    curl += g.curl * amp;
  }
  const float b = sinf(kTwoPi * model.breathePhase());
  const float breathe = ps.p[kBreathe];
  sx *= 1.0f + 0.035f * breathe * b;
  sy *= 1.0f + 0.06f * breathe * (0.5f + 0.5f * b);
  dx += ps.p[kDrift] * (1.0f / 3.0f) * kShiftXPx * sinf(kTwoPi * model.driftPhase());
  const float wave = clampf(ps.p[kWave], 0.0f, 1.0f) * kWavePx;
  const float wave_n = ps.p[kWaveN];
  const float phase = ps.wave_phase;
  // the tongue's wag: its tip sways sideways by the depth below its top
  const float wag = ps.p[kWag];
  const int tp = g_map.poly[kSlotTongue][0];
  float ty0 = 0.0f, th = 1.0f, sway = 0.0f;
  if (wag > 0.0f && cur_.a[tp] > 0) {
    const int tv = g_map.vertex[kSlotTongue][0];
    const int tn = kAnimeSlotN[kSlotTongue];
    float ty1 = (float)cur_.y[tv];
    ty0 = ty1;
    for (int i = 1; i < tn; ++i) {
      ty0 = fminf(ty0, (float)cur_.y[tv + i]);
      ty1 = fmaxf(ty1, (float)cur_.y[tv + i]);
    }
    ty0 *= kUnit;
    th = fmaxf(ty1 * kUnit - ty0, 1.0f);
    sway = wag * 0.5f * th * sinf(kTwoPi * model.wagPhase());
  }

  AnimeFrame& f = frame_;
  f.n_vertices = f.n_polys = f.n_layers = 0;
  f.background = kStyles[t.style].background;
  float xs[kRingMax], ys[kRingMax];
  for (int s = 0; s < kAnimeSlotCount; ++s) {
    const uint16_t color = ps.color[kAnimeSlotElement[s]];
    for (int k = 0; k < kAnimeSlotMax[s]; ++k) {
      const int p = g_map.poly[s][k];
      const int v = g_map.vertex[s][k];
      const int n = kAnimeSlotN[s];
      // the gesture's morph target over the blend
      float alpha = (float)cur_.a[p] / 255.0f;
      float width = (float)cur_.w[p] * kUnit;
      if (toward > 0.0f) {
        alpha += ((float)gest_.a[p] / 255.0f - alpha) * toward;
        width += ((float)gest_.w[p] * kUnit - width) * toward;
      }
      if (alpha < kAbsent) continue;
      const bool follows = sway != 0.0f && (s == kSlotTongue || (cur_.flags[p] & 1u));
      for (int i = 0; i < n; ++i) {
        float x = (float)cur_.x[v + i] * kUnit, y = (float)cur_.y[v + i] * kUnit;
        if (toward > 0.0f) {
          x += ((float)gest_.x[v + i] * kUnit - x) * toward;
          y += ((float)gest_.y[v + i] * kUnit - y) * toward;
        }
        const float rx = x - cx, ry = y - cy;
        const float u = rx / hw;
        float yy = cy + ry * sy + dy + tilt * rx + curl * u * u;
        float xx = cx + rx * sx + dx;
        if (wave > 0.0f) yy += wave * sinf(kTwoPi * (0.5f * (u + 1.0f) * wave_n - phase));
        if (follows) xx += sway * clampf((y - ty0) / th, 0.0f, 1.0f);
        xs[i] = xx;
        ys[i] = yy;
      }
      if (kAnimeSlotKind[s] == kKindAnchor) {  // the drool: a drop stretching from the point
        const float period = t.drool_s * 1000.0f;
        const float kt = (float)(now_ms % (uint32_t)period) / period;
        const float stretch = kt < 0.85f ? ease(kt / 0.85f) : 1.0f - ease((kt - 0.85f) / 0.15f);
        const float len = width * (1.6f + 4.5f * stretch);
        const float rd = width * (0.55f + 0.45f * stretch);
        const float yb = fminf(ys[0] + len, (float)kScreenHeight - kMarginPx - rd - kDroolRim);
        if (begin_layer(&f, ps.color[kElLip], alpha)) {
          push_capsule(&f, xs[0], ys[0], xs[0], yb - rd, kDroolStemPx + kDroolRim);
          push_circle(&f, xs[0], yb - rd, rd + kDroolRim);
          end_layer(&f);
        }
        if (begin_layer(&f, color, alpha)) {
          push_capsule(&f, xs[0], ys[0], xs[0], yb - rd, kDroolStemPx);
          push_circle(&f, xs[0], yb - rd, rd);
          end_layer(&f);
        }
        continue;
      }
      if (!begin_layer(&f, color, alpha)) continue;
      if (kAnimeSlotKind[s] == kKindStroke) {
        push_stroke(&f, xs, ys, n, width);
      } else {
        push_polygon(&f, xs, ys, n);
      }
      end_layer(&f);
    }
  }
  // the overlays (the cheek pair, the shock's stripes): the table's keyframe drawn as it is,
  // over the mouth, at the opacity of the look's parameter; they neither morph nor move with
  // the gestures and the breath
  for (int o = 0; o < kAnimeOverlayCount; ++o) {
    const int frame = t.overlays[o];
    if (frame < 0) continue;
    const float amount = clampf(ps.p[kAnimeOverlayParam[o]], 0.0f, 1.0f);
    if (amount < kAbsent) continue;
    const AnimeKey& kf = t.keyframes[frame];
    for (int k = 0; k < kf.count; ++k) {
      const AnimePoly& poly = t.polys[kf.first + k];
      const int s = poly.slot;
      const int n = kAnimeSlotN[s];
      if (kAnimeSlotKind[s] == kKindAnchor) continue;
      for (int i = 0; i < n; ++i) {
        xs[i] = (float)t.vertices[2 * (poly.first + i)] * kUnit;
        ys[i] = (float)t.vertices[2 * (poly.first + i) + 1] * kUnit;
      }
      if (!begin_layer(&f, ps.color[kAnimeSlotElement[s]], kAnimeSlotAlpha[s] * amount)) continue;
      if (kAnimeSlotKind[s] == kKindStroke) {
        push_stroke(&f, xs, ys, n, (float)poly.width * (1.0f / (float)kAnimeWidthUnit));
      } else {
        push_polygon(&f, xs, ys, n);
      }
      end_layer(&f);
    }
  }
  // the kiss's heart: from beside the mouth's right edge up and out, as the parametric one
  if (ps.heart >= 0.0f) {
    const float kf = ps.heart;
    const float size = kHeartPx * (0.55f + 0.45f * ease(kf * 5.0f));
    const float rise = 1.0f - (1.0f - kf) * (1.0f - kf);
    float hx = x1 + 4.0f + 0.6f * kHeartPx + 12.0f * kf + 5.0f * sinf(kTwoPi * 1.5f * kf);
    float hy = cy - 0.6f * kHeartPx - kHeartRisePx * rise;
    hx = fminf(hx, (float)kScreenWidth - kMarginPx - (kLobeX + kLobeR) * size);
    hy = fmaxf(hy, kMarginPx + kLobeR * size);
    const float alpha = kf < 0.55f ? 1.0f : 1.0f - (kf - 0.55f) / 0.45f;
    if (begin_layer(&f, ps.color[kElHeart], alpha)) {
      push_heart(&f, hx, hy, size);
      end_layer(&f);
    }
  }
  // the sleeper's z: a small z that grows as it floats up and right from the mouth's corner
  if (ps.z >= 0.0f) {
    const float kz = ps.z;
    const float size = kZPx * (0.6f + 0.6f * kz);
    float zx = x1 + 6.0f + 16.0f * kz;
    float zy = y0 - 6.0f - kZRisePx * ease(kz);
    zx = fminf(zx, (float)kScreenWidth - kMarginPx - size - kZLinePx);
    zy = fmaxf(zy, kMarginPx + kZLinePx);
    const float alpha = kz < 0.5f ? 1.0f : 1.0f - (kz - 0.5f) / 0.5f;
    // the glyph's three strokes as one line there and back
    const float lx[6] = {zx, zx + size, zx, zx + size, zx, zx + size};
    const float ly[6] = {zy, zy, zy + size, zy + size, zy + size, zy};
    if (begin_layer(&f, ps.color[kElLip], alpha)) {
      push_stroke(&f, lx, ly, 6, kZLinePx);
      end_layer(&f);
    }
  }
}

void draw_anime(const AnimeFrame& frame, Strip s) {
  const uint16_t bg = s.swapped ? (uint16_t)((frame.background >> 8) | (frame.background << 8))
                                : frame.background;
  const int n = s.rows * kScreenWidth;
  for (int i = 0; i < n; ++i) s.px[i] = bg;
  for (int x = 0; x < kScreenWidth + 2; ++x) g_acc[x] = 0.0f;
  for (int layer = 0; layer < frame.n_layers; ++layer) draw_layer(frame, layer, s);
}

}  // namespace face
