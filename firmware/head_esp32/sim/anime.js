// The shape-keyframe mouth (the anime style and the styles drawn like it): a line-by-line port
// of src/face_anime.cpp, so the simulator draws what the head will. The keyframes are one
// table per shapes style (<style>_table.js, generated from config/<style>/*.svg, listed in
// shape_tables.js); the model follows face.js's FaceModel (the parametric model: its
// transitions, speech level, idle gestures and effects) and morphs the drawings vertex by
// vertex; the frame is rasterized by scanlines with anti-aliased coverage (non-zero winding).
// Math.fround keeps the arithmetic in float32 like the ESP32's; tests/unit/test_face_parity.py
// renders the same scripts with both and compares the pixels.
"use strict";

(function (root) {
  const T = typeof FACE_TABLE !== "undefined" ? FACE_TABLE : require("./face_table.js");
  const TABLES = typeof SHAPE_TABLES !== "undefined" ? SHAPE_TABLES : require("./shape_tables.js");
  const Face = typeof window !== "undefined" ? root.Face : require("./face.js");
  const f = Math.fround;
  const W = T.screen.width;
  const H = T.screen.height;
  const G = T.geometry;
  const P = Face.P;
  const E = Face.E;
  const ease = Face.ease;
  // the layout every table shares (the generator's): the layers, the units, the keys
  const A = TABLES[0];
  const KEYS = A.keys.length;
  const SLOTS = A.slots;
  const SLOT_COUNT = SLOTS.length;
  const SLOT_LINE = SLOTS.findIndex((s) => s.name === "line");
  const SLOT_TONGUE = SLOTS.findIndex((s) => s.name === "tongue");
  const NEUTRAL = Face.NEUTRAL;
  // The table of a shapes style (an index of T.styles), or null for a parametric one: whether
  // the shape engine draws the frame (src/face_anime.cpp shape_table_of).
  const tableOf = (style) => TABLES.find((t) => t.style === style) || null;
  const TWO_PI = f(6.28318530718);
  const PI = f(3.14159265359);
  const UNIT_F = f(1 / A.unit);
  const ABSENT = f(0.004);
  const JOIN_COS = f(0.906);
  const REVERSE_COS = f(-0.9);
  const ARC_STEP = f(0.5236);
  const MITRE_MAX = 2;
  const ARC_MAX = 8;
  const RING_MAX = 64;
  const DROOL_STEM_PX = f(1.2);
  const DROOL_RIM = f(1.2);
  const Z_PX = 9;
  const Z_RISE_PX = 42;
  const Z_LINE_PX = f(2.2);
  const LOBE_X = f(0.45), LOBE_R = f(0.5), TIP_Y = f(1.2), SIDE_X = f(0.8126), SIDE_Y = f(0.3443);
  const LOBE_ARC = 7;
  const LOBE_FROM = f(-2.689), LOBE_SPAN = f(3.448), LOBE_LEFT_FROM = f(2.383);
  const FRAME_POLYS = 80;
  const FRAME_VERTICES = 2048;
  const LAYERS = 40;
  const EDGE_MAX = 512;
  const LINE_UNIT = A.unit / A.width_unit;
  const clampf = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const sinf = (x) => f(Math.sin(x));
  const cosf = (x) => f(Math.cos(x));
  const atan2f = (y, x) => f(Math.atan2(y, x));
  const sqrtf = (x) => f(Math.sqrt(x));
  const q16 = (v) => Math.floor(f(f(v * A.unit) + 0.5));
  const q8 = (v) => Math.floor(f(f(clampf(v, 0, 1) * 255) + 0.5));

  // Where polygon k of slot s sits in a shape set: its first vertex and its polygon index, and
  // the way back.
  const MAP = { vertex: [], poly: [], slotOf: [], indexOf: [], count: 0, vertices: 0 };
  (() => {
    let v = 0, p = 0;
    for (let s = 0; s < SLOT_COUNT; s++) {
      MAP.vertex.push([]);
      MAP.poly.push([]);
      for (let k = 0; k < SLOTS[s].max; k++) {
        MAP.vertex[s].push(v);
        MAP.poly[s].push(p);
        MAP.slotOf.push(s);
        MAP.indexOf.push(k);
        v += SLOTS[s].n;
        p++;
      }
    }
    MAP.count = p;
    MAP.vertices = v;
  })();

  // A materialized set of every slot's polygons (src/face_anime.h ShapeSet).
  function newSet() {
    return {
      x: new Int16Array(MAP.vertices), y: new Int16Array(MAP.vertices),
      w: new Int16Array(MAP.count), a: new Uint8Array(MAP.count), flags: new Uint8Array(MAP.count),
    };
  }

  function copySet(src, dst) {
    dst.x.set(src.x); dst.y.set(src.y); dst.w.set(src.w); dst.a.set(src.a); dst.flags.set(src.flags);
  }

  // A keyframe of a style's table as a set: its polygons in place, every other one absent.
  function load(t, frame, out) {
    out.a.fill(0); out.w.fill(0); out.flags.fill(0);
    if (frame < 0 || frame >= t.frames.length) return;
    for (const poly of t.frames[frame].polys) {
      const s = poly.slot;
      const v = MAP.vertex[s][poly.index];
      const p = MAP.poly[s][poly.index];
      const n = SLOTS[s].n;
      for (let i = 0; i < n; i++) {
        out.x[v + i] = poly.points[2 * i];
        out.y[v + i] = poly.points[2 * i + 1];
      }
      out.w[p] = Math.trunc(poly.width * LINE_UNIT);
      out.a[p] = q8(SLOTS[s].alpha);
      out.flags[p] = poly.flags;
    }
  }

  // A polygon's vertices moved onto the nearest points of the set's outline (line[0]).
  function snapToOutline(ref, p, out) {
    const s = MAP.slotOf[p], k = MAP.indexOf[p];
    const n = SLOTS[s].n;
    const v = MAP.vertex[s][k];
    const lv = MAP.vertex[SLOT_LINE][0];
    const ln = SLOTS[SLOT_LINE].n;
    for (let i = 0; i < n; i++) {
      const px = ref.x[v + i], py = ref.y[v + i];
      let best = 0x7fffffff;
      let bx = out.x[lv], by = out.y[lv];
      for (let j = 0; j < ln; j++) {
        const dx = out.x[lv + j] - px, dy = out.y[lv + j] - py;
        const d = dx * dx + dy * dy;
        if (d < best) { best = d; bx = out.x[lv + j]; by = out.y[lv + j]; }
      }
      out.x[v + i] = bx;
      out.y[v + i] = by;
    }
  }

  // Every polygon `set` lacks that `ref` has: ref's, snapped onto set's outline, invisible.
  function collapse(set, ref) {
    for (let p = 0; p < MAP.count; p++) {
      if (set.a[p] > 0 || ref.a[p] === 0) continue;
      snapToOutline(ref, p, set);
      set.w[p] = ref.w[p];
      set.a[p] = 0;
      set.flags[p] = ref.flags[p];
    }
  }

  const lerp16 = (a, b, t) => Math.floor(f(f(a + f(f(b - a) * t)) + 0.5));

  function blend(a, b, t, out) {
    for (let p = 0; p < MAP.count; p++) {
      out.flags[p] = b.flags[p];
      if (a.a[p] === 0 && b.a[p] === 0) { out.a[p] = 0; out.w[p] = 0; continue; }
      const s = MAP.slotOf[p];
      const v = MAP.vertex[s][MAP.indexOf[p]];
      const n = SLOTS[s].n;
      for (let i = 0; i < n; i++) {
        out.x[v + i] = lerp16(a.x[v + i], b.x[v + i], t);
        out.y[v + i] = lerp16(a.y[v + i], b.y[v + i], t);
      }
      out.w[p] = lerp16(a.w[p], b.w[p], t);
      out.a[p] = Math.floor(f(f(a.a[p] + f(f(b.a[p] - a.a[p]) * t)) + 0.5));
    }
  }

  const gTarget = newSet();
  const gNeutral = newSet();

  // Into gTarget: keyframe `frame` of the table at `intensity`.
  function target(t, frame, intensity) {
    if (intensity >= 1) { load(t, frame, gTarget); return; }
    load(t, frame, gNeutral);
    load(t, NEUTRAL * KEYS, gTarget);
    collapse(gNeutral, gTarget);
    collapse(gTarget, gNeutral);
    blend(gTarget, gNeutral, intensity, gTarget);
  }

  // -- the frame's polygons -----------------------------------------------------------------
  function newFrame() {
    return {
      vx: new Int16Array(FRAME_VERTICES), vy: new Int16Array(FRAME_VERTICES),
      polyFirst: new Uint16Array(FRAME_POLYS), polyN: new Uint16Array(FRAME_POLYS),
      layerFirst: new Uint8Array(LAYERS), layerN: new Uint8Array(LAYERS),
      layerColor: new Uint16Array(LAYERS), layerAlpha: new Uint8Array(LAYERS),
      nVertices: 0, nPolys: 0, nLayers: 0, background: 0,
    };
  }

  function pushVertex(fr, x, y) {
    if (fr.nVertices >= FRAME_VERTICES) return;
    x = clampf(x, 1, W - 2);
    y = clampf(y, -8, H + 8);
    fr.vx[fr.nVertices] = q16(x);
    fr.vy[fr.nVertices] = q16(y);
    fr.nVertices++;
  }

  function endPoly(fr, first) {
    const n = fr.nVertices - first;
    if (n < 3 || fr.nPolys >= FRAME_POLYS) { fr.nVertices = first; return; }
    fr.polyFirst[fr.nPolys] = first;
    fr.polyN[fr.nPolys] = n;
    fr.nPolys++;
  }

  function beginLayer(fr, color, alpha) {
    if (fr.nLayers >= LAYERS) return false;
    fr.layerFirst[fr.nLayers] = fr.nPolys;
    fr.layerN[fr.nLayers] = 0;
    fr.layerColor[fr.nLayers] = color;
    fr.layerAlpha[fr.nLayers] = Math.floor(f(f(clampf(alpha, 0, 1) * 32) + 0.5));
    return true;
  }

  function endLayer(fr) {
    const n = fr.nPolys - fr.layerFirst[fr.nLayers];
    if (n <= 0 || fr.layerAlpha[fr.nLayers] === 0) { fr.nPolys = fr.layerFirst[fr.nLayers]; return; }
    fr.layerN[fr.nLayers] = n;
    fr.nLayers++;
  }

  function pushPolygon(fr, xs, ys, n) {
    const first = fr.nVertices;
    for (let i = 0; i < n; i++) pushVertex(fr, xs[i], ys[i]);
    endPoly(fr, first);
  }

  const sdx = new Float32Array(RING_MAX), sdy = new Float32Array(RING_MAX), sidx = new Int32Array(RING_MAX);

  // One side of a stroke (src/face_anime.cpp offset_side).
  function offsetSide(fr, xs, ys, n, half, side) {
    let m = 0;
    for (let i = 0; i < n && m < RING_MAX; i++) {
      const j = (i + 1) % n;
      const ex = f(xs[j] - xs[i]), ey = f(ys[j] - ys[i]);
      const len = sqrtf(f(f(ex * ex) + f(ey * ey)));
      if (len < 1e-3) continue;
      sdx[m] = f(ex / len);
      sdy[m] = f(ey / len);
      sidx[m] = i;
      m++;
    }
    if (m < 2) return;
    const first = fr.nVertices;
    const sd = side;
    for (let k = 0; k < m; k++) {
      const kp = (k + m - 1) % m;
      const ax = sdx[kp], ay = sdy[kp];
      const bx = sdx[k], by = sdy[k];
      const px = xs[sidx[k]], py = ys[sidx[k]];
      const nax = f(sd * ay), nay = f(-sd * ax);
      const nbx = f(sd * by), nby = f(-sd * bx);
      const dot = f(f(ax * bx) + f(ay * by));
      const cross = f(f(ax * by) - f(ay * bx));
      if (dot > JOIN_COS) {
        const mx = f(nax + nbx), my = f(nay + nby);
        const m2 = f(f(mx * mx) + f(my * my));
        let scale = m2 > 1e-6 ? f(2 / m2) : 1;
        if (scale > MITRE_MAX) scale = MITRE_MAX;
        pushVertex(fr, f(px + f(f(half * mx) * scale)), f(py + f(f(half * my) * scale)));
        continue;
      }
      const angle = dot < REVERSE_COS ? f(sd * PI) : atan2f(cross, dot);
      const a0 = atan2f(nay, nax);
      let steps = Math.trunc(f(f(Math.abs(angle) / ARC_STEP) + 0.999));
      if (steps < 1) steps = 1;
      if (steps > ARC_MAX) steps = ARC_MAX;
      for (let s = 0; s <= steps; s++) {
        const a = f(a0 + f(f(angle * s) / steps));
        pushVertex(fr, f(px + f(half * cosf(a))), f(py + f(half * sinf(a))));
      }
    }
    if (side < 0) {
      for (let i = first, j = fr.nVertices - 1; i < j; i++, j--) {
        const tx = fr.vx[i], ty = fr.vy[i];
        fr.vx[i] = fr.vx[j]; fr.vy[i] = fr.vy[j];
        fr.vx[j] = tx; fr.vy[j] = ty;
      }
    }
    endPoly(fr, first);
  }

  function pushStroke(fr, xs, ys, n, width) {
    if (width <= 0.1) return;
    offsetSide(fr, xs, ys, n, f(0.5 * width), 1);
    offsetSide(fr, xs, ys, n, f(0.5 * width), -1);
  }

  function pushCapsule(fr, ax, ay, bx, by, r) {
    let dx = f(bx - ax), dy = f(by - ay);
    const len = sqrtf(f(f(dx * dx) + f(dy * dy)));
    if (len < 1e-3) { dx = 0; dy = 1; } else { dx = f(dx / len); dy = f(dy / len); }
    const base = atan2f(-dx, dy);
    const first = fr.nVertices;
    for (let s = 0; s <= 6; s++) {
      const a = f(base + f(f(PI * s) / 6));
      pushVertex(fr, f(bx + f(r * cosf(a))), f(by + f(r * sinf(a))));
    }
    for (let s = 0; s <= 6; s++) {
      const a = f(f(base + PI) + f(f(PI * s) / 6));
      pushVertex(fr, f(ax + f(r * cosf(a))), f(ay + f(r * sinf(a))));
    }
    endPoly(fr, first);
  }

  function pushCircle(fr, cx, cy, r) {
    const first = fr.nVertices;
    for (let s = 0; s < 12; s++) {
      const a = f(f(TWO_PI * s) / 12);
      pushVertex(fr, f(cx + f(r * cosf(a))), f(cy + f(r * sinf(a))));
    }
    endPoly(fr, first);
  }

  function pushHeart(fr, x0, y0, s) {
    const first = fr.nVertices;
    for (let k = 0; k <= LOBE_ARC; k++) {
      const a = f(LOBE_FROM + f(f(LOBE_SPAN * k) / LOBE_ARC));
      pushVertex(fr, f(f(x0 + f(LOBE_X * s)) + f(f(LOBE_R * s) * cosf(a))), f(y0 + f(f(LOBE_R * s) * sinf(a))));
    }
    pushVertex(fr, f(x0 + f(SIDE_X * s)), f(y0 + f(SIDE_Y * s)));
    pushVertex(fr, x0, f(y0 + f(TIP_Y * s)));
    pushVertex(fr, f(x0 - f(SIDE_X * s)), f(y0 + f(SIDE_Y * s)));
    for (let k = 0; k <= LOBE_ARC; k++) {
      const a = f(LOBE_LEFT_FROM + f(f(LOBE_SPAN * k) / LOBE_ARC));
      pushVertex(fr, f(f(x0 - f(LOBE_X * s)) + f(f(LOBE_R * s) * cosf(a))), f(y0 + f(f(LOBE_R * s) * sinf(a))));
    }
    endPoly(fr, first);
  }

  // -- the rasterizer ----------------------------------------------------------------------
  const ex0 = new Float32Array(EDGE_MAX), ey0 = new Float32Array(EDGE_MAX), ey1 = new Float32Array(EDGE_MAX);
  const edxdy = new Float32Array(EDGE_MAX), edir = new Float32Array(EDGE_MAX);
  const enext = new Int16Array(EDGE_MAX);
  const rowHead = new Int16Array(H);
  const active = new Int16Array(EDGE_MAX);
  const acc = new Float32Array(W + 2);

  function mix(dst, c, w) {
    const r = ((((dst >> 11) & 31) * (32 - w) + ((c >> 11) & 31) * w) >> 5);
    const g = ((((dst >> 5) & 63) * (32 - w) + ((c >> 5) & 63) * w) >> 5);
    const b = (((dst & 31) * (32 - w) + (c & 31) * w) >> 5);
    return (r << 11) | (g << 5) | b;
  }

  const range = { xmin: 0, xmax: 0 };

  // The signed area an edge's piece within row `y` adds to the row's accumulator.
  function accumulate(e, y) {
    const ya = Math.max(y, ey0[e]);
    const yb = Math.min(y + 1, ey1[e]);
    const dy = f(yb - ya);
    if (dy <= 0) return;
    const x = f(ex0[e] + f(edxdy[e] * f(ya - ey0[e])));
    const xnext = f(x + f(edxdy[e] * dy));
    const d = f(dy * edir[e]);
    let x0 = x, x1 = xnext;
    if (x1 < x0) { x0 = xnext; x1 = x; }
    const x0floor = Math.floor(x0);
    const x0i = x0floor;
    const x1ceil = Math.ceil(x1);
    const x1i = x1ceil;
    if (x0i < range.xmin) range.xmin = x0i;
    if (x1i + 1 > range.xmax) range.xmax = x1i + 1;
    if (x1i <= x0i + 1) {
      const xmf = f(f(0.5 * f(x + xnext)) - x0floor);
      acc[x0i] = f(acc[x0i] + f(d - f(d * xmf)));
      acc[x0i + 1] = f(acc[x0i + 1] + f(d * xmf));
      return;
    }
    const s = f(1 / f(x1 - x0));
    const x0f = f(x0 - x0floor);
    const a0 = f(f(f(0.5 * s) * f(1 - x0f)) * f(1 - x0f));
    const x1f = f(f(x1 - x1ceil) + 1);
    const am = f(f(f(0.5 * s) * x1f) * x1f);
    acc[x0i] = f(acc[x0i] + f(d * a0));
    if (x1i === x0i + 2) {
      acc[x0i + 1] = f(acc[x0i + 1] + f(d * f(f(1 - a0) - am)));
    } else {
      const a1 = f(s * f(1.5 - x0f));
      acc[x0i + 1] = f(acc[x0i + 1] + f(d * f(a1 - a0)));
      for (let xi = x0i + 2; xi < x1i - 1; xi++) acc[xi] = f(acc[xi] + f(d * s));
      const a2 = f(a1 + f((x1i - x0i - 3) * s));
      acc[x1i - 1] = f(acc[x1i - 1] + f(d * f(f(1 - a2) - am)));
    }
    acc[x1i] = f(acc[x1i] + f(d * am));
  }

  // One layer of the frame into the pixels (one strip of `rows` rows from y0).
  function drawLayer(fr, layer, px, y0, rows) {
    const top = y0;
    const bottom = y0 + rows;
    for (let r = 0; r < rows; r++) rowHead[r] = -1;
    let nEdges = 0;
    let rowLo = rows, rowHi = -1;
    const p0 = fr.layerFirst[layer];
    const p1 = p0 + fr.layerN[layer];
    for (let p = p0; p < p1; p++) {
      const first = fr.polyFirst[p];
      const n = fr.polyN[p];
      for (let i = 0; i < n; i++) {
        const j = i + 1 < n ? i + 1 : 0;
        let ax = f(fr.vx[first + i] * UNIT_F), ay = f(fr.vy[first + i] * UNIT_F);
        let bx = f(fr.vx[first + j] * UNIT_F), by = f(fr.vy[first + j] * UNIT_F);
        if (ay === by) continue;
        let dir = 1;
        if (ay > by) {
          const tx = ax, ty = ay;
          ax = bx; ay = by; bx = tx; by = ty;
          dir = -1;
        }
        if (by <= top || ay >= bottom || nEdges >= EDGE_MAX) continue;
        ex0[nEdges] = ax; ey0[nEdges] = ay; ey1[nEdges] = by;
        edxdy[nEdges] = f(f(bx - ax) / f(by - ay));
        edir[nEdges] = dir;
        let row = Math.floor(ay) - y0;
        if (row < 0) row = 0;
        enext[nEdges] = rowHead[row];
        rowHead[row] = nEdges;
        if (row < rowLo) rowLo = row;
        let last = Math.ceil(by) - y0 - 1;
        if (last > rows - 1) last = rows - 1;
        if (last > rowHi) rowHi = last;
        nEdges++;
      }
    }
    if (nEdges === 0) return;
    const color = fr.layerColor[layer];
    const alpha = fr.layerAlpha[layer];
    let nActive = 0;
    for (let r = rowLo; r <= rowHi; r++) {
      const y = y0 + r;
      let m = 0;
      for (let k = 0; k < nActive; k++) {
        if (ey1[active[k]] > y) active[m++] = active[k];
      }
      nActive = m;
      for (let e = rowHead[r]; e >= 0; e = enext[e]) {
        if (nActive < EDGE_MAX) active[nActive++] = e;
      }
      if (nActive === 0) continue;
      range.xmin = W; range.xmax = 0;
      for (let k = 0; k < nActive; k++) accumulate(active[k], y);
      let xmin = range.xmin, xmax = range.xmax;
      if (xmin < 0) xmin = 0;
      if (xmax > W) xmax = W;
      let sum = 0;
      const row = y * W;
      for (let x = xmin; x < xmax; x++) {
        sum = f(sum + acc[x]);
        acc[x] = 0;
        let cov = Math.abs(sum);
        if (cov < ABSENT) continue;
        if (cov > 1) cov = 1;
        const w = Math.trunc(f(f(cov * alpha) + 0.5));
        if (w <= 0) continue;
        const i = row + x;
        px[i] = w >= 32 ? color : mix(px[i], color, w);
      }
      acc[xmax] = 0;
    }
  }

  // The whole screen (one strip of H rows) as RGB565 values.
  function drawAnime(fr, px) {
    px.fill(fr.background);
    acc.fill(0);
    for (let layer = 0; layer < fr.nLayers; layer++) drawLayer(fr, layer, px, 0, H);
    return px;
  }

  // -- the model (src/face_anime.cpp AnimeModel) ----------------------------------------------
  class AnimeModel {
    constructor() {
      this.from = newSet(); this.to = newSet(); this.cur = newSet(); this.gest = newSet();
      this.frame = newFrame();
      this.table = null;  // the style's drawings
      this.starts = 0; this.key = 0; this.t0 = 0; this.dur = 0;
      this.gesture = -1; this.gestureToward = false; this.primed = false;
      this.xs = new Float32Array(RING_MAX); this.ys = new Float32Array(RING_MAX);
    }

    static talkKey(style, level) {
      const shapes = f(T.styles[style].flap_shapes);
      if (shapes < 2) return 0;
      const q = Math.floor(f(f(level * f(shapes - 1)) + 0.5));
      return q <= 0 ? 0 : q >= shapes - 1 ? KEYS - 1 : 1;
    }

    retarget(durationMs, nowMs) {
      copySet(this.cur, this.from);
      copySet(gTarget, this.to);
      collapse(this.to, this.from);
      collapse(this.from, this.to);
      if (this.gestureToward) {
        collapse(this.to, this.gest);
        collapse(this.gest, this.to);
      }
      this.t0 = nowMs;
      this.dur = durationMs > 0 ? durationMs : 0;
      if (this.dur === 0) copySet(this.to, this.cur);
    }

    follow(model, nowMs) {
      const t = tableOf(model.style);
      if (t === null) return;  // a parametric style: the other renderer's frame
      const ps = model.shape();
      const key = AnimeModel.talkKey(t.style, model.level);
      const frame = model.shown * KEYS + key;
      if (!this.primed || t !== this.table) {
        // the first frame, or the style arrived: from another shapes style its drawing morphs
        // over the model's transition, from a parametric style it shows at once
        const morph = this.primed && this.table !== null;
        this.table = t;
        target(t, frame, model.intensity);
        if (morph) {
          if (this.gestureToward) load(t, t.gestures[this.gesture].toward, this.gest);
          this.retarget(model.dur, model.t0);
        } else {
          copySet(gTarget, this.cur);
          this.retarget(0, nowMs);
        }
        this.primed = true;
        this.starts = model.starts;
        this.key = key;
      } else if (model.starts !== this.starts) {
        target(t, frame, model.intensity);
        this.retarget(model.dur, model.t0);
        this.starts = model.starts;
        this.key = key;
      } else if (key !== this.key) {
        target(t, frame, model.intensity);
        this.retarget(f(t.talk_tween_ms), nowMs);
        this.key = key;
      }
      const elapsed = f(nowMs - this.t0);
      const e = this.dur > 0 ? ease(f(elapsed / this.dur)) : 1;
      blend(this.from, this.to, e, this.cur);

      const gesture = model.idleGesture;
      if (gesture !== this.gesture) {
        this.gesture = gesture;
        this.gestureToward = gesture >= 0 && t.gestures[gesture].toward >= 0;
        if (this.gestureToward) {
          load(t, t.gestures[gesture].toward, this.gest);
          collapse(this.gest, this.cur);
          collapse(this.from, this.gest);
          collapse(this.to, this.gest);
          blend(this.from, this.to, e, this.cur);
        }
      }
      this.build(model, ps, nowMs);
    }

    build(model, ps, nowMs) {
      const t = this.table;
      const cur = this.cur, gest = this.gest;
      const gesture = this.gesture;
      const amp = f(model.idleSign * model.idleAmp);
      const mag = Math.abs(amp);
      const toward = gesture >= 0 && this.gestureToward && mag > 0 ? f(mag * f(t.gestures[gesture].amount)) : 0;
      const lv = MAP.vertex[SLOT_LINE][0];
      const ln = SLOTS[SLOT_LINE].n;
      let x0 = cur.x[lv], x1 = x0, y0 = cur.y[lv], y1 = y0;
      for (let i = 1; i < ln; i++) {
        x0 = Math.min(x0, cur.x[lv + i]); x1 = Math.max(x1, cur.x[lv + i]);
        y0 = Math.min(y0, cur.y[lv + i]); y1 = Math.max(y1, cur.y[lv + i]);
      }
      x0 = f(x0 * UNIT_F); x1 = f(x1 * UNIT_F); y0 = f(y0 * UNIT_F); y1 = f(y1 * UNIT_F);
      const cx = f(0.5 * f(x0 + x1)), cy = f(0.5 * f(y0 + y1));
      const hw = Math.max(f(0.5 * f(x1 - x0)), 1);
      let sx = 1, sy = 1, dx = 0, dy = 0, tilt = 0, curl = 0;
      if (gesture >= 0 && mag > 0) {
        const g = t.gestures[gesture];
        sx = f(sx + f(f(f(g.sx) - 1) * mag));
        sy = f(sy + f(f(f(g.sy) - 1) * mag));
        dx = f(dx + f(f(g.dx) * amp));
        dy = f(dy + f(f(g.dy) * mag));
        tilt = f(tilt + f(f(g.tilt) * amp));
        curl = f(curl + f(f(g.curl) * amp));
      }
      const b = sinf(f(TWO_PI * model.breathePhase));
      const breathe = ps.p[P.breathe];
      sx = f(sx * f(1 + f(f(f(0.035) * breathe) * b)));
      sy = f(sy * f(1 + f(f(f(0.06) * breathe) * f(0.5 + f(0.5 * b)))));
      dx = f(dx + f(f(f(ps.p[P.drift] * f(1 / 3)) * f(G.shift_x_px)) * sinf(f(TWO_PI * model.driftPhase))));
      const wave = f(clampf(ps.p[P.wave], 0, 1) * f(G.wave_px));
      const waveN = ps.p[P.wave_n];
      const phase = ps.wavePhase;
      const wag = ps.p[P.wag];
      const tp = MAP.poly[SLOT_TONGUE][0];
      let ty0 = 0, th = 1, sway = 0;
      if (wag > 0 && cur.a[tp] > 0) {
        const tv = MAP.vertex[SLOT_TONGUE][0];
        const tn = SLOTS[SLOT_TONGUE].n;
        let ty1 = cur.y[tv];
        ty0 = ty1;
        for (let i = 1; i < tn; i++) {
          ty0 = Math.min(ty0, cur.y[tv + i]); ty1 = Math.max(ty1, cur.y[tv + i]);
        }
        ty0 = f(ty0 * UNIT_F);
        th = Math.max(f(f(ty1 * UNIT_F) - ty0), 1);
        sway = f(f(f(wag * 0.5) * th) * sinf(f(TWO_PI * model.wagPhase)));
      }

      const fr = this.frame;
      fr.nVertices = fr.nPolys = fr.nLayers = 0;
      fr.background = T.styles[t.style].rgb565.background;
      const xs = this.xs, ys = this.ys;
      for (let s = 0; s < SLOT_COUNT; s++) {
        const color = ps.color[SLOTS[s].element];
        for (let k = 0; k < SLOTS[s].max; k++) {
          const p = MAP.poly[s][k];
          const v = MAP.vertex[s][k];
          const n = SLOTS[s].n;
          let alpha = f(cur.a[p] / 255);
          let width = f(cur.w[p] * UNIT_F);
          if (toward > 0) {
            alpha = f(alpha + f(f(f(gest.a[p] / 255) - alpha) * toward));
            width = f(width + f(f(f(gest.w[p] * UNIT_F) - width) * toward));
          }
          if (alpha < ABSENT) continue;
          const follows = sway !== 0 && (s === SLOT_TONGUE || (cur.flags[p] & 1) !== 0);
          for (let i = 0; i < n; i++) {
            let x = f(cur.x[v + i] * UNIT_F), y = f(cur.y[v + i] * UNIT_F);
            if (toward > 0) {
              x = f(x + f(f(f(gest.x[v + i] * UNIT_F) - x) * toward));
              y = f(y + f(f(f(gest.y[v + i] * UNIT_F) - y) * toward));
            }
            const rx = f(x - cx), ry = f(y - cy);
            const u = f(rx / hw);
            let yy = f(f(f(f(cy + f(ry * sy)) + dy) + f(tilt * rx)) + f(f(curl * u) * u));
            let xx = f(f(cx + f(rx * sx)) + dx);
            if (wave > 0) yy = f(yy + f(wave * sinf(f(TWO_PI * f(f(f(0.5 * f(u + 1)) * waveN) - phase)))));
            if (follows) xx = f(xx + f(sway * clampf(f(f(y - ty0) / th), 0, 1)));
            xs[i] = xx;
            ys[i] = yy;
          }
          if (SLOTS[s].kind === "anchor") {
            const period = f(f(t.drool_s) * 1000);
            const kt = f((nowMs % Math.trunc(period)) / period);
            const stretch = kt < 0.85 ? ease(f(kt / f(0.85))) : f(1 - ease(f(f(kt - f(0.85)) / f(0.15))));
            const len = f(width * f(f(1.6) + f(f(4.5) * stretch)));
            const rd = f(width * f(f(0.55) + f(f(0.45) * stretch)));
            const yb = Math.min(f(ys[0] + len), f(f(f(H - f(G.margin_px)) - rd) - DROOL_RIM));
            if (beginLayer(fr, ps.color[E.lip], alpha)) {
              pushCapsule(fr, xs[0], ys[0], xs[0], f(yb - rd), f(DROOL_STEM_PX + DROOL_RIM));
              pushCircle(fr, xs[0], f(yb - rd), f(rd + DROOL_RIM));
              endLayer(fr);
            }
            if (beginLayer(fr, color, alpha)) {
              pushCapsule(fr, xs[0], ys[0], xs[0], f(yb - rd), DROOL_STEM_PX);
              pushCircle(fr, xs[0], f(yb - rd), rd);
              endLayer(fr);
            }
            continue;
          }
          if (!beginLayer(fr, color, alpha)) continue;
          if (SLOTS[s].kind === "stroke") pushStroke(fr, xs, ys, n, width);
          else pushPolygon(fr, xs, ys, n);
          endLayer(fr);
        }
      }
      // the overlays (the cheek pair, the shock's stripes): the table's keyframe drawn as it
      // is, over the mouth, at the opacity of the look's parameter; they neither morph nor
      // move with the gestures and the breath
      for (const o of t.overlays) {
        if (o.frame < 0) continue;
        const amount = clampf(ps.p[P[o.name]], 0, 1);
        if (amount < ABSENT) continue;
        for (const poly of t.frames[o.frame].polys) {
          const s = poly.slot;
          const n = SLOTS[s].n;
          if (SLOTS[s].kind === "anchor") continue;
          for (let i = 0; i < n; i++) {
            xs[i] = f(poly.points[2 * i] * UNIT_F);
            ys[i] = f(poly.points[2 * i + 1] * UNIT_F);
          }
          if (!beginLayer(fr, ps.color[SLOTS[s].element], f(f(SLOTS[s].alpha) * amount))) continue;
          if (SLOTS[s].kind === "stroke") pushStroke(fr, xs, ys, n, f(poly.width * f(1 / A.width_unit)));
          else pushPolygon(fr, xs, ys, n);
          endLayer(fr);
        }
      }
      if (ps.heart >= 0) {
        const kf = ps.heart;
        const heartPx = f(G.heart_px);
        const size = f(heartPx * f(f(0.55) + f(f(0.45) * ease(f(kf * 5)))));
        const rise = f(1 - f(f(1 - kf) * f(1 - kf)));
        let hx = f(f(f(f(x1 + 4) + f(f(0.6) * heartPx)) + f(12 * kf)) + f(5 * sinf(f(f(TWO_PI * f(1.5)) * kf))));
        let hy = f(f(cy - f(f(0.6) * heartPx)) - f(f(G.heart_rise_px) * rise));
        hx = Math.min(hx, f(f(W - f(G.margin_px)) - f(f(LOBE_X + LOBE_R) * size)));
        hy = Math.max(hy, f(f(G.margin_px) + f(LOBE_R * size)));
        const alpha = kf < 0.55 ? 1 : f(1 - f(f(kf - f(0.55)) / f(0.45)));
        if (beginLayer(fr, ps.color[E.heart], alpha)) {
          pushHeart(fr, hx, hy, size);
          endLayer(fr);
        }
      }
      if (ps.z >= 0) {
        const kz = ps.z;
        const size = f(Z_PX * f(f(0.6) + f(f(0.6) * kz)));
        let zx = f(f(x1 + 6) + f(16 * kz));
        let zy = f(f(y0 - 6) - f(Z_RISE_PX * ease(kz)));
        zx = Math.min(zx, f(f(f(W - f(G.margin_px)) - size) - Z_LINE_PX));
        zy = Math.max(zy, f(f(G.margin_px) + Z_LINE_PX));
        const alpha = kz < 0.5 ? 1 : f(1 - f(f(kz - f(0.5)) / f(0.5)));
        const lx = [zx, f(zx + size), zx, f(zx + size), zx, f(zx + size)];
        const ly = [zy, zy, f(zy + size), f(zy + size), f(zy + size), zy];
        if (beginLayer(fr, ps.color[E.lip], alpha)) {
          pushStroke(fr, lx, ly, 6, Z_LINE_PX);
          endLayer(fr);
        }
      }
    }
  }

  // The frame of an AnimeModel (after follow) as the screen's RGB565 values.
  function render(anime, px) {
    return drawAnime(anime.frame, px || new Uint16Array(W * H));
  }

  const api = { TABLES, KEYS, AnimeModel, drawAnime, render, tableOf };
  if (typeof module !== "undefined") module.exports = api;
  else root.Anime = api;
})(typeof window !== "undefined" ? window : globalThis);
