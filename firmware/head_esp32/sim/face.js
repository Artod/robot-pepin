// The mouth's model and renderer: a line-by-line port of src/face_model.cpp and
// src/face_render.cpp, so the simulator draws what the head will. The table is face_table.js,
// generated from config/face.json. Math.fround keeps the arithmetic in float32 like the ESP32's;
// tests/unit/test_face_parity.py renders the same shapes with both and compares the pixels.
"use strict";

(function (root) {
  const T = typeof FACE_TABLE !== "undefined" ? FACE_TABLE : require("./face_table.js");
  const f = Math.fround;
  const P = {};
  T.params.forEach((p, i) => { P[p.name] = i; });
  const N = T.params.length;
  const W = T.screen.width;
  const H = T.screen.height;
  const G = T.geometry;
  const TM = T.timing;
  const C = T.rgb565;
  const NEUTRAL = T.expressions.findIndex((e) => e.name === "neutral");
  const TWO_PI = f(6.28318530718);
  const BITE_GAP_PX = 1.0;
  const SAMPLES_MAX = 2 * W + 3;
  const DISC_MAX = 2 * 8 + 2;

  const clampf = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const ease = (k) => { k = clampf(k, 0, 1); return f(k * k * f(3 - 2 * k)); };
  const sinf = (x) => f(Math.sin(x));

  // -- the model (face_model.cpp) ------------------------------------------------------------
  class FaceModel {
    constructor() {
      const n = T.expressions[NEUTRAL].values;
      this.from = Float32Array.from(n);
      this.to = Float32Array.from(n);
      this.cur = Float32Array.from(n);
      this.t0 = 0; this.dur = 0; this.expression = NEUTRAL;
      this.levelTarget = 0; this.level = 0; this.levelAt = 0;
      this.last = 0; this.started = false;
      this.wavePhase = 0; this.breathePhase = 0; this.driftPhase = 0;
    }

    setExpression(id, intensity, transitionMs, nowMs) {
      if (id < 0 || id >= T.expressions.length) return;
      intensity = clampf(intensity, 0, 1);
      const target = new Float32Array(N);
      const n = T.expressions[NEUTRAL].values;
      const e = T.expressions[id].values;
      for (let i = 0; i < N; i++) target[i] = f(n[i] + f(f(e[i] - n[i]) * intensity));
      this.expression = id;
      this.setTarget(target, transitionMs, nowMs);
    }

    setTarget(target, transitionMs, nowMs) {
      this.update(nowMs);
      for (let i = 0; i < N; i++) {
        this.from[i] = this.cur[i];
        this.to[i] = clampf(target[i], T.params[i].min, T.params[i].max);
      }
      this.t0 = nowMs;
      this.dur = transitionMs > 0 ? transitionMs : 0;
      if (this.dur === 0) for (let i = 0; i < N; i++) this.cur[i] = this.to[i];
    }

    setMouth(level, nowMs) { this.levelTarget = clampf(level, 0, 1); this.levelAt = nowMs; }

    update(nowMs) {
      if (!this.started) { this.started = true; this.last = nowMs; }
      let dt = f((nowMs - this.last) / 1000);
      dt = clampf(dt, 0, 0.25);
      this.last = nowMs;
      const e = this.dur > 0 ? ease(f((nowMs - this.t0) / this.dur)) : 1;
      for (let i = 0; i < N; i++) this.cur[i] = f(this.from[i] + f(f(this.to[i] - this.from[i]) * e));
      if (nowMs - this.levelAt > TM.mouth_stale_s * 1000) this.levelTarget = 0;
      const tau = this.levelTarget > this.level ? TM.mouth_attack_s : TM.mouth_release_s;
      if (dt > 0) this.level = f(this.level + f(f(this.levelTarget - this.level) * f(1 - Math.exp(-dt / tau))));
      this.wavePhase = f(this.wavePhase + f(this.cur[P.wave_speed] * dt));
      this.wavePhase = f(this.wavePhase - Math.floor(this.wavePhase));
      this.breathePhase = f(this.breathePhase + f(G.breathe_hz * dt));
      this.breathePhase = f(this.breathePhase - Math.floor(this.breathePhase));
      this.driftPhase = f(this.driftPhase + f(G.drift_hz * dt));
      this.driftPhase = f(this.driftPhase - Math.floor(this.driftPhase));
    }

    shape() {
      const c = this.cur;
      const p = Float32Array.from(c);
      const b = sinf(TWO_PI * this.breathePhase);
      const br = c[P.breathe];
      p[P.width] = f(f(c[P.width] * f(1 + f(f(0.035 * br) * b))) * f(1 - f(G.speech_narrow * this.level)));
      const open = f(f(c[P.open] + f(f(br * 0.045) * f(0.5 + f(0.5 * b)))) + f(G.speech_open * this.level));
      p[P.open] = clampf(open, 0, 1);
      p[P.shift_x] = f(c[P.shift_x] + f(f(c[P.drift] * f(1 / 3)) * sinf(TWO_PI * this.driftPhase)));
      return { p, wavePhase: this.wavePhase };
    }
  }

  // -- the renderer (face_render.cpp) ---------------------------------------------------------
  function triangle(ph) {
    const fr = ph - Math.floor(ph);
    if (fr < 0.25) return 4 * fr;
    if (fr < 0.75) return 2 - 4 * fr;
    return 4 * fr - 4;
  }

  function geometry(shape) {
    const p = shape.p;
    const g = {};
    g.half_w = f(0.5 * W * clampf(p[P.width], 0.02, 1));
    g.r = f(0.5 * f(G.thick_min_px + f((G.thick_max_px - G.thick_min_px) * clampf(p[P.thick], 0, 1))));
    g.r = clampf(g.r, 0.5, f(0.5 * (DISC_MAX - 2)));
    let room = f(0.5 * W - g.half_w - g.r - 2);
    if (room < 0) room = 0;
    g.cx = f(0.5 * W + clampf(f(p[P.shift_x] * G.shift_x_px), -room, room));
    g.cy = f(0.5 * H + f(p[P.shift_y] * G.shift_y_px));
    g.open_px = f(clampf(p[P.open], 0, 1) * G.max_open_px);
    g.upper = clampf(p[P.upper], 0, 1);
    g.pexp = clampf(p[P.corner_exp], 1, 12);
    g.smile_px = f(p[P.smile] * G.smile_px);
    g.asym_px = f(p[P.asym] * G.asym_px);
    g.wave_px = f(clampf(p[P.wave], 0, 1) * G.wave_px);
    g.wave_n = p[P.wave_n];
    g.sharp = clampf(p[P.wave_sharp], 0, 1);
    g.phase = shape.wavePhase;
    g.mid = (u) => {
      const u2 = f(u * u);
      const line = f(f(g.cy - f(g.smile_px * u2)) - f(g.asym_px * u));
      if (g.wave_px === 0) return line;
      const taper = f(1 - f(f(u2 * u2) * u2));
      const ph = f(f(f(f(u + 1) * 0.5) * g.wave_n) - g.phase);
      const wave = f(f(f(1 - g.sharp) * sinf(TWO_PI * ph)) + f(g.sharp * triangle(ph)));
      return f(line + f(f(g.wave_px * taper) * wave));
    };
    g.profile = (u) => {
      const a = Math.abs(u);
      if (a >= 1) return 0;
      if (g.pexp === 2) return f(Math.sqrt(f(1 - f(a * a))));
      return f(Math.pow(f(1 - f(Math.pow(a, g.pexp))), f(1 / g.pexp)));
    };
    return g;
  }

  function layout(shape) {
    const g = geometry(shape);
    const teeth = clampf(shape.p[P.teeth], 0, 1);
    const left = f(g.cx - g.half_w);
    const right = f(g.cx + g.half_w);
    const sx = [], st = [], sb = [];
    const k0 = Math.ceil(f(left * 2));
    const k1 = Math.floor(f(right * 2));
    for (let k = k0 - 1; k <= k1 + 1 && sx.length < SAMPLES_MAX; k++) {
      let xs = f(0.5 * k);
      if (k === k0 - 1) xs = left;
      if (k === k1 + 1) xs = right;
      const u = clampf(f(f(xs - g.cx) / g.half_w), -1, 1);
      const m = g.mid(u);
      const o = g.open_px > 0 ? f(g.open_px * g.profile(u)) : 0;
      sx.push(xs);
      st.push(f(m - f(o * g.upper)));
      sb.push(f(m + f(o * f(1 - g.upper))));
    }
    const n = sx.length;
    let x0 = Math.floor(f(left - g.r));
    let x1 = Math.ceil(f(right + g.r));
    x0 = x0 < 0 ? 0 : x0;
    x1 = x1 > W - 1 ? W - 1 : x1;
    const r2 = f(g.r * g.r);
    const disc = [];
    const mmax = Math.trunc(f(2 * g.r));
    for (let m = 0; m <= mmax; m++) disc.push(f(Math.sqrt(Math.max(f(r2 - f(0.25 * (m * m))), 0))));
    const cols = [];
    let first = 0;
    for (let x = 0; x < W; x++) {
      const c = { active: false, gap: false, ot: 0, ob: 0, it: 0, ib: 0, up: 0, lo: 0 };
      cols.push(c);
      if (x < x0 || x > x1) continue;
      const xc = f(x + 0.5);
      let ot = 1e9, it = -1e9, ob = -1e9, ib = 1e9;
      while (first < n && sx[first] < f(xc - g.r)) first++;
      for (let i = first; i < n && sx[i] <= f(xc + g.r); i++) {
        const dx = f(sx[i] - xc);
        let hh;
        if (i === 0 || i === n - 1) {
          hh = f(Math.sqrt(Math.max(f(r2 - f(dx * dx)), 0)));
        } else {
          const m = Math.trunc(f(f(Math.abs(dx) * 2) + 0.5));
          hh = m <= mmax ? disc[m] : 0;
        }
        ot = Math.min(ot, f(st[i] - hh));
        it = Math.max(it, f(st[i] + hh));
        ob = Math.max(ob, f(sb[i] + hh));
        ib = Math.min(ib, f(sb[i] - hh));
        c.active = true;
      }
      if (!c.active) continue;
      c.ot = ot; c.ob = ob; c.it = it; c.ib = ib;
      const cavity = f(ib - it);
      if (cavity > 0 && teeth > 0) {
        let up = f(f(clampf(f(2 * teeth), 0, 1) * 0.5) * cavity);
        let lo = f(f(clampf(f(f(2 * teeth) - 1), 0, 1) * 0.5) * cavity);
        if (lo > 0) {
          up = Math.max(0, Math.min(up, f(f(0.5 * cavity) - BITE_GAP_PX)));
          lo = Math.max(0, Math.min(lo, f(f(0.5 * cavity) - BITE_GAP_PX)));
        }
        c.up = up; c.lo = lo;
        const m = f(f(Math.abs(f(xc - g.cx)) + f(0.5 * G.tooth_px)) % G.tooth_px);
        c.gap = m < G.tooth_gap_px;
      }
    }
    return { cols, x0, x1 };
  }

  function mix(dst, c, w) {
    const r = ((((dst >> 11) & 31) * (32 - w)) + (((c >> 11) & 31) * w)) >> 5;
    const g = ((((dst >> 5) & 63) * (32 - w)) + (((c >> 5) & 63) * w)) >> 5;
    const b = (((dst & 31) * (32 - w)) + ((c & 31) * w)) >> 5;
    return (r << 11) | (g << 5) | b;
  }

  function blend(px, x, y, coverage, c) {
    const w = Math.floor(f(coverage * 32) + 0.5);
    if (w <= 0) return;
    const i = y * W + x;
    px[i] = w >= 32 ? c : mix(px[i], c, w);
  }

  function span(px, x, ya, yb, c) {
    ya = Math.max(ya, 0);
    yb = Math.min(yb, H);
    if (yb <= ya) return;
    const ia = Math.floor(ya);
    const ib = Math.floor(yb);
    if (ia === ib) { blend(px, x, ia, f(yb - ya), c); return; }
    blend(px, x, ia, f(ia + 1 - ya), c);
    for (let y = ia + 1; y < ib; y++) px[y * W + x] = c;
    if (yb > ib && ib < H) blend(px, x, ib, f(yb - ib), c);
  }

  // The whole screen (one strip of H rows) as RGB565 values.
  function draw(lay, px) {
    px.fill(C.background);
    for (let x = lay.x0; x <= lay.x1; x++) {
      const c = lay.cols[x];
      if (!c.active) continue;
      span(px, x, c.ot, c.ob, C.lip);
      if (c.ib <= c.it) continue;
      span(px, x, c.it, c.ib, C.cavity);
      if (c.gap) continue;
      if (c.up > 0) span(px, x, c.it, f(c.it + c.up), C.teeth);
      if (c.lo > 0) span(px, x, f(c.ib - c.lo), c.ib, C.teeth);
    }
    return px;
  }

  function render(shape, px) {
    return draw(layout(shape), px || new Uint16Array(W * H));
  }

  // RGB565 pixels into a canvas ImageData's RGBA bytes.
  function toRGBA(px, rgba) {
    for (let i = 0; i < px.length; i++) {
      const v = px[i];
      const r = (v >> 11) & 31, g = (v >> 5) & 63, b = v & 31;
      rgba[4 * i] = (r << 3) | (r >> 2);
      rgba[4 * i + 1] = (g << 2) | (g >> 4);
      rgba[4 * i + 2] = (b << 3) | (b >> 2);
      rgba[4 * i + 3] = 255;
    }
    return rgba;
  }

  const api = { T, P, W, H, NEUTRAL, FaceModel, layout, draw, render, toRGBA, ease };
  if (typeof module !== "undefined") module.exports = api;
  else root.Face = api;
})(typeof window !== "undefined" ? window : globalThis);
