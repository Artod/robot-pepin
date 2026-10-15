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
  const PI = f(3.14159265359);
  const QUIET_LEVEL = f(0.01);
  const WAVE_SETTLE_S = f(0.3);
  const IDLE = T.idle;
  const BITE_GAP_PX = 1.0;
  const SAMPLES_MAX = 2 * W + 3;
  const DISC_MAX = 2 * 8 + 2;
  const EL = T.elements.length;
  const E = {};
  T.elements.forEach((name, i) => { E[name] = i; });
  const BOW_HALF = f(0.45);
  const DIMPLE_ARC = f(0.95);
  const DIMPLE_POINTS = 7;
  const HOOK_POINTS = 5;
  const HOOK_ARC = f(1.658);
  const BULB_DIP = f(0.6);
  const DECO_MAX = 768;
  const DECO_BACKGROUND = 0xff;
  const LOBE_X = f(0.45), LOBE_R = f(0.5), TIP_Y = f(1.2), SIDE_X = f(0.8126), SIDE_Y = f(0.3443);
  const SIDE_NX = f(0.725), SIDE_NY = f(0.689), TIP_INSET = f(1.45);
  const HEART_HEIGHT = f(TIP_Y + LOBE_R);
  const DROOL_STEM_PX = f(1.2), DROOL_DROP_PX = f(3.6);
  const LOBE_COS = [0, 0.5, 0.8660254, 1, 0.8660254, 0.5, 0].map(f);
  const LOBE_SIN = [-1, -0.8660254, -0.5, 0, 0.5, 0.8660254, 1].map(f);
  const LOBE_STEPS = 6;
  const EFFECT_BITS = { heart: 1, z: 2 };
  const g32 = (name) => f(G[name]);  // a geometry number as the firmware's float
  // a style's numbers as the firmware's floats, and its background as RGB565
  const STYLES = [];
  const restyle = () => {
    STYLES.length = 0;
    for (const s of T.styles) {
      STYLES.push({
        thick_min_px: f(s.thick_min_px), thick_max_px: f(s.thick_max_px),
        tooth_gap_px: f(s.tooth_gap_px), shade: f(s.shade), teeth_line_px: f(s.teeth_line_px),
        deco_line_px: f(s.deco_line_px), pucker_line: f(s.pucker_line),
        tongue_w_px: f(s.tongue_w_px), tongue_shine: f(s.tongue_shine), patch_x: f(s.patch_x),
        teeth_span: f(s.teeth_span), flap_shapes: f(s.flap_shapes),
        background: s.rgb565.background,
      });
    }
  };
  restyle();
  const styleOf = (s) => (s >= 0 && s < STYLES.length ? s : T.style);

  const clampf = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const ease = (k) => { k = clampf(k, 0, 1); return f(k * k * f(3 - 2 * k)); };
  const sinf = (x) => f(Math.sin(x));
  const cosf = (x) => f(Math.cos(x));
  const rgbOf = (c) => [(c >>> 16) & 255, (c >>> 8) & 255, c & 255];
  const rgb565 = (c) => {
    const r = Math.trunc(clampf(c[0], 0, 255)), g = Math.trunc(clampf(c[1], 0, 255));
    const b = Math.trunc(clampf(c[2], 0, 255));
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
  };
  const look = (id, style) => T.expressions[id].looks[style];

  // -- the model (face_model.cpp) ------------------------------------------------------------
  class FaceModel {
    constructor() {
      this.style = T.style;
      const n = look(NEUTRAL, this.style).values;
      this.from = Float32Array.from(n);
      this.to = Float32Array.from(n);
      this.cur = Float32Array.from(n);
      const nc = look(NEUTRAL, this.style).colors;
      this.cfrom = nc.map(rgbOf);
      this.cto = nc.map(rgbOf);
      this.ccur = nc.map(rgbOf);
      this.shown = NEUTRAL; this.intensity = 1; this.heartOn = false; this.heartT0 = 0;
      this.zOn = false; this.zT0 = 0; this.starts = 0;
      this.t0 = 0; this.dur = 0; this.expression = NEUTRAL;
      this.levelTarget = 0; this.level = 0; this.levelAt = 0;
      this.last = 0; this.started = false;
      this.wavePhase = 0; this.breathePhase = 0; this.driftPhase = 0; this.wagPhase = 0;
      this.rng = 0x2545F491;
      this.idleScheduled = false; this.idleNext = 0; this.idleGesture = -1; this.idleT0 = 0;
      this.idleSign = 1; this.idleFading = false; this.idleFadeAt = 0; this.idleAmp = 0;
    }

    seed(s) { this.rng = (s >>> 0) !== 0 ? s >>> 0 : 0x2545F491; }

    nextRandom() {
      let x = this.rng;
      x = (x ^ (x << 13)) >>> 0;
      x = (x ^ (x >>> 17)) >>> 0;
      x = (x ^ (x << 5)) >>> 0;
      this.rng = x;
      return x;
    }

    uniform() { return f((this.nextRandom() >>> 8) * (1 / 16777216)); }

    scheduleIdle(nowMs) {
      this.idleScheduled = true;
      const lo = f(IDLE.every_s[0]), hi = f(IDLE.every_s[1]);
      const span = f(lo + f(f(hi - lo) * this.uniform()));
      this.idleNext = nowMs + Math.trunc(f(span * 1000));
    }

    idleAllowed(nowMs) {
      const id = this.expression;
      if (id < 0 || id >= 32 || !IDLE.expressions.includes(id)) return false;
      const settleMs = f(f(IDLE.settle_s) * 1000);
      if (f(nowMs - this.t0) < f(settleMs + this.dur)) return false;
      if (f(nowMs - this.levelAt) < settleMs) return false;
      return this.level < QUIET_LEVEL;
    }

    updateIdle(nowMs) {
      this.idleAmp = 0;
      if (!IDLE.enabled) return;
      if (!this.idleScheduled) this.scheduleIdle(nowMs);
      const allowed = this.idleAllowed(nowMs);
      const G = IDLE.gestures;
      if (this.idleGesture >= 0) {
        if (!allowed && !this.idleFading) { this.idleFading = true; this.idleFadeAt = nowMs; }
        const u = f(f(nowMs - this.idleT0) / f(f(G[this.idleGesture].duration_s) * 1000));
        let fade = 1;
        if (this.idleFading) fade = f(1 - ease(f(f(nowMs - this.idleFadeAt) / f(f(IDLE.fade_s) * 1000))));
        if (u >= 1 || fade <= 0) { this.idleGesture = -1; this.scheduleIdle(nowMs); return; }
        const sv = sinf(f(PI * clampf(u, 0, 1)));
        this.idleAmp = f(f(sv * sv) * fade);
        return;
      }
      if (nowMs - this.idleNext < 0) return;
      if (!allowed) { this.scheduleIdle(nowMs); return; }
      // the gestures of this style, by weight
      const mine = (g) => g.styles.includes(this.style);
      let total = 0;
      for (const g of G) if (mine(g)) total = f(total + f(g.weight));
      if (total <= 0) { this.scheduleIdle(nowMs); return; }
      let r = f(this.uniform() * total);
      let g = 0;
      for (let k = 0; k < G.length; k++) {
        if (!mine(G[k])) continue;
        g = k;
        if (r < f(G[k].weight)) break;
        r = f(r - f(G[k].weight));
      }
      this.idleGesture = g; this.idleT0 = nowMs; this.idleFading = false;
      this.idleSign = G[g].mirror && (this.nextRandom() & 1) ? -1 : 1;
    }

    // The target of `shown` at `intensity` in the style in force: its values and colours.
    lookOf(shown, intensity) {
      const target = new Float32Array(N);
      const n = look(NEUTRAL, this.style).values;
      const e = look(shown, this.style).values;
      for (let i = 0; i < N; i++) target[i] = f(n[i] + f(f(e[i] - n[i]) * intensity));
      const colors = [];
      for (let c = 0; c < EL; c++) {
        const nc = rgbOf(look(NEUTRAL, this.style).colors[c]);
        const xc = rgbOf(look(shown, this.style).colors[c]);
        colors.push(nc.map((v, k) => f(v + f(f(xc[k] - v) * intensity))));
      }
      return [target, colors];
    }

    setExpression(id, intensity, transitionMs, nowMs) {
      if (id < 0 || id >= T.expressions.length) return;
      intensity = f(clampf(intensity, 0, 1));
      let shown = id;
      const variants = T.expressions[id].variants;
      if (variants.length > 0) {  // itself or one of its variants, alike
        const r = this.nextRandom() % (variants.length + 1);
        if (r > 0) shown = variants[r - 1];
      }
      const [target, colors] = this.lookOf(shown, intensity);
      this.expression = id;
      this.shown = shown;
      this.intensity = intensity;
      this.start(target, colors, transitionMs, nowMs);
      const effects = intensity > 0 ? this.effectsOf(shown) : 0;
      if ((effects & EFFECT_BITS.heart) && !this.heartOn) {
        this.heartOn = true;
        this.heartT0 = nowMs;
      }
      this.startZ(effects, nowMs);
    }

    // The z with the look shown's effects: started (now, then every z_every_s) or stopped.
    startZ(effects, nowMs) {
      if ((effects & EFFECT_BITS.z) && !this.zOn) {
        this.zOn = true;
        this.zT0 = nowMs;
      } else if (!(effects & EFFECT_BITS.z)) {
        this.zOn = false;  // the sleeper woke: no z lingers over the new face
      }
    }

    // The look's effects in the style in force, as bits.
    effectsOf(shown) {
      let bits = 0;
      for (const name of look(shown, this.style).effects) bits |= EFFECT_BITS[name];
      return bits;
    }

    setStyle(style, nowMs) {
      if (style < 0 || style >= STYLES.length || style === this.style) return;
      this.style = style;
      const [target, colors] = this.lookOf(this.shown, this.intensity);
      this.start(target, colors, f(TM.transition_ms), nowMs);
      this.startZ(this.intensity > 0 ? this.effectsOf(this.shown) : 0, nowMs);
    }

    setTarget(target, transitionMs, nowMs) {
      this.start(target, this.cto.map((c) => c.slice()), transitionMs, nowMs);
    }

    start(target, colors, transitionMs, nowMs) {
      this.update(nowMs);
      for (let i = 0; i < N; i++) {
        this.from[i] = this.cur[i];
        this.to[i] = clampf(target[i], T.params[i].min, T.params[i].max);
      }
      for (let c = 0; c < EL; c++) {
        this.cfrom[c] = this.ccur[c].slice();
        this.cto[c] = colors[c].slice();
      }
      this.t0 = nowMs;
      this.dur = transitionMs > 0 ? transitionMs : 0;
      this.starts = (this.starts + 1) >>> 0;
      if (this.dur === 0) {
        for (let i = 0; i < N; i++) this.cur[i] = this.to[i];
        for (let c = 0; c < EL; c++) this.ccur[c] = this.cto[c].slice();
      }
    }

    setMouth(level, nowMs) { this.levelTarget = clampf(level, 0, 1); this.levelAt = nowMs; }

    update(nowMs) {
      if (!this.started) { this.started = true; this.last = nowMs; }
      let dt = f((nowMs - this.last) / 1000);
      dt = clampf(dt, 0, 0.25);
      this.last = nowMs;
      const e = this.dur > 0 ? ease(f((nowMs - this.t0) / this.dur)) : 1;
      for (let i = 0; i < N; i++) this.cur[i] = f(this.from[i] + f(f(this.to[i] - this.from[i]) * e));
      for (let c = 0; c < EL; c++) {
        for (let k = 0; k < 3; k++) {
          this.ccur[c][k] = f(this.cfrom[c][k] + f(f(this.cto[c][k] - this.cfrom[c][k]) * e));
        }
      }
      if (this.heartOn && f(nowMs - this.heartT0) >= f(f(TM.heart_s) * 1000)) this.heartOn = false;
      // the z flies again every z_every_s while the look shown has it; it stops with the look
      if (this.zOn) {
        const wanted = this.intensity > 0 && (this.effectsOf(this.shown) & EFFECT_BITS.z) !== 0;
        const since = f(nowMs - this.zT0);
        if (!wanted && since >= f(f(TM.z_s) * 1000)) {
          this.zOn = false;
        } else if (since >= f(f(TM.z_every_s) * 1000)) {
          if (wanted) this.zT0 = (this.zT0 + Math.trunc(f(f(TM.z_every_s) * 1000))) >>> 0;
          else this.zOn = false;
        }
      }
      if (nowMs - this.levelAt > TM.mouth_stale_s * 1000) this.levelTarget = 0;
      const tau = this.levelTarget > this.level ? TM.mouth_attack_s : TM.mouth_release_s;
      if (dt > 0) this.level = f(this.level + f(f(this.levelTarget - this.level) * f(1 - Math.exp(-dt / tau))));
      if (this.cur[P.wave_speed] !== 0) {
        this.wavePhase = f(this.wavePhase + f(this.cur[P.wave_speed] * dt));
      } else if (this.cur[P.wave] <= 0) {
        this.wavePhase = 0;  // no wave shows: home at once
      } else if (dt > 0) {  // a standing wave settles home
        const home = this.wavePhase < 0.5 ? 0 : 1;
        this.wavePhase = f(this.wavePhase + f(f(home - this.wavePhase) * f(1 - Math.exp(-dt / WAVE_SETTLE_S))));
        if (Math.abs(f(home - this.wavePhase)) < f(1e-3)) this.wavePhase = home;
      }
      this.wavePhase = f(this.wavePhase - Math.floor(this.wavePhase));
      this.breathePhase = f(this.breathePhase + f(G.breathe_hz * dt));
      this.breathePhase = f(this.breathePhase - Math.floor(this.breathePhase));
      this.driftPhase = f(this.driftPhase + f(G.drift_hz * dt));
      this.driftPhase = f(this.driftPhase - Math.floor(this.driftPhase));
      this.wagPhase = f(this.wagPhase + f(G.wag_hz * dt));
      this.wagPhase = f(this.wagPhase - Math.floor(this.wagPhase));
      this.updateIdle(nowMs);
    }

    shape() {
      const c = this.cur;
      const p = Float32Array.from(c);
      const b = sinf(TWO_PI * this.breathePhase);
      const br = c[P.breathe];
      // the speech level as the style takes it: continuous, or the anime lip flap's few shapes
      let level = this.level;
      const shapes = STYLES[this.style].flap_shapes;
      if (shapes >= 2) level = f(Math.floor(f(f(this.level * f(shapes - 1)) + 0.5)) / f(shapes - 1));
      p[P.width] = f(f(c[P.width] * f(1 + f(f(0.035 * br) * b))) * f(1 - f(G.speech_narrow * level)));
      const open = f(f(c[P.open] + f(f(br * 0.045) * f(0.5 + f(0.5 * b)))) + f(G.speech_open * level));
      p[P.open] = clampf(open, 0, 1);
      p[P.shift_x] = f(c[P.shift_x] + f(f(c[P.drift] * f(1 / 3)) * sinf(TWO_PI * this.driftPhase)));
      if (c[P.wag] > 0) {
        const lean = f(c[P.tongue_lean] + f(f(c[P.wag] * 0.5) * sinf(TWO_PI * this.wagPhase)));
        p[P.tongue_lean] = clampf(lean, T.params[P.tongue_lean].min, T.params[P.tongue_lean].max);
      }
      if (this.idleGesture >= 0 && this.idleAmp > 0) {
        const k = f(this.idleSign * this.idleAmp);
        const d = IDLE.gestures[this.idleGesture].deltas;
        for (let i = 0; i < N; i++) {
          if (d[i] !== 0) p[i] = clampf(f(p[i] + f(f(d[i]) * k)), T.params[i].min, T.params[i].max);
        }
      }
      let heart = -1;
      if (this.heartOn) heart = clampf(f(f(this.last - this.heartT0) / f(f(TM.heart_s) * 1000)), 0, 1);
      let z = -1;
      if (this.zOn) {
        const k = f(f(this.last - this.zT0) / f(f(TM.z_s) * 1000));
        if (k < 1) z = clampf(k, 0, 1);
      }
      return { p, wavePhase: this.wavePhase, heart, z, color: this.ccur.map(rgb565), style: this.style };
    }
  }

  // -- the renderer (face_render.cpp) ---------------------------------------------------------
  function triangle(ph) {
    const fr = ph - Math.floor(ph);
    if (fr < 0.25) return 4 * fr;
    if (fr < 0.75) return 2 - 4 * fr;
    return 4 * fr - 4;
  }

  function geometry(shape, st) {
    const p = shape.p;
    const g = {};
    g.half_w = f(0.5 * W * clampf(p[P.width], 0.02, 1));
    g.pucker = clampf(p[P.pucker], 0, 1);  // the line shrinks into the pucker
    g.lips = clampf(p[P.heart_lips], 0, 1);  // and into the heart
    g.shrink = f(f(1 - g.pucker) * f(1 - g.lips));
    if (g.shrink < 1) g.half_w = Math.max(f(g.half_w * g.shrink), 1);
    g.r = f(0.5 * f(st.thick_min_px + f(f(st.thick_max_px - st.thick_min_px) * clampf(p[P.thick], 0, 1))));
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
    g.bow_px = f(clampf(p[P.bow], 0, 1) * G.bow_px);
    g.bulb = clampf(p[P.bulb], 0, 1);
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
      let pr;
      if (g.pexp === 2) pr = f(Math.sqrt(f(1 - f(a * a))));
      else pr = f(Math.pow(f(1 - f(Math.pow(a, g.pexp))), f(1 / g.pexp)));
      if (g.bulb <= 0) return pr;
      const rho = Math.min(0.5, f(g.open_px / f(2 * g.half_w)));
      let bone;
      if (a > f(1 - rho)) {
        const d = f(f(a - f(1 - rho)) / rho);
        bone = f(Math.sqrt(Math.max(f(1 - f(d * d)), 0)));
      } else {
        const t = f(a / f(1 - rho));
        bone = f(1 - f(BULB_DIP * f(1 - f(t * t))));
      }
      return f(pr + f(f(bone - pr) * g.bulb));
    };
    g.lower = (u) => {
      const o = g.open_px > 0 ? f(g.open_px * g.profile(u)) : 0;
      return f(g.mid(u) + f(o * f(1 - g.upper)));
    };
    g.tangent = (u) => {
      const slope = f(f(f(f(-2 * g.smile_px) * u) - g.asym_px) / g.half_w);
      const x = u, y = f(u * slope);
      const n = f(Math.sqrt(f(f(x * x) + f(y * y))));
      return [f(x / n), f(y / n)];
    };
    return g;
  }

  // Cupid's bow: how far the upper lip dips at u, 1 at the middle, 0 from |u| = BOW_HALF.
  function bow(u) {
    const a = f(u / BOW_HALF);
    if (a <= -1 || a >= 1) return 0;
    const k = f(1 - f(a * a));
    return f(k * k);
  }

  // -- the decorations (face_render.cpp) ------------------------------------------------------
  function push(out, x, top, bot, element, alpha) {
    if (x < 0 || x >= W || !(bot > top) || alpha === 0 || out.deco.length >= DECO_MAX) return;
    out.deco.push({ x, top, bot, element, alpha });
  }

  const opacity = (v) => Math.trunc(f(f(clampf(v, 0, 1) * 32) + 0.5));

  function capsule(ax, ay, bx, by, r, xc) {
    let lo = 1e9, hi = -1e9;
    const ex = [ax, bx], ey = [ay, by];
    for (let k = 0; k < 2; k++) {
      const d = f(xc - ex[k]);
      if (Math.abs(d) <= r) {
        const h = f(Math.sqrt(Math.max(f(f(r * r) - f(d * d)), 0)));
        lo = Math.min(lo, f(ey[k] - h));
        hi = Math.max(hi, f(ey[k] + h));
      }
    }
    const dx = f(bx - ax), dy = f(by - ay);
    const len2 = f(f(dx * dx) + f(dy * dy));
    if (len2 > 0) {
      const len = f(Math.sqrt(len2));
      const ux = f(xc - ax);
      let ya = -1e9, yb = 1e9;
      const along = f(ux * dx);
      if (Math.abs(dy) > f(1e-6)) {
        let y0 = f(ay + f(f(0 - along) / dy));
        let y1 = f(ay + f(f(len2 - along) / dy));
        if (y0 > y1) [y0, y1] = [y1, y0];
        ya = Math.max(ya, y0);
        yb = Math.min(yb, y1);
      } else if (along < 0 || along > len2) {
        yb = ya;
      }
      const across = f(ux * dy);
      if (Math.abs(dx) > f(1e-6)) {
        let y0 = f(ay + f(f(across - f(r * len)) / dx));
        let y1 = f(ay + f(f(across + f(r * len)) / dx));
        if (y0 > y1) [y0, y1] = [y1, y0];
        ya = Math.max(ya, y0);
        yb = Math.min(yb, y1);
      } else if (Math.abs(across) > f(r * len)) {
        yb = ya;
      }
      if (yb > ya) {
        lo = Math.min(lo, ya);
        hi = Math.max(hi, yb);
      }
    }
    return hi > lo ? [lo, hi] : null;
  }

  function stroke(xs, ys, r, element, alpha, out) {
    const n = xs.length;
    if (n < 2) return;
    let xLo = xs[0], xHi = xs[0];
    for (let i = 1; i < n; i++) { xLo = Math.min(xLo, xs[i]); xHi = Math.max(xHi, xs[i]); }
    let x0 = Math.floor(f(xLo - r));
    let x1 = Math.ceil(f(xHi + r));
    x0 = x0 < 0 ? 0 : x0;
    x1 = x1 > W - 1 ? W - 1 : x1;
    for (let x = x0; x <= x1; x++) {
      const xc = f(x + 0.5);
      const found = [];
      for (let i = 0; i + 1 < n; i++) {
        const s = capsule(xs[i], ys[i], xs[i + 1], ys[i + 1], r, xc);
        if (!s) continue;
        let j = found.length;
        found.push(s);
        while (j > 0 && found[j - 1][0] > s[0]) { found[j] = found[j - 1]; j--; }
        found[j] = s;
      }
      if (found.length === 0) continue;
      let [a, b] = found[0];
      for (let j = 1; j < found.length; j++) {
        if (found[j][0] <= b) {
          b = Math.max(b, found[j][1]);
        } else {
          push(out, x, a, b, element, alpha);
          [a, b] = found[j];
        }
      }
      push(out, x, a, b, element, alpha);
    }
  }

  function cheeks(amount, out) {
    if (amount <= 0) return;
    const fill = opacity(f(f(0.45) * amount));
    const line = opacity(amount);
    const rx = g32("cheek_rx_px"), ry = g32("cheek_ry_px");
    const cy = f(f(0.5 * H) - g32("cheek_y_px"));
    for (let side = -1; side <= 1; side += 2) {
      const cx = f(f(0.5 * W) + f(side * g32("cheek_x_px")));
      const x0 = Math.floor(f(cx - rx));
      const x1 = Math.ceil(f(cx + rx));
      for (let x = x0; x <= x1; x++) {
        const d = f(f(f(x + 0.5) - cx) / rx);
        if (d <= -1 || d >= 1) continue;
        const h = f(ry * f(Math.sqrt(f(1 - f(d * d)))));
        push(out, x, f(cy - h), f(cy + h), E.cheek, fill);
      }
      for (let k = -1; k <= 1; k++) {
        const sx = f(cx + f(f(k * f(0.42)) * rx));
        const xs = [f(sx - f(f(0.22) * rx)), f(sx + f(f(0.22) * rx))];
        const ys = [f(cy + f(f(0.6) * ry)), f(cy - f(f(0.6) * ry))];
        stroke(xs, ys, f(1.4), E.cheek, line, out);
      }
    }
  }

  function tongue(g, st, amount, where, lean, out) {
    if (amount <= 0) return;
    const line = st.deco_line_px;
    const u = clampf(where, -1, 1);
    const tx = f(g.cx + f(u * g.half_w));
    const root = g.lower(u);
    const len = Math.min(f(amount * g32("tongue_px")), f(f(H - g32("margin_px")) - root));
    if (len <= 0) return;
    const wr = Math.min(f(0.5 * st.tongue_w_px), len);
    const reach = f(len - wr);
    const bx = f(tx + f(lean * reach));
    const by = f(root + reach);
    const x0 = Math.floor(f(Math.min(tx, bx) - wr));
    const x1 = Math.ceil(f(Math.max(tx, bx) + wr));
    for (let pass = line > 0 ? 0 : 1; pass < 2; pass++) {
      const rr = pass === 0 || line <= 0 ? wr : Math.max(f(wr - line), 0.5);
      const element = pass === 0 ? E.lip : E.tongue;
      for (let x = x0; x <= x1; x++) {
        const s = capsule(tx, root, bx, by, rr, f(x + 0.5));
        if (!s) continue;
        let under = root;
        if (x >= out.x0 && x <= out.x1 && out.cols[x].active) {
          const c = out.cols[x];
          under = c.ib > c.it ? f(c.it + f(f(0.4) * f(c.ib - c.it))) : f(0.5 * f(c.ot + c.ob));
        }
        push(out, x, Math.max(s[0], under), s[1], element, 32);
      }
    }
    if (st.tongue_shine > 0 && reach > 0) {  // the highlight: an ellipse high on the tongue
      const sr = f(st.tongue_shine * f(wr - line));
      const sx = f(f(tx + f(f(lean * f(0.35)) * reach)) - f(f(0.25) * wr));
      const sy = f(root + f(f(0.4) * reach));
      const sx0 = Math.floor(f(sx - sr));
      const sx1 = Math.ceil(f(sx + sr));
      for (let x = sx0; x <= sx1; x++) {
        const d = f(f(f(x + 0.5) - sx) / sr);
        if (d <= -1 || d >= 1) continue;
        const h = f(f(f(0.6) * sr) * f(Math.sqrt(f(1 - f(d * d)))));
        push(out, x, Math.max(f(sy - h), root), f(sy + h), E.shade, 32);
      }
    }
    const y0 = f(f(root + g.r) + 1);
    const y1 = f(root + f(f(0.6) * len));
    if (y1 > y0) {
      const xs = [f(tx + f(lean * f(y0 - root))), f(tx + f(lean * f(y1 - root)))];
      stroke(xs, [y0, y1], f(0.8), E.cavity, 10, out);
    }
  }

  function patch(g, st, amount, lay) {
    if (amount <= 0 || g.open_px <= 0) return;
    const span = f(1 + Math.abs(st.patch_x));
    const rise = f(amount * g.open_px);
    for (let x = lay.x0; x <= lay.x1; x++) {
      const c = lay.cols[x];
      if (!c.active || c.ib <= c.it) continue;
      const u = f(f(f(x + 0.5) - g.cx) / g.half_w);
      const d = f(f(u - st.patch_x) / span);
      const dome = f(Math.sqrt(Math.max(f(1 - f(d * d)), 0)));
      const floorY = f(f(c.it + c.up) + (c.up > 0 ? st.teeth_line_px : 0));
      const top = Math.max(f(c.ib - f(rise * dome)), floorY);
      push(lay, x, top, f(c.ib - c.lo), E.tongue, 32);
      if (st.deco_line_px > 0) push(lay, x, Math.max(f(top - st.deco_line_px), floorY), top, E.lip, 32);
    }
  }

  function fangs(g, st, amount, at, sideOf, lay, out) {
    if (amount <= 0) return;
    const fw = g32("fang_w_px");
    for (let side = -1; side <= 1; side += 2) {
      const a = f(amount * Math.min(1, f(1 + f(side * sideOf))));
      if (a <= 0) continue;
      const len = f(a * g32("fang_px"));
      const fx = f(g.cx + f(f(side * at) * g.half_w));
      const x0 = Math.floor(f(fx - fw));
      const x1 = Math.ceil(f(fx + fw));
      for (let x = x0; x <= x1; x++) {
        if (x < lay.x0 || x > lay.x1) continue;
        const c = lay.cols[x];
        if (!c.active) continue;
        const d = f(Math.abs(f(f(x + 0.5) - fx)) / fw);
        if (d >= 1) continue;
        const open = c.ib > c.it;
        let root = open ? c.it : f(0.5 * f(c.ot + c.ob));
        if (open && c.up > 0 && !c.gap) root = f(root + f(c.up + st.teeth_line_px));
        push(out, x, f(root - 1), f(root + f(len * f(1 - d))), E.teeth, 32);
      }
    }
  }

  function dimples(g, amount, left, right, out) {
    if (amount <= 0) return;
    const half = f(amount * DIMPLE_ARC);
    const r = Math.max(1, f(f(0.8) * g.r));
    const rad = f(g32("dimple_px") + g.r);
    for (let side = -1; side <= 1; side += 2) {
      const u = side;
      const ex = side < 0 ? left : right;
      const ey = g.mid(u);
      const [tx, ty] = g.tangent(u);
      const xs = [], ys = [];
      for (let k = 0; k < DIMPLE_POINTS; k++) {
        const phi = f(-half + f(f(f(2 * half) * k) / (DIMPLE_POINTS - 1)));
        const c = cosf(phi), s = sinf(phi);
        xs.push(f(ex + f(rad * f(f(c * tx) - f(s * ty)))));
        ys.push(f(ey + f(rad * f(f(c * ty) + f(s * tx)))));
      }
      stroke(xs, ys, r, E.lip, 32, out);
    }
  }

  function hooks(g, amount, left, right, out) {
    if (amount === 0) return;
    const rad = f(g32("hook_px") * Math.abs(amount));
    for (let side = -1; side <= 1; side += 2) {
      const u = side;
      const ex = side < 0 ? left : right;
      const ey = g.mid(u);
      const [tx, ty] = g.tangent(u);
      let nx = -ty, ny = tx;
      if (ny < 0) { nx = -nx; ny = -ny; }
      if (amount < 0) { nx = -nx; ny = -ny; }
      const ccx = f(ex + f(rad * nx));
      const ccy = f(ey + f(rad * ny));
      const xs = [], ys = [];
      for (let k = 0; k < HOOK_POINTS; k++) {
        const th = f(f(HOOK_ARC * k) / (HOOK_POINTS - 1));
        const c = cosf(th), s = sinf(th);
        xs.push(f(f(ccx - f(f(rad * c) * nx)) + f(f(rad * s) * tx)));
        ys.push(f(f(ccy - f(f(rad * c) * ny)) + f(f(rad * s) * ty)));
      }
      stroke(xs, ys, g.r, E.lip, 32, out);
    }
  }

  function block(cx, top, bot, hw, rc, element, out) {
    const x0 = Math.floor(f(cx - hw));
    const x1 = Math.ceil(f(cx + hw));
    for (let x = x0; x <= x1; x++) {
      const d = Math.abs(f(f(x + 0.5) - cx));
      if (d >= hw) continue;
      let inset = 0;
      if (d > f(hw - rc)) {
        const e = f(f(d - f(hw - rc)) / rc);
        inset = f(rc * f(1 - f(Math.sqrt(Math.max(f(1 - f(e * e)), 0)))));
      }
      push(out, x, f(top + inset), f(bot - inset), element, 32);
    }
  }

  function moustache(g, amount, out) {
    if (amount <= 0) return;
    const h = f(amount * g32("moustache_h_px"));
    const hw = f(0.5 * g32("moustache_w_px"));
    const rim = g32("moustache_rim_px");
    const lip = f(f(g.mid(0) - f(g.open_px * g.upper)) - g.r);
    const bot = f(f(lip - g32("moustache_gap_px")) - rim);
    const top = f(bot - h);
    const rc = Math.min(3, f(0.5 * h));
    if (rim > 0) block(g.cx, f(top - rim), f(bot + rim), f(hw + rim), f(rc + rim), E.lip, out);
    block(g.cx, top, bot, hw, rc, E.moustache, out);
    const x0 = Math.floor(f(g.cx - hw));
    const x1 = Math.ceil(f(g.cx + hw));
    for (let x = x0; x <= x1; x++) {
      const d = Math.abs(f(f(x + 0.5) - g.cx));
      if (d >= f(hw - 2) || f(f(d + 2.5) % 5) >= 1) continue;
      push(out, x, f(bot - f(f(0.35) * h)), f(f(bot + rim) + 1), DECO_BACKGROUND, 32);
    }
  }

  function pucker(g, amount, width, lineShare, out) {
    if (amount <= 0) return;
    // a lobe's radius; a mouth narrower than 0.2 scales the "3" down (a small "ε")
    const rl = f(f(f(0.25 * g32("pucker_px")) * amount) * Math.min(1, f(width / f(0.2))));
    const cx = f(g.cx - f(f(0.3) * rl));
    const cy = g.mid(0);
    const xs = new Array(2 * LOBE_STEPS + 1), ys = new Array(2 * LOBE_STEPS + 1);
    for (let k = 0; k <= LOBE_STEPS; k++) {
      xs[k] = f(cx + f(rl * LOBE_COS[k]));
      ys[k] = f(f(cy - rl) + f(rl * LOBE_SIN[k]));
      xs[LOBE_STEPS + k] = f(cx + f(rl * LOBE_COS[k]));
      ys[LOBE_STEPS + k] = f(f(cy + rl) + f(rl * LOBE_SIN[k]));
    }
    xs[LOBE_STEPS] = f(cx - f(f(0.25) * rl));
    ys[LOBE_STEPS] = cy;
    const r = lineShare > 0 ? Math.max(1, f(lineShare * rl)) : g.r;
    stroke(xs, ys, r, E.lip, 32, out);
  }

  function heartShape(x0, y0, s, inset, element, alpha, out) {
    const lr = f(f(LOBE_R * s) - inset);
    if (lr <= 0) return;
    const sx = f(f(SIDE_X * s) - f(SIDE_NX * inset));
    const sy = f(f(y0 + f(SIDE_Y * s)) - f(SIDE_NY * inset));
    const px = [f(x0 - sx), f(x0 + sx), x0];
    const py = [sy, sy, f(f(y0 + f(TIP_Y * s)) - f(TIP_INSET * inset))];
    const xa = Math.floor(f(f(x0 - f(LOBE_X * s)) - lr));
    const xb = Math.ceil(f(f(x0 + f(LOBE_X * s)) + lr));
    for (let x = xa; x <= xb; x++) {
      const xc = f(x + 0.5);
      let lo = 1e9, hi = -1e9;
      for (let side = -1; side <= 1; side += 2) {
        const d = f(xc - f(x0 + f(f(side * LOBE_X) * s)));
        if (Math.abs(d) < lr) {
          const h = f(Math.sqrt(f(f(lr * lr) - f(d * d))));
          lo = Math.min(lo, f(y0 - h));
          hi = Math.max(hi, f(y0 + h));
        }
      }
      for (let e = 0; e < 3; e++) {
        const q = (e + 1) % 3;
        const xa_ = Math.min(px[e], px[q]), xb_ = Math.max(px[e], px[q]);
        if (xc < xa_ || xc > xb_ || px[e] === px[q]) continue;
        const y = f(py[e] + f(f(f(xc - px[e]) * f(py[q] - py[e])) / f(px[q] - px[e])));
        lo = Math.min(lo, y);
        hi = Math.max(hi, y);
      }
      push(out, x, lo, hi, element, alpha);
    }
  }

  function heartLips(g, amount, out) {
    if (amount <= 0) return;
    const s = f(f(amount * g32("heart_lips_px")) / HEART_HEIGHT);
    const y0 = f(g.mid(0) - f(f(0.35) * s));
    heartShape(g.cx, y0, s, 0, E.lip, 32, out);
    heartShape(g.cx, y0, s, f(2 * g.r), E.cavity, 32, out);
  }

  function drool(g, amount, line, out) {
    if (amount <= 0) return;
    const u = g32("drool_at");
    const x = f(g.cx + f(u * g.half_w));
    const root = f(g.lower(u) + f(0.5 * g.r));
    const len = Math.min(f(amount * g32("drool_px")), f(f(H - g32("margin_px")) - root));
    if (len <= 0) return;
    const dr = Math.min(f(DROOL_DROP_PX * amount), f(0.5 * len));
    const stem = DROOL_STEM_PX;
    const dy = f(f(root + len) - dr);
    for (let pass = line > 0 ? 0 : 1; pass < 2; pass++) {
      const grow = pass === 0 ? 0 : -line;
      const element = pass === 0 ? E.lip : E.shade;
      const rs = Math.max(f(stem + grow), 0.4);
      const rd = Math.max(f(dr + grow), 0.5);
      const x0 = Math.floor(f(x - rd));
      const x1 = Math.ceil(f(x + rd));
      for (let xx = x0; xx <= x1; xx++) {
        const xc = f(xx + 0.5);
        let lo = 1e9, hi = -1e9;
        const s = capsule(x, root, x, dy, rs, xc);
        if (s) { lo = Math.min(lo, s[0]); hi = Math.max(hi, s[1]); }
        const d = f(xc - x);
        if (Math.abs(d) < rd) {
          const h = f(Math.sqrt(f(f(rd * rd) - f(d * d))));
          lo = Math.min(lo, f(dy - h));
          hi = Math.max(hi, f(dy + h));
        }
        if (hi > lo) push(out, xx, lo, hi, element, 32);
      }
    }
  }

  function zGlyph(g, k, edge, out) {
    if (k < 0) return;
    const s = f(g32("z_glyph_px") * f(f(0.55) + f(f(0.45) * k)));
    const rise = f(1 - f(f(1 - k) * f(1 - k)));
    let x0 = f(f(f(f(edge + 6) + f(0.5 * s)) + f(10 * k)) + f(4 * sinf(f(f(TWO_PI * f(1.2)) * k))));
    let y0 = f(f(f(g.mid(1) - f(0.5 * s)) - 6) - f(g32("z_glyph_rise_px") * rise));
    x0 = Math.min(x0, f(f(f(W - g32("margin_px")) - f(0.5 * s)) - g.r));
    y0 = Math.max(y0, f(f(g32("margin_px") + f(0.5 * s)) + g.r));
    const alpha = opacity(k < f(0.6) ? 1 : f(1 - f(f(k - f(0.6)) / f(0.4))));
    const h = f(0.5 * s);
    const xs = [f(x0 - h), f(x0 + h), f(x0 - h), f(x0 + h)];
    const ys = [f(y0 - h), f(y0 - h), f(y0 + h), f(y0 + h)];
    stroke(xs, ys, Math.max(1, f(f(0.7) * g.r)), E.lip, alpha, out);
  }

  function heart(g, k, edge, out) {
    if (k < 0) return;
    const hp = g32("heart_px");
    const s = f(hp * f(f(0.55) + f(f(0.45) * ease(f(k * 5)))));
    const rise = f(1 - f(f(1 - k) * f(1 - k)));
    let x0 = f(f(f(f(edge + 4) + f(f(0.6) * hp)) + f(12 * k)) + f(5 * sinf(f(f(TWO_PI * 1.5) * k))));
    let y0 = f(f(g.mid(1) - f(f(0.6) * hp)) - f(g32("heart_rise_px") * rise));
    x0 = Math.min(x0, f(f(W - g32("margin_px")) - f(f(LOBE_X + LOBE_R) * s)));
    y0 = Math.max(y0, f(g32("margin_px") + f(LOBE_R * s)));
    const alpha = opacity(k < f(0.55) ? 1 : f(1 - f(f(k - f(0.55)) / f(0.45))));
    heartShape(x0, y0, s, 0, E.heart, alpha, out);
  }

  function layout(shape) {
    const style = styleOf(shape.style);
    const st = STYLES[style];
    const g = geometry(shape, st);
    const teeth = clampf(shape.p[P.teeth], 0, 1);
    const left = f(g.cx - g.half_w);
    const right = f(g.cx + g.half_w);
    const sx = [], stp = [], sb = [], so = [];
    const k0 = Math.ceil(f(left * 2));
    const k1 = Math.floor(f(right * 2));
    for (let k = k0 - 1; k <= k1 + 1 && sx.length < SAMPLES_MAX && g.shrink > 0; k++) {
      let xs = f(0.5 * k);
      if (k === k0 - 1) xs = left;
      if (k === k1 + 1) xs = right;
      const u = clampf(f(f(xs - g.cx) / g.half_w), -1, 1);
      const m = g.mid(u);
      const o = g.open_px > 0 ? f(g.open_px * g.profile(u)) : 0;
      sx.push(xs);
      let top = f(m - f(o * g.upper));
      if (g.bow_px > 0) top = f(top + f(g.bow_px * bow(u)));
      stp.push(top);
      sb.push(f(m + f(o * f(1 - g.upper))));
      so.push(o);
    }
    const n = sx.length;
    let fit = 1;
    const yLo = f(g32("margin_px") + g.r);
    const yHi = f(f(H - g32("margin_px")) - g.r);
    for (let i = 0; i < n; i++) {
      const below = f(so[i] * f(1 - g.upper));
      const above = f(so[i] * g.upper);
      if (sb[i] > yHi && below > 0) fit = Math.min(fit, f(f(yHi - f(sb[i] - below)) / below));
      if (stp[i] < yLo && above > 0) fit = Math.min(fit, f(f(f(stp[i] + above) - yLo) / above));
    }
    if (fit < 1) {
      fit = Math.max(fit, 0);
      for (let i = 0; i < n; i++) {
        const cut = f(so[i] * f(1 - fit));
        stp[i] = f(stp[i] + f(cut * g.upper));
        sb[i] = f(sb[i] - f(cut * f(1 - g.upper)));
      }
      g.open_px = f(g.open_px * fit);
    }
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
        ot = Math.min(ot, f(stp[i] - hh));
        it = Math.max(it, f(stp[i] + hh));
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
        if (Math.abs(f(xc - g.cx)) > f(st.teeth_span * g.half_w)) { up = 0; lo = 0; }  // the band's reach
        c.up = up; c.lo = lo;
        const m = f(f(Math.abs(f(xc - g.cx)) + f(0.5 * G.tooth_px)) % G.tooth_px);
        c.gap = m < st.tooth_gap_px;
      }
    }
    const lay = {
      cols, x0, x1, color: shape.color.slice(), background: st.background, style, deco: [], nUnder: 0,
    };
    const p = shape.p;
    cheeks(clampf(p[P.cheeks], 0, 1), lay);
    lay.nUnder = lay.deco.length;
    patch(g, st, clampf(p[P.patch], 0, 1), lay);
    tongue(g, st, clampf(p[P.tongue], 0, 1), p[P.tongue_x], p[P.tongue_lean], lay);
    drool(g, clampf(p[P.drool], 0, 1), st.deco_line_px, lay);
    fangs(g, st, clampf(p[P.fangs], 0, 1), clampf(p[P.fang_at], 0, 1), clampf(p[P.fang_side], -1, 1), lay, lay);
    dimples(g, clampf(p[P.dimples], 0, 1), left, right, lay);
    hooks(g, f(clampf(p[P.hooks], -1, 1) * g.shrink), left, right, lay);
    moustache(g, clampf(p[P.moustache], 0, 1), lay);
    pucker(g, g.pucker, clampf(p[P.width], 0.02, 1), st.pucker_line, lay);
    heartLips(g, g.lips, lay);
    const lobe = f(f(0.25 * g32("pucker_px")) * g.pucker);
    const heartEdge = f(g.cx + f(f(f(0.95) * f(g.lips * g32("heart_lips_px"))) / HEART_HEIGHT));
    heart(g, shape.heart, Math.max(Math.max(right, f(g.cx + f(f(1.2) * lobe))), heartEdge), lay);
    zGlyph(g, shape.z === undefined ? -1 : shape.z, Math.max(right, f(g.cx + f(f(1.2) * lobe))), lay);
    return lay;
  }

  function mix(dst, c, w) {
    const r = ((((dst >> 11) & 31) * (32 - w)) + (((c >> 11) & 31) * w)) >> 5;
    const g = ((((dst >> 5) & 63) * (32 - w)) + (((c >> 5) & 63) * w)) >> 5;
    const b = (((dst & 31) * (32 - w)) + ((c & 31) * w)) >> 5;
    return (r << 11) | (g << 5) | b;
  }

  function blend(px, x, y, coverage, c, alpha) {
    const w = Math.floor(f(coverage * alpha) + 0.5);
    if (w <= 0) return;
    const i = y * W + x;
    px[i] = w >= 32 ? c : mix(px[i], c, w);
  }

  function span(px, x, ya, yb, c, alpha = 32) {
    ya = Math.max(ya, 0);
    yb = Math.min(yb, H);
    if (yb <= ya) return;
    const ia = Math.floor(ya);
    const ib = Math.floor(yb);
    if (ia === ib) { blend(px, x, ia, f(yb - ya), c, alpha); return; }
    blend(px, x, ia, f(ia + 1 - ya), c, alpha);
    if (alpha >= 32) {
      for (let y = ia + 1; y < ib; y++) px[y * W + x] = c;
    } else {
      for (let y = ia + 1; y < ib; y++) px[y * W + x] = mix(px[y * W + x], c, alpha);
    }
    if (yb > ib && ib < H) blend(px, x, ib, f(yb - ib), c, alpha);
  }

  // The whole screen (one strip of H rows) as RGB565 values.
  function draw(lay, px) {
    const st = STYLES[lay.style];
    px.fill(lay.background);
    for (let i = 0; i < lay.nUnder; i++) {
      const d = lay.deco[i];
      span(px, d.x, d.top, d.bot, lay.color[d.element], d.alpha);
    }
    const lip = lay.color[E.lip], cavity = lay.color[E.cavity], teeth = lay.color[E.teeth];
    const shade = lay.color[E.shade];
    for (let x = lay.x0; x <= lay.x1; x++) {
      const c = lay.cols[x];
      if (!c.active) continue;
      span(px, x, c.ot, c.ob, lip);
      if (c.ib <= c.it) continue;
      if (st.shade > 0) {
        const split = f(c.ib - f(st.shade * f(c.ib - c.it)));
        span(px, x, c.it, split, cavity);
        span(px, x, split, c.ib, shade);
      } else {
        span(px, x, c.it, c.ib, cavity);
      }
      if (c.gap) continue;
      if (c.up > 0) {
        const edge = f(c.it + c.up);
        span(px, x, c.it, edge, teeth);
        if (st.teeth_line_px > 0) span(px, x, edge, Math.min(f(edge + st.teeth_line_px), c.ib), lip);
      }
      if (c.lo > 0) {
        const edge = f(c.ib - c.lo);
        span(px, x, edge, c.ib, teeth);
        if (st.teeth_line_px > 0) span(px, x, Math.max(f(edge - st.teeth_line_px), c.it), edge, lip);
      }
    }
    for (let i = lay.nUnder; i < lay.deco.length; i++) {
      const d = lay.deco[i];
      const c = d.element === DECO_BACKGROUND ? lay.background : lay.color[d.element];
      span(px, d.x, d.top, d.bot, c, d.alpha);
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

  // The resting values and colours (RGB565) of expression `id` in `style` (a shape of its own:
  // the dump, the previews).
  const valuesOf = (id, style = T.style) => Float32Array.from(look(id, styleOf(style)).values);
  const colorsOf = (id, style = T.style) => look(id, styleOf(style)).colors.map((c) => rgb565(rgbOf(c)));

  const api = {
    T, P, E, W, H, NEUTRAL, FaceModel, layout, draw, render, toRGBA, ease, valuesOf, colorsOf,
    restyle,
  };
  if (typeof module !== "undefined") module.exports = api;
  else root.Face = api;
})(typeof window !== "undefined" ? window : globalThis);
