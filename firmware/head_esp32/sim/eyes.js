// The eyes' model and power stage: a line-by-line port of src/eyes.cpp, so the simulator
// lights the rings as the head will (the same bytes per LED per frame; tests/unit/
// test_face_parity.py compares the two), and the pictures of the rings: the raw LEDs on
// their ring geometry and the glow through the diffuser film. The table is eye_table.js,
// generated from config/face.json's `eyes` block. Math.fround keeps the arithmetic in float32
// like the ESP32's.
"use strict";

(function (root) {
  const T = typeof EYE_TABLE !== "undefined" ? EYE_TABLE : require("./eye_table.js");
  const Face = typeof window !== "undefined" ? root.Face : require("./face.js");
  const FT = Face.T;
  const f = Math.fround;
  const ease = Face.ease;
  const N = T.params.length;
  const EL = T.elements.length;
  const LEDS = T.leds.deg.length;
  const P = {};
  T.params.forEach((p, i) => { P[p.name] = i; });
  const E = {};
  T.elements.forEach((name, i) => { E[name] = i; });
  const EYES = T.sides.length;
  const LEFT = 0, RIGHT = 1;
  const INNER = 0;
  const G = {};
  for (const [k, v] of Object.entries(T.geometry)) G[k] = f(v);
  const TM = {};
  for (const [k, v] of Object.entries(T.timing)) TM[k] = f(v);
  const PW = {};
  for (const [k, v] of Object.entries(T.power)) PW[k] = f(v);
  const MIN = T.params.map((p) => f(p.min));
  const MAX = T.params.map((p) => f(p.max));
  const FX = {};
  for (const [k, v] of Object.entries(T.effect_numbers)) FX[k] = f(v);
  const FXID = {};
  T.effects.forEach((name, i) => { FXID[name] = i; });
  const EFFECTS = T.effects.length;
  const oneShot = (e) => e === FXID.sparkle || e === FXID.flash;
  const NEUTRAL = Face.NEUTRAL;
  const FULL_TURN = 360, HALF_TURN = 180;
  const BROW_MAX_DEG = 85;
  const TEST_LEVEL = f(0.4), TEST_TAIL = f(0.12);
  const RATE_FLOOR = f(0.05);
  const COLOR_SCALE = f(1 / 255);
  const LID_TOPMOST = f(1 + f(G.lid_soft * 0.5));
  const LID_SPAN = f(2 + G.lid_soft);
  const OVERRIDE_BYTES = 16, OVERRIDE_BYTES_EFFECT = 19;
  const KEEP_I8 = 0x80, KEEP_U8 = 0xff;
  const MODE = { expression: 0, standing: 1, timed: 2, selfTest: 3 };

  // the chain's geometry, live: the simulator's layout controls rebuild it (the table's
  // numbers are the firmware's)
  const leds = { eye: [], kind: [], deg: [], sin: [], cos: [], innerFirst: [0, 0], innerN: [8, 8] };
  function innerRings(rings) {
    for (const r of rings) {
      if (r.kind !== INNER) continue;
      leds.innerFirst[r.eye] = r.first;
      leds.innerN[r.eye] = r.n;
    }
  }
  function relayout(rings) {
    leds.eye.length = leds.kind.length = leds.deg.length = leds.sin.length = leds.cos.length = 0;
    innerRings(rings);
    for (const r of rings) {
      const step = (360 / r.n) * (r.clockwise ? 1 : -1);
      for (let k = 0; k < r.n; k++) {
        const deg = (((r.first_deg + k * step) % 360) + 360) % 360;
        const sin = Number((Math.sin((deg * Math.PI) / 180)).toPrecision(9));
        const cos = Number((Math.cos((deg * Math.PI) / 180)).toPrecision(9));
        leds.eye.push(r.eye);
        leds.kind.push(r.kind);
        leds.deg.push(f(deg));
        leds.sin.push(f(Math.abs(sin) < 1e-9 ? 0 : sin));
        leds.cos.push(f(Math.abs(cos) < 1e-9 ? 0 : cos));
      }
    }
  }
  // the table's own LED angles (pepin.face wrote the same decimals the firmware compiles)
  function tableLayout() {
    innerRings(T.rings);
    leds.eye = T.leds.eye.slice();
    leds.kind = T.leds.kind.slice();
    leds.deg = T.leds.deg.map(f);
    leds.sin = T.leds.sin.map(f);
    leds.cos = T.leds.cos.map(f);
  }
  tableLayout();

  const clampf = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const wrap = (deg) => {
    while (deg > HALF_TURN) deg = f(deg - FULL_TURN);
    while (deg <= -HALF_TURN) deg = f(deg + FULL_TURN);
    return deg;
  };
  const rgbOf = (c) => [(c >>> 16) & 255, (c >>> 8) & 255, c & 255];
  const arc = (ledDeg, centreDeg, halfDeg) => {
    const d = Math.abs(wrap(f(ledDeg - centreDeg)));
    return f(1 - ease(f(f(f(d - halfDeg) / G.brow_soft_deg) + 0.5)));
  };
  const look = (set, id, style) => T.looks[set][id][style];
  const SETS = T.sets.length;
  const SHUT_ARC = T.sets.map((s) => s.shut_arc);
  const SHUT_LID = T.sets.map((s) => f(s.shut_lid));
  const SET_OF_STYLE = T.style_sets;
  // A smooth wave of the phase (periods), -1..1: a triangle eased.
  const waveOf = (phase) => {
    phase = f(phase - Math.floor(phase));
    const tri = phase < 0.5 ? f(2 * phase) : f(2 - f(2 * phase));
    return f(f(2 * ease(tri)) - 1);
  };

  // A 'Y' payload's 16 or 19 bytes as the override (protocol.h has the layout and the
  // sentinels).
  function decodeOverride(p) {
    const o = {
      mode: p[0], ms: p[1] | (p[2] << 8), mask: 0, value: new Float32Array(N), tinted: false,
      tint: [0, 0, 0], blink: p[14], side: p[15], hasEffect: false, effect: 0, effectMs: 0,
    };
    if (p.length >= OVERRIDE_BYTES_EFFECT && p[16] !== KEEP_U8) {
      o.hasEffect = true;
      o.effect = p[16] < EFFECTS ? p[16] : 0;
      o.effectMs = p[17] | (p[18] << 8);
    }
    const fields = [
      [3, P.brow_pos, true], [4, P.brow_raise, true], [5, P.brow_width, false],
      [6, P.lid_top, false], [7, P.lid_bottom, false], [8, P.gaze_x, true], [9, P.gaze_y, true],
      [13, P.glow, false],
    ];
    for (const [at, param, wide] of fields) {
      const b = p[at];
      if (wide) {
        if (b === KEEP_I8) continue;
        let v = b < 128 ? b : b - 256;
        v = v < -100 ? -100 : v > 100 ? 100 : v;
        o.value[param] = f(v * f(0.01));
      } else {
        if (b === KEEP_U8) continue;
        o.value[param] = f(b * f(1 / 254));
      }
      o.value[param] = clampf(o.value[param], MIN[param], MAX[param]);
      o.mask |= 1 << param;
    }
    o.tinted = (p[10] | p[11] | p[12]) !== 0;
    for (let k = 0; k < 3; k++) o.tint[k] = p[10 + k];
    return o;
  }

  // -- the model (eyes.cpp) --------------------------------------------------------------------
  class EyeModel {
    constructor() {
      const style = FT.style;
      this.setChoice = 0;
      this.set = SET_OF_STYLE[style];
      this.look = Float32Array.from(look(this.set, NEUTRAL, style).values);
      this.clook = look(this.set, NEUTRAL, style).colors.map(rgbOf);
      this.ovMask = [0, 0];
      this.ovValue = [new Float32Array(N), new Float32Array(N)];
      this.ovTinted = [false, false];
      this.ovTint = [[0, 0, 0], [0, 0, 0]];
      this.ovTimed = false;
      this.ovUntil = 0;
      this.from = [Float32Array.from(this.look), Float32Array.from(this.look)];
      this.to = [Float32Array.from(this.look), Float32Array.from(this.look)];
      this.cur = [Float32Array.from(this.look), Float32Array.from(this.look)];
      const copy = () => this.clook.map((c) => c.slice());
      this.cfrom = [copy(), copy()];
      this.cto = [copy(), copy()];
      this.ccur = [copy(), copy()];
      this.t0 = 0; this.dur = 0; this.primed = false; this.starts = 0; this.style = -1;
      this.started = false; this.last = 0; this.breathPhase = 0;
      this.rng = 0x9e3779b9;
      this.blinkOn = [false, false]; this.blinkT0 = [0, 0];
      this.blinkClose = [0, 0]; this.blinkHold = [0, 0]; this.blinkOpen = [0, 0];
      this.blinkCount = [1, 1];
      this.lookEffect = 0; this.hostEffectOn = false; this.hostEffect = 0; this.hostEffectUntil = 0;
      this.effectSource = 0xff; this.effect = 0; this.effectT0 = 0;
      this.twinkleT0 = new Array(LEDS).fill(0); this.twinkleOn = new Array(LEDS).fill(false);
      this.flickerLevel = 1; this.flickerNext = 0;
      this.blinkScheduled = false; this.blinkNext = 0;
      this.saccScheduled = false; this.saccNext = 0; this.saccT0 = 0;
      this.saccFrom = [0, 0]; this.saccTo = [0, 0];
      this.heart = -1; this.z = -1;
      this.testOn = false; this.testT0 = 0;
      this.brightness = T.power.brightness;
      this.limitMa = 0; this.ramping = false; this.rampFrom = 0; this.rampTo = 0; this.rampT0 = 0;
      this.light = new Float32Array(LEDS * 3);
      this.frame = { rgb: new Uint8Array(LEDS * 3), milliamps: 0 };
    }

    seed(s) { this.rng = (s >>> 0) !== 0 ? s >>> 0 : 0x9e3779b9; }

    nextRandom() {
      let x = this.rng;
      x = (x ^ (x << 13)) >>> 0;
      x = (x ^ (x >>> 17)) >>> 0;
      x = (x ^ (x << 5)) >>> 0;
      this.rng = x;
      return x;
    }

    uniform() { return f((this.nextRandom() >>> 8) * (1 / 16777216)); }

    setSet(choice) { this.setChoice = choice; }

    lookOf(set, style, shown, intensity) {
      const nv = look(set, NEUTRAL, style).values, xv = look(set, shown, style).values;
      for (let i = 0; i < N; i++) {
        const n = f(nv[i]);
        this.look[i] = f(n + f(f(f(xv[i]) - n) * intensity));
      }
      const nc = look(set, NEUTRAL, style).colors, xc = look(set, shown, style).colors;
      for (let e = 0; e < EL; e++) {
        const n = rgbOf(nc[e]), x = rgbOf(xc[e]);
        for (let k = 0; k < 3; k++) this.clook[e][k] = f(n[k] + f(f(x[k] - n[k]) * intensity));
      }
    }

    retarget(ms, t0) {
      for (let eye = 0; eye < EYES; eye++) {
        for (let i = 0; i < N; i++) {
          this.from[eye][i] = this.cur[eye][i];
          this.to[eye][i] = (this.ovMask[eye] >>> i) & 1 ? this.ovValue[eye][i] : this.look[i];
        }
        for (let e = 0; e < EL; e++) {
          const tinted = this.ovTinted[eye] && (e === E.iris || e === E.brow);
          for (let k = 0; k < 3; k++) {
            this.cfrom[eye][e][k] = this.ccur[eye][e][k];
            this.cto[eye][e][k] = tinted ? this.ovTint[eye][k] : this.clook[e][k];
          }
        }
      }
      this.t0 = t0;
      this.dur = ms > 0 ? f(ms) : 0;
      if (this.dur === 0) {
        for (let eye = 0; eye < EYES; eye++) {
          for (let i = 0; i < N; i++) this.cur[eye][i] = this.to[eye][i];
          for (let e = 0; e < EL; e++) for (let k = 0; k < 3; k++) this.ccur[eye][e][k] = this.cto[eye][e][k];
        }
      }
    }

    setOverride(o, nowMs) {
      const left = o.side !== 2, right = o.side !== 1;
      if (o.mode === 0) {
        for (let eye = 0; eye < EYES; eye++) { this.ovMask[eye] = 0; this.ovTinted[eye] = false; }
        this.ovTimed = false;
      } else {
        for (let eye = 0; eye < EYES; eye++) {
          if (!(eye === LEFT ? left : right)) continue;
          this.ovMask[eye] = o.mask;
          for (let i = 0; i < N; i++) this.ovValue[eye][i] = o.value[i];
          this.ovTinted[eye] = o.tinted;
          for (let k = 0; k < 3; k++) this.ovTint[eye][k] = o.tint[k];
        }
        this.ovTimed = o.mode === 2;
        this.ovUntil = nowMs + o.ms;
      }
      this.retarget(TM.override_ms, nowMs);
      if (o.blink === 1 || o.blink === 2) this.startBlink(LEFT, o.blink === 2, 1, 1, nowMs);
      if (o.blink === 1 || o.blink === 3) this.startBlink(RIGHT, o.blink === 3, 1, 1, nowMs);
      if (o.mode === 0 || (o.hasEffect && o.effect === FXID.none)) {
        if (this.hostEffectOn) {
          this.hostEffectOn = false;
          this.effectSource = this.lookEffect;
          this.effect = oneShot(this.lookEffect) ? FXID.none : this.lookEffect;
          this.effectT0 = nowMs;
        }
      } else if (o.hasEffect) {
        this.hostEffectOn = true;
        this.hostEffect = o.effect;
        this.hostEffectUntil = o.effectMs > 0 ? nowMs + o.effectMs : 0;
        this.effectSource = 0xff;
      }
    }

    setBrightness(b) { this.brightness = b; }

    selfTest(nowMs) {
      this.testOn = true;
      this.testT0 = nowMs;
      this.limitMa = PW.cap_ma;
      this.ramping = false;
    }

    mode() {
      if (this.testOn) return MODE.selfTest;
      for (let eye = 0; eye < EYES; eye++) {
        if (this.ovMask[eye] !== 0 || this.ovTinted[eye]) return this.ovTimed ? MODE.timed : MODE.standing;
      }
      return MODE.expression;
    }

    startBlink(eye, wink, rate, count, nowMs) {
      this.blinkOn[eye] = true;
      this.blinkT0[eye] = nowMs;
      this.blinkClose[eye] = f(TM.blink_close_ms / rate);
      this.blinkHold[eye] = wink ? TM.wink_hold_ms : 0;
      this.blinkOpen[eye] = f(TM.blink_open_ms / rate);
      this.blinkCount[eye] = count;
    }

    blinkOf(eye, nowMs) {
      if (!this.blinkOn[eye]) return 0;
      let t = f(nowMs - this.blinkT0[eye]);
      const cycle = f(f(this.blinkClose[eye] + this.blinkHold[eye]) + this.blinkOpen[eye]);
      let n = 0;
      while (t >= cycle && n < this.blinkCount[eye]) { t = f(t - cycle); n += 1; }
      if (n >= this.blinkCount[eye]) { this.blinkOn[eye] = false; return 0; }
      if (t < this.blinkClose[eye]) return ease(f(t / this.blinkClose[eye]));
      const held = f(t - this.blinkClose[eye]);
      if (held < this.blinkHold[eye]) return 1;
      const opening = f(held - this.blinkHold[eye]);
      if (opening < this.blinkOpen[eye]) return f(1 - ease(f(opening / this.blinkOpen[eye])));
      this.blinkOn[eye] = false;
      return 0;
    }

    follow(model, nowMs) {
      const set = this.setChoice >= 1 && this.setChoice <= SETS ? this.setChoice - 1 : SET_OF_STYLE[model.style];
      const setChanged = set !== this.set;
      if (!this.primed || model.starts !== this.starts || model.style !== this.style || setChanged) {
        this.set = set;
        this.lookOf(this.set, model.style, model.shown, f(model.intensity));
        this.lookEffect = model.intensity > 0 ? look(this.set, model.shown, model.style).effect : FXID.none;
        if (oneShot(this.lookEffect) && !this.hostEffectOn && !setChanged) this.effectSource = 0xff;
        if (!this.primed) {
          this.retarget(0, nowMs);
          this.primed = true;
        } else if (setChanged) {
          this.retarget(TM.override_ms, nowMs);
        } else {
          this.retarget(model.dur, model.t0);
        }
        this.starts = model.starts;
        this.style = model.style;
      }
      if (this.ovTimed && nowMs - this.ovUntil >= 0) {
        for (let eye = 0; eye < EYES; eye++) { this.ovMask[eye] = 0; this.ovTinted[eye] = false; }
        this.ovTimed = false;
        this.retarget(TM.override_ms, nowMs);
      }
      const ps = model.shape();
      this.heart = ps.heart;
      this.z = ps.z;
      this.update(nowMs);
    }

    update(nowMs) {
      if (!this.started) { this.started = true; this.last = nowMs; }
      let dt = f(f(nowMs - this.last) * f(0.001));
      dt = clampf(dt, 0, f(0.25));
      this.last = nowMs;

      const elapsed = f(nowMs - this.t0);
      const e = this.dur > 0 ? ease(f(elapsed / this.dur)) : 1;
      for (let eye = 0; eye < EYES; eye++) {
        const from = this.from[eye], to = this.to[eye], cur = this.cur[eye];
        for (let i = 0; i < N; i++) cur[i] = f(from[i] + f(f(to[i] - from[i]) * e));
        for (let c = 0; c < EL; c++) {
          for (let k = 0; k < 3; k++) {
            this.ccur[eye][c][k] = f(this.cfrom[eye][c][k] + f(f(this.cto[eye][c][k] - this.cfrom[eye][c][k]) * e));
          }
        }
      }
      this.breathPhase = f(this.breathPhase + f(TM.breath_hz * dt));
      this.breathPhase = f(this.breathPhase - Math.floor(this.breathPhase));

      if (!this.blinkScheduled) {
        this.blinkScheduled = true;
        this.blinkNext = nowMs + Math.trunc(f(f(TM.blink_min_s + f(f(TM.blink_max_s - TM.blink_min_s) * this.uniform())) * 1000));
      }
      if (nowMs - this.blinkNext >= 0) {
        const rate = clampf(this.cur[LEFT][P.blink_rate], 0, MAX[P.blink_rate]);
        let span = f(TM.blink_min_s + f(f(TM.blink_max_s - TM.blink_min_s) * this.uniform()));
        if (rate >= RATE_FLOOR) {
          span = f(span / rate);
          if (!this.blinkOn[LEFT] && !this.blinkOn[RIGHT]) {
            const count = this.uniform() < TM.flutter ? 3 : 1;
            this.startBlink(LEFT, false, rate, count, nowMs);
            this.startBlink(RIGHT, false, rate, count, nowMs);
          }
        }
        this.blinkNext = nowMs + Math.trunc(f(span * 1000));
      }

      if (!this.saccScheduled) {
        this.saccScheduled = true;
        this.saccT0 = nowMs;
        this.saccNext = nowMs + Math.trunc(f(f(TM.saccade_min_s + f(f(TM.saccade_max_s - TM.saccade_min_s) * this.uniform())) * 1000));
      }
      if (nowMs - this.saccNext >= 0) {
        const k = clampf(f(f(nowMs - this.saccT0) / TM.saccade_ms), 0, 1);
        for (let a = 0; a < 2; a++) {
          this.saccFrom[a] = f(this.saccFrom[a] + f(f(this.saccTo[a] - this.saccFrom[a]) * ease(k)));
          this.saccTo[a] = f(TM.saccade_amp * f(f(2 * this.uniform()) - 1));
        }
        this.saccT0 = nowMs;
        this.saccNext = nowMs + Math.trunc(f(f(TM.saccade_min_s + f(f(TM.saccade_max_s - TM.saccade_min_s) * this.uniform())) * 1000));
      }

      this.updateEffect(nowMs, dt);

      if (this.testOn && f(nowMs - this.testT0) >= TM.self_test_ms) this.testOn = false;
      if (this.testOn) {
        this.renderSelfTest(nowMs);
        this.power(nowMs, true);
      } else {
        this.render(nowMs);
        this.power(nowMs, false);
      }
    }

    updateEffect(nowMs, dt) {
      let wanted = this.lookEffect;
      if (this.hostEffectOn) {
        if (this.hostEffectUntil !== 0 && nowMs - this.hostEffectUntil >= 0) {
          this.hostEffectOn = false;
          this.effectSource = this.lookEffect;
          this.effect = oneShot(this.lookEffect) ? FXID.none : this.lookEffect;
          this.effectT0 = nowMs;
        } else {
          wanted = this.hostEffect;
        }
      }
      if (wanted !== this.effectSource) {
        this.effectSource = wanted;
        this.effect = wanted;
        this.effectT0 = nowMs;
        this.twinkleOn.fill(false);
        this.flickerLevel = 1;
        this.flickerNext = nowMs;
      }
      const elapsed = f(nowMs - this.effectT0);
      if (this.effect === FXID.sparkle) {
        if (elapsed >= f(FX.sparkle_ms + f(2 * FX.sparkle_rise_ms))) {
          this.effect = FXID.none;
        } else if (elapsed < FX.sparkle_ms) {
          for (let eye = 0; eye < EYES; eye++) {
            if (this.uniform() < f(FX.sparkle_rate * dt)) {
              const led = leds.innerFirst[eye] + (this.nextRandom() % leds.innerN[eye]);
              this.twinkleOn[led] = true;
              this.twinkleT0[led] = nowMs;
            }
          }
        }
      } else if (this.effect === FXID.flash) {
        if (elapsed >= FX.flash_ms) this.effect = FXID.none;
      } else if (this.effect === FXID.flicker) {
        if (nowMs - this.flickerNext >= 0) {
          this.flickerLevel = f(FX.flicker_floor + f(f(1 - FX.flicker_floor) * this.uniform()));
          this.flickerNext = nowMs + Math.trunc(f(FX.flicker_min_ms + f(f(FX.flicker_max_ms - FX.flicker_min_ms) * this.uniform())));
        }
      }
    }

    renderSelfTest(nowMs) {
      const k = f(f(nowMs - this.testT0) / TM.self_test_ms);
      let at = Math.trunc(f(k * LEDS));
      if (at < 0) at = 0;
      if (at >= LEDS) at = LEDS - 1;
      const behind = at > 0 ? at - 1 : LEDS - 1;
      for (let i = 0; i < LEDS; i++) {
        const v = i === at ? TEST_LEVEL : i === behind ? TEST_TAIL : 0;
        this.light[3 * i] = this.light[3 * i + 1] = this.light[3 * i + 2] = v;
      }
    }

    render(nowMs) {
      const tri = this.breathPhase < 0.5 ? f(2 * this.breathPhase) : f(2 - f(2 * this.breathPhase));
      const wave = f(f(2 * ease(tri)) - 1);
      const sk = clampf(f(f(nowMs - this.saccT0) / TM.saccade_ms), 0, 1);
      const saccX = f(this.saccFrom[0] + f(f(this.saccTo[0] - this.saccFrom[0]) * ease(sk)));
      const saccY = f(this.saccFrom[1] + f(f(this.saccTo[1] - this.saccFrom[1]) * ease(sk)));
      const lobePower = Math.trunc(G.highlight_power);
      const fxElapsedS = f(f(nowMs - this.effectT0) * f(0.001));
      let fxGain = 1, fxWhite = 0, fxSink = 0, spinHead = 0;
      let breathOn = true;
      if (this.effect === FXID.flash) {
        fxWhite = f(FX.flash_gain * f(1 - ease(f(f(fxElapsedS * 1000) / FX.flash_ms))));
      } else if (this.effect === FXID.spin) {
        spinHead = f(f(fxElapsedS * FX.spin_hz) * FULL_TURN);
        spinHead = f(spinHead - f(Math.floor(f(spinHead / FULL_TURN)) * FULL_TURN));
        fxGain = FX.spin_dim;
      } else if (this.effect === FXID.pulse) {
        fxGain = f(1 + f(FX.pulse_depth * waveOf(f(fxElapsedS * FX.pulse_hz))));
      } else if (this.effect === FXID.flicker) {
        fxGain = this.flickerLevel;
      } else if (this.effect === FXID.doze) {
        const exhale = f(0.5 + f(0.5 * waveOf(f(fxElapsedS * FX.doze_hz))));
        fxGain = f(1 - f(FX.doze_depth * exhale));
        fxSink = f(FX.doze_sink * exhale);
        breathOn = false;
      }
      let tearK = -1;
      if (this.effect === FXID.tear) {
        let inPeriod = f(fxElapsedS * 1000);
        inPeriod = f(inPeriod - f(Math.floor(f(inPeriod / FX.tear_period_ms)) * FX.tear_period_ms));
        if (inPeriod < FX.tear_fall_ms) tearK = f(inPeriod / FX.tear_fall_ms);
      }
      const tearRgb = [FX.tear_r, FX.tear_g, FX.tear_b];

      for (let eye = 0; eye < EYES; eye++) {
        const p = this.cur[eye];
        const side = eye === LEFT ? 1 : -1;
        const nose = eye === RIGHT ? 1 : -1;
        const raise = clampf(f(p[P.brow_raise] + f(p[P.brow_asym] * side)), -1, 1);
        const wink = clampf(f(p[P.wink] * side), 0, 1);
        const blink = this.blinkOf(eye, nowMs);
        const shut = wink > blink ? wink : blink;
        let lidTop = p[P.lid_top];
        let lidBottom = p[P.lid_bottom];
        if (SHUT_ARC[this.set]) {
          const lid = SHUT_LID[this.set];
          if (lidBottom < lid) lidBottom = f(lidBottom + f(f(lid - lidBottom) * shut));
        } else {
          lidTop = f(lidTop + f(f(1 - lidTop) * shut));
        }
        if (this.z >= 0) lidTop = f(lidTop + f(G.z_sink * this.z));
        lidTop = f(lidTop + fxSink);
        lidTop = clampf(lidTop, 0, 1);
        lidBottom = clampf(lidBottom, 0, 1);
        const edgeTop = f(LID_TOPMOST - f(f(LID_SPAN * lidTop) * f(Math.sqrt(lidTop))));
        const edgeBottom = f(-LID_TOPMOST + f(f(LID_SPAN * lidBottom) * f(Math.sqrt(lidBottom))));
        let browAt = f(f(f(-p[P.brow_pos]) * G.brow_span_deg) * f(1 - f(G.brow_pull * raise)));
        if (raise < 0) browAt = f(browAt + f(raise * G.brow_sag_deg));
        browAt = clampf(browAt, -BROW_MAX_DEG, BROW_MAX_DEG);
        const browHalf = f(G.brow_half_min_deg + f(f(G.brow_half_max_deg - G.brow_half_min_deg) * p[P.brow_width]));
        const browBright = f(p[P.brow_glow] * clampf(f(G.brow_dim + f(G.brow_lift * raise)), 0, 1));
        const lowerHalf = f(G.lower_lid_half_min_deg + f(f(G.lower_lid_half_max_deg - G.lower_lid_half_min_deg) * lidBottom));
        const lowerBright = f(G.lower_lid_glow * lidBottom);
        const cheekBright = f(G.cheek_glow * p[P.cheek]);
        const gazeX = f(f(p[P.gaze_x] * G.gaze_gain) + saccX);
        const gazeY = f(f(p[P.gaze_y] * G.gaze_gain) + saccY);
        const px = f(f(-gazeX) - f(nose * G.catchlight_out));
        const py = f(gazeY + G.catchlight_up);
        const pm = f(Math.sqrt(f(f(px * px) + f(py * py))));
        const breath = breathOn ? f(1 + f(f(G.breath_depth * p[P.breath]) * wave)) : 1;
        const glow = f(f(p[P.glow] * breath) * fxGain);
        const sparkleAt = f(f(TM.sparkle_turns * FULL_TURN) * this.heart);
        const cc = this.ccur[eye];

        for (let i = 0; i < LEDS; i++) {
          if (leds.eye[i] !== eye) continue;
          const sx = leds.sin[i];
          const h = leds.cos[i];
          const c = [0, 0, 0];
          if (leds.kind[i] === INNER) {
            const coverTop = ease(f(f(f(h - edgeTop) / G.lid_soft) + 0.5));
            const coverBottom = ease(f(f(f(edgeBottom - h) / G.lid_soft) + 0.5));
            let lit = f(f(1 - coverTop) * f(1 - coverBottom));
            if (p[P.cross] > 0) {
              const diag = f(f(2 * sx) * h);
              lit = f(lit * f(f(1 - p[P.cross]) + f(f(p[P.cross] * diag) * diag)));
            }
            let lobe = 0;
            if (pm > 1e-4) {
              const d = f(f(f(sx * px) + f(h * py)) / pm);
              if (d > 0) {
                lobe = 1;
                for (let n = 0; n < lobePower; n++) lobe = f(lobe * d);
                lobe = f(lobe * (pm < 1 ? pm : 1));
              }
            }
            let k = f(lobe * p[P.highlight]);
            let extra = fxWhite;
            if (this.effect === FXID.spin) {
              const d = wrap(f(leds.deg[i] - spinHead));
              let w = f(1 - f(Math.abs(d) / FX.spin_head_deg));
              if (w < 0) w = 0;
              if (d < 0 && -d < FX.spin_tail_deg) {
                const tail = f(f(0.6) * f(1 + f(d / FX.spin_tail_deg)));
                if (tail > w) w = tail;
              }
              k = f(k + f(f(1 - k) * w));
              extra = f(extra + f(w * f(1 - FX.spin_dim)));
            } else if (this.effect === FXID.sparkle && this.twinkleOn[i]) {
              const tw = f(f(nowMs - this.twinkleT0[i]) / FX.sparkle_rise_ms);
              if (tw >= 2) this.twinkleOn[i] = false;
              else extra = f(extra + f(FX.sparkle_gain * (tw < 1 ? tw : f(2 - tw))));
            }
            const spot = f(glow * f(1 + f(G.highlight_gain * k)));
            for (let ch = 0; ch < 3; ch++) {
              const iris = cc[E.iris][ch];
              const mixed = f(iris + f(f(cc[E.highlight][ch] - iris) * k));
              c[ch] = f(f(f(mixed * spot) * lit) + f(f(255 * extra) * lit));
            }
          } else {
            const phi = wrap(f(nose * leds.deg[i]));
            const above = ease(f(f(h - G.brow_floor) / G.brow_floor_soft));
            const below = ease(f(f(f(-h) - G.brow_floor) / G.brow_floor_soft));
            const brow = f(f(browBright * arc(phi, browAt, browHalf)) * above);
            const lower = f(f(lowerBright * arc(phi, HALF_TURN, lowerHalf)) * below);
            const cheek = f(f(cheekBright * arc(phi, HALF_TURN, G.cheek_half_deg)) * below);
            let sparkle = 0;
            if (this.heart >= 0) {
              const d = Math.abs(wrap(f(phi - sparkleAt)));
              const w = f(1 - f(d / G.sparkle_half_deg));
              if (w > 0) sparkle = f(w * f(1 - this.heart));
            }
            let tear = 0;
            if (this.effect === FXID.tear && tearK >= 0) {
              const at = f(100 + f(80 * tearK));
              const d = f(phi - at);
              if (d >= 0) tear = f(1 - f(d / FX.tear_tail_deg));
              else tear = f(1 + f(d / 12));
              if (tear < 0) tear = 0;
              tear = f(f(tear * below) * f(1 - ease(f(f(tearK - f(0.7)) / f(0.3)))));
            }
            for (let ch = 0; ch < 3; ch++) {
              c[ch] = f(f(f(f(f(cc[E.brow][ch] * brow) + f(cc[E.lid][ch] * lower)) + f(cc[E.cheek][ch] * cheek)) + f(255 * f(sparkle + fxWhite))) + f(tearRgb[ch] * tear));
            }
          }
          for (let ch = 0; ch < 3; ch++) this.light[3 * i + ch] = clampf(f(c[ch] * COLOR_SCALE), 0, 1);
        }
      }
    }

    power(nowMs, full) {
      const gain = full ? 1 : f(this.brightness * COLOR_SCALE);
      const duty = new Uint8Array(LEDS * 3);
      let sum = 0;
      for (let i = 0; i < LEDS * 3; i++) {
        let q = Math.trunc(f(f(this.light[i] * 255) + 0.5));
        q = q < 0 ? 0 : q > 255 ? 255 : q;
        duty[i] = T.gamma[q];
        sum = f(sum + duty[i]);
      }
      const base = f(LEDS * PW.quiescent_ma);
      const demand = f(f(f(PW.led_ma * sum) * COLOR_SCALE) * gain);
      let allowed = f(PW.cap_ma - base);
      if (allowed < 0) allowed = 0;
      let scale = demand > allowed && demand > 0 ? f(allowed / demand) : 1;
      const capped = f(demand * scale);
      if (capped > this.limitMa) {
        if (!this.ramping) {
          this.ramping = true;
          this.rampFrom = this.limitMa;
          this.rampT0 = nowMs;
        }
        this.rampTo = capped;
        const k = clampf(f(f(nowMs - this.rampT0) / PW.soft_start_ms), 0, 1);
        this.limitMa = f(this.rampFrom + f(f(this.rampTo - this.rampFrom) * k));
        if (k >= 1) this.ramping = false;
      } else {
        this.limitMa = capped;
        this.ramping = false;
      }
      if (capped > 0 && this.limitMa < capped) scale = f(scale * f(this.limitMa / capped));
      scale = f(scale * gain);
      let outSum = 0;
      for (let i = 0; i < LEDS * 3; i++) {
        let v = Math.trunc(f(duty[i] * scale));
        v = v < 0 ? 0 : v > 255 ? 255 : v;
        this.frame.rgb[i] = v;
        outSum = f(outSum + v);
      }
      const ma = f(base + f(f(PW.led_ma * outSum) * COLOR_SCALE));
      this.frame.milliamps = Math.trunc(f(ma + 0.5));
    }
  }

  // -- the pictures ------------------------------------------------------------------------
  // The rings as seen facing the robot: the robot's right eye on the left. Millimetres from
  // the table's layout; `scale` pixels per mm.
  const L = T.layout;
  const MARGIN_MM = 4;
  function geometry(scale) {
    const half = L.outer_radius_mm + L.led_mm + MARGIN_MM;
    const w = Math.round((L.eye_gap_mm + 2 * half) * scale);
    const h = Math.round(2 * half * scale);
    const centres = [
      { x: (half + L.eye_gap_mm) * scale, y: half * scale },  // the robot's left eye: the viewer's right
      { x: half * scale, y: half * scale },                   // its right eye: the viewer's left
    ];
    const pos = [];
    for (let i = 0; i < LEDS; i++) {
      const r = (leds.kind[i] === INNER ? L.inner_radius_mm : L.outer_radius_mm) * scale;
      const c = centres[leds.eye[i]];
      const a = (leds.deg[i] * Math.PI) / 180;
      pos.push({ x: c.x + r * Math.sin(a), y: c.y - r * Math.cos(a) });
    }
    return { w, h, centres, pos, scale };
  }

  // The LED's visible brightness: its duty (linear light) encoded for the screen.
  const encode = (linear) => Math.round(255 * Math.pow(Math.min(Math.max(linear, 0), 1), 1 / 2.2));

  function fill(rgba, w, h, r, g, b) {
    for (let i = 0; i < w * h; i++) { rgba[4 * i] = r; rgba[4 * i + 1] = g; rgba[4 * i + 2] = b; rgba[4 * i + 3] = 255; }
  }

  function disc(rgba, w, h, cx, cy, radius, r, g, b) {
    const x0 = Math.max(0, Math.floor(cx - radius)), x1 = Math.min(w - 1, Math.ceil(cx + radius));
    const y0 = Math.max(0, Math.floor(cy - radius)), y1 = Math.min(h - 1, Math.ceil(cy + radius));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const dx = x + 0.5 - cx, dy = y + 0.5 - cy;
        if (dx * dx + dy * dy > radius * radius) continue;
        const i = 4 * (y * w + x);
        rgba[i] = r; rgba[i + 1] = g; rgba[i + 2] = b;
      }
    }
  }

  function annulus(rgba, w, h, cx, cy, inner, outer, r, g, b) {
    const x0 = Math.max(0, Math.floor(cx - outer)), x1 = Math.min(w - 1, Math.ceil(cx + outer));
    const y0 = Math.max(0, Math.floor(cy - outer)), y1 = Math.min(h - 1, Math.ceil(cy + outer));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const dx = x + 0.5 - cx, dy = y + 0.5 - cy;
        const d2 = dx * dx + dy * dy;
        if (d2 > outer * outer || d2 < inner * inner) continue;
        const i = 4 * (y * w + x);
        rgba[i] = r; rgba[i + 1] = g; rgba[i + 2] = b;
      }
    }
  }

  // The rings' bodies and the lens barrels (both views start from this).
  function paintBodies(rgba, geo) {
    fill(rgba, geo.w, geo.h, 8, 9, 11);
    for (const c of geo.centres) {
      const s = geo.scale;
      annulus(rgba, geo.w, geo.h, c.x, c.y, (L.outer_radius_mm - 4) * s, (L.outer_radius_mm + 4) * s, 20, 22, 25);
      annulus(rgba, geo.w, geo.h, c.x, c.y, (L.inner_radius_mm - 3.5) * s, (L.inner_radius_mm + 3.5) * s, 20, 22, 25);
      disc(rgba, geo.w, geo.h, c.x, c.y, L.lens_radius_mm * s, 3, 4, 5);
    }
  }

  // The raw LEDs: each a disc in its own colour (an unlit one a dark square of the package).
  function paintRaw(rgb, rgba, geo) {
    paintBodies(rgba, geo);
    for (let i = 0; i < LEDS; i++) {
      const p = geo.pos[i];
      const r = encode(rgb[3 * i] / 255), g = encode(rgb[3 * i + 1] / 255), b = encode(rgb[3 * i + 2] / 255);
      disc(rgba, geo.w, geo.h, p.x, p.y, L.led_mm * geo.scale, Math.max(r, 34), Math.max(g, 36), Math.max(b, 40));
    }
    return rgba;
  }

  // Through the diffuser film: every LED a Gaussian blob of film_sigma_mm, added in linear
  // light, then encoded; the lens barrel stays dark.
  let kernel = null;  // the blob, once per scale
  function blob(sigma) {
    const reach = Math.ceil(3 * sigma);
    if (kernel && kernel.sigma === sigma) return kernel;
    const size = 2 * reach + 1;
    const k = new Float32Array(size * size);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const dx = x - reach, dy = y - reach;
        k[y * size + x] = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      }
    }
    kernel = { sigma, reach, size, k };
    return kernel;
  }

  function paintFilm(rgb, rgba, geo) {
    paintBodies(rgba, geo);
    const { reach, size, k } = blob(L.film_sigma_mm * geo.scale);
    const acc = new Float32Array(geo.w * geo.h * 3);
    for (let i = 0; i < LEDS; i++) {
      const p = geo.pos[i];
      const r = rgb[3 * i] / 255, g = rgb[3 * i + 1] / 255, b = rgb[3 * i + 2] / 255;
      if (r === 0 && g === 0 && b === 0) continue;
      const cx = Math.round(p.x), cy = Math.round(p.y);
      for (let y = 0; y < size; y++) {
        const py = cy - reach + y;
        if (py < 0 || py >= geo.h) continue;
        for (let x = 0; x < size; x++) {
          const px = cx - reach + x;
          if (px < 0 || px >= geo.w) continue;
          const w = k[y * size + x];
          const j = 3 * (py * geo.w + px);
          acc[j] += r * w; acc[j + 1] += g * w; acc[j + 2] += b * w;
        }
      }
    }
    for (let i = 0; i < geo.w * geo.h; i++) {
      const r = acc[3 * i], g = acc[3 * i + 1], b = acc[3 * i + 2];
      if (r === 0 && g === 0 && b === 0) continue;
      // the film lifts the body's grey under the glow too
      rgba[4 * i] = Math.min(255, rgba[4 * i] + encode(r));
      rgba[4 * i + 1] = Math.min(255, rgba[4 * i + 1] + encode(g));
      rgba[4 * i + 2] = Math.min(255, rgba[4 * i + 2] + encode(b));
    }
    for (const c of geo.centres) disc(rgba, geo.w, geo.h, c.x, c.y, L.lens_radius_mm * geo.scale, 3, 4, 5);
    return rgba;
  }

  const api = {
    T, P, E, N, LEDS, MODE, FXID, OVERRIDE_BYTES, OVERRIDE_BYTES_EFFECT, KEEP_I8, KEEP_U8,
    EyeModel, decodeOverride, leds, relayout, tableLayout, geometry, paintRaw, paintFilm, encode,
  };
  if (typeof module !== "undefined") module.exports = api;
  else root.Eyes = api;
})(typeof window !== "undefined" ? window : globalThis);
