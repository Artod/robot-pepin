// The face as PNG files, drawn by the simulator's model and renderer (face.js) under node, no
// browser and no packages:
//
//   node sim/preview.js [--out DIR] [--scale N] [--style NAME] [--at MS,MS,...]
//                       [--level L,L,...] [--gesture NAME,NAME,...] [--background #RRGGBB]
//                       NAME|all ...
//   node sim/preview.js --eyes [--out DIR] [--style NAME] [--set NAME] [--brightness B]
//                       [--mouth-sim DIR] NAME|all|sheets ...
//
// With --eyes the rings instead of the mouth (eyes.js): each NAME is written to
// DIR/eyes_NAME.png, the look at rest seen through the diffuser film (at --brightness, default
// 255; --set classic|kawaii picks the look set, default the style's); `sheets` writes
// DIR/eyes_all.png (every expression in the pepin and the anime palette),
// DIR/eyes_strips.png (a blink, a wink, the gaze, brow, raise and lid sweeps, the sleeper's lids
// sinking with the z, transitions, the kiss's sparkle, the boot self-test) and
// DIR/eyes_raw_vs_film.png (the raw LEDs beside the film view), DIR/eyes_groups.png (the
// expressions by emotion family with the effects at their peak), DIR/eyes_brows.png (one
// eye: the brow's sweeps), DIR/eyes_kawaii_all.png (every expression in the kawaii set),
// DIR/eyes_classic_vs_kawaii.png (the two sets side by side) and, with --mouth-sim (a
// simulator directory whose preview.js draws the mouth in the kawaii style: this one or
// another checkout's), DIR/eyes_kawaii_with_mouth.png (the rings over the mouth panel at the head's
// real spacing, nine expressions).
//
// Each NAME (an expression of config/face.json; `all`: every one) is written to DIR/NAME.png as
// the table draws it at rest (its own parameters: no breath, no drift, no variant drawn). With
// --at, also DIR/NAME_<ms>ms.png: the model that many milliseconds after the expression was set
// (the breath, the drift, a variant drawn at random, the kiss's heart in flight, the tongue's
// wag). With --level, also DIR/NAME_l<L>.png: the mouth at that speech level (0..1), as lip sync
// opens it. With --gesture, also DIR/NAME_g<NAME>.png: the face with that idle gesture at its
// peak. --style draws in that style (pepin, rose, manga, anime, kawaii, comics; default:
// config/face.json's; a shapes style draws with the shape engine, anime.js: its stills are
// the model at rest);
// --background draws on that colour instead of the style's (a comparison, not the face);
// --scale enlarges the pixels (nearest neighbour; default 2).
"use strict";

const fs = require("fs");
const path = require("path");
const zlib = require("zlib");
const Face = require("./face.js");
const Anime = require("./anime.js");
const Eyes = require("./eyes.js");

const CRC_TABLE = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(bytes) {
  let c = 0xffffffff;
  for (const b of bytes) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const head = Buffer.alloc(8);
  head.writeUInt32BE(data.length, 0);
  head.write(type, 4, "ascii");
  const tail = Buffer.alloc(4);
  tail.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), data])), 0);
  return Buffer.concat([head, data, tail]);
}

// RGB565 pixels (W x H) as an 8-bit RGB PNG, each pixel `scale` x `scale`.
function png(px, scale) {
  const w = Face.W * scale;
  const h = Face.H * scale;
  const rgba = Face.toRGBA(px, new Uint8Array(px.length * 4));
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) {
    const row = y * (w * 3 + 1);
    raw[row] = 0;  // no filter
    for (let x = 0; x < w; x++) {
      const i = 4 * (Math.floor(y / scale) * Face.W + Math.floor(x / scale));
      raw.set(rgba.subarray(i, i + 3), row + 1 + 3 * x);
    }
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(w, 0);
  header.writeUInt32BE(h, 4);
  header.set([8, 2, 0, 0, 0], 8);  // 8 bits, RGB
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", header),
    chunk("IDAT", zlib.deflateSync(raw)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

// The frame of a model (and the shape engine's, in a shapes style) as pixels.
function frameOf(model, anime, ms) {
  if (Anime.tableOf(model.style) !== null) {
    anime.follow(model, ms);
    return Anime.render(anime);
  }
  return Face.render(model.shape());
}

// A fresh model in `style` showing `id` at `ms` after it was set (no tween), with the shape
// engine following it; for a shapes style the only way to a still.
function models(id, style, ms) {
  const model = new Face.FaceModel();
  const anime = new Anime.AnimeModel();
  model.setStyle(style, 0);
  model.setExpression(id, 1, 0, 0);
  model.update(0);
  if (Anime.tableOf(style) !== null) anime.follow(model, 0);
  model.update(ms);
  return [model, anime];
}

// The expression as the table draws it at rest in `style`, with an idle gesture's deltas at
// their peak when one is named (a shapes style: the gesture at its peak, 0.5 s in).
function still(id, style = Face.T.style, gesture = null) {
  if (Anime.tableOf(style) !== null) {
    const [model, anime] = models(id, style, 0);
    if (gesture) {
      const g = Face.T.idle.gestures.findIndex((x) => x.name === gesture);
      if (g < 0) throw new Error(`no idle gesture ${gesture}`);
      model.idleGesture = g; model.idleT0 = 0; model.idleFading = false; model.idleSign = 1;
      model.idleAmp = 1;
    }
    return frameOf(model, anime, 0);
  }
  const p = Face.valuesOf(id, style);
  if (gesture) {
    const g = Face.T.idle.gestures.find((x) => x.name === gesture);
    if (!g) throw new Error(`no idle gesture ${gesture}`);
    for (let i = 0; i < p.length; i++) {
      if (g.deltas[i] === 0) continue;
      const q = Face.T.params[i];
      p[i] = Math.min(Math.max(p[i] + g.deltas[i], q.min), q.max);
    }
  }
  return Face.render({ p, wavePhase: 0, heart: -1, z: -1, color: Face.colorsOf(id, style), style });
}

// The model `ms` after the expression was set (no tween), at speech level `level`.
function live(id, ms, style = Face.T.style, level = 0) {
  const [model, anime] = models(id, style, ms);
  if (level > 0) model.setMouth(level, ms);
  model.update(ms);
  let at = ms;
  if (level > 0) {  // the level's attack settled: a few more frames at the same instant
    for (let k = 0; k < 20; k++) {
      model.setMouth(level, ms + 20 * k);
      model.update(ms + 20 * k);
      if (Anime.tableOf(style) !== null) anime.follow(model, ms + 20 * k);
      at = ms + 20 * k;
    }
  }
  return frameOf(model, anime, at);
}

// RGBA pixels (w x h) as an 8-bit RGB PNG.
function pngRGBA(rgba, w, h) {
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) {
    const row = y * (w * 3 + 1);
    raw[row] = 0;
    for (let x = 0; x < w; x++) {
      const i = 4 * (y * w + x);
      raw[row + 1 + 3 * x] = rgba[i];
      raw[row + 2 + 3 * x] = rgba[i + 1];
      raw[row + 3 + 3 * x] = rgba[i + 2];
    }
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(w, 0);
  header.writeUInt32BE(h, 4);
  header.set([8, 2, 0, 0, 0], 8);
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", header),
    chunk("IDAT", zlib.deflateSync(raw)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

// -- the eyes -------------------------------------------------------------------------------
// A fresh pair of models showing `id` in `style` from t = 0 at `brightness`.
function eyeModels(id, style, brightness, set = 0) {
  const model = new Face.FaceModel();
  const eyes = new Eyes.EyeModel();
  eyes.setBrightness(brightness);
  eyes.setSet(set);  // 0: the style's set; else the set id + 1
  model.setStyle(style, 0);
  model.setExpression(id, 1, 0, 0);
  model.update(0);
  eyes.follow(model, 0);
  return [model, eyes];
}

// The frame at `ms`, stepped from the last one at 20 ms (the soft start and the clocks run).
function eyeFrame(model, eyes, ms, step = 20) {
  let t = eyes.last;
  while (t + step <= ms) {
    t += step;
    model.update(t);
    eyes.follow(model, t);
  }
  if (t < ms) {
    model.update(ms);
    eyes.follow(model, ms);
  }
  return Uint8Array.from(eyes.frame.rgb);
}

// The look at rest: 600 ms in (the soft start done; the first blink and saccade come later).
function eyeStill(id, style, brightness = 255, ms = 600, set = 0) {
  const [model, eyes] = eyeModels(id, style, brightness, set);
  return eyeFrame(model, eyes, ms);
}

// The id's still in a look set (1 classic, 2 kawaii), with a 'Y' applied before the frame
// when one is given.
function eyeStillIn(id, set, brightness = 255, fields = null) {
  const [model, eyes] = eyeModels(id, PEPIN, brightness, set);
  eyeFrame(model, eyes, 600);
  if (fields) eyes.setOverride(Eyes.decodeOverride(yPayload(fields)), 600);
  return eyeFrame(model, eyes, 1000);
}

// A picture of a frame: {rgba, w, h}.
function eyePicture(rgb, scale, film) {
  const geo = Eyes.geometry(scale);
  const rgba = new Uint8ClampedArray(geo.w * geo.h * 4);
  (film ? Eyes.paintFilm : Eyes.paintRaw)(rgb, rgba, geo);
  return { rgba, w: geo.w, h: geo.h };
}

// A 3x5 font for the sheets' labels.
const GLYPHS = {
  0: "111101101101111", 1: "010110010010111", 2: "111001111100111", 3: "111001111001111",
  4: "101101111001001", 5: "111100111001111", 6: "111100111101111", 7: "111001001001001",
  8: "111101111101111", 9: "111101111001111", a: "010101111101101", b: "110101110101110",
  c: "111100100100111", d: "110101101101110", e: "111100110100111", f: "111100110100100",
  g: "111100101101111", h: "101101111101101", i: "111010010010111", j: "001001001101111",
  k: "101101110101101", l: "100100100100111", m: "101111111101101", n: "110101101101101",
  o: "111101101101111", p: "111101111100100", q: "111101101111001", r: "110101110101101",
  s: "111100111001111", t: "111010010010010", u: "101101101101111", v: "101101101101010",
  w: "101101111111101", x: "101101010101101", y: "101101010010010", z: "111001010100111",
  _: "000000000000111", "-": "000000111000000", ":": "000010000010000", ".": "000000000000010",
  " ": "000000000000000", "+": "000010111010000", "/": "001001010100100", "%": "101001010100101",
  "(": "010100100100010", ")": "010001001001010", ",": "000000000010100",
};

function text(sheet, x, y, str, scale = 2, rgb = [216, 225, 234]) {
  for (const ch of str.toLowerCase()) {
    const g = GLYPHS[ch] || GLYPHS[" "];
    for (let r = 0; r < 5; r++) {
      for (let c = 0; c < 3; c++) {
        if (g[r * 3 + c] !== "1") continue;
        for (let dy = 0; dy < scale; dy++) {
          for (let dx = 0; dx < scale; dx++) {
            const px = x + c * scale + dx, py = y + r * scale + dy;
            if (px < 0 || py < 0 || px >= sheet.w || py >= sheet.h) continue;
            const i = 4 * (py * sheet.w + px);
            sheet.rgba[i] = rgb[0]; sheet.rgba[i + 1] = rgb[1]; sheet.rgba[i + 2] = rgb[2];
          }
        }
      }
    }
    x += 4 * scale;
  }
}

function sheet(w, h) {
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) { rgba[4 * i] = 13; rgba[4 * i + 1] = 16; rgba[4 * i + 2] = 19; rgba[4 * i + 3] = 255; }
  return { rgba, w, h };
}

function blit(dst, src, x, y) {
  for (let r = 0; r < src.h; r++) {
    const dy = y + r;
    if (dy < 0 || dy >= dst.h) continue;
    for (let c = 0; c < src.w; c++) {
      const dx = x + c;
      if (dx < 0 || dx >= dst.w) continue;
      const i = 4 * (dy * dst.w + dx), j = 4 * (r * src.w + c);
      dst.rgba[i] = src.rgba[j]; dst.rgba[i + 1] = src.rgba[j + 1]; dst.rgba[i + 2] = src.rgba[j + 2];
    }
  }
}

// A 'Y' payload from its fields (the others kept), as the head server would pack it.
function yPayload(fields, { mode = 1, ms = 0, blink = 0, side = 0, iris = null, effect = null, effectMs = 0 } = {}) {
  const p = new Uint8Array(effect === null ? Eyes.OVERRIDE_BYTES : Eyes.OVERRIDE_BYTES_EFFECT);
  p[0] = mode; p[1] = ms & 255; p[2] = ms >> 8;
  const at = { brow_pos: [3, true], brow_raise: [4, true], brow_width: [5, false], lid_top: [6, false],
               lid_bottom: [7, false], gaze_x: [8, true], gaze_y: [9, true], glow: [13, false] };
  for (const [name, [index, signed]] of Object.entries(at)) {
    const v = fields[name];
    if (v === undefined) { p[index] = signed ? Eyes.KEEP_I8 : Eyes.KEEP_U8; continue; }
    p[index] = signed ? (Math.round(v * 100) & 255) : Math.round(v * 254);
  }
  if (iris) { p[10] = iris[0]; p[11] = iris[1]; p[12] = iris[2]; }
  p[14] = blink; p[15] = side;
  if (effect !== null) { p[16] = effect; p[17] = effectMs & 255; p[18] = effectMs >> 8; }
  return p;
}

const PEPIN = 0;
const ANIME = Face.T.styles.findIndex((s) => s.name === "anime");

// Every expression in the pepin and the anime palette, through the film.
function sheetAll(brightness) {
  const names = Face.T.expressions.map((e) => e.name);
  const scale = 2;
  const cell = Eyes.geometry(scale);
  const label = 110, gap = 6, columns = 3;
  const cw = label + 2 * (cell.w + gap), ch = cell.h + 14 + gap;
  const rows = Math.ceil(names.length / columns);
  const out = sheet(columns * cw + gap, rows * ch + 30);
  text(out, gap, 8, `eyes at rest, every expression: pepin (left) and anime (right) palettes, film view, brightness ${brightness}`);
  names.forEach((name, id) => {
    const x0 = gap + (id % columns) * cw, y0 = 30 + Math.floor(id / columns) * ch;
    text(out, x0, y0 + 4, `${id} ${name}`);
    blit(out, eyePicture(eyeStill(id, PEPIN, brightness), scale, true), x0 + label, y0);
    blit(out, eyePicture(eyeStill(id, ANIME, brightness), scale, true), x0 + label + cell.w + gap, y0);
  });
  return out;
}

// Strips over time and over a field's range.
function sheetStrips(brightness) {
  const scale = 1.5;
  const cell = Eyes.geometry(scale);
  const gap = 4, label = 330, rowH = cell.h + 18;
  const ids = Object.fromEntries(Face.T.expressions.map((e, i) => [e.name, i]));
  const strips = [];
  const over = (title, id, style, apply, times) => {
    const [model, eyes] = eyeModels(id, style, brightness);
    eyeFrame(model, eyes, 600);
    if (apply) apply(model, eyes, 600);
    const frames = times.map((t) => eyeFrame(model, eyes, 600 + t, 10));
    strips.push([title, frames.map((rgb) => eyePicture(rgb, scale, true))]);
  };
  const sweep = (title, id, style, field, values) => {
    const frames = values.map((v) => {
      const [model, eyes] = eyeModels(id, style, brightness);
      eyeFrame(model, eyes, 600);
      eyes.setOverride(Eyes.decodeOverride(yPayload({ [field]: v })), 600);
      return eyeFrame(model, eyes, 1000);
    });
    strips.push([title, frames.map((rgb) => eyePicture(rgb, scale, true))]);
  };
  const steps = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
  over("blink (0-240 ms, 20 ms)", ids.neutral, PEPIN,
       (m, e, t) => e.setOverride(Eyes.decodeOverride(yPayload({}, { blink: 1 })), t), steps(0, 240, 13));
  over("wink left (0-400 ms, 40 ms)", ids.neutral, PEPIN,
       (m, e, t) => e.setOverride(Eyes.decodeOverride(yPayload({}, { blink: 2 })), t), steps(0, 400, 11));
  sweep("gaze_x -1..1 (the robot's left..right)", ids.neutral, PEPIN, "gaze_x", steps(-1, 1, 9));
  sweep("gaze_y -1..1 (down..up)", ids.neutral, PEPIN, "gaze_y", steps(-1, 1, 9));
  sweep("brow_pos -1..1 (nose..outward)", ids.neutral, PEPIN, "brow_pos", steps(-1, 1, 9));
  sweep("brow_raise -1..1 (low..high)", ids.neutral, PEPIN, "brow_raise", steps(-1, 1, 9));
  sweep("brow_width 0..1", ids.neutral, PEPIN, "brow_width", steps(0, 1, 9));
  sweep("lid_top 0..1", ids.neutral, PEPIN, "lid_top", steps(0, 1, 9));
  sweep("lid_bottom 0..1", ids.neutral, PEPIN, "lid_bottom", steps(0, 1, 9));
  sweep("glow 0..1", ids.neutral, PEPIN, "glow", steps(0, 1, 9));
  over("sleepy (anime): the z sinks the lids (0-4.8 s, 0.4 s)", ids.sleepy, ANIME, null, steps(0, 4800, 13));
  over("neutral to happy (280 ms, 40 ms)", ids.neutral, PEPIN,
       (m, e, t) => m.setExpression(ids.happy, 1, 280, t), steps(0, 320, 9));
  over("happy to sad (280 ms, 40 ms)", ids.happy, PEPIN,
       (m, e, t) => m.setExpression(ids.sad, 1, 280, t), steps(0, 320, 9));
  over("kiss_heart: the sparkle (0-1.8 s, 150 ms)", ids.kiss_heart, PEPIN, null, steps(0, 1800, 13));
  over("breath (neutral, 0-5 s, 0.5 s)", ids.neutral, PEPIN, null, steps(0, 5000, 11));
  over("the boot self-test (0-1 s, 100 ms)", ids.neutral, PEPIN, (m, e, t) => e.selfTest(t), steps(0, 990, 11));
  const fx = (name, times) => over(`effect ${name} ('Y' byte 16: ${Eyes.FXID[name]}) ${times[times.length - 1]} ms`,
    ids.neutral, PEPIN, (m, e, t) => e.setOverride(Eyes.decodeOverride(yPayload({}, { effect: Eyes.FXID[name] })), t), times);
  fx("sparkle", steps(0, 1400, 15));
  fx("flash", steps(0, 420, 15));
  fx("spin", steps(0, 1000, 11));
  fx("pulse", steps(0, 1500, 16));
  fx("flicker", steps(0, 700, 15));
  fx("doze", steps(0, 8000, 17));
  over("a flutter: three short closures (0-840 ms, 60 ms)", ids.neutral, PEPIN,
       (m, e, t) => { e.startBlink(0, false, 1, 3, t); e.startBlink(1, false, 1, 3, t); }, steps(0, 840, 15));
  const widest = Math.max(...strips.map(([, f]) => f.length));
  const out = sheet(label + widest * (cell.w + gap) + gap, strips.length * rowH + 30);
  text(out, gap, 8, `eye strips: the host's blinks and 'Y' sweeps, the effects and the clocks, film view, brightness ${brightness}`);
  strips.forEach(([title, frames], row) => {
    const y = 30 + row * rowH;
    text(out, gap, y + 4, title);
    frames.forEach((pic, i) => blit(out, pic, label + i * (cell.w + gap), y));
  });
  return out;
}

// A few expressions: the raw LEDs beside the film view.
function sheetRawVsFilm(brightness) {
  const scale = 3;
  const cell = Eyes.geometry(scale);
  const gap = 8, label = 24;
  const names = ["neutral", "happy", "sad", "sinister", "surprised", "thinking", "sleepy", "cat"];
  const out = sheet(2 * (cell.w + gap) + gap, names.length * (cell.h + label + gap) + 30);
  text(out, gap, 8, `raw leds (left) and through the film (right), brightness ${brightness}`);
  names.forEach((name, row) => {
    const id = Face.T.expressions.findIndex((e) => e.name === name);
    const y = 30 + row * (cell.h + label + gap);
    text(out, gap, y + 4, `${id} ${name}`);
    const rgb = eyeStill(id, PEPIN, brightness);
    blit(out, eyePicture(rgb, scale, false), gap, y + label);
    blit(out, eyePicture(rgb, scale, true), 2 * gap + cell.w, y + label);
  });
  return out;
}

function crop(pic, x, y, w, h) {
  const out = sheet(w, h);
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const i = 4 * (r * w + c), j = 4 * ((y + r) * pic.w + (x + c));
      out.rgba[i] = pic.rgba[j]; out.rgba[i + 1] = pic.rgba[j + 1]; out.rgba[i + 2] = pic.rgba[j + 2];
    }
  }
  return out;
}

// The 33 expressions by emotion family at rest, pepin palette, and a row of the effects at
// their peak.
const FAMILIES = [
  ["calm", ["neutral", "flat", "focused", "listening", "moustache"]],
  ["happy", ["smile", "happy", "grin", "smile_wide", "happy_plain"]],
  ["sad", ["sad", "worried", "pout"]],
  ["angry-red", ["clenched", "struggling", "angry", "sinister", "vampire", "terminator"]],
  ["surprised", ["surprised", "o_mouth", "shocked"]],
  ["thinking", ["thinking", "thinking_flat"]],
  ["sleepy", ["sleepy", "tired"]],
  ["playful", ["kiss", "kiss_heart", "tease", "blush", "cat", "cat_blush"]],
];

// The frame of a host effect at its most telling moment: the brightest frame for sparkle,
// flash and pulse, the dimmest for flicker and doze, the spin a quarter turn in.
function effectPeak(name, brightness) {
  const ids = Object.fromEntries(Face.T.expressions.map((e, i) => [e.name, i]));
  const [model, eyes] = eyeModels(ids.neutral, PEPIN, brightness);
  eyeFrame(model, eyes, 600);
  eyes.setOverride(Eyes.decodeOverride(yPayload({}, { effect: Eyes.FXID[name] })), 600);
  const span = { sparkle: 1400, flash: 400, spin: 240, pulse: 800, flicker: 700, doze: 8400 }[name];
  let best = null, bestScore = null;
  for (let t = 20; t <= span; t += 20) {
    const rgb = eyeFrame(model, eyes, 600 + t, 20);
    let total = 0;
    for (let i = 0; i < 24; i++) total += rgb[3 * i] + rgb[3 * i + 1] + rgb[3 * i + 2];  // the inner left ring
    const score = name === "flicker" || name === "doze" ? -total : total;
    const pick = name === "spin" ? best === null && t >= span : bestScore === null || score > bestScore;
    if (pick) { best = rgb; bestScore = score; }
  }
  return best;
}

function sheetGroups(brightness) {
  const ids = Object.fromEntries(Face.T.expressions.map((e, i) => [e.name, i]));
  const scale = 2;
  const cell = Eyes.geometry(scale);
  const gap = 6, label = 96, rowH = cell.h + 20 + gap;
  const widest = Math.max(...FAMILIES.map(([, names]) => names.length), 6);
  const out = sheet(label + widest * (cell.w + gap) + gap, (FAMILIES.length + 1) * rowH + 30);
  text(out, gap, 8, `eyes by family at rest, pepin palette, film view, brightness ${brightness}; the effects at their peak below`);
  FAMILIES.forEach(([family, names], row) => {
    const y = 30 + row * rowH;
    text(out, gap, y + 4, family);
    names.forEach((name, i) => {
      const x = label + i * (cell.w + gap);
      text(out, x, y + 4, `${ids[name]} ${name}`);
      blit(out, eyePicture(eyeStill(ids[name], PEPIN, brightness), scale, true), x, y + 20);
    });
  });
  const y = 30 + FAMILIES.length * rowH;
  text(out, gap, y + 4, "effects");
  ["sparkle", "flash", "spin", "pulse", "flicker", "doze"].forEach((name, i) => {
    const x = label + i * (cell.w + gap);
    text(out, x, y + 4, `${Eyes.FXID[name]} ${name}`);
    blit(out, eyePicture(effectPeak(name, brightness), scale, true), x, y + 20);
  });
  return out;
}

// One eye (the robot's right, the nose to the viewer's right): the brow from the nose to the
// outside in seven steps, then low and high, then narrow and wide.
function sheetBrows(brightness) {
  const ids = Object.fromEntries(Face.T.expressions.map((e, i) => [e.name, i]));
  const scale = 3;
  const full = Eyes.geometry(scale);
  const w = Math.round(full.w / 2), h = full.h;
  const gap = 8, label = 20;
  const cells = [];
  const steps = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
  for (const v of steps(-1, 1, 7)) cells.push([`brow_pos ${v.toFixed(2)}${v < 0 ? " (nose)" : v > 0 ? " (outward)" : " (top)"}`, { brow_pos: v }]);
  cells.push(["brow_raise -1 (low)", { brow_raise: -1 }], ["brow_raise +1 (high)", { brow_raise: 1 }]);
  cells.push(["brow_width 0 (one led)", { brow_width: 0 }], ["brow_width 1 (the top half)", { brow_width: 1 }]);
  cells.push(["lid_top 0.5 + lid_bottom 0.5 (a slit)", { lid_top: 0.5, lid_bottom: 0.5 }]);
  const columns = 4;
  const rows = Math.ceil(cells.length / columns);
  const out = sheet(columns * (w + gap) + gap, rows * (h + label + gap) + 30);
  text(out, gap, 8, `one eye (the robot's right; the nose is to the right): the brow's language, 'Y' sweeps on neutral, brightness ${brightness}`);
  cells.forEach(([title, fields], i) => {
    const x = gap + (i % columns) * (w + gap), y = 30 + Math.floor(i / columns) * (h + label + gap);
    const [model, eyes] = eyeModels(ids.neutral, PEPIN, brightness);
    eyeFrame(model, eyes, 600);
    eyes.setOverride(Eyes.decodeOverride(yPayload(fields)), 600);
    const pic = crop(eyePicture(eyeFrame(model, eyes, 1000), scale, true), 0, 0, w, h);
    text(out, x, y + 4, title, 2);
    blit(out, pic, x, y + label);
  });
  return out;
}

const KAWAII_SET = 2;  // the 'C' byte's value for the kawaii set
const CLASSIC_SET = 1;

// Every expression in the kawaii set, film view.
function sheetKawaiiAll(brightness) {
  const names = Face.T.expressions.map((e) => e.name);
  const scale = 2;
  const cell = Eyes.geometry(scale);
  const gap = 6, columns = 4, cw = cell.w + gap, ch = cell.h + 20 + gap;
  const rows = Math.ceil(names.length / columns);
  const out = sheet(columns * cw + gap, rows * ch + 30);
  text(out, gap, 8, `eyes at rest, the kawaii set, every expression, pepin palette, film view, brightness ${brightness}`);
  names.forEach((name, id) => {
    const x = gap + (id % columns) * cw, y = 30 + Math.floor(id / columns) * ch;
    text(out, x, y + 4, `${id} ${name}`);
    blit(out, eyePicture(eyeStillIn(id, KAWAII_SET, brightness), scale, true), x, y + 20);
  });
  return out;
}

// The two sets side by side, per expression.
function sheetClassicVsKawaii(brightness) {
  const names = Face.T.expressions.map((e) => e.name);
  const scale = 2;
  const cell = Eyes.geometry(scale);
  const gap = 6, columns = 2, pair = 2 * cell.w + gap, cw = pair + 2 * gap, ch = cell.h + 20 + gap;
  const rows = Math.ceil(names.length / columns);
  const out = sheet(columns * cw + gap, rows * ch + 30);
  text(out, gap, 8, `classic (left) and kawaii (right) per expression, pepin palette, film view, brightness ${brightness}`);
  names.forEach((name, id) => {
    const x = gap + (id % columns) * cw, y = 30 + Math.floor(id / columns) * ch;
    text(out, x, y + 4, `${id} ${name}`);
    blit(out, eyePicture(eyeStillIn(id, CLASSIC_SET, brightness), scale, true), x, y + 20);
    blit(out, eyePicture(eyeStillIn(id, KAWAII_SET, brightness), scale, true), x + cell.w + gap, y + 20);
  });
  return out;
}

// The tandem: the rings through the film over the mouth panel (41 x 22 mm, its centre 50 mm
// under the eyes' line) at the head's real spacing, the mouth drawn by a simulator's
// preview.js (this directory's or another checkout's) in the kawaii style, for nine
// expressions.
function sheetWithMouth(brightness, mouthDir) {
  const MouthPreview = require(path.resolve(mouthDir, "preview.js"));
  const MouthFace = require(path.resolve(mouthDir, "face.js"));
  const mouthStyle = MouthFace.T.styles.findIndex((s) => s.name === "kawaii");
  if (mouthStyle < 0) throw new Error(`no kawaii style in ${mouthDir}`);
  const mouthIds = Object.fromEntries(MouthFace.T.expressions.map((e, i) => [e.name, i]));
  const ids = Object.fromEntries(Face.T.expressions.map((e, i) => [e.name, i]));
  const scale = 3;  // px per mm
  const rings = Eyes.geometry(scale);
  const panelW = Math.round(41 * scale), panelH = Math.round(22 * scale);
  const drop = Math.round(50 * scale);  // the panel's centre under the eyes' line
  const faceH = Math.round(rings.h / 2 + drop + panelH / 2 + 8);
  const cellW = rings.w, gap = 10, label = 20;
  const cells = [
    ["neutral", "neutral", null], ["smile", "smile", null], ["happy", "happy", null],
    ["sad", "sad", null], ["angry", "angry", null],
    ["surprised", "surprised", null], ["sleepy", "sleepy", null], ["tease", "tease", null], ["cat", "cat", null],
  ];
  const columns = 3;
  const rows = Math.ceil(cells.length / columns);
  const out = sheet(columns * (cellW + gap) + gap, rows * (faceH + label + gap) + 30);
  text(out, gap, 8, `the tandem: kawaii eyes over the kawaii mouth, 61 mm between the eyes, the 41x22 mm panel 50 mm below, brightness ${brightness}`);
  cells.forEach(([title, name, fields], i) => {
    const x = gap + (i % columns) * (cellW + gap), y = 30 + Math.floor(i / columns) * (faceH + label + gap);
    text(out, x, y + 4, title);
    const face = sheet(cellW, faceH);
    for (let k = 0; k < face.w * face.h; k++) { face.rgba[4 * k] = 16; face.rgba[4 * k + 1] = 18; face.rgba[4 * k + 2] = 22; }
    blit(face, eyePicture(eyeStillIn(ids[name], KAWAII_SET, brightness, fields), scale, true), 0, 0);
    // the mouth panel: the 320x170 frame scaled onto 41 x 22 mm
    const px = MouthPreview.still(mouthIds[name], mouthStyle);
    const rgba = MouthFace.toRGBA(px, new Uint8Array(px.length * 4));
    const panel = sheet(panelW, panelH);
    for (let yy = 0; yy < panelH; yy++) {
      for (let xx = 0; xx < panelW; xx++) {
        const sx = Math.min(MouthFace.W - 1, Math.floor((xx / panelW) * MouthFace.W));
        const sy = Math.min(MouthFace.H - 1, Math.floor((yy / panelH) * MouthFace.H));
        const j = 4 * (sy * MouthFace.W + sx), k = 4 * (yy * panelW + xx);
        panel.rgba[k] = rgba[j]; panel.rgba[k + 1] = rgba[j + 1]; panel.rgba[k + 2] = rgba[j + 2];
      }
    }
    blit(face, panel, Math.round(cellW / 2 - panelW / 2), Math.round(rings.h / 2 + drop - panelH / 2));
    blit(out, face, x, y + label);
  });
  return out;
}

function eyesMain(argv) {
  let out = ".";
  let style = Face.T.style;
  let brightness = 255;
  let set = 0;
  let mouthDir = null;
  const names = [];
  const styles = Face.T.styles.map((s) => s.name);
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--out") out = argv[++i];
    else if (a === "--brightness") brightness = Math.min(255, Math.max(0, Math.trunc(Number(argv[++i]))));
    else if (a === "--mouth-sim") mouthDir = argv[++i];
    else if (a === "--set") {
      const sets = Eyes.T.sets.map((s) => s.name);
      set = sets.indexOf(argv[++i]) + 1;
      if (set < 1) {
        process.stderr.write(`no set ${argv[i]}; the eyes have ${sets.join(", ")}\n`);
        return 1;
      }
    }
    else if (a === "--style") {
      style = styles.indexOf(argv[++i]);
      if (style < 0) {
        process.stderr.write(`no style ${argv[i]}; the face has ${styles.join(", ")}\n`);
        return 1;
      }
    } else if (a === "--eyes") continue;
    else names.push(a);
  }
  const known = Face.T.expressions.map((e) => e.name);
  const wanted = names.includes("all") ? known : names.filter((n) => n !== "sheets");
  if (wanted.length === 0 && !names.includes("sheets")) {
    process.stderr.write("usage: node sim/preview.js --eyes [--out DIR] [--style NAME] [--brightness B] NAME|all|sheets ...\n");
    return 2;
  }
  fs.mkdirSync(out, { recursive: true });
  const write = (file, pic) => {
    fs.writeFileSync(path.join(out, file), pngRGBA(pic.rgba, pic.w, pic.h));
    process.stdout.write(`${path.join(out, file)}\n`);
  };
  for (const name of wanted) {
    const id = known.indexOf(name);
    if (id < 0) {
      process.stderr.write(`no expression ${name}; the face knows ${known.join(", ")}\n`);
      return 1;
    }
    write(`eyes_${name}.png`, eyePicture(eyeStill(id, style, brightness, 600, set), 3, true));
  }
  if (names.includes("sheets")) {
    write("eyes_all.png", sheetAll(brightness));
    write("eyes_strips.png", sheetStrips(brightness));
    write("eyes_raw_vs_film.png", sheetRawVsFilm(brightness));
    write("eyes_groups.png", sheetGroups(brightness));
    write("eyes_brows.png", sheetBrows(brightness));
    write("eyes_kawaii_all.png", sheetKawaiiAll(brightness));
    write("eyes_classic_vs_kawaii.png", sheetClassicVsKawaii(brightness));
    if (mouthDir) write("eyes_kawaii_with_mouth.png", sheetWithMouth(brightness, mouthDir));
  }
  return 0;
}

function main(argv) {
  if (argv.includes("--eyes")) return eyesMain(argv);
  let out = ".";
  let scale = 2;
  let at = [];
  let levels = [];
  let gestures = [];
  let style = Face.T.style;
  const names = [];
  const styles = Face.T.styles.map((s) => s.name);
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--out") out = argv[++i];
    else if (a === "--scale") scale = Math.max(1, Math.trunc(Number(argv[++i])));
    else if (a === "--at") at = argv[++i].split(",").map(Number);
    else if (a === "--level") levels = argv[++i].split(",").map(Number);
    else if (a === "--gesture") gestures = argv[++i].split(",");
    else if (a === "--background") {  // the style's background overridden, for a comparison
      const c = parseInt(argv[++i].replace("#", ""), 16);
      const r = (c >> 16) & 255, g = (c >> 8) & 255, b = c & 255;
      const bg = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
      for (const s of Face.T.styles) s.rgb565.background = bg;
      Face.restyle();
    }
    else if (a === "--style") {
      style = styles.indexOf(argv[++i]);
      if (style < 0) {
        process.stderr.write(`no style ${argv[i]}; the face has ${styles.join(", ")}\n`);
        return 1;
      }
    } else names.push(a);
  }
  const known = Face.T.expressions.map((e) => e.name);
  const wanted = names.includes("all") ? known : names;
  if (wanted.length === 0) {
    process.stderr.write("usage: node sim/preview.js [--out DIR] [--scale N] [--style NAME] [--at MS,..] [--level L,..] [--gesture NAME,..] [--background #RRGGBB] NAME|all ...\n");
    return 2;
  }
  fs.mkdirSync(out, { recursive: true });
  for (const name of wanted) {
    const id = known.indexOf(name);
    if (id < 0) {
      process.stderr.write(`no expression ${name}; the face knows ${known.join(", ")}\n`);
      return 1;
    }
    const files = [
      [`${name}.png`, still(id, style)],
      ...at.map((ms) => [`${name}_${ms}ms.png`, live(id, ms, style)]),
      ...levels.map((l) => [`${name}_l${l}.png`, live(id, 400, style, l)]),
      ...gestures.map((g) => [`${name}_g${g}.png`, still(id, style, g)]),
    ];
    for (const [file, px] of files) {
      fs.writeFileSync(path.join(out, file), png(px, scale));
      process.stdout.write(`${path.join(out, file)}\n`);
    }
  }
  return 0;
}

module.exports = { png, pngRGBA, still, live, eyeStill, eyePicture };
if (require.main === module) process.exitCode = main(process.argv.slice(2));
