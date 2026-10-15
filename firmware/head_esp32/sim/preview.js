// The face as PNG files, drawn by the simulator's model and renderer (face.js) under node, no
// browser and no packages:
//
//   node sim/preview.js [--out DIR] [--scale N] [--style NAME] [--at MS,MS,...]
//                       [--level L,L,...] [--gesture NAME,NAME,...] [--background #RRGGBB]
//                       NAME|all ...
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

function main(argv) {
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

module.exports = { png, still, live };
if (require.main === module) process.exitCode = main(process.argv.slice(2));
