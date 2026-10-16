// The simulator's model and renderer under node, reading the script test_native/dump.cpp reads
// (see there) and writing the same output: tests/unit/test_face_parity.py compares the two.
"use strict";

const Face = require("./face.js");
const Anime = require("./anime.js");
const Eyes = require("./eyes.js");

const words = require("fs").readFileSync(0, "utf8").split(/\s+/).filter((w) => w.length);
const model = new Face.FaceModel();
const anime = new Anime.AnimeModel();
const eyes = new Eyes.EyeModel();
let style = Face.T.style;
const out = [];
let i = 0;
const next = () => Number(words[i++]);
while (i < words.length) {
  const op = words[i++];
  if (op === "R") {
    const p = new Float32Array(Face.T.params.length);
    for (let k = 0; k < p.length; k++) p[k] = next();
    const wavePhase = Math.fround(next());
    const heart = Math.fround(next());
    const z = Math.fround(next());
    const color = Face.T.elements.map(() => next());
    const px = Face.render({ p, wavePhase, heart, z, color, style });
    out.push(Buffer.from(px.buffer, px.byteOffset, px.byteLength));
  } else if (op === "D") {
    const t = next();
    model.update(t);
    let px;
    if (Anime.tableOf(model.style) !== null) {
      anime.follow(model, t);
      px = Anime.render(anime);
    } else {
      px = Face.render(model.shape());
    }
    out.push(Buffer.from(px.buffer, px.byteOffset, px.byteLength));
  } else if (op === "Y") {
    style = next();
    model.setStyle(style, 0);
  } else if (op === "C") {
    const s = next(), t = next();
    model.setStyle(s, t);
  } else if (op === "E") {
    const id = next(), intensity = next(), ms = next(), t = next();
    model.setExpression(id, intensity, ms, t);
  } else if (op === "M") {
    const level = next(), t = next();
    model.setMouth(level, t);
  } else if (op === "G") {
    const t = next();
    model.update(t);
    eyes.follow(model, t);
    const hex = Array.from(eyes.frame.rgb, (v) => v.toString(16).padStart(2, "0")).join("");
    out.push(Buffer.from(`${hex} ${eyes.frame.milliamps} ${eyes.mode()} ${eyes.set}\n`));
  } else if (op === "O") {
    const hex = words[i++], t = next();
    const bytes = Buffer.from(hex, "hex");
    if (bytes.length === Eyes.OVERRIDE_BYTES || bytes.length === Eyes.OVERRIDE_BYTES_EFFECT) {
      eyes.setOverride(Eyes.decodeOverride(bytes), t);
    }
  } else if (op === "B") {
    const v = next();
    next();
    eyes.setBrightness(v);
  } else if (op === "K") {
    eyes.selfTest(next());
  } else if (op === "X") {
    const v = next();
    next();
    eyes.setSet(v);
  } else if (op === "S") {
    model.update(next());
    const s = model.shape();
    const text = [...s.p, s.wavePhase, s.heart, s.z].map((v) => v.toFixed(5)).join(" ");
    out.push(Buffer.from(`${text} ${s.color.join(" ")} ${s.style} ${model.shown} ${model.expression}\n`));
  }
}
process.stdout.write(Buffer.concat(out));
