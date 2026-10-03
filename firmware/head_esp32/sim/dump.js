// The simulator's model and renderer under node, reading the script test_native/dump.cpp reads
// (see there) and writing the same output: tests/unit/test_face_parity.py compares the two.
"use strict";

const Face = require("./face.js");

const words = require("fs").readFileSync(0, "utf8").split(/\s+/).filter((w) => w.length);
const model = new Face.FaceModel();
const out = [];
let i = 0;
const next = () => Number(words[i++]);
while (i < words.length) {
  const op = words[i++];
  if (op === "R") {
    const p = new Float32Array(Face.T.params.length);
    for (let k = 0; k < p.length; k++) p[k] = next();
    const px = Face.render({ p, wavePhase: Math.fround(next()) });
    out.push(Buffer.from(px.buffer, px.byteOffset, px.byteLength));
  } else if (op === "E") {
    const id = next(), intensity = next(), ms = next(), t = next();
    model.setExpression(id, intensity, ms, t);
  } else if (op === "M") {
    const level = next(), t = next();
    model.setMouth(level, t);
  } else if (op === "S") {
    model.update(next());
    const s = model.shape();
    out.push(Buffer.from([...s.p].map((v) => v.toFixed(5)).join(" ") + " " + s.wavePhase.toFixed(5) + "\n"));
  }
}
process.stdout.write(Buffer.concat(out));
