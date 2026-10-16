// The simulator's page: the firmware's loop (src/main.cpp) and the head server's arbiter
// (pepin.head_server.FaceArbiter) around the shared model and renderer (face.js).
"use strict";

(function () {
  const T = FACE_TABLE;
  const F = Face;
  const $ = (id) => document.getElementById(id);
  const now = () => performance.now();
  const ids = Object.fromEntries(T.expressions.map((e, i) => [e.name, i]));

  const canvas = $("face");
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(F.W, F.H);
  const px = new Uint16Array(F.W * F.H);
  const model = new F.FaceModel();
  const anime = new Anime.AnimeModel();  // the shape engine, following the model in a shapes style
  const eyes = new Eyes.EyeModel();      // the rings, following the model and the 'Y' panel

  // -- the head server's arbiter: each source one standing and one timed request, newest wins
  class Arbiter {
    constructor() { this.held = new Map(); this.seq = 0; }
    set(source, name, intensity, ms, holdS, t) {
      this.seq += 1;
      const timed = holdS !== null && holdS !== undefined;
      this.held.set(source + (timed ? "#timed" : "#standing"), {
        source, name, intensity, ms, until: timed ? t + holdS * 1000 : null, seq: this.seq,
      });
    }
    clear(source) { this.held.delete(source + "#timed"); this.held.delete(source + "#standing"); }
    showing(t) {
      for (const [key, r] of this.held) if (r.until !== null && r.until <= t) this.held.delete(key);
      let best = null;
      for (const r of this.held.values()) if (!best || r.seq > best.seq) best = r;
      return best;
    }
  }
  const arbiter = new Arbiter();
  let sentKey = null;
  let shownBy = null;

  // -- the firmware's state (main.cpp loop) ------------------------------------------------
  let hostConnected = true;
  let lastRx = now();
  let mode = "face";
  let hostExpr = 0;
  let hostIntensity = 1;
  let info = null;
  let infoUntil = 0;

  function receiveExpression(id, intensity, ms) {
    lastRx = now();
    hostExpr = id;
    hostIntensity = intensity;
    if (mode !== "asleep") model.setExpression(id, intensity, ms, now());
  }

  function receiveMouth(level) {
    if (!hostConnected) return;
    lastRx = now();
    model.setMouth(level, now());
    $("meter").style.width = `${Math.round(level * 100)}%`;
  }

  function serverTick(t) {
    const w = arbiter.showing(t);
    const name = w ? w.name : "neutral";
    const intensity = w ? w.intensity : 1;
    shownBy = w ? w.source : null;
    const key = `${name}/${intensity}`;
    if (key !== sentKey && hostConnected) {
      receiveExpression(ids[name], intensity, w ? w.ms : T.timing.transition_ms);
      sentKey = key;
    }
  }

  function firmwareTick(t) {
    if (hostConnected) lastRx = t; // the head server pings at 2 Hz
    const silent = t - lastRx > T.timing.host_silent_s * 1000;
    if (silent && mode !== "asleep") {
      mode = "asleep";
      model.setExpression(ids.sleepy, 1, 900, t);
    } else if (!silent && mode === "asleep") {
      mode = "face";
      model.setExpression(hostExpr, hostIntensity, T.timing.transition_ms, t);
    }
  }

  // -- the info screen (display.cpp display_info) -------------------------------------------
  function showItems(text) { // pepin.head_link.show_items
    const items = [];
    const bar = /^\s*(-?\d+(?:\.\d+)?)\s*(?:\/\s*(\d+(?:\.\d+)?)|(%))\s*(.*)$/;
    for (let line of text.replace(/\|/g, "\n").split("\n")) {
      line = line.trim();
      if (!line) continue;
      const at = line.indexOf(":");
      const key = at >= 0 ? line.slice(0, at).trim() : "";
      const value = at >= 0 ? line.slice(at + 1).trim() : "";
      if (at >= 0 && key && value) {
        const m = value.match(bar);
        if (m) {
          const full = m[3] ? 100 : Number(m[2]);
          items.push({ bar: key, frac: full > 0 ? Number(m[1]) / full : 0, value });
        } else {
          items.push({ key, value });
        }
      } else {
        items.push({ text: line });
      }
    }
    return items.slice(0, 8);
  }

  function drawInfo(items) {
    const c = T.colors;
    ctx.fillStyle = c.background;
    ctx.fillRect(0, 0, F.W, F.H);
    ctx.textBaseline = "top";
    let y = 6;
    let title = true;
    for (const item of items) {
      if (y >= F.H - 14) break;
      ctx.textAlign = "left";
      if (item.text !== undefined) {
        ctx.font = `${title ? 24 : 20}px "Hiragino Sans", sans-serif`;
        ctx.fillStyle = title ? c.accent : c.text;
        ctx.fillText(item.text, 8, y);
        y += title ? 30 : 24;
      } else if (item.key !== undefined) {
        ctx.font = '20px "Hiragino Sans", sans-serif';
        ctx.fillStyle = c.accent;
        ctx.fillText(item.key, 8, y);
        ctx.fillStyle = c.text;
        ctx.textAlign = "right";
        ctx.fillText(item.value, F.W - 8, y);
        y += 24;
      } else {
        ctx.font = '16px "Hiragino Sans", sans-serif';
        ctx.fillStyle = c.text;
        ctx.fillText(item.bar, 8, y + 3);
        // the bar between the label and the value (display.cpp: the same arithmetic)
        const bx = Math.max(110, 16 + Math.round(ctx.measureText(item.bar).width));
        const bw = F.W - bx - Math.round(ctx.measureText(item.value).width) - 16;
        const bh = 14;
        if (bw >= 30) {
          ctx.strokeStyle = c.accent;
          ctx.beginPath(); ctx.roundRect(bx + 0.5, y + 3.5, bw - 1, bh - 1, 4); ctx.stroke();
          const fill = Math.round((bw - 4) * Math.min(Math.max(item.frac, 0), 1));
          if (fill > 0) {
            ctx.fillStyle = c.lip;
            ctx.beginPath(); ctx.roundRect(bx + 2, y + 5, fill, bh - 4, 3); ctx.fill();
          }
        }
        ctx.textAlign = "right";
        ctx.fillText(item.value, F.W - 8, y + 3);
        y += 22;
      }
      title = false;
    }
  }

  // -- the loop -------------------------------------------------------------------------------
  let frames = 0;
  let fpsAt = now();
  let renderMs = 0;
  function frame() {
    const t = now();
    serverTick(t);
    firmwareTick(t);
    const showingInfo = info && t < infoUntil;
    if (showingInfo) {
      drawInfo(info);
    } else {
      info = null;
      model.update(t);
      const t0 = now();
      if (Anime.tableOf(model.style) !== null) {
        anime.follow(model, t);
        Anime.render(anime, px);
      } else {
        F.render(model.shape(), px);
      }
      renderMs = 0.9 * renderMs + 0.1 * (now() - t0);
      F.toRGBA(px, image.data);
      ctx.putImageData(image, 0, 0);
    }
    eyesTick(t);
    frames += 1;
    if (t - fpsAt >= 1000) {
      $("st-fps").textContent = String(frames);
      frames = 0;
      fpsAt = t;
    }
    const shown = model.shown !== model.expression ? ` (${T.expressions[model.shown].name})` : "";
    $("st-expr").textContent = T.expressions[model.expression].name + shown;
    $("st-by").textContent = mode === "asleep" ? "firmware (host silent)" : shownBy || "-";
    $("st-mode").textContent = showingInfo ? "info" : mode;
    $("st-level").textContent = model.level.toFixed(2);
    $("st-ms").textContent = renderMs.toFixed(2);
    requestAnimationFrame(frame);
  }

  // -- controls ------------------------------------------------------------------------------
  const intensity = () => Number($("intensity").value);
  const transition = () => Number($("transition").value);
  $("intensity").oninput = () => { $("intensity-v").textContent = intensity().toFixed(2); };
  $("transition").oninput = () => { $("transition-v").textContent = `${transition()} ms`; };
  $("silent-s").textContent = String(T.timing.host_silent_s);

  for (const e of T.expressions) {
    const b = document.createElement("button");
    b.textContent = e.name;
    b.title = e.brief ? `${e.brief}: ${e.note}` : e.note;
    b.onclick = () => arbiter.set("llm", e.name, intensity(), transition(), null, now());
    $("expressions").appendChild(b);
  }
  const clearLlm = document.createElement("button");
  clearLlm.textContent = "(clear)";
  clearLlm.title = "the expression buttons' source asks for nothing any more";
  clearLlm.onclick = () => arbiter.clear("llm");
  $("expressions").appendChild(clearLlm);

  // Which source sends each event, and whether it ends what that source held (the goal
  // server's drive end; pepin.face_events).
  const EVENT_SOURCES = {
    goal_accepted: ["goal", false], recovery: ["goal", false], arrived: ["goal", true],
    goal_failed: ["goal", true], goal_cancelled: ["goal", true], listening: ["voice", false],
    thinking: ["voice", false], speaking: ["voice", false], brain_lost: ["brain", false],
  };
  let speakingTimer = null;
  for (const [name, ev] of Object.entries(T.events)) {
    const [source, end] = EVENT_SOURCES[name] || ["sim", false];
    const b = document.createElement("button");
    b.textContent = name.replace(/_/g, " ");
    b.title = `${source}: ${ev.name}${ev.hold_s ? ` for ${ev.hold_s} s` : ", held"}`;
    b.onclick = () => {
      const t = now();
      if (name === "brain_lost" && arbiter.held.has("brain#standing")) {
        arbiter.clear("brain");
        b.classList.remove("on");
        return;
      }
      if (end) arbiter.clear(source);
      arbiter.set(source, ev.name, ev.intensity, T.timing.transition_ms, ev.hold_s, t);
      if (name === "brain_lost") b.classList.add("on");
      if (name === "speaking") { // the voice loop: speaks, then clears its own state
        startBabble();
        clearTimeout(speakingTimer);
        speakingTimer = setTimeout(() => { stopBabble(); arbiter.clear("voice"); }, 3500);
      }
    };
    $("events").appendChild(b);
  }
  const voiceDone = document.createElement("button");
  voiceDone.textContent = "voice done";
  voiceDone.title = "the voice loop clears its own state";
  voiceDone.onclick = () => arbiter.clear("voice");
  $("events").appendChild(voiceDone);

  $("host").onchange = () => { hostConnected = $("host").checked; if (hostConnected) sentKey = null; };
  T.styles.forEach((s, i) => {  // the 'C' config's style byte, live
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = s.name;
    option.selected = i === T.style;
    $("style").appendChild(option);
  });
  $("style").onchange = () => model.setStyle(Number($("style").value), now());
  $("smooth").onchange = () => canvas.classList.toggle("smooth", $("smooth").checked);
  $("info-s").oninput = () => { $("info-s-v").textContent = $("info-s").value; };
  $("info-show").onclick = () => {
    if (!hostConnected) return;
    info = showItems($("info-text").value);
    infoUntil = now() + Number($("info-s").value) * 1000;
  };

  // -- speech -------------------------------------------------------------------------------
  const L = T.lipsync;
  const levelOf = (rms) => {
    const db = 20 * Math.log10(rms + 1e-9);
    return Math.min(Math.max((db - L.floor_db) / (L.full_db - L.floor_db), 0), 1);
  };
  let speechTimer = null;
  let speechStop = null;
  function startSpeech(next, label, stop) {
    stopSpeech();
    speechStop = stop || null;
    speechTimer = setInterval(() => receiveMouth(next()), L.window_s * 1000);
    $("speech-note").textContent = label;
  }
  function stopSpeech() {
    if (speechTimer) clearInterval(speechTimer);
    speechTimer = null;
    if (speechStop) speechStop();
    speechStop = null;
    for (const id of ["babble", "mic"]) $(id).classList.remove("on");
    $("meter").style.width = "0";
  }

  // A synthetic voice: syllables of 100-260 ms with raised-cosine loudness, words of 1-4
  // syllables, pauses between words (levels, not audio).
  function babbleSource() {
    let syllable = 0, length = 0, peak = 0, pause = 0, left = 0;
    return () => {
      if (pause > 0) { pause -= 1; return 0; }
      if (syllable >= length) {
        length = Math.round((0.1 + Math.random() * 0.16) / L.window_s);
        peak = 0.35 + Math.random() * 0.65;
        syllable = 0;
        if (left <= 0) { left = 1 + Math.floor(Math.random() * 4); pause = Math.floor(Math.random() * 6); }
        left -= 1;
      }
      syllable += 1;
      return peak * Math.sin(Math.PI * syllable / (length + 1));
    };
  }
  function startBabble() {
    startSpeech(babbleSource(), "babble: synthetic syllables");
    $("babble").classList.add("on");
  }
  function stopBabble() { if ($("babble").classList.contains("on")) stopSpeech(); }
  $("babble").onclick = () => ($("babble").classList.contains("on") ? stopSpeech() : startBabble());

  let audio = null;
  const audioContext = () => (audio = audio || new AudioContext());
  $("mic").onclick = async () => {
    if ($("mic").classList.contains("on")) { stopSpeech(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ac = audioContext();
      const analyser = ac.createAnalyser();
      analyser.fftSize = 1024;
      ac.createMediaStreamSource(stream).connect(analyser);
      const buf = new Float32Array(analyser.fftSize);
      startSpeech(() => {
        analyser.getFloatTimeDomainData(buf);
        let sum = 0;
        for (const v of buf) sum += v * v;
        return levelOf(Math.sqrt(sum / buf.length));
      }, "microphone: RMS per 40 ms", () => stream.getTracks().forEach((tr) => tr.stop()));
      $("mic").classList.add("on");
    } catch (error) {
      $("speech-note").innerHTML = `<span class="warn">no microphone: ${error.message}</span>`;
    }
  };
  $("file").onchange = async () => {
    const file = $("file").files[0];
    if (!file) return;
    const ac = audioContext();
    const buffer = await ac.decodeAudioData(await file.arrayBuffer());
    const data = buffer.getChannelData(0);
    const win = Math.round(L.window_s * buffer.sampleRate);
    const levels = [];
    for (let i = 0; i + win <= data.length; i += win) {
      let sum = 0;
      for (let k = i; k < i + win; k++) sum += data[k] * data[k];
      levels.push(levelOf(Math.sqrt(sum / win)));
    }
    const source = ac.createBufferSource();
    source.buffer = buffer;
    source.connect(ac.destination);
    const start = ac.currentTime + 0.05;
    source.start(start);
    startSpeech(() => {
      const i = Math.floor((ac.currentTime - start) / L.window_s);
      if (i >= levels.length) { setTimeout(stopSpeech, 0); return 0; }
      return i >= 0 ? levels[i] : 0;
    }, `${file.name}: ${levels.length} windows`, () => { try { source.stop(); } catch (_) {} });
  };

  // -- tuning -------------------------------------------------------------------------------
  const sliders = [];
  T.params.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "param";
    row.title = p.note;
    const name = document.createElement("span");
    name.textContent = p.name;
    const input = document.createElement("input");
    Object.assign(input, { type: "range", min: p.min, max: p.max, step: (p.max - p.min) / 200 });
    const out = document.createElement("output");
    input.oninput = () => {
      out.textContent = Number(input.value).toFixed(2);
      model.setTarget(Float32Array.from(sliders.map((s) => Number(s.value))), 0, now());
    };
    row.append(name, input, out);
    $("params").appendChild(row);
    sliders[i] = input;
    sliders[i].out = out;
  });
  $("tune").ontoggle = () => {
    if (!$("tune").open) return;
    sliders.forEach((s, i) => { s.value = model.to[i]; s.out.textContent = Number(model.to[i]).toFixed(2); });
  };
  $("copy").onclick = async () => {
    const params = {};
    T.params.forEach((p, i) => {
      const v = Math.round(Number(sliders[i].value) * 100) / 100;
      if (Math.abs(v - p.default) > 1e-6) params[p.name] = v;
    });
    const text = JSON.stringify({ name: "new", params, note: "" });
    try { await navigator.clipboard.writeText(text); $("copied").textContent = "copied"; }
    catch (_) { $("copied").textContent = text; }
  };

  // -- the eyes ------------------------------------------------------------------------------
  const ET = Eyes.T;
  const eyeScale = 3;  // px per mm
  let geo = Eyes.geometry(eyeScale);
  const headCanvas = $("eyes"), rawCanvas = $("eyes-raw");
  const headCtx = headCanvas.getContext("2d"), rawCtx = rawCanvas.getContext("2d");
  let headImage = headCtx.createImageData(geo.w, geo.h);
  let rawImage = rawCtx.createImageData(geo.w, geo.h);
  const EYE_MODES = ["expression", "override standing", "override timed", "self-test"];
  let paintMs = 0;
  $("st-eyes-layout").textContent = `0x${ET.layout_id.toString(16).padStart(2, "0")}`;

  function markChain(ctx) {  // the first LED of each ring ringed, an arrow the chain's way
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5;
    for (const r of chainRings()) {
      const p = geo.pos[r.first], q = geo.pos[r.first + 1];
      ctx.beginPath(); ctx.arc(p.x, p.y, ET.layout.led_mm * eyeScale + 2.5, 0, 2 * Math.PI); ctx.stroke();
      const dx = q.x - p.x, dy = q.y - p.y, n = Math.hypot(dx, dy) || 1;
      const ax = p.x + (dx / n) * 10, ay = p.y + (dy / n) * 10;
      ctx.beginPath(); ctx.moveTo(p.x + (dx / n) * 5, p.y + (dy / n) * 5); ctx.lineTo(ax, ay); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ax, ay);
      ctx.lineTo(ax - (dx / n) * 4 - (dy / n) * 3, ay - (dy / n) * 4 + (dx / n) * 3);
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - (dx / n) * 4 + (dy / n) * 3, ay - (dy / n) * 4 - (dx / n) * 3); ctx.stroke();
    }
    ctx.fillStyle = "#8495a5"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("R", geo.centres[1].x, geo.h - 4);
    ctx.fillText("L", geo.centres[0].x, geo.h - 4);
  }

  function eyesTick(t) {
    eyes.follow(model, t);
    const t0 = now();
    const rgb = eyes.frame.rgb;
    if ($("eyes-film").checked) Eyes.paintFilm(rgb, headImage.data, geo);
    else Eyes.paintRaw(rgb, headImage.data, geo);
    headCtx.putImageData(headImage, 0, 0);
    Eyes.paintRaw(rgb, rawImage.data, geo);
    rawCtx.putImageData(rawImage, 0, 0);
    markChain(rawCtx);
    paintMs = 0.9 * paintMs + 0.1 * (now() - t0);
    $("st-eyes-ma").textContent = String(eyes.frame.milliamps);
    $("st-eyes-b").textContent = String(eyes.brightness);
    $("st-eyes-mode").textContent = EYE_MODES[eyes.mode()];
    $("st-eyes-fx").textContent = ET.effects[eyes.effect];
    $("st-eyes-set").textContent = ET.sets[eyes.set].name;
    $("st-eyes-ms").textContent = paintMs.toFixed(2);
  }

  $("eyes-brightness").oninput = () => {  // the 'C' config's ninth byte
    eyes.setBrightness(Number($("eyes-brightness").value));
    $("eyes-brightness-v").textContent = $("eyes-brightness").value;
  };
  $("eyes-selftest").onclick = () => eyes.selfTest(now());
  $("eyes-set").onchange = () => eyes.setSet(Number($("eyes-set").value));  // the 'C' config's tenth byte

  // -- the layout: four chain positions, each a ring, its first LED's angle and its winding
  const ringNames = ET.sides.flatMap((s) => ET.ring_kinds.map((k) => `${s} ${k}`));
  const ringRows = [];
  ET.rings.forEach((r, k) => {
    const row = document.createElement("div");
    row.className = "ring";
    const pos = document.createElement("span");
    pos.textContent = `chain ${k}`;
    const which = document.createElement("select");
    ringNames.forEach((name, i) => {
      const o = document.createElement("option");
      o.value = String(i); o.textContent = name; o.selected = i === r.eye * ET.ring_kinds.length + r.kind;
      which.appendChild(o);
    });
    const deg = document.createElement("input");
    Object.assign(deg, { type: "range", min: 0, max: 359, step: 1, value: r.first_deg });
    const out = document.createElement("output");
    out.textContent = `${r.first_deg} deg`;
    const cw = document.createElement("label");
    const cwBox = document.createElement("input");
    cwBox.type = "checkbox"; cwBox.checked = r.clockwise;
    cw.append(cwBox, "clockwise");
    row.append(pos, which, deg, out, cw);
    $("rings").appendChild(row);
    ringRows.push({ which, deg, out, cwBox });
    which.onchange = deg.oninput = cwBox.onchange = relayout;
  });
  function chainRings() {
    let first = 0;
    return ringRows.map((r) => {
      const i = Number(r.which.value);
      const eye = Math.floor(i / ET.ring_kinds.length), kind = i % ET.ring_kinds.length;
      const n = ET.rings.find((x) => x.kind === kind).n;
      const ring = { eye, kind, n, first, first_deg: Number(r.deg.value), clockwise: r.cwBox.checked };
      first += n;
      return ring;
    });
  }
  function relayout() {
    ringRows.forEach((r) => { r.out.textContent = `${r.deg.value} deg`; });
    Eyes.relayout(chainRings());
    geo = Eyes.geometry(eyeScale);
  }
  $("layout-copy").onclick = async () => {
    const rings = chainRings().map((r) => ({
      eye: ET.sides[r.eye], ring: ET.ring_kinds[r.kind], first_deg: r.first_deg, clockwise: r.clockwise,
    }));
    const text = JSON.stringify(rings);
    try { await navigator.clipboard.writeText(text); $("layout-copied").textContent = "copied"; }
    catch (_) { $("layout-copied").textContent = text; }
  };

  // -- the 'Y' panel: the frame's fields, built as the head server would and decoded as the
  // firmware does
  const Y_FIELDS = [  // name, byte, signed
    ["brow_pos", 3, true], ["brow_raise", 4, true], ["brow_width", 5, false], ["lid_top", 6, false],
    ["lid_bottom", 7, false], ["gaze_x", 8, true], ["gaze_y", 9, true], ["glow", 13, false],
  ];
  const yRows = {};
  for (const [name, , signed] of Y_FIELDS) {
    const row = document.createElement("div");
    row.className = "yfield";
    const label = document.createElement("span");
    label.textContent = name;
    const keep = document.createElement("label");
    const keepBox = document.createElement("input");
    keepBox.type = "checkbox"; keepBox.checked = true;
    keep.append(keepBox, "keep");
    const input = document.createElement("input");
    Object.assign(input, { type: "range", min: signed ? -100 : 0, max: signed ? 100 : 254, step: 1, value: 0 });
    const out = document.createElement("output");
    out.textContent = "0";
    input.oninput = () => { out.textContent = input.value; keepBox.checked = false; };
    row.append(label, keep, input, out);
    $("y-fields").appendChild(row);
    yRows[name] = { keepBox, input };
  }
  ["keep", ...ET.effects].forEach((name, i) => {  // the effect byte: 0xFF keeps, else an id
    const o = document.createElement("option");
    o.value = i === 0 ? "255" : String(i - 1);
    o.textContent = i === 0 ? "(keep)" : name;
    $("y-effect").appendChild(o);
  });
  function yPayload(blink, modeOverride) {
    const effect = Number($("y-effect").value);
    const p = new Uint8Array(effect === 255 ? Eyes.OVERRIDE_BYTES : Eyes.OVERRIDE_BYTES_EFFECT);
    const mode = modeOverride !== undefined ? modeOverride : Number($("y-mode").value);
    const ms = Math.min(65535, Math.max(0, Number($("y-ms").value) || 0));
    p[0] = mode; p[1] = ms & 255; p[2] = ms >> 8;
    for (const [name, at, signed] of Y_FIELDS) {
      const r = yRows[name];
      if (r.keepBox.checked) { p[at] = signed ? Eyes.KEEP_I8 : Eyes.KEEP_U8; continue; }
      const v = Number(r.input.value);
      p[at] = signed ? (v & 255) : v;
    }
    if (!$("y-iris-keep").checked) {
      const c = parseInt($("y-iris").value.slice(1), 16);
      p[10] = (c >> 16) & 255; p[11] = (c >> 8) & 255; p[12] = c & 255;
    }
    p[14] = blink;
    p[15] = Number($("y-side").value);
    if (effect !== 255) {
      const fxMs = Math.min(65535, Math.max(0, Number($("y-effect-ms").value) || 0));
      p[16] = effect; p[17] = fxMs & 255; p[18] = fxMs >> 8;
    }
    return p;
  }
  function sendY(blink, mode) {
    if (!hostConnected) return;
    const p = yPayload(blink, mode);
    $("y-hex").textContent = Array.from(p, (v) => v.toString(16).padStart(2, "0")).join(" ");
    eyes.setOverride(Eyes.decodeOverride(p), now());
  }
  $("y-send").onclick = () => sendY(0);
  $("y-blink").onclick = () => sendY(1);
  $("y-wink-l").onclick = () => sendY(2);
  $("y-wink-r").onclick = () => sendY(3);
  $("y-release").onclick = () => sendY(0, 0);
  $("y-iris").oninput = () => { $("y-iris-keep").checked = false; };

  requestAnimationFrame(frame);
})();
