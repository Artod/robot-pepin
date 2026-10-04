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
      F.render(model.shape(), px);
      renderMs = 0.9 * renderMs + 0.1 * (now() - t0);
      F.toRGBA(px, image.data);
      ctx.putImageData(image, 0, 0);
    }
    frames += 1;
    if (t - fpsAt >= 1000) {
      $("st-fps").textContent = String(frames);
      frames = 0;
      fpsAt = t;
    }
    $("st-expr").textContent = T.expressions[model.expression].name;
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
    b.title = e.note;
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

  requestAnimationFrame(frame);
})();
