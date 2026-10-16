#include "eyes.h"

#include <math.h>

namespace eyes {

static_assert(kStyleCount == face::kStyleCount, "eye_table.h and face_table.h disagree");
static_assert(kExpressionCount == face::kExpressionCount, "eye_table.h and face_table.h disagree");
static_assert(kLedCount <= 255, "the self-test's index is a byte");

namespace {

constexpr float kFullTurn = 360.0f;
constexpr float kHalfTurn = 180.0f;
constexpr float kBrowMaxDeg = 85.0f;      // the brow's centre never leaves the top half
constexpr float kTestLevel = 0.4f;        // the self-test's LED, perceptual
constexpr float kTestTail = 0.12f;        // the one behind it
constexpr float kRateFloor = 0.05f;       // a blink rate under this: no blinks
constexpr float kColorScale = 1.0f / 255.0f;
constexpr float kLidTopmost = 1.0f + kLidSoft * 0.5f;  // a lid's edge at 0: no LED touched
constexpr float kLidSpan = 2.0f + kLidSoft;            // and at 1: every LED covered

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Degrees wrapped to (-180, 180].
float wrap(float deg) {
  while (deg > kHalfTurn) deg -= kFullTurn;
  while (deg <= -kHalfTurn) deg += kFullTurn;
  return deg;
}

// 0xRRGGBB as three floats, 0..255.
void rgb_of(uint32_t c, float out[3]) {
  out[0] = (float)((c >> 16) & 0xFFu);
  out[1] = (float)((c >> 8) & 0xFFu);
  out[2] = (float)(c & 0xFFu);
}

// A LED's share of an arc centred `centre_deg` with the half width `half_deg`, its edge
// softened over kBrowSoftDeg: 1 inside, 0 outside.
float arc(float led_deg, float centre_deg, float half_deg) {
  const float d = fabsf(wrap(led_deg - centre_deg));
  return 1.0f - face::ease((d - half_deg) / kBrowSoftDeg + 0.5f);
}

}  // namespace

Override decode_override(const uint8_t* p, size_t n) {
  Override o = {};
  o.mode = p[0];
  o.ms = (uint16_t)(p[1] | (p[2] << 8));
  // i8 fields: -100..100 in hundredths, 0x80 keeps; u8 fields: 0..254 over the range, 0xFF keeps
  const struct { int at; int param; bool wide; } fields[] = {
      {3, kBrowPos, true},    {4, kBrowRaise, true},  {5, kBrowWidth, false}, {6, kLidTop, false},
      {7, kLidBottom, false}, {8, kGazeX, true},      {9, kGazeY, true},      {13, kGlow, false},
  };
  for (const auto& f : fields) {
    const uint8_t b = p[f.at];
    if (f.wide) {
      if (b == kKeepI8) continue;
      const int v = b < 128 ? (int)b : (int)b - 256;
      o.value[f.param] = (float)(v < -100 ? -100 : (v > 100 ? 100 : v)) * 0.01f;
    } else {
      if (b == kKeepU8) continue;
      o.value[f.param] = (float)b * (1.0f / 254.0f);
    }
    o.value[f.param] = clampf(o.value[f.param], kParamMin[f.param], kParamMax[f.param]);
    o.mask |= 1u << f.param;
  }
  o.tinted = (p[10] | p[11] | p[12]) != 0;
  for (int k = 0; k < 3; ++k) o.tint[k] = (float)p[10 + k];
  o.blink = p[14];
  o.side = p[15];
  if (n >= kOverrideBytesEffect && p[16] != kKeepU8) {
    o.has_effect = true;
    o.effect = p[16] < kEffectCount ? p[16] : (uint8_t)kFxNone;
    o.effect_ms = (uint16_t)(p[17] | (p[18] << 8));
  }
  return o;
}

namespace {

bool one_shot(uint8_t effect) { return effect == kFxSparkle || effect == kFxFlash; }

// A smooth wave of the phase (periods), -1..1: a triangle eased (no transcendental call: the
// simulator must land on the same bytes).
float wave_of(float phase) {
  phase -= floorf(phase);
  const float tri = phase < 0.5f ? 2.0f * phase : 2.0f - 2.0f * phase;
  return 2.0f * face::ease(tri) - 1.0f;
}

}  // namespace

EyeModel::EyeModel() {
  set_ = kEyeSetOfStyle[face::kStyleDefault];
  for (int i = 0; i < kParamCount; ++i) look_[i] = kLooks[set_][face::kStyleDefault][face::kNeutral][i];
  for (int e = 0; e < kElementCount; ++e) {
    rgb_of(kColors[set_][face::kStyleDefault][face::kNeutral][e], clook_[e]);
  }
  for (int eye = 0; eye < kEyeCount; ++eye) {
    ov_mask_[eye] = 0;
    ov_tinted_[eye] = false;
    for (int k = 0; k < 3; ++k) ov_tint_[eye][k] = 0.0f;
    for (int i = 0; i < kParamCount; ++i) {
      ov_value_[eye][i] = 0.0f;
      from_[eye][i] = to_[eye][i] = cur_[eye][i] = look_[i];
    }
    for (int e = 0; e < kElementCount; ++e) {
      for (int k = 0; k < 3; ++k) cfrom_[eye][e][k] = cto_[eye][e][k] = ccur_[eye][e][k] = clook_[e][k];
    }
    blink_on_[eye] = false;
    blink_t0_[eye] = 0;
    blink_close_[eye] = blink_hold_[eye] = blink_open_[eye] = 0.0f;
    blink_count_[eye] = 1;
    inner_first_[eye] = inner_n_[eye] = 0;
  }
  for (int r = 0; r < kRingCount; ++r) {
    if (kRings[r].kind == kInner) {
      inner_first_[kRings[r].eye] = kRings[r].first;
      inner_n_[kRings[r].eye] = kRings[r].n;
    }
  }
  for (int i = 0; i < kLedCount; ++i) {
    twinkle_t0_[i] = 0;
    twinkle_on_[i] = false;
  }
  sacc_from_[0] = sacc_from_[1] = sacc_to_[0] = sacc_to_[1] = 0.0f;
  for (int i = 0; i < kLedCount * 3; ++i) light_[i] = 0.0f;
  for (int i = 0; i < kLedCount * 3; ++i) frame_.rgb[i] = 0;
  frame_.milliamps = 0;
}

void EyeModel::seed(uint32_t s) { rng_ = s != 0u ? s : 0x9E3779B9u; }

uint32_t EyeModel::nextRandom() {
  uint32_t x = rng_;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  rng_ = x;
  return x;
}

float EyeModel::uniform() { return (float)(nextRandom() >> 8) * (1.0f / 16777216.0f); }

void EyeModel::lookOf(int set, int style, int shown, float intensity) {
  for (int i = 0; i < kParamCount; ++i) {
    const float n = kLooks[set][style][face::kNeutral][i];
    look_[i] = n + (kLooks[set][style][shown][i] - n) * intensity;
  }
  for (int e = 0; e < kElementCount; ++e) {
    float n[3], x[3];
    rgb_of(kColors[set][style][face::kNeutral][e], n);
    rgb_of(kColors[set][style][shown][e], x);
    for (int k = 0; k < 3; ++k) clook_[e][k] = n[k] + (x[k] - n[k]) * intensity;
  }
}

// The tween to the effective target (the look, the override's fields over it) from where
// the eyes are now.
void EyeModel::retarget(float ms, uint32_t t0_ms) {
  for (int eye = 0; eye < kEyeCount; ++eye) {
    for (int i = 0; i < kParamCount; ++i) {
      from_[eye][i] = cur_[eye][i];
      to_[eye][i] = ((ov_mask_[eye] >> i) & 1u) ? ov_value_[eye][i] : look_[i];
    }
    for (int e = 0; e < kElementCount; ++e) {
      const bool tinted = ov_tinted_[eye] && (e == kElIris || e == kElBrow);
      for (int k = 0; k < 3; ++k) {
        cfrom_[eye][e][k] = ccur_[eye][e][k];
        cto_[eye][e][k] = tinted ? ov_tint_[eye][k] : clook_[e][k];
      }
    }
  }
  t0_ms_ = t0_ms;
  dur_ms_ = ms > 0.0f ? ms : 0.0f;
  if (dur_ms_ == 0.0f) {
    for (int eye = 0; eye < kEyeCount; ++eye) {
      for (int i = 0; i < kParamCount; ++i) cur_[eye][i] = to_[eye][i];
      for (int e = 0; e < kElementCount; ++e) {
        for (int k = 0; k < 3; ++k) ccur_[eye][e][k] = cto_[eye][e][k];
      }
    }
  }
}

void EyeModel::setOverride(const Override& o, uint32_t now_ms) {
  const bool left = o.side != 2, right = o.side != 1;
  if (o.mode == 0) {  // release: both eyes back to the expression's look
    for (int eye = 0; eye < kEyeCount; ++eye) {
      ov_mask_[eye] = 0;
      ov_tinted_[eye] = false;
    }
    ov_timed_ = false;
  } else {
    for (int eye = 0; eye < kEyeCount; ++eye) {
      if (!(eye == kLeft ? left : right)) continue;
      ov_mask_[eye] = o.mask;
      for (int i = 0; i < kParamCount; ++i) ov_value_[eye][i] = o.value[i];
      ov_tinted_[eye] = o.tinted;
      for (int k = 0; k < 3; ++k) ov_tint_[eye][k] = o.tint[k];
    }
    ov_timed_ = o.mode == 2;
    ov_until_ms_ = now_ms + o.ms;
  }
  retarget(kOverrideMs, now_ms);
  if (o.blink == 1 || o.blink == 2) startBlink(kLeft, o.blink == 2, 1.0f, 1, now_ms);
  if (o.blink == 1 || o.blink == 3) startBlink(kRight, o.blink == 3, 1.0f, 1, now_ms);
  if (o.mode == 0 || (o.has_effect && o.effect == kFxNone)) {
    if (host_effect_on_) {  // back to the look's own; a one-shot of the look does not refire
      host_effect_on_ = false;
      effect_source_ = look_effect_;
      effect_ = one_shot(look_effect_) ? (uint8_t)kFxNone : look_effect_;
      effect_t0_ms_ = now_ms;
    }
  } else if (o.has_effect) {
    host_effect_on_ = true;
    host_effect_ = o.effect;
    host_effect_until_ms_ = o.effect_ms > 0 ? now_ms + o.effect_ms : 0;
    effect_source_ = 0xFF;  // starts (again) on the next frame
  }
}

void EyeModel::selfTest(uint32_t now_ms) {
  test_on_ = true;
  test_t0_ms_ = now_ms;
  limit_ma_ = kCapMa;  // the sweep needs no soft start; the look after it ramps up from it
  ramping_ = false;
}

uint8_t EyeModel::mode() const {
  if (test_on_) return kModeSelfTest;
  for (int eye = 0; eye < kEyeCount; ++eye) {
    if (ov_mask_[eye] != 0 || ov_tinted_[eye]) return ov_timed_ ? kModeTimed : kModeStanding;
  }
  return kModeExpression;
}

void EyeModel::startBlink(int eye, bool wink, float rate, int count, uint32_t now_ms) {
  blink_on_[eye] = true;
  blink_t0_[eye] = now_ms;
  blink_close_[eye] = kBlinkCloseMs / rate;
  blink_hold_[eye] = wink ? kWinkHoldMs : 0.0f;
  blink_open_[eye] = kBlinkOpenMs / rate;
  blink_count_[eye] = count;
}

// The blink's closure on `eye` at now_ms, 0 open .. 1 closed, `count` closures in a row; it
// ends itself.
float EyeModel::blinkOf(int eye, uint32_t now_ms) {
  if (!blink_on_[eye]) return 0.0f;
  float t = (float)(int32_t)(now_ms - blink_t0_[eye]);
  const float cycle = blink_close_[eye] + blink_hold_[eye] + blink_open_[eye];
  int n = 0;
  while (t >= cycle && n < blink_count_[eye]) {
    t -= cycle;
    ++n;
  }
  if (n >= blink_count_[eye]) {
    blink_on_[eye] = false;
    return 0.0f;
  }
  if (t < blink_close_[eye]) return face::ease(t / blink_close_[eye]);
  const float held = t - blink_close_[eye];
  if (held < blink_hold_[eye]) return 1.0f;
  const float opening = held - blink_hold_[eye];
  if (opening < blink_open_[eye]) return 1.0f - face::ease(opening / blink_open_[eye]);
  blink_on_[eye] = false;
  return 0.0f;
}

void EyeModel::follow(const face::FaceModel& model, uint32_t now_ms) {
  // the set: the 'C' choice, else the one the face style follows; a change retargets now
  uint8_t set = set_choice_ >= 1 && set_choice_ <= kEyeSetCount ? (uint8_t)(set_choice_ - 1)
                                                                 : kEyeSetOfStyle[model.style()];
  const bool set_changed = set != set_;
  if (!primed_ || model.starts() != starts_ || model.style() != style_ || set_changed) {
    set_ = set;
    lookOf(set_, model.style(), model.shown(), model.intensity());
    look_effect_ = model.intensity() > 0.0f ? kLookEffect[set_][model.style()][model.shown()]
                                            : (uint8_t)kFxNone;
    if (one_shot(look_effect_) && !host_effect_on_ && !set_changed) effect_source_ = 0xFF;
    if (!primed_) {
      retarget(0.0f, now_ms);
      primed_ = true;
    } else if (set_changed) {
      retarget(kOverrideMs, now_ms);  // the other set's look, over the override's tween
    } else {
      retarget(model.transitionMs(), model.transitionStartMs());  // on the model's clock
    }
    starts_ = model.starts();
    style_ = model.style();
  }
  if (ov_timed_ && (int32_t)(now_ms - ov_until_ms_) >= 0) {  // the timed override lapsed
    for (int eye = 0; eye < kEyeCount; ++eye) {
      ov_mask_[eye] = 0;
      ov_tinted_[eye] = false;
    }
    ov_timed_ = false;
    retarget(kOverrideMs, now_ms);
  }
  const face::Shape ps = model.shape();
  heart_ = ps.heart;
  z_ = ps.z;
  update(now_ms);
}

void EyeModel::update(uint32_t now_ms) {
  if (!started_) {
    started_ = true;
    last_ms_ = now_ms;
  }
  float dt = (float)(int32_t)(now_ms - last_ms_) * 0.001f;
  dt = clampf(dt, 0.0f, 0.25f);
  last_ms_ = now_ms;

  const float elapsed = (float)(int32_t)(now_ms - t0_ms_);
  const float e = dur_ms_ > 0.0f ? face::ease(elapsed / dur_ms_) : 1.0f;
  for (int eye = 0; eye < kEyeCount; ++eye) {
    for (int i = 0; i < kParamCount; ++i) cur_[eye][i] = from_[eye][i] + (to_[eye][i] - from_[eye][i]) * e;
    for (int c = 0; c < kElementCount; ++c) {
      for (int k = 0; k < 3; ++k) {
        ccur_[eye][c][k] = cfrom_[eye][c][k] + (cto_[eye][c][k] - cfrom_[eye][c][k]) * e;
      }
    }
  }
  breath_phase_ += kBreathHz * dt;
  breath_phase_ -= floorf(breath_phase_);

  // the blinks: both eyes together every kBlinkMinS..kBlinkMaxS, divided by the look's rate
  if (!blink_scheduled_) {
    blink_scheduled_ = true;
    blink_next_ms_ = now_ms + (uint32_t)((kBlinkMinS + (kBlinkMaxS - kBlinkMinS) * uniform()) * 1000.0f);
  }
  if ((int32_t)(now_ms - blink_next_ms_) >= 0) {
    const float rate = clampf(cur_[kLeft][kBlinkRate], 0.0f, kParamMax[kBlinkRate]);
    float span_s = kBlinkMinS + (kBlinkMaxS - kBlinkMinS) * uniform();
    if (rate >= kRateFloor) {
      span_s /= rate;
      if (!blink_on_[kLeft] && !blink_on_[kRight]) {
        const int count = uniform() < kFlutter ? 3 : 1;  // now and then a flutter of three
        startBlink(kLeft, false, rate, count, now_ms);
        startBlink(kRight, false, rate, count, now_ms);
      }
    }
    blink_next_ms_ = now_ms + (uint32_t)(span_s * 1000.0f);
  }

  // the micro-saccades: a new small offset of the gaze now and then, reached in kSaccadeMs
  if (!sacc_scheduled_) {
    sacc_scheduled_ = true;
    sacc_t0_ms_ = now_ms;
    sacc_next_ms_ = now_ms + (uint32_t)((kSaccadeMinS + (kSaccadeMaxS - kSaccadeMinS) * uniform()) * 1000.0f);
  }
  if ((int32_t)(now_ms - sacc_next_ms_) >= 0) {
    const float k = clampf((float)(int32_t)(now_ms - sacc_t0_ms_) / kSaccadeMs, 0.0f, 1.0f);
    for (int a = 0; a < 2; ++a) {
      sacc_from_[a] = sacc_from_[a] + (sacc_to_[a] - sacc_from_[a]) * face::ease(k);
      sacc_to_[a] = kSaccadeAmp * (2.0f * uniform() - 1.0f);
    }
    sacc_t0_ms_ = now_ms;
    sacc_next_ms_ = now_ms + (uint32_t)((kSaccadeMinS + (kSaccadeMaxS - kSaccadeMinS) * uniform()) * 1000.0f);
  }

  updateEffect(now_ms, dt);

  if (test_on_ && (float)(int32_t)(now_ms - test_t0_ms_) >= kSelfTestMs) test_on_ = false;
  if (test_on_) {
    renderSelfTest(now_ms);
    power(now_ms, true);
  } else {
    render(now_ms);
    power(now_ms, false);
  }
}

// Which effect runs: the host's while it stands (its life over: back to the look's), else
// the look's default; a change starts the new one; a one-shot ends on its own. The sparkle's
// twinkles are spawned and the flicker's level drawn here.
void EyeModel::updateEffect(uint32_t now_ms, float dt) {
  uint8_t wanted = look_effect_;
  if (host_effect_on_) {
    if (host_effect_until_ms_ != 0 && (int32_t)(now_ms - host_effect_until_ms_) >= 0) {
      host_effect_on_ = false;
      effect_source_ = look_effect_;  // the look's own again, a one-shot of it not refired
      effect_ = one_shot(look_effect_) ? (uint8_t)kFxNone : look_effect_;
      effect_t0_ms_ = now_ms;
    } else {
      wanted = host_effect_;
    }
  }
  if (wanted != effect_source_) {
    effect_source_ = wanted;
    effect_ = wanted;
    effect_t0_ms_ = now_ms;
    for (int i = 0; i < kLedCount; ++i) twinkle_on_[i] = false;
    flicker_level_ = 1.0f;
    flicker_next_ms_ = now_ms;
  }
  const float elapsed = (float)(int32_t)(now_ms - effect_t0_ms_);
  if (effect_ == kFxSparkle) {
    if (elapsed >= kSparkleMs + 2.0f * kSparkleRiseMs) {
      effect_ = kFxNone;
    } else if (elapsed < kSparkleMs) {
      for (int eye = 0; eye < kEyeCount; ++eye) {
        if (uniform() < kSparkleRate * dt) {
          const int led = inner_first_[eye] + (int)(nextRandom() % (uint32_t)inner_n_[eye]);
          twinkle_on_[led] = true;
          twinkle_t0_[led] = now_ms;
        }
      }
    }
  } else if (effect_ == kFxFlash) {
    if (elapsed >= kFlashMs) effect_ = kFxNone;
  } else if (effect_ == kFxFlicker) {
    if ((int32_t)(now_ms - flicker_next_ms_) >= 0) {
      flicker_level_ = kFlickerFloor + (1.0f - kFlickerFloor) * uniform();
      flicker_next_ms_ = now_ms + (uint32_t)(kFlickerMinMs + (kFlickerMaxMs - kFlickerMinMs) * uniform());
    }
  }
}

void EyeModel::renderSelfTest(uint32_t now_ms) {
  const float k = (float)(int32_t)(now_ms - test_t0_ms_) / kSelfTestMs;
  int at = (int)(k * (float)kLedCount);
  if (at < 0) at = 0;
  if (at >= kLedCount) at = kLedCount - 1;
  const int behind = at > 0 ? at - 1 : kLedCount - 1;
  for (int i = 0; i < kLedCount; ++i) {
    const float v = i == at ? kTestLevel : (i == behind ? kTestTail : 0.0f);
    light_[3 * i] = light_[3 * i + 1] = light_[3 * i + 2] = v;
  }
}

void EyeModel::render(uint32_t now_ms) {
  // the breath: a smooth triangle of the phase, -1..1 (no transcendental call: the simulator
  // must land on the same bytes)
  const float tri = breath_phase_ < 0.5f ? 2.0f * breath_phase_ : 2.0f - 2.0f * breath_phase_;
  const float wave = 2.0f * face::ease(tri) - 1.0f;
  const float sk = clampf((float)(int32_t)(now_ms - sacc_t0_ms_) / kSaccadeMs, 0.0f, 1.0f);
  const float sacc_x = sacc_from_[0] + (sacc_to_[0] - sacc_from_[0]) * face::ease(sk);
  const float sacc_y = sacc_from_[1] + (sacc_to_[1] - sacc_from_[1]) * face::ease(sk);
  const int lobe_power = (int)kHighlightPower;
  // the effect's modulation this frame: a gain on the glow, white added to both rings, the
  // lids sinking, the spin's head
  const float fx_elapsed_s = (float)(int32_t)(now_ms - effect_t0_ms_) * 0.001f;
  float fx_gain = 1.0f, fx_white = 0.0f, fx_sink = 0.0f, spin_head = 0.0f;
  bool breath_on = true;
  if (effect_ == kFxFlash) {
    fx_white = kFlashGain * (1.0f - face::ease(fx_elapsed_s * 1000.0f / kFlashMs));
  } else if (effect_ == kFxSpin) {
    spin_head = fx_elapsed_s * kSpinHz * kFullTurn;
    spin_head -= floorf(spin_head / kFullTurn) * kFullTurn;
    fx_gain = kSpinDim;
  } else if (effect_ == kFxPulse) {
    fx_gain = 1.0f + kPulseDepth * wave_of(fx_elapsed_s * kPulseHz);
  } else if (effect_ == kFxFlicker) {
    fx_gain = flicker_level_;
  } else if (effect_ == kFxDoze) {
    const float exhale = 0.5f + 0.5f * wave_of(fx_elapsed_s * kDozeHz);
    fx_gain = 1.0f - kDozeDepth * exhale;
    fx_sink = kDozeSink * exhale;
    breath_on = false;
  }
  // the tear's progress down the arc, 0..1 while it falls, -1 between drops
  float tear_k = -1.0f;
  if (effect_ == kFxTear) {
    float in_period = fx_elapsed_s * 1000.0f;
    in_period -= floorf(in_period / kTearPeriodMs) * kTearPeriodMs;
    if (in_period < kTearFallMs) tear_k = in_period / kTearFallMs;
  }
  const float tear_rgb[3] = {kTearR, kTearG, kTearB};

  for (int eye = 0; eye < kEyeCount; ++eye) {
    const float* p = cur_[eye];
    const float side = eye == kLeft ? 1.0f : -1.0f;  // the asymmetries' sign
    // the nose: clockwise from the top (as seen) on the robot's right eye, which sits on the
    // viewer's left; the other way on its left eye
    const float nose = eye == kRight ? 1.0f : -1.0f;
    const float raise = clampf(p[kBrowRaise] + p[kBrowAsym] * side, -1.0f, 1.0f);
    const float wink = clampf(p[kWink] * side, 0.0f, 1.0f);
    const float blink = blinkOf(eye, now_ms);
    const float shut = wink > blink ? wink : blink;
    // a closing eye: dark (the top lid down over everything), or in a shut_arc set the upper
    // arc alone (the bottom lid rising to the set's shut_lid)
    float lid_top = p[kLidTop];
    float lid_bottom = p[kLidBottom];
    if (kEyeSetShutArc[set_]) {
      const float lid = kEyeSetShutLid[set_];
      if (lid_bottom < lid) lid_bottom += (lid - lid_bottom) * shut;
    } else {
      lid_top += (1.0f - lid_top) * shut;
    }
    if (z_ >= 0.0f) lid_top += kZSink * z_;  // the sleeper's z: the lids sink as it rises
    lid_top += fx_sink;
    lid_top = clampf(lid_top, 0.0f, 1.0f);
    lid_bottom = clampf(lid_bottom, 0.0f, 1.0f);
    // a lid's edge by height: from above the top LED (nothing covered) to below the bottom one
    // (all covered) as lid^1.5, so the first tenths only touch the LED nearest the lid
    const float edge_top = kLidTopmost - kLidSpan * lid_top * sqrtf(lid_top);
    const float edge_bottom = -kLidTopmost + kLidSpan * lid_bottom * sqrtf(lid_bottom);
    float brow_at = -p[kBrowPos] * kBrowSpanDeg * (1.0f - kBrowPull * raise);
    if (raise < 0.0f) brow_at += raise * kBrowSagDeg;  // a lowered brow sags outward
    brow_at = clampf(brow_at, -kBrowMaxDeg, kBrowMaxDeg);
    const float brow_half = kBrowHalfMinDeg + (kBrowHalfMaxDeg - kBrowHalfMinDeg) * p[kBrowWidth];
    const float brow_bright = p[kBrowGlow] * clampf(kBrowDim + kBrowLift * raise, 0.0f, 1.0f);
    const float lower_half =
        kLowerLidHalfMinDeg + (kLowerLidHalfMaxDeg - kLowerLidHalfMinDeg) * lid_bottom;
    const float lower_bright = kLowerLidGlow * lid_bottom;
    const float cheek_bright = kCheekGlow * p[kCheek];
    // the pupil highlight: toward the gaze (the robot's right is the viewer's left) plus the
    // resting catchlight (outward and up), in the viewer's frame
    const float gaze_x = p[kGazeX] * kGazeGain + sacc_x;
    const float gaze_y = p[kGazeY] * kGazeGain + sacc_y;
    const float px = -gaze_x - nose * kCatchlightOut;
    const float py = gaze_y + kCatchlightUp;
    const float pm = sqrtf(px * px + py * py);
    const float breath = breath_on ? 1.0f + kBreathDepth * p[kBreath] * wave : 1.0f;
    const float glow = p[kGlow] * breath * fx_gain;
    const float sparkle_at = kSparkleTurns * kFullTurn * heart_;  // degrees toward the nose

    for (int i = 0; i < kLedCount; ++i) {
      if (kLedEye[i] != eye) continue;
      const float sx = kLedSin[i];
      const float h = kLedCos[i];
      float c[3];
      if (kLedKind[i] == kInner) {
        const float cover_top = face::ease((h - edge_top) / kLidSoft + 0.5f);
        const float cover_bottom = face::ease((edge_bottom - h) / kLidSoft + 0.5f);
        float lit = (1.0f - cover_top) * (1.0f - cover_bottom);
        if (p[kCross] > 0.0f) {  // the x: the diagonals alone (sin^2 of twice the angle)
          const float diag = 2.0f * sx * h;
          lit *= (1.0f - p[kCross]) + p[kCross] * diag * diag;
        }
        float lobe = 0.0f;
        if (pm > 1e-4f) {
          const float d = (sx * px + h * py) / pm;
          if (d > 0.0f) {
            lobe = 1.0f;
            for (int n = 0; n < lobe_power; ++n) lobe *= d;
            lobe *= pm < 1.0f ? pm : 1.0f;
          }
        }
        float k = lobe * p[kHighlight];
        float extra = fx_white;  // white over the eye: the flash, a twinkle
        if (effect_ == kFxSpin) {  // the head and its tail, chasing clockwise as seen
          const float d = wrap(kLedDeg[i] - spin_head);
          float w = 1.0f - fabsf(d) / kSpinHeadDeg;
          if (w < 0.0f) w = 0.0f;
          if (d < 0.0f && -d < kSpinTailDeg) {
            const float tail = 0.6f * (1.0f + d / kSpinTailDeg);
            if (tail > w) w = tail;
          }
          k = k + (1.0f - k) * w;
          extra += w * (1.0f - kSpinDim);  // the head glows past the dimmed body
        } else if (effect_ == kFxSparkle && twinkle_on_[i]) {
          const float tw = (float)(int32_t)(now_ms - twinkle_t0_[i]) / kSparkleRiseMs;
          if (tw >= 2.0f) {
            twinkle_on_[i] = false;
          } else {
            extra += kSparkleGain * (tw < 1.0f ? tw : 2.0f - tw);
          }
        }
        const float spot = glow * (1.0f + kHighlightGain * k);  // the highlight a brighter spot
        for (int ch = 0; ch < 3; ++ch) {
          const float iris = ccur_[eye][kElIris][ch];
          const float mixed = iris + (ccur_[eye][kElHighlight][ch] - iris) * k;
          c[ch] = mixed * spot * lit + 255.0f * extra * lit;
        }
      } else {
        const float phi = wrap(nose * kLedDeg[i]);  // toward the nose positive
        const float above = face::ease((h - kBrowFloor) / kBrowFloorSoft);
        const float below = face::ease((-h - kBrowFloor) / kBrowFloorSoft);
        const float brow = brow_bright * arc(phi, brow_at, brow_half) * above;
        const float lower = lower_bright * arc(phi, kHalfTurn, lower_half) * below;
        const float cheek = cheek_bright * arc(phi, kHalfTurn, kCheekHalfDeg) * below;
        float sparkle = 0.0f;
        if (heart_ >= 0.0f) {
          const float d = fabsf(wrap(phi - sparkle_at));
          const float w = 1.0f - d / kSparkleHalfDeg;
          if (w > 0.0f) sparkle = w * (1.0f - heart_);
        }
        float tear = 0.0f;  // a drop from the nose side's horizontal down to the bottom
        if (effect_ == kFxTear && tear_k >= 0.0f) {
          const float at = 100.0f + 80.0f * tear_k;
          const float d = phi - at;  // behind the drop (toward where it came from): a tail
          if (d >= 0.0f) {
            tear = 1.0f - d / kTearTailDeg;
          } else {
            tear = 1.0f + d / 12.0f;
          }
          if (tear < 0.0f) tear = 0.0f;
          tear *= below * (1.0f - face::ease((tear_k - 0.7f) / 0.3f));
        }
        for (int ch = 0; ch < 3; ++ch) {
          c[ch] = ccur_[eye][kElBrow][ch] * brow + ccur_[eye][kElLid][ch] * lower +
                  ccur_[eye][kElCheek][ch] * cheek + 255.0f * (sparkle + fx_white) +
                  tear_rgb[ch] * tear;
        }
      }
      for (int ch = 0; ch < 3; ++ch) light_[3 * i + ch] = clampf(c[ch] * kColorScale, 0.0f, 1.0f);
    }
  }
}

// The gamma (the look's values are perceptual), the brightness as a linear gain on the duty,
// the current cap and the soft start; `full`: at full brightness (the self-test).
void EyeModel::power(uint32_t now_ms, bool full) {
  const float gain = full ? 1.0f : (float)brightness_ * kColorScale;
  uint8_t duty[kLedCount * 3];
  float sum = 0.0f;
  for (int i = 0; i < kLedCount * 3; ++i) {
    int q = (int)(light_[i] * 255.0f + 0.5f);
    q = q < 0 ? 0 : (q > 255 ? 255 : q);
    duty[i] = kGammaTable[q];
    sum += (float)duty[i];
  }
  const float base = (float)kLedCount * kQuiescentMa;
  const float demand = kLedMa * sum * kColorScale * gain;  // the LEDs' current before the cap
  float allowed = kCapMa - base;
  if (allowed < 0.0f) allowed = 0.0f;
  float scale = demand > allowed && demand > 0.0f ? allowed / demand : 1.0f;
  const float capped = demand * scale;
  // the soft start: a rise of the demand reaches its target over kSoftStartMs from where the
  // ceiling stood; a fall is followed at once
  if (capped > limit_ma_) {
    if (!ramping_) {
      ramping_ = true;
      ramp_from_ = limit_ma_;
      ramp_t0_ms_ = now_ms;
    }
    ramp_to_ = capped;
    const float k = clampf((float)(int32_t)(now_ms - ramp_t0_ms_) / kSoftStartMs, 0.0f, 1.0f);
    limit_ma_ = ramp_from_ + (ramp_to_ - ramp_from_) * k;
    if (k >= 1.0f) ramping_ = false;
  } else {
    limit_ma_ = capped;
    ramping_ = false;
  }
  if (capped > 0.0f && limit_ma_ < capped) scale *= limit_ma_ / capped;
  scale *= gain;
  float out_sum = 0.0f;
  for (int i = 0; i < kLedCount * 3; ++i) {
    int v = (int)((float)duty[i] * scale);  // truncated: the sum never passes the cap
    v = v < 0 ? 0 : (v > 255 ? 255 : v);
    frame_.rgb[i] = (uint8_t)v;
    out_sum += (float)v;
  }
  const float ma = base + kLedMa * out_sum * kColorScale;
  frame_.milliamps = (uint16_t)(ma + 0.5f);
}

}  // namespace eyes
