#include "face_model.h"

#include <math.h>

namespace face {

namespace {

constexpr float kTwoPi = 6.28318530718f;
constexpr float kPi = 3.14159265359f;
constexpr float kQuietLevel = 0.01f;  // a speech level under this is no speech
// A standing wave (wave_speed 0) settles to its home phase with this time constant: the cat's
// "w" and the worried wobble read as drawn only at phase 0, and a crawl leaves any phase.
constexpr float kWaveSettleS = 0.3f;

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// 0xRRGGBB as three floats, 0..255.
void rgb_of(uint32_t c, float out[3]) {
  out[0] = (float)((c >> 16) & 0xFFu);
  out[1] = (float)((c >> 8) & 0xFFu);
  out[2] = (float)(c & 0xFFu);
}

// Three floats, 0..255, as RGB565 (truncated, as pepin.face.rgb565 does: a settled colour is
// exactly the table's).
uint16_t rgb565(const float c[3]) {
  const int r = (int)clampf(c[0], 0.0f, 255.0f);
  const int g = (int)clampf(c[1], 0.0f, 255.0f);
  const int b = (int)clampf(c[2], 0.0f, 255.0f);
  return (uint16_t)(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
}

}  // namespace

float ease(float k) {
  k = clampf(k, 0.0f, 1.0f);
  return k * k * (3.0f - 2.0f * k);
}

FaceModel::FaceModel() {
  for (int i = 0; i < kParamCount; ++i) {
    from_[i] = to_[i] = cur_[i] = kExpressions[style_][kNeutral][i];
  }
  for (int e = 0; e < kElementCount; ++e) {
    rgb_of(kExpressionColors[style_][kNeutral][e], ccur_[e]);
    for (int k = 0; k < 3; ++k) cfrom_[e][k] = cto_[e][k] = ccur_[e][k];
  }
}

void FaceModel::lookOf(int shown, float intensity, float target[kParamCount],
                       float colors[kElementCount][3]) const {
  for (int i = 0; i < kParamCount; ++i) {
    const float n = kExpressions[style_][kNeutral][i];
    target[i] = n + (kExpressions[style_][shown][i] - n) * intensity;
  }
  for (int e = 0; e < kElementCount; ++e) {
    float n[3], x[3];
    rgb_of(kExpressionColors[style_][kNeutral][e], n);
    rgb_of(kExpressionColors[style_][shown][e], x);
    for (int k = 0; k < 3; ++k) colors[e][k] = n[k] + (x[k] - n[k]) * intensity;
  }
}

void FaceModel::setExpression(int id, float intensity, float transition_ms, uint32_t now_ms) {
  if (id < 0 || id >= kExpressionCount) return;
  intensity = clampf(intensity, 0.0f, 1.0f);
  int shown = id;
  if (kVariantCount[id] > 0) {  // itself or one of its variants, alike
    const uint32_t r = nextRandom() % (uint32_t)(kVariantCount[id] + 1);
    if (r > 0) shown = kVariants[id][r - 1];
  }
  float target[kParamCount];
  float colors[kElementCount][3];
  lookOf(shown, intensity, target, colors);
  expression_ = id;
  shown_ = shown;
  intensity_ = intensity;
  start(target, colors, transition_ms, now_ms);
  const uint8_t effects = intensity > 0.0f ? kExpressionEffects[style_][shown] : 0;
  if ((effects & kEffectHeart) && !heart_on_) {
    heart_on_ = true;  // one heart at a time: a resend while it flies does not restart it
    heart_t0_ms_ = now_ms;
  }
  startZ(effects, now_ms);
}

void FaceModel::startZ(uint8_t effects, uint32_t now_ms) {
  if ((effects & kEffectZ) && !z_on_) {
    z_on_ = true;
    z_t0_ms_ = now_ms;
  } else if (!(effects & kEffectZ)) {
    z_on_ = false;  // the sleeper woke: no z lingers over the new face
  }
}

void FaceModel::setStyle(int style, uint32_t now_ms) {
  if (style < 0 || style >= kStyleCount || style == style_) return;
  style_ = style;
  float target[kParamCount];
  float colors[kElementCount][3];
  lookOf(shown_, intensity_, target, colors);
  start(target, colors, kTransitionMs, now_ms);
  startZ(intensity_ > 0.0f ? kExpressionEffects[style_][shown_] : 0, now_ms);
}

void FaceModel::setTarget(const float target[kParamCount], float transition_ms, uint32_t now_ms) {
  float colors[kElementCount][3];
  for (int e = 0; e < kElementCount; ++e) {
    for (int k = 0; k < 3; ++k) colors[e][k] = cto_[e][k];
  }
  start(target, colors, transition_ms, now_ms);
}

void FaceModel::start(const float target[kParamCount], const float colors[kElementCount][3],
                      float transition_ms, uint32_t now_ms) {
  update(now_ms);  // cur_ is where the mouth is at this moment: the new tween starts there
  for (int i = 0; i < kParamCount; ++i) {
    from_[i] = cur_[i];
    to_[i] = clampf(target[i], kParamMin[i], kParamMax[i]);
  }
  for (int e = 0; e < kElementCount; ++e) {
    for (int k = 0; k < 3; ++k) {
      cfrom_[e][k] = ccur_[e][k];
      cto_[e][k] = colors[e][k];
    }
  }
  t0_ms_ = now_ms;
  dur_ms_ = transition_ms > 0.0f ? transition_ms : 0.0f;
  ++starts_;
  if (dur_ms_ == 0.0f) {
    for (int i = 0; i < kParamCount; ++i) cur_[i] = to_[i];
    for (int e = 0; e < kElementCount; ++e) {
      for (int k = 0; k < 3; ++k) ccur_[e][k] = cto_[e][k];
    }
  }
}

void FaceModel::setMouth(float level, uint32_t now_ms) {
  level_target_ = clampf(level, 0.0f, 1.0f);
  level_at_ms_ = now_ms;
}

void FaceModel::seed(uint32_t s) { rng_ = s != 0u ? s : 0x2545F491u; }

uint32_t FaceModel::nextRandom() {
  uint32_t x = rng_;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  rng_ = x;
  return x;
}

float FaceModel::uniform() { return (float)(nextRandom() >> 8) * (1.0f / 16777216.0f); }

void FaceModel::scheduleIdle(uint32_t now_ms) {
  idle_scheduled_ = true;
  const float span_s = kIdleEveryMinS + (kIdleEveryMaxS - kIdleEveryMinS) * uniform();
  idle_next_ms_ = now_ms + (uint32_t)(span_s * 1000.0f);
}

bool FaceModel::idleAllowed(uint32_t now_ms) const {
  if (expression_ < 0 || expression_ >= 32 || !((kIdleExpressionMask >> expression_) & 1u)) {
    return false;
  }
  const float settle_ms = kIdleSettleS * 1000.0f;
  if ((float)(int32_t)(now_ms - t0_ms_) < settle_ms + dur_ms_) return false;  // a change
  if ((float)(int32_t)(now_ms - level_at_ms_) < settle_ms) return false;      // speech
  return level_ < kQuietLevel;
}

void FaceModel::updateIdle(uint32_t now_ms) {
  idle_amp_ = 0.0f;
  if (!kIdleOn) return;
  if (!idle_scheduled_) scheduleIdle(now_ms);
  const bool allowed = idleAllowed(now_ms);
  if (idle_gesture_ >= 0) {
    if (!allowed && !idle_fading_) {
      idle_fading_ = true;
      idle_fade_ms_ = now_ms;
    }
    const float u = (float)(int32_t)(now_ms - idle_t0_ms_) /
                    (kIdleGestureDurationS[idle_gesture_] * 1000.0f);
    float fade = 1.0f;
    if (idle_fading_) {
      fade = 1.0f - ease((float)(int32_t)(now_ms - idle_fade_ms_) / (kIdleFadeS * 1000.0f));
    }
    if (u >= 1.0f || fade <= 0.0f) {
      idle_gesture_ = -1;
      scheduleIdle(now_ms);
      return;
    }
    const float s = sinf(kPi * clampf(u, 0.0f, 1.0f));
    idle_amp_ = s * s * fade;
    return;
  }
  if ((int32_t)(now_ms - idle_next_ms_) < 0) return;
  if (!allowed) {  // due while something else shows: wait a whole span again, no burst after
    scheduleIdle(now_ms);
    return;
  }
  // the gestures of this style, by weight
  const uint8_t bit = (uint8_t)(1u << style_);
  float total = 0.0f;
  for (int g = 0; g < kIdleGestureCount; ++g) {
    if (kIdleGestureStyles[g] & bit) total += kIdleGestureWeight[g];
  }
  if (total <= 0.0f) {
    scheduleIdle(now_ms);
    return;
  }
  float r = uniform() * total;
  int g = 0;
  for (int k = 0; k < kIdleGestureCount; ++k) {
    if (!(kIdleGestureStyles[k] & bit)) continue;
    g = k;
    if (r < kIdleGestureWeight[k]) break;
    r -= kIdleGestureWeight[k];
  }
  idle_gesture_ = g;
  idle_t0_ms_ = now_ms;
  idle_fading_ = false;
  idle_sign_ = (kIdleGestureMirror[g] && (nextRandom() & 1u)) ? -1.0f : 1.0f;
}

void FaceModel::update(uint32_t now_ms) {
  if (!started_) {
    started_ = true;
    last_ms_ = now_ms;
  }
  float dt = (float)(int32_t)(now_ms - last_ms_) / 1000.0f;
  dt = clampf(dt, 0.0f, 0.25f);  // a stalled frame must not make the idle motion jump
  last_ms_ = now_ms;

  const float elapsed = (float)(int32_t)(now_ms - t0_ms_);
  const float e = dur_ms_ > 0.0f ? ease(elapsed / dur_ms_) : 1.0f;
  for (int i = 0; i < kParamCount; ++i) cur_[i] = from_[i] + (to_[i] - from_[i]) * e;
  for (int c = 0; c < kElementCount; ++c) {
    for (int k = 0; k < 3; ++k) ccur_[c][k] = cfrom_[c][k] + (cto_[c][k] - cfrom_[c][k]) * e;
  }
  if (heart_on_ && (float)(int32_t)(now_ms - heart_t0_ms_) >= kHeartS * 1000.0f) {
    heart_on_ = false;
  }
  // the z flies again every kZEveryS while the look shown has it; it stops with the look
  if (z_on_) {
    const bool wanted = intensity_ > 0.0f && (kExpressionEffects[style_][shown_] & kEffectZ);
    const float since = (float)(int32_t)(now_ms - z_t0_ms_);
    if (!wanted && since >= kZS * 1000.0f) {
      z_on_ = false;
    } else if (since >= kZEveryS * 1000.0f) {
      if (wanted) {
        z_t0_ms_ += (uint32_t)(kZEveryS * 1000.0f);
      } else {
        z_on_ = false;
      }
    }
  }

  if ((float)(int32_t)(now_ms - level_at_ms_) > kMouthStaleS * 1000.0f) level_target_ = 0.0f;
  const float tau = level_target_ > level_ ? kMouthAttackS : kMouthReleaseS;
  if (dt > 0.0f) level_ += (level_target_ - level_) * (1.0f - expf(-dt / tau));

  if (cur_[kWaveSpeed] != 0.0f) {
    wave_phase_ += cur_[kWaveSpeed] * dt;
  } else if (cur_[kWave] <= 0.0f) {
    wave_phase_ = 0.0f;  // no wave shows: home at once
  } else if (dt > 0.0f) {  // a standing wave settles home
    const float home = wave_phase_ < 0.5f ? 0.0f : 1.0f;
    wave_phase_ += (home - wave_phase_) * (1.0f - expf(-dt / kWaveSettleS));
    if (fabsf(home - wave_phase_) < 1e-3f) wave_phase_ = home;
  }
  wave_phase_ -= floorf(wave_phase_);
  breathe_phase_ += kBreatheHz * dt;
  breathe_phase_ -= floorf(breathe_phase_);
  drift_phase_ += kDriftHz * dt;
  drift_phase_ -= floorf(drift_phase_);
  wag_phase_ += kWagHz * dt;
  wag_phase_ -= floorf(wag_phase_);

  updateIdle(now_ms);
}

Shape FaceModel::shape() const {
  Shape s;
  for (int i = 0; i < kParamCount; ++i) s.p[i] = cur_[i];
  const float b = sinf(kTwoPi * breathe_phase_);
  const float breathe = cur_[kBreathe];
  // the speech level as the style takes it: continuous, or the anime lip flap's few shapes
  // (closed, half open, open), cut on the level
  float level = level_;
  const float shapes = kStyles[style_].flap_shapes;
  if (shapes >= 2.0f) level = floorf(level_ * (shapes - 1.0f) + 0.5f) / (shapes - 1.0f);
  s.p[kWidth] = cur_[kWidth] * (1.0f + 0.035f * breathe * b) * (1.0f - kSpeechNarrow * level);
  const float open = cur_[kOpen] + breathe * 0.045f * (0.5f + 0.5f * b) + kSpeechOpen * level;
  s.p[kOpen] = clampf(open, 0.0f, 1.0f);
  s.p[kShiftX] = cur_[kShiftX] + cur_[kDrift] * (1.0f / 3.0f) * sinf(kTwoPi * drift_phase_);
  if (cur_[kWag] > 0.0f) {
    const float lean = cur_[kTongueLean] + cur_[kWag] * 0.5f * sinf(kTwoPi * wag_phase_);
    s.p[kTongueLean] = clampf(lean, kParamMin[kTongueLean], kParamMax[kTongueLean]);
  }
  if (idle_gesture_ >= 0 && idle_amp_ > 0.0f) {
    const float k = idle_sign_ * idle_amp_;
    for (int i = 0; i < kParamCount; ++i) {
      const float d = kIdleGestures[idle_gesture_][i];
      if (d != 0.0f) s.p[i] = clampf(s.p[i] + d * k, kParamMin[i], kParamMax[i]);
    }
  }
  s.wave_phase = wave_phase_;
  s.heart = -1.0f;
  if (heart_on_) {
    s.heart = clampf((float)(int32_t)(last_ms_ - heart_t0_ms_) / (kHeartS * 1000.0f), 0.0f, 1.0f);
  }
  s.z = -1.0f;
  if (z_on_) {
    const float k = (float)(int32_t)(last_ms_ - z_t0_ms_) / (kZS * 1000.0f);
    if (k < 1.0f) s.z = clampf(k, 0.0f, 1.0f);
  }
  for (int e = 0; e < kElementCount; ++e) s.color[e] = rgb565(ccur_[e]);
  s.style = (uint8_t)style_;
  return s;
}

}  // namespace face
