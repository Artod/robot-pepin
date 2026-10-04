#include "face_model.h"

#include <math.h>

namespace face {

namespace {

constexpr float kTwoPi = 6.28318530718f;

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

}  // namespace

float ease(float k) {
  k = clampf(k, 0.0f, 1.0f);
  return k * k * (3.0f - 2.0f * k);
}

FaceModel::FaceModel() {
  for (int i = 0; i < kParamCount; ++i) {
    from_[i] = to_[i] = cur_[i] = kExpressions[kNeutral][i];
  }
}

void FaceModel::setExpression(int id, float intensity, float transition_ms, uint32_t now_ms) {
  if (id < 0 || id >= kExpressionCount) return;
  intensity = clampf(intensity, 0.0f, 1.0f);
  float target[kParamCount];
  for (int i = 0; i < kParamCount; ++i) {
    const float n = kExpressions[kNeutral][i];
    target[i] = n + (kExpressions[id][i] - n) * intensity;
  }
  expression_ = id;
  setTarget(target, transition_ms, now_ms);
}

void FaceModel::setTarget(const float target[kParamCount], float transition_ms, uint32_t now_ms) {
  update(now_ms);  // cur_ is where the mouth is at this moment: the new tween starts there
  for (int i = 0; i < kParamCount; ++i) {
    from_[i] = cur_[i];
    to_[i] = clampf(target[i], kParamMin[i], kParamMax[i]);
  }
  t0_ms_ = now_ms;
  dur_ms_ = transition_ms > 0.0f ? transition_ms : 0.0f;
  if (dur_ms_ == 0.0f) {
    for (int i = 0; i < kParamCount; ++i) cur_[i] = to_[i];
  }
}

void FaceModel::setMouth(float level, uint32_t now_ms) {
  level_target_ = clampf(level, 0.0f, 1.0f);
  level_at_ms_ = now_ms;
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

  if ((float)(int32_t)(now_ms - level_at_ms_) > kMouthStaleS * 1000.0f) level_target_ = 0.0f;
  const float tau = level_target_ > level_ ? kMouthAttackS : kMouthReleaseS;
  if (dt > 0.0f) level_ += (level_target_ - level_) * (1.0f - expf(-dt / tau));

  wave_phase_ += cur_[kWaveSpeed] * dt;
  wave_phase_ -= floorf(wave_phase_);
  breathe_phase_ += kBreatheHz * dt;
  breathe_phase_ -= floorf(breathe_phase_);
  drift_phase_ += kDriftHz * dt;
  drift_phase_ -= floorf(drift_phase_);
}

Shape FaceModel::shape() const {
  Shape s;
  for (int i = 0; i < kParamCount; ++i) s.p[i] = cur_[i];
  const float b = sinf(kTwoPi * breathe_phase_);
  const float breathe = cur_[kBreathe];
  s.p[kWidth] = cur_[kWidth] * (1.0f + 0.035f * breathe * b) * (1.0f - kSpeechNarrow * level_);
  const float open = cur_[kOpen] + breathe * 0.045f * (0.5f + 0.5f * b) + kSpeechOpen * level_;
  s.p[kOpen] = clampf(open, 0.0f, 1.0f);
  s.p[kShiftX] = cur_[kShiftX] + cur_[kDrift] * (1.0f / 3.0f) * sinf(kTwoPi * drift_phase_);
  s.wave_phase = wave_phase_;
  return s;
}

}  // namespace face
