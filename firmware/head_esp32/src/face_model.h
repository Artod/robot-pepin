// The mouth's state over time: the expression being tweened to, the speech level on top of it,
// and the idle motion. Pure C++ (no Arduino): the firmware and the native tests share it, and
// sim/face.js is its line-by-line port.
#pragma once

#include <stdint.h>

#include "face_table.h"

namespace face {

// What the renderer draws in one frame: the tweened parameters with the speech level and the
// idle motion already applied, and the wave's phase.
struct Shape {
  float p[kParamCount];
  float wave_phase;  // periods, [0, 1)
};

class FaceModel {
 public:
  FaceModel();

  // Tween to expression `id` (an index of kExpressions) at `intensity` (0 neutral .. 1 the
  // expression itself) over `transition_ms`, starting from wherever the mouth is now.
  void setExpression(int id, float intensity, float transition_ms, uint32_t now_ms);

  // Tween to an explicit parameter set (the simulator's sliders; the firmware uses ids).
  void setTarget(const float target[kParamCount], float transition_ms, uint32_t now_ms);

  // A speech level, 0..1 (an 'M' frame's 0..255 / 255); it stands for mouth_stale_s.
  void setMouth(float level, uint32_t now_ms);

  // Advance to `now_ms`: the tween, the level's attack/release, the wave and the idle clocks.
  void update(uint32_t now_ms);

  // The frame to draw at the last update().
  Shape shape() const;

  int expression() const { return expression_; }
  float level() const { return level_; }

 private:
  float from_[kParamCount];
  float to_[kParamCount];
  float cur_[kParamCount];
  uint32_t t0_ms_ = 0;
  float dur_ms_ = 0.0f;
  int expression_ = kNeutral;
  float level_target_ = 0.0f;
  float level_ = 0.0f;
  uint32_t level_at_ms_ = 0;
  uint32_t last_ms_ = 0;
  bool started_ = false;
  float wave_phase_ = 0.0f;    // periods, wrapped to [0, 1)
  float breathe_phase_ = 0.0f;  // the idle breath, periods
  float drift_phase_ = 0.0f;    // the idle drift, periods
};

// Smoothstep: the tween's easing, 0..1 -> 0..1.
float ease(float k);

}  // namespace face
