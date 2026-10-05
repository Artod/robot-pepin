// The mouth's state over time: the expression being tweened to, the speech level on top of it,
// and the idle motion: the breath, the drift, and the idle gestures (config/face.json `idle`: on
// a resting expression, now and then a brief wider smile, a tilt or a blink). Pure C++ (no
// Arduino): the firmware and the native tests share it, and sim/face.js is its line-by-line port.
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

  // Seed the idle gestures' random choices (the firmware: esp_random() at boot; the tests keep
  // the default seed, which the simulator shares).
  void seed(uint32_t s);

  // Advance to `now_ms`: the tween, the level's attack/release, the wave and the idle clocks.
  void update(uint32_t now_ms);

  // The frame to draw at the last update().
  Shape shape() const;

  int expression() const { return expression_; }
  float level() const { return level_; }
  // The idle gesture under way (an index of kIdleGestures), or -1.
  int idleGesture() const { return idle_gesture_; }

 private:
  uint32_t nextRandom();  // xorshift32
  float uniform();        // [0, 1), 24 bits
  void scheduleIdle(uint32_t now_ms);
  bool idleAllowed(uint32_t now_ms) const;
  void updateIdle(uint32_t now_ms);

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
  uint32_t rng_ = 0x2545F491u;
  bool idle_scheduled_ = false;
  uint32_t idle_next_ms_ = 0;   // the next gesture may start from here
  int idle_gesture_ = -1;
  uint32_t idle_t0_ms_ = 0;
  float idle_sign_ = 1.0f;      // a mirror gesture's side
  bool idle_fading_ = false;    // a change or speech came: the gesture leaves
  uint32_t idle_fade_ms_ = 0;
  float idle_amp_ = 0.0f;       // the gesture's share in this frame, 0..1
};

// Smoothstep: the tween's easing, 0..1 -> 0..1.
float ease(float k);

}  // namespace face
