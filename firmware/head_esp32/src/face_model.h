// The mouth's state over time: the expression being tweened to (its parameters and its elements'
// colours in the style in force), the speech level on top of it, the idle motion (the breath,
// the drift, and the idle gestures: config/face.json `idle`, on a resting expression now and
// then a brief wider smile, a tilt or a blink), an expression's variants drawn at random and
// its effects (the kiss's heart once, the sleeper's "z" again and again). Pure C++ (no
// Arduino): the firmware and the native tests share it, and sim/face.js is its line-by-line
// port.
#pragma once

#include <stdint.h>

#include "face_table.h"

namespace face {

// What the renderer draws in one frame: the tweened parameters with the speech level and the
// idle motion already applied, the wave's phase, the heart's progress, the elements' colours
// and the style they are drawn in.
struct Shape {
  float p[kParamCount];
  float wave_phase;                // periods, [0, 1)
  float heart;                     // the heart effect's flight, [0, 1], or -1: none
  float z;                         // the sleeper's z's flight, [0, 1], or -1: none
  uint16_t color[kElementCount];   // RGB565, by Element
  uint8_t style;                   // a Style
};

class FaceModel {
 public:
  FaceModel();

  // Tween to expression `id` (an index of kExpressions) at `intensity` (0 neutral .. 1 the
  // expression itself) over `transition_ms`, starting from wherever the mouth is now. An
  // expression with variants shows itself or one of them, drawn at random each time; one with
  // the heart effect starts it (a heart still flying is not restarted); one with the z
  // effect sends a "z" up now and every kZEveryS while it shows (in the style in force).
  void setExpression(int id, float intensity, float transition_ms, uint32_t now_ms);

  // Draw in `style` (a Style) from now on: the expression shown tweens into its look in that
  // style over the default transition; the variant drawn and a heart in flight stay; the z
  // starts or stops with the look's effects in that style.
  void setStyle(int style, uint32_t now_ms);

  // Tween to an explicit parameter set (the simulator's sliders; the firmware uses ids); the
  // colours keep tweening to the last expression's.
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

  // The expression asked for (what the status reports) and the one drawn for it (a variant).
  int expression() const { return expression_; }
  int shown() const { return shown_; }
  int style() const { return style_; }
  float level() const { return level_; }
  // The idle gesture under way (an index of kIdleGestures), or -1, and its share in this
  // frame with its side (-1..1: a mirror gesture's sign times its bump).
  int idleGesture() const { return idle_gesture_; }
  float idleAmp() const { return idle_sign_ * idle_amp_; }
  // The transitions started so far (a counter: the shape engine sees a new one), the one
  // under way's duration, and the idle clocks (periods, [0, 1)).
  uint32_t starts() const { return starts_; }
  uint32_t transitionStartMs() const { return t0_ms_; }
  float transitionMs() const { return dur_ms_; }
  float breathePhase() const { return breathe_phase_; }
  float driftPhase() const { return drift_phase_; }
  float wagPhase() const { return wag_phase_; }
  float intensity() const { return intensity_; }

 private:
  uint32_t nextRandom();  // xorshift32
  float uniform();        // [0, 1), 24 bits
  void scheduleIdle(uint32_t now_ms);
  bool idleAllowed(uint32_t now_ms) const;
  void updateIdle(uint32_t now_ms);
  void lookOf(int shown, float intensity, float target[kParamCount],
              float colors[kElementCount][3]) const;
  // The z with the look shown's effects: started (now, then every kZEveryS) or stopped.
  void startZ(uint8_t effects, uint32_t now_ms);
  void start(const float target[kParamCount], const float colors[kElementCount][3],
             float transition_ms, uint32_t now_ms);

  float from_[kParamCount];
  float to_[kParamCount];
  float cur_[kParamCount];
  float cfrom_[kElementCount][3];  // the elements' colours, RGB 0..255
  float cto_[kElementCount][3];
  float ccur_[kElementCount][3];
  uint32_t t0_ms_ = 0;
  float dur_ms_ = 0.0f;
  int style_ = kStyleDefault;
  int expression_ = kNeutral;
  int shown_ = kNeutral;
  float intensity_ = 1.0f;
  bool heart_on_ = false;
  uint32_t heart_t0_ms_ = 0;
  bool z_on_ = false;           // the sleeper's z: a flight every kZEveryS while it shows
  uint32_t z_t0_ms_ = 0;
  uint32_t starts_ = 0;
  float level_target_ = 0.0f;
  float level_ = 0.0f;
  uint32_t level_at_ms_ = 0;
  uint32_t last_ms_ = 0;
  bool started_ = false;
  float wave_phase_ = 0.0f;    // periods, wrapped to [0, 1)
  float breathe_phase_ = 0.0f;  // the idle breath, periods
  float drift_phase_ = 0.0f;    // the idle drift, periods
  float wag_phase_ = 0.0f;      // the tongue's wag, periods
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
