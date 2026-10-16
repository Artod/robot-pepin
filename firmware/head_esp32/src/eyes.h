// The eyes: two WS2812B rings round each camera lens (an inner 8 and an outer 16, eye_table.h
// has the chain), lit from the same expression table as the mouth. The model follows the
// face model (the expression shown, its intensity and style, its transitions and effects),
// tweens every eye parameter and colour on its own, blinks, breathes and makes micro-saccades
// on its own clock, takes the host's 'Y' override over any of its fields, and turns the lot
// into one perceptual RGB per LED; the power stage then applies the brightness, the gamma,
// the current cap and the soft start and yields the bytes the driver sends. Pure C++ (no
// Arduino): the firmware and the native tests share it, and sim/eyes.js is its line-by-line
// port.
//
// Sides are the robot's own: its left eye sits on the viewer's right. Angles round a ring run
// from the top, clockwise as seen facing the robot; the two eyes mirror about the nose.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "eye_table.h"
#include "face_model.h"

namespace eyes {

// A 'Y' frame decoded: which params it sets (a bit per Param in `mask`) and their values,
// its tint (the iris and the brow), the blink it asks for, which eyes, for how long, and the
// inner ring's effect it calls (the 19-byte form).
struct Override {
  uint8_t mode;              // 0 release, 1 standing, 2 timed by `ms`
  uint16_t ms;
  uint32_t mask;             // a bit per Param set (the rest stay the expression's)
  float value[kParamCount];  // in the param's own units
  bool tinted;               // an iris colour given (not 0,0,0)
  float tint[3];             // RGB 0..255: the iris and the brow
  uint8_t blink;             // 0 none, 1 both eyes, 2 wink left, 3 wink right
  uint8_t side;              // 0 both eyes, 1 the left only, 2 the right only
  bool has_effect;           // the effect byte given (not 0xFF, or the 16-byte form)
  uint8_t effect;            // an Effect: 0 none (back to the look's own)
  uint16_t effect_ms;        // its life; 0: a one-shot effect's own length, a standing one
                             // until the next effect byte or a release
};

constexpr size_t kOverrideBytes = 16;        // the 'Y' without the effect
constexpr size_t kOverrideBytesEffect = 19;  // with effect u8 and effect_ms u16
constexpr uint8_t kKeepI8 = 0x80;  // an i8 field left to the expression
constexpr uint8_t kKeepU8 = 0xFF;  // a u8 field left to the expression (0..254 set it)

// A 'Y' payload of 16 or 19 bytes (protocol.h has the layout and the sentinel rule).
Override decode_override(const uint8_t* p, size_t n);

enum Mode : uint8_t { kModeExpression = 0, kModeStanding = 1, kModeTimed = 2, kModeSelfTest = 3 };

// What the driver sends: RGB per LED in chain order after the power stage, and the frame's
// current estimate.
struct Frame {
  uint8_t rgb[kLedCount * 3];
  uint16_t milliamps;
};

class EyeModel {
 public:
  EyeModel();

  // Seed the blinks' and saccades' random choices (the firmware: esp_random() at boot; the
  // tests and the simulator keep the default).
  void seed(uint32_t s);

  // Follow the face model's state at `now_ms` (after its update()): a new transition retargets
  // to the look of the expression shown in the style in force, the effects are read off it;
  // a timed override that lapsed is released; then the frame is computed.
  void follow(const face::FaceModel& model, uint32_t now_ms);

  // The host's 'Y': release, or stand (timed or not) over the expression's look on the eyes it
  // names, and blink or wink now.
  void setOverride(const Override& o, uint32_t now_ms);

  // The 'C' config's eyes brightness (0: the rings dark).
  void setBrightness(uint8_t b) { brightness_ = b; }

  // The 'C' config's eye-set byte: 0 follows the face style (eye_table.h kEyeSetOfStyle),
  // else the set id + 1; the look is retargeted at the next follow().
  void setSet(uint8_t choice) { set_choice_ = choice; }
  uint8_t set() const { return set_; }

  // The boot sweep from now: one LED chases round the chain over kSelfTestMs at full
  // brightness, then the look.
  void selfTest(uint32_t now_ms);

  const Frame& frame() const { return frame_; }
  uint8_t mode() const;
  uint8_t brightness() const { return brightness_; }
  // Per LED, RGB 0..1 (perceptual) before the power stage, at the last follow().
  const float* light() const { return light_; }
  // The tweened value of a param on one eye (the tests).
  float value(int eye, int param) const { return cur_[eye][param]; }
  // The effect running (an Effect; kFxNone between them).
  uint8_t effect() const { return effect_; }

 private:
  uint32_t nextRandom();
  float uniform();
  void lookOf(int set, int style, int shown, float intensity);
  void retarget(float ms, uint32_t t0_ms);
  void startBlink(int eye, bool wink, float rate, int count, uint32_t now_ms);
  float blinkOf(int eye, uint32_t now_ms);
  void updateEffect(uint32_t now_ms, float dt);
  void update(uint32_t now_ms);
  void render(uint32_t now_ms);
  void renderSelfTest(uint32_t now_ms);
  void power(uint32_t now_ms, bool full);

  // the expression's look (both eyes alike; the asymmetries are applied when rendering)
  float look_[kParamCount];
  float clook_[kElementCount][3];
  // the override per eye
  uint32_t ov_mask_[kEyeCount];
  float ov_value_[kEyeCount][kParamCount];
  bool ov_tinted_[kEyeCount];
  float ov_tint_[kEyeCount][3];
  bool ov_timed_ = false;
  uint32_t ov_until_ms_ = 0;
  // the tween per eye
  float from_[kEyeCount][kParamCount];
  float to_[kEyeCount][kParamCount];
  float cur_[kEyeCount][kParamCount];
  float cfrom_[kEyeCount][kElementCount][3];
  float cto_[kEyeCount][kElementCount][3];
  float ccur_[kEyeCount][kElementCount][3];
  uint32_t t0_ms_ = 0;
  float dur_ms_ = 0.0f;
  bool primed_ = false;
  uint32_t starts_ = 0;
  int style_ = -1;
  uint8_t set_choice_ = 0;  // 0 follow the style, else a set id + 1
  uint8_t set_ = kSetClassic;
  // the clocks
  bool started_ = false;
  uint32_t last_ms_ = 0;
  float breath_phase_ = 0.0f;
  uint32_t rng_ = 0x9E3779B9u;
  bool blink_on_[kEyeCount];
  uint32_t blink_t0_[kEyeCount];
  float blink_close_[kEyeCount];
  float blink_hold_[kEyeCount];
  float blink_open_[kEyeCount];
  int blink_count_[kEyeCount];  // closures in a row (the flutter: three)
  bool blink_scheduled_ = false;
  uint32_t blink_next_ms_ = 0;
  bool sacc_scheduled_ = false;
  uint32_t sacc_next_ms_ = 0;
  uint32_t sacc_t0_ms_ = 0;
  float sacc_from_[2];
  float sacc_to_[2];
  // the effects read off the face model
  float heart_ = -1.0f;
  float z_ = -1.0f;
  // the inner ring's effect: the look's default, the host's over it, the one running
  uint8_t look_effect_ = kFxNone;
  bool host_effect_on_ = false;
  uint8_t host_effect_ = kFxNone;
  uint32_t host_effect_until_ms_ = 0;  // 0: until released
  uint8_t effect_source_ = 0xFF;       // the id the running effect was started from
  uint8_t effect_ = kFxNone;
  uint32_t effect_t0_ms_ = 0;
  uint32_t twinkle_t0_[kLedCount];     // the sparkle's twinkle per LED
  bool twinkle_on_[kLedCount];
  float flicker_level_ = 1.0f;
  uint32_t flicker_next_ms_ = 0;
  uint8_t inner_first_[kEyeCount];     // each eye's inner ring in the chain
  uint8_t inner_n_[kEyeCount];
  // the self-test
  bool test_on_ = false;
  uint32_t test_t0_ms_ = 0;
  // the power stage
  uint8_t brightness_ = kBrightnessDefault;
  float limit_ma_ = 0.0f;  // the soft start's ceiling on the LEDs' current
  bool ramping_ = false;
  float ramp_from_ = 0.0f;
  float ramp_to_ = 0.0f;
  uint32_t ramp_t0_ms_ = 0;
  float light_[kLedCount * 3];
  Frame frame_;
};

}  // namespace eyes
