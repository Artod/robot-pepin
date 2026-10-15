// The shape-keyframe mouth (the anime style and the styles drawn like it): every expression
// is three drawings (at rest, half open, open), each a set of closed outlines per layer (one
// table per shapes style, <style>_table.h, generated from config/<style>/*.svg and listed in
// shape_tables.h), and the model morphs them vertex by vertex: an expression change, a talk
// cut (the speech level cut into the three drawings), a switch between two shapes styles or
// an idle gesture is a tween from the shape on the screen to the target's, the layers a
// keyframe lacks collapsing onto its outline as they fade. The frame is rasterized by
// scanlines with anti-aliased coverage (non-zero winding: the signed coverage of every edge
// accumulated and its magnitude clamped) straight into the display's strips. Pure C++ (no
// Arduino): the firmware and the native tests share it, and sim/anime.js is its line-by-line
// port.
#pragma once

#include <stdint.h>

#include "face_model.h"
#include "face_render.h"
#include "face_table.h"
#include "shape_tables.h"

namespace face {

// The table of a shapes style (an index of kStyles), or nullptr for a parametric one: whether
// the shape engine draws the frame.
const ShapeTable* shape_table_of(int style);

constexpr int kAnimeFramePolys = 80;  // polygons a frame may hold (offset strokes, effects)
constexpr int kAnimeFrameVertices = 2048;  // their vertices
constexpr int kAnimeLayers = 40;

// A materialized set of every slot's polygons: positions (1/16 px), each polygon's stroke
// width (strokes) or size (anchors) in 1/16 px, its opacity (0..255; 0: absent) and its flags
// (the table's: it follows the tongue). Polygon k of slot s sits at a fixed place.
struct ShapeSet {
  int16_t x[kAnimeSlotVertices];
  int16_t y[kAnimeSlotVertices];
  int16_t w[kAnimeSlotPolys];
  uint8_t a[kAnimeSlotPolys];
  uint8_t flags[kAnimeSlotPolys];
};

// One frame to rasterize: closed polygons (x, y in 1/16 px) grouped into layers drawn in order,
// each layer one colour at one opacity, its polygons filled together (non-zero winding).
struct AnimeFrame {
  int16_t vx[kAnimeFrameVertices];
  int16_t vy[kAnimeFrameVertices];
  uint16_t poly_first[kAnimeFramePolys];
  uint16_t poly_n[kAnimeFramePolys];
  uint8_t layer_first[kAnimeLayers];  // the layer's first polygon
  uint8_t layer_n[kAnimeLayers];
  uint16_t layer_color[kAnimeLayers];
  uint8_t layer_alpha[kAnimeLayers];  // 0..32
  int n_vertices;
  int n_polys;
  int n_layers;
  uint16_t background;
};

class AnimeModel {
 public:
  AnimeModel();

  // Follow the parametric model's state at `now_ms` (after its update): a new transition or
  // style switch retargets to the expression shown (from another shapes style's drawing over
  // the model's transition, from a parametric style at once), a talk cut retargets to its
  // drawing, the idle gesture and the effects are read off it; then the frame is built.
  // Nothing happens while the model's style is parametric.
  void follow(const FaceModel& model, uint32_t now_ms);

  // The frame to draw after follow().
  const AnimeFrame& frame() const { return frame_; }

  // The blend on the screen at the last follow(), before the gesture and the breath (the
  // tests).
  const ShapeSet& shape() const { return cur_; }

  // The talk drawing a speech level cuts to in `style` (0 rest, 1 half open, 2 open).
  static int talkKey(int style, float level);

 private:
  void retarget(float duration_ms, uint32_t now_ms);
  void build(const FaceModel& model, const Shape& ps, uint32_t now_ms);

  ShapeSet from_;
  ShapeSet to_;
  ShapeSet cur_;   // the blend of from_ and to_ at the last follow()
  ShapeSet gest_;  // the gesture's morph target, when it has one
  AnimeFrame frame_;
  const ShapeTable* table_ = nullptr;  // the style's drawings
  uint32_t starts_ = 0;
  int key_ = 0;
  uint32_t t0_ms_ = 0;
  float dur_ms_ = 0.0f;
  int gesture_ = -1;
  bool gesture_toward_ = false;
  bool primed_ = false;
};

// The strip cleared to the background and the frame's layers drawn into it.
void draw_anime(const AnimeFrame& frame, Strip strip);

}  // namespace face
