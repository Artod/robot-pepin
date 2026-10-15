// The mouth drawn column by column: each screen column holds at most a lip band, a cavity and
// two rows of teeth, each a vertical span with anti-aliased ends; the decorations (the blush,
// the tongue, the fangs, the cheek folds, the moustache, the pucker, the corner hooks, the
// heart-shaped lips, the drool, the kiss's heart) are more such spans, drawn under or over the
// mouth in their elements' colours. The style (kStyles) sets the line's thickness, whether the
// teeth are one band, the shade band and the outlines round the decorations, and the
// background. Pure C++ (no Arduino): the firmware and the native tests share it, and
// sim/face.js is its line-by-line port.
#pragma once

#include <stdint.h>

#include "face_model.h"
#include "face_table.h"

namespace face {

// The spans of one screen column, in screen pixels (y down); a column is drawn only if active.
struct Column {
  bool active;
  bool tooth_gap;   // a gap between two teeth: the cavity shows through
  float outer_top;  // the lip band: [outer_top, outer_bot)
  float outer_bot;
  float inner_top;  // the cavity: [inner_top, inner_bot), empty when inner_bot <= inner_top
  float inner_bot;
  float teeth_up;   // the upper row's height from inner_top
  float teeth_lo;   // the lower row's height up from inner_bot
};

// One span of a decoration: [top, bot) of column x in an element's colour (or the background's:
// the moustache's bristles) at an opacity of alpha/32.
struct Deco {
  float top;
  float bot;
  int16_t x;
  uint8_t element;  // an Element, or kDecoBackground
  uint8_t alpha;    // 0..32
};
constexpr uint8_t kDecoBackground = 0xFF;
constexpr int kDecoMax = 768;  // every decoration at once takes about 500

// One frame's geometry: computed once, then drawn into as many strips as the buffer allows.
struct Layout {
  Column col[kScreenWidth];
  int x_first;  // the columns that may be active: [x_first, x_last]
  int x_last;
  uint16_t color[kElementCount];  // the shape's colours, RGB565
  uint16_t background;            // the style's, RGB565
  uint8_t style;                  // a Style: how the mouth's spans are drawn
  int n_under;  // deco[0, n_under) is drawn under the mouth (the blush),
  int n_deco;   // deco[n_under, n_deco) over it (the tongue, the fangs, the folds, ...)
  Deco deco[kDecoMax];
};

// A horizontal strip of the screen: rows [y0, y0 + rows) of a kScreenWidth-wide RGB565 buffer.
// `swapped` stores each pixel byte-swapped, the order the panel's SPI takes (LovyanGFX's
// swap565_t), so the strip goes to the display by DMA without a conversion.
struct Strip {
  uint16_t* px;
  int y0;
  int rows;
  bool swapped;
};

// The columns and decorations of `shape`.
void layout(const Shape& shape, Layout* out);

// The strip cleared to the background and the mouth drawn into it.
void draw(const Layout& layout, Strip strip);

}  // namespace face
