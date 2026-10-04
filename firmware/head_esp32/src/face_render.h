// The mouth drawn column by column: each screen column holds at most a lip band, a cavity and
// two rows of teeth, each a vertical span with anti-aliased ends. Pure C++ (no Arduino): the
// firmware and the native tests share it, and sim/face.js is its line-by-line port.
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

// One frame's geometry: computed once, then drawn into as many strips as the buffer allows.
struct Layout {
  Column col[kScreenWidth];
  int x_first;  // the columns that may be active: [x_first, x_last]
  int x_last;
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

// The columns of `shape`.
void layout(const Shape& shape, Layout* out);

// The strip cleared to the background and the mouth drawn into it.
void draw(const Layout& layout, Strip strip);

}  // namespace face
