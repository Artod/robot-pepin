// The 1.9" ST7789 (170x320, landscape): the face as two DMA-pushed strips, the info screen as
// text drawn straight to the panel.
#pragma once

#include <stdint.h>

#include "face_render.h"
#include "protocol.h"

namespace head {

// Panel, backlight and the two strip buffers (2 x 54 KB of DMA-capable RAM); false when the
// buffers cannot be had.
bool display_begin(uint8_t brightness);

void display_brightness(uint8_t brightness);

// One face frame: each strip rendered while the other one is on its way to the panel.
void display_face(const face::Layout& layout);

// The info screen, drawn once (the face resumes with the next display_face).
void display_info(const InfoScreen& info);

}  // namespace head
