// The eyes' WS2812B chain on the ESP32's RMT peripheral (NeoPixelBus): one data pin, the
// transfer started from the face loop on core 1 and finished by the RMT's own interrupt there,
// so the IMU task on core 0, the panel's SPI DMA and the UART never wait for it.
#pragma once

#include <stdint.h>

namespace head {

// Claim the pin and an RMT channel; the rings dark.
void eyes_begin(int pin);

// One frame of RGB bytes (eyes::kLedCount x 3, chain order) onto the wire: ~1.5 ms in the
// background. A frame that comes while the last one is still going out is skipped (never at
// 50 fps: the wire takes 1.5 ms of the 20).
void eyes_show(const uint8_t* rgb);

}  // namespace head
