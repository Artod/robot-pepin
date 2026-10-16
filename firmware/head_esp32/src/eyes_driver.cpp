#include "eyes_driver.h"

#include <NeoPixelBus.h>

#include "eye_table.h"

namespace head {

namespace {

// RMT channel 0 with the WS2812x timing (an 800 kHz bit stream, a 300 us reset): the driver
// translates the bytes into RMT items in its interrupt as the channel drains, no DMA and no
// item buffer; Show() returns as soon as the transfer has started.
using Bus = NeoPixelBus<NeoGrbFeature, NeoEsp32Rmt0Ws2812xMethod>;
Bus* g_bus = nullptr;

}  // namespace

void eyes_begin(int pin) {
  g_bus = new Bus((uint16_t)eyes::kLedCount, (uint8_t)pin);
  g_bus->Begin();
  g_bus->ClearTo(RgbColor(0, 0, 0));
  g_bus->Show();
}

void eyes_show(const uint8_t* rgb) {
  if (!g_bus || !g_bus->CanShow()) return;
  for (int i = 0; i < eyes::kLedCount; ++i) {
    g_bus->SetPixelColor((uint16_t)i, RgbColor(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]));
  }
  g_bus->Show();
}

}  // namespace head
