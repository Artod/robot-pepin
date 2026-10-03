#include "display.h"

#include <Arduino.h>
#include <esp_heap_caps.h>

#define LGFX_USE_V1
#include <LovyanGFX.hpp>

#include "face_table.h"
#include "pins.h"

namespace head {

namespace {

// The ideaspark board's panel: ST7789V, 170 columns of a 240-column controller (35 in), IPS
// (inverted colours), on VSPI's native pins.
class Panel : public lgfx::LGFX_Device {
 public:
  Panel() {
    {
      auto cfg = bus_.config();
      cfg.spi_host = VSPI_HOST;
      cfg.spi_mode = 0;
      cfg.freq_write = 80000000;  // 10.9 ms a full frame; 40 MHz if the panel shows noise
      cfg.freq_read = 16000000;
      cfg.spi_3wire = true;
      cfg.use_lock = true;
      cfg.dma_channel = SPI_DMA_CH_AUTO;
      cfg.pin_sclk = pins::kTftSclk;
      cfg.pin_mosi = pins::kTftMosi;
      cfg.pin_miso = -1;
      cfg.pin_dc = pins::kTftDc;
      bus_.config(cfg);
      panel_.setBus(&bus_);
    }
    {
      auto cfg = panel_.config();
      cfg.pin_cs = pins::kTftCs;
      cfg.pin_rst = pins::kTftRst;
      cfg.pin_busy = -1;
      cfg.memory_width = 240;
      cfg.memory_height = 320;
      cfg.panel_width = 170;
      cfg.panel_height = 320;
      cfg.offset_x = 35;
      cfg.offset_y = 0;
      cfg.offset_rotation = 0;
      cfg.readable = false;
      cfg.invert = true;
      cfg.rgb_order = false;
      cfg.dlen_16bit = false;
      cfg.bus_shared = false;
      panel_.config(cfg);
    }
    {
      auto cfg = light_.config();
      cfg.pin_bl = pins::kTftBacklight;
      cfg.invert = false;
      cfg.freq = 12000;
      cfg.pwm_channel = 7;
      light_.config(cfg);
      panel_.setLight(&light_);
    }
    setPanel(&panel_);
  }

 private:
  lgfx::Panel_ST7789 panel_;
  lgfx::Bus_SPI bus_;
  lgfx::Light_PWM light_;
};

Panel g_lcd;
constexpr int kTopRows = face::kScreenHeight / 2;
constexpr int kBottomRows = face::kScreenHeight - kTopRows;
uint16_t* g_strip[2] = {nullptr, nullptr};
bool g_dma_busy = false;

}  // namespace

bool display_begin(uint8_t brightness) {
  g_lcd.init();
  g_lcd.setRotation(face::kScreenRotation);
  g_lcd.setBrightness(brightness);
  g_lcd.fillScreen(face::kColorBackground);
  const size_t bytes = (size_t)face::kScreenWidth * kBottomRows * sizeof(uint16_t);
  for (auto& strip : g_strip) {
    strip = (uint16_t*)heap_caps_malloc(bytes, MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    if (!strip) return false;
  }
  return true;
}

void display_brightness(uint8_t brightness) { g_lcd.setBrightness(brightness); }

void display_face(const face::Layout& layout) {
  // A transfer starts only once the one before it is done (LovyanGFX waits for the bus), so:
  // strip 0 is free here (the last frame's strip 1 started after it had gone out), and strip 1
  // is free once strip 0's push has started. Each strip is drawn while the other is sent.
  face::draw(layout, face::Strip{g_strip[0], 0, kTopRows, true});
  if (!g_dma_busy) {
    g_lcd.startWrite();
    g_dma_busy = true;
  }
  g_lcd.pushImageDMA(0, 0, face::kScreenWidth, kTopRows, (const lgfx::swap565_t*)g_strip[0]);
  face::draw(layout, face::Strip{g_strip[1], kTopRows, kBottomRows, true});
  g_lcd.pushImageDMA(0, kTopRows, face::kScreenWidth, kBottomRows,
                     (const lgfx::swap565_t*)g_strip[1]);
}

void display_info(const InfoScreen& info) {
  if (g_dma_busy) {
    g_lcd.waitDMA();
    g_lcd.endWrite();
    g_dma_busy = false;
  }
  g_lcd.startWrite();
  g_lcd.fillScreen(face::kColorBackground);
  const int w = face::kScreenWidth;
  int y = 6;
  bool title = true;
  for (int i = 0; i < info.n && y < face::kScreenHeight - 14; ++i) {
    const InfoItem& item = info.items[i];
    g_lcd.setTextDatum(lgfx::top_left);
    if (item.kind == kText) {
      g_lcd.setFont(title ? &fonts::lgfxJapanGothic_24 : &fonts::lgfxJapanGothic_20);
      g_lcd.setTextColor(title ? face::kColorAccent : face::kColorText);
      g_lcd.drawString(item.value, 8, y);
      y += title ? 30 : 24;
    } else if (item.kind == kKeyValue) {
      g_lcd.setFont(&fonts::lgfxJapanGothic_20);
      g_lcd.setTextColor(face::kColorAccent);
      g_lcd.drawString(item.key, 8, y);
      g_lcd.setTextColor(face::kColorText);
      g_lcd.setTextDatum(lgfx::top_right);
      g_lcd.drawString(item.value, w - 8, y);
      y += 24;
    } else {  // a bar: the label, the bar, the value at its end
      g_lcd.setFont(&fonts::lgfxJapanGothic_16);
      g_lcd.setTextColor(face::kColorText);
      g_lcd.drawString(item.key, 8, y + 3);
      const int bx = 110, bw = w - bx - 70, bh = 14;
      g_lcd.drawRoundRect(bx, y + 3, bw, bh, 4, face::kColorAccent);
      const int fill = (int)((bw - 4) * (item.frac / 255.0f));
      if (fill > 0) g_lcd.fillRoundRect(bx + 2, y + 5, fill, bh - 4, 3, face::kColorLip);
      g_lcd.setTextDatum(lgfx::top_right);
      g_lcd.drawString(item.value, w - 8, y + 3);
      y += 22;
    }
    title = false;
  }
  g_lcd.endWrite();
}

}  // namespace head
