// The ideaspark ESP32 1.9" board's wiring (README.md has the table for the soldering iron).
#pragma once

namespace pins {

// The ST7789 on the board itself (VSPI's own pins, so SPI runs without the GPIO matrix).
constexpr int kTftMosi = 23;
constexpr int kTftSclk = 18;
constexpr int kTftCs = 15;
constexpr int kTftDc = 2;
constexpr int kTftRst = 4;
constexpr int kTftBacklight = 32;

// The MPU6050 (GY-521) on 3-5 cm of wire: the Arduino core's default I2C pins, free on this
// board, and its data-ready line on a free input. Without the INT wire the firmware polls.
constexpr int kImuSda = 21;
constexpr int kImuScl = 22;
constexpr int kImuInt = 27;

}  // namespace pins
