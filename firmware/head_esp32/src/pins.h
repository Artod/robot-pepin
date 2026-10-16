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

// The eyes' WS2812B chain (both rings of both eyes in series, eye_table.h has the order): one
// data line, 3.3 V from the ESP32 into the first ring's DIN (that ring's 5 V through a 1N4007,
// so its 4.3 V rail reads 3.3 V as high). The board is potted: the pin is whichever free GPIO
// the iron can reach. Free on this board: 26 (the default), 25, 33, 13, 14, 19, 5, 16, 17;
// taken: 27 (the IMU's INT), 32 (the backlight), 21/22 (I2C), 23/18/15/2/4 (the panel).
constexpr int kEyesData = 26;

}  // namespace pins
