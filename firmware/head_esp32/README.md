# Head firmware (ESP32): the mouth and the head IMU

The robot's head is a stereo camera on a pan/tilt neck. Its two lenses are the eyes; under them
sits a 1.9" ST7789 screen (320x170, landscape) that is the mouth, driven by an ESP32 board that
also samples an MPU6050 glued rigidly to the camera body. The ESP32 talks to the robot's board
over USB serial only; WiFi and Bluetooth are never started.

```
board (Orange Pi)                                    head
pepin.head_server ── /dev/pepin-head (CH340, 921600) ── ESP32 ── SPI 80 MHz ── ST7789 (mouth)
  TCP 3340: face commands in, IMU lines out                    └─ I2C 400 kHz ── MPU6050 (head IMU)
```

## Hardware

**Board:** ideaspark ESP32 development board, ESP32-WROOM (classic), 16 MB flash, integrated
1.9" ST7789 170x320 TFT, CH340 USB-UART, USB-C.

**Display pins (on the board, fixed):** MOSI 23, SCLK 18, CS 15, DC 2, RST 4, backlight 32 (PWM).
These are VSPI's native pins, so the panel runs at 80 MHz without the GPIO matrix.

**MPU6050 wiring (GY-521, 3-5 cm of wire):**

| GY-521 | ESP32 board | Note |
| --- | --- | --- |
| VCC | 5V (VIN) | the GY-521 has its own 3.3 V regulator and pull-ups to it; 3V3 also works |
| GND | GND | |
| SDA | GPIO 21 | the Arduino core's default I2C pins, free on this board |
| SCL | GPIO 22 | |
| INT | GPIO 27 | optional but recommended: exact sample stamps; without it the firmware polls |
| AD0 | open | pulled low on the module: address 0x68 (0x69 is tried too) |
| XDA, XCL | open | |

Glue the GY-521 flat and rigid to the camera body (not to the screen or the bracket), and note
which way its printed X and Y arrows point relative to the camera: the head IMU's mount is a
number the visual-inertial side needs.

## Build and flash

PlatformIO needs no install beyond `uv`:

```bash
cd firmware/head_esp32
uvx --from platformio pio run                                    # build
uvx --from platformio pio run -t upload --upload-port /dev/cu.usbserial-XXXX   # flash
```

The board enters its bootloader by itself (DTR/RTS auto-reset). Flash from the laptop's USB, then
plug the board into the robot. The serial line carries binary frames, so a serial monitor shows
noise; use `pepin.head_server` (below) instead.

**Bench check on the laptop** before the robot: run the head server against the board's port and
talk to it with the CLI:

```bash
uv run python -m pepin.head_server --config config/head.json --device /dev/cu.usbserial-XXXX
uv run python -m pepin.head_link --host 127.0.0.1 status        # link, fps, IMU rate, clock
uv run python -m pepin.head_link --host 127.0.0.1 express happy 3
uv run python -m pepin.head_link --host 127.0.0.1 show 'Temps|left: 41/70 C' 6
uv run python -m pepin.head_link --host 127.0.0.1 imu 5         # samples in m/s^2 and rad/s
```

## The mouth

The mouth is procedural: 16 numbers draw it (`config/face.json`, `params`): width, opening,
corner lift, how the opening splits above and below the mouth's line, the opening's profile (a
lens, an ellipse or a rounded rectangle), teeth (none, the upper row, clenched), a wave along it
(a worried wobble or a crawling zigzag), asymmetry, position, line thickness, and two idle
motions (breathing, drift). An expression is a full set of these numbers, and the firmware tweens
every number on its own (smoothstep, 280 ms by default), so any expression turns into any other.
The speech level ('M' frames, sent by the board's audio server while it plays) opens and narrows
whatever expression shows, with a fast attack and a slower release; without fresh levels for
250 ms the mouth closes by itself.

**At rest the face is alive** (`config/face.json`, `idle`): while a resting expression shows
(neutral, smile), every 8-20 s one gesture passes over it as a smooth bump: a brief wider smile, a
small tilt either way (asymmetry and a sideways shift) or a blink-like thinning of the line. None
starts within 3 s of an expression change or of a speech level, and one under way fades out in
0.25 s when an event or speech comes. The choices are random (seeded from `esp_random()` at
boot); the simulator draws the same ones from the same seed.

Expressions, by wire id (append only):

| id | name | looks like |
| --- | --- | --- |
| 0 | neutral | a short, slightly lifted line |
| 1 | smile | a closed smile |
| 2 | happy | an open D-shaped smile, upper teeth |
| 3 | grin | a wide grin, upper teeth |
| 4 | clenched | a grin with clenched teeth |
| 5 | sad | corners down |
| 6 | worried | a wobbly, slightly open frown |
| 7 | surprised | an O |
| 8 | thinking | a short skewed line off to the side, drifting |
| 9 | struggling | a crawling zigzag |
| 10 | flat | a dead-straight line |
| 11 | sleepy | a small round mouth breathing slowly |
| 12 | focused | a short firm line |
| 13 | listening | a small open, attentive mouth |

Rendering is column by column: for each screen column the lip curves (sampled every half
pixel) are swept with a disc of the line's radius, which gives round ends and an even line on
steep parts, and each column becomes at most four vertical spans (lip, cavity, two rows of teeth)
with anti-aliased ends. The frame is drawn into two 320x85 strips (2 x 54 KB of DMA-capable RAM)
and each strip is pushed to the panel by DMA while the other is drawn. The face is capped at
50 fps; the frame rate is in the status line.

**Host silence:** with no frame from the board for 3 s (the head server pings at 2 Hz) the mouth
falls asleep (`sleepy`) and wakes into the board's expression when it speaks again. An info screen
('T') covers the face for its seconds.

`src/face_table.h` and `sim/face_table.js` are generated from `config/face.json`:

```bash
uv run python -m pepin.face --write     # after editing config/face.json
```

## The simulator

`sim/index.html` runs the same model and renderer (a line-by-line port in `sim/face.js`, checked
pixel for pixel against the C++ by `tests/unit/test_face_parity.py`) with the head server's
arbiter, robot-event buttons, lip sync from a synthetic babble, the microphone or an audio file,
the info screen, and sliders for every parameter (with a copy-as-JSON button for new expressions):

```bash
open firmware/head_esp32/sim/index.html
```

## The serial protocol

Both ways: `0xA5 | type | length (u16 LE) | payload | CRC-8` (polynomial 0x07, initial 0, over
type, length and payload). `src/protocol.h` and `src/pepin/head_link.py` document every payload.

| type | direction | payload |
| --- | --- | --- |
| `I` | ESP32 -> host | IMU samples, n x 17 bytes: micros u32, ax ay az gx gy gz i16 (raw), config id u8 |
| `S` | ESP32 -> host | status once a second: fps, IMU rate, I2C errors, dropped samples, frame errors, ... |
| `P` | ESP32 -> host | pong: the ping's id, the ESP32's micros at receipt |
| `E` | host -> ESP32 | expression: id u8, intensity u8, transition ms u16 |
| `M` | host -> ESP32 | mouth openness u8 (0..255) |
| `T` | host -> ESP32 | info screen: duration ms u16, items (text, key/value, bar) |
| `Q` | host -> ESP32 | ping: id u32 |
| `C` | host -> ESP32 | config: id u8, IMU rate u16, DLPF u8, accel FS u8, gyro FS u8, brightness u8 |

The IMU samples at the output rate the host's 'C' asks for (the head server asks 200 Hz,
`config/head.json`; the firmware boots at 1000 Hz until then), DLPF_CFG 3, +-4 g, +-500 deg/s,
CLKSEL 1 (the gyro's PLL), on its own core-0 task while the display runs on core 1. Each sample
is stamped with `micros()` in the data-ready interrupt, so the INT wire is required: without
edges the firmware streams nothing and reports `imu_state` 3 (a polled fallback beats against
the chip's own oscillator and is compiled out, `HEAD_IMU_POLL_FALLBACK`). Samples go out every
20 ms (4 at 200 Hz: 73 bytes, 4 % of the link; 20 at 1 kHz: 345 bytes, 19 %); a frame that does
not fit the serial buffer is dropped and counted, never waited for. The status line carries the
chip's rate measured in its own stamps (198-202 at a nominal 200 is the chip's oscillator, not a
fault) and the gaps longer than 1.5 periods.

## Built on

- [LovyanGFX](https://github.com/lovyan03/LovyanGFX) (panel driver, DMA, fonts; the info screen's
  IPA Gothic font covers Cyrillic)
- the Arduino core for ESP32 2.0.17 (PlatformIO espressif32 6.x)
