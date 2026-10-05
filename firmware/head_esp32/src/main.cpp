// Pepin's head: the mouth on the 1.9" screen, the head IMU, and the serial link to the board's
// head server (pepin.head_server). Core 1 draws the face; core 0 samples the IMU; the UART's
// event task parses the host's frames. WiFi is never started and Bluetooth's RAM is released.
#include <Arduino.h>
#include <esp_bt.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>

#include "display.h"
#include "face_model.h"
#include "face_render.h"
#include "face_table.h"
#include "imu.h"
#include "protocol.h"

namespace {

constexpr uint32_t kBaud = 921600;
constexpr uint32_t kFramePeriodMs = 20;  // the face's ceiling, 50 fps
constexpr uint32_t kStatusPeriodMs = 1000;
constexpr uint8_t kBootBrightness = 200;

enum Mode : uint8_t { kModeFace = 0, kModeInfo = 1, kModeAsleep = 2 };

struct FaceCommand {
  uint8_t type;  // 'E', 'M', or 'B' (brightness, from a 'C')
  uint8_t a;     // expression id / mouth level / brightness
  uint8_t b;     // intensity
  uint16_t ms;   // transition
};

SemaphoreHandle_t g_tx;        // one frame at a time on the wire
QueueHandle_t g_commands;      // the host's face commands, for the face loop
SemaphoreHandle_t g_info_lock;
head::InfoScreen g_info;       // the newest 'T', under g_info_lock
volatile bool g_info_new = false;
volatile uint32_t g_last_rx_ms = 0;
volatile uint32_t g_dropped = 0;
volatile uint32_t g_sent_samples = 0;
head::Parser g_parser;

face::FaceModel g_model;
face::Layout g_layout;

// One frame onto the wire. `droppable`: when the TX buffer cannot take it whole, it is counted
// and dropped rather than waited for (the IMU must not stall on a slow reader). The frame is
// built in the caller's `frame` (the UART event task's stack is small: a pong brings 13 bytes).
bool send_in(uint8_t* frame, size_t room, uint8_t type, const uint8_t* payload, size_t n,
             bool droppable) {
  if (n + head::kOverhead > room) return false;
  const size_t size = head::encode(type, payload, n, frame);
  xSemaphoreTake(g_tx, portMAX_DELAY);
  bool sent = true;
  if (droppable && Serial.availableForWrite() < (int)size) {
    sent = false;
  } else {
    Serial.write(frame, size);
  }
  xSemaphoreGive(g_tx);
  return sent;
}

bool send(uint8_t type, const uint8_t* payload, size_t n, bool droppable) {
  uint8_t frame[head::kOverhead + 20 * head::kImuSampleBytes];
  return send_in(frame, sizeof frame, type, payload, n, droppable);
}

void on_imu(const head::ImuSample* s, size_t n) {
  uint8_t payload[20 * head::kImuSampleBytes];
  uint8_t* p = payload;
  for (size_t i = 0; i < n; ++i) {
    p = head::put_u32(p, s[i].t_us);
    for (int k = 0; k < 3; ++k) p = head::put_u16(p, (uint16_t)s[i].a[k]);
    for (int k = 0; k < 3; ++k) p = head::put_u16(p, (uint16_t)s[i].g[k]);
    *p++ = s[i].cfg;
  }
  if (send(head::kImu, payload, (size_t)(p - payload), true)) {
    g_sent_samples += n;
  } else {
    g_dropped += n;
  }
}

// One complete host frame (on the UART event task: light work only).
void dispatch(uint8_t type, const uint8_t* p, size_t n) {
  const uint32_t now_us = micros();
  g_last_rx_ms = millis();
  FaceCommand cmd = {};
  switch (type) {
    case head::kPing:
      if (n == 4) {
        uint8_t pong[8];
        uint8_t frame[sizeof pong + head::kOverhead];
        head::put_u32(head::put_u32(pong, head::get_u32(p)), now_us);
        send_in(frame, sizeof frame, head::kPong, pong, sizeof pong, false);
      }
      return;
    case head::kExpression:
      if (n != 4) return;
      cmd = {head::kExpression, p[0], p[1], head::get_u16(p + 2)};
      break;
    case head::kMouth:
      if (n != 1) return;
      cmd = {head::kMouth, p[0], 0, 0};
      break;
    case head::kInfo:
      if (xSemaphoreTake(g_info_lock, pdMS_TO_TICKS(5)) == pdTRUE) {
        if (head::parse_info(p, n, &g_info)) g_info_new = true;
        xSemaphoreGive(g_info_lock);
      }
      return;
    case head::kConfig:
      if (n != head::kConfigBytes) return;
      head::imu_configure({p[0], head::get_u16(p + 1), p[3], p[4], p[5]});
      cmd = {'B', p[6], 0, 0};
      break;
    default:
      return;
  }
  xQueueSend(g_commands, &cmd, 0);
}

void on_receive() {
  while (Serial.available() > 0) {
    if (g_parser.feed((uint8_t)Serial.read())) {
      dispatch(g_parser.type(), g_parser.payload(), g_parser.size());
    }
  }
}

}  // namespace

void setup() {
  esp_bt_controller_mem_release(ESP_BT_MODE_BTDM);  // Bluetooth never runs: its RAM back
  g_tx = xSemaphoreCreateMutex();
  g_info_lock = xSemaphoreCreateMutex();
  g_commands = xQueueCreate(32, sizeof(FaceCommand));
  Serial.setRxBufferSize(4096);
  Serial.setTxBufferSize(8192);  // ~90 ms of the IMU stream at 921600 baud
  Serial.begin(kBaud);
  Serial.setRxTimeout(2);  // a frame is handed over 2 symbols (22 us) after its last byte
  Serial.onReceive(on_receive, false);
  head::display_begin(kBootBrightness);
  g_model.seed(esp_random());  // the idle gestures differ from boot to boot
  head::imu_start(on_imu);
  g_last_rx_ms = millis();
}

void loop() {
  static uint32_t next_frame = millis();
  static uint32_t next_status = millis() + kStatusPeriodMs;
  static uint32_t frames = 0;
  static uint32_t samples_at_status = 0;
  static uint32_t stamp_at_status = 0;
  static uint16_t fps_x10 = 0;
  static Mode mode = kModeFace;
  static uint32_t info_until = 0;
  static int host_expression = face::kNeutral;
  static uint8_t host_intensity = 255;
  const uint32_t now = millis();

  FaceCommand cmd;
  while (xQueueReceive(g_commands, &cmd, 0) == pdTRUE) {
    if (cmd.type == head::kExpression) {
      host_expression = cmd.a;
      host_intensity = cmd.b;
      if (mode != kModeAsleep) g_model.setExpression(cmd.a, cmd.b / 255.0f, cmd.ms, now);
    } else if (cmd.type == head::kMouth) {
      g_model.setMouth(cmd.a / 255.0f, now);
    } else if (cmd.type == 'B') {
      head::display_brightness(cmd.a);
    }
  }

  // The host silent: fall asleep; the host back: its expression again.
  const bool silent = now - g_last_rx_ms > (uint32_t)(face::kHostSilentS * 1000.0f);
  if (silent && mode != kModeAsleep) {
    mode = kModeAsleep;
    g_model.setExpression(face::kSleepy, 1.0f, 900.0f, now);
  } else if (!silent && mode == kModeAsleep) {
    mode = kModeFace;
    g_model.setExpression(host_expression, host_intensity / 255.0f, face::kTransitionMs, now);
  }

  if (g_info_new && xSemaphoreTake(g_info_lock, pdMS_TO_TICKS(2)) == pdTRUE) {
    head::InfoScreen info = g_info;
    g_info_new = false;
    xSemaphoreGive(g_info_lock);
    if (info.duration_ms > 0 && info.n > 0) {
      head::display_info(info);
      mode = mode == kModeAsleep ? kModeAsleep : kModeInfo;
      info_until = now + info.duration_ms;
    } else {
      info_until = now;  // an empty 'T': back to the face
    }
  }
  const bool showing_info = info_until && (int32_t)(info_until - now) > 0;
  if (!showing_info) {
    if (mode == kModeInfo) mode = kModeFace;
    info_until = 0;
    g_model.update(now);
    face::layout(g_model.shape(), &g_layout);
    head::display_face(g_layout);
    ++frames;
  }

  if ((int32_t)(now - next_status) >= 0) {
    next_status += kStatusPeriodMs;
    fps_x10 = (uint16_t)(frames * 10);
    frames = 0;
    const head::ImuStats imu = head::imu_stats();
    head::Status s = {};
    s.micros = micros();
    s.fps_x10 = fps_x10;
    // The chip's rate in its stamps' own time (its oscillator is not the ESP32's: 198-202 at a
    // nominal 200 is normal), over the samples since the last status line.
    const uint32_t span_us = imu.last_stamp - stamp_at_status;
    const uint32_t n = imu.samples - samples_at_status;
    s.imu_rate_x10 = (span_us > 0 && samples_at_status > 0)
                         ? (uint16_t)((uint64_t)n * 10000000ull / span_us)
                         : 0;
    samples_at_status = imu.samples;
    stamp_at_status = imu.last_stamp;
    s.i2c_errors = imu.i2c_errors;
    s.dropped = g_dropped;
    s.gaps = imu.gaps;
    s.rx_errors = g_parser.crc_errors + g_parser.length_errors;
    s.rx_frames = g_parser.frames;
    s.expression = (uint8_t)g_model.expression();
    s.mode = showing_info ? kModeInfo : mode;
    s.imu_state = imu.state;
    s.who_am_i = imu.who_am_i;
    s.config_id = imu.config_id;
    s.version = head::kFirmwareVersion;
    s.free_heap_kb = (uint16_t)(ESP.getFreeHeap() / 1024);
    uint8_t payload[head::kStatusBytes];
    send(head::kStatus, payload, head::pack_status(s, payload), true);
  }

  next_frame += kFramePeriodMs;
  const int32_t wait = (int32_t)(next_frame - millis());
  if (wait > 0) {
    delay((uint32_t)wait);
  } else {
    next_frame = millis();  // behind: do not try to catch up
  }
}
