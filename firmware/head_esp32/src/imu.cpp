#include "imu.h"

#include <Arduino.h>
#include <Wire.h>
#include <esp_timer.h>
#include <string.h>

#include "pins.h"

namespace head {

namespace {

// MPU-6050 register map (rev 4.2).
constexpr uint8_t kSmplrtDiv = 0x19;
constexpr uint8_t kConfigReg = 0x1A;
constexpr uint8_t kGyroConfig = 0x1B;
constexpr uint8_t kAccelConfig = 0x1C;
constexpr uint8_t kIntPinCfg = 0x37;
constexpr uint8_t kIntEnable = 0x38;
constexpr uint8_t kAccelXoutH = 0x3B;
constexpr uint8_t kPwrMgmt1 = 0x6B;
constexpr uint8_t kWhoAmI = 0x75;
constexpr uint32_t kI2cHz = 400000;  // the chip's own ceiling
constexpr int kBatchMax = 8;
constexpr int kErrorsBeforeReset = 50;  // consecutive failed reads: the chip is set up again

TaskHandle_t g_task = nullptr;
ImuSink g_sink = nullptr;
esp_timer_handle_t g_poll_timer = nullptr;
portMUX_TYPE g_lock = portMUX_INITIALIZER_UNLOCKED;
ImuConfig g_pending = kImuDefaults;
volatile bool g_has_pending = false;
ImuStats g_stats = {};
uint8_t g_addr = 0x68;

void IRAM_ATTR on_data_ready() {
  BaseType_t woken = pdFALSE;
  xTaskNotifyFromISR(g_task, (uint32_t)micros(), eSetValueWithOverwrite, &woken);
  if (woken) portYIELD_FROM_ISR();
}

void on_poll(void*) { xTaskNotify(g_task, (uint32_t)micros(), eSetValueWithOverwrite); }

bool write_reg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(g_addr);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool read_regs(uint8_t reg, uint8_t* buf, size_t n) {
  Wire.beginTransmission(g_addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(g_addr, (uint8_t)n, (uint8_t) true) != n) return false;
  for (size_t i = 0; i < n; ++i) buf[i] = (uint8_t)Wire.read();
  return true;
}

void stop_sources() {
  detachInterrupt(digitalPinToInterrupt(pins::kImuInt));
  if (g_poll_timer) esp_timer_stop(g_poll_timer);
}

// Find the chip, write the configuration, pick the data-ready interrupt if its wire answers
// within 50 ms, else poll at twice the rate. False while no chip answers.
bool setup_chip(const ImuConfig& cfg) {
  stop_sources();
  uint8_t who = 0;
  bool found = false;
  for (uint8_t addr : {(uint8_t)0x68, (uint8_t)0x69}) {
    g_addr = addr;
    if (read_regs(kWhoAmI, &who, 1)) {
      found = true;
      break;
    }
  }
  g_stats.who_am_i = found ? who : 0;
  if (!found) {
    g_stats.state = kImuAbsent;
    return false;
  }
  const uint8_t dlpf = (cfg.dlpf >= 1 && cfg.dlpf <= 6) ? cfg.dlpf : 3;
  const uint16_t rate = cfg.rate_hz < 4 ? 4 : (cfg.rate_hz > 1000 ? 1000 : cfg.rate_hz);
  bool ok = write_reg(kPwrMgmt1, 0x80);  // reset
  delay(100);
  ok = ok && write_reg(kPwrMgmt1, 0x01);  // awake, clocked by the X gyro's PLL
  ok = ok && write_reg(kSmplrtDiv, (uint8_t)(1000 / rate - 1));
  ok = ok && write_reg(kConfigReg, dlpf);
  ok = ok && write_reg(kGyroConfig, (uint8_t)((cfg.gyro_fs & 3) << 3));
  ok = ok && write_reg(kAccelConfig, (uint8_t)((cfg.accel_fs & 3) << 3));
  ok = ok && write_reg(kIntPinCfg, 0x10);  // active high, push-pull, 50 us pulse, any read clears
  ok = ok && write_reg(kIntEnable, 0x01);  // DATA_RDY
  if (!ok) {
    ++g_stats.i2c_errors;
    g_stats.state = kImuAbsent;
    return false;
  }
  g_stats.config_id = cfg.id;
  ulTaskNotifyValueClear(g_task, 0xFFFFFFFF);
  xTaskNotifyStateClear(g_task);
  attachInterrupt(digitalPinToInterrupt(pins::kImuInt), on_data_ready, RISING);
  uint32_t edge = 0;
  if (xTaskNotifyWait(0, 0xFFFFFFFF, &edge, pdMS_TO_TICKS(50)) == pdTRUE) {
    g_stats.state = kImuInterrupt;
    return true;
  }
  detachInterrupt(digitalPinToInterrupt(pins::kImuInt));
  if (!g_poll_timer) {
    esp_timer_create_args_t args = {};
    args.callback = on_poll;
    args.name = "imu_poll";
    esp_timer_create(&args, &g_poll_timer);
  }
  esp_timer_start_periodic(g_poll_timer, 500000u / rate);
  g_stats.state = kImuPolled;
  return true;
}

void imu_task(void*) {
  Wire.begin(pins::kImuSda, pins::kImuScl, kI2cHz);
  Wire.setTimeOut(5);
  pinMode(pins::kImuInt, INPUT_PULLDOWN);  // an unconnected INT stays low: no edges, polling
  ImuConfig cfg = kImuDefaults;
  bool running = false;
  ImuSample batch[kBatchMax];
  int count = 0;
  int batch_n = 5;
  uint32_t period_us = 1000;
  uint32_t last_stamp = 0;
  uint8_t last_raw[14] = {0};
  int errors = 0;
  for (;;) {
    if (g_has_pending) {
      portENTER_CRITICAL(&g_lock);
      cfg = g_pending;
      g_has_pending = false;
      portEXIT_CRITICAL(&g_lock);
      running = false;
    }
    if (!running) {
      count = 0;
      last_stamp = 0;
      errors = 0;
      running = setup_chip(cfg);
      if (!running) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        continue;
      }
      const uint16_t rate = cfg.rate_hz < 4 ? 4 : (cfg.rate_hz > 1000 ? 1000 : cfg.rate_hz);
      period_us = 1000000u / rate;
      batch_n = rate / 200;  // about 200 frames a second: <= 5 ms of batching
      batch_n = batch_n < 1 ? 1 : (batch_n > kBatchMax ? kBatchMax : batch_n);
    }
    uint32_t stamp = 0;
    if (xTaskNotifyWait(0, 0xFFFFFFFF, &stamp, pdMS_TO_TICKS(100)) != pdTRUE) {
      running = false;  // no edge and no poll for 100 ms: set the chip up again
      continue;
    }
    const bool polled = g_stats.state == kImuPolled;
    if (polled) stamp -= period_us / 4;  // polled at twice the rate: half a poll period old
    uint8_t raw[14];
    if (!read_regs(kAccelXoutH, raw, sizeof raw)) {
      ++g_stats.i2c_errors;
      if (++errors > kErrorsBeforeReset) running = false;
      continue;
    }
    errors = 0;
    if (polled) {
      // accel (0..5) and gyro (8..13) equal to the last read: the chip has no new sample yet
      if (memcmp(raw, last_raw, 6) == 0 && memcmp(raw + 8, last_raw + 8, 6) == 0) {
        ++g_stats.duplicates;
        continue;
      }
    } else if (last_stamp && stamp - last_stamp > period_us + period_us / 2) {
      g_stats.missed += (stamp - last_stamp + period_us / 2) / period_us - 1;
    }
    memcpy(last_raw, raw, sizeof raw);
    last_stamp = stamp;
    ImuSample& s = batch[count++];
    s.t_us = stamp;
    for (int i = 0; i < 3; ++i) {
      s.a[i] = (int16_t)((raw[2 * i] << 8) | raw[2 * i + 1]);
      s.g[i] = (int16_t)((raw[8 + 2 * i] << 8) | raw[8 + 2 * i + 1]);
    }
    s.cfg = cfg.id;
    if (count >= batch_n) {
      if (g_sink) g_sink(batch, (size_t)count);
      g_stats.samples += (uint32_t)count;
      count = 0;
    }
  }
}

}  // namespace

void imu_start(ImuSink sink) {
  g_sink = sink;
  xTaskCreatePinnedToCore(imu_task, "imu", 4096, nullptr, configMAX_PRIORITIES - 2, &g_task, 0);
}

void imu_configure(const ImuConfig& config) {
  portENTER_CRITICAL(&g_lock);
  g_pending = config;
  g_has_pending = true;
  portEXIT_CRITICAL(&g_lock);
}

ImuStats imu_stats() {
  ImuStats s;
  portENTER_CRITICAL(&g_lock);
  s = g_stats;
  portEXIT_CRITICAL(&g_lock);
  return s;
}

}  // namespace head
