// aleks2: aurora::device gyro/accel for the Switch over libnx's six-axis
// sensor. Ported from Aleks' pad_switch.cpp (aurora 08167a62). NX's SDL3-switch
// backend has no sensor support, and dolphin/pad/pad.cpp already falls back to
// aurora::device when the SDL gamepad reports no sensor, so this is the only
// missing piece for gyro aim.
#ifdef __SWITCH__
#include <aurora/aurora.h>
#include <switch.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "device.hpp"
#include "motion_queue.hpp"

namespace aurora::device {
namespace {

PadState s_pad{};
bool s_padInitialized = false;
HidSixAxisSensorHandle s_handle{};
bool s_started = false;
motion::Cursor s_primaryCursor;  // motion_samples(Primary): where it stopped in s_handle's queue

void ensure_pad() {
  if (s_padInitialized) {
    return;
  }
  // aleks2: four, not one -- see platforms/switch/source/switch_input_hid.cpp.
  // At one, pairing a Pro Controller stopped the Joy-Con from turning on.
  padConfigureInput(4, HidNpadStyleSet_NpadStandard);
  padInitializeDefault(&s_pad);
  s_padInitialized = true;
}

bool choose_style(u32 styleSet, HidNpadStyleTag& outStyle) {
  if ((styleSet & HidNpadStyleTag_NpadFullKey) != 0) {
    outStyle = HidNpadStyleTag_NpadFullKey;
    return true;
  }
  if ((styleSet & HidNpadStyleTag_NpadJoyDual) != 0) {
    outStyle = HidNpadStyleTag_NpadJoyDual;
    return true;
  }
  if ((styleSet & HidNpadStyleTag_NpadJoyRight) != 0) {
    outStyle = HidNpadStyleTag_NpadJoyRight;
    return true;
  }
  if ((styleSet & HidNpadStyleTag_NpadJoyLeft) != 0) {
    outStyle = HidNpadStyleTag_NpadJoyLeft;
    return true;
  }
  return false;
}

bool resolve_handle(HidSixAxisSensorHandle& outHandle) {
  ensure_pad();
  padUpdate(&s_pad);

  HidNpadIdType id = HidNpadIdType_No1;
  HidNpadStyleTag style = static_cast<HidNpadStyleTag>(0);
  if (padIsHandheld(&s_pad)) {
    id = HidNpadIdType_Handheld;
    style = HidNpadStyleTag_NpadHandheld;
  } else if (padIsNpadActive(&s_pad, HidNpadIdType_No1)) {
    if (!choose_style(hidGetNpadStyleSet(HidNpadIdType_No1), style)) {
      return false;
    }
  } else {
    return false;
  }

  HidSixAxisSensorHandle handles[2]{};
  if (style == HidNpadStyleTag_NpadJoyDual) {
    if (R_SUCCEEDED(hidGetSixAxisSensorHandles(handles, 2, id, style))) {
      // Detached dual Joy-Con has one sensor per controller. Use the right
      // Joy-Con for aiming.
      outHandle = handles[1];
      return true;
    }
  }
  if (R_FAILED(hidGetSixAxisSensorHandles(handles, 1, id, style))) {
    return false;
  }
  outHandle = handles[0];
  return true;
}

bool ensure_started() {
  HidSixAxisSensorHandle handle{};
  if (!resolve_handle(handle)) {
    return false;
  }
  if (s_started && s_handle.type_value != handle.type_value) {
    hidStopSixAxisSensor(s_handle);
    s_started = false;
  }
  if (s_handle.type_value != handle.type_value) {
    s_primaryCursor = motion::Cursor{};
  }
  s_handle = handle;
  if (!s_started) {
    if (R_FAILED(hidStartSixAxisSensor(s_handle))) {
      return false;
    }
    s_started = true;
  }
  return true;
}

bool read(bool gyro, float* data, const int n_values) {
  if (data == nullptr || n_values <= 0) {
    return false;
  }
  std::fill(data, data + n_values, 0.0f);
  if (!ensure_started()) {
    return false;
  }
  HidSixAxisSensorState state{};
  if (hidGetSixAxisSensorStates(s_handle, &state, 1) == 0 ||
      (state.attributes & HidSixAxisSensorAttribute_IsConnected) == 0) {
    return false;
  }
  // UNITS. libnx and SDL do not agree, and everything downstream is tuned for
  // SDL because that is what the desktop build feeds it:
  //   angular_velocity  libnx = REVOLUTIONS/s   SDL_SENSOR_GYRO  = radians/s
  //   acceleration      libnx = g               SDL_SENSOR_ACCEL = m/s^2
  // Passing libnx's numbers through raw made every gyro reading 2*pi times too
  // small: dusk::gyro's absolute deadband (0.04) then zeroed gentle yaw turns
  // outright while pitch flicks merely felt sluggish. Converting at this
  // boundary keeps the deadband, smoothing and sensitivity defaults untouched.
  constexpr float kRevPerSecToRadPerSec = 6.28318530717958647692f;  // 2*pi
  constexpr float kGToMetresPerSecSq = 9.80665f;
  const float scale = gyro ? kRevPerSecToRadPerSec : kGToMetresPerSecSq;
  const HidVector& src = gyro ? state.angular_velocity : state.acceleration;
  if (n_values > 0) {
    data[0] = src.x * scale;
  }
  if (n_values > 1) {
    data[1] = src.y * scale;
  }
  if (n_values > 2) {
    data[2] = src.z * scale;
  }
  return true;
}

// ---------------------------------------------------------------------------
// aleks2 (H38): the two sensors of a detached Joy-Con pair, separately.
//
// Primary above is the right Joy-Con (gyro aim). Wii-style motion controls also need the left one -- the Nunchuk --
// so sensor_layout() resolves both handles of a JoyDual pad once per tick and starts them on request, and the side
// reads below use that cached resolution: no HID service call besides the state read itself. Left and Right are
// answered only for a detached pair; one Pro Controller, the handheld console or a lone Joy-Con has just Primary.
struct SideSensor {
  HidSixAxisSensorHandle handle{};
  bool resolved = false;
  bool started = false;
  bool owned = false;  // this module started the hardware sensor for the side; Right shares Primary's, which is not stopped
  motion::Cursor cursor;  // where the one reader of this sensor's state queue stopped (motion_samples)
};
SideSensor s_side[2];  // [0] Left, [1] Right

SideSensor* side_sensor(const SensorSide side) {
  return side == SensorSide::Left ? &s_side[0] : side == SensorSide::Right ? &s_side[1] : nullptr;
}

void stop_side(SideSensor& s) {
  if (s.started && s.owned) {
    hidStopSixAxisSensor(s.handle);
  }
  s = SideSensor{};
}

// Converts libnx's g to the m/s^2 the rest of aurora (and SDL) use.
constexpr float kGToMetresPerSecSq = 9.80665f;

bool detached_pair_style(HidNpadIdType& id, HidNpadStyleTag& style) {
  ensure_pad();
  padUpdate(&s_pad);
  if (padIsHandheld(&s_pad) || !padIsNpadActive(&s_pad, HidNpadIdType_No1)) {
    return false;
  }
  id = HidNpadIdType_No1;
  if (!choose_style(hidGetNpadStyleSet(HidNpadIdType_No1), style)) {
    return false;
  }
  return style == HidNpadStyleTag_NpadJoyDual;
}

// The Left/Right handles come in a pair; a pad that is no longer a detached pair drops them.
SensorLayout layout(const bool start) {
  SensorLayout out;
  HidNpadIdType id = HidNpadIdType_No1;
  HidNpadStyleTag style = static_cast<HidNpadStyleTag>(0);
  if (!detached_pair_style(id, style)) {
    ensure_pad();
    // Handheld, Pro Controller or a lone Joy-Con: one sensor, Primary.
    out.primary = padIsHandheld(&s_pad) ||
                  (padIsNpadActive(&s_pad, HidNpadIdType_No1) && hidGetNpadStyleSet(HidNpadIdType_No1) != 0);
    for (auto& s : s_side) {
      stop_side(s);
    }
    return out;
  }
  out.primary = true;
  out.detached_pair = true;
  HidSixAxisSensorHandle handles[2]{};
  if (R_FAILED(hidGetSixAxisSensorHandles(handles, 2, id, style))) {
    return out;
  }
  for (int i = 0; i < 2; ++i) {
    SideSensor& s = s_side[i];
    if (s.resolved && s.handle.type_value != handles[i].type_value) {
      stop_side(s);
    }
    s.handle = handles[i];
    s.resolved = true;
    if (start && !s.started) {
      if (i == 1) {
        // The right Joy-Con is Primary's sensor: start it the way a Primary read does, and leave it to that path.
        s.started = ensure_started();
      } else {
        s.started = R_SUCCEEDED(hidStartSixAxisSensor(s.handle));
        s.owned = s.started;
      }
    }
  }
  out.left = s_side[0].resolved && (!start || s_side[0].started);
  out.right = s_side[1].resolved && (!start || s_side[1].started);
  return out;
}

// The one reader of a sensor's state queue (libnx keeps 17 states, newest first): everything since the last call, oldest
// first, in m/s^2 and rad/s with the time each covers. See motion_queue.hpp for what a gap and delta_time mean.
int read_motion(const HidSixAxisSensorHandle handle, motion::Cursor& cursor, MotionSample* out, const int max_samples) {
  constexpr int kQueue = 17;
  HidSixAxisSensorState states[kQueue]{};
  const s32 got = hidGetSixAxisSensorStates(handle, states, kQueue);
  if (got <= 0) {
    return 0;
  }
  motion::RawState raw[kQueue];
  for (s32 i = 0; i < got; ++i) {
    raw[i].samplingNumber = states[i].sampling_number;
    raw[i].deltaTime = states[i].delta_time;
    raw[i].accel[0] = states[i].acceleration.x;
    raw[i].accel[1] = states[i].acceleration.y;
    raw[i].accel[2] = states[i].acceleration.z;
    raw[i].gyro[0] = states[i].angular_velocity.x;
    raw[i].gyro[1] = states[i].angular_velocity.y;
    raw[i].gyro[2] = states[i].angular_velocity.z;
    raw[i].connected = (states[i].attributes & HidSixAxisSensorAttribute_IsConnected) != 0;
  }
  motion::Sample ready[kQueue];
  const int n = motion::ingest(cursor, raw, got, ready, std::min(max_samples, kQueue));
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      out[i].accel[k] = ready[i].accel[k];
      out[i].gyro[k] = ready[i].gyro[k];
    }
    out[i].sampling_number = ready[i].samplingNumber;
    out[i].delta_us = ready[i].deltaUs;
    out[i].flags = ready[i].flags;
  }
  return n;
}

// ---------------------------------------------------------------------------
// aleks2: rumble over libnx.
//
// SDL3's Switch backend answers SDL_Unsupported() to SDL_JoystickRumble, so
// aurora::device::rumble fell into the no-op branch of device.cpp and the
// console had no vibration at all -- neither the companion's touch haptics nor
// the game's own, since PADControlMotor routes through here whenever the SDL
// controller reports no rumble (see dolphin/pad/pad.cpp,
// should_use_device_rumble).
//
// The two callers ask for different things. PAD starts an indefinite pulse
// (durationMs == 0) and ends it with an explicit rumble(0, 0, 0); the companion
// fires 12-70 ms one-shots and never stops them. So a timed pulse needs someone
// to switch the motors off later, and sleeping out 70 ms on the caller would
// stall the game thread -- hence one thread that does nothing but wait out the
// deadline.

constexpr float kMotorFreqLow = 160.0f;
constexpr float kMotorFreqHigh = 320.0f;

std::mutex s_rumbleMutex;
std::condition_variable s_rumbleWake;
std::thread s_rumbleThread;
bool s_rumbleQuit = false;
bool s_rumblePulsing = false;
std::chrono::steady_clock::time_point s_rumbleUntil{};

HidVibrationDeviceHandle s_motors[2]{};
s32 s_motorCount = 0;
HidNpadIdType s_motorId = HidNpadIdType_No1;
HidNpadStyleTag s_motorStyle = static_cast<HidNpadStyleTag>(0);
bool s_motorsReady = false;

// Callers hold s_rumbleMutex.
void send_motors(const float ampLow, const float ampHigh) {
  if (!s_motorsReady) {
    return;
  }
  HidVibrationValue values[2]{};
  for (s32 i = 0; i < s_motorCount; i++) {
    values[i].amp_low = ampLow;
    values[i].freq_low = kMotorFreqLow;
    values[i].amp_high = ampHigh;
    values[i].freq_high = kMotorFreqHigh;
  }
  hidSendVibrationValues(s_motors, values, s_motorCount);
}

// Game thread only: it calls padUpdate, which the worker never does.
bool resolve_motors() {
  ensure_pad();
  padUpdate(&s_pad);

  HidNpadIdType id = HidNpadIdType_No1;
  HidNpadStyleTag style = static_cast<HidNpadStyleTag>(0);
  if (padIsHandheld(&s_pad)) {
    id = HidNpadIdType_Handheld;
    style = HidNpadStyleTag_NpadHandheld;
  } else if (padIsNpadActive(&s_pad, HidNpadIdType_No1)) {
    if (!choose_style(hidGetNpadStyleSet(HidNpadIdType_No1), style)) {
      return false;
    }
  } else {
    return false;
  }

  if (s_motorsReady && id == s_motorId && style == s_motorStyle) {
    return true;
  }
  // A lone Joy-Con has one motor; handheld, Pro and dual have two.
  const s32 count =
      (style == HidNpadStyleTag_NpadJoyLeft || style == HidNpadStyleTag_NpadJoyRight) ? 1 : 2;
  if (R_FAILED(hidInitializeVibrationDevices(s_motors, count, id, style))) {
    s_motorsReady = false;
    return false;
  }
  s_motorCount = count;
  s_motorId = id;
  s_motorStyle = style;
  s_motorsReady = true;
  return true;
}

void rumble_worker() {
  std::unique_lock lock{s_rumbleMutex};
  while (!s_rumbleQuit) {
    if (!s_rumblePulsing) {
      s_rumbleWake.wait(lock);
      continue;
    }
    if (s_rumbleWake.wait_until(lock, s_rumbleUntil) == std::cv_status::timeout &&
        s_rumblePulsing && std::chrono::steady_clock::now() >= s_rumbleUntil) {
      send_motors(0.0f, 0.0f);
      s_rumblePulsing = false;
    }
  }
}

void ensure_rumble_worker() {
  if (!s_rumbleThread.joinable() && !s_rumbleQuit) {
    s_rumbleThread = std::thread{rumble_worker};
  }
}

}  // namespace

bool gyro_available(const SensorSide side) noexcept {
  const SideSensor* s = side_sensor(side);
  return s == nullptr ? true : (s->resolved && s->started);
}
bool gyro(const SensorSide side, float* data, const int n_values) noexcept {
  SideSensor* s = side_sensor(side);
  if (s == nullptr) {
    return read(true, data, n_values);
  }
  if (data == nullptr || n_values <= 0) {
    return false;
  }
  std::fill(data, data + n_values, 0.0f);
  if (!s->resolved || !s->started) {
    return false;
  }
  HidSixAxisSensorState state{};
  if (hidGetSixAxisSensorStates(s->handle, &state, 1) == 0 ||
      (state.attributes & HidSixAxisSensorAttribute_IsConnected) == 0) {
    return false;
  }
  constexpr float kRevPerSecToRadPerSec = 6.28318530717958647692f;
  const float v[3] = {state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z};
  for (int i = 0; i < n_values && i < 3; ++i) {
    data[i] = v[i] * kRevPerSecToRadPerSec;
  }
  return true;
}
bool accel_available(const SensorSide side) noexcept { return gyro_available(side); }
bool accel(const SensorSide side, float* data, const int n_values) noexcept {
  SideSensor* s = side_sensor(side);
  if (s == nullptr) {
    return read(false, data, n_values);
  }
  float xyz[3];
  if (data == nullptr || n_values <= 0) {
    return false;
  }
  std::fill(data, data + n_values, 0.0f);
  HidSixAxisSensorState state{};
  if (!s->resolved || !s->started || hidGetSixAxisSensorStates(s->handle, &state, 1) == 0 ||
      (state.attributes & HidSixAxisSensorAttribute_IsConnected) == 0) {
    return false;
  }
  xyz[0] = state.acceleration.x * kGToMetresPerSecSq;
  xyz[1] = state.acceleration.y * kGToMetresPerSecSq;
  xyz[2] = state.acceleration.z * kGToMetresPerSecSq;
  for (int i = 0; i < n_values && i < 3; ++i) {
    data[i] = xyz[i];
  }
  return true;
}

int motion_samples(const SensorSide side, MotionSample* out, const int max_samples) noexcept {
  if (out == nullptr || max_samples < 1) {
    return 0;
  }
  SideSensor* s = side_sensor(side);
  if (s != nullptr) {
    if (!s->resolved || !s->started) {
      return 0;
    }
    return read_motion(s->handle, s->cursor, out, max_samples);
  }
  // Primary: the sensor read() uses; this call keeps its own cursor over the same (read-only) state queue.
  if (!ensure_started()) {
    return 0;
  }
  return read_motion(s_handle, s_primaryCursor, out, max_samples);
}

bool motion_stats(const SensorSide side, MotionStats* out) noexcept {
  if (out == nullptr) {
    return false;
  }
  const motion::Cursor* c = nullptr;
  SideSensor* s = side_sensor(side);
  if (s != nullptr) {
    if (!s->resolved) {
      return false;
    }
    c = &s->cursor;
  } else {
    c = &s_primaryCursor;
  }
  out->dropped = c->dropped;
  out->gaps = c->gaps;
  out->raw_delta_min = c->timing.samples != 0 ? c->timing.rawMin : 0;
  out->raw_delta_max = c->timing.rawMax;
  out->as_nanoseconds = c->timing.asNanoseconds;
  out->as_microseconds = c->timing.asMicroseconds;
  out->fallback = c->timing.fallback;
  out->samples = c->timing.samples;
  return true;
}

// The latest acceleration of `side`, one sample, for callers that predate motion_samples(). It reads the newest state without
// touching the cursor of the queue reader: motion_samples() is the only consumer of a sensor's queue, so this cannot take
// samples from it (or hand them out twice).
int accel_samples(const SensorSide side, float* xyz, const int max_samples) noexcept {
  if (xyz == nullptr || max_samples < 1) {
    return 0;
  }
  SideSensor* s = side_sensor(side);
  if (s == nullptr) {
    // Primary: the sensor read() uses, newest sample only.
    return accel(SensorSide::Primary, xyz, 3) ? 1 : 0;
  }
  if (!s->resolved || !s->started) {
    return 0;
  }
  HidSixAxisSensorState state{};
  if (hidGetSixAxisSensorStates(s->handle, &state, 1) == 0 ||
      (state.attributes & HidSixAxisSensorAttribute_IsConnected) == 0) {
    return 0;
  }
  xyz[0] = state.acceleration.x * kGToMetresPerSecSq;
  xyz[1] = state.acceleration.y * kGToMetresPerSecSq;
  xyz[2] = state.acceleration.z * kGToMetresPerSecSq;
  return 1;
}

SensorLayout sensor_layout(const bool start_sensors) noexcept { return layout(start_sensors); }

void stop_side_sensors() noexcept {
  for (auto& s : s_side) {
    stop_side(s);
  }
}

bool stick_clicks(bool* left, bool* right) noexcept {
  ensure_pad();
  padUpdate(&s_pad);
  const u64 held = padGetButtons(&s_pad);
  if (left != nullptr) *left = (held & HidNpadButton_StickL) != 0;
  if (right != nullptr) *right = (held & HidNpadButton_StickR) != 0;
  return padIsNpadActive(&s_pad, HidNpadIdType_No1) || padIsHandheld(&s_pad);
}

bool rumble_available() noexcept { return true; }

void rumble(const uint16_t lowFreqIntensity, const uint16_t highFreqIntensity,
            const uint16_t durationMs) noexcept {
  if (!resolve_motors()) {
    return;
  }
  const float ampLow = std::clamp(static_cast<float>(lowFreqIntensity) / 65535.0f, 0.0f, 1.0f);
  const float ampHigh = std::clamp(static_cast<float>(highFreqIntensity) / 65535.0f, 0.0f, 1.0f);
  // Zero amplitude is how PAD says stop; a zero duration means "until I do".
  const bool timed = durationMs != 0 && (ampLow > 0.0f || ampHigh > 0.0f);
  if (timed) {
    ensure_rumble_worker();
  }

  std::lock_guard lock{s_rumbleMutex};
  send_motors(ampLow, ampHigh);
  s_rumblePulsing = timed;
  if (timed) {
    s_rumbleUntil = std::chrono::steady_clock::now() + std::chrono::milliseconds{durationMs};
  }
  s_rumbleWake.notify_one();
}

namespace detail {
void shutdown_rumble() noexcept {
  {
    std::lock_guard lock{s_rumbleMutex};
    s_rumbleQuit = true;
    s_rumbleWake.notify_one();
  }
  if (s_rumbleThread.joinable()) {
    s_rumbleThread.join();
  }
  // Also covers an indefinite pulse, which the worker leaves running.
  std::lock_guard lock{s_rumbleMutex};
  send_motors(0.0f, 0.0f);
}
}  // namespace detail

}  // namespace aurora::device
#endif
