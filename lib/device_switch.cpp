// aleks2: aurora::device gyro/accel for the Switch over libnx's six-axis
// sensor. Ported from Aleks' pad_switch.cpp (aurora 08167a62). NX's SDL3-switch
// backend has no sensor support, and dolphin/pad/pad.cpp already falls back to
// aurora::device when the SDL gamepad reports no sensor, so this is the only
// missing piece for gyro aim.
#ifdef __SWITCH__
#include <aurora/aurora.h>
#include <switch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include <aurora/hd_rumble.h>

#include "device.hpp"
#include "logging.hpp"
#include "motion_queue.hpp"
#include "player_one_rule.hpp"

namespace aurora::device {
namespace {

constexpr Module Log{"aurora::device"};

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

// ---------------------------------------------------------------------------
// aleks2 (rc17 console bug): the npad that is player 1.
//
// The game's pad 0 is SDL's one Switch joystick, which reads Handheld and No1 together (padInitializeDefault). So
// both drive player 1, and vibration and six-axis must go to the one in use. rc16 took Handheld whenever the
// Joy-Con were attached: with a DualSense on No1 (MissionControl, seen as a Pro Controller) and the Joy-Con on the
// console, HD Rumble buzzed the Joy-Con and gyro aim read a console lying still.
//
// The rule: an npad that is connected on its own wins. When both are connected, the one that last had input (a
// button press or a stick off-centre) wins; until one does, No1 (an external controller was connected for a
// reason). player_one_track() follows this every frame (aurora_player_one_tick); everything below asks
// player_one().
PadState s_padNo1{};
PadState s_padHandheld{};
bool s_routeInit = false;
bool s_p1Known = false;
HidNpadIdType s_p1Id = HidNpadIdType_No1;

bool npad_style(const HidNpadIdType id, HidNpadStyleTag& style) {
  const u32 set = hidGetNpadStyleSet(id);
  if (id == HidNpadIdType_Handheld) {
    if ((set & HidNpadStyleTag_NpadHandheld) == 0) {
      return false;
    }
    style = HidNpadStyleTag_NpadHandheld;
    return true;
  }
  return choose_style(set, style);
}

bool had_input(PadState& pad) {
  padUpdate(&pad);
  if (padGetButtonsDown(&pad) != 0) {
    return true;
  }
  constexpr s32 kOffCentre = 9000;  // of 32767: a stick deliberately pushed, not drift
  for (int stick = 0; stick < 2; ++stick) {
    const HidAnalogStickState s = padGetStickPos(&pad, stick);
    if (s.x > kOffCentre || s.x < -kOffCentre || s.y > kOffCentre || s.y < -kOffCentre) {
      return true;
    }
  }
  return false;
}

const char* style_name(const HidNpadStyleTag style) {
  switch (style) {
  case HidNpadStyleTag_NpadFullKey: return "Pro/FullKey";
  case HidNpadStyleTag_NpadHandheld: return "Handheld";
  case HidNpadStyleTag_NpadJoyDual: return "JoyDual";
  case HidNpadStyleTag_NpadJoyLeft: return "JoyLeft";
  case HidNpadStyleTag_NpadJoyRight: return "JoyRight";
  default: return "?";
  }
}

void player_one_track() {
  ensure_pad();
  if (!s_routeInit) {
    padInitialize(&s_padNo1, HidNpadIdType_No1);
    padInitialize(&s_padHandheld, HidNpadIdType_Handheld);
    s_routeInit = true;
  }
  using player_one_rule::Npad;
  HidNpadStyleTag style{};
  player_one_rule::Seen seen;
  seen.no1Connected = npad_style(HidNpadIdType_No1, style);
  seen.handheldConnected = npad_style(HidNpadIdType_Handheld, style);
  seen.no1Input = had_input(s_padNo1);
  seen.handheldInput = had_input(s_padHandheld);
  const bool no1 = seen.no1Connected;
  const bool handheld = seen.handheldConnected;
  const Npad previous = !s_p1Known ? Npad::None : s_p1Id == HidNpadIdType_Handheld ? Npad::Handheld : Npad::No1;
  const Npad pick = player_one_rule::choose(seen, previous);
  if (pick == Npad::None) {
    s_p1Known = false;
    return;
  }
  const HidNpadIdType chosen = pick == Npad::Handheld ? HidNpadIdType_Handheld : HidNpadIdType_No1;
  if (!s_p1Known || chosen != s_p1Id) {
    HidNpadStyleTag chosenStyle{};
    npad_style(chosen, chosenStyle);
    Log.info("player 1 is npad {} ({}){}", chosen == HidNpadIdType_Handheld ? "Handheld" : "No1",
             style_name(chosenStyle), no1 && handheld ? "; the other one is connected too" : "");
  }
  s_p1Id = chosen;
  s_p1Known = true;
}

// The npad and style that vibration and six-axis go to.
bool player_one(HidNpadIdType& id, HidNpadStyleTag& style) {
  if (!s_p1Known) {
    player_one_track();
    if (!s_p1Known) {
      return false;
    }
  }
  id = s_p1Id;
  if (npad_style(id, style)) {
    return true;
  }
  // It went away since the last tick (undocked, a controller switched off): look again now.
  player_one_track();
  if (!s_p1Known) {
    return false;
  }
  id = s_p1Id;
  return npad_style(id, style);
}

// aleks2 (rc17): a sensor that gives no motion data at all -- common for non-Nintendo pads behind MissionControl,
// which take the Pro Controller's place without its six-axis -- is noted per npad and style. gyro_available() is
// then false for it, so the game keeps the classic controls; the gyro setting stays as the player set it. The
// reason goes to the log once. When another controller becomes player 1, its sensor is tried afresh.
constexpr auto kMotionGrace = std::chrono::milliseconds{1500};
bool s_motionSeen = false;
bool s_noMotion = false;
HidNpadIdType s_noMotionId = HidNpadIdType_No1;
HidNpadStyleTag s_noMotionStyle{};
std::chrono::steady_clock::time_point s_motionSince{};

void note_motion(const bool connected, const HidSixAxisSensorState& state) {
  if (s_motionSeen || s_noMotion) {
    return;
  }
  // A working sensor always shows gravity; one that answers with zeros, or not at all, has nothing.
  const HidVector& a = state.acceleration;
  if (connected && (a.x != 0.0f || a.y != 0.0f || a.z != 0.0f)) {
    s_motionSeen = true;
    return;
  }
  if (std::chrono::steady_clock::now() - s_motionSince < kMotionGrace) {
    return;
  }
  HidNpadIdType id{};
  HidNpadStyleTag style{};
  if (!player_one(id, style)) {
    return;
  }
  s_noMotion = true;
  s_noMotionId = id;
  s_noMotionStyle = style;
  Log.warn("gyro: player 1 (npad {}, {}) gives no motion data{}; gyro aim falls back to the classic controls",
           id == HidNpadIdType_Handheld ? "Handheld" : "No1", style_name(style),
           connected ? " (all zero)" : " (sensor not connected)");
}

bool no_motion_for_player_one() {
  if (!s_noMotion) {
    return false;
  }
  HidNpadIdType id{};
  HidNpadStyleTag style{};
  if (!player_one(id, style)) {
    return true;
  }
  if (id == s_noMotionId && style == s_noMotionStyle) {
    return true;
  }
  s_noMotion = false;  // another controller is player 1 now: try its sensor
  return false;
}

bool resolve_handle(HidSixAxisSensorHandle& outHandle) {
  HidNpadIdType id = HidNpadIdType_No1;
  HidNpadStyleTag style = static_cast<HidNpadStyleTag>(0);
  if (!player_one(id, style)) {
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
    // aleks2 (rc17): a newly started sensor gets its grace period to show motion data.
    s_motionSeen = false;
    s_motionSince = std::chrono::steady_clock::now();
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
  const bool got = hidGetSixAxisSensorStates(s_handle, &state, 1) != 0;
  const bool connected = got && (state.attributes & HidSixAxisSensorAttribute_IsConnected) != 0;
  note_motion(connected, state);  // aleks2 (rc17)
  if (!connected) {
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
  // aleks2 (rc17): player 1's npad, not "No1 unless the console has Joy-Con attached".
  if (!player_one(id, style)) {
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
    // Handheld, Pro Controller or a lone Joy-Con: one sensor, Primary. aleks2 (rc17): none when player 1's
    // controller gives no motion data.
    out.primary = player_one(id, style) && !no_motion_for_player_one();
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
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        raw[i].dir[r * 3 + c] = states[i].direction.direction[r][c];
      }
    }
    raw[i].connected = (states[i].attributes & HidSixAxisSensorAttribute_IsConnected) != 0;
  }
  motion::Sample ready[kQueue];
  const int n = motion::ingest(cursor, raw, got, ready, std::min(max_samples, kQueue));
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      out[i].accel[k] = ready[i].accel[k];
      out[i].gyro[k] = ready[i].gyro[k];
    }
    for (int k = 0; k < 9; ++k) {
      out[i].dir[k] = ready[i].dir[k];
    }
    out[i].sampling_number = ready[i].samplingNumber;
    out[i].delta_us = ready[i].deltaUs;
    out[i].flags = ready[i].flags | kMotionHasDirection;
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

// aleks2 (V1-rc16): the intensity setting, applied to everything sent to the motors.
std::atomic<float> s_rumbleGain{1.0f};

// Callers hold s_rumbleMutex.
void send_motors(const float ampLow, const float ampHigh) {
  if (!s_motorsReady) {
    return;
  }
  const float gain = s_rumbleGain.load(std::memory_order_relaxed);
  HidVibrationValue values[2]{};
  for (s32 i = 0; i < s_motorCount; i++) {
    values[i].amp_low = ampLow * gain;
    values[i].freq_low = kMotorFreqLow;
    values[i].amp_high = ampHigh * gain;
    values[i].freq_high = kMotorFreqHigh;
  }
  hidSendVibrationValues(s_motors, values, s_motorCount);
}

// aleks2 (V1-rc16): HD Rumble (aurora/hd_rumble.h). Two layers of keyframes, played by the rumble worker.
struct HdLayer {
  std::vector<AuroraHdKey> keys;
  bool loop = false;
  bool active = false;
  std::chrono::steady_clock::time_point start{};
  uint32_t totalMs = 0;
};
HdLayer s_hd[2];
bool s_hdSent = false;  // the last thing sent was an HD value (so the end sends one stop)

// The layer's value at `now`, or false when it has ended. Callers hold s_rumbleMutex.
bool hd_eval(HdLayer& layer, const std::chrono::steady_clock::time_point now, AuroraHdKey& out) {
  if (!layer.active || layer.keys.empty() || layer.totalMs == 0) {
    layer.active = false;
    return false;
  }
  uint32_t t = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now - layer.start).count());
  if (t >= layer.totalMs) {
    if (!layer.loop) {
      layer.active = false;
      return false;
    }
    t %= layer.totalMs;
  }
  for (size_t i = 0; i < layer.keys.size(); ++i) {
    const AuroraHdKey& a = layer.keys[i];
    if (t < a.ms) {
      // towards the next key (the last one holds, or eases towards the first when looping)
      const bool last = i + 1 == layer.keys.size();
      const AuroraHdKey& b = last ? (layer.loop ? layer.keys.front() : a) : layer.keys[i + 1];
      const float f = a.ms == 0 ? 0.0f : static_cast<float>(t) / static_cast<float>(a.ms);
      out.amp_low = a.amp_low + (b.amp_low - a.amp_low) * f;
      out.freq_low = a.freq_low + (b.freq_low - a.freq_low) * f;
      out.amp_high = a.amp_high + (b.amp_high - a.amp_high) * f;
      out.freq_high = a.freq_high + (b.freq_high - a.freq_high) * f;
      out.pan = a.pan + (b.pan - a.pan) * f;
      return true;
    }
    t -= a.ms;
  }
  return false;
}

// One side of the pad from a band value and its balance.
float side_amp(const float amp, const float pan, const bool right) {
  return std::clamp(amp * (right ? std::min(1.0f, 1.0f + pan) : std::min(1.0f, 1.0f - pan)), 0.0f, 1.0f);
}

// Callers hold s_rumbleMutex. Each band takes the stronger layer; a lone Joy-Con gets both sides at once.
void send_hd(const AuroraHdKey* const layers[2]) {
  if (!s_motorsReady) {
    return;
  }
  const float gain = s_rumbleGain.load(std::memory_order_relaxed);
  const bool single = s_motorCount == 1;
  HidVibrationValue values[2]{};
  for (s32 m = 0; m < s_motorCount; ++m) {
    float ampLow = 0.0f, ampHigh = 0.0f, freqLow = kMotorFreqLow, freqHigh = kMotorFreqHigh;
    for (int l = 0; l < 2; ++l) {
      const AuroraHdKey* k = layers[l];
      if (k == nullptr) {
        continue;
      }
      const float low = single ? k->amp_low : side_amp(k->amp_low, k->pan, m == 1);
      const float high = single ? k->amp_high : side_amp(k->amp_high, k->pan, m == 1);
      if (low > ampLow) {
        ampLow = low;
        freqLow = k->freq_low;
      }
      if (high > ampHigh) {
        ampHigh = high;
        freqHigh = k->freq_high;
      }
    }
    values[m].amp_low = ampLow * gain;
    values[m].freq_low = std::clamp(freqLow, 40.0f, 640.0f);
    values[m].amp_high = ampHigh * gain;
    values[m].freq_high = std::clamp(freqHigh, 80.0f, 1280.0f);
  }
  hidSendVibrationValues(s_motors, values, s_motorCount);
  s_hdSent = true;
}

// Callers hold s_rumbleMutex. True while a layer still plays.
bool hd_step(const std::chrono::steady_clock::time_point now) {
  AuroraHdKey values[2]{};
  const AuroraHdKey* layers[2]{};
  bool any = false;
  for (int l = 0; l < 2; ++l) {
    if (hd_eval(s_hd[l], now, values[l])) {
      layers[l] = &values[l];
      any = true;
    }
  }
  if (any) {
    send_hd(layers);
  } else if (s_hdSent) {
    send_motors(0.0f, 0.0f);
    s_hdSent = false;
  }
  return any;
}

// Game thread only: it calls padUpdate, which the worker never does.
bool resolve_motors() {
  // aleks2 (rc17): player 1's npad and style (Pro/FullKey, JoyDual, Handheld, JoyLeft/Right), never "Handheld
  // whenever the Joy-Con are attached".
  HidNpadIdType id = HidNpadIdType_No1;
  HidNpadStyleTag style = static_cast<HidNpadStyleTag>(0);
  if (!player_one(id, style)) {
    return false;
  }

  if (s_motorsReady && id == s_motorId && style == s_motorStyle) {
    return true;
  }
  if (s_motorsReady) {
    // Player 1 changed: the controller left behind must not keep buzzing.
    std::lock_guard lock{s_rumbleMutex};
    send_motors(0.0f, 0.0f);
    s_motorsReady = false;
  }
  // A lone Joy-Con has one motor; handheld, Pro and dual have two.
  const s32 count =
      (style == HidNpadStyleTag_NpadJoyLeft || style == HidNpadStyleTag_NpadJoyRight) ? 1 : 2;
  HidVibrationDeviceHandle motors[2]{};
  const bool ok = R_SUCCEEDED(hidInitializeVibrationDevices(motors, count, id, style));
  // aleks2 (V1-rc16): the worker sends HD values from these handles, so they change under its lock.
  std::lock_guard lock{s_rumbleMutex};
  if (!ok) {
    s_motorsReady = false;
    return false;
  }
  s_motors[0] = motors[0];
  s_motors[1] = motors[1];
  s_motorCount = count;
  s_motorId = id;
  s_motorStyle = style;
  s_motorsReady = true;
  return true;
}

void rumble_worker() {
  std::unique_lock lock{s_rumbleMutex};
  while (!s_rumbleQuit) {
    // aleks2 (V1-rc16): an HD envelope plays at ~5 ms steps and takes precedence over a timed pulse.
    if (s_hd[0].active || s_hd[1].active || s_hdSent) {
      if (hd_step(std::chrono::steady_clock::now())) {
        s_rumbleWake.wait_for(lock, std::chrono::milliseconds{5});
      }
      continue;
    }
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
  // aleks2 (rc17): Primary is unavailable while player 1's controller gives no motion data (classic controls).
  return s == nullptr ? !no_motion_for_player_one() : (s->resolved && s->started);
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
  if (no_motion_for_player_one() || !ensure_started()) {
    return 0;
  }
  const int n = read_motion(s_handle, s_primaryCursor, out, max_samples);
  if (!s_motionSeen && !s_noMotion) {  // aleks2 (rc17): the no-motion check, as read() does it
    HidSixAxisSensorState state{};
    const bool got = hidGetSixAxisSensorStates(s_handle, &state, 1) != 0;
    note_motion(got && (state.attributes & HidSixAxisSensorAttribute_IsConnected) != 0, state);
  }
  return n;
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

// aleks2 (V1-rc16): aurora/hd_rumble.h. aurora_hd_rumble_available() lives in dolphin/pad/pad.cpp (it needs the
// pad's controller); the rest is here.
extern "C" void aurora_hd_rumble_play(const int layer, const AuroraHdKey* keys, const int count, const bool loop) {
  using namespace aurora::device;
  if (layer < 0 || layer > 1) {
    return;
  }
  if (count > 0 && !resolve_motors()) {
    return;
  }
  {
    std::lock_guard lock{s_rumbleMutex};
    HdLayer& l = s_hd[layer];
    l.keys.assign(keys, keys + std::max(count, 0));
    l.loop = loop;
    l.totalMs = 0;
    for (const auto& k : l.keys) {
      l.totalMs += k.ms;
    }
    l.start = std::chrono::steady_clock::now();
    l.active = !l.keys.empty() && l.totalMs != 0;
    s_rumblePulsing = false;  // a classic timed pulse gives way
  }
  ensure_rumble_worker();
  s_rumbleWake.notify_one();
}

extern "C" void aurora_hd_rumble_stop(void) {
  using namespace aurora::device;
  std::lock_guard lock{s_rumbleMutex};
  s_hd[0].active = s_hd[1].active = false;
  if (s_hdSent) {
    send_motors(0.0f, 0.0f);
    s_hdSent = false;
  }
  s_rumbleWake.notify_one();
}

extern "C" void aurora_rumble_set_gain(const float gain) {
  aurora::device::s_rumbleGain.store(std::clamp(gain, 0.0f, 1.0f), std::memory_order_relaxed);
}

// aleks2 (rc17): which npad is player 1, followed once per frame (game thread). A change also moves the motors
// right away, so a loop that is playing (a quake, a gallop) continues on the new controller and the old one stops.
extern "C" void aurora_player_one_tick(void) {
  using namespace aurora::device;
  const bool known = s_p1Known;
  const HidNpadIdType before = s_p1Id;
  player_one_track();
  if (!s_motorsReady || (known && s_p1Known && s_p1Id == before)) {
    return;
  }
  if (!s_p1Known) {
    std::lock_guard lock{s_rumbleMutex};
    send_motors(0.0f, 0.0f);
    s_motorsReady = false;
    return;
  }
  resolve_motors();
}
#endif
