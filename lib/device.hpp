#pragma once

#include <cstdint>

namespace aurora::device {

// aleks2 (H38): which motion sensor a read asks for. Primary is the one the pad API has always used (gyro aim): the
// controller's own sensor, or the right Joy-Con of a detached pair. Left and Right name the two Joy-Cons of a
// detached pair separately; every other backend, and every other controller, answers false for them.
enum class SensorSide { Primary, Left, Right };

// What the sensors of the active controller are right now.
struct SensorLayout {
  bool primary = false;        // a Primary sensor exists
  bool left = false;           // a separate Left sensor exists (detached Joy-Con pair)
  bool right = false;          // a separate Right sensor exists (detached Joy-Con pair)
  bool detached_pair = false;  // two Joy-Con, as opposed to one controller with one sensor
};

bool rumble_available() noexcept;
void rumble(uint16_t lowFreq, uint16_t highFreq, uint16_t durationMs) noexcept;

bool gyro_available(SensorSide side) noexcept;
bool gyro(SensorSide side, float* data, int n_values) noexcept;

bool accel_available(SensorSide side) noexcept;
bool accel(SensorSide side, float* data, int n_values) noexcept;

// The latest acceleration of `side` as an x, y, z triple in m/s^2 (one sample; the return value is how many, 0 when there is
// no such sensor). Since H38-c3 it no longer drains the sensor's queue: use motion_samples() for the samples since the last
// tick.
int accel_samples(SensorSide side, float* xyz, int max_samples) noexcept;

// aleks2 (H38-c3): accelerometer and gyroscope together, with the time each sample covers. The one reader of a sensor's
// state queue: it hands out the states since its last call, oldest first, and remembers where it stopped, so a second
// reader would find nothing. accel_samples() below does not read that queue: it returns the latest acceleration only.
constexpr uint8_t kMotionGapBefore = 1;  // MotionSample::flags: states were lost (or skipped) just before this sample
constexpr uint8_t kMotionHasDirection = 2;  // aleks2 (V1-rc12): MotionSample::dir holds HID's attitude
struct MotionSample {
  float accel[3] = {};  // m/s^2, gravity included
  float gyro[3] = {};   // rad/s
  float dir[9] = {};    // aleks2 (V1-rc12): HID's fused attitude (3x3, row-major) when kMotionHasDirection is set
  uint64_t sampling_number = 0;
  uint32_t delta_us = 0;  // microseconds this sample covers; 0 when the backend does not know (SDL: latest sample only)
  uint8_t flags = 0;
};
int motion_samples(SensorSide side, MotionSample* out, int max_samples) noexcept;

// What the queue reads so far looked like for `side`: states lost, gap events and the raw delta_time range with how it
// was interpreted. Returns false when there is no such sensor.
struct MotionStats {
  uint64_t dropped = 0;
  uint32_t gaps = 0;
  uint64_t raw_delta_min = 0;
  uint64_t raw_delta_max = 0;
  uint32_t as_nanoseconds = 0;
  uint32_t as_microseconds = 0;
  uint32_t fallback = 0;
  uint32_t samples = 0;
};
bool motion_stats(SensorSide side, MotionStats* out) noexcept;

// Looks at the active controller. With start_sensors the Left and Right sensors of a detached pair are switched
// on (they are otherwise left off: a running six-axis sensor costs battery); without it only the layout is
// reported. Call it once per tick before reading a side sensor.
SensorLayout sensor_layout(bool start_sensors) noexcept;
// Switches the Left and Right sensors off again.
void stop_side_sensors() noexcept;

// The functions that predate sides: the Primary sensor.
inline bool gyro_available() noexcept { return gyro_available(SensorSide::Primary); }
inline bool gyro(float* data, int n_values) noexcept { return gyro(SensorSide::Primary, data, n_values); }
inline bool accel_available() noexcept { return accel_available(SensorSide::Primary); }
inline bool accel(float* data, int n_values) noexcept { return accel(SensorSide::Primary, data, n_values); }

// aleks2 (H38-c4): whether the stick buttons (L3, R3) of the active controller are held right now. Both false, and a false
// return, when there is no controller. Used for the Wii-style motion recenter chord; the game's own pad mapping has no
// stick buttons, so this reads the controller directly.
bool stick_clicks(bool* left, bool* right) noexcept;

void shutdown() noexcept;

} // namespace aurora::device
