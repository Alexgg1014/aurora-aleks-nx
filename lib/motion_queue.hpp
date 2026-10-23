#pragma once

// aleks2 (H38-c3): the bookkeeping for reading a six-axis sensor's state queue, with no libnx in it so it can be tested on
// the host. libnx keeps up to 17 states per sensor (about 85 ms at 200 Hz), newest first; every read returns the states it
// has, not only the new ones. This turns that into "the samples since the last read, oldest first", and says what it could
// not deliver: a state that was overwritten before it was read (a long frame), a counter that restarted (a reconnected
// controller) or a state flagged as not connected. A gap is not an error to hide: the consumer must not treat a movement
// that spans one as a clean gesture.
//
// delta_time is documented by libnx only as "DeltaTime". The unit is not assumed: a value that can be a plausible sample
// period in nanoseconds (1..40 ms) or in microseconds is taken as such, and anything else falls back to the sampling
// number's own cadence (nominal 5 ms per step). The raw range seen is kept so the log can say what the console really sends.

#include <algorithm>
#include <cstdint>

namespace aurora::device::motion {

struct RawState {
  uint64_t samplingNumber = 0;
  uint64_t deltaTime = 0;  // as libnx reports it
  float accel[3] = {};     // g
  float gyro[3] = {};      // revolutions per second
  float dir[9] = {};       // aleks2 (V1-rc12): HID's fused attitude (HidDirectionState, row-major), as given
  bool connected = true;
};

constexpr uint8_t kGapBefore = 1;  // samples were lost (or skipped) just before this one

struct Sample {
  float accel[3] = {};  // m/s^2
  float gyro[3] = {};   // rad/s
  float dir[9] = {};    // aleks2 (V1-rc12): HID's attitude matrix, as given (row-major)
  uint64_t samplingNumber = 0;
  uint32_t deltaUs = 0;  // microseconds this sample covers, always within kMinUs..kMaxUs
  uint8_t flags = 0;
};

constexpr uint32_t kNominalUs = 5000;
constexpr uint32_t kMinUs = 1000;
constexpr uint32_t kMaxUs = 40000;
constexpr float kGToMetresPerSecSq = 9.80665f;
constexpr float kRevToRad = 6.28318530717958647692f;

struct TimingStats {
  uint64_t rawMin = UINT64_MAX;
  uint64_t rawMax = 0;
  uint32_t asNanoseconds = 0;  // samples whose delta_time read as a period in ns
  uint32_t asMicroseconds = 0;
  uint32_t fallback = 0;  // samples timed from the sampling number instead
  uint32_t samples = 0;
};

struct Cursor {
  uint64_t last = 0;
  bool have = false;
  uint64_t dropped = 0;  // states lost, cumulative
  uint32_t gaps = 0;     // events: an overflow, a counter restart or a disconnected state
  bool pendingGap = false;  // a gap with no sample after it yet: the next sample carries the flag
  TimingStats timing;
};

inline uint32_t period_us(uint64_t raw, uint64_t step, TimingStats* stats) {
  if (stats != nullptr) {
    stats->rawMin = std::min(stats->rawMin, raw);
    stats->rawMax = std::max(stats->rawMax, raw);
    ++stats->samples;
  }
  if (raw >= uint64_t{kMinUs} * 1000 && raw <= uint64_t{kMaxUs} * 1000) {
    if (stats != nullptr) ++stats->asNanoseconds;
    return static_cast<uint32_t>(raw / 1000);
  }
  if (raw >= kMinUs && raw <= kMaxUs) {
    if (stats != nullptr) ++stats->asMicroseconds;
    return static_cast<uint32_t>(raw);
  }
  if (stats != nullptr) ++stats->fallback;
  const uint64_t span = std::max<uint64_t>(step, 1) * kNominalUs;
  return static_cast<uint32_t>(std::clamp<uint64_t>(span, kMinUs, kMaxUs));
}

// `states` is newest first (states[0]); returns the number of samples written to `out`, oldest first.
inline int ingest(Cursor& c, const RawState* states, int got, Sample* out, int maxOut) {
  if (states == nullptr || out == nullptr || got <= 0 || maxOut <= 0) {
    return 0;
  }
  bool carryGap = c.pendingGap;
  c.pendingGap = false;
  if (c.have && states[0].samplingNumber < c.last) {  // the counter restarted: a new controller session
    c.have = false;
    ++c.gaps;
    carryGap = true;
  }
  if (!c.have) {
    // The queue is history when we have not read before: start from the newest state, hand out none of the old ones.
    c.last = states[0].samplingNumber;
    c.have = true;
    c.pendingGap = carryGap;
    return 0;
  }
  int fresh = 0;
  while (fresh < got && states[fresh].samplingNumber > c.last) {
    ++fresh;
  }
  if (fresh == 0) {
    return 0;
  }
  int take = std::min(fresh, maxOut);
  if (fresh > take) {  // the caller could not take them all: the oldest are dropped
    c.dropped += static_cast<uint64_t>(fresh - take);
    ++c.gaps;
    carryGap = true;
  }
  int n = 0;
  uint64_t prev = c.last;
  if (take < fresh) {
    prev = states[take].samplingNumber;
  }
  for (int i = take - 1; i >= 0; --i) {
    const RawState& s = states[i];
    if (s.samplingNumber > prev + 1) {  // numbers missing between two states: overwritten before we read them
      c.dropped += s.samplingNumber - prev - 1;
      ++c.gaps;
      carryGap = true;
    }
    const uint64_t step = s.samplingNumber - prev;
    prev = s.samplingNumber;
    if (!s.connected) {
      ++c.gaps;
      carryGap = true;
      continue;
    }
    Sample& o = out[n++];
    for (int k = 0; k < 3; ++k) {
      o.accel[k] = s.accel[k] * kGToMetresPerSecSq;
      o.gyro[k] = s.gyro[k] * kRevToRad;
    }
    for (int k = 0; k < 9; ++k) {
      o.dir[k] = s.dir[k];
    }
    o.samplingNumber = s.samplingNumber;
    o.deltaUs = period_us(s.deltaTime, step, &c.timing);
    o.flags = carryGap ? kGapBefore : 0;
    carryGap = false;
  }
  c.last = states[0].samplingNumber;
  c.pendingGap = carryGap;  // a skipped trailing state has no sample to mark yet
  return n;
}

}  // namespace aurora::device::motion
