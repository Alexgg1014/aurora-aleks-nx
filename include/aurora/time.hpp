#pragma once

#include <chrono>
#include <cstdint>

namespace aurora::time {

inline constexpr float kMaximumTimeScale = 16.0f;

class native_clock {
public:
  using rep = int64_t;
  using period = std::nano;
  using duration = std::chrono::duration<rep, period>;
  using time_point = std::chrono::time_point<native_clock>;

  static constexpr bool is_steady = true;

  static time_point now() noexcept;
};

class game_clock {
public:
  using rep = int64_t;
  using period = std::nano;
  using duration = std::chrono::duration<rep, period>;
  using time_point = std::chrono::time_point<game_clock>;

  static constexpr bool is_steady = false;

  static time_point now() noexcept;
};

using wall_clock = std::chrono::system_clock;

/** Sets the clock timescale. Default 1.0f. 0.0f is paused. Range 0.0f-kMaximumTimeScale. */
void set_scale(float scale) noexcept;

float scale() noexcept;

/** aleks2 (Switch 1.0.1): leaves out of game_clock a span the app spent suspended (the console asleep), which the
 *  platform only learns about after it ends. The span is measured back from now and never reaches before the clock's
 *  last rebase, and the clock never goes below a time it already handed out. Returns how much game time was actually
 *  left out. Wall-clock time (OSGetSystemTime, wall_clock) is not affected. */
std::chrono::nanoseconds skip_suspended(std::chrono::nanoseconds span) noexcept;

} // namespace aurora::time
