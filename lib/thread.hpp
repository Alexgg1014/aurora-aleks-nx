#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <stop_token>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

namespace aurora::thread {

enum class Priority { Low, Normal, High };
enum class Affinity {
  None,
  SharedCache,
  Logic,
  Render,
  // GX -> WebGPU translation. Its own core on Switch: the game thread blocks
  // on this worker inside gfx::create_pass/resolve_pass, so whatever shares a
  // core with it is taken straight out of the frame.
  Fifo,
  // Shader compilation (NAK). Deliberately NOT the Fifo core.
  Compile,
};

struct Options {
  std::string name;
  Priority priority = Priority::Normal;
  Affinity affinity = Affinity::None;
};

// Applies thread options to the current thread
void set_current(const Options& options) noexcept;

#if defined(__SWITCH__)
// A thread whose stack size is chosen by the caller. std::thread cannot ask for one,
// and the shader compiler Dawn runs on the pipeline thread needs far more stack than
// libnx gives a thread by default.
class LargeStackThread {
public:
  LargeStackThread() noexcept = default;
  LargeStackThread(const LargeStackThread&) = delete;
  LargeStackThread& operator=(const LargeStackThread&) = delete;
  ~LargeStackThread() { join(); }

  // False if the system could not create the thread.
  bool start(size_t stackBytes, std::function<void()> body);
  // Waits for the thread to finish; does nothing if there is none.
  void join() noexcept;

private:
  struct Impl;
  Impl* mImpl = nullptr;
};
#endif

class Thread {
public:
  Thread() noexcept = default;

  template <typename Function>
    requires std::invocable<std::decay_t<Function>&, std::stop_token>
  explicit Thread(Options options, Function&& function)
  : mThread{[options = std::move(options), function = std::forward<Function>(function)](std::stop_token token) mutable {
    set_current(options);
    std::invoke(function, token);
  }} {}

  Thread(Thread&&) noexcept = default;
  Thread& operator=(Thread&&) noexcept = default;
  Thread(const Thread&) = delete;
  Thread& operator=(const Thread&) = delete;
  ~Thread() = default;

  [[nodiscard]] bool joinable() const noexcept { return mThread.joinable(); }
  [[nodiscard]] std::thread::id get_id() const noexcept { return mThread.get_id(); }
  [[nodiscard]] std::stop_token get_stop_token() const noexcept { return mThread.get_stop_token(); }
  bool request_stop() noexcept { return mThread.request_stop(); }
  void join() { mThread.join(); }
  std::jthread::native_handle_type native_handle() { return mThread.native_handle(); }

private:
  std::jthread mThread;
};

} // namespace aurora::thread
