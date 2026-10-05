#include "fifo.hpp"

#include "../thread.hpp"
#include "command_processor.hpp"
#include "fifo_stats.hpp"

#include <aurora/tuning.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>

#include <tracy/Tracy.hpp>

namespace aurora::gx::fifo {
namespace detail {
uint8_t* sBufferData = nullptr;
uint32_t sBufferSize = 0;
uint32_t sBufferCapacity = 0;
bool sInDisplayList = false;
uint8_t* sDlBuffer = nullptr;
uint32_t sDlSize = 0;
uint32_t sDlWritePos = 0;
} // namespace detail

namespace {
constexpr Module Log{"aurora::gx::fifo"};
constexpr auto kProcessingMode = ProcessingMode::Thread;
constexpr uint32_t kDrawBatchSize = 16;

bool sFrameActive = false;
uint32_t sPendingDraws = 0;
std::atomic<uint64_t> sPublished{0};
std::atomic<uint64_t> sProcessed{0};
uint64_t sStreamBase = 0;
std::mutex sBufferMutex;
std::atomic<uint32_t> sWorkerWake{0};
thread::Thread sWorkerThread;
std::atomic<DrawDoneCallback> sDrawDoneCallback{nullptr};
// aleks2 (H35): telemetry only; see fifo_stats.hpp.
stats::Counters sStats;
DrainSite sDrainTag = kUntagged; // game thread only

uint64_t now_ns() noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void dispatch_draw_done() noexcept {
  if (const auto callback = sDrawDoneCallback.load(std::memory_order_acquire); callback != nullptr) {
    callback();
  }
}

void wake_worker() noexcept {
  stats::add(sStats.producer.wakeRequests, 1);
  sWorkerWake.fetch_add(1, std::memory_order_release);
  sWorkerWake.notify_all();
}

void process_to(uint64_t target, std::memory_order order) noexcept {
  uint64_t processed = sProcessed.load(std::memory_order_relaxed);
  while (processed < target) {
    ProcessResult result{};
    const uint64_t busyStart = now_ns();
    {
      std::lock_guard lock{sBufferMutex};
      AURORA_ASSERT(processed >= sStreamBase && target <= sStreamBase + detail::sBufferSize,
                    "FIFO processing range [{}, {}) is outside buffered range [{}, {})", processed, target, sStreamBase,
                    sStreamBase + detail::sBufferSize);
      const auto start = static_cast<uint32_t>(processed - sStreamBase);
      const auto size = static_cast<uint32_t>(target - processed);
      result = process(detail::sBufferData + start, size);
    }
    stats::record_process(sStats.worker, result.bytesProcessed, now_ns() - busyStart);
    AURORA_ASSERT(result.bytesProcessed > 0 && result.bytesProcessed <= target - processed,
                  "FIFO processor made invalid progress: processed {} of {} remaining bytes", result.bytesProcessed,
                  target - processed);
    if (result.drawDone) {
      dispatch_draw_done();
    }
    processed += result.bytesProcessed;
    sProcessed.store(processed, order);
    sProcessed.notify_all();
  }
}

void worker_main(std::stop_token token) noexcept {
  std::stop_callback wakeOnStop{token, wake_worker};
  while (true) {
    const uint32_t event = sWorkerWake.load(std::memory_order_acquire);
    const uint64_t processed = sProcessed.load(std::memory_order_relaxed);
    const uint64_t published = sPublished.load(std::memory_order_acquire);
    if (published != processed) {
      process_to(published, std::memory_order_release);
      continue;
    }

    if (token.stop_requested()) {
      break;
    }
    stats::add(sStats.worker.sleeps, 1);
    sWorkerWake.wait(event, std::memory_order_acquire);
  }
}

void start_worker() {
  if (kProcessingMode != ProcessingMode::Thread || sWorkerThread.joinable()) {
    return;
  }
  sWorkerThread = thread::Thread{{
                                     .name = "Aurora FIFO processor",
#if defined(__SWITCH__)
                                     .affinity = thread::Affinity::Fifo,
#else
                                     .affinity = thread::Affinity::SharedCache,
#endif
                                 },
                                 worker_main};
}

void stop_worker() {
  if (!sWorkerThread.joinable()) {
    return;
  }
  sWorkerThread.request_stop();
  sWorkerThread.join();
}
} // namespace

ProcessingMode processing_mode() noexcept { return kProcessingMode; }

void init() {
  stop_worker();

  constexpr uint32_t initialCapacity = 64 * 1024;
  free(detail::sBufferData);
  detail::sBufferData = static_cast<uint8_t*>(malloc(initialCapacity));
  AURORA_ASSERT(detail::sBufferData != nullptr, "fifo::init: failed to allocate {} bytes", initialCapacity);
  detail::sBufferSize = 0;
  detail::sBufferCapacity = initialCapacity;
  detail::sInDisplayList = false;
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;

  sFrameActive = false;
  sPendingDraws = 0;
  sStreamBase = 0;
  sPublished.store(0, std::memory_order_relaxed);
  sProcessed.store(0, std::memory_order_relaxed);
  sWorkerWake.store(0, std::memory_order_relaxed);

  start_worker();
}

void shutdown() { stop_worker(); }

void begin_frame() noexcept { sFrameActive = true; }

void end_frame() noexcept {
  sFrameActive = false;
  clear_draw_cache(); // command_processor
}

void write_data_grow(const void* data, uint32_t length) {
  const uint64_t needed64 = static_cast<uint64_t>(detail::sBufferSize) + length;
  AURORA_ASSERT(needed64 <= std::numeric_limits<uint32_t>::max(), "fifo::write_data: buffer size overflow");
  const auto needed = static_cast<uint32_t>(needed64);
  const auto doubledCapacity = static_cast<uint64_t>(detail::sBufferCapacity) * 2;
  const auto newCapacity = static_cast<uint32_t>(
      std::min<uint64_t>(std::max(doubledCapacity, needed64), std::numeric_limits<uint32_t>::max()));
  const auto grow = [newCapacity] {
    auto* resized = static_cast<uint8_t*>(realloc(detail::sBufferData, newCapacity));
    AURORA_ASSERT(resized != nullptr, "fifo::write_data: failed to allocate {} bytes", newCapacity);
    detail::sBufferData = resized;
  };
  if (sWorkerThread.joinable()) {
    std::lock_guard lock{sBufferMutex};
    grow();
  } else {
    grow();
  }
  std::memcpy(detail::sBufferData + detail::sBufferSize, data, length);
  detail::sBufferSize = needed;
  detail::sBufferCapacity = newCapacity;
}

void publish() noexcept {
  if (!sFrameActive || kProcessingMode == ProcessingMode::Drain || detail::sInDisplayList) {
    return;
  }

  const uint64_t target = sStreamBase + detail::sBufferSize;
  if (target > sPublished.load(std::memory_order_relaxed)) {
    sPendingDraws = 0;
    stats::add(sStats.producer.publishes, 1);
    sPublished.store(target, std::memory_order_release);
    if (kProcessingMode == ProcessingMode::Thread) {
      wake_worker();
    } else {
      process_to(target, std::memory_order_relaxed);
    }
  }
}

DrawDoneCallback set_draw_done_callback(DrawDoneCallback callback) noexcept {
  return sDrawDoneCallback.exchange(callback, std::memory_order_acq_rel);
}

void finish_draw() noexcept {
  if (!sFrameActive || kProcessingMode == ProcessingMode::Drain || detail::sInDisplayList) {
    return;
  }
  if (++sPendingDraws >= kDrawBatchSize) {
    publish();
  }
}

void patch_u32(uint32_t offset, uint32_t val) {
  AURORA_ASSERT(!detail::sInDisplayList && offset <= detail::sBufferSize &&
                    sizeof(uint32_t) <= detail::sBufferSize - offset,
                "fifo::patch_u32: invalid patch offset {} (buffer size {})", offset, detail::sBufferSize);
  AURORA_ASSERT(sStreamBase + offset >= sPublished.load(std::memory_order_relaxed),
                "fifo::patch_u32: offset {} is below the published watermark", offset);
  const auto out = bswap(val);
  std::memcpy(detail::sBufferData + offset, &out, sizeof(out));
}

void begin_display_list(uint8_t* buf, uint32_t size) {
  detail::sInDisplayList = true;
  detail::sDlBuffer = buf;
  detail::sDlSize = size;
  detail::sDlWritePos = 0;
}

uint32_t end_display_list() {
  detail::sInDisplayList = false;
  const uint32_t bytesWritten = detail::sDlWritePos;
  const uint32_t padded = (bytesWritten + 31) & ~31u;
  while (detail::sDlWritePos < padded && detail::sDlWritePos < detail::sDlSize) {
    detail::sDlBuffer[detail::sDlWritePos++] = 0;
  }
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;
  return padded;
}

bool in_display_list() { return detail::sInDisplayList; }

void drain(DrainSite site) {
  const DrainSite countedAs = stats::resolve_site(site, sDrainTag);
  if (detail::sBufferSize == 0) {
    stats::record_drain(sStats.producer, countedAs, false, 0);
    return;
  }

  ZoneScoped;
  const uint64_t target = sStreamBase + detail::sBufferSize;
  if (target > sPublished.load(std::memory_order_relaxed)) {
    stats::add(sStats.producer.publishes, 1);
  }

  bool waited = false;
  uint64_t waitNs = 0;
  switch (kProcessingMode) {
  case ProcessingMode::Drain:
  case ProcessingMode::Inline: {
    // The caller translates the rest itself: that is its wait.
    const uint64_t start = now_ns();
    sPublished.store(target, std::memory_order_relaxed);
    waited = sProcessed.load(std::memory_order_relaxed) < target;
    process_to(target, std::memory_order_relaxed);
    waitNs = waited ? now_ns() - start : 0;
    break;
  }
  case ProcessingMode::Thread: {
    sPublished.store(target, std::memory_order_release);
    wake_worker();

    uint64_t processed = sProcessed.load(std::memory_order_acquire);
    if (processed < target) {
      waited = true;
      const uint64_t start = now_ns();
      do {
        sProcessed.wait(processed, std::memory_order_acquire);
        processed = sProcessed.load(std::memory_order_acquire);
      } while (processed < target);
      waitNs = now_ns() - start;
    }
    break;
  }
  }
  stats::record_drain(sStats.producer, countedAs, waited, waitNs);

  {
    std::lock_guard lock{sBufferMutex};
    sStreamBase = target;
    detail::sBufferSize = 0;
  }
  sPendingDraws = 0;
}

stats::Counters& counters() noexcept { return sStats; }

DrainSite set_drain_tag(DrainSite tag) noexcept {
  const DrainSite previous = sDrainTag;
  sDrainTag = static_cast<size_t>(tag) < kDrainSiteCount ? tag : kUntagged;
  return previous;
}

const uint8_t* get_buffer_data() { return detail::sBufferData; }
uint32_t get_buffer_size() { return detail::sBufferSize; }

void clear_buffer() {
  const uint64_t processed = sProcessed.load(std::memory_order_acquire);
  AURORA_ASSERT(sPublished.load(std::memory_order_acquire) == processed,
                "fifo::clear_buffer: published commands are still pending");
  std::lock_guard lock{sBufferMutex};
  sStreamBase = processed;
  detail::sBufferSize = 0;
  sPendingDraws = 0;
}

} // namespace aurora::gx::fifo

extern "C" AuroraFifoDrainSite aurora_gx_fifo_set_drain_tag(AuroraFifoDrainSite tag) {
  using aurora::gx::fifo::DrainSite;
  const auto value = static_cast<unsigned>(tag);
  const DrainSite next = value < aurora::gx::fifo::kDrainSiteCount ? static_cast<DrainSite>(value)
                                                                   : aurora::gx::fifo::kUntagged;
  return static_cast<AuroraFifoDrainSite>(aurora::gx::fifo::set_drain_tag(next));
}

extern "C" void aurora_gx_fifo_stats(AuroraFifoStats* out) {
  if (out == nullptr) {
    return;
  }
  using aurora::gx::fifo::stats::get;
  const auto& c = aurora::gx::fifo::counters();
  *out = {};
  out->busy_ns = get(c.worker.busyNs);
  out->processed_bytes = get(c.worker.processedBytes);
  out->process_calls = get(c.worker.processCalls);
  out->worker_sleeps = get(c.worker.sleeps);
  out->publishes = get(c.producer.publishes);
  out->wake_requests = get(c.producer.wakeRequests);
  out->drains = {get(c.producer.total.calls), get(c.producer.total.waits), get(c.producer.total.waitNs)};
  static_assert(aurora::gx::fifo::kDrainSiteCount == AURORA_FIFO_DRAIN_SITE_COUNT);
  for (size_t i = 0; i < aurora::gx::fifo::kDrainSiteCount; ++i) {
    const auto& site = c.producer.sites[i];
    out->drain_sites[i] = {get(site.calls), get(site.waits), get(site.waitNs)};
  }
  out->processing_mode = static_cast<uint32_t>(aurora::gx::fifo::processing_mode());
}
