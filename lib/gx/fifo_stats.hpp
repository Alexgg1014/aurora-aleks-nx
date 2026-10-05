#pragma once

// aleks2 (H35): cumulative FIFO telemetry. Read with aurora_gx_fifo_stats(); never
// reset, so a reader takes two readings and subtracts. Only the standard library, so
// the host tests can use it on its own.
//
// Each group is written by one thread: Worker by whichever thread runs the command
// processor (the FIFO worker in Thread mode), Producer by the game thread. The groups
// sit on their own cache lines so the two writers do not share one. Counters are
// relaxed atomics: they order nothing, and nothing reads them to decide anything.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace aurora::gx::fifo {

enum class DrainSite : uint8_t {
  DrawDone = 0,
  EndFrame = 1,
  Companion = 2,
  Explicit = 3,
  AuxPass = 4,
  Other = 5,
};
inline constexpr size_t kDrainSiteCount = 6;
// Used only as the "no tag" value of the caller-side tag.
inline constexpr DrainSite kUntagged = static_cast<DrainSite>(kDrainSiteCount);

namespace stats {

struct DrainCounters {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> waits{0};
  std::atomic<uint64_t> waitNs{0};
};

struct alignas(64) WorkerGroup {
  std::atomic<uint64_t> busyNs{0};
  std::atomic<uint64_t> processedBytes{0};
  std::atomic<uint64_t> processCalls{0};
  std::atomic<uint64_t> sleeps{0};
};

struct alignas(64) ProducerGroup {
  std::atomic<uint64_t> publishes{0};
  std::atomic<uint64_t> wakeRequests{0};
  DrainCounters total;
  std::array<DrainCounters, kDrainSiteCount> sites;
};

struct Counters {
  WorkerGroup worker;
  ProducerGroup producer;
};

inline void add(std::atomic<uint64_t>& counter, uint64_t value) noexcept {
  counter.fetch_add(value, std::memory_order_relaxed);
}

inline uint64_t get(const std::atomic<uint64_t>& counter) noexcept {
  return counter.load(std::memory_order_relaxed);
}

// A drain request from `site`. `waited` says whether the caller had to wait (or do the
// translation itself); `waitNs` is then how long.
inline void record_drain(ProducerGroup& producer, DrainSite site, bool waited, uint64_t waitNs) noexcept {
  const auto index = static_cast<size_t>(site);
  auto& bySite = producer.sites[index < kDrainSiteCount ? index : static_cast<size_t>(DrainSite::Other)];
  add(producer.total.calls, 1);
  add(bySite.calls, 1);
  if (waited) {
    add(producer.total.waits, 1);
    add(bySite.waits, 1);
    add(producer.total.waitNs, waitNs);
    add(bySite.waitNs, waitNs);
  }
}

inline void record_process(WorkerGroup& worker, uint64_t bytes, uint64_t busyNs) noexcept {
  add(worker.processCalls, 1);
  add(worker.processedBytes, bytes);
  add(worker.busyNs, busyNs);
}

// The site a drain counts under: the caller's tag if there is one, else the site the
// drain itself names.
constexpr DrainSite resolve_site(DrainSite own, DrainSite tag) noexcept {
  return static_cast<size_t>(tag) < kDrainSiteCount ? tag : own;
}

// Difference of two cumulative readings. Unsigned, so a counter that wrapped between
// the readings still gives the right window.
constexpr uint64_t delta(uint64_t before, uint64_t after) noexcept { return after - before; }

} // namespace stats

stats::Counters& counters() noexcept;
// The tag drains on the game thread are counted under; kUntagged for none.
DrainSite set_drain_tag(DrainSite tag) noexcept;

} // namespace aurora::gx::fifo
