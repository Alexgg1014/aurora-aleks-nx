// aleks2 (H35): FIFO telemetry. The counters only measure: these tests check that
// they accumulate, give correct windows by subtraction, classify each drain under the
// right site, and that tagging changes nothing about the drain itself.

#include "gx_test_common.hpp"
#include "gx/fifo_stats.hpp"

#include <aurora/tuning.h>

#include <atomic>
#include <chrono>
#include <limits>
#include <thread>

namespace fifo = aurora::gx::fifo;
namespace stats = aurora::gx::fifo::stats;
using fifo::DrainSite;

namespace {
AuroraFifoStats read_stats() {
  AuroraFifoStats out{};
  aurora_gx_fifo_stats(&out);
  return out;
}

const AuroraFifoDrainStats& site(const AuroraFifoStats& s, AuroraFifoDrainSite which) {
  return s.drain_sites[static_cast<size_t>(which)];
}

uint64_t calls_delta(const AuroraFifoStats& a, const AuroraFifoStats& b, AuroraFifoDrainSite which) {
  return stats::delta(site(a, which).calls, site(b, which).calls);
}

std::atomic<bool> sHoldEntered{false};
std::atomic<bool> sHoldRelease{false};
void holding_draw_done_callback() {
  sHoldEntered.store(true, std::memory_order_release);
  while (!sHoldRelease.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

bool wait_true(const std::atomic<bool>& flag) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (!flag.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

void emit_triangle() {
  GXClearVtxDesc();
  GXSetVtxDesc(GX_VA_POS, GX_DIRECT);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_U8, 0);
  GXBegin(GX_TRIANGLES, GX_VTXFMT0, 3);
  GXPosition3u8(0, 1, 2);
  GXPosition3u8(3, 4, 5);
  GXPosition3u8(6, 7, 8);
  GXEnd();
}
} // namespace

// --- The counter block on its own -------------------------------------------------

TEST(FifoStatsCounters, AccumulateBySiteAndTotal) {
  stats::ProducerGroup producer;
  stats::record_drain(producer, DrainSite::DrawDone, true, 1500);
  stats::record_drain(producer, DrainSite::DrawDone, true, 500);
  stats::record_drain(producer, DrainSite::EndFrame, false, 0);
  stats::record_drain(producer, DrainSite::Companion, true, 7000);
  EXPECT_EQ(stats::get(producer.total.calls), 4u);
  EXPECT_EQ(stats::get(producer.total.waits), 3u);
  EXPECT_EQ(stats::get(producer.total.waitNs), 9000u);
  const auto& drawDone = producer.sites[static_cast<size_t>(DrainSite::DrawDone)];
  EXPECT_EQ(stats::get(drawDone.calls), 2u);
  EXPECT_EQ(stats::get(drawDone.waitNs), 2000u);
  const auto& endFrame = producer.sites[static_cast<size_t>(DrainSite::EndFrame)];
  EXPECT_EQ(stats::get(endFrame.calls), 1u);
  EXPECT_EQ(stats::get(endFrame.waits), 0u);
  EXPECT_EQ(stats::get(endFrame.waitNs), 0u);
  // A wait reported as "not waited" never adds time.
  stats::record_drain(producer, DrainSite::Other, false, 99999);
  EXPECT_EQ(stats::get(producer.total.waitNs), 9000u);
}

TEST(FifoStatsCounters, OutOfRangeSiteCountsAsOther) {
  stats::ProducerGroup producer;
  stats::record_drain(producer, static_cast<DrainSite>(200), true, 10);
  EXPECT_EQ(stats::get(producer.sites[static_cast<size_t>(DrainSite::Other)].calls), 1u);
  EXPECT_EQ(stats::get(producer.total.calls), 1u);
}

TEST(FifoStatsCounters, TagOverridesOwnSite) {
  EXPECT_EQ(stats::resolve_site(DrainSite::AuxPass, fifo::kUntagged), DrainSite::AuxPass);
  EXPECT_EQ(stats::resolve_site(DrainSite::AuxPass, DrainSite::Companion), DrainSite::Companion);
  EXPECT_EQ(stats::resolve_site(DrainSite::DrawDone, static_cast<DrainSite>(77)), DrainSite::DrawDone);
}

TEST(FifoStatsCounters, DeltaIsCorrectAcrossWrap) {
  constexpr uint64_t max = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(stats::delta(100, 250), 150u);
  EXPECT_EQ(stats::delta(max - 4, 5), 10u);
  std::atomic<uint64_t> counter{max - 1};
  const uint64_t before = stats::get(counter);
  stats::add(counter, 3);
  EXPECT_EQ(stats::delta(before, stats::get(counter)), 3u);
}

TEST(FifoStatsCounters, ProcessAccumulates) {
  stats::WorkerGroup worker;
  stats::record_process(worker, 64, 1000);
  stats::record_process(worker, 32, 250);
  EXPECT_EQ(stats::get(worker.processCalls), 2u);
  EXPECT_EQ(stats::get(worker.processedBytes), 96u);
  EXPECT_EQ(stats::get(worker.busyNs), 1250u);
}

TEST(FifoStatsCounters, WriterGroupsDoNotShareACacheLine) {
  static_assert(alignof(stats::WorkerGroup) >= 64);
  static_assert(alignof(stats::ProducerGroup) >= 64);
  stats::Counters counters;
  const auto worker = reinterpret_cast<uintptr_t>(&counters.worker);
  const auto producer = reinterpret_cast<uintptr_t>(&counters.producer);
  EXPECT_GE(producer - worker, sizeof(stats::WorkerGroup));
  EXPECT_EQ(producer % 64, 0u);
  EXPECT_EQ(worker % 64, 0u);
}

// --- The real FIFO (worker-thread mode, the only one compiled in) -----------------

TEST_F(GXFifoTest, StatsAreCumulativeAndWindowsSubtract) {
  ASSERT_EQ(fifo::processing_mode(), fifo::ProcessingMode::Thread);
  fifo::init();
  fifo::begin_frame();
  const AuroraFifoStats start = read_stats();
  EXPECT_EQ(start.processing_mode, 2u);

  emit_triangle();
  const uint32_t buffered = fifo::get_buffer_size();
  ASSERT_GT(buffered, 0u);
  GXDrawDone();
  const AuroraFifoStats afterFirst = read_stats();
  // Everything buffered plus GXDrawDone's own register write got translated.
  EXPECT_GE(stats::delta(start.processed_bytes, afterFirst.processed_bytes), buffered);
  EXPECT_GE(stats::delta(start.process_calls, afterFirst.process_calls), 1u);
  EXPECT_GE(stats::delta(start.publishes, afterFirst.publishes), 1u);
  EXPECT_GE(stats::delta(start.wake_requests, afterFirst.wake_requests), 1u);
  EXPECT_EQ(calls_delta(start, afterFirst, AURORA_FIFO_DRAIN_DRAW_DONE), 1u);
  EXPECT_EQ(stats::delta(start.drains.calls, afterFirst.drains.calls), 1u);

  emit_triangle();
  GXDrawDone();
  const AuroraFifoStats afterSecond = read_stats();
  // Never reset: the second window adds to the first.
  EXPECT_GE(afterSecond.processed_bytes, afterFirst.processed_bytes);
  EXPECT_EQ(calls_delta(start, afterSecond, AURORA_FIFO_DRAIN_DRAW_DONE), 2u);
  EXPECT_EQ(calls_delta(afterFirst, afterSecond, AURORA_FIFO_DRAIN_DRAW_DONE), 1u);
  // Window sums: the total equals the sum over sites.
  uint64_t sumCalls = 0;
  uint64_t sumWaitNs = 0;
  for (size_t i = 0; i < AURORA_FIFO_DRAIN_SITE_COUNT; ++i) {
    sumCalls += afterSecond.drain_sites[i].calls - start.drain_sites[i].calls;
    sumWaitNs += afterSecond.drain_sites[i].wait_ns - start.drain_sites[i].wait_ns;
  }
  EXPECT_EQ(sumCalls, stats::delta(start.drains.calls, afterSecond.drains.calls));
  EXPECT_EQ(sumWaitNs, stats::delta(start.drains.wait_ns, afterSecond.drains.wait_ns));
  fifo::end_frame();
}

TEST_F(GXFifoTest, EmptyDrainCountsACallButNoWait) {
  fifo::init();
  fifo::begin_frame();
  ASSERT_EQ(fifo::get_buffer_size(), 0u);
  const AuroraFifoStats before = read_stats();
  fifo::drain(DrainSite::EndFrame);
  const AuroraFifoStats after = read_stats();
  EXPECT_EQ(calls_delta(before, after, AURORA_FIFO_DRAIN_END_FRAME), 1u);
  EXPECT_EQ(site(after, AURORA_FIFO_DRAIN_END_FRAME).waits, site(before, AURORA_FIFO_DRAIN_END_FRAME).waits);
  EXPECT_EQ(after.drains.wait_ns, before.drains.wait_ns);
  EXPECT_EQ(after.publishes, before.publishes);
  fifo::end_frame();
}

TEST_F(GXFifoTest, DrainWaitIsMeasuredWhileTheWorkerIsBusy) {
  sHoldEntered.store(false, std::memory_order_relaxed);
  sHoldRelease.store(false, std::memory_order_relaxed);
  fifo::init();
  fifo::begin_frame();
  GXSetDrawDoneCallback(holding_draw_done_callback);
  GXSetDrawDone(); // published, not drained: the worker blocks in the callback
  ASSERT_TRUE(wait_true(sHoldEntered));
  emit_triangle();

  const AuroraFifoStats before = read_stats();
  std::thread releaser{[] {
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    sHoldRelease.store(true, std::memory_order_release);
  }};
  GXDrawDone();
  releaser.join();
  const AuroraFifoStats after = read_stats();

  const auto& b = site(before, AURORA_FIFO_DRAIN_DRAW_DONE);
  const auto& a = site(after, AURORA_FIFO_DRAIN_DRAW_DONE);
  EXPECT_EQ(a.calls - b.calls, 1u);
  EXPECT_EQ(a.waits - b.waits, 1u);
  // The game thread was held for at least most of the 20 ms the callback slept.
  EXPECT_GE(a.wait_ns - b.wait_ns, 10'000'000u);
  // busy_ns is command translation only. The draw-done callback runs after process()
  // returns, so the 20 ms it held the worker is wait for the game thread, not busy.
  EXPECT_LT(after.busy_ns - before.busy_ns, a.wait_ns - b.wait_ns);
  GXSetDrawDoneCallback(nullptr);
  fifo::end_frame();
}

TEST_F(GXFifoTest, DrainSitesAreClassified) {
  fifo::init();
  fifo::begin_frame();
  const AuroraFifoStats start = read_stats();

  emit_triangle();
  GXDrawDone();
  emit_triangle();
  AuroraGXSync();
  emit_triangle();
  fifo::drain(DrainSite::AuxPass);
  emit_triangle();
  fifo::drain();
  emit_triangle();
  fifo::drain(DrainSite::EndFrame);

  const AuroraFifoStats end = read_stats();
  EXPECT_EQ(calls_delta(start, end, AURORA_FIFO_DRAIN_DRAW_DONE), 1u);
  EXPECT_EQ(calls_delta(start, end, AURORA_FIFO_DRAIN_EXPLICIT), 1u);
  EXPECT_EQ(calls_delta(start, end, AURORA_FIFO_DRAIN_AUX_PASS), 1u);
  EXPECT_EQ(calls_delta(start, end, AURORA_FIFO_DRAIN_OTHER), 1u);
  EXPECT_EQ(calls_delta(start, end, AURORA_FIFO_DRAIN_END_FRAME), 1u);
  EXPECT_EQ(calls_delta(start, end, AURORA_FIFO_DRAIN_COMPANION), 0u);
  fifo::end_frame();
}

TEST_F(GXFifoTest, CompanionTagWrapsEveryDrainOfTheCapture) {
  fifo::init();
  fifo::begin_frame();
  const AuroraFifoStats start = read_stats();

  // What dualscreen.cpp does around its capture: create_pass, drain_fifo and
  // resolve_pass each drain; all of them belong to the companion.
  const AuroraFifoDrainSite previous = aurora_gx_fifo_set_drain_tag(AURORA_FIFO_DRAIN_COMPANION);
  EXPECT_EQ(previous, AURORA_FIFO_DRAIN_UNTAGGED);
  emit_triangle();
  fifo::drain(DrainSite::AuxPass);
  emit_triangle();
  AuroraGXSync();
  emit_triangle();
  fifo::drain(DrainSite::AuxPass);
  EXPECT_EQ(aurora_gx_fifo_set_drain_tag(previous), AURORA_FIFO_DRAIN_COMPANION);

  // Untagged again: back to the drain's own site.
  emit_triangle();
  GXDrawDone();

  const AuroraFifoStats end = read_stats();
  EXPECT_EQ(calls_delta(start, end, AURORA_FIFO_DRAIN_COMPANION), 3u);
  EXPECT_EQ(calls_delta(start, end, AURORA_FIFO_DRAIN_AUX_PASS), 0u);
  EXPECT_EQ(calls_delta(start, end, AURORA_FIFO_DRAIN_EXPLICIT), 0u);
  EXPECT_EQ(calls_delta(start, end, AURORA_FIFO_DRAIN_DRAW_DONE), 1u);
  // An invalid tag clears it rather than counting under a bogus index.
  aurora_gx_fifo_set_drain_tag(static_cast<AuroraFifoDrainSite>(42));
  EXPECT_EQ(aurora_gx_fifo_set_drain_tag(AURORA_FIFO_DRAIN_UNTAGGED), AURORA_FIFO_DRAIN_UNTAGGED);
  fifo::end_frame();
}

TEST_F(GXFifoTest, TaggingDoesNotChangeWhatTheDrainDoes) {
  fifo::init();
  fifo::begin_frame();
  const std::array<u8, 5> bpWrite{GX_LOAD_BP_REG, 0x41, 0x0A, 0x0B, 0x0C};
  fifo::write_data(bpWrite.data(), bpWrite.size());
  const AuroraFifoDrainSite previous = aurora_gx_fifo_set_drain_tag(AURORA_FIFO_DRAIN_COMPANION);
  fifo::drain(DrainSite::AuxPass);
  aurora_gx_fifo_set_drain_tag(previous);
  // Fully processed and the buffer consumed, exactly as an untagged drain.
  EXPECT_EQ(aurora::gx::g_gxState.bpRegCache[0x41], 0x410A0B0Cu);
  EXPECT_EQ(fifo::get_buffer_size(), 0u);
  fifo::end_frame();
}

TEST(FifoStatsApi, NullOutputIsIgnored) { aurora_gx_fifo_stats(nullptr); }
