// aleks2 (H35): the render worker's public counters, used from a translation unit other
// than render_worker.cpp, the way lib/gfx/frame.cpp uses them.
//
// 67e405f defined items_run() and work_ns() inside render_worker.cpp's anonymous namespace,
// which made them different, internal functions: the declarations in render_worker.hpp had no
// definition at all. The tests that lived in render_worker.cpp's own reach never noticed, and
// the Switch link of frame.cpp failed with "undefined reference to
// aurora::gfx::render_worker::items_run()". This file references both through the header, so
// that mistake fails this test's link the same way.
#include "../lib/gfx/render_worker.hpp"

#include <atomic>
#include <chrono>
#include <thread>

#include <gtest/gtest.h>

namespace {
using namespace std::chrono_literals;
namespace render_worker = aurora::gfx::render_worker;

TEST(RenderWorkerCounters, LinkFromAnotherUnitAndCountWorkerItems) {
  render_worker::initialize();
  const uint64_t itemsBefore = render_worker::items_run();
  const uint64_t nsBefore = render_worker::work_ns();

  std::atomic_int ran = 0;
  render_worker::enqueue_work([&] {
    std::this_thread::sleep_for(2ms);
    ran.fetch_add(1, std::memory_order_acq_rel);
  });
  render_worker::enqueue_work([&] { ran.fetch_add(1, std::memory_order_acq_rel); });
  render_worker::synchronize();

  EXPECT_EQ(ran.load(std::memory_order_acquire), 2);
  // Two work items, plus whatever synchronize() queued itself to wait on.
  EXPECT_GE(render_worker::items_run() - itemsBefore, 2u);
  // The first item slept 2 ms on the worker; the time is the worker's, not the caller's.
  EXPECT_GE(render_worker::work_ns() - nsBefore, 1'000'000u);
  render_worker::shutdown();
}
} // namespace
