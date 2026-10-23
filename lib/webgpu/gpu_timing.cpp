#include "gpu_timing.hpp"

#include "../internal.hpp"
#include "gpu.hpp"

#include <aurora/tuning.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>

namespace aurora::webgpu::gpu_timing {
namespace {
Module Log("aurora::webgpu::gpu_timing");

constexpr uint32_t RingDepth = 4;
// ResolveQuerySet wants a 256-byte aligned destination offset: one stride per slot.
constexpr uint64_t ResolveStride = 256;
constexpr uint64_t SlotBytes = 2 * sizeof(uint64_t);
// A frame that took longer than this on the GPU is a suspend or a broken pair, not a frame.
constexpr uint64_t SaneFrameTicks = UINT64_C(2'000'000'000);
// Begin-to-begin pairs needed before the tick period estimate is trusted.
constexpr uint64_t CalibrationPairs = 120;

enum class SlotState : uint8_t { Free, Recording, InFlight, Mapped, Failed };

struct Slot {
  wgpu::Buffer readback;
  int64_t cpuBeginNs = 0;
  std::atomic<SlotState> state{SlotState::Free};
};

bool g_supported = false;
std::atomic<bool> g_requested{false};
wgpu::QuerySet g_querySet;
wgpu::Buffer g_resolveBuffer;
wgpu::Texture g_dummyTexture;
wgpu::TextureView g_dummyView;
std::array<Slot, RingDepth> g_slots;
uint32_t g_recordSlot = 0;
uint32_t g_emitSlot = 0;
bool g_frameActive = false;
bool g_framePending = false;

// Calibration of the tick period: GPU begin-to-begin against CPU begin-to-begin.
bool g_havePrev = false;
uint64_t g_prevBeginTicks = 0;
int64_t g_prevCpuNs = 0;
double g_pairCpuNs = 0.0;
double g_pairTicks = 0.0;
uint64_t g_pairs = 0;

// Published to any thread.
std::atomic<uint64_t> g_frames{0};
std::atomic<uint64_t> g_gpuNsTotal{0};
std::atomic<uint64_t> g_lastNs{0};
std::atomic<uint64_t> g_skipped{0};
std::atomic<uint32_t> g_periodPpm{1'000'000}; // ns per tick, x 1e6
std::atomic<bool> g_calibrated{false};

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void empty_pass(const wgpu::CommandEncoder& encoder, uint32_t beginIndex, uint32_t endIndex) {
  const wgpu::PassTimestampWrites writes{
      .querySet = g_querySet,
      .beginningOfPassWriteIndex = beginIndex,
      .endOfPassWriteIndex = endIndex,
  };
  const wgpu::RenderPassColorAttachment attachment{
      .view = g_dummyView,
      .loadOp = wgpu::LoadOp::Load,
      .storeOp = wgpu::StoreOp::Store,
  };
  const wgpu::RenderPassDescriptor desc{
      .label = "GPU timing",
      .colorAttachmentCount = 1,
      .colorAttachments = &attachment,
      .timestampWrites = &writes,
  };
  auto pass = encoder.BeginRenderPass(&desc);
  pass.End();
}

void consume(Slot& slot) {
  const auto* ts = static_cast<const uint64_t*>(slot.readback.GetConstMappedRange(0, SlotBytes));
  if (ts == nullptr) {
    g_skipped.fetch_add(1, std::memory_order_relaxed);
    g_havePrev = false;
    return;
  }
  const uint64_t begin = ts[0];
  const uint64_t end = ts[1];
  if (end <= begin || end - begin > SaneFrameTicks) {
    g_skipped.fetch_add(1, std::memory_order_relaxed);
    g_havePrev = false;
    return;
  }
  // Calibrate on consecutive frames only (a skipped one would double the CPU side).
  if (g_havePrev && begin > g_prevBeginTicks && slot.cpuBeginNs > g_prevCpuNs) {
    g_pairTicks += static_cast<double>(begin - g_prevBeginTicks);
    g_pairCpuNs += static_cast<double>(slot.cpuBeginNs - g_prevCpuNs);
    if (++g_pairs == CalibrationPairs) {
      const double period = g_pairTicks > 0.0 ? g_pairCpuNs / g_pairTicks : 1.0;
      // Vulkan on this GPU counts nanoseconds (period 1); keep exactly 1 unless the
      // estimate is clearly something else, so jitter never skews the numbers.
      const double used = std::fabs(period - 1.0) < 0.05 ? 1.0 : period;
      g_periodPpm.store(static_cast<uint32_t>(std::lround(used * 1e6)), std::memory_order_relaxed);
      g_calibrated.store(true, std::memory_order_relaxed);
      Log.info("GPU timestamp period: estimated {:.4f} ns/tick over {} frames, using {:.4f}", period, g_pairs, used);
    }
  }
  g_havePrev = true;
  g_prevBeginTicks = begin;
  g_prevCpuNs = slot.cpuBeginNs;

  const double period = static_cast<double>(g_periodPpm.load(std::memory_order_relaxed)) / 1e6;
  const auto ns = static_cast<uint64_t>(static_cast<double>(end - begin) * period);
  g_lastNs.store(ns, std::memory_order_relaxed);
  g_gpuNsTotal.fetch_add(ns, std::memory_order_relaxed);
  g_frames.fetch_add(1, std::memory_order_release);
}
} // namespace

void initialize() {
  g_supported = g_device && g_device.HasFeature(wgpu::FeatureName::TimestampQuery);
  if (!g_supported) {
    Log.info("GPU timing unavailable: the device has no TimestampQuery");
    return;
  }
  const wgpu::QuerySetDescriptor querySetDescriptor{
      .label = "GPU timing timestamps",
      .type = wgpu::QueryType::Timestamp,
      .count = RingDepth * 2,
  };
  g_querySet = g_device.CreateQuerySet(&querySetDescriptor);
  const wgpu::BufferDescriptor resolveDescriptor{
      .label = "GPU timing resolve",
      .usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc,
      .size = ResolveStride * RingDepth,
  };
  g_resolveBuffer = g_device.CreateBuffer(&resolveDescriptor);
  const wgpu::BufferDescriptor readbackDescriptor{
      .label = "GPU timing readback",
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = SlotBytes,
  };
  for (auto& slot : g_slots) {
    slot.readback = g_device.CreateBuffer(&readbackDescriptor);
    slot.state = SlotState::Free;
  }
  const wgpu::TextureDescriptor dummyDescriptor{
      .label = "GPU timing target",
      .usage = wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {1, 1, 1},
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  g_dummyTexture = g_device.CreateTexture(&dummyDescriptor);
  g_dummyView = g_dummyTexture.CreateView();
  g_recordSlot = g_emitSlot = 0;
  g_frameActive = g_framePending = false;
  Log.info("GPU timing available (off until requested)");
}

void shutdown() {
  g_querySet = {};
  g_resolveBuffer = {};
  g_dummyView = {};
  g_dummyTexture = {};
  for (auto& slot : g_slots) {
    slot.readback = {};
    slot.state = SlotState::Free;
  }
  g_supported = false;
  g_frameActive = g_framePending = false;
}

void frame_begin(const wgpu::CommandEncoder& encoder) {
  g_frameActive = false;
  if (!g_supported || !g_requested.load(std::memory_order_relaxed)) {
    g_havePrev = false;
    return;
  }
  auto& slot = g_slots[g_recordSlot];
  if (slot.state.load(std::memory_order_acquire) != SlotState::Free) {
    g_skipped.fetch_add(1, std::memory_order_relaxed);
    g_havePrev = false;
    return;
  }
  slot.state.store(SlotState::Recording, std::memory_order_relaxed);
  slot.cpuBeginNs = now_ns();
  g_frameActive = true;
  empty_pass(encoder, g_recordSlot * 2, wgpu::kQuerySetIndexUndefined);
}

void frame_end(const wgpu::CommandEncoder& encoder) {
  if (!g_frameActive) {
    return;
  }
  g_frameActive = false;
  auto& slot = g_slots[g_recordSlot];
  empty_pass(encoder, wgpu::kQuerySetIndexUndefined, g_recordSlot * 2 + 1);
  encoder.ResolveQuerySet(g_querySet, g_recordSlot * 2, 2, g_resolveBuffer, g_recordSlot * ResolveStride);
  encoder.CopyBufferToBuffer(g_resolveBuffer, g_recordSlot * ResolveStride, slot.readback, 0, SlotBytes);
  g_framePending = true;
}

void after_submit() {
  if (!g_supported) {
    return;
  }
  if (g_framePending) {
    g_framePending = false;
    auto& slot = g_slots[g_recordSlot];
    slot.state.store(SlotState::InFlight, std::memory_order_release);
    // AllowSpontaneous: the per-frame ProcessEvents in gfx (frame.cpp) drives it.
    slot.readback.MapAsync(wgpu::MapMode::Read, 0, SlotBytes, wgpu::CallbackMode::AllowSpontaneous,
                           [&slot](wgpu::MapAsyncStatus status, wgpu::StringView) {
                             slot.state.store(status == wgpu::MapAsyncStatus::Success ? SlotState::Mapped
                                                                                       : SlotState::Failed,
                                              std::memory_order_release);
                           });
    g_recordSlot = (g_recordSlot + 1) % RingDepth;
  }
  // In order, so begin-to-begin calibration sees frames as they were submitted.
  while (true) {
    auto& slot = g_slots[g_emitSlot];
    const auto state = slot.state.load(std::memory_order_acquire);
    if (state == SlotState::Mapped) {
      consume(slot);
      slot.readback.Unmap();
    } else if (state == SlotState::Failed) {
      g_skipped.fetch_add(1, std::memory_order_relaxed);
      g_havePrev = false;
    } else {
      break;
    }
    slot.state.store(SlotState::Free, std::memory_order_release);
    g_emitSlot = (g_emitSlot + 1) % RingDepth;
  }
}

bool requested() { return g_requested.load(std::memory_order_relaxed); }

} // namespace aurora::webgpu::gpu_timing

void aurora_gpu_timing_set_enabled(bool enabled) {
  aurora::webgpu::gpu_timing::g_requested.store(enabled, std::memory_order_relaxed);
}

void aurora_gpu_timing_stats(AuroraGpuTimingStats* out) {
  using namespace aurora::webgpu::gpu_timing;
  if (out == nullptr) {
    return;
  }
  std::memset(out, 0, sizeof(*out));
  // aleks2 (V1-rc12): with the deko3d renderer the GPU time is deko3d's own per-frame timestamps (lib/deko); Dawn's
  // device is the Null one there, so these counters never moved and every consumer (frame telemetry, the overlay,
  // the tour's summary and CSV) said "GPU time not available".
  AuroraRenderCpuStats render{};
  aurora_gfx_render_cpu_stats(&render);
  if (render.deko3d) {
    out->supported = 1;
    out->enabled = g_requested.load(std::memory_order_relaxed) ? 1 : 0;  // V1-rc16: deko3d writes them only on request
    out->frames = render.gpu_frames;
    out->gpu_ns = render.gpu_ns;
    out->period_ppm = 1000000;
    out->calibrated = 1;
    return;
  }
  out->supported = g_supported ? 1 : 0;
  out->enabled = g_supported && g_requested.load(std::memory_order_relaxed) ? 1 : 0;
  out->frames = g_frames.load(std::memory_order_acquire);
  out->gpu_ns = g_gpuNsTotal.load(std::memory_order_relaxed);
  out->last_ns = g_lastNs.load(std::memory_order_relaxed);
  out->skipped = g_skipped.load(std::memory_order_relaxed);
  out->period_ppm = g_periodPpm.load(std::memory_order_relaxed);
  out->calibrated = g_calibrated.load(std::memory_order_relaxed) ? 1 : 0;
}
