#pragma once

#include "gx.hpp"

#include <atomic>
#include <cstdint>

namespace aurora::gx {
ShaderInfo build_shader_info(const ShaderConfig& config) noexcept;
// Fills and pushes the uniform block for the current GX state (the H34 path).
gfx::Range build_uniform(const ShaderInfo& info) noexcept;
// aleks2 (H35): the uniform block for a draw whose uniform is dirty, following the
// reuse mode (aurora_gx_set_uniform_reuse_mode). `pipelineOnly` is true when the only
// reason it is dirty is a pipeline change (no state write, no bind group change);
// `current` is the range the draw cache holds. In Off this is build_uniform.
gfx::Range rebuild_uniform(const ShaderInfo& info, bool pipelineOnly, gfx::Range current) noexcept;
// The uniform buffer starts over (new frame): no previous range may be reused.
void forget_last_uniform() noexcept;
// The draw cache dropped its uniform range.
void invalidate_uniform_reuse() noexcept;

// aleks2 (FIFO-opt): AURORA_FIFO_FAST_* (command_processor.cpp) and their counters.
// The FIFO thread is the only writer of the counters.
struct FifoFastCounters {
  std::atomic<uint64_t> bindMemoHits{0};
  std::atomic<uint64_t> bindMemoMisses{0};
  std::atomic<uint64_t> uniformInPlaceBlocks{0};
  std::atomic<uint64_t> uniformInPlaceRepeats{0};
};
extern std::atomic<uint32_t> g_fifoFast;
extern FifoFastCounters g_fifoFastStats;

namespace testing {
// Host tests only: clears the tracker and the mismatch latch, which otherwise lasts
// for the whole process by design.
void rearm_uniform_reuse() noexcept;
} // namespace testing
u8 color_channel(GXChannelID id) noexcept;
}; // namespace aurora::gx
