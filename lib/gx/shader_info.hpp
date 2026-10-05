#pragma once

#include "gx.hpp"

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

namespace testing {
// Host tests only: clears the tracker and the mismatch latch, which otherwise lasts
// for the whole process by design.
void rearm_uniform_reuse() noexcept;
} // namespace testing
u8 color_channel(GXChannelID id) noexcept;
}; // namespace aurora::gx
