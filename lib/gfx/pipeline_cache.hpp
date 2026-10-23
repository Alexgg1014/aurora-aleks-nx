#pragma once

#include "types.hpp"

namespace aurora::gfx::clear {
struct PipelineConfig;
} // namespace aurora::gfx::clear

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::rmlui {
struct PipelineConfig;
} // namespace aurora::rmlui

namespace aurora::gfx {

enum class ShaderType : uint8_t {
  Clear = 0,
  GX = 1,
  Rml = 2,
};

struct CompiledPipeline {
  wgpu::RenderPipeline main;
  wgpu::RenderPipeline prepass;
#if defined(AURORA_GFX_DEKO3D)
  // aleks2 (deko3d P1): the deko3d renderer's pipelines (lib/deko/dk_gfx.hpp); main and prepass stay
  // null with it.
  const void* dkMain = nullptr;
  const void* dkPrepass = nullptr;
#endif

  [[nodiscard]] uint32_t pipeline_count() const {
#if defined(AURORA_GFX_DEKO3D)
    if (dkMain != nullptr) {
      return 1u + static_cast<uint32_t>(dkPrepass != nullptr);
    }
#endif
    return static_cast<uint32_t>(static_cast<bool>(main)) + static_cast<uint32_t>(static_cast<bool>(prepass));
  }
};

void initialize_pipeline_cache();
void shutdown_pipeline_cache();
void begin_pipeline_frame();
void end_pipeline_frame();
void rebuild_pipeline_cache();

// aleks2 (H31): bumped whenever the pipeline tables are initialised, torn down or
// rebuilt. Anything that remembers a PipelineRef across calls (the GX lookup
// memo) is only valid while this number is unchanged.
uint32_t pipeline_generation() noexcept;

PipelineRef find_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout);
PipelineRef find_pipeline(const clear::PipelineConfig& config, const RenderTargetLayout& layout);
PipelineRef find_pipeline(const rmlui::PipelineConfig& config);

bool get_pipeline(PipelineRef ref, CompiledPipeline& pipeline);

} // namespace aurora::gfx
