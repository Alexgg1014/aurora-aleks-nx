#include "pipeline.hpp"

#include "../gfx/encoding.hpp"
#include "../gfx/resources.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "../gfx/resource_cache.hpp"

#include "gx_fmt.hpp"
#include "shader_info.hpp"

#include <tracy/Tracy.hpp>

#include <atomic>
#include <chrono>

#if defined(AURORA_GFX_DEKO3D)
#include "dk_gfx.hpp"
#include "dk_runtime.hpp"
#endif

namespace aurora::gfx {
extern std::atomic<uint64_t> g_gxPipelineNs;  // gfx/pipeline_cache.cpp (V1-rc1)
extern std::atomic<uint32_t> g_gxPipelinesTimed;
} // namespace aurora::gfx

namespace aurora::gx {

#if defined(AURORA_GFX_DEKO3D)
// aleks2 (deko3d P1, after SwitchWaker patch 0014): the same passes as below, built by lib/deko (the
// module's DKSH from the caches, or Tint + the uam worker for a miss). Keep the destination-alpha
// choice in step with create_pipeline's.
static gfx::CompiledPipeline create_pipeline_deko3d(const PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  const auto create = [&](const PipelineOptions& options) {
    const auto start = std::chrono::steady_clock::now();
    const void* built = gfx::dk::build_gx_pipeline(config, layout, options);
    gfx::g_gxPipelineNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                            std::chrono::steady_clock::now() - start)
                                                            .count()),
                                  std::memory_order_relaxed);
    gfx::g_gxPipelinesTimed.fetch_add(1, std::memory_order_relaxed);
    return built;
  };
  PipelineOptions options{
      .colorUpdate = config.colorUpdate,
      .alphaUpdate = config.alphaUpdate,
      .depthUpdate = config.depthUpdate,
  };
  gfx::CompiledPipeline pipeline;
  if (config.alphaUpdate && config.dstAlpha != UINT32_MAX && config.dstAlpha != 0) {
    const auto usesSourceAlpha = [](GXBlendFactor factor) {
      return factor == GX_BL_SRCALPHA || factor == GX_BL_INVSRCALPHA;
    };
    const bool needsSourceAlpha = config.colorUpdate && config.blendMode == GX_BM_BLEND &&
                                  (usesSourceAlpha(config.blendFacSrc) || usesSourceAlpha(config.blendFacDst));
    if (!needsSourceAlpha) {
      options.dstAlphaMode = DstAlphaMode::Replace;
    } else if (gfx::dk::dual_source_blending() && layout.colorAttachmentCount == 1) {
      options.dstAlphaMode = DstAlphaMode::DualSource;
    } else {
      auto alpha = options;
      alpha.dstAlphaMode = DstAlphaMode::Replace;
      alpha.colorUpdate = false;
      alpha.depthUpdate = false;
      pipeline.dkPrepass = create(alpha);
      if (pipeline.dkPrepass == nullptr) {
        return {};
      }
      options.alphaUpdate = false;
    }
  }
  pipeline.dkMain = create(options);
  if (pipeline.dkMain == nullptr) {
    return {};
  }
  return pipeline;
}
#endif

gfx::CompiledPipeline create_pipeline(const PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  ZoneScoped;
#if defined(AURORA_GFX_DEKO3D)
  if (gfx::dk::active()) {
    return create_pipeline_deko3d(config, layout);
  }
#endif
  const auto hash = xxh3_hash(layout.key, xxh3_hash(config, static_cast<HashType>(gfx::ShaderType::GX)));
  const auto create = [&](const PipelineOptions& options, const char* passName) {
    const auto shader = build_shader(config.shaderConfig, layout, options.dstAlphaMode);
    const auto label = fmt::format("GX Pipeline {:x} {}", hash, passName);
    const auto start = std::chrono::steady_clock::now();
    auto built = build_pipeline(config, layout, options, {}, shader, label.c_str());
    gfx::g_gxPipelineNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                            std::chrono::steady_clock::now() - start)
                                                            .count()),
                                  std::memory_order_relaxed);
    gfx::g_gxPipelinesTimed.fetch_add(1, std::memory_order_relaxed);
    return built;
  };

  PipelineOptions options{
      .colorUpdate = config.colorUpdate,
      .alphaUpdate = config.alphaUpdate,
      .depthUpdate = config.depthUpdate,
  };
  gfx::CompiledPipeline pipeline;
  if (config.alphaUpdate && config.dstAlpha != UINT32_MAX && config.dstAlpha != 0) {
    const auto usesSourceAlpha = [](GXBlendFactor factor) {
      return factor == GX_BL_SRCALPHA || factor == GX_BL_INVSRCALPHA;
    };
    const bool needsSourceAlpha = config.colorUpdate && config.blendMode == GX_BM_BLEND &&
                                  (usesSourceAlpha(config.blendFacSrc) || usesSourceAlpha(config.blendFacDst));
    if (!needsSourceAlpha) {
      options.dstAlphaMode = DstAlphaMode::Replace;
    } else if (webgpu::g_dualSourceBlendingSupported && layout.colorAttachmentCount == 1) {
      options.dstAlphaMode = DstAlphaMode::DualSource;
    } else {
      // Write alpha before RGB, no depth write
      auto alpha = options;
      alpha.dstAlphaMode = DstAlphaMode::Replace;
      alpha.colorUpdate = false;
      alpha.depthUpdate = false;
      pipeline.prepass = create(alpha, "alpha");
      if (!pipeline.prepass) {
        return {};
      }
      options.alphaUpdate = false;
    }
  }
  pipeline.main = create(options, "main");
  if (!pipeline.main) {
    return {};
  }
  return pipeline;
}

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
  gfx::CompiledPipeline pipeline;
  if (!gfx::get_pipeline(data.pipeline, pipeline)) {
    return;
  }
  const auto& resources = gfx::detail::resources();
  pass.SetImmediates(0, &data.immediateData, sizeof(data.immediateData));
  const std::array offsets{data.uniformRange.offset};
  pass.SetBindGroup(1, resources.uniformBindGroup, offsets.size(), offsets.data());
  if (data.bindGroups.textureBindGroup) {
    gfx::pass_binds().texture_group(pass, data.bindGroups.textureBindGroup);
  }
  pass.SetIndexBuffer(resources.indexBuffer, wgpu::IndexFormat::Uint16, data.idxRange.offset, data.idxRange.size);
  if (data.dstAlpha != UINT32_MAX) {
    gfx::pass_binds().blend_alpha(pass, data.dstAlpha);
  }
  const auto draw = [&] {
    if (data.indexCount == 0) {
      pass.Draw(data.vtxCount, data.instanceCount);
    } else {
      pass.DrawIndexed(data.indexCount, data.instanceCount);
    }
  };
  if (pipeline.prepass) {
    gfx::bind_pipeline(pipeline.prepass, pass);
    draw();
  }
  gfx::bind_pipeline(pipeline.main, pass);
  draw();
}

} // namespace aurora::gx
