#pragma once

#include "../gfx/pipeline_cache.hpp"
#include "gx.hpp"

namespace aurora::gx {
struct DrawData {
  gfx::PipelineRef pipeline;
  GXBindGroups bindGroups;
  gfx::Range vertRange;
  gfx::Range idxRange;
  gfx::Range uniformRange;
  DrawImmediateData immediateData;
  uint32_t vtxCount;
  uint32_t indexCount;
  uint32_t instanceCount;
  uint32_t dstAlpha;
};

constexpr uint32_t GXPipelineConfigVersion = 13;
struct PipelineConfig {
  uint32_t version = GXPipelineConfigVersion;
  uint32_t msaaSamples = 1; // deprecated
  ShaderConfig shaderConfig;
  GXCompare depthFunc;
  GXCullMode cullMode;
  GXBlendMode blendMode;
  GXBlendFactor blendFacSrc, blendFacDst;
  GXLogicOp blendOp;
  uint32_t dstAlpha;
  uint32_t polygonOffsetBits;
  uint32_t polygonOffsetScaleBits;
  uint32_t polygonOffsetClampBits;
  bool depthCompare, depthUpdate, alphaUpdate, colorUpdate;
};
static_assert(std::has_unique_object_representations_v<PipelineConfig>);

struct PipelineOptions {
  DstAlphaMode dstAlphaMode = DstAlphaMode::None;
  bool colorUpdate = false;
  bool alphaUpdate = false;
  bool depthUpdate = false;
};

gfx::CompiledPipeline create_pipeline(const PipelineConfig& config, const gfx::RenderTargetLayout& layout);
#if defined(AURORA_GFX_DEKO3D)
// aleks2 (deko3d P1): build_pipeline's fixed state for a pass of a config, the same values it gives
// Dawn, without creating a WebGPU pipeline (lib/deko/dk_pipeline.cpp)
struct PipelineStateDesc {
  bool hasDepth = false;
  wgpu::DepthStencilState depthStencil;
  wgpu::BlendState blend;
  wgpu::PrimitiveState primitive;
  uint32_t colorCount = 0;
  std::array<wgpu::ColorWriteMask, gfx::MaxColorAttachments> writeMasks{};
};
PipelineStateDesc pipeline_state_desc(const PipelineConfig& config, const gfx::RenderTargetLayout& layout,
                                      const PipelineOptions& options) noexcept;
#endif
void render(const DrawData& data, const wgpu::RenderPassEncoder& pass);

void queue_surface(const u8* dlStart, uint32_t dlSize, bool bigEndian) noexcept;
} // namespace aurora::gx
