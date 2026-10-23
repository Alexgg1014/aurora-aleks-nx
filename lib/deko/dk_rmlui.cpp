// aleks2 (deko3d P1): RmlUi (the menus, Settings, the mod dialogs) drawn by the deko3d renderer. MIT,
// this project's; SwitchWaker has no RmlUi (it builds Aurora without it).
//
// Pipelines (Aurora's compile thread, pipeline_cache.cpp resolve_pipeline): the 12 kinds of
// rmlui::PipelineKind, each a vertex module (3 layouts: geometry, fullscreen, blur fullscreen) and a
// fragment module, both with entry point "main". Their stages come from the DKSH cache's named records
// ("rmlui_vs_<layout>.main", "rmlui_fs_<kind>.main", research/deko3d/dksh_cache) or, when the cache lacks
// them, from the WGSL of rmlui/pipeline.cpp through Tint (the Sequential binding scheme) and the uam
// worker, as a GX miss. The fixed state is create_pipeline's: premultiplied-alpha blending or none, the
// colour write mask, no culling, the stencil modes of the clip masks (compare, pass op, read and write
// mask 0xFF, the draw's reference).
//
// Draws (render worker, dk_encode.cpp encode_pass): rmlui::render's bindings. Group 0 is the common
// bind group (the uniforms at the draw's dynamic offset, the linear sampler), groups 1 and 2 the draw's
// (a texture, or uniforms at their dynamic offset). Uniforms are Aurora's shared uniform buffer, which
// with deko3d is the frame's staging block: the address is the staging uniform region + the entry's
// offset + the dynamic offset. Every texture is sampled with the common sampler (the only sampler the
// shaders declare). Geometry draws read the vertex region (Rml::Vertex: position, colour, tex_coord)
// and u32 indices; fullscreen draws are a vertex count with no vertex buffer.
#include "dk_encode.hpp"
#include "dk_gfx.hpp"

#include "dksh_file.h"
#include "gfx/frame.hpp"
#include "gfx/pipeline_cache.hpp"
#include "gfx/resource_cache.hpp"
#include "gfx/resources.hpp"
#include "rmlui/pipeline.hpp"

#include <RmlUi/Core/Vertex.h>

#include <cstddef>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace aurora::gfx::dk {
using namespace aurora::gfx::detail;
using swdk::dklog;
using swdk::R;
using swdk::ShaderEntry;
using swdk::ShaderStage;
using swdk::ShaderState;
using swdk::SlotBinding;

namespace {

// Aurora's staging layout (encoding.cpp), as dk_encode.cpp's
constexpr uint64_t kVertexOffset = 0;
constexpr uint64_t kUniformOffset = kVertexOffset + VertexBufferSize;
constexpr uint64_t kIndexOffset = kUniformOffset + UniformBufferSize;

struct RmlStage {
  ShaderEntry* entry = nullptr;
  std::vector<SlotBinding> bindings;
};

struct RmlPipe {
  RmlStage vs, fs;
  bool vertexBuffer = false;
  bool stencil = false;
  DkRasterizerState rasterizer{};
  DkColorState color{};
  DkColorWriteState colorWrite{};
  DkBlendState blend{};
  DkDepthStencilState depthStencil{};
};

std::mutex g_translateMutex;
uint64_t g_built = 0, g_fromCache = 0, g_translated = 0, g_failed = 0;

const char* layout_name(rmlui::VertexLayoutKind k) {
  switch (k) {
  case rmlui::VertexLayoutKind::Geometry: return "geometry";
  case rmlui::VertexLayoutKind::BlurFullscreen: return "blur_fullscreen";
  case rmlui::VertexLayoutKind::Fullscreen:
  default: return "fullscreen";
  }
}

const char* kind_name(rmlui::PipelineKind k) {
  switch (k) {
  case rmlui::PipelineKind::Geometry: return "geometry";
  case rmlui::PipelineKind::Gradient: return "gradient";
  case rmlui::PipelineKind::Blit: return "blit";
  case rmlui::PipelineKind::OpaqueBlit: return "opaque_blit";
  case rmlui::PipelineKind::SeedResample: return "seed_resample";
  case rmlui::PipelineKind::SimpleFilter: return "simple_filter";
  case rmlui::PipelineKind::Blur: return "blur";
  case rmlui::PipelineKind::RegionBlit: return "region_blit";
  case rmlui::PipelineKind::DropShadow: return "drop_shadow";
  case rmlui::PipelineKind::MaskImage: return "mask_image";
  case rmlui::PipelineKind::Glass: return "glass";
  case rmlui::PipelineKind::ImageEffects: return "image_effects";
  default: return "unknown";
  }
}

// One stage: the cache's named record, else Tint here and the uam worker (waited for: this is the
// compile thread). False (logged) when it cannot be made.
bool make_stage(const std::string& name, std::string_view wgsl, ShaderStage stage, RmlStage& out) {
  if (const swdk::NamedShader* named = swdk::shader_cache_named(name.c_str())) {
    out.entry = named->entry;
    out.bindings = named->bindings;
    std::lock_guard lock{g_translateMutex};
    g_fromCache++;
    return out.entry != nullptr;
  }
  swdk::TranslateResult t;
  {
    std::lock_guard lock{g_translateMutex};
    t = swdk::wgsl_to_glsl(std::string{wgsl}, "main", stage, swdk::BindingScheme::Sequential);
  }
  if (!t.ok) {
    dklog("rmlui: %s does not translate: %s", name.c_str(), t.error.substr(0, 300).c_str());
    return false;
  }
  out.entry = swdk::shader_entry(stage, swdk::glsl_hash(t.glsl));
  out.bindings = std::move(t.bindings);
  swdk::shader_compile(out.entry, t.glsl, true);
  if (!swdk::shader_wait(out.entry)) {
    dklog("rmlui: %s: uam failed, its draws stay skipped", name.c_str());
    return false;
  }
  std::lock_guard lock{g_translateMutex};
  g_translated++;
  return true;
}

DkStencilOp stencil_pass_op(rmlui::StencilMode m) {
  switch (m) {
  case rmlui::StencilMode::ClipReplace: return DkStencilOp_Replace;
  case rmlui::StencilMode::ClipIntersect: return DkStencilOp_Incr;  // WebGPU IncrementClamp
  default: return DkStencilOp_Keep;
  }
}

DkCompareOp stencil_compare(rmlui::StencilMode m) {
  switch (m) {
  case rmlui::StencilMode::EqualKeep:
  case rmlui::StencilMode::ClipIntersect: return DkCompareOp_Equal;
  default: return DkCompareOp_Always;
  }
}

// the entry of a bind group's binding (null if none)
const BindGroup::Entry* group_entry(BindGroupRef ref, uint32_t binding) {
  if (ref == 0) {
    return nullptr;
  }
  WGPUBindGroup raw = find_bind_group_raw(ref);
  BindGroup* g = raw ? find_bind_group(raw) : nullptr;
  if (!g) {
    return nullptr;
  }
  for (const auto& e : g->entries) {
    if (e.binding == binding) {
      return &e;
    }
  }
  return nullptr;
}

uint32_t group_dynamic_offset(const rmlui::DrawData& d, uint32_t group) {
  if (!rmlui::deko3d_group_dynamic(d, group)) {
    return 0;
  }
  switch (group) {
  case 0: return d.uniformRange.offset;
  case 1: return d.bindGroup1DynamicOffset;
  case 2: return d.bindGroup2DynamicOffset;
  default: return 0;
  }
}

void log_once(const char* what) {
  static std::vector<std::string> seen;
  for (const auto& s : seen) {
    if (s == what) {
      return;
    }
  }
  seen.emplace_back(what);
  dklog("rmlui: %s (logged once)", what);
}

// a stage's bindings for one draw; false when something it needs is missing (the draw is skipped)
bool bind_stage(const RmlStage& st, DkStage stage, const rmlui::DrawData& d, DkGpuAddr staging, BindGroupRef groups[3],
                Samp* sampler) {
  const WGPUBuffer uniformBuffer = gfx::detail::resources().uniformBuffer.Get();
  for (const SlotBinding& b : st.bindings) {
    if (b.group > 2) {
      log_once("a binding outside groups 0-2");
      return false;
    }
    const BindGroup::Entry* e = group_entry(groups[b.group], b.binding);
    if (!e) {
      log_once("a binding whose bind group entry is missing");
      return false;
    }
    switch (b.kind) {
    case SlotBinding::Kind::Uniform: {
      if (e->buffer != uniformBuffer) {
        log_once("a uniform binding that is not Aurora's shared uniform buffer");
        return false;
      }
      const uint64_t offset = e->offset + group_dynamic_offset(d, b.group);
      const uint32_t size = uint32_t((e->size + 15u) & ~uint64_t(15u));
      dkCmdBufBindUniformBuffer(R.cmd, stage, b.slot, staging + kUniformOffset + offset, size);
      break;
    }
    case SlotBinding::Kind::Sampler:
      if (!e->view) {
        log_once("a texture binding without a view");
        return false;
      }
      dkCmdBufBindTexture(R.cmd, stage, b.slot, texture_handle(e->view, sampler));
      break;
    default:
      log_once("a storage or immediates binding (RmlUi has none)");
      return false;
    }
  }
  return true;
}

} // namespace

const void* build_rml_pipeline(const rmlui::PipelineConfig& config) {
  const auto kind = static_cast<rmlui::PipelineKind>(config.kind);
  const auto layoutKind = static_cast<rmlui::VertexLayoutKind>(config.vertexLayout);
  auto* p = new RmlPipe;
  const std::string vsName = std::string{"rmlui_vs_"} + layout_name(layoutKind) + ".main";
  const std::string fsName = std::string{"rmlui_fs_"} + kind_name(kind) + ".main";
  if (!make_stage(vsName, rmlui::vertex_wgsl(layoutKind), ShaderStage::Vertex, p->vs) ||
      !make_stage(fsName, rmlui::fragment_wgsl(kind), ShaderStage::Fragment, p->fs)) {
    std::lock_guard lock{g_translateMutex};
    g_failed++;
    delete p;
    return nullptr;
  }
  p->vertexBuffer = layoutKind == rmlui::VertexLayoutKind::Geometry;
  // zeroed first: the states are bound as they are
  memset(&p->rasterizer, 0, sizeof p->rasterizer);
  memset(&p->color, 0, sizeof p->color);
  memset(&p->colorWrite, 0, sizeof p->colorWrite);
  memset(&p->blend, 0, sizeof p->blend);
  memset(&p->depthStencil, 0, sizeof p->depthStencil);
  dkRasterizerStateDefaults(&p->rasterizer);
  dkColorStateDefaults(&p->color);
  dkColorWriteStateDefaults(&p->colorWrite);
  dkBlendStateDefaults(&p->blend);
  dkDepthStencilStateDefaults(&p->depthStencil);
  p->rasterizer.cullMode = DkFace_None;
  p->rasterizer.frontFace = DkFrontFace_CW;
  const bool blend = static_cast<rmlui::BlendMode>(config.blendMode) == rmlui::BlendMode::Premultiplied;
  dkColorStateSetBlendEnable(&p->color, 0, blend);
  dkBlendStateSetOps(&p->blend, DkBlendOp_Add, DkBlendOp_Add);
  dkBlendStateSetFactors(&p->blend, DkBlendFactor_One, DkBlendFactor_InvSrcAlpha, DkBlendFactor_One,
                         DkBlendFactor_InvSrcAlpha);
  dkColorWriteStateSetMask(&p->colorWrite, 0, config.colorWriteMask & 0xF);
  DkDepthStencilState& ds = p->depthStencil;
  ds.depthTestEnable = false;
  ds.depthWriteEnable = false;
  const auto stencilMode = static_cast<rmlui::StencilMode>(config.stencilMode);
  p->stencil = stencilMode != rmlui::StencilMode::None;
  ds.stencilTestEnable = p->stencil;
  ds.stencilFrontFailOp = ds.stencilBackFailOp = DkStencilOp_Keep;
  ds.stencilFrontDepthFailOp = ds.stencilBackDepthFailOp = DkStencilOp_Keep;
  ds.stencilFrontPassOp = ds.stencilBackPassOp = stencil_pass_op(stencilMode);
  ds.stencilFrontCompareOp = ds.stencilBackCompareOp = stencil_compare(stencilMode);
  std::lock_guard lock{g_translateMutex};
  if (g_built++ == 0) {
    dklog("rmlui: first pipeline (kind %s, layout %s) built", kind_name(kind), layout_name(layoutKind));
  }
  return p;
}

bool draw_rml(const void* payload, DkGpuAddr staging, uint32_t tw, uint32_t th) {
  (void)tw;
  (void)th;
  rmlui::DrawData d;
  memcpy(&d, payload, sizeof d);
  CompiledPipeline cp;
  if (!get_pipeline(d.pipeline, cp) || cp.dkMain == nullptr) {
    return false;
  }
  const auto* p = static_cast<const RmlPipe*>(cp.dkMain);
  if (!swdk::shader_usable(p->vs.entry) || !swdk::shader_usable(p->fs.entry)) {
    return false;
  }
  BindGroupRef groups[3] = {rmlui::deko3d_common_bind_group(), d.bindGroup1, d.bindGroup2};
  // the shaders' only sampler: the common group's binding 1
  const BindGroup::Entry* samplerEntry = group_entry(groups[0], 1);
  Samp* sampler = samplerEntry ? samplerEntry->sampler : nullptr;

  const DkShader* shaders[] = {&p->vs.entry->shader, &p->fs.entry->shader};
  if (swdk::code_take_written()) {
    dkCmdBufBarrier(R.cmd, DkBarrier_None, DkInvalidateFlags_Shader);
  }
  dkCmdBufBindShaders(R.cmd, DkStageFlag_GraphicsMask, shaders, 2);
  dkCmdBufBindRasterizerState(R.cmd, &p->rasterizer);
  dkCmdBufBindColorState(R.cmd, &p->color);
  dkCmdBufBindColorWriteState(R.cmd, &p->colorWrite);
  dkCmdBufBindBlendStates(R.cmd, 0, &p->blend, 1);
  dkCmdBufBindDepthStencilState(R.cmd, &p->depthStencil);
  if (p->stencil) {
    dkCmdBufSetStencil(R.cmd, DkFace_FrontAndBack, 0xFF, uint8_t(d.stencilRef), 0xFF);
  }
  if (d.hasBlendConstant != 0) {
    dkCmdBufSetBlendConst(R.cmd, d.blendConstant[0], d.blendConstant[1], d.blendConstant[2], d.blendConstant[3]);
  }
  // (a GX pass may have flipped y: RmlUi draws upright)
  const DkViewportSwizzle identity = {DkSwizzle_PositiveX, DkSwizzle_PositiveY, DkSwizzle_PositiveZ,
                                      DkSwizzle_PositiveW};
  dkCmdBufSetViewportSwizzles(R.cmd, 0, &identity, 1);
  bool ok = bind_stage(p->vs, DkStage_Vertex, d, staging, groups, sampler) &&
            bind_stage(p->fs, DkStage_Fragment, d, staging, groups, sampler);
  if (ok) {
    if (p->vertexBuffer && static_cast<rmlui::DrawKind>(d.drawKind) == rmlui::DrawKind::Geometry) {
      const DkVtxAttribState attribs[] = {
          {0, 0, offsetof(Rml::Vertex, position), DkVtxAttribSize_2x32, DkVtxAttribType_Float, 0},
          {0, 0, offsetof(Rml::Vertex, tex_coord), DkVtxAttribSize_2x32, DkVtxAttribType_Float, 0},
          {0, 0, offsetof(Rml::Vertex, colour), DkVtxAttribSize_4x8, DkVtxAttribType_Unorm, 0},
      };
      const DkVtxBufferState buffer = {sizeof(Rml::Vertex), 0};
      dkCmdBufBindVtxAttribState(R.cmd, attribs, 3);
      dkCmdBufBindVtxBufferState(R.cmd, &buffer, 1);
      dkCmdBufBindVtxBuffer(R.cmd, 0, staging + kVertexOffset + d.vertexRange.offset, d.vertexRange.size);
      dkCmdBufBindIdxBuffer(R.cmd, DkIdxFormat_Uint32, staging + kIndexOffset + d.indexRange.offset);
      dkCmdBufDrawIndexed(R.cmd, DkPrimitive_Triangles, d.indexCount, 1, 0, 0, 0);
    } else {
      dkCmdBufBindVtxAttribState(R.cmd, nullptr, 0);
      dkCmdBufBindVtxBufferState(R.cmd, nullptr, 0);
      dkCmdBufDraw(R.cmd, DkPrimitive_Triangles, d.vertexCount, 1, 0, 0);
    }
  }
  // the encoder's state cache no longer knows what is bound
  invalidate_state();
  return ok;
}

RmlStats rml_stats() {
  std::lock_guard lock{g_translateMutex};
  return {g_built, g_fromCache, g_translated, g_failed};
}

} // namespace aurora::gfx::dk
