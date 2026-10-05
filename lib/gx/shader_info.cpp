#include "shader_info.hpp"

#include "../gfx/hash.hpp"
#include "../gfx/recording.hpp"

#include <aurora/tuning.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <optional>

#include <tracy/Tracy.hpp>

namespace aurora::gx {
namespace {
constexpr Module Log{"aurora::gx"};

bool is_alpha_bump_channel(GXChannelID id) { return id == GX_ALPHA_BUMP || id == GX_ALPHA_BUMPN; }

Vec4<float> texture_size_bias(const gfx::TextureBind& tex) {
  auto width = static_cast<float>(tex.texObj.width());
  auto height = static_cast<float>(tex.texObj.height());
  float vpBias = 0.f;
  if (tex.ref && tex.ref->hasArbitraryMips) {
    const float viewportScale =
        std::min(g_gxState.renderViewport.width / std::max(g_gxState.logicalViewport.width, 1.f),
                 g_gxState.renderViewport.height / std::max(g_gxState.logicalViewport.height, 1.f));
    const float replacementScale = static_cast<float>(tex.ref->size.width) / std::max(width, 1.f);
    vpBias = std::log2(viewportScale / std::max(replacementScale, 0.001f));
  }
  return {width, height, tex.texObj.lod_bias() + vpBias, 0.0f};
}

void color_arg_reg_info(GXTevColorArg arg, const TevStage& stage, ShaderInfo& info) {
  switch (arg) {
  case GX_CC_CPREV:
  case GX_CC_APREV:
    if (!info.writesTevReg.test(GX_TEVPREV)) {
      info.loadsTevReg.set(GX_TEVPREV);
    }
    break;
  case GX_CC_C0:
  case GX_CC_A0:
    if (!info.writesTevReg.test(GX_TEVREG0)) {
      info.loadsTevReg.set(GX_TEVREG0);
    }
    break;
  case GX_CC_C1:
  case GX_CC_A1:
    if (!info.writesTevReg.test(GX_TEVREG1)) {
      info.loadsTevReg.set(GX_TEVREG1);
    }
    break;
  case GX_CC_C2:
  case GX_CC_A2:
    if (!info.writesTevReg.test(GX_TEVREG2)) {
      info.loadsTevReg.set(GX_TEVREG2);
    }
    break;
  case GX_CC_TEXC:
  case GX_CC_TEXA:
    if (stage.texMapId != GX_TEXMAP_NULL) {
      CHECK(stage.texCoordId != GX_TEXCOORD_NULL, "tex coord not bound");
      info.sampledTexCoords.set(stage.texCoordId);
      info.sampledTextures.set(stage.texMapId);
    }
    break;
  case GX_CC_RASC:
  case GX_CC_RASA:
    if (stage.channelId != GX_COLOR_NULL && stage.channelId != GX_COLOR_ZERO &&
        !is_alpha_bump_channel(stage.channelId)) {
      info.sampledColorChannels.set(color_channel(stage.channelId));
    }
    break;
  case GX_CC_KONST:
    switch (stage.kcSel) {
    case GX_TEV_KCSEL_K0:
    case GX_TEV_KCSEL_K0_R:
    case GX_TEV_KCSEL_K0_G:
    case GX_TEV_KCSEL_K0_B:
    case GX_TEV_KCSEL_K0_A:
      info.sampledKColors.set(0);
      break;
    case GX_TEV_KCSEL_K1:
    case GX_TEV_KCSEL_K1_R:
    case GX_TEV_KCSEL_K1_G:
    case GX_TEV_KCSEL_K1_B:
    case GX_TEV_KCSEL_K1_A:
      info.sampledKColors.set(1);
      break;
    case GX_TEV_KCSEL_K2:
    case GX_TEV_KCSEL_K2_R:
    case GX_TEV_KCSEL_K2_G:
    case GX_TEV_KCSEL_K2_B:
    case GX_TEV_KCSEL_K2_A:
      info.sampledKColors.set(2);
      break;
    case GX_TEV_KCSEL_K3:
    case GX_TEV_KCSEL_K3_R:
    case GX_TEV_KCSEL_K3_G:
    case GX_TEV_KCSEL_K3_B:
    case GX_TEV_KCSEL_K3_A:
      info.sampledKColors.set(3);
      break;
    default:
      break;
    }
    break;
  default:
    break;
  }
}

void alpha_arg_reg_info(GXTevAlphaArg arg, const TevStage& stage, ShaderInfo& info) {
  switch (arg) {
  case GX_CA_APREV:
    if (!info.writesTevReg.test(GX_TEVPREV)) {
      info.loadsTevReg.set(GX_TEVPREV);
    }
    break;
  case GX_CA_A0:
    if (!info.writesTevReg.test(GX_TEVREG0)) {
      info.loadsTevReg.set(GX_TEVREG0);
    }
    break;
  case GX_CA_A1:
    if (!info.writesTevReg.test(GX_TEVREG1)) {
      info.loadsTevReg.set(GX_TEVREG1);
    }
    break;
  case GX_CA_A2:
    if (!info.writesTevReg.test(GX_TEVREG2)) {
      info.loadsTevReg.set(GX_TEVREG2);
    }
    break;
  case GX_CA_TEXA:
    if (stage.texMapId != GX_TEXMAP_NULL) {
      CHECK(stage.texCoordId != GX_TEXCOORD_NULL, "tex coord not bound");
      info.sampledTexCoords.set(stage.texCoordId);
      info.sampledTextures.set(stage.texMapId);
    }
    break;
  case GX_CA_RASA:
    if (stage.channelId != GX_COLOR_NULL && stage.channelId != GX_COLOR_ZERO &&
        !is_alpha_bump_channel(stage.channelId)) {
      info.sampledColorChannels.set(color_channel(stage.channelId));
    }
    break;
  case GX_CA_KONST:
    switch (stage.kaSel) {
    case GX_TEV_KASEL_K0_R:
    case GX_TEV_KASEL_K0_G:
    case GX_TEV_KASEL_K0_B:
    case GX_TEV_KASEL_K0_A:
      info.sampledKColors.set(0);
      break;
    case GX_TEV_KASEL_K1_R:
    case GX_TEV_KASEL_K1_G:
    case GX_TEV_KASEL_K1_B:
    case GX_TEV_KASEL_K1_A:
      info.sampledKColors.set(1);
      break;
    case GX_TEV_KASEL_K2_R:
    case GX_TEV_KASEL_K2_G:
    case GX_TEV_KASEL_K2_B:
    case GX_TEV_KASEL_K2_A:
      info.sampledKColors.set(2);
      break;
    case GX_TEV_KASEL_K3_R:
    case GX_TEV_KASEL_K3_G:
    case GX_TEV_KASEL_K3_B:
    case GX_TEV_KASEL_K3_A:
      info.sampledKColors.set(3);
      break;
    default:
      break;
    }
    break;
  default:
    break;
  }
}

f32 tex_offset(GXTexOffset offs) noexcept {
  switch (offs) {
  default:
  case GX_TO_ZERO:
    return 0.f;
  case GX_TO_SIXTEENTH:
    return 1.f / 16.f;
  case GX_TO_EIGHTH:
    return 1.f / 8.f;
  case GX_TO_FOURTH:
    return 1.f / 4.f;
  case GX_TO_HALF:
    return 1.f / 2.f;
  case GX_TO_ONE:
    return 1.f;
  }
}

u32 point_texcoord_mask() noexcept {
  u32 mask = 0;
  for (int i = 0; i < MaxTexCoord; ++i) {
    if (g_gxState.texCoordScales[i].pointOffset) {
      mask |= 1 << i;
    }
  }
  return mask;
}

u32 line_texcoord_mask() noexcept {
  u32 mask = 0;
  for (int i = 0; i < MaxTexCoord; ++i) {
    if (g_gxState.texCoordScales[i].lineOffset) {
      mask |= 1 << i;
    }
  }
  return mask;
}
} // namespace

ShaderInfo build_shader_info(const ShaderConfig& config) noexcept {
  ZoneScoped;

  ShaderInfo info{
      // render/logical viewport size, proj
      .uniformSize = 8 + 8 + 64,
  };

  if (config.lineMode != 0) {
    info.uniformSize += 4 + 4 + 4 + 4; // line_width, line_aspect_y, line_tex_offset, line_texcoord_mask
    info.lineMode = config.lineMode;
  }

  for (int attr = 0; attr < config.attrs.size(); attr++) {
    const auto attrType = config.attrs[attr].attrType;
    if ((attr == GX_VA_PNMTXIDX || (attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX)) && attrType == GX_DIRECT) {
      info.indexAttr.set(attr);
    } else if (attrType == GX_INDEX8 || attrType == GX_INDEX16) {
      info.indexAttr.set(attr);
    }
  }

  // 10 position matrices, 10 texture matrices, 10 normal matrices.
  info.uniformSize += sizeof(Mat3x4<float>) * 30;

  for (int i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    // Color pass
    color_arg_reg_info(stage.colorPass.a, stage, info);
    color_arg_reg_info(stage.colorPass.b, stage, info);
    color_arg_reg_info(stage.colorPass.c, stage, info);
    color_arg_reg_info(stage.colorPass.d, stage, info);
    info.writesTevReg.set(stage.colorOp.outReg);

    // Alpha pass
    alpha_arg_reg_info(stage.alphaPass.a, stage, info);
    alpha_arg_reg_info(stage.alphaPass.b, stage, info);
    alpha_arg_reg_info(stage.alphaPass.c, stage, info);
    alpha_arg_reg_info(stage.alphaPass.d, stage, info);
    if (!info.writesTevReg.test(stage.alphaOp.outReg)) {
      // If we're writing alpha to a register that's not been
      // written to in the shader, load from uniform buffer
      info.loadsTevReg.set(stage.alphaOp.outReg);
      info.writesTevReg.set(stage.alphaOp.outReg);
    }
  }
  for (int i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    // Skip if not enabled
    if (stage.indTexStage >= config.numIndStages) {
      continue;
    }
    const bool usesIndStage = stage.indTexMtxId != GX_ITM_OFF || is_alpha_bump_channel(stage.channelId);
    const bool usesTevTexCoord = stage.indTexMtxId != GX_ITM_OFF || stage.indTexWrapS != GX_ITW_OFF ||
                                 stage.indTexWrapT != GX_ITW_OFF || stage.indTexAddPrev;
    if (usesTevTexCoord && stage.texCoordId != GX_TEXCOORD_NULL) {
      info.sampledTexCoords.set(stage.texCoordId);
    }
    if (usesTevTexCoord && stage.texMapId != GX_TEXMAP_NULL &&
        (stage.indTexMtxId != GX_ITM_OFF || stage.indTexAddPrev)) {
      info.sampledTextures.set(stage.texMapId);
    }
    if (!usesIndStage) {
      continue;
    }
    info.usedIndStages.set(stage.indTexStage);
    const auto& indStage = config.indStages[stage.indTexStage];
    info.sampledTextures.set(indStage.texMapId);
    info.sampledTexCoords.set(indStage.texCoordId);
    info.sampledIndTextures.set(indStage.texMapId);
    // Track which indirect matrix is used
    if (stage.indTexMtxId >= GX_ITM_0 && stage.indTexMtxId <= GX_ITM_2) {
      info.usedIndTexMtxs.set(stage.indTexMtxId - GX_ITM_0);
    } else if (stage.indTexMtxId >= GX_ITM_S0 && stage.indTexMtxId <= GX_ITM_S2) {
      info.usedIndTexMtxs.set(stage.indTexMtxId - GX_ITM_S0);
    } else if (stage.indTexMtxId >= GX_ITM_T0 && stage.indTexMtxId <= GX_ITM_T2) {
      info.usedIndTexMtxs.set(stage.indTexMtxId - GX_ITM_T0);
    }
  }

  // Emboss bump needs its source texcoord generated and a light enabled
  for (int i = 0; i < info.sampledTexCoords.size(); ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue;
    }
    const auto& tcg = config.tcgs[i];
    if (tcg.type >= GX_TG_BUMP0 && tcg.type <= GX_TG_BUMP7) {
      info.sampledTexCoords.set(tcg.embossSrc);
      info.lightingEnabled = true;
    }
  }

  info.uniformSize += info.loadsTevReg.count() * sizeof(Vec4<float>);
  for (int i = 0; i < info.sampledColorChannels.size(); ++i) {
    if (info.sampledColorChannels.test(i)) {
      const auto& cc = config.colorChannels[i];
      const auto& cca = config.colorChannels[i + GX_ALPHA0];
      if (cc.lightingEnabled || cca.lightingEnabled) {
        info.lightingEnabled = true;
      }
    }
  }
  if (info.lightingEnabled) {
    // Lights + light state for all channels
    info.uniformSize += 16 + sizeof(Light) * GX::MaxLights;
  }
  for (int i = 0; i < info.sampledColorChannels.size(); ++i) {
    if (info.sampledColorChannels.test(i)) {
      const auto& cc = config.colorChannels[i];
      if (cc.lightingEnabled && cc.ambSrc == GX_SRC_REG) {
        info.uniformSize += sizeof(Vec4<float>);
      }
      if (cc.matSrc == GX_SRC_REG) {
        info.uniformSize += sizeof(Vec4<float>);
      }
      const auto& cca = config.colorChannels[i + GX_ALPHA0];
      if (cca.lightingEnabled && cca.ambSrc == GX_SRC_REG) {
        info.uniformSize += sizeof(Vec4<float>);
      }
      if (cca.matSrc == GX_SRC_REG) {
        info.uniformSize += sizeof(Vec4<float>);
      }
    }
  }
  info.uniformSize += info.sampledKColors.count() * sizeof(Vec4<float>);
  for (int i = 0; i < info.sampledTexCoords.size(); ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue;
    }
    const auto& tcg = config.tcgs[i];
    if (tcg.postMtx != GX_PTIDENTITY) {
      u32 postMtxIdx = (tcg.postMtx - GX_PTTEXMTX0) / 3;
      info.usesPTTexMtx.set(postMtxIdx);
    }
  }
  if (info.usesPTTexMtx.any()) {
    info.uniformSize += sizeof(Mat3x4<float>) * MaxPTTexMtx;
  }
  if (config.fogType != GX_FOG_NONE) {
    info.usesFog = true;
    info.uniformSize += sizeof(Fog);
  }
  info.uniformSize += MaxTexCoord * sizeof(Vec4<float>);
  if (info.usedIndTexMtxs.any()) {
    info.uniformSize += MaxIndTexMtxs * sizeof(Mat2x4<float>);
  }
  info.uniformSize += info.sampledTextures.count() * sizeof(Vec4<float>);
  info.uniformSize = gfx::align_uniform(info.uniformSize);
  if (info.uniformSize > MaxUniformSize) {
    Log.fatal("Uniform size exceeds maximum: {} > {}", info.uniformSize, MaxUniformSize);
  }
  return info;
}

namespace {
// aleks2 (H35): the sections of the uniform block, in the order fill_uniform writes
// them. Only used to say where a Verify mismatch is.
enum class UniformSection : u8 {
  Viewport,
  LinePoint,
  Projection,
  PosMtx,
  TexMtx,
  NrmMtx,
  TevRegs,
  Lights,
  ChannelColors,
  KColors,
  PtTexMtx,
  Fog,
  TexCoordScales,
  IndTexMtx,
  TextureSizes,
  Count,
};
constexpr std::array<const char*, static_cast<size_t>(UniformSection::Count)> kUniformSectionNames{
    "viewport",       "line_point", "projection", "pos_mtx", "tex_mtx",         "nrm_mtx",
    "tev_regs",       "lights",     "channel_colors", "kcolors", "pt_tex_mtx", "fog",
    "texcoord_scales", "ind_tex_mtx", "texture_sizes",
};

// The normal fill marks nothing; this compiles away.
struct NoSectionMarks {
  void operator()(UniformSection, size_t) const noexcept {}
};

// Where each section started in one fill, to name the section of a byte offset.
struct SectionMarks {
  std::array<size_t, static_cast<size_t>(UniformSection::Count)> start{};
  std::array<bool, static_cast<size_t>(UniformSection::Count)> present{};

  void operator()(UniformSection section, size_t offset) noexcept {
    start[static_cast<size_t>(section)] = offset;
    present[static_cast<size_t>(section)] = true;
  }
  UniformSection at(size_t offset) const noexcept {
    UniformSection found = UniformSection::Viewport;
    size_t best = 0;
    for (size_t i = 0; i < start.size(); ++i) {
      if (present[i] && start[i] <= offset && start[i] >= best) {
        best = start[i];
        found = static_cast<UniformSection>(i);
      }
    }
    return found;
  }
};
} // namespace

template <typename Mark>
static void fill_uniform(ByteBuffer& buf, const ShaderInfo& info, Mark&& mark) noexcept {
  buf.reserve_extra(info.uniformSize);

  mark(UniformSection::Viewport, buf.size());
  buf.append<f32>(g_gxState.renderViewport.width);
  buf.append<f32>(g_gxState.renderViewport.height);
  buf.append<f32>(g_gxState.logicalViewport.width);
  buf.append<f32>(g_gxState.logicalViewport.height);
  if (info.lineMode != 0) {
    mark(UniformSection::LinePoint, buf.size());
    if (info.lineMode == 3) { // GX_POINTS
      buf.append<f32>(static_cast<f32>(g_gxState.pointSize) / 6.f);
      buf.append<f32>(1.0f);
      buf.append<f32>(tex_offset(g_gxState.pointTexOffset));
      buf.append<u32>(point_texcoord_mask());
    } else { // GX_LINES / GX_LINESTRIP
      buf.append<f32>(static_cast<f32>(g_gxState.lineWidth) / 6.f);
      buf.append<f32>(g_gxState.lineHalfAspect ? 0.5f : 1.f);
      buf.append<f32>(tex_offset(g_gxState.lineTexOffset));
      buf.append<u32>(line_texcoord_mask());
    }
  }
  auto proj = g_gxState.proj;
  if constexpr (UseReversedZ) {
    proj.m2 = proj.m2 * Vec4{-1.f, -1.f, -1.f, -1.f};
  } else {
    proj.m2 = proj.m2 + proj.m3;
  }
  mark(UniformSection::Projection, buf.size());
  buf.append(proj);

  mark(UniformSection::PosMtx, buf.size());
  for (int i = 0; i < MaxPnMtx; i++) {
    buf.append(g_gxState.pnMtx[i].pos);
  }

  mark(UniformSection::TexMtx, buf.size());
  for (int i = 0; i < MaxTexMtx; i++) {
    buf.append(g_gxState.texMtxs[i]);
  }

  mark(UniformSection::NrmMtx, buf.size());
  for (int i = 0; i < MaxPnMtx; i++) {
    buf.append(g_gxState.pnMtx[i].nrm);
  }

  mark(UniformSection::TevRegs, buf.size());
  for (int i = 0; i < info.loadsTevReg.size(); ++i) {
    if (info.loadsTevReg.test(i)) {
      buf.append(g_gxState.colorRegs[i]);
    }
  }
  if (info.lightingEnabled) {
    mark(UniformSection::Lights, buf.size());
    // Lights
    static_assert(sizeof(g_gxState.lights) == 80 * GX::MaxLights);
    buf.append(g_gxState.lights);
    // Light state for all channels
    for (int i = 0; i < 4; ++i) {
      buf.append<u32>(g_gxState.colorChannelState[i].lightMask.to_ulong());
    }
  }
  mark(UniformSection::ChannelColors, buf.size());
  for (int i = 0; i < info.sampledColorChannels.size(); ++i) {
    if (!info.sampledColorChannels.test(i)) {
      continue;
    }
    const auto& ccc = g_gxState.colorChannelConfig[i];
    const auto& ccs = g_gxState.colorChannelState[i];
    if (ccc.lightingEnabled && ccc.ambSrc == GX_SRC_REG) {
      buf.append(ccs.ambColor);
    }
    if (ccc.matSrc == GX_SRC_REG) {
      buf.append(ccs.matColor);
    }
    const auto& ccca = g_gxState.colorChannelConfig[i + GX_ALPHA0];
    const auto& ccsa = g_gxState.colorChannelState[i + GX_ALPHA0];
    if (ccca.lightingEnabled && ccca.ambSrc == GX_SRC_REG) {
      buf.append(ccsa.ambColor);
    }
    if (ccca.matSrc == GX_SRC_REG) {
      buf.append(ccsa.matColor);
    }
  }
  mark(UniformSection::KColors, buf.size());
  for (int i = 0; i < info.sampledKColors.size(); ++i) {
    if (info.sampledKColors.test(i)) {
      buf.append(g_gxState.kcolors[i]);
    }
  }
  if (info.usesPTTexMtx.any()) {
    mark(UniformSection::PtTexMtx, buf.size());
    for (int i = 0; i < info.usesPTTexMtx.size(); ++i) {
      buf.append(g_gxState.ptTexMtxs[i]);
    }
  }
  if (info.usesFog) {
    mark(UniformSection::Fog, buf.size());
    const auto& state = g_gxState.fog;
    const float logicalWidth = std::max(g_gxState.logicalViewport.width, 1.f);
    const float renderWidth = std::max(g_gxState.renderViewport.width, 1.f);
    Fog fog{
        .color = state.color,
        .a = state.a,
        .b = state.b,
        .c = state.c,
        .rangeCenter = ((static_cast<float>(state.rangeCenter) - g_gxState.logicalViewport.left) / logicalWidth) * 2.f -
                       1.f + (g_gxState.renderViewport.left / renderWidth) * 2.f,
    };
    for (u32 i = 0; i < state.rangeK.size(); ++i) {
      const u32 source = (i & ~1u) | (1u - (i & 1u));
      fog.rangeK[i / 4][i % 4] = static_cast<float>(state.rangeK[source]) / 64.f;
    }
    fog.rangeK[2][2] = fog.rangeK[2][1];
    fog.rangeK[2][3] = fog.rangeK[2][1];
    buf.append(fog);
  }
  mark(UniformSection::TexCoordScales, buf.size());
  for (const auto& scale : g_gxState.texCoordScales) {
    buf.append(Vec4{static_cast<f32>(scale.scaleS) + 1.0f, static_cast<f32>(scale.scaleT) + 1.0f, 0.0f, 0.0f});
  }
  if (info.usedIndTexMtxs.any()) {
    mark(UniformSection::IndTexMtx, buf.size());
    for (u32 i = 0; i < MaxIndTexMtxs; ++i) {
      const auto& mtx = g_gxState.indTexMtxs[i];
      buf.append(Vec4{mtx.mtx.m0.x, mtx.mtx.m0.y, mtx.mtx.m1.x, mtx.mtx.m1.y});
      buf.append(Vec4{mtx.mtx.m2.x, mtx.mtx.m2.y, std::exp2f(mtx.scaleExp), 0.0f});
    }
  }
  mark(UniformSection::TextureSizes, buf.size());
  for (int i = 0; i < info.sampledTextures.size(); ++i) {
    if (!info.sampledTextures.test(i)) {
      continue;
    }
    buf.append(texture_size_bias(get_texture(static_cast<GXTexMapID>(i))));
  }
}

namespace {
// Draws that differ only in state this shader does not read produce the same
// uniform block as the draw before. Pushing it again would only grow the uniform
// buffer, so the last block is remembered by size and hash and its range reused.
class UniformRepeat {
public:
  void forget() noexcept { mLast.reset(); }

  gfx::Range push(const ByteBuffer& block, bool& repeated) {
    const uint64_t hash = xxh3_hash_s(block.data(), block.size());
    repeated = mLast && mLast->size == block.size() && mLast->hash == hash;
    if (repeated) {
      return mLast->range;
    }
    const gfx::Range range = gfx::push_uniform(block.data(), block.size());
    mLast = Last{block.size(), hash, range};
    return range;
  }

private:
  struct Last {
    size_t size;
    uint64_t hash;
    gfx::Range range;
  };
  std::optional<Last> mLast;
};

UniformRepeat g_uniformRepeat;

// ---------------------------------------------------------------------------------
// aleks2 (H35): reuse of the uniform block when only the pipeline changed. The input
// audit and the argument for each condition are in docs/switch-h35-fifo-plan.md of
// the main repository. Everything here belongs to the FIFO thread except the mode,
// the disable latch and the counters, which other threads read.
// ---------------------------------------------------------------------------------

// Everything that decides which sections fill_uniform writes and how long they are:
// every ShaderInfo field it branches on, and the live channel configuration it reads
// (not part of ShaderInfo). With equal keys, the bytes depend only on GX state values.
struct UniformLayoutKey {
  u32 uniformSize = 0;
  u32 sampledTextures = 0;
  u8 lineMode = 0;
  u8 loadsTevReg = 0;
  u8 sampledColorChannels = 0;
  u8 sampledKColors = 0;
  // bit0 lighting, bit1 fog, bit2 any post-transform matrix, bit3 any indirect matrix
  u8 flags = 0;
  // Per channel slot (COLOR0, COLOR1, ALPHA0, ALPHA1) of a sampled channel: bit0 = the
  // ambient colour is appended (lighting on, ambient from register), bit1 = the
  // material colour is appended (material from register).
  std::array<u8, MaxColorChannels> channelRegs{};

  bool operator==(const UniformLayoutKey&) const = default;
};

UniformLayoutKey layout_key(const ShaderInfo& info) noexcept {
  UniformLayoutKey key{
      .uniformSize = info.uniformSize,
      .sampledTextures = static_cast<u32>(info.sampledTextures.to_ulong()),
      .lineMode = info.lineMode,
      .loadsTevReg = static_cast<u8>(info.loadsTevReg.to_ulong()),
      .sampledColorChannels = static_cast<u8>(info.sampledColorChannels.to_ulong()),
      .sampledKColors = static_cast<u8>(info.sampledKColors.to_ulong()),
      .flags = static_cast<u8>((info.lightingEnabled ? 1u : 0u) | (info.usesFog ? 2u : 0u) |
                               (info.usesPTTexMtx.any() ? 4u : 0u) | (info.usedIndTexMtxs.any() ? 8u : 0u)),
  };
  for (int i = 0; i < info.sampledColorChannels.size(); ++i) {
    if (!info.sampledColorChannels.test(i)) {
      continue;
    }
    for (const int slot : {i, i + GX_ALPHA0}) {
      const auto& ccc = g_gxState.colorChannelConfig[slot];
      key.channelRegs[slot] = static_cast<u8>((ccc.lightingEnabled && ccc.ambSrc == GX_SRC_REG ? 1u : 0u) |
                                              (ccc.matSrc == GX_SRC_REG ? 2u : 0u));
    }
  }
  return key;
}

// The texture size/bias section, computed exactly as fill_uniform writes it. Texture
// bindings can change without DirtyUniform (a new texture object with the same bind
// group, a replacement finishing), so reuse compares this section itself.
struct TextureSection {
  std::array<Vec4<float>, MaxTextures> values{};
  u32 count = 0;

  bool operator==(const TextureSection& rhs) const noexcept {
    return count == rhs.count && std::memcmp(values.data(), rhs.values.data(), count * sizeof(Vec4<float>)) == 0;
  }
};
static_assert(sizeof(Vec4<float>) == 16);

TextureSection texture_section(const ShaderInfo& info) noexcept {
  TextureSection section;
  for (int i = 0; i < info.sampledTextures.size(); ++i) {
    if (info.sampledTextures.test(i)) {
      section.values[section.count++] = texture_size_bias(get_texture(static_cast<GXTexMapID>(i)));
    }
  }
  return section;
}

// The last block filled in full under Verify or On, and what it was filled from.
struct ReuseTracker {
  bool valid = false;
  uint64_t epoch = 0;
  gfx::Range range{};
  UniformLayoutKey key{};
  TextureSection textures{};
  bool bytesValid = false; // Verify keeps a copy of the bytes to compare against
  ByteBuffer bytes;
};
ReuseTracker g_tracker;
// Moves whenever the uniform buffer the ranges point into starts over.
uint64_t g_uniformEpoch = 1;

std::atomic<uint8_t> g_reuseMode{AURORA_UNIFORM_REUSE_OFF};
std::atomic_bool g_reuseDisabled{false};

struct ReuseCounters {
  std::atomic<uint64_t> fullBuilds{0};
  std::atomic<uint64_t> repeatHits{0};
  std::atomic<uint64_t> candidates{0};
  std::atomic<uint64_t> reuses{0};
  std::atomic<uint64_t> checks{0};
  std::atomic<uint64_t> mismatches{0};
  std::atomic<uint32_t> lastMismatchOffset{0};
  std::atomic<uint32_t> lastMismatchSection{0};
  std::atomic<uint32_t> lastMismatchSize{0};
};
ReuseCounters g_reuseStats;

// The FIFO thread is the only writer, so a plain load/store is enough (no
// read-modify-write on the per-draw path); readers only ever see whole values.
void bump(std::atomic<uint64_t>& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

ByteBuffer& scratch_block() noexcept {
  static ByteBuffer buf;
  return buf;
}

gfx::Range push_full(const ByteBuffer& block) noexcept {
  bool repeated = false;
  const gfx::Range range = g_uniformRepeat.push(block, repeated);
  bump(g_reuseStats.fullBuilds);
  if (repeated) {
    bump(g_reuseStats.repeatHits);
  }
  return range;
}

constexpr uint32_t kLoggedMismatches = 8;

void report_mismatch(const ShaderInfo& info, const ByteBuffer& correct, const ByteBuffer& reused) noexcept {
  const size_t common = std::min(correct.size(), reused.size());
  size_t offset = 0;
  while (offset < common && correct.data()[offset] == reused.data()[offset]) {
    ++offset;
  }
  // Fill once more with section marks to name the section; rare, so cost is moot.
  SectionMarks marks;
  ByteBuffer marked;
  fill_uniform(marked, info, marks);
  const auto section = marks.at(offset);

  bump(g_reuseStats.mismatches);
  g_reuseStats.lastMismatchOffset.store(static_cast<uint32_t>(offset), std::memory_order_relaxed);
  g_reuseStats.lastMismatchSection.store(static_cast<uint32_t>(section), std::memory_order_relaxed);
  g_reuseStats.lastMismatchSize.store(static_cast<uint32_t>(correct.size()), std::memory_order_relaxed);
  const bool wasDisabled = g_reuseDisabled.exchange(true, std::memory_order_relaxed);

  const uint64_t count = g_reuseStats.mismatches.load(std::memory_order_relaxed);
  if (count <= kLoggedMismatches) {
    Log.warn("uniform reuse: Verify mismatch #{} in section {} at byte {} (correct block {} B, reused block {} B); "
             "key: lineMode={} tevRegs={:#x} lighting={} channels={:#x} kcolors={:#x} ptTex={} fog={} indMtx={} "
             "textures={:#x}. The correct block was drawn{}",
             count, kUniformSectionNames[static_cast<size_t>(section)], offset, correct.size(), reused.size(),
             static_cast<u32>(info.lineMode), info.loadsTevReg.to_ulong(), info.lightingEnabled,
             info.sampledColorChannels.to_ulong(), info.sampledKColors.to_ulong(), info.usesPTTexMtx.any(),
             info.usesFog, info.usedIndTexMtxs.any(), info.sampledTextures.to_ulong(),
             wasDisabled ? "." : "; reuse is disabled for the rest of this run.");
  }
}
} // namespace

void forget_last_uniform() noexcept {
  g_uniformRepeat.forget();
  ++g_uniformEpoch;
}

void invalidate_uniform_reuse() noexcept { g_tracker.valid = false; }

void testing::rearm_uniform_reuse() noexcept {
  g_tracker.valid = false;
  g_tracker.bytesValid = false;
  g_reuseDisabled.store(false, std::memory_order_relaxed);
}

gfx::Range build_uniform(const ShaderInfo& info) noexcept {
  ZoneScoped;
  auto& buf = scratch_block();
  buf.clear();
  fill_uniform(buf, info, NoSectionMarks{});
  return push_full(buf);
}

gfx::Range rebuild_uniform(const ShaderInfo& info, bool pipelineOnly, gfx::Range current) noexcept {
  const auto mode = g_reuseMode.load(std::memory_order_relaxed);
  if (mode != AURORA_UNIFORM_REUSE_VERIFY && mode != AURORA_UNIFORM_REUSE_ON) {
    // Off: the H34 path. The tracker no longer describes the block in use.
    g_tracker.valid = false;
    return build_uniform(info);
  }
  ZoneScoped;

  const UniformLayoutKey key = layout_key(info);
  std::optional<TextureSection> textures;
  bool candidate = false;
  // Every condition is required; see the plan document for why each one is.
  if (pipelineOnly && current.size != 0 && !g_reuseDisabled.load(std::memory_order_relaxed) && g_tracker.valid &&
      g_tracker.epoch == g_uniformEpoch && g_tracker.range == current && key == g_tracker.key) {
    textures = texture_section(info);
    candidate = *textures == g_tracker.textures;
  }
  if (candidate) {
    bump(g_reuseStats.candidates);
    if (mode == AURORA_UNIFORM_REUSE_ON) {
      bump(g_reuseStats.reuses);
      return current;
    }
  }

  auto& buf = scratch_block();
  buf.clear();
  fill_uniform(buf, info, NoSectionMarks{});
  if (candidate && g_tracker.bytesValid) {
    bump(g_reuseStats.checks);
    if (buf.size() != g_tracker.bytes.size() || std::memcmp(buf.data(), g_tracker.bytes.data(), buf.size()) != 0) {
      report_mismatch(info, buf, g_tracker.bytes);
    }
  }
  // Whatever the comparison said, the block drawn is the one just filled.
  const gfx::Range range = push_full(buf);

  g_tracker.valid = true;
  g_tracker.epoch = g_uniformEpoch;
  g_tracker.range = range;
  g_tracker.key = key;
  g_tracker.textures = textures ? *textures : texture_section(info);
  g_tracker.bytesValid = mode == AURORA_UNIFORM_REUSE_VERIFY;
  if (g_tracker.bytesValid) {
    g_tracker.bytes.clear();
    g_tracker.bytes.append(buf.data(), buf.size());
  }
  return range;
}
} // namespace aurora::gx

extern "C" void aurora_gx_set_uniform_reuse_mode(AuroraUniformReuseMode mode) {
  const auto value = static_cast<uint8_t>(mode);
  aurora::gx::g_reuseMode.store(value <= AURORA_UNIFORM_REUSE_ON ? value : AURORA_UNIFORM_REUSE_OFF,
                                std::memory_order_relaxed);
}

extern "C" AuroraUniformReuseMode aurora_gx_get_uniform_reuse_mode(void) {
  return static_cast<AuroraUniformReuseMode>(aurora::gx::g_reuseMode.load(std::memory_order_relaxed));
}

extern "C" void aurora_gx_uniform_stats(AuroraUniformStats* out) {
  if (out == nullptr) {
    return;
  }
  const auto& c = aurora::gx::g_reuseStats;
  constexpr auto r = std::memory_order_relaxed;
  *out = {
      .full_builds = c.fullBuilds.load(r),
      .repeat_hits = c.repeatHits.load(r),
      .reuse_candidates = c.candidates.load(r),
      .reuses = c.reuses.load(r),
      .verification_checks = c.checks.load(r),
      .verification_mismatches = c.mismatches.load(r),
      .disabled_after_mismatch = aurora::gx::g_reuseDisabled.load(r) ? 1u : 0u,
      .last_mismatch_offset = c.lastMismatchOffset.load(r),
      .last_mismatch_section = c.lastMismatchSection.load(r),
      .last_mismatch_size = c.lastMismatchSize.load(r),
  };
}

extern "C" const char* aurora_gx_uniform_section_name(uint32_t section) {
  return section < aurora::gx::kUniformSectionNames.size() ? aurora::gx::kUniformSectionNames[section] : "unknown";
}
