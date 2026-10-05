// aleks2 (H35): uniform block reuse when only the pipeline changed.
//
// Two levels:
//  - Direct: rebuild_uniform() on this thread with ShaderInfos from the real
//    build_shader_info(), checked against an oracle that lays the block out from the
//    GX state independently of shader_info.cpp. Includes randomized sequences in Off,
//    Verify and On.
//  - End to end: GX API calls through the real FIFO worker and command processor, so
//    the "pipeline-only" decision of push_gx_draw is exercised with real dirty flags.
// The bytes drawn must always be exactly the oracle's; reuse may only change how many
// blocks get filled and pushed.

#include "gx_test_common.hpp"
#include "gx/pipeline.hpp"
#include "gx/shader_info.hpp"

#include <aurora/tuning.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <optional>
#include <random>

namespace aurora::gfx {
extern std::vector<std::pair<Range, std::vector<uint8_t>>> g_testUniformPushes;
extern gx::DrawData g_testLastDraw;
extern bool g_testNormalAttachment;
} // namespace aurora::gfx
namespace aurora::gx {
extern PipelineConfig g_testPipelineConfig;
}

using aurora::gfx::Range;
using aurora::gx::g_gxState;
using aurora::gx::ShaderConfig;
using aurora::gx::ShaderInfo;
namespace gx = aurora::gx;
namespace gfx = aurora::gfx;

namespace {

// ---- Oracle: the block layout, written from the GX state without shader_info.cpp ----

f32 oracle_tex_offset(GXTexOffset offs) {
  switch (offs) {
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
  default:
    return 0.f;
  }
}

std::vector<u8> oracle_block(const ShaderInfo& info) {
  std::vector<u8> out;
  const auto put = [&](const auto& value) {
    const auto* p = reinterpret_cast<const u8*>(&value);
    out.insert(out.end(), p, p + sizeof(value));
  };
  const auto& st = g_gxState;
  put(st.renderViewport.width);
  put(st.renderViewport.height);
  put(st.logicalViewport.width);
  put(st.logicalViewport.height);
  if (info.lineMode != 0) {
    u32 mask = 0;
    for (int i = 0; i < gx::MaxTexCoord; ++i) {
      if (info.lineMode == 3 ? st.texCoordScales[i].pointOffset : st.texCoordScales[i].lineOffset) {
        mask |= 1u << i;
      }
    }
    if (info.lineMode == 3) {
      put(static_cast<f32>(st.pointSize) / 6.f);
      put(1.0f);
      put(oracle_tex_offset(st.pointTexOffset));
    } else {
      put(static_cast<f32>(st.lineWidth) / 6.f);
      put(st.lineHalfAspect ? 0.5f : 1.f);
      put(oracle_tex_offset(st.lineTexOffset));
    }
    put(mask);
  }
  auto proj = st.proj;
  if constexpr (gx::UseReversedZ) {
    proj.m2 = proj.m2 * aurora::Vec4{-1.f, -1.f, -1.f, -1.f};
  } else {
    proj.m2 = proj.m2 + proj.m3;
  }
  put(proj);
  for (const auto& m : st.pnMtx) {
    put(m.pos);
  }
  for (const auto& m : st.texMtxs) {
    put(m);
  }
  for (const auto& m : st.pnMtx) {
    put(m.nrm);
  }
  for (size_t i = 0; i < info.loadsTevReg.size(); ++i) {
    if (info.loadsTevReg.test(i)) {
      put(st.colorRegs[i]);
    }
  }
  if (info.lightingEnabled) {
    put(st.lights);
    for (int i = 0; i < 4; ++i) {
      put(static_cast<u32>(st.colorChannelState[i].lightMask.to_ulong()));
    }
  }
  for (size_t i = 0; i < info.sampledColorChannels.size(); ++i) {
    if (!info.sampledColorChannels.test(i)) {
      continue;
    }
    for (const size_t slot : {i, i + GX_ALPHA0}) {
      const auto& cfg = st.colorChannelConfig[slot];
      const auto& cs = st.colorChannelState[slot];
      if (cfg.lightingEnabled && cfg.ambSrc == GX_SRC_REG) {
        put(cs.ambColor);
      }
      if (cfg.matSrc == GX_SRC_REG) {
        put(cs.matColor);
      }
    }
  }
  for (size_t i = 0; i < info.sampledKColors.size(); ++i) {
    if (info.sampledKColors.test(i)) {
      put(st.kcolors[i]);
    }
  }
  if (info.usesPTTexMtx.any()) {
    put(st.ptTexMtxs);
  }
  if (info.usesFog) {
    const auto& f = st.fog;
    gx::Fog fog{
        .color = f.color,
        .a = f.a,
        .b = f.b,
        .c = f.c,
        .rangeCenter = ((static_cast<float>(f.rangeCenter) - st.logicalViewport.left) /
                        std::max(st.logicalViewport.width, 1.f)) *
                               2.f -
                           1.f + (st.renderViewport.left / std::max(st.renderViewport.width, 1.f)) * 2.f,
    };
    for (u32 i = 0; i < f.rangeK.size(); ++i) {
      const u32 source = (i & ~1u) | (1u - (i & 1u));
      fog.rangeK[i / 4][i % 4] = static_cast<float>(f.rangeK[source]) / 64.f;
    }
    fog.rangeK[2][2] = fog.rangeK[2][1];
    fog.rangeK[2][3] = fog.rangeK[2][1];
    put(fog);
  }
  for (const auto& scale : st.texCoordScales) {
    put(aurora::Vec4<float>{static_cast<f32>(scale.scaleS) + 1.0f, static_cast<f32>(scale.scaleT) + 1.0f, 0.f, 0.f});
  }
  if (info.usedIndTexMtxs.any()) {
    for (const auto& m : st.indTexMtxs) {
      put(aurora::Vec4<float>{m.mtx.m0.x, m.mtx.m0.y, m.mtx.m1.x, m.mtx.m1.y});
      put(aurora::Vec4<float>{m.mtx.m2.x, m.mtx.m2.y, std::exp2f(m.scaleExp), 0.f});
    }
  }
  for (size_t i = 0; i < info.sampledTextures.size(); ++i) {
    if (!info.sampledTextures.test(i)) {
      continue;
    }
    const auto& tex = st.textures[i];
    // No texture refs in these tests, so no arbitrary-mip bias term.
    put(aurora::Vec4<float>{static_cast<float>(tex.texObj.width()), static_cast<float>(tex.texObj.height()),
                            tex.texObj.lod_bias(), 0.f});
  }
  return out;
}

const std::vector<u8>* pushed_bytes(Range range) {
  for (const auto& [r, bytes] : gfx::g_testUniformPushes) {
    if (r == range) {
      return &bytes;
    }
  }
  return nullptr;
}

AuroraUniformStats uniform_stats() {
  AuroraUniformStats s{};
  aurora_gx_uniform_stats(&s);
  return s;
}

float& mat_at(aurora::Mat3x4<float>& m, size_t row, size_t col) {
  return (row == 0 ? m.m0 : row == 1 ? m.m1 : m.m2)[col];
}

// ---- Shader configurations ------------------------------------------------------

ShaderConfig base_config() {
  ShaderConfig c{};
  c.tevStageCount = 1;
  auto& st = c.tevStages[0];
  st.colorPass = {GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC};
  st.alphaPass = {GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_RASA};
  st.colorOp.outReg = GX_TEVPREV;
  st.alphaOp.outReg = GX_TEVPREV;
  st.channelId = GX_COLOR0A0;
  return c;
}

// Same uniform layout as base_config, different pipeline.
ShaderConfig base_config_other_pipeline() {
  ShaderConfig c = base_config();
  c.tevStages[0].colorOp.clamp = !c.tevStages[0].colorOp.clamp;
  c.alphaCompare.ref0 ^= 0x40;
  return c;
}

ShaderConfig konst_config(GXTevKColorSel sel) {
  ShaderConfig c = base_config();
  c.tevStages[0].colorPass.c = GX_CC_KONST;
  c.tevStages[0].kcSel = sel;
  return c;
}

ShaderConfig texture_config(GXTexMapID map) {
  ShaderConfig c = base_config();
  c.tevStages[0].colorPass.b = GX_CC_TEXC;
  c.tevStages[0].texMapId = map;
  c.tevStages[0].texCoordId = GX_TEXCOORD0;
  return c;
}

ShaderConfig fog_config() {
  ShaderConfig c = base_config();
  c.fogType = GX_FOG_PERSP_LIN;
  return c;
}

ShaderConfig lit_config() {
  ShaderConfig c = base_config();
  c.colorChannels[GX_COLOR0].lightingEnabled = true;
  c.colorChannels[GX_COLOR0].ambSrc = GX_SRC_REG;
  c.colorChannels[GX_COLOR0].matSrc = GX_SRC_REG;
  return c;
}

ShaderConfig tev_reg_config() {
  ShaderConfig c = base_config();
  c.tevStages[0].colorPass.a = GX_CC_C0;
  return c;
}

ShaderConfig pt_tex_config() {
  ShaderConfig c = texture_config(GX_TEXMAP0);
  c.tcgs[0].postMtx = GX_PTTEXMTX0;
  return c;
}

ShaderConfig ind_config() {
  ShaderConfig c = texture_config(GX_TEXMAP0);
  c.numIndStages = 1;
  c.indStages[0].texCoordId = GX_TEXCOORD1;
  c.indStages[0].texMapId = GX_TEXMAP1;
  c.tevStages[0].indTexStage = GX_INDTEXSTAGE0;
  c.tevStages[0].indTexMtxId = GX_ITM_0;
  return c;
}

ShaderConfig line_config(u8 lineMode) {
  ShaderConfig c = base_config();
  c.lineMode = lineMode;
  return c;
}

// The live channel configuration fill_uniform reads follows the pipeline's, as
// populate_pipeline_config would have it.
ShaderInfo info_for(const ShaderConfig& config) {
  for (size_t i = 0; i < config.colorChannels.size(); ++i) {
    g_gxState.colorChannelConfig[i] = config.colorChannels[i];
  }
  return gx::build_shader_info(config);
}

void seed_state(std::mt19937& rng) {
  std::uniform_real_distribution<float> f(-4.f, 4.f);
  auto& st = g_gxState;
  st.renderViewport = {0.f, 0.f, 1280.f, 720.f, 0.f, 1.f};
  st.logicalViewport = {0.f, 0.f, 640.f, 480.f, 0.f, 1.f};
  for (auto& m : st.pnMtx) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c) {
        mat_at(m.pos, r, c) = f(rng);
      }
    }
  }
  for (auto& k : st.kcolors) {
    k = {f(rng), f(rng), f(rng), 1.f};
  }
  for (auto& c : st.colorRegs) {
    c = {f(rng), f(rng), f(rng), f(rng)};
  }
  for (auto& light : st.lights) {
    light.pos = {f(rng), f(rng), f(rng)};
    light.color = {0.5f, 0.25f, 1.f, 1.f};
  }
  st.fog.color = {0.1f, 0.2f, 0.3f, 1.f};
  st.fog.a = 1.5f;
  st.fog.c = 0.25f;
  for (auto& k : st.fog.rangeK) {
    k = static_cast<u16>(rng() & 0xFFF);
  }
  for (auto& t : st.texCoordScales) {
    t.scaleS = static_cast<u16>(rng() & 0xFF);
    t.scaleT = static_cast<u16>(rng() & 0xFF);
  }
  for (auto& m : st.indTexMtxs) {
    m.mtx.m0 = {f(rng), f(rng)};
    m.scaleExp = static_cast<s8>(static_cast<int>(rng() % 9) - 4);
  }
  for (auto& t : st.textures) {
    t.texObj.mWidth = 64;
    t.texObj.mHeight = 32;
  }
}

// One "draw" at the direct level: what refresh_uniform does when the uniform is dirty.
struct DirectDraws {
  Range current{};
  Range draw(const ShaderInfo& info, bool pipelineOnly) {
    current = gx::rebuild_uniform(info, pipelineOnly, current);
    return current;
  }
  void end_frame() {
    current = {};
    gx::invalidate_uniform_reuse();
    gx::forget_last_uniform();
  }
};

class UniformReuseDirect : public ::testing::Test {
protected:
  void SetUp() override {
    g_gxState = gx::GXState{};
    gfx::g_testUniformPushes.clear();
    gx::testing::rearm_uniform_reuse();
    gx::forget_last_uniform();
    std::mt19937 rng{1234};
    seed_state(rng);
  }
  void TearDown() override {
    aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_OFF);
    gx::testing::rearm_uniform_reuse();
  }

  void expect_drawn(Range range, const ShaderInfo& info) {
    const auto* bytes = pushed_bytes(range);
    ASSERT_NE(bytes, nullptr);
    EXPECT_EQ(*bytes, oracle_block(info));
  }
};

} // namespace

// ---- Direct level ------------------------------------------------------------------

TEST_F(UniformReuseDirect, ModeApiDefaultsOffAndClampsBadValues) {
  EXPECT_EQ(aurora_gx_get_uniform_reuse_mode(), AURORA_UNIFORM_REUSE_OFF);
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_VERIFY);
  EXPECT_EQ(aurora_gx_get_uniform_reuse_mode(), AURORA_UNIFORM_REUSE_VERIFY);
  aurora_gx_set_uniform_reuse_mode(static_cast<AuroraUniformReuseMode>(9));
  EXPECT_EQ(aurora_gx_get_uniform_reuse_mode(), AURORA_UNIFORM_REUSE_OFF);
  EXPECT_STREQ(aurora_gx_uniform_section_name(0), "viewport");
  EXPECT_STREQ(aurora_gx_uniform_section_name(14), "texture_sizes");
  EXPECT_STREQ(aurora_gx_uniform_section_name(15), "unknown");
  aurora_gx_uniform_stats(nullptr);
}

TEST_F(UniformReuseDirect, OracleMatchesEveryLayout) {
  // The oracle and fill_uniform agree on every layout used below, in Off.
  DirectDraws d;
  for (const auto& config :
       {base_config(), konst_config(GX_TEV_KCSEL_K2), texture_config(GX_TEXMAP3), fog_config(), lit_config(),
        tev_reg_config(), pt_tex_config(), ind_config(), line_config(1), line_config(3)}) {
    const ShaderInfo info = info_for(config);
    expect_drawn(d.draw(info, false), info);
  }
}

TEST_F(UniformReuseDirect, OffIsTheH34Path) {
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_OFF);
  const auto before = uniform_stats();
  DirectDraws d;
  const ShaderInfo a = info_for(base_config());
  const ShaderInfo b = info_for(base_config_other_pipeline());
  const Range first = d.draw(a, false);
  const Range second = d.draw(b, true); // would be a candidate in the other modes
  const auto after = uniform_stats();
  // Every dirty uniform is filled; identical bytes are pushed once (the repeat check
  // that already existed) and nothing is ever counted as a candidate or reused.
  EXPECT_EQ(after.full_builds - before.full_builds, 2u);
  EXPECT_EQ(after.repeat_hits - before.repeat_hits, 1u);
  EXPECT_EQ(after.reuse_candidates, before.reuse_candidates);
  EXPECT_EQ(after.reuses, before.reuses);
  EXPECT_EQ(first, second);
  EXPECT_EQ(gfx::g_testUniformPushes.size(), 1u);
  expect_drawn(second, b);
}

TEST_F(UniformReuseDirect, PipelineChangeWithIdenticalUniformsIsReused) {
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_ON);
  DirectDraws d;
  const ShaderInfo a = info_for(base_config());
  const ShaderInfo b = info_for(base_config_other_pipeline());
  const Range first = d.draw(a, false);
  const auto before = uniform_stats();
  const Range second = d.draw(b, true);
  const auto after = uniform_stats();
  EXPECT_EQ(second, first);
  EXPECT_EQ(after.reuse_candidates - before.reuse_candidates, 1u);
  EXPECT_EQ(after.reuses - before.reuses, 1u);
  EXPECT_EQ(after.full_builds, before.full_builds);
  expect_drawn(second, b);
}

TEST_F(UniformReuseDirect, PipelineChangeWithDifferentUniformsIsRebuilt) {
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_ON);
  DirectDraws d;
  const std::array configs{konst_config(GX_TEV_KCSEL_K1), fog_config(),  lit_config(),       tev_reg_config(),
                           pt_tex_config(),                ind_config(),  line_config(1),     line_config(3),
                           texture_config(GX_TEXMAP2),     konst_config(GX_TEV_KCSEL_K3)};
  for (const auto& config : configs) {
    d.draw(info_for(base_config()), false);
    const auto before = uniform_stats();
    const ShaderInfo next = info_for(config);
    const Range range = d.draw(next, true);
    const auto after = uniform_stats();
    EXPECT_EQ(after.reuses, before.reuses);
    EXPECT_EQ(after.full_builds - before.full_builds, 1u);
    expect_drawn(range, next);
  }
}

TEST_F(UniformReuseDirect, LiveChannelConfigIsPartOfTheSignature) {
  // matSrc decides whether the material colour is in the block; it lives in the GX
  // state, not in ShaderInfo.
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_ON);
  DirectDraws d;
  ShaderConfig regs = base_config();
  regs.colorChannels[GX_COLOR0].matSrc = GX_SRC_REG;
  ShaderConfig vtx = base_config();
  vtx.colorChannels[GX_COLOR0].matSrc = GX_SRC_VTX;
  d.draw(info_for(regs), false);
  const auto before = uniform_stats();
  const ShaderInfo next = info_for(vtx);
  const Range range = d.draw(next, true);
  EXPECT_EQ(uniform_stats().reuses, before.reuses);
  expect_drawn(range, next);
}

TEST_F(UniformReuseDirect, SampledTextureChangeIsCaught) {
  // A texture change that leaves the bind group (and so DirtyUniform) alone must still
  // not reuse a block with the old size/bias.
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_ON);
  DirectDraws d;
  const ShaderInfo a = info_for(texture_config(GX_TEXMAP0));
  ShaderConfig other = texture_config(GX_TEXMAP0);
  other.tevStages[0].colorOp.clamp = !other.tevStages[0].colorOp.clamp;
  const ShaderInfo b = info_for(other);
  d.draw(a, false);

  for (int variant = 0; variant < 2; ++variant) {
    if (variant == 0) {
      g_gxState.textures[0].texObj.mWidth = 128; // size
    } else {
      g_gxState.textures[0].texObj.mode0 ^= 0x10u << 9; // LOD bias
    }
    const auto before = uniform_stats();
    const Range range = d.draw(variant == 0 ? b : a, true);
    EXPECT_EQ(uniform_stats().reuses, before.reuses) << variant;
    expect_drawn(range, variant == 0 ? b : a);
  }
  // Unchanged texture: reused.
  const auto before = uniform_stats();
  d.draw(b, true);
  EXPECT_EQ(uniform_stats().reuses - before.reuses, 1u);
}

TEST_F(UniformReuseDirect, StateWritesAreNeverPipelineOnly) {
  // Matrices, lights, fog, KColors, indirect and post-transform matrices are guarded by
  // DirtyUniform; a draw whose uniform is dirty from state is never a candidate, in any
  // layout that reads the changed state.
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_ON);
  DirectDraws d;
  struct Case {
    ShaderConfig config;
    void (*change)();
  };
  const std::array cases{
      Case{base_config(), [] { mat_at(g_gxState.pnMtx[3].pos, 0, 1) += 1.f; }},
      Case{lit_config(), [] { g_gxState.lights[2].color[0] = 0.75f; }},
      Case{fog_config(), [] { g_gxState.fog.c += 1.f; }},
      Case{konst_config(GX_TEV_KCSEL_K0), [] { g_gxState.kcolors[0][1] = 0.5f; }},
      Case{ind_config(), [] { g_gxState.indTexMtxs[0].mtx.m1.x = 0.125f; }},
      Case{pt_tex_config(), [] { mat_at(g_gxState.ptTexMtxs[0], 0, 0) = 3.f; }},
      Case{base_config(), [] { g_gxState.texCoordScales[1].scaleS += 1; }},
  };
  for (const auto& c : cases) {
    const ShaderInfo info = info_for(c.config);
    d.draw(info, false);
    c.change();
    const auto before = uniform_stats();
    const Range range = d.draw(info, false);
    EXPECT_EQ(uniform_stats().reuse_candidates, before.reuse_candidates);
    expect_drawn(range, info);
  }
}

TEST_F(UniformReuseDirect, FirstDrawAndFrameBoundariesNeverReuse) {
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_ON);
  DirectDraws d;
  const ShaderInfo a = info_for(base_config());
  const ShaderInfo b = info_for(base_config_other_pipeline());
  // First draw: nothing to reuse.
  auto before = uniform_stats();
  d.draw(a, true);
  EXPECT_EQ(uniform_stats().reuse_candidates, before.reuse_candidates);

  // End of frame: the cache drops its range and the uniform buffer starts over.
  d.end_frame();
  before = uniform_stats();
  d.draw(b, true);
  EXPECT_EQ(uniform_stats().reuse_candidates, before.reuse_candidates);

  // Start of frame alone (buffer starts over, the cache still names the old range).
  gx::forget_last_uniform();
  before = uniform_stats();
  d.draw(a, true);
  EXPECT_EQ(uniform_stats().reuse_candidates, before.reuse_candidates);

  // Draw cache cleared alone.
  gx::invalidate_uniform_reuse();
  before = uniform_stats();
  d.draw(b, true);
  EXPECT_EQ(uniform_stats().reuse_candidates, before.reuse_candidates);

  // And a range other than the one last built is never taken for it.
  d.current.offset += 256;
  before = uniform_stats();
  d.draw(a, true);
  EXPECT_EQ(uniform_stats().reuse_candidates, before.reuse_candidates);
}

TEST_F(UniformReuseDirect, VerifyChecksEveryCandidateAndAlwaysDrawsTheFullBlock) {
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_VERIFY);
  DirectDraws d;
  const ShaderInfo a = info_for(base_config());
  const ShaderInfo b = info_for(base_config_other_pipeline());
  d.draw(a, false);
  const auto before = uniform_stats();
  for (int i = 0; i < 10; ++i) {
    const Range range = d.draw(i % 2 ? a : b, true);
    expect_drawn(range, i % 2 ? a : b);
  }
  const auto after = uniform_stats();
  EXPECT_EQ(after.reuse_candidates - before.reuse_candidates, 10u);
  EXPECT_EQ(after.verification_checks - before.verification_checks, 10u);
  EXPECT_EQ(after.verification_mismatches, before.verification_mismatches);
  EXPECT_EQ(after.reuses, before.reuses);
  EXPECT_EQ(after.full_builds - before.full_builds, 10u);
  EXPECT_EQ(after.disabled_after_mismatch, 0u);
}

TEST_F(UniformReuseDirect, ArtificialMismatchFallsBackSafelyAndDisablesReuse) {
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_VERIFY);
  DirectDraws d;
  const ShaderInfo a = info_for(konst_config(GX_TEV_KCSEL_K0));
  ShaderConfig otherConfig = konst_config(GX_TEV_KCSEL_K0);
  otherConfig.tevStages[0].colorOp.clamp = !otherConfig.tevStages[0].colorOp.clamp;
  const ShaderInfo b = info_for(otherConfig);
  const Range first = d.draw(a, false);

  // A K colour changed with no dirty flag: exactly the bug class Verify exists for.
  g_gxState.kcolors[0][2] = 0.875f;
  const auto before = uniform_stats();
  const Range range = d.draw(b, true);
  const auto after = uniform_stats();
  EXPECT_EQ(after.verification_checks - before.verification_checks, 1u);
  EXPECT_EQ(after.verification_mismatches - before.verification_mismatches, 1u);
  EXPECT_EQ(after.disabled_after_mismatch, 1u);
  EXPECT_STREQ(aurora_gx_uniform_section_name(after.last_mismatch_section), "kcolors");
  EXPECT_EQ(after.last_mismatch_size, oracle_block(b).size());
  // The block drawn is the correct one, never the stale one.
  EXPECT_NE(range, first);
  expect_drawn(range, b);

  // Disabled for the rest of the run: neither Verify nor On treat anything as a
  // candidate any more, so nothing is reused.
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_ON);
  const auto onBefore = uniform_stats();
  const Range again = d.draw(a, true);
  const auto onAfter = uniform_stats();
  EXPECT_EQ(onAfter.reuse_candidates, onBefore.reuse_candidates);
  EXPECT_EQ(onAfter.reuses, onBefore.reuses);
  EXPECT_EQ(onAfter.full_builds - onBefore.full_builds, 1u);
  EXPECT_EQ(onAfter.disabled_after_mismatch, 1u);
  expect_drawn(again, a);
}

TEST_F(UniformReuseDirect, ModeSwitchesMidFrameStayCorrect) {
  DirectDraws d;
  const std::array infos{info_for(base_config()), info_for(base_config_other_pipeline()),
                         info_for(konst_config(GX_TEV_KCSEL_K1))};
  const std::array modes{AURORA_UNIFORM_REUSE_OFF, AURORA_UNIFORM_REUSE_ON, AURORA_UNIFORM_REUSE_VERIFY,
                         AURORA_UNIFORM_REUSE_ON, AURORA_UNIFORM_REUSE_OFF, AURORA_UNIFORM_REUSE_VERIFY};
  const auto before = uniform_stats();
  for (int i = 0; i < 60; ++i) {
    aurora_gx_set_uniform_reuse_mode(modes[i % modes.size()]);
    const auto& info = infos[(i * 7) % infos.size()];
    const Range range = d.draw(info, i != 0);
    expect_drawn(range, info);
  }
  EXPECT_EQ(uniform_stats().verification_mismatches, before.verification_mismatches);
}

// Randomized: state writes (always reported as such), texture changes that do not
// dirty the uniform, pipeline changes, frame ends and mode switches. Whatever is drawn
// must be exactly the oracle's block for the state and shader at that moment.
TEST_F(UniformReuseDirect, RandomizedSequencesDrawExactlyTheOracleBlock) {
  const std::array configs{base_config(),        base_config_other_pipeline(), konst_config(GX_TEV_KCSEL_K0),
                           konst_config(GX_TEV_KCSEL_K2), texture_config(GX_TEXMAP0), texture_config(GX_TEXMAP1),
                           fog_config(),         lit_config(),                 tev_reg_config(),
                           pt_tex_config(),      ind_config(),                 line_config(1),
                           line_config(3)};
  for (const auto mode : {AURORA_UNIFORM_REUSE_OFF, AURORA_UNIFORM_REUSE_VERIFY, AURORA_UNIFORM_REUSE_ON}) {
    gx::testing::rearm_uniform_reuse();
    aurora_gx_set_uniform_reuse_mode(mode);
    std::mt19937 rng{static_cast<uint32_t>(42 + mode)};
    DirectDraws d;
    gfx::g_testUniformPushes.clear();
    gx::forget_last_uniform();
    const auto before = uniform_stats();
    size_t current = 0;
    uint64_t draws = 0;
    for (int i = 0; i < 4000; ++i) {
      bool stateDirty = false;
      bool pipelineChanged = false;
      switch (rng() % 8) {
      case 0:
        mat_at(g_gxState.pnMtx[rng() % gx::MaxPnMtx].pos, rng() % 3, rng() % 4) = static_cast<float>(rng() % 100);
        stateDirty = true;
        break;
      case 1:
        g_gxState.kcolors[rng() % 4][rng() % 4] = static_cast<float>(rng() % 256) / 255.f;
        stateDirty = true;
        break;
      case 2:
        g_gxState.textures[rng() % 2].texObj.mWidth = 16u << (rng() % 4); // no dirty flag
        break;
      case 3:
        d.end_frame();
        break;
      default:
        current = rng() % configs.size();
        pipelineChanged = true;
        break;
      }
      if (!stateDirty && !pipelineChanged && d.current.size != 0) {
        continue; // the uniform would not be dirty: no refresh happens
      }
      const ShaderInfo info = info_for(configs[current]);
      const Range range = d.draw(info, pipelineChanged && !stateDirty);
      ++draws;
      const auto* bytes = pushed_bytes(range);
      ASSERT_NE(bytes, nullptr);
      ASSERT_EQ(*bytes, oracle_block(info)) << "mode " << mode << " step " << i;
    }
    const auto after = uniform_stats();
    EXPECT_EQ(after.verification_mismatches, before.verification_mismatches);
    if (mode == AURORA_UNIFORM_REUSE_ON) {
      EXPECT_GT(after.reuses, before.reuses);
      EXPECT_EQ(after.full_builds - before.full_builds + after.reuses - before.reuses, draws);
    } else if (mode == AURORA_UNIFORM_REUSE_VERIFY) {
      EXPECT_GT(after.verification_checks, before.verification_checks);
      EXPECT_EQ(after.full_builds - before.full_builds, draws);
    } else {
      EXPECT_EQ(after.reuse_candidates, before.reuse_candidates);
      EXPECT_EQ(after.full_builds - before.full_builds, draws);
    }
  }
}

// ---- End to end through the FIFO worker and the command processor ---------------

namespace {
class UniformReuseFifo : public GXFifoTest {
protected:
  void SetUp() override {
    GXFifoTest::SetUp();
    gx::testing::rearm_uniform_reuse();
    gfx::g_testNormalAttachment = false;
    gx::g_testPipelineConfig = gx::PipelineConfig{};
    gx::g_testPipelineConfig.msaaSamples = gfx::sample_count();
    set_shader(base_config());
    aurora::gx::fifo::init();
    aurora::gx::fifo::begin_frame();
    aurora::gx::fifo::end_frame(); // drop any draw cache range left by another test
    aurora::gx::fifo::begin_frame();
    gx::forget_last_uniform();
    gfx::g_testUniformPushes.clear();
    GXClearVtxDesc();
    GXSetVtxDesc(GX_VA_POS, GX_DIRECT);
    GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_U8, 0);
  }
  void TearDown() override {
    aurora::gx::fifo::drain();
    aurora::gx::fifo::end_frame();
    aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_OFF);
    gx::testing::rearm_uniform_reuse();
    gfx::g_testNormalAttachment = false;
    GXFifoTest::TearDown();
  }

  // Only between drains: the FIFO worker reads it while translating.
  void set_shader(const ShaderConfig& config) {
    gx::g_testPipelineConfig.shaderConfig = config;
    shader = config;
  }

  // Makes the pipeline dirty the way a material change does. The blend register value
  // alternates so the write is never filtered as redundant.
  void touch_pipeline() {
    blendToggle = !blendToggle;
    GXSetBlendMode(GX_BM_BLEND, blendToggle ? GX_BL_SRCALPHA : GX_BL_ONE, GX_BL_INVSRCALPHA, GX_LO_CLEAR);
  }

  // Draws one triangle and waits until the FIFO worker has translated it; returns the
  // uniform block that draw used.
  std::vector<u8> draw_and_read() {
    GXBegin(GX_TRIANGLES, GX_VTXFMT0, 3);
    GXPosition3u8(0, 1, 2);
    GXPosition3u8(3, 4, 5);
    GXPosition3u8(6, 7, 8);
    GXEnd();
    GXDrawDone();
    const auto* bytes = pushed_bytes(gfx::g_testLastDraw.uniformRange);
    return bytes != nullptr ? *bytes : std::vector<u8>{};
  }

  std::vector<u8> expected() {
    // fill_uniform reads the live channel configuration; populate_pipeline_config is
    // stubbed here, so compare against the shader's own (equal in these tests).
    return oracle_block(gx::build_shader_info(shader));
  }

  ShaderConfig shader{};
  bool blendToggle = false;
};

void load_pos_mtx(float tx) {
  f32 m[3][4]{{1.f, 0.f, 0.f, tx}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}};
  GXLoadPosMtxImm(m, GX_PNMTX0);
}
} // namespace

TEST_F(UniformReuseFifo, PipelineOnlyChangeIsReusedAndExact) {
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_ON);
  ASSERT_EQ(draw_and_read(), expected());
  const auto before = uniform_stats();
  for (int i = 0; i < 6; ++i) {
    set_shader(i % 2 ? base_config() : base_config_other_pipeline());
    touch_pipeline();
    ASSERT_EQ(draw_and_read(), expected()) << i;
  }
  const auto after = uniform_stats();
  EXPECT_EQ(after.reuses - before.reuses, 6u);
  EXPECT_EQ(after.full_builds, before.full_builds);
}

TEST_F(UniformReuseFifo, StateWriteWithPipelineChangeIsRebuilt) {
  // Each case: change a uniform input through the GX API *and* the pipeline in the same
  // draw. The state write sets DirtyUniform first, so this is never pipeline-only.
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_ON);
  struct Case {
    const char* name;
    ShaderConfig config;
    void (*write)(int step);
  };
  const std::array cases{
      Case{"position matrix", base_config(), [](int s) { load_pos_mtx(static_cast<float>(s)); }},
      Case{"kcolor", konst_config(GX_TEV_KCSEL_K1),
           [](int s) { GXSetTevKColor(GX_KCOLOR1, GXColor{static_cast<u8>(10 * s), 20, 30, 255}); }},
      Case{"tev register", tev_reg_config(),
           [](int s) { GXSetTevColor(GX_TEVREG0, GXColor{static_cast<u8>(5 * s), 1, 2, 3}); }},
      Case{"fog", fog_config(),
           [](int s) {
             GXSetFog(GX_FOG_PERSP_LIN, 10.f + static_cast<float>(s), 500.f, 1.f, 1000.f, GXColor{1, 2, 3, 255});
           }},
      Case{"indirect matrix", ind_config(),
           [](int s) {
             f32 m[2][3] = {{0.5f, 0.f, 0.f}, {0.f, 0.25f * static_cast<float>(s + 1), 0.f}};
             GXSetIndTexMtx(GX_ITM_0, m, 1);
           }},
      Case{"post-transform matrix", pt_tex_config(),
           [](int s) {
             f32 m[3][4]{{1.f, 0.f, 0.f, static_cast<float>(s)}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}};
             GXLoadTexMtxImm(m, GX_PTTEXMTX0, GX_MTX3x4);
           }},
      Case{"light", lit_config(),
           [](int s) {
             GXLightObj light{};
             GXInitLightAttn(&light, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f);
             GXInitLightDir(&light, 0.f, 0.f, -1.f);
             GXInitLightPos(&light, static_cast<float>(s), 2.f, 3.f);
             GXInitLightColor(&light, GXColor{200, 100, static_cast<u8>(s), 255});
             GXLoadLightObjImm(&light, GX_LIGHT0);
           }},
      Case{"channel colour", lit_config(),
           [](int s) { GXSetChanMatColor(GX_COLOR0A0, GXColor{static_cast<u8>(s * 3), 4, 5, 6}); }},
  };
  for (const auto& c : cases) {
    set_shader(c.config);
    touch_pipeline();
    ASSERT_EQ(draw_and_read(), expected()) << c.name;
    for (int step = 1; step <= 3; ++step) {
      const auto before = uniform_stats();
      // Same uniform layout, different pipeline.
      ShaderConfig variant = c.config;
      variant.alphaCompare.ref0 = static_cast<u8>(step);
      set_shader(variant);
      c.write(step);
      touch_pipeline();
      const auto bytes = draw_and_read();
      EXPECT_EQ(bytes, expected()) << c.name << " step " << step;
      EXPECT_EQ(uniform_stats().reuses, before.reuses) << c.name << " step " << step;
    }
  }
}

TEST_F(UniformReuseFifo, RenderTargetLayoutChangeReusesIdenticalBytes) {
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_VERIFY);
  ASSERT_EQ(draw_and_read(), expected());
  const auto before = uniform_stats();
  for (int i = 0; i < 4; ++i) {
    gfx::g_testNormalAttachment = !gfx::g_testNormalAttachment; // a new layout key
    ASSERT_EQ(draw_and_read(), expected()) << i;
  }
  const auto after = uniform_stats();
  EXPECT_EQ(after.verification_checks - before.verification_checks, 4u);
  EXPECT_EQ(after.verification_mismatches, before.verification_mismatches);
}

TEST_F(UniformReuseFifo, SampledTextureChangeWithoutBindGroupChangeIsRebuilt) {
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_ON);
  set_shader(texture_config(GX_TEXMAP0));
  g_gxState.textures[0].texObj.mWidth = 64;
  g_gxState.textures[0].texObj.mHeight = 64;
  touch_pipeline();
  ASSERT_EQ(draw_and_read(), expected());
  // Between drains the worker is idle; the stubbed bind groups never change, as with a
  // new texture object in an otherwise identical binding.
  g_gxState.textures[0].texObj.mWidth = 256;
  ShaderConfig other = texture_config(GX_TEXMAP0);
  other.alphaCompare.ref1 ^= 1;
  set_shader(other);
  touch_pipeline();
  const auto before = uniform_stats();
  EXPECT_EQ(draw_and_read(), expected());
  EXPECT_EQ(uniform_stats().reuses, before.reuses);
}

TEST_F(UniformReuseFifo, FrameEndInvalidatesAndFirstDrawBuilds) {
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_ON);
  ASSERT_EQ(draw_and_read(), expected());
  aurora::gx::fifo::end_frame();
  gx::forget_last_uniform();
  aurora::gx::fifo::begin_frame();
  set_shader(base_config_other_pipeline());
  touch_pipeline();
  const auto before = uniform_stats();
  ASSERT_EQ(draw_and_read(), expected());
  const auto after = uniform_stats();
  EXPECT_EQ(after.reuse_candidates, before.reuse_candidates);
  EXPECT_EQ(after.full_builds - before.full_builds, 1u);
}

TEST_F(UniformReuseFifo, VerifyCatchesAMissingDirtyFlagAndDrawsTheCorrectBlock) {
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_VERIFY);
  set_shader(konst_config(GX_TEV_KCSEL_K0));
  touch_pipeline();
  ASSERT_EQ(draw_and_read(), expected());
  // Worker idle after the drain: change a K colour behind the dirty flags' back.
  g_gxState.kcolors[0][0] = 0.3125f;
  ShaderConfig other = konst_config(GX_TEV_KCSEL_K0);
  other.alphaCompare.ref0 ^= 2;
  set_shader(other);
  touch_pipeline();
  const auto before = uniform_stats();
  EXPECT_EQ(draw_and_read(), expected());
  const auto after = uniform_stats();
  EXPECT_EQ(after.verification_mismatches - before.verification_mismatches, 1u);
  EXPECT_EQ(after.disabled_after_mismatch, 1u);

  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_ON);
  set_shader(konst_config(GX_TEV_KCSEL_K0));
  touch_pipeline();
  const auto onBefore = uniform_stats();
  EXPECT_EQ(draw_and_read(), expected());
  EXPECT_EQ(uniform_stats().reuses, onBefore.reuses);
}

TEST_F(UniformReuseFifo, OffMatchesTheOracleForEveryCase) {
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_OFF);
  const auto before = uniform_stats();
  int draws = 0;
  for (const auto& config : {base_config(), base_config_other_pipeline(), konst_config(GX_TEV_KCSEL_K2), fog_config(),
                             lit_config(), tev_reg_config(), pt_tex_config(), ind_config()}) {
    set_shader(config);
    load_pos_mtx(static_cast<float>(draws));
    touch_pipeline();
    const auto got = draw_and_read();
    const auto want = expected();
    size_t at = 0;
    while (at < std::min(got.size(), want.size()) && got[at] == want[at]) {
      ++at;
    }
    ASSERT_EQ(got, want) << draws << " first difference at byte " << at << " of " << got.size() << "/"
                         << want.size();
    ++draws;
  }
  const auto after = uniform_stats();
  EXPECT_EQ(after.reuse_candidates, before.reuse_candidates);
  EXPECT_EQ(after.reuses, before.reuses);
}
