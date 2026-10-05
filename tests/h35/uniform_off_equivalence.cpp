// aleks2 (H35) differential driver for uniform reuse mode Off. Built twice by
// run-uniform-off-equivalence.sh: against lib/gx/shader_info.cpp of the H34 base
// (build_uniform) and of this tree (rebuild_uniform in Off). Both must print the
// same digest: every pushed block, every pushed range and every returned range.
#include "gx/gx.hpp"
#include "gx/shader_info.hpp"
#include "internal.hpp"
#ifdef H35
#include <aurora/tuning.h>
#endif
#include <cstdio>
#include <random>
#include <vector>
#include <fmt/format.h>

namespace aurora {
AuroraConfig g_config{};
void log_internal(AuroraLogLevel, const char* m, const char* msg, unsigned int len) noexcept { fprintf(stderr, "%s %.*s\n", m, len, msg); }
}
auto fmt::formatter<AuroraLogLevel>::format(AuroraLogLevel level, format_context& ctx) const -> format_context::iterator {
  return fmt::format_to(ctx.out(), "{}", static_cast<int>(level));
}
namespace aurora::gx {
GXState g_gxState{};
const gfx::TextureBind& get_texture(GXTexMapID id) noexcept { return g_gxState.textures[id]; }
}
namespace aurora::gfx {
uint32_t align_uniform(uint32_t v) { return (v + 255u) & ~255u; }
static uint32_t next = 0;
static uint64_t digest = 1469598103934665603ull;
static void mix(const void* p, size_t n) { auto* b = (const uint8_t*)p; for (size_t i = 0; i < n; ++i) { digest ^= b[i]; digest *= 1099511628211ull; } }
Range push_uniform(const uint8_t* data, size_t length) { Range r{next, (uint32_t)length}; next = align_uniform(next + length); mix(data, length); mix(&r, sizeof r); return r; }
}
using namespace aurora::gx;
static float& at(aurora::Mat3x4<float>& m, int r, int c) { return (r == 0 ? m.m0 : r == 1 ? m.m1 : m.m2)[c]; }
int main() {
  std::mt19937 rng{7};
  std::vector<ShaderConfig> configs;
  for (int k = 0; k < 64; ++k) {
    ShaderConfig c{};
    c.tevStageCount = 1 + rng() % 3;
    for (u32 s = 0; s < c.tevStageCount; ++s) {
      auto& st = c.tevStages[s];
      const GXTevColorArg args[] = {GX_CC_ZERO, GX_CC_KONST, GX_CC_C0, GX_CC_C1, GX_CC_CPREV, GX_CC_TEXC, GX_CC_RASC, GX_CC_ONE};
      st.colorPass = {args[rng() % 8], args[rng() % 8], args[rng() % 8], args[rng() % 8]};
      st.alphaPass = {GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_RASA};
      st.kcSel = static_cast<GXTevKColorSel>(GX_TEV_KCSEL_K0 + rng() % 4);
      st.texMapId = static_cast<GXTexMapID>(rng() % 4);
      st.texCoordId = static_cast<GXTexCoordID>(rng() % 4);
      st.channelId = rng() % 2 ? GX_COLOR0A0 : GX_COLOR1A1;
      st.colorOp.outReg = static_cast<GXTevRegID>(rng() % 4);
      st.alphaOp.outReg = static_cast<GXTevRegID>(rng() % 4);
    }
    c.fogType = rng() % 3 == 0 ? GX_FOG_PERSP_LIN : GX_FOG_NONE;
    c.lineMode = rng() % 5 == 0 ? (1 + rng() % 3) : 0;
    for (auto& cc : c.colorChannels) { cc.lightingEnabled = rng() % 2; cc.ambSrc = rng() % 2 ? GX_SRC_REG : GX_SRC_VTX; cc.matSrc = rng() % 2 ? GX_SRC_REG : GX_SRC_VTX; }
    if (rng() % 3 == 0) c.tcgs[rng() % 4].postMtx = GX_PTTEXMTX0;
    configs.push_back(c);
  }
#ifdef H35
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_OFF);
  aurora::gfx::Range current{};
#endif
  for (int i = 0; i < 200000; ++i) {
    switch (rng() % 6) {
    case 0: at(g_gxState.pnMtx[rng() % 10].pos, rng() % 3, rng() % 4) = float(rng() % 64); break;
    case 1: g_gxState.kcolors[rng() % 4][rng() % 4] = float(rng() % 256) / 255.f; break;
    case 2: g_gxState.textures[rng() % 4].texObj.mWidth = 8u << (rng() % 5); break;
    case 3: g_gxState.fog.c = float(rng() % 100); g_gxState.lights[rng() % 8].pos[0] = float(rng() % 9); break;
    case 4: if (rng() % 50 == 0) forget_last_uniform(); break;
    default: break;
    }
    const auto& cfg = configs[rng() % configs.size()];
    for (size_t k = 0; k < cfg.colorChannels.size(); ++k) g_gxState.colorChannelConfig[k] = cfg.colorChannels[k];
    const ShaderInfo info = build_shader_info(cfg);
    const bool pipelineOnly = rng() % 2; // drawn in both builds, used only by H35
#ifdef H35
    current = rebuild_uniform(info, pipelineOnly, current);
    aurora::gfx::mix(&current, sizeof current);
#else
    (void)pipelineOnly;
    const auto r = build_uniform(info);
    aurora::gfx::mix(&r, sizeof r);
#endif
  }
  printf("%016llx pushes_end=%u\n", (unsigned long long)aurora::gfx::digest, aurora::gfx::next);
}
namespace aurora::gx {
u8 color_channel(GXChannelID id) noexcept { return (id == GX_COLOR1 || id == GX_ALPHA1 || id == GX_COLOR1A1) ? 1 : 0; }
}
