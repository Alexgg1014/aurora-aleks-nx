// aleks2 (FIFO-opt) differential driver for AURORA_FIFO_FAST_UNIFORM_IN_PLACE, after
// tests/h35/uniform_off_equivalence.cpp. The real shader_info.cpp, a real uniform
// ByteBuffer behind push_uniform and begin_uniform_block, the same 200 000 randomized
// draws, run with the fast path off (argv[1] = 0) and on (argv[1] = 4). Both must print
// the same digest: every returned range, the bytes the GPU reads there, and each frame's
// buffer size. The padding between blocks is not compared: the GPU never reads it, and
// the original path does not define it either (a buffer reused over frames keeps old
// bytes there; the in-place path may leave a taken-back block's bytes).
// Run: tests/fifo_opt/run-uniform-in-place-equivalence.sh
#include "gx/gx.hpp"
#include "gx/shader_info.hpp"
#include "internal.hpp"
#include <aurora/tuning.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <fmt/format.h>
#include <absl/container/internal/hashtablez_sampler.h>

// raw_hash_set.cc's table sampling, which this driver never enables.
namespace absl::container_internal {
HashtablezInfoHandle ForcedTrySample(size_t, size_t, size_t, uint16_t) { return HashtablezInfoHandle{}; }
} // namespace absl::container_internal

namespace aurora {
AuroraConfig g_config{};
void log_internal(AuroraLogLevel, const char* m, const char* msg, unsigned int len) noexcept {
  fprintf(stderr, "%s %.*s\n", m, len, msg);
}
} // namespace aurora
auto fmt::formatter<AuroraLogLevel>::format(AuroraLogLevel level, format_context& ctx) const
    -> format_context::iterator {
  return fmt::format_to(ctx.out(), "{}", static_cast<int>(level));
}
namespace aurora::gx {
GXState g_gxState{};
std::atomic<uint32_t> g_fifoFast{0};
FifoFastCounters g_fifoFastStats;
const gfx::TextureBind& get_texture(GXTexMapID id) noexcept { return g_gxState.textures[id]; }
u8 color_channel(GXChannelID id) noexcept {
  return (id == GX_COLOR1 || id == GX_ALPHA1 || id == GX_COLOR1A1) ? 1 : 0;
}
} // namespace aurora::gx
// command_processor.cpp's switch, which this driver does not link.
extern "C" void aurora_gx_set_fifo_fast(uint32_t mask) {
  aurora::gx::g_fifoFast.store(mask & AURORA_FIFO_FAST_ALL, std::memory_order_relaxed);
}
namespace aurora::gfx {
uint32_t align_uniform(uint32_t v) { return (v + 255u) & ~255u; }
constexpr size_t kAlign = 256;
static ByteBuffer uniforms;
static uint64_t digest = 1469598103934665603ull;
void mix(const void* p, size_t n) {
  auto* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) {
    digest ^= b[i];
    digest *= 1099511628211ull;
  }
}
static void pad() {
  const size_t begin = uniforms.size();
  const size_t aligned = (begin + kAlign - 1) & ~(kAlign - 1);
  if (aligned > begin) {
    uniforms.append_zeroes(aligned - begin);
  }
}
Range push_uniform(const uint8_t* data, size_t length) { // as recording.cpp's push()
  pad();
  const auto begin = uniforms.size();
  uniforms.append(data, length);
  return {static_cast<uint32_t>(begin), static_cast<uint32_t>(length)};
}
ByteBuffer* begin_uniform_block(size_t& unpadded) {
  unpadded = uniforms.size();
  pad();
  return &uniforms;
}
static void end_frame() {
  const uint64_t size = uniforms.size();
  mix(&size, sizeof size);
  uniforms.clear();
}
static void mix_block(Range r) {
  mix(&r, sizeof r);
  if (r.size != 0) {
    mix(uniforms.data() + r.offset, r.size);
  }
}
} // namespace aurora::gfx
using namespace aurora::gx;
static float& at(aurora::Mat3x4<float>& m, int r, int c) { return (r == 0 ? m.m0 : r == 1 ? m.m1 : m.m2)[c]; }
int main(int argc, char** argv) {
  aurora_gx_set_fifo_fast(argc > 1 ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 0)) : 0);
  std::mt19937 rng{7};
  std::vector<ShaderConfig> configs;
  for (int k = 0; k < 64; ++k) {
    ShaderConfig c{};
    c.tevStageCount = 1 + rng() % 3;
    for (u32 s = 0; s < c.tevStageCount; ++s) {
      auto& st = c.tevStages[s];
      const GXTevColorArg args[] = {GX_CC_ZERO, GX_CC_KONST, GX_CC_C0,   GX_CC_C1,
                                    GX_CC_CPREV, GX_CC_TEXC, GX_CC_RASC, GX_CC_ONE};
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
    for (auto& cc : c.colorChannels) {
      cc.lightingEnabled = rng() % 2;
      cc.ambSrc = rng() % 2 ? GX_SRC_REG : GX_SRC_VTX;
      cc.matSrc = rng() % 2 ? GX_SRC_REG : GX_SRC_VTX;
    }
    if (rng() % 3 == 0) {
      c.tcgs[rng() % 4].postMtx = GX_PTTEXMTX0;
    }
    configs.push_back(c);
  }
  aurora_gx_set_uniform_reuse_mode(AURORA_UNIFORM_REUSE_OFF);
  aurora::gfx::Range current{};
  for (int i = 0; i < 200000; ++i) {
    // Fewer state writes than the H35 driver, so that repeated blocks (the case the
    // fast path takes back) are common: about one draw in three repeats.
    switch (rng() % 12) {
    case 0: at(g_gxState.pnMtx[rng() % 10].pos, rng() % 3, rng() % 4) = float(rng() % 64); break;
    case 1: g_gxState.kcolors[rng() % 4][rng() % 4] = float(rng() % 256) / 255.f; break;
    case 2: g_gxState.textures[rng() % 4].texObj.mWidth = 8u << (rng() % 5); break;
    case 3:
      g_gxState.fog.c = float(rng() % 100);
      g_gxState.lights[rng() % 8].pos[0] = float(rng() % 9);
      break;
    case 4:
      if (rng() % 50 == 0) { // a new frame: new uniform buffer
        aurora::gfx::end_frame();
        forget_last_uniform();
        current = {};
      }
      break;
    default: break;
    }
    const auto& cfg = configs[rng() % (i % 7 == 0 ? configs.size() : 4)];
    for (size_t k = 0; k < cfg.colorChannels.size(); ++k) {
      g_gxState.colorChannelConfig[k] = cfg.colorChannels[k];
    }
    const ShaderInfo info = build_shader_info(cfg);
    current = rebuild_uniform(info, rng() % 2, current);
    aurora::gfx::mix_block(current);
  }
  aurora::gfx::end_frame();
  AuroraUniformStats u{};
  aurora_gx_uniform_stats(&u);
  printf("%016llx builds=%llu repeats=%llu in_place=%llu/%llu\n", static_cast<unsigned long long>(aurora::gfx::digest),
         static_cast<unsigned long long>(u.full_builds), static_cast<unsigned long long>(u.repeat_hits),
         static_cast<unsigned long long>(g_fifoFastStats.uniformInPlaceRepeats.load()),
         static_cast<unsigned long long>(g_fifoFastStats.uniformInPlaceBlocks.load()));
}
