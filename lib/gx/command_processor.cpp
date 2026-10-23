#include "command_processor.hpp"

#include "../gfx/depth_peek.hpp"
#include "../gfx/recording.hpp"
#include "../internal.hpp"
#include "dolphin/gd/GDGeometry.h"
#include "dolphin/gx/GXAurora.h"
#include "gx.hpp"
#include "pipeline.hpp"
#include "regs.hpp"
#include "shader_info.hpp"
#include "fifo.hpp"
#include "texture.hpp"
#include "../gfx/hash.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "lookup_memo.hpp"
#include "index_gen.hpp"
#include <bit>
#include <memory>
#include <aurora/tuning.h>

#include <tracy/Tracy.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <atomic>
#include <cstring>
#include <span>
#include <vector>

namespace aurora::gx {
// aleks2 (H31): the memo of resolved pipelines. It belongs to the FIFO thread; the
// switch and the counters are the only things the other threads touch. See
// docs/switch-h31-pipeline-memo.md.
namespace {
constexpr uint32_t kMemoKindGx = 1;
std::atomic_bool g_pipelineMemoEnabled{true};
LookupMemo<PipelineConfig, gfx::PipelineRef, ShaderInfo, 256, 4> g_pipelineMemo;
} // namespace

// aleks2 (FIFO-opt): the fast-path switches (AURORA_FIFO_FAST_*) and their counters.
// Only the FIFO thread writes the counters (plain load/store, as the uniform ones).
std::atomic<uint32_t> g_fifoFast{0};
FifoFastCounters g_fifoFastStats;
} // namespace aurora::gx

extern "C" void aurora_gx_set_fifo_fast(uint32_t mask) {
  aurora::gx::g_fifoFast.store(mask & AURORA_FIFO_FAST_ALL, std::memory_order_relaxed);
}

extern "C" uint32_t aurora_gx_get_fifo_fast(void) { return aurora::gx::g_fifoFast.load(std::memory_order_relaxed); }

extern "C" void aurora_gx_fifo_fast_stats(AuroraFifoFastStats* out) {
  if (out == nullptr) {
    return;
  }
  const auto& c = aurora::gx::g_fifoFastStats;
  out->bind_memo_hits = c.bindMemoHits.load(std::memory_order_relaxed);
  out->bind_memo_misses = c.bindMemoMisses.load(std::memory_order_relaxed);
  out->uniform_in_place_blocks = c.uniformInPlaceBlocks.load(std::memory_order_relaxed);
  out->uniform_in_place_repeats = c.uniformInPlaceRepeats.load(std::memory_order_relaxed);
}

extern "C" void aurora_gx_set_pipeline_memo(bool enabled) {
  aurora::gx::g_pipelineMemoEnabled.store(enabled, std::memory_order_relaxed);
}

extern "C" bool aurora_gx_get_pipeline_memo() {
  return aurora::gx::g_pipelineMemoEnabled.load(std::memory_order_relaxed);
}

extern "C" void aurora_gx_pipeline_memo_stats(AuroraPipelineMemoStats* out) {
  if (out == nullptr) {
    return;
  }
  const auto stats = aurora::gx::g_pipelineMemo.stats();
  out->hits = stats.hits;
  out->misses = stats.misses;
  out->evictions = stats.evictions;
  out->occupied = stats.occupied;
  out->max_occupied = stats.maxOccupied;
  out->capacity = stats.capacity;
}

namespace aurora::gx::fifo {
namespace {
constexpr Module Log{"aurora::gx::fifo"};

u16 prepare_idx_buffer(ByteBuffer& buf, GXPrimitive prim, u16 vtxStart, u16 vtxCount) noexcept {
  u16 numIndices = 0;
  if (prim == GX_QUADS) {
    buf.reserve_extra((vtxCount / 4) * 6 * sizeof(u16));

    for (u16 v = 0; v < vtxCount; v += 4) {
      u16 idx0 = vtxStart + v;
      u16 idx1 = vtxStart + v + 1;
      u16 idx2 = vtxStart + v + 2;
      u16 idx3 = vtxStart + v + 3;

      buf.append(idx0);
      buf.append(idx1);
      buf.append(idx2);
      numIndices += 3;

      buf.append(idx2);
      buf.append(idx3);
      buf.append(idx0);
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLES) {
    buf.reserve_extra(vtxCount * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      buf.append(idx);
      ++numIndices;
    }
  } else if (prim == GX_TRIANGLEFAN) {
    buf.reserve_extra(((u32(vtxCount) - 3) * 3 + 3) * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      buf.append(std::array{vtxStart, static_cast<u16>(idx - 1), idx});
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLESTRIP) {
    buf.reserve_extra(((static_cast<u32>(vtxCount) - 3) * 3 + 3) * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      if ((v & 1) == 0) {
        buf.append(std::array{static_cast<u16>(idx - 2), static_cast<u16>(idx - 1), idx});
      } else {
        buf.append(std::array{static_cast<u16>(idx - 1), static_cast<u16>(idx - 2), idx});
      }
      numIndices += 3;
    }
  } else if (prim == GX_LINES || prim == GX_LINESTRIP || prim == GX_POINTS) {
    buf.reserve_extra(6 * sizeof(u16));
    buf.append<u16>(0);
    buf.append<u16>(1);
    buf.append<u16>(3);
    buf.append<u16>(3);
    buf.append<u16>(2);
    buf.append<u16>(0);
    numIndices = 6;
  } else
    UNLIKELY FATAL("unsupported primitive type {}", static_cast<u32>(prim));
  return numIndices;
}

// GX FIFO opcodes - use CP_ prefix to avoid clashing with GXCommandList.h macros
constexpr u8 CP_CMD_NOP = GX_NOP;
constexpr u8 CP_CMD_LOAD_CP_REG = GX_LOAD_CP_REG;
constexpr u8 CP_CMD_LOAD_XF_REG = GX_LOAD_XF_REG;
constexpr u8 CP_CMD_LOAD_INDX_A = GX_LOAD_INDX_A;
constexpr u8 CP_CMD_LOAD_INDX_B = GX_LOAD_INDX_B;
constexpr u8 CP_CMD_LOAD_INDX_C = GX_LOAD_INDX_C;
constexpr u8 CP_CMD_LOAD_INDX_D = GX_LOAD_INDX_D;
constexpr u8 CP_CMD_CALL_DL = GX_CMD_CALL_DL;
constexpr u8 CP_CMD_INVAL_VTX = GX_CMD_INVL_VC;
constexpr u8 CP_CMD_LOAD_BP_REG = GX_LOAD_BP_REG & GX_OPCODE_MASK;

// Primitive type mask
constexpr u8 CP_OPCODE_MASK = GX_OPCODE_MASK;
constexpr u8 CP_VAT_MASK = GX_VAT_MASK;

struct FogRangeLutKey {
  std::array<u16, 10> rangeK;
  f32 rangeCenter;
  f32 renderWidth;
  u32 targetWidth;

  bool operator==(const FogRangeLutKey&) const = default;
};

struct FogRangeLutEntry {
  FogRangeLutKey key;
  std::vector<f32> factors;
};

constexpr size_t MaxFogRangeLuts = 32;
std::vector<FogRangeLutEntry> sFogRangeLuts;

struct DrawCache {
  uint64_t targetLayoutKey = 0;
  PipelineConfig config{};
  ShaderInfo shaderInfo{};
  gfx::PipelineRef pipelineRef{};
  GXBindGroups bindGroups{};
  uint64_t bindGeneration = 0;
  GXVtxFmt fmt = GX_MAX_VTXFMT;
  u8 lineMode = 0;
  bool hasPipeline = false;
  gfx::Range uniformRange{};
  gfx::Range fogRange{};
  FogRangeLutKey fogRangeKey{};
  bool hasFogRange = false;
  GXVtxFmt lastDrawFmt = GX_MAX_VTXFMT;
  // The configuration draw_prim already derived while deciding whether the
  // pipeline really changed, kept so push_gx_draw does not derive it again for
  // the same draw. Only meaningful between those two calls; probeValid says so.
  PipelineConfig probe{};
  GXPrimitive probePrim = GX_TRIANGLES;
  GXVtxFmt probeFmt = GX_MAX_VTXFMT;
  bool probeValid = false;
};

extern DrawCache sDrawCache;

DrawCache sDrawCache;

FogRangeLutKey fog_range_lut_key() noexcept {
  const auto& state = g_gxState.fog;
  const f32 logicalWidth = std::max(g_gxState.logicalViewport.width, 1.f);
  const f32 renderWidth = std::max(g_gxState.renderViewport.width, 1.f);
  return {
      .rangeK = state.rangeK,
      .rangeCenter = ((static_cast<f32>(state.rangeCenter) - g_gxState.logicalViewport.left) / logicalWidth) * 2.f -
                     1.f + (g_gxState.renderViewport.left / renderWidth) * 2.f,
      .renderWidth = renderWidth,
      .targetWidth = gfx::get_render_target_size().x,
  };
}

std::vector<f32> build_fog_range_lut(const FogRangeLutKey& key) {
  std::array<f32, 10> rangeK;
  for (u32 i = 0; i < rangeK.size(); ++i) {
    const u32 source = (i & ~1u) | (1u - (i & 1u));
    rangeK[i] = static_cast<f32>(key.rangeK[source]) / 64.f;
  }

  std::vector<f32> lut(key.targetWidth);
  for (u32 x = 0; x < key.targetWidth; ++x) {
    const f32 screenX = ((static_cast<f32>(x) + 0.5f) / key.renderWidth) * 2.f - 1.f;
    const f32 offset = screenX - key.rangeCenter;
    const f32 rangeIndex = std::clamp(9.f - std::abs(offset) * 9.f, 0.f, 9.f);
    const u32 lower = static_cast<u32>(rangeIndex);
    const u32 upper = std::min(lower + 1, 9u);
    const f32 fraction = rangeIndex - static_cast<f32>(lower);
    const f32 k = std::max(rangeK[lower] * (1.f - fraction) + rangeK[upper] * fraction, 0.000001f);
    lut[x] = std::sqrt(offset * offset + k * k) / k;
  }
  return lut;
}

const std::vector<f32>& resolve_fog_range_lut(const FogRangeLutKey& key) {
  for (const auto& entry : sFogRangeLuts) {
    if (entry.key == key) {
      return entry.factors;
    }
  }
  if (sFogRangeLuts.size() == MaxFogRangeLuts) {
    sFogRangeLuts.erase(sFogRangeLuts.begin());
  }
  sFogRangeLuts.emplace_back(FogRangeLutEntry{key, build_fog_range_lut(key)});
  return sFogRangeLuts.back().factors;
}

gfx::Range push_fog_range_lut(const FogRangeLutKey& key) {
  const auto& lut = resolve_fog_range_lut(key);
  return gfx::push_storage(reinterpret_cast<const u8*>(lut.data()), lut.size() * sizeof(f32));
}

u8 line_mode_for_prim(GXPrimitive prim) noexcept {
  switch (prim) {
  case GX_LINES:
    return 1;
  case GX_LINESTRIP:
    return 2;
  case GX_POINTS:
    return 3;
  default:
    return 0;
  }
}
} // namespace

static void handle_draw(u8 cmd, ByteReader& reader) noexcept;
static void handle_aurora(ByteReader& reader) noexcept;

ProcessResult process(const u8* data, u32 size) noexcept {
  ZoneScoped;
  ByteReader reader{{data, size}};

  while (!reader.empty()) {
    const u8 cmd = reader.read<u8>();
    u8 opcode = cmd & CP_OPCODE_MASK;

    switch (opcode) {
    case CP_CMD_NOP:
      continue;

    case CP_CMD_LOAD_BP_REG: {
      const u32 value = reader.read<u32>();
      handle_bp(value);
      if (reg_get(value, 8, 24) == GX_BP_REG_DRAWDONE) {
        return {static_cast<u32>(reader.offset()), true};
      }
      break;
    }

    case CP_CMD_LOAD_CP_REG: {
      const u8 addr = reader.read<u8>();
      handle_cp(addr, reader.read<u32>());
      break;
    }

    case CP_CMD_LOAD_XF_REG: {
      const u32 header = reader.read<u32>();
      const u32 count = ((header >> 16) & 0xFFFF) + 1;
      const u16 addr = header & 0xFFFF;
      handle_xf(addr, reader.take(count * sizeof(u32)));
      break;
    }

    case CP_CMD_LOAD_INDX_A:
    case CP_CMD_LOAD_INDX_B:
    case CP_CMD_LOAD_INDX_C:
    case CP_CMD_LOAD_INDX_D: {
      ZoneScopedN("LOAD_INDX");
      const u32 arrayType = GX_POS_MTX_ARRAY + (opcode - CP_CMD_LOAD_INDX_A) / 0x08;
      const u16 srcArrayIdx = reader.read<u16>();
      const u16 addrLen = reader.read<u16>();

      const u16 len = (addrLen >> 12) + 1;
      const u16 dstAddr = addrLen & 0x0FFF;
      auto const& array = g_gxState.arrays[arrayType];
      const u32 srcOffset = static_cast<u32>(srcArrayIdx) * array.stride;
      const u32 srcSize = static_cast<u32>(len) * sizeof(u32);
      AURORA_ASSERT(array.data != nullptr, "indexed XF load from unmapped array {}", arrayType);
      AURORA_ASSERT(srcOffset <= array.size && srcSize <= array.size - srcOffset,
                    "indexed XF load outside array {}: offset={}, size={}, array size={}", arrayType, srcOffset,
                    srcSize, array.size);
      auto const* srcData = static_cast<const u8*>(array.data) + srcOffset;
      if (!copy_xf_data(dstAddr, srcData, len, array.le ? std::endian::little : std::endian::big)) {
#ifndef NDEBUG
        Log.debug("Unimplemented indexed XF load (opcode 0x{:02X}, dstAddr=%04x)", opcode, dstAddr);
#endif
      }
      break;
    }

    case CP_CMD_CALL_DL: {
      // Call display list: 8 bytes (address + size)
      Log.warn("Ignoring nested GX_CMD_CALL_DL");
      reader.skip(8);
      break;
    }

    case CP_CMD_INVAL_VTX: {
      for (auto& array : g_gxState.arrays) {
        array.cachedRange = {};
      }
      g_gxState.dirty |= DirtyImmediates;
      break;
    }

    case GX_AURORA: {
      handle_aurora(reader);
      break;
    }

    // Draw commands: 0x80-0xBF
    case GX_DRAW_QUADS:
    case GX_DRAW_TRIANGLES:
    case GX_DRAW_TRIANGLE_STRIP:
    case GX_DRAW_TRIANGLE_FAN:
    case GX_DRAW_LINES:
    case GX_DRAW_LINE_STRIP:
    case GX_DRAW_POINTS: {
      handle_draw(cmd, reader);
      break;
    }

    default:
      // Check if it's a draw command (0x80-0xBF range)
      if (cmd >= 0x80) {
        handle_draw(cmd, reader);
      } else {
        // Hex dump surrounding bytes for debugging
        {
          const size_t pos = reader.offset();
          size_t dumpStart = (pos > 17) ? pos - 17 : 0;
          size_t dumpEnd = (pos + 16 < size) ? pos + 16 : size;
          std::string hex;
          for (size_t i = dumpStart; i < dumpEnd; i++) {
            if (i == pos - 1)
              hex += fmt::format("[{:02x}]", data[i]);
            else
              hex += fmt::format(" {:02x}", data[i]);
          }
          Log.error("  hex dump (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
        }
        FATAL("command_processor: unknown opcode 0x{:02X} at pos {}", cmd, reader.offset() - 1);
      }
      break;
    }
  }
  return {size, false};
}

[[noreturn]] static void handle_draw_overrun(size_t totalVtxBytes, const ByteReader& reader) noexcept {
  // Hex dump around the draw command for debugging
  const size_t pos = reader.offset();
  const size_t size = reader.size();
  const u8* data = reader.data();
  size_t cmdPos = pos - 2 - 1; // opcode byte position (before vtxCount and pos++)
  size_t dumpStart = (cmdPos > 16) ? cmdPos - 16 : 0;
  size_t dumpEnd = (cmdPos + 32 < size) ? cmdPos + 32 : size;
  std::string hex;
  for (size_t i = dumpStart; i < dumpEnd; i++) {
    if (i == cmdPos)
      hex += fmt::format("[{:02x}]", data[i]);
    else
      hex += fmt::format(" {:02x}", data[i]);
  }
  Log.error("  hex dump around draw cmd (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
  FATAL("draw vertex data overrun: need {} bytes at pos {}, have {}", totalVtxBytes, pos, reader.remaining());
}

static u32 calc_vtx_size(GXVtxFmt fmt) noexcept {
  u32 vtxSize = 0;
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto& attrFmt = vtxFmt.attrs[i];
    switch (g_gxState.vtxDesc[i]) {
    case GX_NONE:
      break;
    case GX_DIRECT: {
      const auto attr = static_cast<GXAttr>(i);
      vtxSize += comp_type_size(attr, attrFmt.type) * comp_cnt_count(attr, attrFmt.cnt);
      break;
    }
    case GX_INDEX8:
      vtxSize += i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 ? 3 : 1;
      break;
    case GX_INDEX16:
      vtxSize += i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 ? 6 : 2;
      break;
    }
  }
  g_gxState.lastVtxFmt = fmt;
  g_gxState.lastVtxSize = vtxSize;
  return vtxSize;
}

using LayoutKey = decltype(gfx::get_render_target_layout().key);

// aleks2 (FIFO-opt, PIPELINE_HASH / PIPELINE_COMPARE): the TEV stages past tevStageCount.
// populate_pipeline_config writes stages [0, count) into a value-initialised config, so
// in every config it builds (and in cache.config, always one of them) the stages past the
// count hold the same default bytes. Hashing or comparing without them gives the same
// answer for these configs, at a fraction of the 2112 bytes the 16 stages take.
constexpr size_t kTevStagesBegin = offsetof(PipelineConfig, shaderConfig) + offsetof(ShaderConfig, tevStages);
constexpr size_t kTevStagesEnd = offsetof(PipelineConfig, shaderConfig) + offsetof(ShaderConfig, tevStageCount);
static_assert(kTevStagesEnd - kTevStagesBegin == sizeof(TevStage) * MaxTevStages);

static size_t active_stages_end(const PipelineConfig& config) noexcept {
  const u32 count = std::min<u32>(config.shaderConfig.tevStageCount, MaxTevStages);
  return kTevStagesBegin + count * sizeof(TevStage);
}

static uint64_t pipeline_hash_active(const PipelineConfig& config) noexcept {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&config);
  const uint64_t head = xxh3_hash_s(bytes, active_stages_end(config));
  return xxh3_hash_s(bytes + kTevStagesEnd, sizeof(PipelineConfig) - kTevStagesEnd, head);
}

static bool pipeline_equal_active(const PipelineConfig& a, const PipelineConfig& b) noexcept {
  const size_t end = active_stages_end(a);
  if (end != active_stages_end(b)) {
    return false; // different stage counts
  }
  const auto* pa = reinterpret_cast<const uint8_t*>(&a);
  const auto* pb = reinterpret_cast<const uint8_t*>(&b);
  return std::memcmp(pa, pb, end) == 0 &&
         std::memcmp(pa + kTevStagesEnd, pb + kTevStagesEnd, sizeof(PipelineConfig) - kTevStagesEnd) == 0;
}

// Makes the cached pipeline (its configuration, shader analysis and handle) the one the
// current GX state asks for.
static void refresh_pipeline(DrawCache& cache, GXPrimitive prim, GXVtxFmt fmt, u8 lineMode,
                             LayoutKey layoutKey) noexcept {
  auto& state = g_gxState;

  // draw_prim leaves the configuration it derived for this very draw; use it rather
  // than deriving it a second time.
  PipelineConfig derived{};
  const bool haveProbe = cache.probeValid && cache.probePrim == prim && cache.probeFmt == fmt;
  if (!haveProbe) {
    populate_pipeline_config(derived, prim, fmt);
  }
  const PipelineConfig& wanted = haveProbe ? cache.probe : derived;
  cache.probeValid = false;
  const uint32_t fast = g_fifoFast.load(std::memory_order_relaxed);
  const bool fastCompare = (fast & AURORA_FIFO_FAST_PIPELINE_COMPARE) != 0;

  cache.fmt = fmt;
  cache.lineMode = lineMode;
  // A probe from settle_pipeline_dirty was already found different from cache.config
  // (nothing changes it in between), so PIPELINE_COMPARE does not compare it again.
  if (layoutKey == cache.targetLayoutKey &&
      (fastCompare ? !haveProbe && pipeline_equal_active(wanted, cache.config)
                   : std::memcmp(&wanted, &cache.config, sizeof(PipelineConfig)) == 0)) {
    // The state was written but came out where it already was.
    state.dirty &= ~DirtyPipeline;
    return;
  }

  gfx::detail::count_draw_event(gfx::detail::DrawEvent::PipelineRebuild);
  const bool hadPipeline = cache.hasPipeline;
  const auto sampledBefore = cache.shaderInfo.sampledTextures;
  const auto sampledIndirectBefore = cache.shaderInfo.sampledIndTextures;
  cache.config = wanted;
  // The first time a configuration is seen it takes the full lookup; afterwards the
  // memo replays what that produced. With the memo off only the full lookup runs.
  memo_resolve(
      g_pipelineMemo, g_pipelineMemoEnabled.load(std::memory_order_relaxed), kMemoKindGx, layoutKey,
      gfx::pipeline_generation(), cache.config.version, cache.config,
      [&] {
        return (fast & AURORA_FIFO_FAST_PIPELINE_HASH) != 0 ? pipeline_hash_active(cache.config)
                                                            : static_cast<uint64_t>(xxh3_hash(cache.config));
      },
      cache.pipelineRef, cache.shaderInfo,
      [&] {
        cache.shaderInfo = build_shader_info(cache.config.shaderConfig);
        cache.pipelineRef = gfx::pipeline_ref(cache.config);
      });
  cache.targetLayoutKey = layoutKey;
  cache.hasPipeline = true;
  state.dirty = (state.dirty & ~DirtyPipeline) | DirtyUniform;
  if (!hadPipeline || sampledBefore != cache.shaderInfo.sampledTextures ||
      sampledIndirectBefore != cache.shaderInfo.sampledIndTextures) {
    cache.bindGeneration = 0;
    // aleks2 (V1, black ground): the texture bind group was built for the previous shader's slots. Without this,
    // refresh_bind_groups' "only the generation moved" check compares the new shader's slots with themselves
    // (already resolved by earlier draws), finds them unchanged and keeps the old group, whose extra slots hold the
    // empty 1x1 texture: Twilight Princess' 3-texture terrain after a 2-texture draw sampled black.
    state.dirty |= DirtyTextures;
  }
}

// The textures the current shader samples, by slot; slots it does not sample are null.
static std::array<const void*, MaxTextures> sampled_texture_refs(const ShaderInfo& info) noexcept {
  std::array<const void*, MaxTextures> refs{};
  for (u32 slot = 0; slot < MaxTextures; ++slot) {
    if (info.sampledTextures[slot] || info.sampledIndTextures[slot]) {
      refs[slot] = g_gxState.textures[slot].ref.get();
    }
  }
  return refs;
}

// aleks2 (FIFO-opt, BIND_MEMO): texture bind groups of this frame by what
// build_bind_groups reads, each sampled slot's texture view and sampler descriptor.
// A hit returns the BindGroupRef the same descriptor produced earlier in the frame;
// that bind group is in the cache, marked used this frame, and holds its views, so no
// view address can be reused while the entry can match. Entries are valid for their
// frame only (clear_draw_cache), so every texture set still goes through
// bind_group_ref once a frame. Direct-mapped; a collision only costs a miss.
struct BindMemoKey {
  std::array<uintptr_t, MaxTextures> views{};
  // addressModeU/V/W, magFilter, minFilter, mipmapFilter, lodMinClamp, lodMaxClamp
  // (bits), compare, maxAnisotropy: get_descriptor's output, field by field.
  std::array<std::array<uint32_t, 10>, MaxTextures> samplers{};
};
static_assert(std::has_unique_object_representations_v<BindMemoKey>);

struct BindMemoEntry {
  uint64_t frame = 0;
  uint64_t hash = 0;
  BindMemoKey key;
  GXBindGroups groups;
};
constexpr size_t kBindMemoSize = 256;
std::unique_ptr<BindMemoEntry[]> sBindMemo;
uint64_t sBindMemoFrame = 1;

static void bump_fast(std::atomic<uint64_t>& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

static GXBindGroups memo_bind_groups(const ShaderInfo& info) noexcept {
  if (!info.sampledTextures.any() && !info.sampledIndTextures.any()) {
    return {}; // as build_bind_groups
  }
  BindMemoKey key{};
  for (u32 i = 0; i < MaxTextures; ++i) {
    const auto& tex = g_gxState.textures[i];
    if (!tex || !(info.sampledTextures[i] || info.sampledIndTextures[i])) {
      continue; // build_bind_groups binds the empty view and sampler here
    }
    key.views[i] = reinterpret_cast<uintptr_t>(tex.ref->sampleTextureView.Get());
    const wgpu::SamplerDescriptor d = tex.get_descriptor();
    key.samplers[i] = {static_cast<uint32_t>(d.addressModeU),  static_cast<uint32_t>(d.addressModeV),
                       static_cast<uint32_t>(d.addressModeW),  static_cast<uint32_t>(d.magFilter),
                       static_cast<uint32_t>(d.minFilter),     static_cast<uint32_t>(d.mipmapFilter),
                       std::bit_cast<uint32_t>(d.lodMinClamp), std::bit_cast<uint32_t>(d.lodMaxClamp),
                       static_cast<uint32_t>(d.compare),       static_cast<uint32_t>(d.maxAnisotropy)};
  }
  const uint64_t hash = xxh3_hash_s(&key, sizeof(key));
  if (!sBindMemo) {
    sBindMemo = std::make_unique<BindMemoEntry[]>(kBindMemoSize);
  }
  BindMemoEntry& e = sBindMemo[hash & (kBindMemoSize - 1)];
  if (e.frame == sBindMemoFrame && e.hash == hash && std::memcmp(&e.key, &key, sizeof(key)) == 0) {
    bump_fast(g_fifoFastStats.bindMemoHits);
    return e.groups;
  }
  bump_fast(g_fifoFastStats.bindMemoMisses);
  e.frame = sBindMemoFrame;
  e.hash = hash;
  e.key = key;
  e.groups = build_bind_groups(info);
  return e.groups;
}

// Rebuilds the texture bind groups if the bound textures may have changed. Returns
// whether that dirtied the uniform (the texture size/bias section).
static bool refresh_bind_groups(DrawCache& cache) noexcept {
  auto& state = g_gxState;
  const bool registersWritten = (state.dirty & DirtyTextures) != 0;
  if (!registersWritten && cache.bindGeneration == texture::current_bind_generation()) {
    return false;
  }

  // If only the generation moved (a texture was replaced somewhere) and the slots this
  // shader samples still resolve to the same textures, the groups are still right.
  std::array<const void*, MaxTextures> before{};
  if (!registersWritten) {
    before = sampled_texture_refs(cache.shaderInfo);
  }
  const auto previousGroup = cache.bindGroups.textureBindGroup;
  resolve_sampled_textures(cache.shaderInfo);
  if (!registersWritten && sampled_texture_refs(cache.shaderInfo) == before) {
    cache.bindGeneration = texture::current_bind_generation();
    state.dirty &= ~DirtyTextures;
    return false;
  }

  gfx::detail::count_draw_event(gfx::detail::DrawEvent::BindGroupRebuild);
  cache.bindGroups = (g_fifoFast.load(std::memory_order_relaxed) & AURORA_FIFO_FAST_BIND_MEMO) != 0
                         ? memo_bind_groups(cache.shaderInfo)
                         : build_bind_groups(cache.shaderInfo);
  cache.bindGeneration = texture::current_bind_generation();
  state.dirty &= ~DirtyTextures;
  // For texture_size_bias uniform
  if (cache.bindGroups.textureBindGroup != previousGroup) {
    state.dirty |= DirtyUniform;
    return true;
  }
  return false;
}

// `pipelineOnly`: the uniform is dirty only because refresh_pipeline changed the
// pipeline (H35 reuse; ignored in the default Off mode).
static void refresh_uniform(DrawCache& cache, bool pipelineOnly) noexcept {
  auto& state = g_gxState;
  if ((state.dirty & DirtyUniform) == 0 && cache.uniformRange.size != 0) {
    return;
  }
  gfx::detail::count_draw_event(gfx::detail::DrawEvent::UniformRebuild);
  cache.uniformRange = rebuild_uniform(cache.shaderInfo, pipelineOnly, cache.uniformRange);
  state.dirty &= ~DirtyUniform;
}

static void push_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange, gfx::Range idxRange,
                         u32 numIndices) noexcept {
  auto& state = g_gxState;
  auto& cache = sDrawCache;

  DrawImmediateData immediates{.vtxStart = vertRange.offset, .currentPnMtx = state.currentPnMtx};
  for (int i = GX_VA_POS; i <= GX_VA_TEX7; ++i) {
    if (state.vtxDesc[i] != GX_INDEX8 && state.vtxDesc[i] != GX_INDEX16) {
      continue;
    }
    auto& array = state.arrays[i];
    if (array.cachedRange.size == 0) {
      array.cachedRange = gfx::push_storage(static_cast<const uint8_t*>(array.data), array.size);
    }
    immediates.arrayStart[i - GX_VA_POS] = array.cachedRange.offset;
  }

  const u8 lineMode = line_mode_for_prim(prim);
  const auto targetLayoutKey = gfx::get_render_target_layout().key;
  const bool pipelineValid = cache.hasPipeline && (state.dirty & DirtyPipeline) == 0 && cache.fmt == fmt &&
                             cache.lineMode == lineMode && cache.targetLayoutKey == targetLayoutKey;
  // Whether GX state writes dirtied the uniform before this draw; refresh_pipeline
  // and refresh_bind_groups may add their own reasons below.
  const bool uniformDirtyFromState = (state.dirty & DirtyUniform) != 0;
  if (!pipelineValid) {
    refresh_pipeline(cache, prim, fmt, lineMode, targetLayoutKey);
  }
  const bool uniformDirtyFromPipeline = !uniformDirtyFromState && (state.dirty & DirtyUniform) != 0;
  const bool uniformDirtyFromTextures = refresh_bind_groups(cache);
  refresh_uniform(cache, uniformDirtyFromPipeline && !uniformDirtyFromTextures);
  if (cache.config.shaderConfig.fogRangeEnabled) {
    const auto key = fog_range_lut_key();
    if (!cache.hasFogRange || cache.fogRangeKey != key) {
      cache.fogRange = push_fog_range_lut(key);
      cache.fogRangeKey = key;
      cache.hasFogRange = true;
    }
  }
  immediates.fogRangeBase = cache.fogRange.offset / sizeof(u32);

  state.dirty &= ~DirtyImmediates;

  uint32_t instanceCount = 1;
  if (prim == GX_LINES) {
    instanceCount = vtxCount / 2;
  } else if (prim == GX_LINESTRIP) {
    instanceCount = vtxCount - 1;
  } else if (prim == GX_POINTS) {
    instanceCount = vtxCount;
  }
  cache.lastDrawFmt = fmt;
  gfx::push_draw_command(DrawData{
      .pipeline = cache.pipelineRef,
      .bindGroups = cache.bindGroups,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = cache.uniformRange,
      .immediateData = immediates,
      .vtxCount = vtxCount,
      .indexCount = numIndices,
      .instanceCount = instanceCount,
      .dstAlpha = state.dstAlpha,
  });
}

static void handle_draw_unmerged(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange) noexcept {
  ZoneScoped;
  u32 numIndices = 0;
  gfx::Range idxRange;

  if (prim != GX_TRIANGLES && (g_fifoFast.load(std::memory_order_relaxed) & AURORA_FIFO_FAST_DIRECT_INDICES) != 0) {
    // aleks2 (FIFO-opt): the same indices, written in place.
    const u32 count = index_gen::count(prim, vtxCount);
    if (u8* out = gfx::map_indices(count * sizeof(u16), 4, idxRange)) {
      index_gen::write(out, prim, 0, vtxCount);
    }
    numIndices = static_cast<u16>(count);
  } else if (prim != GX_TRIANGLES) {
    ZoneScopedN("build idx buffer");
    static ByteBuffer idxBuf;
    numIndices = prepare_idx_buffer(idxBuf, prim, 0, vtxCount);
    idxRange = gfx::push_indices(idxBuf.data(), idxBuf.size(), 4);
    idxBuf.clear();
  }

  push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, numIndices);
}

// The pipeline-dirty flag may be set only because state was written back to what it
// already was (a material set again to the same values). Deriving the configuration
// and comparing finds that out, and the draw can then still merge. If the
// configuration did change it is kept for push_gx_draw: this draw cannot merge, so it
// is the next to need it.
static void settle_pipeline_dirty(GXPrimitive prim, GXVtxFmt fmt, bool fmtMatch) noexcept {
  auto& cache = sDrawCache;
  if ((g_gxState.dirty & DirtyPipeline) == 0 || !fmtMatch || !cache.hasPipeline ||
      cache.config.msaaSamples != gfx::sample_count()) {
    return;
  }
  cache.probe = PipelineConfig{};
  populate_pipeline_config(cache.probe, prim, fmt);
  const bool same = (g_fifoFast.load(std::memory_order_relaxed) & AURORA_FIFO_FAST_PIPELINE_COMPARE) != 0
                        ? pipeline_equal_active(cache.probe, cache.config)
                        : std::memcmp(&cache.probe, &cache.config, sizeof(PipelineConfig)) == 0;
  if (same) {
    g_gxState.dirty &= ~DirtyPipeline;
    return;
  }
  cache.probePrim = prim;
  cache.probeFmt = fmt;
  cache.probeValid = true;
}

// For the diagnostics: what stopped this draw from joining the one before it.
static void count_merge_blocker(bool fmtMatch) noexcept {
  const auto* previous = gfx::get_last_draw_command<DrawData>();
  if (previous == nullptr || previous->instanceCount != 1) {
    return;
  }
  using gfx::detail::DrawEvent;
  DrawEvent reason = DrawEvent::MergeBlockedUniform;
  if (!fmtMatch) {
    reason = DrawEvent::MergeBlockedFormat;
  } else if ((g_gxState.dirty & DirtyPipeline) != 0) {
    reason = DrawEvent::MergeBlockedPipeline;
  } else if ((g_gxState.dirty & DirtyTextures) != 0) {
    reason = DrawEvent::MergeBlockedTextures;
  }
  gfx::detail::count_draw_event(reason);
}

static void draw_prim(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, ByteReader& reader) noexcept {
  ZoneScoped;
  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt)
    LIKELY { vtxSize = g_gxState.lastVtxSize; }
  else
    UNLIKELY { vtxSize = calc_vtx_size(fmt); }

  u32 totalVtxBytes = vtxCount * vtxSize;
  if (totalVtxBytes > reader.remaining())
    UNLIKELY { handle_draw_overrun(totalVtxBytes, reader); }

  const bool fmtMatch = fmt == sDrawCache.lastDrawFmt && sDrawCache.lineMode == 0 && prim != GX_LINES &&
                        prim != GX_LINESTRIP && prim != GX_POINTS;
  settle_pipeline_dirty(prim, fmt, fmtMatch);
  const bool cleanState = g_gxState.dirty == 0 && fmtMatch;
  auto* lastDraw = cleanState ? gfx::get_last_draw_command<DrawData>() : nullptr;
  const bool canMerge = lastDraw != nullptr && lastDraw->instanceCount == 1;
  if (!canMerge) {
    count_merge_blocker(fmtMatch);
  }

  // Push raw vertex data to buffer. Merged draws must remain contiguous with the previous range.
  const auto vertexData = reader.take(totalVtxBytes);
  gfx::Range vertRange = gfx::push_verts(vertexData.data(), vertexData.size(), canMerge ? 0 : 4);

  // Try to merge with previous draw call
  if (canMerge) {
    u32 numIndices = 0;
    gfx::Range idxRange;
    static ByteBuffer idxBuf;
    const bool hadIndexRange = lastDraw->idxRange.size != 0;
    const bool directIdx = (g_fifoFast.load(std::memory_order_relaxed) & AURORA_FIFO_FAST_DIRECT_INDICES) != 0;
    if (directIdx) {
      // aleks2 (FIFO-opt): what the path below appends to idxBuf and pushes, written in
      // place: the previous draw's triangle list if it had none yet, then this one's.
      u32 previousCount = 0;
      if (lastDraw->indexCount == 0 && prim != GX_TRIANGLES) {
        previousCount = index_gen::count(GX_TRIANGLES, lastDraw->vtxCount);
        lastDraw->indexCount = static_cast<u16>(previousCount);
      }
      if (lastDraw->indexCount != 0) {
        const u32 count = index_gen::count(prim, vtxCount);
        if (u8* out = gfx::map_indices((previousCount + count) * sizeof(u16), hadIndexRange ? 0 : 4, idxRange)) {
          if (previousCount != 0) {
            index_gen::write(out, GX_TRIANGLES, 0, lastDraw->vtxCount);
          }
          index_gen::write(out + previousCount * sizeof(u16), prim, lastDraw->vtxCount, vtxCount);
        }
        numIndices += static_cast<u16>(count);
      }
    } else if (lastDraw->indexCount == 0 && prim != GX_TRIANGLES) {
      // Generate triangle index buffer for previous draw
      lastDraw->indexCount = prepare_idx_buffer(idxBuf, GX_TRIANGLES, 0, lastDraw->vtxCount);
    }
    if (directIdx) {
      // written above
    } else if (lastDraw->indexCount != 0) {
      numIndices += prepare_idx_buffer(idxBuf, prim, lastDraw->vtxCount, vtxCount);
      idxRange = gfx::push_indices(idxBuf.data(), idxBuf.size(), hadIndexRange ? 0 : 4);
      idxBuf.clear();
    }
    CHECK(lastDraw->vertRange.offset + lastDraw->vertRange.size == vertRange.offset,
          "Non-consecutive vertex ranges ({} < {})", lastDraw->vertRange.offset + lastDraw->vertRange.size,
          vertRange.offset);
    if (hadIndexRange) {
      CHECK(lastDraw->idxRange.offset + lastDraw->idxRange.size == idxRange.offset,
            "Non-consecutive index ranges ({} < {})", lastDraw->idxRange.offset + lastDraw->idxRange.size,
            idxRange.offset);
    }
    lastDraw->vertRange.size += vertRange.size;
    if (lastDraw->idxRange.size == 0) {
      lastDraw->idxRange = idxRange;
    } else {
      lastDraw->idxRange.size += idxRange.size;
    }
    lastDraw->vtxCount += vtxCount;
    lastDraw->indexCount += numIndices;
    gfx::detail::increment_merged_draw_count();
    return;
  }

  handle_draw_unmerged(prim, fmt, vtxCount, vertRange);
  sDrawCache.probeValid = false;
}

static void handle_draw(u8 cmd, ByteReader& reader) noexcept {
  const auto fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
  const auto prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
  draw_prim(prim, fmt, reader.read<u16>(), reader);
}

void handle_aurora(ByteReader& reader) noexcept {
  ZoneScoped;
  const u16 subCmd = reader.read<u16>();

  if (subCmd == GX_AURORA_LOAD_VIEWPORT_RENDER) {
    const f32 left = reader.read<f32>();
    const f32 top = reader.read<f32>();
    const f32 width = reader.read<f32>();
    const f32 height = reader.read<f32>();
    const f32 nearZ = reader.read<f32>();
    const f32 farZ = reader.read<f32>();
    set_render_viewport({
        .left = left,
        .top = top,
        .width = width,
        .height = height,
        .znear = nearZ,
        .zfar = farZ,
    });
  } else if (subCmd == GX_AURORA_LOAD_SCISSOR_RENDER) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    set_render_scissor({left, top, width, height});
  } else if (subCmd == GX_AURORA_LOAD_PROJECTION_FULL) {
    auto& proj = g_gxState.proj;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        proj[r][c] = reader.read<f32>();
      }
    }
    // Invalidate projection XF regs
    for (u32 reg = 0x20; reg <= 0x26; ++reg) {
      g_gxState.xfRegValid.reset(reg);
    }
    g_gxState.dirty |= DirtyUniform;
  } else if (subCmd >= GX_AURORA_LOAD_ARRAYBASE && subCmd <= (GX_AURORA_LOAD_ARRAYBASE | 0x0f)) {
    const u32 attrIdx = subCmd - GX_AURORA_LOAD_ARRAYBASE + GX_VA_POS;
    const u64 arrayAddr = reader.read<u64>();
    const u32 arraySize = reader.read<u32>();
    const bool le = reader.read<u8>() == 1;

    auto& array = g_gxState.arrays[attrIdx];
    const auto newData = reinterpret_cast<void*>(arrayAddr);
    if (array.data != newData || array.size != arraySize || array.le != le) {
      if (array.le != le) {
        // Endianness is baked into the shader
        g_gxState.dirty |= DirtyPipeline;
      }
      array.data = newData;
      array.size = arraySize;
      array.le = le;
      array.cachedRange = {};
      g_gxState.dirty |= DirtyImmediates;
    }
  } else if (subCmd == GX_AURORA_LOAD_TEXOBJ) {
    const auto texMapId = reader.read<u8>();
    CHECK(texMapId < MaxTextures, "invalid texture map id {}", texMapId);
    auto& slot = g_gxState.loadedTextures[texMapId];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const u32 newWidth = reader.read<u32>();
    const u32 newHeight = reader.read<u32>();
    const auto newFormat = static_cast<GXTexFmt>(reader.read<u32>());
    const auto newTlut = static_cast<GXTlut>(reader.read<u32>());
    u8 newFlags = slot.flags & ~0x80u; // Reset no-cache flag
    if (reader.read<u8>() != 0) {
      newFlags |= 1u;
    } else {
      newFlags &= ~1u;
    }
    const u32 newTexObjId = reader.read<u32>();
    const u32 newTexDataVersion = reader.read<u32>();
    if (slot.data != newData || slot.mWidth != newWidth || slot.mHeight != newHeight ||
        slot.mFormat != static_cast<u32>(newFormat) || slot.tlut != newTlut || slot.flags != newFlags ||
        slot.texObjId != newTexObjId || slot.texDataVersion != newTexDataVersion) {
      slot.data = newData;
      slot.mWidth = newWidth;
      slot.mHeight = newHeight;
      slot.mFormat = newFormat;
      slot.tlut = newTlut;
      slot.flags = newFlags;
      slot.texObjId = newTexObjId;
      slot.texDataVersion = newTexDataVersion;
      g_gxState.dirty |= DirtyTextures;
    }
  } else if (subCmd == GX_AURORA_LOAD_TLUT) {
    const auto idx = reader.read<u8>();
    CHECK(idx < MaxTluts, "invalid tlut slot {}", idx);
    auto& slot = g_gxState.loadedTluts[idx];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const auto newFormat = static_cast<GXTlutFmt>(reader.read<u32>());
    const u16 newNumEntries = reader.read<u16>();
    const u32 newTlutObjId = reader.read<u32>();
    const u32 newTlutDataVersion = reader.read<u32>();
    const u8 newFlags = slot.flags & ~0x80u; // Reset no-cache flag
    if (slot.data != newData || slot.format != newFormat || slot.numEntries != newNumEntries ||
        slot.tlutObjId != newTlutObjId || slot.tlutDataVersion != newTlutDataVersion || slot.flags != newFlags) {
      if (slot.tlutObjId != newTlutObjId || slot.tlutDataVersion != newTlutDataVersion) {
        texture::invalidate_bindings();
      }
      slot.data = newData;
      slot.format = newFormat;
      slot.numEntries = newNumEntries;
      slot.tlutObjId = newTlutObjId;
      slot.tlutDataVersion = newTlutDataVersion;
      slot.flags = newFlags;
      g_gxState.dirty |= DirtyTextures;
    }
  } else if (subCmd == GX2_SET_POLYGON_OFFSET) {
    const f32 frontOffset = reader.read<f32>();
    const f32 frontScale = reader.read<f32>();
    const f32 backOffset = reader.read<f32>();
    const f32 backScale = reader.read<f32>();
    const f32 clamp = reader.read<f32>();
    if (g_gxState.frontOffset != frontOffset || g_gxState.frontScale != frontScale ||
        g_gxState.backOffset != backOffset || g_gxState.backScale != backScale || g_gxState.clamp != clamp) {
      g_gxState.frontOffset = frontOffset;
      g_gxState.frontScale = frontScale;
      g_gxState.backOffset = backOffset;
      g_gxState.backScale = backScale;
      g_gxState.clamp = clamp;
      g_gxState.dirty |= DirtyPipeline;
    }
  } else if (subCmd == GX_AURORA_LOAD_COPY_SRC) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    g_gxState.texCopySrc = {left, top, width, height};
  } else if (subCmd == GX_AURORA_LOAD_COPY_DST) {
    g_gxState.texCopyDstWidth = reader.read<u32>();
    g_gxState.texCopyDstHeight = reader.read<u32>();
    g_gxState.texCopyFmt = static_cast<GXTexFmt>(reader.read<u32>());
    reader.skip(1); // mipmap is not implemented, but remains part of the command payload
    g_gxState.texCopyDstWide = true;
  } else if (subCmd == GX_AURORA_LOAD_COPY_DEST) {
    g_gxState.texCopyDest = reinterpret_cast<const void*>(reader.read<u64>());
  } else if (subCmd == GX_AURORA_REQUEST_DEPTH_SNAPSHOT) {
    gfx::depth_peek::request_snapshot();
  } else if (subCmd == GX_AURORA_BEGIN_OFFSCREEN) {
    const u32 width = reader.read<u32>();
    const u32 height = reader.read<u32>();
    gfx::begin_offscreen(width, height);
  } else if (subCmd == GX_AURORA_END_OFFSCREEN) {
    gfx::end_offscreen();
  } else if (subCmd == GX_AURORA_DESTROY_TEXOBJ) {
    evict_texture_object(reader.read<u32>());
  } else if (subCmd == GX_AURORA_DESTROY_TLUT) {
    evict_tlut_object(reader.read<u32>());
  } else if (subCmd == GX_AURORA_DESTROY_COPY_TEX) {
    evict_copy_texture(reinterpret_cast<const void*>(reader.read<u64>()));
  } else if (subCmd == GX_AURORA_DRAW_SIZED) {
    const u8 cmd = reader.read<u8>();
    const u32 byteLen = reader.read<u32>();
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    if (byteLen != 0) {
      u32 vtxSize;
      if (g_gxState.lastVtxFmt == fmt) {
        vtxSize = g_gxState.lastVtxSize;
      } else {
        vtxSize = calc_vtx_size(fmt);
      }
      // aleks2 (rc19): as DRAW_INDEXED below, a malformed sized draw is dropped with the rest of its buffer.
      if (vtxSize == 0 || byteLen % vtxSize != 0 || byteLen / vtxSize > 0xFFFF || reader.remaining() < byteLen) {
#ifndef NDEBUG
        FATAL("GX_AURORA_DRAW_SIZED: malformed ({} bytes, size-{} vertices, {} bytes left)", byteLen, vtxSize,
              reader.remaining());
#endif
        static std::atomic<u32> s_dropped{0};
        const u32 n = s_dropped.fetch_add(1, std::memory_order_relaxed);
        if (n < 4 || (n & (n - 1)) == 0) {
          Log.error("GX_AURORA_DRAW_SIZED: malformed draw dropped with the rest of its buffer ({} bytes, size-{} "
                    "vertices, {} bytes left); {} so far",
                    byteLen, vtxSize, reader.remaining(), n + 1);
        }
        reader.skip(reader.remaining());
        return;
      }
      u32 vtxCount = byteLen / vtxSize;
      draw_prim(prim, fmt, static_cast<u16>(vtxCount), reader);
    }
  } else if (subCmd == GX_AURORA_DRAW_INDEXED) {
    ZoneScopedN("DRAW_INDEXED");
    const u8 cmd = reader.read<u8>();
    const u16 vtxCount = reader.read<u16>();
    const u32 indexCount = reader.read<u32>();
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    const size_t idxBytes = static_cast<size_t>(indexCount) * sizeof(u16);
    u32 vtxSize;
    if (g_gxState.lastVtxFmt == fmt) {
      vtxSize = g_gxState.lastVtxSize;
    } else {
      vtxSize = calc_vtx_size(fmt);
    }
    const u32 totalVtxBytes = vtxCount * vtxSize;
    // aleks2 (rc19): a malformed indexed draw (not triangles, or its index or vertex data running past the commands
    // it came in) is dropped with the rest of that buffer, whose following bytes cannot be trusted either. rc18 on
    // the console aborted here in Reader::take, from a display list that was corrupt by the time the game drew it
    // (a warp into a boss event, with a mod's models being created). Debug builds still stop on it.
    if (prim != GX_TRIANGLES || reader.remaining() < idxBytes + static_cast<size_t>(totalVtxBytes)) {
#ifndef NDEBUG
      FATAL("GX_AURORA_DRAW_INDEXED: malformed (primitive {}, {} indices, {} vertices of {} bytes, {} bytes left)",
            static_cast<u32>(prim), indexCount, vtxCount, vtxSize, reader.remaining());
#endif
      static std::atomic<u32> s_dropped{0};
      const u32 n = s_dropped.fetch_add(1, std::memory_order_relaxed);
      if (n < 4 || (n & (n - 1)) == 0) {  // the first few, then at powers of two
        Log.error("GX_AURORA_DRAW_INDEXED: malformed draw dropped with the rest of its buffer (primitive {}, {} "
                  "indices, {} vertices of {} bytes, {} bytes left); {} so far",
                  static_cast<u32>(prim), indexCount, vtxCount, vtxSize, reader.remaining(), n + 1);
      }
      reader.skip(reader.remaining());
      return;
    }
    // Index data is always host-endian; push it to the GPU buffer as-is
    const auto indexData = reader.take(idxBytes);
    const gfx::Range idxRange = gfx::push_indices(indexData.data(), indexData.size(), 4);
    const auto vertexData = reader.take(totalVtxBytes);
    const gfx::Range vertRange = gfx::push_verts(vertexData.data(), vertexData.size(), 4);
    if (indexCount != 0) {
      push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, indexCount);
    }
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_PUSH) {
    auto label = reader.read_string();
    gfx::push_debug_group(std::move(label));
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_POP) {
    pop_debug_group();
  } else if (subCmd == GX_AURORA_DEBUG_MARKER_INSERT) {
    auto label = reader.read_string();
    gfx::insert_debug_marker(std::move(label));
  }

  else {
    Log.error("Unknown Aurora subcommand: {:04X}", subCmd);
  }
}

void clear_draw_cache() noexcept {
  sDrawCache.bindGeneration = 0;
  sDrawCache.uniformRange = {};
  invalidate_uniform_reuse();
  sDrawCache.fogRange = {};
  sDrawCache.hasFogRange = false;
  sDrawCache.probeValid = false;
  ++sBindMemoFrame; // aleks2 (FIFO-opt): this frame's bind groups no longer match
}

} // namespace aurora::gx::fifo
