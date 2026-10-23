// aleks2: the few pieces of Aurora that exist to be measured rather than
// assumed.
#ifndef AURORA_TUNING_H
#define AURORA_TUNING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* H31 diagnostic: whether the GX draw path replays a stored PipelineRef/ShaderInfo
 * for a configuration it has already resolved. Off = the H30 path, exactly. On by
 * default. Safe to change at any time. */
void aurora_gx_set_pipeline_memo(bool enabled);
bool aurora_gx_get_pipeline_memo(void);

/* Cumulative since launch; the caller subtracts two readings to get a window.
 * `occupied` is the current entry count, `max_occupied` the highest it has been. */
typedef struct {
  uint64_t hits;
  uint64_t misses;
  uint64_t evictions;
  uint32_t occupied;
  uint32_t max_occupied;
  uint32_t capacity;
} AuroraPipelineMemoStats;
void aurora_gx_pipeline_memo_stats(AuroraPipelineMemoStats* out);

/* H35: where the game thread waited for the FIFO processor. GXDrawDone is the
 * per-frame wait inside the game's draw (JFWDisplay::endFrame); END_FRAME is the
 * one aurora_end_frame does; COMPANION is any drain made while the caller has
 * tagged the second-screen/HUD capture; EXPLICIT is AuroraGXSync and
 * aurora::gfx::drain_fifo() outside a tag; AUX_PASS is create_pass, resolve_pass,
 * push_custom_draw and push_encoder_task; OTHER is anything untagged. */
typedef enum {
  AURORA_FIFO_DRAIN_DRAW_DONE = 0,
  AURORA_FIFO_DRAIN_END_FRAME = 1,
  AURORA_FIFO_DRAIN_COMPANION = 2,
  AURORA_FIFO_DRAIN_EXPLICIT = 3,
  AURORA_FIFO_DRAIN_AUX_PASS = 4,
  AURORA_FIFO_DRAIN_OTHER = 5,
  AURORA_FIFO_DRAIN_SITE_COUNT = 6,
  /* Not a site: passed to aurora_gx_fifo_set_drain_tag to clear the tag. */
  AURORA_FIFO_DRAIN_UNTAGGED = AURORA_FIFO_DRAIN_SITE_COUNT,
} AuroraFifoDrainSite;

/* Makes every drain on the calling (game) thread count as `tag` until the tag is
 * set again; returns the previous tag so callers can restore it. Measuring only:
 * it changes nothing about when or how the FIFO is drained. */
AuroraFifoDrainSite aurora_gx_fifo_set_drain_tag(AuroraFifoDrainSite tag);

typedef struct {
  uint64_t calls;   /* drain requests, including ones with nothing buffered */
  uint64_t waits;   /* the ones where the caller had to wait for translation */
  uint64_t wait_ns; /* time the caller spent in those waits */
} AuroraFifoDrainStats;

/* Cumulative since launch, never reset: read it before and after a window and
 * subtract (unsigned subtraction stays right across a wrap). All counters are
 * relaxed atomics, so a reading taken while the worker runs can be a few
 * hundred nanoseconds out of step between fields, never torn within one. */
typedef struct {
  uint64_t busy_ns;         /* time translating commands (FIFO worker); excludes the draw-done callback */
  uint64_t processed_bytes; /* FIFO bytes translated */
  uint64_t process_calls;   /* command_processor::process() calls */
  uint64_t publishes;       /* times the published watermark advanced (incl. drains) */
  uint64_t wake_requests;   /* notifications sent to the worker */
  uint64_t worker_sleeps;   /* times the worker found nothing to do and waited */
  AuroraFifoDrainStats drains;                                  /* all sites */
  AuroraFifoDrainStats drain_sites[AURORA_FIFO_DRAIN_SITE_COUNT]; /* by site */
  uint32_t processing_mode; /* 0 = at drain, 1 = inline at publish, 2 = worker thread */
} AuroraFifoStats;
void aurora_gx_fifo_stats(AuroraFifoStats* out);

/* H35 diagnostic: reuse of the uniform block when only the pipeline changed.
 * OFF    = the H34 path exactly: every dirty uniform is filled, hashed and pushed.
 * VERIFY = the reuse decision is made, but the block is always filled in full and
 *          compared byte for byte with the one reuse would have kept; the full
 *          block is what is drawn. A difference counts a mismatch, is logged and
 *          disables reuse for the rest of the process.
 * ON     = reuse decisions are applied (never after a mismatch).
 * OFF by default. Safe to change at any time, from any thread: each draw reads the
 * mode once, and a reuse can only point at a block filled earlier in the same
 * frame under the same conditions. */
typedef enum {
  AURORA_UNIFORM_REUSE_OFF = 0,
  AURORA_UNIFORM_REUSE_VERIFY = 1,
  AURORA_UNIFORM_REUSE_ON = 2,
} AuroraUniformReuseMode;
void aurora_gx_set_uniform_reuse_mode(AuroraUniformReuseMode mode);
AuroraUniformReuseMode aurora_gx_get_uniform_reuse_mode(void);

/* Cumulative since launch (subtract two readings). */
typedef struct {
  uint64_t full_builds;      /* blocks filled in full */
  uint64_t repeat_hits;      /* full blocks equal to the previous one (pushed once) */
  uint64_t reuse_candidates; /* pipeline-only changes that met every reuse condition */
  uint64_t reuses;           /* candidates actually reused (ON) */
  uint64_t verification_checks;     /* candidates compared byte for byte (VERIFY) */
  uint64_t verification_mismatches; /* comparisons that differed */
  uint32_t disabled_after_mismatch; /* 1 once a mismatch has disabled reuse */
  uint32_t last_mismatch_offset;    /* byte offset of the first difference */
  uint32_t last_mismatch_section;   /* aurora_gx_uniform_section_name() index */
  uint32_t last_mismatch_size;      /* size of the correct block */
} AuroraUniformStats;
void aurora_gx_uniform_stats(AuroraUniformStats* out);
/* Name of a uniform block section ("viewport", "pos_mtx", ...); "unknown" if out of
 * range. */
const char* aurora_gx_uniform_section_name(uint32_t section);

/* H35: where the frame waits outside the FIFO, cumulative since launch (subtract two
 * readings). frame_slot: aurora_begin_frame waiting for one of the frame slots the
 * render worker still holds (render backpressure). staging: waiting for a mapped
 * staging buffer. render_work: the render worker running queued work (encoding,
 * submit, present); present can block on the display, so this is not pure CPU. */
typedef struct {
  uint64_t frame_slot_acquires;
  uint64_t frame_slot_waits; /* acquires that found no free slot at first */
  uint64_t frame_slot_wait_ns;
  uint64_t staging_acquires;
  uint64_t staging_waits; /* acquires that found no free or mapped buffer at first */
  uint64_t staging_wait_ns;
  uint64_t render_items;
  uint64_t render_work_ns;
} AuroraFrameWaitStats;
void aurora_gfx_frame_wait_stats(AuroraFrameWaitStats* out);

/* aleks2 (deko3d P1): the render worker's CPU as THREAD time (the kernel's tick count of that thread on
 * Switch, 0 elsewhere), not wall time: render_work_ns includes waits for the display. Cumulative since
 * launch (subtract two readings); draws = the draw commands of the frames encoded (drawCallCount
 * summed), so thread_cpu_ns / draws is the render worker's CPU per draw with either backend. deko3d:
 * true when the deko3d renderer drives the GPU; gpu_ns / gpu_frames its GPU time (timestamps; with Dawn
 * use aurora_gpu_timing_stats). */
typedef struct {
  uint64_t thread_cpu_ns;
  uint64_t frames;
  uint64_t draws;
  uint64_t gpu_ns;
  uint64_t gpu_frames;
  bool deko3d;
} AuroraRenderCpuStats;
void aurora_gfx_render_cpu_stats(AuroraRenderCpuStats* out);

/* V1-c1: whole-frame GPU time from timestamp queries (lib/webgpu/gpu_timing.cpp), without
 * Tracy. Each frame is bracketed by two empty passes whose timestamps are read back two or
 * three frames late. Off until a consumer turns it on; supported = 0 when the device has no
 * TimestampQuery or AURORA_GPU_TIMING=0 kept it from being requested. Counters are
 * cumulative since launch (subtract two readings). period_ppm is the GPU tick length in
 * nanoseconds x 1e6 that the figures were converted with (1e6 until calibrated against the
 * CPU clock, and kept at 1e6 when the estimate is within 5 % of it). */
typedef struct {
  uint32_t supported;
  uint32_t enabled;
  uint64_t frames;  /* frames whose GPU time was read back */
  uint64_t gpu_ns;  /* their summed GPU time */
  uint64_t last_ns; /* the most recent one */
  uint64_t skipped; /* frames not measured: readback ring busy, map failed or a broken pair */
  uint32_t period_ppm;
  uint32_t calibrated;
} AuroraGpuTimingStats;
void aurora_gpu_timing_set_enabled(bool enabled);
void aurora_gpu_timing_stats(AuroraGpuTimingStats* out);

/* V1-c1: render-target sets kept per internal resolution, so that going back to a size
 * already visited reuses its textures instead of allocating a new set (dynamic resolution).
 * 0 = previous behaviour (a new set on every change, the old one dropped); at most 6. Main
 * thread. Lowering it releases the least recently used sets, never the one in use. */
void aurora_set_render_target_pool(uint32_t max_sets);
typedef struct {
  uint64_t resizes;     /* internal-resolution changes applied to the render targets */
  uint64_t allocations; /* render-target sets created */
  uint64_t pool_hits;   /* changes served by a set kept from before */
  uint64_t evictions;   /* sets released to stay within the pool size */
  uint64_t live_bytes_estimate; /* every set held now, from sizes and formats */
  uint32_t pool_sets;
  uint32_t pool_max;
  uint32_t width;  /* the set in use */
  uint32_t height;
} AuroraRenderTargetStats;
void aurora_render_target_stats(AuroraRenderTargetStats* out);

/* Which graphics API and adapter the device was created on, as text, for
 * reports that have to say what they were measured on. Each buffer is always
 * NUL-terminated ("unknown" if the adapter is not up yet); any may be
 * NULL. `driver` is the adapter's description string, which for Vulkan carries
 * the driver name and version. */
void aurora_get_adapter_info(char* backend, size_t backend_size, char* device, size_t device_size, char* driver,
                             size_t driver_size);

/* aleks2 (FIFO-opt): fast paths of the GX FIFO translation, one bit each, all off by
 * default. Each one draws exactly what the original path draws (host tests compare
 * them); they only remove work from the FIFO thread.
 * BIND_MEMO        = texture bind groups remembered for the frame by each sampled
 *                    slot's view and sampler: a texture set seen earlier in the frame
 *                    skips the sampler cache and the bind group cache (hashes, locks).
 * DIRECT_INDICES   = generated indices written straight into the frame's index
 *                    buffer instead of a scratch buffer copied in.
 * UNIFORM_IN_PLACE = the uniform block (Uniform Reuse Off only) filled straight into
 *                    the frame's uniform buffer; a block equal to the previous one is
 *                    found with memcmp and taken back, instead of an xxh3 of every
 *                    block and a copy.
 * PIPELINE_HASH    = the pipeline memo's hash covers only the TEV stages in use (the
 *                    16 stages are 2112 of the config's 2772 bytes); the memo still
 *                    confirms a hit with a full memcmp.
 * PIPELINE_COMPARE = settle/refresh compare configs without the unused TEV stages
 *                    (value-initialised alike in every config populate builds), and
 *                    refresh skips the compare settle already made.
 * Safe to change at any time, from any thread: each use reads the mask once. */
typedef enum {
  AURORA_FIFO_FAST_BIND_MEMO = 1u << 0,
  AURORA_FIFO_FAST_DIRECT_INDICES = 1u << 1,
  AURORA_FIFO_FAST_UNIFORM_IN_PLACE = 1u << 2,
  AURORA_FIFO_FAST_PIPELINE_HASH = 1u << 3,
  AURORA_FIFO_FAST_PIPELINE_COMPARE = 1u << 4,
  AURORA_FIFO_FAST_ALL = 0x1Fu,
} AuroraFifoFastPath;
void aurora_gx_set_fifo_fast(uint32_t mask);
/* aleks2 (V1-rc8): texture-replacement cache budget in MB (0 = platform default: 640 on Switch, 4096 elsewhere). */
void aurora_gx_set_texrep_budget_mb(uint32_t mb);
uint32_t aurora_gx_get_texrep_budget_mb(void);
/* aleks2 (V1-rc12): texture-replacement usage report. Off by default (nothing recorded). enable() starts a window;
 * take() hands over every replacement made resident since the last take (name, size, format, resident bytes), the
 * resident peak after eviction and the evictions in the window, then starts the next window. The callback runs
 * outside the registry lock. registered_keys() = textures that have a replacement installed (0: no pack). */
typedef struct AuroraTexrepUsageRow {
  uint64_t id;
  const char* name;
  uint32_t width;
  uint32_t height;
  uint32_t mips;
  const char* format;
  uint64_t bytes;
  uint32_t full; /* 0 = only the thumbnail tier so far */
  uint32_t loads;
} AuroraTexrepUsageRow;
typedef struct AuroraTexrepUsageTotals {
  uint64_t residentBytes;
  uint64_t peakBytes;
  uint64_t budgetBytes;
  uint64_t evictions;
  uint64_t totalEvictions;
  uint32_t residentTextures;
} AuroraTexrepUsageTotals;
typedef void (*AuroraTexrepUsageFn)(void* user, const AuroraTexrepUsageRow* row);
void aurora_gx_texrep_usage_enable(uint32_t on);
uint64_t aurora_gx_texrep_registered_keys(void);
void aurora_gx_texrep_usage_take(AuroraTexrepUsageFn fn, void* user, AuroraTexrepUsageTotals* totals);
uint32_t aurora_gx_get_fifo_fast(void);
/* aleks2 (V1-rc11): bumped every time the EFB-copy texture cache is dropped (a framebuffer size change, shutdown):
 * a caller that keeps a copy texture alive across frames (the minimap's panned base) re-renders it when this moves. */
uint32_t aurora_gx_copy_cache_generation(void);
/* aleks2 (V1-rc11): cumulative EFB copies, and copies to a destination last used with another size or format. */
void aurora_gx_copy_stats(uint64_t* copies, uint64_t* reshapes);
/* aleks2 (V1-rc12): texture binds inside an EFB copy's memory but not at its start (read from RAM, not from the
 * copy), and binds at a copy's start with another size than the copy. Cumulative. */
void aurora_gx_copy_bind_stats(uint64_t* insideCopy, uint64_t* sizeMismatch);
/* aleks2 (V1-rc12): debug switches for the black-ground bisect (perf-lab.ini, developer mode). Bits:
 * AURORA_DEBUG_FOG_OFF every pipeline without fog; AURORA_DEBUG_BASE_MIP samplers clamped to mip 0;
 * AURORA_DEBUG_NO_ANISO anisotropy 1. Set once at boot; 0 = normal. */
typedef enum {
  AURORA_DEBUG_FOG_OFF = 1u << 0,
  AURORA_DEBUG_BASE_MIP = 1u << 1,
  AURORA_DEBUG_NO_ANISO = 1u << 2,
  AURORA_DEBUG_COPY_SHAPE_MATCH = 1u << 3,  /* a copy address sampled at another size binds the copy of that size */
} AuroraDebugFlag;
void aurora_debug_set_flags(uint32_t flags);
uint32_t aurora_debug_flags(void);

/* Cumulative since launch (subtract two readings). */
typedef struct {
  uint64_t bind_memo_hits;
  uint64_t bind_memo_misses;
  uint64_t uniform_in_place_blocks;  /* blocks filled in place */
  uint64_t uniform_in_place_repeats; /* of those, equal to the previous one and taken back */
} AuroraFifoFastStats;
void aurora_gx_fifo_fast_stats(AuroraFifoFastStats* out);

#ifdef __cplusplus
}
#endif

#endif /* AURORA_TUNING_H */
