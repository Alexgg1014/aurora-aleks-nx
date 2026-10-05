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

/* Which graphics API and adapter the device was created on, as text, for
 * reports that have to say what they were measured on. Each buffer is always
 * NUL-terminated ("unknown" if the adapter is not up yet); any may be
 * NULL. `driver` is the adapter's description string, which for Vulkan carries
 * the driver name and version. */
void aurora_get_adapter_info(char* backend, size_t backend_size, char* device, size_t device_size, char* driver,
                             size_t driver_size);

#ifdef __cplusplus
}
#endif

#endif /* AURORA_TUNING_H */
