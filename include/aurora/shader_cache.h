#ifndef AURORA_SHADER_CACHE_H
#define AURORA_SHADER_CACHE_H
// aleks2 (SHADERS-c1): what the game needs to know about shader preparation.
//
// At startup Aurora queues every pipeline recipe it knows (its pipeline_cache.db plus
// the bundled romfs initial_pipeline_cache.db) for compilation in the background; Dawn
// keeps the compiled result in dawn_cache.db. This header reports that pass ("warmup")
// so the game can show a preparation screen, decide whether it has already been done for
// this console, and keep it out of benchmarks; it also reports the health of both cache
// files and exports the recipe list for the Shader Tour.

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>

extern "C" {
#else
#include <stddef.h>
#include <stdint.h>
#endif

typedef struct {
  /* The warmup pass: recipes queued for background compilation since the last
     (re)build of the pipeline cache, and how many of those are compiled. A new pass
     (a render target layout change) increments warmupGeneration and restarts both. */
  uint32_t warmupGeneration;
  uint32_t warmupQueued;
  uint32_t warmupDone;
  uint32_t knownPipelines;  /* recipes known: pipeline_cache.db + seed */
  double warmupCompileMs;   /* time spent compiling warmup pipelines (worker thread) */
  double warmupCompileMaxMs;
  uint64_t layoutKey;       /* render target layout the warmup compiled for */

  /* The bundled seed (romfs:/initial_pipeline_cache.db). */
  uint32_t seedRowsMerged;  /* rows merged this launch (0: no seed or unreadable) */
  uint64_t seedSignature;   /* 64-bit hash of the seed's (type, hash, version) rows; 0: none */

  /* Versions that make old recipes or blobs useless. */
  uint32_t pipelineSchema;
  uint32_t gxConfigVersion;
  uint32_t clearConfigVersion;
  uint32_t rmlConfigVersion;
  uint32_t dawnSchema;

  /* Dawn's blob cache, this launch. */
  uint32_t dawnHits;
  uint32_t dawnMisses;
  uint32_t dawnStores;
  double dawnStoreMsTotal;  /* per stored blob: compress + one transaction */
  double dawnStoreMsMax;

  /* Health of each file, this launch. ok: open and usable. dirty: the last session did
     not close it (crash, forced power-off). repaired: it was damaged, set aside under a
     dated name and recreated. */
  uint8_t pipelineCacheOk, pipelineCacheDirty, pipelineCacheRepaired;
  uint8_t dawnCacheOk, dawnCacheDirty, dawnCacheRepaired;
  double pipelineCheckMs, dawnCheckMs; /* quick_check time, 0 if it did not run */
  char pipelineJournal[12];            /* journal_mode in force */
  char dawnJournal[12];

  /* SHADERS-c2. Identity of the code that turns a recipe into a pipeline (shader
     generation, pipeline creation, the Dawn revision): when it changes, everything the
     console compiled is stale even if the recipes are not. */
  uint64_t runtimeIdentity;
  /* Background pacing of the warmup (aurora_set_shader_warmup_duty), in percent. */
  uint32_t warmupDuty;
  /* V1-rc1. Where the time of a GX pipeline goes (worker thread, this launch): building the WGSL
     text, creating the shader module (Tint parses and validates it; the SPIR-V itself comes from
     dawn_cache.db), and creating the pipeline (Dawn + the driver, VkPipelineCache from the blob
     cache). Modules reused: same shader text as an earlier pipeline, nothing built or parsed. */
  double gxSourceMs, gxModuleMs, gxPipelineMs;
  uint32_t gxPipelinesTimed, gxModulesReused;
  /* V1-rc1. Scenes (aurora_set_shader_scene): the current one, the one the previous session
     ended in (its recipes go first), how many recipes have a scene recorded, and how many queued
     recipes the last scene change moved to the front of the warmup. */
  uint32_t sceneTag, lastSceneTag, recipesWithScene, scenePromoted;
  /* Warmup entries dropped because the render target layout changed before they ran. */
  uint32_t warmupDropped;
} AuroraShaderCacheStatus;

void aurora_get_shader_cache_status(AuroraShaderCacheStatus* out);

/* V1-c3: on-console self-test of the file layer under SQLite (lib/sqlite_selftest.cpp),
   in `dir` (created and removed). Safe before aurora_initialize. Writes a text report
   (probes, then every journal_mode/synchronous pair, without and with the libnx I/O shims)
   and the preferred pair that passed. Returns 1 (passed with the shims, the console's
   normal case), 0 (passed without them) or -1 (nothing passed). */
int aurora_sqlite_selftest(const char* dir, char* report, size_t reportSize, char* journal, size_t journalSize,
                           char* sync, size_t syncSize);

/* Paces the warmup pass: after compiling a warmup pipeline the worker rests so that it
   works `percent` of the time (100: no rest, the default; 50: compile 1 ms, rest 1 ms).
   Only warmup pipelines rest, and never while the game is waiting for one it needs.
   A lab option for now: whether it keeps 30 fps during play is to be measured. */
void aurora_set_shader_warmup_duty(uint32_t percent);
/* V1-rc1. The scene (stage) the game is in, by name; NULL or "" for none. Every recipe used while
   a scene is set is remembered for it (cache/pipeline_scenes.bin). Setting a scene moves its
   recipes that are still waiting to the front of the warmup, and the next start begins with the
   recipes of the scene this session was last in. */
void aurora_set_shader_scene(const char* name);
/* V1-rc5. Normal starts (the cache already holds the list): the warmup keeps only the recipes of
   the last scene, of the scenes that share most recipes with it, the ones used in most scenes
   (up to `sharedRecipes`) and the first ones the game uses (menus, HUD); the rest are compiled
   when the game asks for them (~20 ms each from the cache). From then on, entering a scene warms
   that scene's recipes in the background. Returns the recipes kept queued. */
uint32_t aurora_scope_shader_warmup(uint32_t sharedRecipes);
/* V1-rc5. Queues every known recipe again (the whole list), and ends the scene scope. */
void aurora_warm_all_shaders(void);
/* V1-rc5. While the warmup is scoped: its pipelines still waiting (the game's own requests are
   not counted), so that the compile indicator shows only what the game is waiting for. 0 otherwise. */
uint32_t aurora_scoped_warmup_pending(void);
/* V1-rc7. Pipelines the current scene still waits for: everything the game has asked for, plus the background
   warmup's recipes known for this scene. 0 when there is no scene. For a loading screen to wait on. */
uint32_t aurora_shader_scene_pending(void);
/* aleks2 (V1-rc9): nonzero lets the compile thread run on core 0 too, at the game's priority (a loading screen
   waiting for its shaders); 0 puts it back on core 2 at its own priority. Applied before its next pipeline. */
void aurora_shader_set_compile_boost(uint32_t on);
/* aleks2 (V1-rc12): warmup memory. Background (warmup) pipelines the current scene does not use are created -- filling
   the Dawn disk cache -- and released; keep_all(1) keeps them all as before. paused(1) holds the background warmup
   (the game's own requests still compile); paused(0) resumes it. */
void aurora_shader_set_warmup_paused(uint32_t on);
void aurora_shader_set_warmup_keep_all(uint32_t on);
uint32_t aurora_pipeline_live_count(void);
uint32_t aurora_shader_warmup_released(void);

/* "<device>|<driver description>" as Dawn reports them; the same string a Dawn seed is
   keyed on. Writes at most size bytes (NUL-terminated); returns the full length. */
size_t aurora_get_driver_identity(char* out, size_t size);

/* Copies the recipe list (pipeline_cache.db) to `path` with SQLite's backup API, so the
   copy is consistent while the game keeps writing. Returns the row count, or -1. */
int64_t aurora_export_pipeline_cache(const char* path);

#ifdef __cplusplus
}
#endif

#endif
