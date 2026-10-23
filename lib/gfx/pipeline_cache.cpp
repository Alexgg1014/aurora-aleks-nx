#include "pipeline_cache.hpp"

#include <aurora/shader_cache.h>

#include "clear.hpp"
#include "frame_packet.hpp"
#include "resources.hpp"
#include "hash.hpp"
#include "../gx/pipeline.hpp"
#include "../io.hpp"
#include "../thread.hpp"
#ifdef AURORA_ENABLE_RMLUI
#include "../rmlui/pipeline.hpp"
#endif
#include "../sqlite_utils.hpp"
#include "../webgpu/gpu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <system_error>
#include <limits>
#include <mutex>
#include <optional>
#include <ranges>
#include <variant>
#include <thread>

#include <SDL3/SDL_iostream.h>
#include <SDL3/SDL_stdinc.h>
#if defined(AURORA_CACHE_USE_ZSTD)
#include <zstd.h>
#endif
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <fmt/format.h>
#include <tracy/Tracy.hpp>
#if defined(AURORA_GFX_DEKO3D)
#include "dk_gfx.hpp"
#include "dk_runtime.hpp"
#endif

#ifdef __SWITCH__
#include <switch.h>  // aleks2 (V1-rc9): the compile boost (thread core mask and priority)
#endif

namespace aurora::gfx {
static Module Log("aurora::gfx::pipeline_cache");

constexpr int PipelineCacheSchema = 1;
constexpr const char* InitialPipelineCacheName = "initial_pipeline_cache.db";
constexpr const char* SdlVfsName = "aurora_pipeline_cache_sdl_vfs";

using NewPipelineCallback = std::function<CompiledPipeline()>;

struct PendingPipeline {
  PipelineRef hash;
  NewPipelineCallback create;
  HashType configKey = 0;  // aleks2 (V1-rc1): the recipe (g_knownPipelines key), for scene ordering
};

struct PipelineCacheWrite {
  ShaderType type;
  PipelineRef hash;
  uint32_t configVersion;
  ByteBuffer config;
  uint32_t firstFrameUsed = UINT32_MAX;
};

using SavedPipelineConfig = std::variant<gx::PipelineConfig, clear::PipelineConfig
#ifdef AURORA_ENABLE_RMLUI
                                         ,
                                         rmlui::PipelineConfig
#endif
                                         >;

struct KnownPipeline {
  ShaderType type;
  SavedPipelineConfig config;
  uint32_t firstFrameUsed;
};

struct SdlVfsSqliteFile {
  sqlite3_file base;
  SDL_IOStream* io = nullptr;
};

static std::mutex g_pipelineMutex;
static bool g_hasPipelineThread = false;
static size_t g_pipelinesPerFrame = 0;
// For synchronous pipeline fallback (OpenGL)
#ifdef NDEBUG
constexpr size_t BuildPipelinesPerFrame = 5;
#else
constexpr size_t BuildPipelinesPerFrame = 1;
#endif
#ifdef __SWITCH__
static thread::LargeStackThread g_pipelineThread;
#else
static std::thread g_pipelineThread;
#endif
static std::atomic_bool g_pipelineThreadEnd = false;
static std::condition_variable g_pipelineQueueCv;
static std::condition_variable g_pipelineReadyCv;
static absl::flat_hash_map<PipelineRef, CompiledPipeline> g_pipelines;
static absl::flat_hash_map<HashType, KnownPipeline> g_knownPipelines;
static std::optional<uint64_t> g_pipelineLayoutKey;
static std::deque<PendingPipeline> g_pipelineQueue;
static std::deque<PendingPipeline> g_backgroundPipelineQueue;
static absl::flat_hash_set<PipelineRef> g_pendingPipelines;
static std::atomic_bool g_gpuCachePrunePending = false;

static sqlite3* g_pipelineCacheDb = nullptr;
static sqlite3_stmt* g_pipelineCacheLoadStmt = nullptr;
static sqlite3_stmt* g_pipelineCacheUpsertStmt = nullptr;
static bool g_pipelineCacheBroken = false;
static std::thread g_pipelineCacheWriterThread;
static std::condition_variable g_pipelineCacheWriterCv;
static std::mutex g_pipelineCacheWriterMutex;
static std::deque<PipelineCacheWrite> g_pipelineCacheWriteQueue;
static bool g_pipelineCacheWriterStop = false;
static int g_sdlVfsRegisterResult = SQLITE_ERROR;

// aleks2 (SHADERS-c1): cache health, the seed, and the warmup pass, for
// aurora_get_shader_cache_status (include/aurora/shader_cache.h).
static std::string g_pipelineCachePath;
static sqlite::CacheOpenReport g_pipelineOpenReport;
static bool g_pipelineCacheOpened = false;
static bool g_pipelineSawDamage = false;
static uint32_t g_seedRowsMerged = 0;
static uint64_t g_seedSignature = 0;
// Hashes queued by the current warmup pass and not compiled yet (under g_pipelineMutex).
static absl::flat_hash_set<PipelineRef> g_warmupPending;
static uint32_t g_warmupGeneration = 0;
static uint32_t g_warmupQueued = 0;
static uint32_t g_warmupDone = 0;
static double g_warmupCompileMs = 0.0;
static double g_warmupCompileMaxMs = 0.0;
static std::atomic<uint32_t> g_warmupDuty{100};
static uint32_t g_warmupDropped = 0;

// aleks2 (V1-rc1): scenes (aurora_set_shader_scene). Which scenes used each recipe, so that a
// scene's recipes go to the front of the warmup when it loads, and the last scene's first at the
// next start. Kept in cache/pipeline_scenes.bin (small, rewritten whole, atomically).
static std::atomic<uint32_t> g_sceneTag{0};
static uint32_t g_lastSceneTag = 0;
static uint32_t g_scenePromoted = 0;
static absl::flat_hash_map<HashType, std::vector<uint32_t>> g_pipelineScenes;  // under g_pipelineMutex
static bool g_pipelineScenesDirty = false;                                     // under g_pipelineMutex
// V1-rc5: aurora_scope_shader_warmup was called: entering a scene queues that scene's recipes.
static std::atomic<bool> g_warmupScoped{false};
// aleks2 (V1-rc12): warmup memory. A tester's rc11 first launch warmed the whole list (6811 recipes) in the background
// during play and kept every pipeline alive: the heap's free arena fell from 159 to 65 MiB, then the capture service
// (grc) aborted and fatal itself panicked. A background (warmup) pipeline that the current scene does not use is now
// created -- which is what writes it to the Dawn disk cache -- and released at once; a draw that needs it later gets
// it back from that cache. aurora_shader_set_warmup_keep_all(1) restores the old behaviour;
// aurora_shader_set_warmup_paused(1) holds the background warmup (the game's low-memory guard).
static std::atomic<bool> g_warmupKeepAll{false};
static std::atomic<bool> g_warmupPaused{false};
static uint32_t g_warmupReleased = 0;  // under g_pipelineMutex

// Under g_pipelineMutex. Unknown recipe: assume it is used.
static bool scene_uses_recipe(HashType configKey) {
  if (configKey == 0) {
    return true;
  }
  const uint32_t scene = g_sceneTag.load(std::memory_order_relaxed);
  const auto it = g_pipelineScenes.find(configKey);
  return it != g_pipelineScenes.end() && scene != 0 && std::ranges::find(it->second, scene) != it->second.end();
}
static void warm_scene_recipes(uint32_t tag, const char* name);
constexpr size_t MaxScenesPerRecipe = 24;
constexpr uint32_t PipelineScenesMagic = 0x31535041;  // "APS1"

// V1-rc1: where a GX pipeline's time goes (gx/pipeline.cpp, gx/shader.cpp).
std::atomic<uint64_t> g_gxSourceNs{0};
std::atomic<uint64_t> g_gxModuleNs{0};
std::atomic<uint64_t> g_gxPipelineNs{0};
std::atomic<uint32_t> g_gxPipelinesTimed{0};
std::atomic<uint32_t> g_gxModulesReused{0};

static SdlVfsSqliteFile* sdl_vfs_file(sqlite3_file* file) { return reinterpret_cast<SdlVfsSqliteFile*>(file); }

static sqlite3_vfs* default_vfs(sqlite3_vfs* vfs) { return static_cast<sqlite3_vfs*>(vfs->pAppData); }

static int sdl_vfs_close(sqlite3_file* file) {
  auto* vfsFile = sdl_vfs_file(file);
  if (vfsFile->io != nullptr) {
    SDL_CloseIO(vfsFile->io);
    vfsFile->io = nullptr;
  }
  return SQLITE_OK;
}

static int sdl_vfs_read(sqlite3_file* file, void* buffer, int amount, sqlite3_int64 offset) {
  auto* vfsFile = sdl_vfs_file(file);
  if (vfsFile->io == nullptr || offset < 0 || amount < 0) {
    return SQLITE_IOERR_READ;
  }
  if (SDL_SeekIO(vfsFile->io, offset, SDL_IO_SEEK_SET) < 0) {
    return SQLITE_IOERR_SEEK;
  }

  auto* dst = static_cast<uint8_t*>(buffer);
  int total = 0;
  while (total < amount) {
    const size_t read = SDL_ReadIO(vfsFile->io, dst + total, static_cast<size_t>(amount - total));
    if (read == 0) {
      if (SDL_GetIOStatus(vfsFile->io) == SDL_IO_STATUS_EOF) {
        std::memset(dst + total, 0, static_cast<size_t>(amount - total));
        return SQLITE_IOERR_SHORT_READ;
      }
      return SQLITE_IOERR_READ;
    }
    if (read > static_cast<size_t>(std::numeric_limits<int>::max() - total)) {
      return SQLITE_IOERR_READ;
    }
    total += static_cast<int>(read);
  }
  return SQLITE_OK;
}

static int sdl_vfs_write(sqlite3_file*, const void*, int, sqlite3_int64) { return SQLITE_READONLY; }

static int sdl_vfs_truncate(sqlite3_file*, sqlite3_int64) { return SQLITE_READONLY; }

static int sdl_vfs_sync(sqlite3_file*, int) { return SQLITE_OK; }

static int sdl_vfs_file_size(sqlite3_file* file, sqlite3_int64* size) {
  auto* vfsFile = sdl_vfs_file(file);
  if (vfsFile->io == nullptr || size == nullptr) {
    return SQLITE_IOERR_FSTAT;
  }

  const auto ioSize = SDL_GetIOSize(vfsFile->io);
  if (ioSize < 0) {
    return SQLITE_IOERR_FSTAT;
  }
  *size = static_cast<sqlite3_int64>(ioSize);
  return SQLITE_OK;
}

static int sdl_vfs_lock(sqlite3_file*, int) { return SQLITE_OK; }

static int sdl_vfs_unlock(sqlite3_file*, int) { return SQLITE_OK; }

static int sdl_vfs_check_reserved_lock(sqlite3_file*, int* reserved) {
  if (reserved != nullptr) {
    *reserved = 0;
  }
  return SQLITE_OK;
}

static int sdl_vfs_file_control(sqlite3_file*, int op, void* arg) {
  switch (op) {
  case SQLITE_FCNTL_LOCKSTATE:
    *static_cast<int*>(arg) = SQLITE_LOCK_NONE;
    return SQLITE_OK;
  case SQLITE_FCNTL_HAS_MOVED:
    *static_cast<int*>(arg) = 0;
    return SQLITE_OK;
  case SQLITE_FCNTL_SIZE_HINT:
    return SQLITE_OK;
  default:
    return SQLITE_NOTFOUND;
  }
}

static int sdl_vfs_sector_size(sqlite3_file*) { return 4096; }

static int sdl_vfs_device_characteristics(sqlite3_file*) {
  return SQLITE_IOCAP_IMMUTABLE | SQLITE_IOCAP_UNDELETABLE_WHEN_OPEN;
}

static constexpr sqlite3_io_methods SdlVfsIoMethods{
    .iVersion = 1,
    .xClose = sdl_vfs_close,
    .xRead = sdl_vfs_read,
    .xWrite = sdl_vfs_write,
    .xTruncate = sdl_vfs_truncate,
    .xSync = sdl_vfs_sync,
    .xFileSize = sdl_vfs_file_size,
    .xLock = sdl_vfs_lock,
    .xUnlock = sdl_vfs_unlock,
    .xCheckReservedLock = sdl_vfs_check_reserved_lock,
    .xFileControl = sdl_vfs_file_control,
    .xSectorSize = sdl_vfs_sector_size,
    .xDeviceCharacteristics = sdl_vfs_device_characteristics,
};

static int sdl_vfs_open(sqlite3_vfs*, sqlite3_filename name, sqlite3_file* file, int flags, int* outFlags) {
  auto* vfsFile = sdl_vfs_file(file);
  vfsFile->base.pMethods = nullptr;
  vfsFile->io = nullptr;

  if (name == nullptr || (flags & SQLITE_OPEN_READWRITE) != 0 || (flags & SQLITE_OPEN_READONLY) == 0) {
    return SQLITE_CANTOPEN;
  }

  vfsFile->io = SDL_IOFromFile(name, "rb");
  if (vfsFile->io == nullptr) {
    return SQLITE_CANTOPEN;
  }

  vfsFile->base.pMethods = &SdlVfsIoMethods;
  if (outFlags != nullptr) {
    *outFlags = SQLITE_OPEN_READONLY;
  }
  return SQLITE_OK;
}

static int sdl_vfs_delete(sqlite3_vfs*, const char*, int) { return SQLITE_READONLY; }

static int sdl_vfs_access(sqlite3_vfs*, const char* name, int flags, int* result) {
  if (result == nullptr) {
    return SQLITE_IOERR_ACCESS;
  }
  if (name == nullptr || flags == SQLITE_ACCESS_READWRITE) {
    *result = 0;
    return SQLITE_OK;
  }

  auto* io = SDL_IOFromFile(name, "rb");
  *result = io != nullptr ? 1 : 0;
  if (io != nullptr) {
    SDL_CloseIO(io);
  }
  return SQLITE_OK;
}

static int sdl_vfs_full_pathname(sqlite3_vfs*, const char* name, int outSize, char* out) {
  if (name == nullptr || out == nullptr || outSize <= 0) {
    return SQLITE_CANTOPEN;
  }
  sqlite3_snprintf(outSize, out, "%s", name);
  return SQLITE_OK;
}

static void* sdl_vfs_dl_open(sqlite3_vfs*, const char*) { return nullptr; }

static void sdl_vfs_dl_error(sqlite3_vfs*, int bytes, char* message) {
  if (message != nullptr && bytes > 0) {
    sqlite3_snprintf(bytes, message, "%s", "Dynamic loading is unsupported");
  }
}

static void (*sdl_vfs_dl_sym(sqlite3_vfs*, void*, const char*))(void) { return nullptr; }

static void sdl_vfs_dl_close(sqlite3_vfs*, void*) {}

static int sdl_vfs_randomness(sqlite3_vfs* vfs, int bytes, char* out) {
  if (auto* base = default_vfs(vfs); base != nullptr && base->xRandomness != nullptr) {
    return base->xRandomness(base, bytes, out);
  }
  if (out != nullptr && bytes > 0) {
    std::memset(out, 0, static_cast<size_t>(bytes));
  }
  return bytes;
}

static int sdl_vfs_sleep(sqlite3_vfs* vfs, int microseconds) {
  if (auto* base = default_vfs(vfs); base != nullptr && base->xSleep != nullptr) {
    return base->xSleep(base, microseconds);
  }
  return microseconds;
}

static int sdl_vfs_current_time(sqlite3_vfs* vfs, double* time) {
  if (auto* base = default_vfs(vfs); base != nullptr && base->xCurrentTime != nullptr) {
    return base->xCurrentTime(base, time);
  }
  if (time != nullptr) {
    *time = 2440587.5;
  }
  return SQLITE_OK;
}

static int sdl_vfs_get_last_error(sqlite3_vfs*, int, char*) { return SQLITE_OK; }

static bool register_sdl_vfs() {
  static std::once_flag registerOnce;
  std::call_once(registerOnce, [] {
    auto* baseVfs = sqlite3_vfs_find(nullptr);
    static sqlite3_vfs sdlVfs{
        .iVersion = 1,
        .szOsFile = static_cast<int>(sizeof(SdlVfsSqliteFile)),
        .mxPathname = 4096,
        .pNext = nullptr,
        .zName = SdlVfsName,
        .pAppData = baseVfs,
        .xOpen = sdl_vfs_open,
        .xDelete = sdl_vfs_delete,
        .xAccess = sdl_vfs_access,
        .xFullPathname = sdl_vfs_full_pathname,
        .xDlOpen = sdl_vfs_dl_open,
        .xDlError = sdl_vfs_dl_error,
        .xDlSym = sdl_vfs_dl_sym,
        .xDlClose = sdl_vfs_dl_close,
        .xRandomness = sdl_vfs_randomness,
        .xSleep = sdl_vfs_sleep,
        .xCurrentTime = sdl_vfs_current_time,
        .xGetLastError = sdl_vfs_get_last_error,
    };
    g_sdlVfsRegisterResult = sqlite3_vfs_register(&sdlVfs, 0);
  });
  return g_sdlVfsRegisterResult == SQLITE_OK;
}

#if defined(__cpp_lib_atomic_ref)
static std::atomic_ref queuedPipelines{detail::resources().stats.queuedPipelines};
static std::atomic_ref createdPipelines{detail::resources().stats.createdPipelines};
#else
struct AtomicStatRef {
  uint32_t& ref;
  uint32_t operator++() { return __atomic_add_fetch(&ref, 1, __ATOMIC_RELAXED); }
  uint32_t operator--() { return __atomic_sub_fetch(&ref, 1, __ATOMIC_RELAXED); }
  uint32_t operator++(int) { return __atomic_fetch_add(&ref, 1, __ATOMIC_RELAXED); }
  uint32_t operator--(int) { return __atomic_fetch_sub(&ref, 1, __ATOMIC_RELAXED); }
  uint32_t operator+=(uint32_t val) { return __atomic_add_fetch(&ref, val, __ATOMIC_RELAXED); }
  uint32_t operator=(uint32_t val) {
    __atomic_store_n(&ref, val, __ATOMIC_RELAXED);
    return val;
  }
};
static AtomicStatRef queuedPipelines{detail::resources().stats.queuedPipelines};
static AtomicStatRef createdPipelines{detail::resources().stats.createdPipelines};
#endif

template <typename PipelineConfig>
static PipelineCacheWrite make_pipeline_cache_write(ShaderType type, PipelineRef hash, const PipelineConfig& config,
                                                    uint32_t firstFrameUsed) {
  static_assert(std::has_unique_object_representations_v<PipelineConfig>);

  PipelineCacheWrite write{
      .type = type,
      .hash = hash,
      .configVersion = config.version,
      .config = ByteBuffer(sizeof(config)),
      .firstFrameUsed = firstFrameUsed,
  };
  std::memcpy(write.config.data(), &config, sizeof(config));
  return write;
}

static void enqueue_pipeline_cache_write(PipelineCacheWrite write) {
  if (g_pipelineCacheBroken || g_pipelineCacheDb == nullptr) {
    return;
  }

  {
    std::lock_guard lock{g_pipelineCacheWriterMutex};
    g_pipelineCacheWriteQueue.emplace_back(std::move(write));
  }
  g_pipelineCacheWriterCv.notify_one();
}

template <typename Queue>
static auto find_pending_pipeline(Queue& queue, PipelineRef hash) {
  return std::find_if(queue.begin(), queue.end(), [=](const PendingPipeline& pending) { return pending.hash == hash; });
}

enum class PipelinePriority {
  Background, // cache warmup
  Normal,     // async skip draw
  Blocking,   // block until compiled
};

static void promote_pending_pipeline(PipelineRef hash, PipelinePriority priority) {
  if (priority == PipelinePriority::Background) {
    return;
  }

  auto backgroundIt = find_pending_pipeline(g_backgroundPipelineQueue, hash);
  if (backgroundIt == g_backgroundPipelineQueue.end()) {
    return;
  }
  if (priority == PipelinePriority::Blocking) {
    g_pipelineQueue.emplace_front(std::move(*backgroundIt));
  } else {
    g_pipelineQueue.emplace_back(std::move(*backgroundIt));
  }
  g_backgroundPipelineQueue.erase(backgroundIt);
}

static std::optional<PendingPipeline> take_pending_pipeline(PipelineRef hash) {
  auto priorityIt = find_pending_pipeline(g_pipelineQueue, hash);
  if (priorityIt != g_pipelineQueue.end()) {
    PendingPipeline pending = std::move(*priorityIt);
    g_pipelineQueue.erase(priorityIt);
    g_pendingPipelines.erase(hash);
    return pending;
  }

  auto backgroundIt = find_pending_pipeline(g_backgroundPipelineQueue, hash);
  if (backgroundIt != g_backgroundPipelineQueue.end()) {
    PendingPipeline pending = std::move(*backgroundIt);
    g_backgroundPipelineQueue.erase(backgroundIt);
    g_pendingPipelines.erase(hash);
    return pending;
  }

  return std::nullopt;
}

static void notify_pipeline_ready(bool queued, uint32_t pipelineCount) {
  createdPipelines += pipelineCount;
  if (queued && --queuedPipelines == 0 && g_gpuCachePrunePending.exchange(false, std::memory_order_acq_rel)) {
    // Prune GPU cache entries after fully loading the pipeline cache.
    webgpu::cache_prune();
  }
  g_pipelineReadyCv.notify_all();
}

template <typename Config>
static void remember_pipeline_config(ShaderType type, const Config& config, uint32_t firstFrameUsed, bool persist) {
  const auto cacheKey = xxh3_hash(config, static_cast<HashType>(type));
  bool changed = false;
  {
    std::lock_guard lock{g_pipelineMutex};
    auto [it, inserted] = g_knownPipelines.try_emplace(cacheKey, KnownPipeline{type, config, firstFrameUsed});
    changed = inserted || firstFrameUsed < it->second.firstFrameUsed;
    it->second.firstFrameUsed = std::min(it->second.firstFrameUsed, firstFrameUsed);
    if (const uint32_t scene = g_sceneTag.load(std::memory_order_relaxed); scene != 0 && persist) {
      auto& scenes = g_pipelineScenes[cacheKey];
      if (scenes.size() < MaxScenesPerRecipe && std::ranges::find(scenes, scene) == scenes.end()) {
        scenes.push_back(scene);
        g_pipelineScenesDirty = true;
      }
    }
  }
  if (persist && changed) {
    enqueue_pipeline_cache_write(make_pipeline_cache_write(type, cacheKey, config, firstFrameUsed));
  }
}

static PipelineRef find_pipeline_impl(PipelineRef runtimeKey, NewPipelineCallback&& cb,
                                      PipelinePriority priority = PipelinePriority::Normal, HashType configKey = 0) {
  ZoneScoped;

  const bool blocking = priority == PipelinePriority::Blocking;
  bool notifyWorker = false;
  bool pipelineReady = false;
  bool createdPipeline = false;
  uint32_t pipelineCount = 0;
  bool queued = false;
  {
    std::scoped_lock guard{g_pipelineMutex};
    auto pipelineIt = g_pipelines.find(runtimeKey);
    if (pipelineIt != g_pipelines.end()) {
      pipelineReady = true;
    } else if (g_pendingPipelines.contains(runtimeKey)) {
      if (blocking && !g_hasPipelineThread) {
        auto pending = take_pending_pipeline(runtimeKey);
        if (pending) {
          if (g_warmupPending.erase(runtimeKey) != 0) {
            ++g_warmupDone;
          }
          auto result = pending->create();
          pipelineCount = result.pipeline_count();
          g_pipelines.try_emplace(runtimeKey, std::move(result));
          pipelineReady = true;
          g_pipelinesPerFrame += std::max(1u, pipelineCount);
          createdPipeline = true;
          queued = true;
        }
      } else {
        promote_pending_pipeline(runtimeKey, priority);
        notifyWorker = priority != PipelinePriority::Background;
      }
    } else if (!g_hasPipelineThread && (blocking || g_pipelinesPerFrame < BuildPipelinesPerFrame)) {
      auto result = cb();
      pipelineCount = result.pipeline_count();
      g_pipelines.try_emplace(runtimeKey, std::move(result));
      pipelineReady = true;
      g_pipelinesPerFrame += std::max(1u, pipelineCount);
      createdPipeline = true;
    } else {
      PendingPipeline pending{
          .hash = runtimeKey,
          .create = std::move(cb),
          .configKey = configKey,
      };
      switch (priority) {
      case PipelinePriority::Background:
        g_backgroundPipelineQueue.emplace_back(std::move(pending));
        if (g_warmupPending.insert(runtimeKey).second) {
          ++g_warmupQueued;
        }
        break;
      case PipelinePriority::Normal:
        g_pipelineQueue.emplace_back(std::move(pending));
        break;
      case PipelinePriority::Blocking:
        g_pipelineQueue.emplace_front(std::move(pending));
        break;
      }
      g_pendingPipelines.insert(runtimeKey);
      ++queuedPipelines;
      notifyWorker = true;
    }
  }

  if (createdPipeline) {
    notify_pipeline_ready(queued, pipelineCount);
  }

  if (notifyWorker) {
    g_pipelineQueueCv.notify_one();
  }

  if (blocking && !pipelineReady) {
    std::unique_lock lock{g_pipelineMutex};
    g_pipelineReadyCv.wait(lock, [=] { return g_pipelines.contains(runtimeKey) || g_pipelineThreadEnd; });
  }
  return runtimeKey;
}

static PipelineRef resolve_pipeline(ShaderType type, const gx::PipelineConfig& config, const RenderTargetLayout& layout,
                                    PipelinePriority priority) {
  const auto configKey = xxh3_hash(config, static_cast<HashType>(type));
  const auto runtimeKey = xxh3_hash(layout.key, configKey);
  return find_pipeline_impl(runtimeKey, [config, layout] { return create_pipeline(config, layout); }, priority,
                            configKey);
}

static PipelineRef resolve_pipeline(ShaderType type, const clear::PipelineConfig& config,
                                    const RenderTargetLayout& layout, PipelinePriority priority) {
  const auto configKey = xxh3_hash(config, static_cast<HashType>(type));
  const auto runtimeKey = xxh3_hash(layout.key, configKey);
#if defined(AURORA_GFX_DEKO3D)
  if (dk::active()) {
    // aleks2 (deko3d P1): the fixed clear shaders of the DKSH cache (lib/deko/dk_pipeline.cpp)
    return find_pipeline_impl(
        runtimeKey, [config, layout] { return CompiledPipeline{.dkMain = dk::build_clear_pipeline(config, layout)}; },
        priority, configKey);
  }
#endif
  return find_pipeline_impl(
      runtimeKey, [config, layout] { return CompiledPipeline{.main = create_pipeline(config, layout)}; }, priority,
      configKey);
}

#ifdef AURORA_ENABLE_RMLUI
static PipelineRef resolve_pipeline(ShaderType type, const rmlui::PipelineConfig& config, const RenderTargetLayout&,
                                    PipelinePriority priority) {
  const auto configKey = xxh3_hash(config, static_cast<HashType>(type));
#if defined(AURORA_GFX_DEKO3D)
  if (dk::active()) {
    // aleks2 (deko3d P1): RmlUi's pipelines for deko3d (lib/deko/dk_rmlui.cpp)
    return find_pipeline_impl(
        configKey, [config] { return CompiledPipeline{.dkMain = dk::build_rml_pipeline(config)}; }, priority,
        configKey);
  }
#endif
  return find_pipeline_impl(
      configKey, [config] { return CompiledPipeline{.main = rmlui::create_pipeline(config)}; }, priority, configKey);
}
#endif

static void pipeline_cache_abort() {
  g_pipelineCacheBroken = true;
  if (g_pipelineCacheLoadStmt != nullptr) {
    sqlite3_finalize(g_pipelineCacheLoadStmt);
    g_pipelineCacheLoadStmt = nullptr;
  }
  if (g_pipelineCacheUpsertStmt != nullptr) {
    sqlite3_finalize(g_pipelineCacheUpsertStmt);
    g_pipelineCacheUpsertStmt = nullptr;
  }
  if (g_pipelineCacheDb != nullptr) {
    // Clean unless this session saw damage: then the open marker stays and the next
    // start checks the file and sets it aside.
    sqlite::close_cache(g_pipelineCacheDb, g_pipelineCachePath, !g_pipelineSawDamage);
    g_pipelineCacheDb = nullptr;
  }
}

// A write that failed because the file is damaged: say so once; the abort that follows
// keeps the open marker, so the next start repairs it.
static void note_pipeline_cache_failure(const char* what) {
  if (g_pipelineCacheDb == nullptr || g_pipelineSawDamage) {
    return;
  }
  if (sqlite::is_corruption(sqlite3_extended_errcode(g_pipelineCacheDb))) {
    g_pipelineSawDamage = true;
    Log.error("pipeline_cache.db is damaged ({}: {}); off for this session, the next start sets it aside and "
              "starts a new one",
              what, sqlite3_errmsg(g_pipelineCacheDb));
  }
}

static bool write_pipeline_cache_record(const PipelineCacheWrite& write);

static std::string pipeline_cache_seed_path() {
  if (g_config.resourcesPath == nullptr || g_config.resourcesPath[0] == '\0') {
    return InitialPipelineCacheName;
  }

  std::string path{g_config.resourcesPath};
  if (path.back() != '/' && path.back() != '\\') {
    path += '/';
  }
  path += InitialPipelineCacheName;
  return path;
}

// aleks2 (SHADERS-c1): the list compresses ~40x (a GX config is mostly zeros), so it can
// ship as <name>.zst: read whole, decompressed into SQLite-owned memory and opened as an
// in-memory database (sqlite3_deserialize), read-only. nullptr when there is none.
static sqlite3* open_compressed_seed_db(const std::string& path) {
#if defined(AURORA_CACHE_USE_ZSTD)
  const std::string zstPath = path + ".zst";
  size_t packedSize = 0;
  void* packed = SDL_LoadFile(zstPath.c_str(), &packedSize);
  if (packed == nullptr) {
    return nullptr;
  }
  const auto rawSize = ZSTD_getFrameContentSize(packed, packedSize);
  if (rawSize == ZSTD_CONTENTSIZE_ERROR || rawSize == ZSTD_CONTENTSIZE_UNKNOWN || rawSize == 0 ||
      rawSize > 256ull * 1024 * 1024) {
    Log.warn("Bundled pipeline cache '{}' is not a zstd frame with a known size", zstPath);
    SDL_free(packed);
    return nullptr;
  }
  auto* raw = static_cast<unsigned char*>(sqlite3_malloc64(rawSize));
  const auto got = raw != nullptr ? ZSTD_decompress(raw, rawSize, packed, packedSize) : 0;
  SDL_free(packed);
  if (raw == nullptr || ZSTD_isError(got) || got != rawSize) {
    Log.warn("Could not decompress bundled pipeline cache '{}'", zstPath);
    sqlite3_free(raw);
    return nullptr;
  }
  sqlite3* seedDb = nullptr;
  if (sqlite3_open_v2(":memory:", &seedDb, SQLITE_OPEN_READWRITE | SQLITE_OPEN_MEMORY, nullptr) != SQLITE_OK) {
    sqlite3_close(seedDb);
    sqlite3_free(raw);
    return nullptr;
  }
  const int ret = sqlite3_deserialize(seedDb, "main", raw, static_cast<sqlite3_int64>(rawSize),
                                      static_cast<sqlite3_int64>(rawSize),
                                      SQLITE_DESERIALIZE_FREEONCLOSE | SQLITE_DESERIALIZE_READONLY);
  if (ret != SQLITE_OK) {
    Log.warn("Bundled pipeline cache '{}' is not a database: {}", zstPath, sqlite3_errmsg(seedDb));
    sqlite3_close(seedDb);
    return nullptr;
  }
  Log.info("Bundled pipeline cache '{}': {} bytes, {} uncompressed", zstPath, packedSize, rawSize);
  return seedDb;
#else
  (void)path;
  return nullptr;
#endif
}

static sqlite3* open_pipeline_cache_seed_db(const std::string& path) {
  sqlite3* seedDb = open_compressed_seed_db(path);
  if (seedDb == nullptr) {
    if (!register_sdl_vfs()) {
      Log.warn("Failed to register SDL pipeline cache seed VFS");
      return nullptr;
    }
    const auto ret =
        sqlite3_open_v2(path.c_str(), &seedDb, SQLITE_OPEN_READONLY | SQLITE_OPEN_PRIVATECACHE, SdlVfsName);
    if (ret != SQLITE_OK) {
      if (seedDb != nullptr) {
        sqlite3_close(seedDb);
      }
      Log.info("No bundled initial pipeline cache found at '{}'", path);
      return nullptr;
    }
  }

  bool schemaMatch = false;
  const auto schemaQuery = fmt::format("SELECT 1 FROM aurora_schema WHERE value = {}", PipelineCacheSchema);
  const auto schemaRet =
      sqlite::exec(seedDb, schemaQuery.c_str(), [&schemaMatch](int, char**, char**) { schemaMatch = true; });
  if (schemaRet != SQLITE_OK) {
    Log.warn("Failed to read bundled pipeline cache schema from '{}': {}", path, sqlite3_errmsg(seedDb));
    sqlite3_close(seedDb);
    return nullptr;
  }
  if (!schemaMatch) {
    Log.warn("Bundled pipeline cache '{}' does not use schema version {}", path, PipelineCacheSchema);
    sqlite3_close(seedDb);
    return nullptr;
  }

  return seedDb;
}

// V1-c5: AURORA_SEED_LIMIT=N keeps only the first N recipes of the bundled list (in
// first-use order), in case thousands of compiled pipelines are too much memory on the
// console. 0 or unset: all of them.
static uint32_t seed_limit() {
  const char* v = std::getenv("AURORA_SEED_LIMIT");
  return v != nullptr ? static_cast<uint32_t>(std::strtoul(v, nullptr, 10)) : 0;
}

static const char* seed_select_sql() {
  return "SELECT type, hash, config_version, config_size, config, first_frame_used FROM pipeline_cache "
         "ORDER BY first_frame_used ASC, rowid ASC";
}

// The signature the list would have (same rows the merge would take), without writing.
[[maybe_unused]] static uint64_t seed_signature(sqlite3* seedDb, uint32_t& rows) {
  sqlite3_stmt* stmt = nullptr;
  uint64_t signature = 0;
  rows = 0;
  const uint32_t limit = seed_limit();
  if (sqlite3_prepare_v2(seedDb, seed_select_sql(), -1, &stmt, nullptr) != SQLITE_OK) {
    return 0;
  }
  while (sqlite3_step(stmt) == SQLITE_ROW && (limit == 0 || rows < limit)) {
    const uint64_t rowKey[3] = {static_cast<uint64_t>(sqlite3_column_int(stmt, 0)),
                                static_cast<uint64_t>(sqlite3_column_int64(stmt, 1)),
                                static_cast<uint64_t>(sqlite3_column_int(stmt, 2))};
    signature += XXH3_64bits(rowKey, sizeof(rowKey));
    ++rows;
  }
  sqlite3_finalize(stmt);
  return signature;
}

// The tables, inside one BEGIN IMMEDIATE (where a damaged file first shows). Returns
// SQLITE_OK or the failing (extended) result code, so open_cache can tell damage apart.
static int init_pipeline_cache_schema() {
  int ret = SQLITE_OK;
  bool schemaFailed = false;
  {
    sqlite::Transaction tx(g_pipelineCacheDb, Log, true);
    if (!tx) {
      Log.error("Failed to begin pipeline cache schema transaction");
      schemaFailed = true;
    } else {
      ret = sqlite::exec(g_pipelineCacheDb, "CREATE TABLE IF NOT EXISTS aurora_schema(value INTEGER);");
      if (ret != SQLITE_OK) {
        Log.error("Failed to create pipeline cache schema table: {}", sqlite3_errmsg(g_pipelineCacheDb));
        schemaFailed = true;
      }
    }

    bool schemaMatch = false;
    if (!schemaFailed) {
      const auto schemaQuery = fmt::format("SELECT 1 FROM aurora_schema WHERE value = {}", PipelineCacheSchema);
      ret = sqlite::exec(g_pipelineCacheDb, schemaQuery.c_str(),
                         [&schemaMatch](int, char**, char**) { schemaMatch = true; });
      if (ret != SQLITE_OK) {
        Log.error("Failed to read pipeline cache schema version: {}", sqlite3_errmsg(g_pipelineCacheDb));
        schemaFailed = true;
      }
    }

    if (!schemaFailed && !schemaMatch) {
      const auto schemaSql = fmt::format(
          R"(DROP TABLE IF EXISTS pipeline_cache;
CREATE TABLE pipeline_cache (
  type INTEGER NOT NULL,
  hash INTEGER NOT NULL,
  config_version INTEGER NOT NULL,
  config_size INTEGER NOT NULL,
  config BLOB NOT NULL,
  first_frame_used INTEGER NOT NULL,
  PRIMARY KEY (type, hash)
);
CREATE INDEX pipeline_cache_load_order_idx
  ON pipeline_cache(type, config_version, first_frame_used);
DELETE FROM aurora_schema;
INSERT INTO aurora_schema VALUES ({});)",
          PipelineCacheSchema);
      ret = sqlite::exec(g_pipelineCacheDb, schemaSql.c_str());
      if (ret != SQLITE_OK) {
        Log.error("Failed to initialize pipeline cache schema: {}", sqlite3_errmsg(g_pipelineCacheDb));
        schemaFailed = true;
      }
    }

    if (!schemaFailed) {
      tx.commit();
    }
  }

  if (schemaFailed) {
    const int code = sqlite3_extended_errcode(g_pipelineCacheDb);
    return code != SQLITE_OK ? code : SQLITE_ERROR;
  }
  return SQLITE_OK;
}

static bool prepare_pipeline_cache_db() {
  if (g_pipelineCacheBroken) {
    return false;
  }
  if (g_pipelineCacheDb != nullptr) {
    return true;
  }

  const auto dbPath = io::fs_path_from_string(g_config.cachePath) / "pipeline_cache.db";
  const auto path = io::fs_path_to_string(dbPath);
  std::error_code createError;
  std::filesystem::create_directories(dbPath.parent_path(), createError);
  // aleks2 (SHADERS-c1): opened, checked after an unclean exit, and set aside and
  // recreated if damaged; the schema step below is part of that check. A recreated file
  // gets the bundled seed back right after (seed_pipeline_cache).
  g_pipelineCachePath = path;
  sqlite3* opened = nullptr;
  auto ret = sqlite::open_cache(path, &opened, [](sqlite3* candidate) {
    g_pipelineCacheDb = candidate;
    const int code = init_pipeline_cache_schema();
    g_pipelineCacheDb = nullptr;
    return code;
  }, g_pipelineOpenReport, Log);
  if (ret != SQLITE_OK) {
    Log.error("Failed to open pipeline cache database: {}", sqlite3_errstr(ret));
    pipeline_cache_abort();
    return false;
  }
  g_pipelineCacheDb = opened;
  g_pipelineCacheOpened = true;

  ret = sqlite3_prepare_v3(g_pipelineCacheDb,
                           "SELECT config, first_frame_used FROM pipeline_cache "
                           "WHERE type = ? AND config_version = ? "
                           "ORDER BY first_frame_used ASC, rowid ASC",
                           -1, SQLITE_PREPARE_PERSISTENT, &g_pipelineCacheLoadStmt, nullptr);
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare pipeline cache load statement: {}", sqlite3_errmsg(g_pipelineCacheDb));
    pipeline_cache_abort();
    return false;
  }

  ret = sqlite3_prepare_v3(
      g_pipelineCacheDb,
      "INSERT INTO pipeline_cache (type, hash, config_version, config_size, config, first_frame_used) "
      "VALUES (?, ?, ?, ?, ?, ?) "
      "ON CONFLICT(type, hash) DO UPDATE SET "
      "config_version = excluded.config_version, "
      "config_size = excluded.config_size, "
      "config = excluded.config, "
      "first_frame_used = MIN(pipeline_cache.first_frame_used, excluded.first_frame_used)",
      -1, SQLITE_PREPARE_PERSISTENT, &g_pipelineCacheUpsertStmt, nullptr);
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare pipeline cache upsert statement: {}", sqlite3_errmsg(g_pipelineCacheDb));
    pipeline_cache_abort();
    return false;
  }

  // aleks2 (V1-rc2): the bundled list is no longer merged into pipeline_cache.db (seed_pipeline_cache).
  // On the console that merge cost 23.5 s of black screen in aurora_initialize (rc1, empty cache), and the
  // rc1 attempt to write it from the writer thread instead put a 28 MB transaction on the card at the same
  // time as Dawn's stores and the log (the thread dump of the rc1 crash). The list stays in romfs and is
  // read into memory at every start (load_seed_from_bundle); the card keeps only what the game adds.
  if (g_pipelineCacheBroken) {
    return false;
  }

  return true;
}

static void prune_old_pipeline_cache_versions() {
  if (!prepare_pipeline_cache_db()) {
    return;
  }

  const auto clearDelete = fmt::format("DELETE FROM pipeline_cache WHERE type = {} AND config_version < {}",
                                       underlying(ShaderType::Clear), clear::ClearPipelineConfigVersion);
  auto ret = sqlite::exec(g_pipelineCacheDb, clearDelete.c_str());
  if (ret != SQLITE_OK) {
    Log.error("Failed to prune clear pipeline cache rows: {}", sqlite3_errmsg(g_pipelineCacheDb));
    pipeline_cache_abort();
    return;
  }

  const auto gxDelete = fmt::format("DELETE FROM pipeline_cache WHERE type = {} AND config_version < {}",
                                    underlying(ShaderType::GX), gx::GXPipelineConfigVersion);
  ret = sqlite::exec(g_pipelineCacheDb, gxDelete.c_str());
  if (ret != SQLITE_OK) {
    Log.error("Failed to prune GX pipeline cache rows: {}", sqlite3_errmsg(g_pipelineCacheDb));
    pipeline_cache_abort();
    return;
  }

#ifdef AURORA_ENABLE_RMLUI
  const auto rmlDelete = fmt::format("DELETE FROM pipeline_cache WHERE type = {} AND config_version < {}",
                                     underlying(ShaderType::Rml), rmlui::RmlPipelineConfigVersion);
  ret = sqlite::exec(g_pipelineCacheDb, rmlDelete.c_str());
  if (ret != SQLITE_OK) {
    Log.error("Failed to prune RmlUi pipeline cache rows: {}", sqlite3_errmsg(g_pipelineCacheDb));
    pipeline_cache_abort();
  }
#endif
}

static bool write_pipeline_cache_record(const PipelineCacheWrite& write) {
  const auto fail = [&]() {
    sqlite3_reset(g_pipelineCacheUpsertStmt);
    sqlite3_clear_bindings(g_pipelineCacheUpsertStmt);
    return false;
  };

  auto ret = sqlite3_bind_int(g_pipelineCacheUpsertStmt, 1, underlying(write.type));
  if (ret != SQLITE_OK) {
    Log.error("Failed to bind pipeline cache type: {}", sqlite3_errmsg(g_pipelineCacheDb));
    return fail();
  }
  ret = sqlite3_bind_int64(g_pipelineCacheUpsertStmt, 2, static_cast<sqlite3_int64>(write.hash));
  if (ret != SQLITE_OK) {
    Log.error("Failed to bind pipeline cache hash: {}", sqlite3_errmsg(g_pipelineCacheDb));
    return fail();
  }
  ret = sqlite3_bind_int(g_pipelineCacheUpsertStmt, 3, static_cast<int>(write.configVersion));
  if (ret != SQLITE_OK) {
    Log.error("Failed to bind pipeline cache config version: {}", sqlite3_errmsg(g_pipelineCacheDb));
    return fail();
  }
  ret = sqlite3_bind_int(g_pipelineCacheUpsertStmt, 4, static_cast<int>(write.config.size()));
  if (ret != SQLITE_OK) {
    Log.error("Failed to bind pipeline cache config size: {}", sqlite3_errmsg(g_pipelineCacheDb));
    return fail();
  }
  ret = sqlite3_bind_blob64(g_pipelineCacheUpsertStmt, 5, write.config.data(), write.config.size(), SQLITE_TRANSIENT);
  if (ret != SQLITE_OK) {
    Log.error("Failed to bind pipeline cache config blob: {}", sqlite3_errmsg(g_pipelineCacheDb));
    return fail();
  }
  ret = sqlite3_bind_int64(g_pipelineCacheUpsertStmt, 6, static_cast<sqlite3_int64>(write.firstFrameUsed));
  if (ret != SQLITE_OK) {
    Log.error("Failed to bind pipeline cache first-frame-used: {}", sqlite3_errmsg(g_pipelineCacheDb));
    return fail();
  }

  ret = sqlite3_step(g_pipelineCacheUpsertStmt);
  if (ret != SQLITE_DONE) {
    Log.error("Failed to upsert pipeline cache row: {}", sqlite3_errmsg(g_pipelineCacheDb));
    return fail();
  }

  sqlite3_reset(g_pipelineCacheUpsertStmt);
  sqlite3_clear_bindings(g_pipelineCacheUpsertStmt);
  return true;
}

static void pipeline_cache_writer() {
#ifdef TRACY_ENABLE
  tracy::SetThreadName("Pipeline cache writer thread");
#endif

  while (true) {
    std::deque<PipelineCacheWrite> batch;
    {
      std::unique_lock lock{g_pipelineCacheWriterMutex};
      g_pipelineCacheWriterCv.wait(lock,
                                   [] { return g_pipelineCacheWriterStop || !g_pipelineCacheWriteQueue.empty(); });
      if (g_pipelineCacheWriterStop && g_pipelineCacheWriteQueue.empty()) {
        return;
      }
      batch.swap(g_pipelineCacheWriteQueue);
    }

    bool writeFailed = false;
    {
      sqlite::Transaction tx(g_pipelineCacheDb, Log, true);
      if (!tx) {
        Log.error("Failed to begin pipeline cache write transaction");
        writeFailed = true;
        note_pipeline_cache_failure("begin");
      } else {
        for (const auto& write : batch) {
          if (!write_pipeline_cache_record(write)) {
            writeFailed = true;
            note_pipeline_cache_failure("upsert");
            break;
          }
        }

        if (!writeFailed) {
          tx.commit();
          if (sqlite3_get_autocommit(g_pipelineCacheDb) == 0) {
            writeFailed = true;  // still inside the transaction: the commit failed
            note_pipeline_cache_failure("commit");
          }
        }
      }
    }

    if (writeFailed) {
      pipeline_cache_abort();
      return;
    }
  }
}

// aleks2 (V1-rc9): while a loading screen waits for its scene's shaders (screen black, the game thread mostly
// idle), the compile thread may also run on core 0 and at the game's priority; outside that it stays on core 2 at
// low priority. rc8 console: 70-136 ms per cached pipeline during gameplay against ~21 ms at boot -- the thread
// was starved, not the cache cold. The worker applies a change itself, before its next pipeline.
static std::atomic<uint32_t> g_compileBoost{0};
static void apply_compile_boost() {
#ifdef __SWITCH__
  static uint32_t applied = 0;
  static s32 savedPriority = -1;
  const uint32_t want = g_compileBoost.load(std::memory_order_relaxed);
  if (want == applied) {
    return;
  }
  applied = want;
  const Handle self = ::threadGetCurHandle();
  if (want != 0) {
    s32 prio = 0;
    if (R_SUCCEEDED(::svcGetThreadPriority(&prio, self))) {
      savedPriority = prio;
    }
    ::svcSetThreadCoreMask(self, 2, 0b101);
    ::svcSetThreadPriority(self, 0x2C);
  } else {
    ::svcSetThreadCoreMask(self, 2, 0b100);
    if (savedPriority >= 0) {
      ::svcSetThreadPriority(self, static_cast<u32>(savedPriority));
    }
  }
#endif
}

static void pipeline_worker() {
#ifdef TRACY_ENABLE
  tracy::SetThreadName("Pipeline compilation thread");
#endif

  bool hasMore = false;
  while (g_hasPipelineThread || g_pipelinesPerFrame < BuildPipelinesPerFrame) {
    PendingPipeline pending;
    bool fromBackground = false;  // aleks2 (V1-rc12): a warmup pipeline nobody asked for yet
    {
      std::unique_lock lock{g_pipelineMutex};
      if (g_hasPipelineThread) {
        if (!hasMore) {
          g_pipelineQueueCv.wait(lock, [] {
            return !g_pipelineQueue.empty() || !g_backgroundPipelineQueue.empty() || g_pipelineThreadEnd;
          });
        }
      } else if (g_pipelineQueue.empty() && g_backgroundPipelineQueue.empty()) {
        return;
      }
      if (g_pipelineThreadEnd) {
        break;
      }
      if (g_hasPipelineThread && g_pipelineQueue.empty() && webgpu::cache_copy_running()) {
        // aleks2 (V1-rc3): the bundled shader cache is being copied; warming up now would compile it all cold.
        // What the game asks for (g_pipelineQueue) still goes ahead.
        g_pipelineQueueCv.wait_for(lock, std::chrono::milliseconds(100));
        hasMore = true;
        continue;
      }
      if (g_hasPipelineThread && g_pipelineQueue.empty() && g_warmupPaused.load(std::memory_order_relaxed)) {
        // aleks2 (V1-rc12): the game asked to hold the background warmup (low memory); its own requests still run.
        g_pipelineQueueCv.wait_for(lock, std::chrono::milliseconds(200));
        hasMore = true;
        continue;
      }
      auto& source = !g_pipelineQueue.empty() ? g_pipelineQueue : g_backgroundPipelineQueue;
      fromBackground = &source == &g_backgroundPipelineQueue;
      pending = std::move(source.front());
      source.pop_front();
    }
    apply_compile_boost();
    const auto compileStart = std::chrono::steady_clock::now();
    double restMs = 0.0;
    auto result = pending.create();
    const double compileMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - compileStart).count();
    const auto pipelineCount = result.pipeline_count();
    {
      std::lock_guard lock{g_pipelineMutex};
      if (fromBackground && !g_warmupKeepAll.load(std::memory_order_relaxed) && !scene_uses_recipe(pending.configKey)) {
        ++g_warmupReleased;  // V1-rc12: in the disk cache now; `result` is released when this iteration ends
      } else {
        g_pipelines.try_emplace(pending.hash, std::move(result));
      }
      g_pendingPipelines.erase(pending.hash);
      if (g_warmupPending.erase(pending.hash) != 0) {
        ++g_warmupDone;
        g_warmupCompileMs += compileMs;
        g_warmupCompileMaxMs = std::max(g_warmupCompileMaxMs, compileMs);
        // aleks2 (SHADERS-c2): background pacing (aurora_set_shader_warmup_duty). Rest in
        // proportion to the compile just done, but not if the game is waiting for one.
        const uint32_t duty = g_warmupDuty.load(std::memory_order_relaxed);
        if (duty < 100 && g_pipelineQueue.empty()) {
          restMs = std::min(compileMs * (100.0 - duty) / duty, 2000.0);
        }
      }
      hasMore = !g_pipelineQueue.empty() || !g_backgroundPipelineQueue.empty();
    }
    if (!g_hasPipelineThread) {
      g_pipelinesPerFrame += std::max(1u, pipelineCount);
    }
    notify_pipeline_ready(true, pipelineCount);
    if (restMs > 0.0) {
      // Woken early by anything the game needs (or shutdown).
      std::unique_lock lock{g_pipelineMutex};
      g_pipelineQueueCv.wait_for(lock, std::chrono::duration<double, std::milli>(restMs),
                                 [] { return !g_pipelineQueue.empty() || g_pipelineThreadEnd.load(); });
    }
  }
}

template <typename PipelineConfig>
static size_t load_pipeline_cache_entries(ShaderType type, uint32_t configVersion) {
  if (!prepare_pipeline_cache_db()) {
    return 0;
  }

  auto ret = sqlite3_bind_int(g_pipelineCacheLoadStmt, 1, underlying(type));
  if (ret != SQLITE_OK) {
    Log.error("Failed to bind pipeline cache load type: {}", sqlite3_errmsg(g_pipelineCacheDb));
    pipeline_cache_abort();
    return 0;
  }
  ret = sqlite3_bind_int(g_pipelineCacheLoadStmt, 2, static_cast<int>(configVersion));
  if (ret != SQLITE_OK) {
    Log.error("Failed to bind pipeline cache load config version: {}", sqlite3_errmsg(g_pipelineCacheDb));
    pipeline_cache_abort();
    return 0;
  }

  size_t acceptedRows = 0;
  while ((ret = sqlite3_step(g_pipelineCacheLoadStmt)) == SQLITE_ROW) {
    const auto* configBlob = static_cast<const uint8_t*>(sqlite3_column_blob(g_pipelineCacheLoadStmt, 0));
    const auto configSize = sqlite3_column_bytes(g_pipelineCacheLoadStmt, 0);
    const auto firstFrameUsed = static_cast<uint32_t>(sqlite3_column_int64(g_pipelineCacheLoadStmt, 1));
    if (configSize != static_cast<int>(sizeof(PipelineConfig)) || (configSize != 0 && configBlob == nullptr)) {
      continue;
    }

    PipelineConfig config;
    std::memcpy(&config, configBlob, sizeof(config));
    if (config.version != configVersion) {
      continue;
    }

    remember_pipeline_config(type, config, firstFrameUsed, false);
    ++acceptedRows;
  }

  if (ret != SQLITE_DONE) {
    Log.error("Failed to read pipeline cache rows: {}", sqlite3_errmsg(g_pipelineCacheDb));
    pipeline_cache_abort();
  }

  sqlite3_reset(g_pipelineCacheLoadStmt);
  sqlite3_clear_bindings(g_pipelineCacheLoadStmt);
  return acceptedRows;
}

// V1-c3: without a usable pipeline_cache.db (V1-c1/c2 on the console) the shipped list was
// never read -- the seed is merged into that file. Read it straight into memory then, so
// the warmup still compiles it (into Dawn's cache if that one works, else for this session).
template <typename PipelineConfig>
static bool remember_seed_row(ShaderType type, uint32_t configVersion, int rowVersion, const void* blob, int size,
                              uint32_t firstFrameUsed) {
  if (rowVersion != static_cast<int>(configVersion) || size != static_cast<int>(sizeof(PipelineConfig)) ||
      blob == nullptr) {
    return false;
  }
  PipelineConfig config;
  std::memcpy(&config, blob, sizeof(config));
  if (config.version != configVersion) {
    return false;
  }
  remember_pipeline_config(type, config, firstFrameUsed, false);
  return true;
}

static size_t load_seed_from_bundle(bool cacheUsable) {
  const auto loadStart = std::chrono::steady_clock::now();
  const auto seedPath = pipeline_cache_seed_path();
  sqlite3* seedDb = open_pipeline_cache_seed_db(seedPath);
  if (seedDb == nullptr) {
    if (!cacheUsable) {
      Log.warn("No pipeline cache and no bundled list at '{}': nothing to prepare", seedPath);
    }
    return 0;
  }
  // The ORDER BY sorts 28 MB of rows: in memory, not in temporary files on the card.
  sqlite::exec(seedDb, "PRAGMA temp_store=MEMORY;");
  sqlite3_stmt* stmt = nullptr;
  size_t accepted = 0;
  size_t skipped = 0;
  uint64_t signature = 0;
  const uint32_t limit = seed_limit();
  if (sqlite3_prepare_v2(seedDb,
                         "SELECT type, hash, config_version, config, first_frame_used FROM pipeline_cache "
                         "ORDER BY first_frame_used ASC, rowid ASC",
                         -1, &stmt, nullptr) == SQLITE_OK) {
    while (sqlite3_step(stmt) == SQLITE_ROW && (limit == 0 || accepted + skipped < limit)) {
      const int type = sqlite3_column_int(stmt, 0);
      const auto hash = sqlite3_column_int64(stmt, 1);
      const int version = sqlite3_column_int(stmt, 2);
      const void* blob = sqlite3_column_blob(stmt, 3);
      const int size = sqlite3_column_bytes(stmt, 3);
      const auto first = static_cast<uint32_t>(std::clamp<sqlite3_int64>(sqlite3_column_int64(stmt, 4), 0, UINT32_MAX));
      bool ok = false;
      if (type == static_cast<int>(ShaderType::GX)) {
        ok = remember_seed_row<gx::PipelineConfig>(ShaderType::GX, gx::GXPipelineConfigVersion, version, blob, size, first);
      } else if (type == static_cast<int>(ShaderType::Clear)) {
        ok = remember_seed_row<clear::PipelineConfig>(ShaderType::Clear, clear::ClearPipelineConfigVersion, version, blob,
                                                      size, first);
#ifdef AURORA_ENABLE_RMLUI
      } else if (type == static_cast<int>(ShaderType::Rml)) {
        ok = remember_seed_row<rmlui::PipelineConfig>(ShaderType::Rml, rmlui::RmlPipelineConfigVersion, version, blob,
                                                      size, first);
#endif
      }
      if (ok) {
        ++accepted;
        const uint64_t rowKey[3] = {static_cast<uint64_t>(type), static_cast<uint64_t>(hash),
                                    static_cast<uint64_t>(version)};
        signature += XXH3_64bits(rowKey, sizeof(rowKey));
      } else {
        ++skipped;
      }
    }
    sqlite3_finalize(stmt);
  } else {
    Log.warn("Could not read the bundled list '{}': {}", seedPath, sqlite3_errmsg(seedDb));
  }
  sqlite3_close(seedDb);
  g_seedRowsMerged = static_cast<uint32_t>(accepted);
  g_seedSignature = accepted > 0 ? signature : 0;
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loadStart).count();
  if (cacheUsable) {
    Log.info("Bundled list read into memory from '{}' ({} recipes, {} skipped, signature {:016x}) in {:.0f} ms",
             seedPath, accepted, skipped, g_seedSignature, ms);
  } else {
    Log.warn("pipeline_cache.db unusable: bundled list read into memory from '{}' ({} recipes, {} skipped, "
             "signature {:016x}) in {:.0f} ms; compiled this session only unless the Dawn cache works",
             seedPath, accepted, skipped, g_seedSignature, ms);
  }
  return accepted;
}

static size_t load_pipeline_cache() {
  if (!prepare_pipeline_cache_db()) {
    return load_seed_from_bundle(false);
  }
  prune_old_pipeline_cache_versions();
  if (g_pipelineCacheBroken) {
    return 0;
  }

  size_t acceptedRows = 0;
#ifdef AURORA_ENABLE_RMLUI
  acceptedRows += load_pipeline_cache_entries<rmlui::PipelineConfig>(ShaderType::Rml, rmlui::RmlPipelineConfigVersion);
#endif
  acceptedRows +=
      load_pipeline_cache_entries<clear::PipelineConfig>(ShaderType::Clear, clear::ClearPipelineConfigVersion);
  acceptedRows += load_pipeline_cache_entries<gx::PipelineConfig>(ShaderType::GX, gx::GXPipelineConfigVersion);
  // V1-rc2: the bundled list, from romfs into memory (not merged into the card any more).
  acceptedRows += load_seed_from_bundle(true);
  return acceptedRows;
}

static void start_pipeline_cache_writer() {
  if (!prepare_pipeline_cache_db()) {
    return;
  }

  g_pipelineCacheWriterStop = false;
  g_pipelineCacheWriterThread = std::thread(pipeline_cache_writer);
}

static void stop_pipeline_cache_writer() {
  if (g_pipelineCacheWriterThread.joinable()) {
    {
      std::lock_guard lock{g_pipelineCacheWriterMutex};
      g_pipelineCacheWriterStop = true;
    }
    g_pipelineCacheWriterCv.notify_one();
    g_pipelineCacheWriterThread.join();
  } else {
    g_pipelineCacheWriterStop = false;
  }

  g_pipelineCacheWriteQueue.clear();
}

PipelineRef find_pipeline(const clear::PipelineConfig& config, const RenderTargetLayout& layout) {
  remember_pipeline_config(ShaderType::Clear, config, current_frame(), true);
  return resolve_pipeline(ShaderType::Clear, config, layout, PipelinePriority::Normal);
}

PipelineRef find_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout) {
  remember_pipeline_config(ShaderType::GX, config, current_frame(), true);
  return resolve_pipeline(ShaderType::GX, config, layout, PipelinePriority::Normal);
}

#ifdef AURORA_ENABLE_RMLUI
PipelineRef find_pipeline(const rmlui::PipelineConfig& config) {
  remember_pipeline_config(ShaderType::Rml, config, 0, true);
  return resolve_pipeline(ShaderType::Rml, config, {}, PipelinePriority::Blocking);
}
#endif

// aleks2 (V1-rc1): cache/pipeline_scenes.bin: magic, the last scene, then per recipe its key and
// scenes. stdio and rename, as the other cache files on this platform.
static std::string pipeline_scenes_path() {
  return io::fs_path_to_string(io::fs_path_from_string(g_config.cachePath) / "pipeline_scenes.bin");
}

static void load_pipeline_scenes() {
  const auto path = pipeline_scenes_path();
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    return;
  }
  uint32_t header[3] = {};
  size_t loaded = 0;
  if (std::fread(header, sizeof(header), 1, f) == 1 && header[0] == PipelineScenesMagic) {
    std::lock_guard lock{g_pipelineMutex};
    g_lastSceneTag = header[1];
    for (uint32_t i = 0; i < header[2]; ++i) {
      uint64_t key = 0;
      uint8_t count = 0;
      if (std::fread(&key, sizeof(key), 1, f) != 1 || std::fread(&count, 1, 1, f) != 1 ||
          count > MaxScenesPerRecipe) {
        break;
      }
      std::vector<uint32_t> scenes(count);
      if (count != 0 && std::fread(scenes.data(), sizeof(uint32_t), count, f) != count) {
        break;
      }
      g_pipelineScenes[key] = std::move(scenes);
      ++loaded;
    }
  }
  std::fclose(f);
  Log.info("Pipeline scenes: {} recipes with a scene, last scene {:08x}", loaded, g_lastSceneTag);
}

static void save_pipeline_scenes() {
  std::vector<uint8_t> out;
  {
    std::lock_guard lock{g_pipelineMutex};
    if (!g_pipelineScenesDirty) {
      return;
    }
    g_pipelineScenesDirty = false;
    const uint32_t header[3] = {PipelineScenesMagic, g_sceneTag.load(std::memory_order_relaxed),
                                static_cast<uint32_t>(g_pipelineScenes.size())};
    out.reserve(sizeof(header) + g_pipelineScenes.size() * 16);
    const auto put = [&out](const void* p, size_t n) {
      const auto* b = static_cast<const uint8_t*>(p);
      out.insert(out.end(), b, b + n);
    };
    put(header, sizeof(header));
    for (const auto& [key, scenes] : g_pipelineScenes) {
      const uint64_t k = key;
      const uint8_t count = static_cast<uint8_t>(scenes.size());
      put(&k, sizeof(k));
      put(&count, 1);
      put(scenes.data(), scenes.size() * sizeof(uint32_t));
    }
  }
  const auto path = pipeline_scenes_path();
  const auto tmp = path + ".tmp";
  std::FILE* f = std::fopen(tmp.c_str(), "wb");
  if (f == nullptr) {
    return;
  }
  const bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
  std::fclose(f);
  if (ok) {
    std::remove(path.c_str());
    std::rename(tmp.c_str(), path.c_str());
  }
}

static void set_shader_scene(const char* name) {
  uint32_t tag = 0;
  if (name != nullptr && name[0] != '\0') {
    tag = static_cast<uint32_t>(xxh3_hash_s(name, std::strlen(name), 0x5343454eu));
    tag = tag == 0 ? 1 : tag;
  }
  const uint32_t previous = g_sceneTag.exchange(tag, std::memory_order_relaxed);
  if (tag == 0 || tag == previous) {
    return;
  }
  uint32_t moved = 0;
  {
    std::lock_guard lock{g_pipelineMutex};
    g_pipelineScenesDirty = true;  // the last scene is part of the file
    const auto inScene = [tag](const PendingPipeline& pending) {
      const auto it = g_pipelineScenes.find(pending.configKey);
      return it != g_pipelineScenes.end() && std::ranges::find(it->second, tag) != it->second.end();
    };
    const auto split = std::stable_partition(g_backgroundPipelineQueue.begin(), g_backgroundPipelineQueue.end(),
                                             inScene);
    moved = static_cast<uint32_t>(std::distance(g_backgroundPipelineQueue.begin(), split));
    g_scenePromoted = moved;
  }
  Log.info("Scene '{}' ({:08x}): {} queued recipes moved to the front of the warmup", name, tag, moved);
  // Small and rewritten whole; this runs while the game loads the scene.
  save_pipeline_scenes();
  if (g_warmupScoped.load(std::memory_order_relaxed)) {
    warm_scene_recipes(tag, name);
  }
}

static std::atomic<uint32_t> g_pipelineGeneration{0};

uint32_t pipeline_generation() noexcept { return g_pipelineGeneration.load(std::memory_order_acquire); }

void rebuild_pipeline_cache() {
  ZoneScoped;
  g_pipelineGeneration.fetch_add(1, std::memory_order_release);
  const auto scene = scene_render_target_layout();
  auto offscreen = scene;
  offscreen.colorAttachmentCount = 1;
  offscreen.sampleCount = 1;
  detail::finalize_render_target_layout(offscreen);
  g_pipelineLayoutKey = scene.key;

  std::vector<KnownPipeline> known;
  {
    std::lock_guard lock{g_pipelineMutex};
    // aleks2 (V1-rc1): the warmup still queued for the previous layout is dropped. It used to keep
    // compiling (thousands of pipelines, minutes of the core the render worker shares) after a
    // layout change -- Flip Grip on and off -- for a layout no longer drawn. What the game asks
    // for again is queued again on demand.
    for (const auto& pending : g_backgroundPipelineQueue) {
      g_pendingPipelines.erase(pending.hash);
      g_warmupPending.erase(pending.hash);
      --queuedPipelines;
      ++g_warmupDropped;
    }
    g_backgroundPipelineQueue.clear();
    // A new warmup pass: what the previous one left queued still counts.
    ++g_warmupGeneration;
    g_warmupQueued = static_cast<uint32_t>(g_warmupPending.size());
    g_warmupDone = 0;
    g_warmupCompileMs = 0.0;
    g_warmupCompileMaxMs = 0.0;
    known.reserve(g_knownPipelines.size());
    std::vector<KnownPipeline> lastScene;
    for (const auto& [key, pipeline] : g_knownPipelines) {
      const auto scenes = g_pipelineScenes.find(key);
      const bool inLast = g_lastSceneTag != 0 && scenes != g_pipelineScenes.end() &&
                          std::ranges::find(scenes->second, g_lastSceneTag) != scenes->second.end();
      (inLast ? lastScene : known).push_back(pipeline);
    }
    // V1-rc1: the scene this console was last in goes first, then everything in order of first use.
    std::ranges::sort(lastScene, {}, &KnownPipeline::firstFrameUsed);
    std::ranges::sort(known, {}, &KnownPipeline::firstFrameUsed);
    known.insert(known.begin(), lastScene.begin(), lastScene.end());
  }
  for (const auto& pipeline : known) {
    std::visit(
        [&](const auto& config) {
          resolve_pipeline(pipeline.type, config, scene, PipelinePriority::Background);
          if (offscreen.key != scene.key) {
            resolve_pipeline(pipeline.type, config, offscreen, PipelinePriority::Background);
          }
        },
        pipeline.config);
  }
}

// V1-rc5: queues `pipelines` in the background for the scene layout (and the offscreen one).
static void warm_known_pipelines(const std::vector<KnownPipeline>& pipelines) {
  const auto scene = scene_render_target_layout();
  auto offscreen = scene;
  offscreen.colorAttachmentCount = 1;
  offscreen.sampleCount = 1;
  detail::finalize_render_target_layout(offscreen);
  for (const auto& pipeline : pipelines) {
    std::visit(
        [&](const auto& config) {
          resolve_pipeline(pipeline.type, config, scene, PipelinePriority::Background);
          if (offscreen.key != scene.key) {
            resolve_pipeline(pipeline.type, config, offscreen, PipelinePriority::Background);
          }
        },
        pipeline.config);
  }
}

static void warm_scene_recipes(uint32_t tag, const char* name) {
  std::vector<KnownPipeline> inScene;
  {
    std::lock_guard lock{g_pipelineMutex};
    for (const auto& [key, scenes] : g_pipelineScenes) {
      if (std::ranges::find(scenes, tag) == scenes.end()) {
        continue;
      }
      if (const auto it = g_knownPipelines.find(key); it != g_knownPipelines.end()) {
        inScene.push_back(it->second);
      }
    }
  }
  std::ranges::sort(inScene, {}, &KnownPipeline::firstFrameUsed);
  const auto warmupQueued = [] {
    std::lock_guard lock{g_pipelineMutex};
    return g_warmupQueued;
  };
  const uint32_t before = warmupQueued();
  warm_known_pipelines(inScene);  // compiled ones are skipped
  const uint32_t after = warmupQueued();
  Log.info("Scene '{}' ({:08x}): {} recipes known, {} queued in the background (not compiled yet)", name, tag,
           inScene.size(), after >= before ? after - before : 0u);
}

static uint32_t scope_shader_warmup(uint32_t sharedRecipes) {
  absl::flat_hash_set<HashType> keep;
  uint32_t kept = 0;
  uint32_t deferred = 0;
  size_t neighbourCount = 0;
  {
    std::lock_guard lock{g_pipelineMutex};
    // The scenes that share most recipes with the last one: the stages next to it, as a rule.
    absl::flat_hash_map<uint32_t, uint32_t> shared;
    for (const auto& [key, scenes] : g_pipelineScenes) {
      if (g_lastSceneTag == 0 || std::ranges::find(scenes, g_lastSceneTag) == scenes.end()) {
        continue;
      }
      keep.insert(key);
      for (const uint32_t scene : scenes) {
        if (scene != g_lastSceneTag) {
          ++shared[scene];
        }
      }
    }
    std::vector<std::pair<uint32_t, uint32_t>> neighbours(shared.begin(), shared.end());
    std::ranges::sort(neighbours, std::greater{}, &std::pair<uint32_t, uint32_t>::second);
    neighbours.resize(std::min<size_t>(neighbours.size(), 3));
    neighbourCount = neighbours.size();
    // The recipes used in most scenes (Link, the HUD, effects), then the first ones of any session (menus).
    std::vector<std::pair<HashType, size_t>> byScenes;
    for (const auto& [key, scenes] : g_pipelineScenes) {
      if (std::ranges::any_of(neighbours, [&](const auto& n) { return std::ranges::find(scenes, n.first) != scenes.end(); })) {
        keep.insert(key);
      } else if (scenes.size() >= 2) {
        byScenes.emplace_back(key, scenes.size());
      }
    }
    std::ranges::sort(byScenes, std::greater{}, &std::pair<HashType, size_t>::second);
    for (size_t i = 0; i < byScenes.size() && i < sharedRecipes; ++i) {
      keep.insert(byScenes[i].first);
    }
    std::vector<std::pair<uint32_t, HashType>> first;
    first.reserve(g_knownPipelines.size());
    for (const auto& [key, pipeline] : g_knownPipelines) {
      first.emplace_back(pipeline.firstFrameUsed, key);
    }
    const size_t firstCount = std::min<size_t>(first.size(), sharedRecipes / 2);
    std::ranges::partial_sort(first, first.begin() + static_cast<ptrdiff_t>(firstCount));
    for (size_t i = 0; i < firstCount; ++i) {
      keep.insert(first[i].second);
    }
    // What is not kept leaves the warmup, as on a layout change: the game queues it when it draws it.
    std::deque<PendingPipeline> remaining;
    for (auto& pending : g_backgroundPipelineQueue) {
      if (keep.contains(pending.configKey)) {
        remaining.emplace_back(std::move(pending));
        ++kept;
        continue;
      }
      g_pendingPipelines.erase(pending.hash);
      if (g_warmupPending.erase(pending.hash) != 0 && g_warmupQueued != 0) {
        --g_warmupQueued;
      }
      --queuedPipelines;
      ++deferred;
    }
    g_backgroundPipelineQueue = std::move(remaining);
    g_warmupScoped = true;
    g_gpuCachePrunePending = false;  // an emptied queue must not prune what this session did not use
  }
  Log.info("Warmup scoped: last scene {:08x}, {} neighbour scenes, {} recipes kept; {} queued pipelines kept, {} "
           "left to compile on demand",
           g_lastSceneTag, neighbourCount, keep.size(), kept, deferred);
  return kept;
}

static void warm_all_shaders() {
  g_warmupScoped = false;
  Log.info("Warmup: every known recipe queued again");
  rebuild_pipeline_cache();
}

void initialize_pipeline_cache() {
  g_pipelineGeneration.fetch_add(1, std::memory_order_release);
  g_warmupScoped = false;
  g_pipelineCacheBroken = false;
  g_pipelineSawDamage = false;
  g_pipelineCacheOpened = false;
  g_seedRowsMerged = 0;
  g_seedSignature = 0;
  g_pipelineCacheWriterStop = false;
  g_pipelineThreadEnd = false;
  g_gpuCachePrunePending = false;

  if (webgpu::g_backendType == wgpu::BackendType::WebGPU) {
    g_hasPipelineThread = false;
  } else {
    g_hasPipelineThread = true;
#ifdef __SWITCH__
    // Eight megabytes: the shader compiler is deeply recursive.
    const bool started = g_pipelineThread.start(8u * 1024u * 1024u, [] {
      thread::set_current(thread::Options{.name = "Pipeline compilation",
                                          .priority = thread::Priority::Low,
                                          .affinity = thread::Affinity::Compile});
      pipeline_worker();
    });
    if (!started) {
      Log.error("Pipeline worker thread creation failed; using synchronous compile");
      g_hasPipelineThread = false;
    }
#else
    g_pipelineThread = std::thread(pipeline_worker);
#endif
  }

  const size_t loadedCount = load_pipeline_cache();
  load_pipeline_scenes();
  rebuild_pipeline_cache();
#ifndef __SWITCH__
  if (!g_pipelineCacheBroken && loadedCount > 0) {
    g_gpuCachePrunePending = true;
  }
#else
  // aleks2 (V1-rc5): never pruned on the Switch. The prune keeps only the keys this session used, which with
  // the per-scene warmup (scope_shader_warmup) deleted most of the bundled cache -- and a pruned cache still
  // matched its .seed, so it was never copied again -- then ran a VACUUM of ~75 MB on the card under the cache
  // mutex. The bundled cache is the console's cache: it stays whole.
  (void)loadedCount;
#endif

  if (!g_pipelineCacheBroken) {
    start_pipeline_cache_writer();
  }
}

void shutdown_pipeline_cache() {
  save_pipeline_scenes();
  g_pipelineGeneration.fetch_add(1, std::memory_order_release);
  if (g_hasPipelineThread) {
    g_pipelineThreadEnd = true;
    g_pipelineQueueCv.notify_all();
    g_pipelineReadyCv.notify_all();
    g_pipelineThread.join();
  }
  g_hasPipelineThread = false;

  stop_pipeline_cache_writer();
  pipeline_cache_abort();
  g_pipelineCacheBroken = false;
  g_pipelinesPerFrame = 0;
  g_gpuCachePrunePending = false;
  g_pipelines.clear();
  g_knownPipelines.clear();
  g_pipelineLayoutKey.reset();
  g_pipelineQueue.clear();
  g_backgroundPipelineQueue.clear();
  g_pendingPipelines.clear();
  g_warmupPending.clear();
  g_warmupQueued = 0;
  g_warmupDone = 0;
  g_pipelineScenes.clear();
  g_pipelineScenesDirty = false;

  queuedPipelines = 0;
  createdPipelines = 0;
}

void begin_pipeline_frame() {
  if (!g_hasPipelineThread) {
    g_pipelinesPerFrame = 0;
  }
  if (g_pipelineLayoutKey != scene_render_target_layout().key) {
    rebuild_pipeline_cache();
  }
}

void end_pipeline_frame() {
  if (!g_hasPipelineThread) {
    pipeline_worker();
  }
}

bool get_pipeline(PipelineRef ref, CompiledPipeline& pipeline) {
  std::lock_guard guard{g_pipelineMutex};
  const auto it = g_pipelines.find(ref);
#if defined(AURORA_GFX_DEKO3D)
  if (it == g_pipelines.end() || (!it->second.main && it->second.dkMain == nullptr)) {
#else
  if (it == g_pipelines.end() || !it->second.main) {
#endif
    return false;
  }
  pipeline = it->second;
  return true;
}

} // namespace aurora::gfx

namespace aurora::gfx {
// aleks2 (SHADERS-c1): include/aurora/shader_cache.h.
static void copy_text(char* out, size_t size, const std::string& text) {
  const size_t n = std::min(size - 1, text.size());
  std::memcpy(out, text.data(), n);
  out[n] = 0;
}

static void fill_shader_cache_status(AuroraShaderCacheStatus& st) {
  st = {};
  {
    std::lock_guard lock{g_pipelineMutex};
    st.warmupGeneration = g_warmupGeneration;
    st.warmupQueued = g_warmupQueued;
    st.warmupDone = g_warmupDone;
    st.warmupCompileMs = g_warmupCompileMs;
    st.warmupCompileMaxMs = g_warmupCompileMaxMs;
    st.knownPipelines = static_cast<uint32_t>(g_knownPipelines.size());
    st.layoutKey = g_pipelineLayoutKey.value_or(0);
    st.recipesWithScene = static_cast<uint32_t>(g_pipelineScenes.size());
    st.lastSceneTag = g_lastSceneTag;
    st.scenePromoted = g_scenePromoted;
    st.warmupDropped = g_warmupDropped;
  }
  st.sceneTag = g_sceneTag.load(std::memory_order_relaxed);
  st.gxSourceMs = static_cast<double>(g_gxSourceNs.load(std::memory_order_relaxed)) / 1e6;
  st.gxModuleMs = static_cast<double>(g_gxModuleNs.load(std::memory_order_relaxed)) / 1e6;
  st.gxPipelineMs = static_cast<double>(g_gxPipelineNs.load(std::memory_order_relaxed)) / 1e6;
  st.gxPipelinesTimed = g_gxPipelinesTimed.load(std::memory_order_relaxed);
  st.gxModulesReused = g_gxModulesReused.load(std::memory_order_relaxed);
  st.seedRowsMerged = g_seedRowsMerged;
  st.seedSignature = g_seedSignature;
  st.pipelineSchema = PipelineCacheSchema;
  st.gxConfigVersion = gx::GXPipelineConfigVersion;
  st.clearConfigVersion = clear::ClearPipelineConfigVersion;
#ifdef AURORA_ENABLE_RMLUI
  st.rmlConfigVersion = rmlui::RmlPipelineConfigVersion;
#endif
  st.dawnSchema = 2;  // CACHE_SCHEMA in webgpu/gpu_cache.cpp
#ifdef AURORA_PIPELINE_IDENTITY
  st.runtimeIdentity = AURORA_PIPELINE_IDENTITY;
#endif
  st.warmupDuty = g_warmupDuty.load(std::memory_order_relaxed);

  st.pipelineCacheOk = g_pipelineCacheOpened && !g_pipelineCacheBroken && g_pipelineCacheDb != nullptr;
  st.pipelineCacheDirty = g_pipelineOpenReport.dirtyAtOpen;
  st.pipelineCacheRepaired = g_pipelineOpenReport.repaired;
  st.pipelineCheckMs = g_pipelineOpenReport.checkMs;
  copy_text(st.pipelineJournal, sizeof(st.pipelineJournal), g_pipelineOpenReport.journalMode);

  const auto dawn = webgpu::cache_counters();
  st.dawnHits = dawn.hits;
  st.dawnMisses = dawn.misses;
  st.dawnStores = dawn.stores;
  st.dawnStoreMsTotal = dawn.storeMsTotal;
  st.dawnStoreMsMax = dawn.storeMsMax;
  st.dawnCacheOk = dawn.opened;
  st.dawnCacheDirty = dawn.dirty;
  st.dawnCacheRepaired = dawn.repaired;
  st.dawnCheckMs = dawn.checkMs;
  copy_text(st.dawnJournal, sizeof(st.dawnJournal), dawn.journal);
}

static int64_t export_pipeline_cache(const char* path) {
  if (path == nullptr || g_pipelineCacheDb == nullptr || g_pipelineCacheBroken) {
    return -1;
  }
  sqlite3* out = nullptr;
  const auto& options = sqlite::cache_options(Log);
  if (sqlite3_open_v2(path, &out, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, options.vfs) != SQLITE_OK) {
    Log.error("Shader Tour export: cannot create '{}': {}", path, out != nullptr ? sqlite3_errmsg(out) : "?");
    sqlite3_close(out);
    return -1;
  }
  // The connection is serialized (SQLITE_THREADSAFE=1), so the writer thread just waits.
  int64_t rows = -1;
  sqlite3_backup* backup = sqlite3_backup_init(out, "main", g_pipelineCacheDb, "main");
  if (backup != nullptr) {
    sqlite3_backup_step(backup, -1);
    if (sqlite3_backup_finish(backup) == SQLITE_OK) {
      sqlite::exec(out, "PRAGMA journal_mode=delete;");
      sqlite::exec(out, "SELECT COUNT(*) FROM pipeline_cache", [&rows](int, char** argv, char**) {
        rows = argv[0] != nullptr ? std::strtoll(argv[0], nullptr, 10) : -1;
      });
    }
  }
  if (rows < 0) {
    Log.error("Shader Tour export to '{}' failed: {}", path, sqlite3_errmsg(out));
  } else {
    Log.info("Shader Tour export: {} pipeline recipes -> {}", rows, path);
  }
  sqlite3_close(out);
  return rows;
}
} // namespace aurora::gfx

void aurora_get_shader_cache_status(AuroraShaderCacheStatus* out) {
  if (out != nullptr) {
    aurora::gfx::fill_shader_cache_status(*out);
  }
}

size_t aurora_get_driver_identity(char* out, size_t size) {
  const std::string id = aurora::webgpu::driver_identity();
  if (out != nullptr && size > 0) {
    aurora::gfx::copy_text(out, size, id);
  }
  return id.size();
}

int64_t aurora_export_pipeline_cache(const char* path) { return aurora::gfx::export_pipeline_cache(path); }

void aurora_set_shader_scene(const char* name) { aurora::gfx::set_shader_scene(name); }

uint32_t aurora_scope_shader_warmup(uint32_t sharedRecipes) { return aurora::gfx::scope_shader_warmup(sharedRecipes); }

void aurora_warm_all_shaders(void) { aurora::gfx::warm_all_shaders(); }

void aurora_shader_set_compile_boost(uint32_t on) {
  aurora::gfx::g_compileBoost.store(on != 0 ? 1u : 0u, std::memory_order_relaxed);
}

uint32_t aurora_shader_scene_pending(void) {
  const uint32_t tag = aurora::gfx::g_sceneTag.load(std::memory_order_relaxed);
  std::lock_guard lock{aurora::gfx::g_pipelineMutex};
  uint32_t pending = static_cast<uint32_t>(aurora::gfx::g_pipelineQueue.size());
  if (tag == 0) {
    return pending;
  }
  for (const auto& queued : aurora::gfx::g_backgroundPipelineQueue) {
    const auto it = aurora::gfx::g_pipelineScenes.find(queued.configKey);
    if (it != aurora::gfx::g_pipelineScenes.end() && std::ranges::find(it->second, tag) != it->second.end()) {
      ++pending;
    }
  }
  return pending;
}

uint32_t aurora_scoped_warmup_pending(void) {
  if (!aurora::gfx::g_warmupScoped.load(std::memory_order_relaxed)) {
    return 0;
  }
  std::lock_guard lock{aurora::gfx::g_pipelineMutex};
  return static_cast<uint32_t>(aurora::gfx::g_warmupPending.size());
}

void aurora_shader_set_warmup_paused(uint32_t on) {
  aurora::gfx::g_warmupPaused.store(on != 0, std::memory_order_relaxed);
  if (on == 0) {
    aurora::gfx::g_pipelineQueueCv.notify_one();
  }
}

void aurora_shader_set_warmup_keep_all(uint32_t on) {
  aurora::gfx::g_warmupKeepAll.store(on != 0, std::memory_order_relaxed);
}

uint32_t aurora_pipeline_live_count(void) {
  std::lock_guard lock{aurora::gfx::g_pipelineMutex};
  return static_cast<uint32_t>(aurora::gfx::g_pipelines.size());
}

uint32_t aurora_shader_warmup_released(void) {
  std::lock_guard lock{aurora::gfx::g_pipelineMutex};
  return aurora::gfx::g_warmupReleased;
}

void aurora_set_shader_warmup_duty(uint32_t percent) {
  aurora::gfx::g_warmupDuty.store(std::clamp<uint32_t>(percent, 10, 100), std::memory_order_relaxed);
}
