#pragma once
// aleks2 (H31): a bounded memo of "configuration -> resolved handle" for the GX
// draw path. Owned by the FIFO thread. See docs/switch-h31-pipeline-memo.md for why
// replaying a stored result is equivalent to calling the full lookup again.
//
// Deliberately free of Aurora/Dawn types so it can be tested on a PC with a plain
// compiler (platforms/switch/tools/test-pipeline-memo.sh).
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

namespace aurora::gx {

struct LookupMemoStats {
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t evictions = 0;
  uint32_t occupied = 0;
  uint32_t maxOccupied = 0;
  uint32_t capacity = 0;
};

// Sets must be a power of two. Replacement is LRU inside a set, by a monotonic
// stamp, so the same sequence of calls always evicts the same entry.
template <class Config, class Ref, class Info, size_t Sets, size_t Ways>
class LookupMemo {
  static_assert((Sets & (Sets - 1)) == 0, "Sets must be a power of two");
  static_assert(std::has_unique_object_representations_v<Config>,
                "Config is compared and hashed as raw bytes");

public:
  static constexpr size_t kCapacity = Sets * Ways;

  // A hit needs the same kind, layout, version, hash AND byte-identical config; a
  // generation different from the one the memo was filled under empties it first.
  // Counts the hit or the miss. Writes ref/info only on a hit. No locks, and no
  // allocation: the table is reserved by the first insert().
  bool find(uint32_t kind, uint64_t layoutKey, uint32_t generation, uint32_t configVersion, uint64_t hash,
            const Config& config, Ref& ref, Info& info) noexcept {
    sync_generation(generation);
    if (mEntries != nullptr) {
      Entry* set = &mEntries[(hash & (Sets - 1)) * Ways];
      for (size_t way = 0; way < Ways; ++way) {
        Entry& e = set[way];
        if (e.valid && e.hash == hash && e.kind == kind && e.layoutKey == layoutKey &&
            e.configVersion == configVersion && std::memcmp(&e.config, &config, sizeof(Config)) == 0) {
          e.stamp = ++mClock;
          ref = e.ref;
          info = e.info;
          bump(mHits);
          return true;
        }
      }
    }
    bump(mMisses);
    return false;
  }

  // Stores a result that came from the full lookup. May allocate (once).
  void insert(uint32_t kind, uint64_t layoutKey, uint32_t generation, uint32_t configVersion, uint64_t hash,
              const Config& config, const Ref& ref, const Info& info) {
    sync_generation(generation);
    if (mEntries == nullptr) {
      mEntries = std::make_unique<Entry[]>(kCapacity);
    }
    Entry* set = &mEntries[(hash & (Sets - 1)) * Ways];
    Entry* victim = nullptr;
    for (size_t way = 0; way < Ways; ++way) {
      Entry& e = set[way];
      if (!e.valid) {
        victim = &e;
        break;
      }
      if (victim == nullptr || e.stamp < victim->stamp) {
        victim = &e;
      }
    }
    if (victim->valid) {
      bump(mEvictions);
    } else {
      ++mOccupied;
      if (mOccupied > mMaxOccupied) {
        mMaxOccupied = mOccupied;
      }
      mOccupiedPub.store(mOccupied, std::memory_order_relaxed);
      mMaxOccupiedPub.store(mMaxOccupied, std::memory_order_relaxed);
    }
    victim->valid = true;
    victim->kind = kind;
    victim->layoutKey = layoutKey;
    victim->configVersion = configVersion;
    victim->hash = hash;
    victim->stamp = ++mClock;
    victim->config = config;
    victim->ref = ref;
    victim->info = info;
  }

  // Empties the table (keeps the allocation and the cumulative counters).
  void clear() noexcept {
    if (mEntries != nullptr) {
      for (size_t i = 0; i < kCapacity; ++i) {
        mEntries[i].valid = false;
      }
    }
    mOccupied = 0;
    mOccupiedPub.store(0, std::memory_order_relaxed);
  }

  // Readable from another thread; the values are approximate while the owner runs.
  LookupMemoStats stats() const noexcept {
    return {mHits.load(std::memory_order_relaxed),
            mMisses.load(std::memory_order_relaxed),
            mEvictions.load(std::memory_order_relaxed),
            mOccupiedPub.load(std::memory_order_relaxed),
            mMaxOccupiedPub.load(std::memory_order_relaxed),
            static_cast<uint32_t>(kCapacity)};
  }

private:
  struct Entry {
    Config config{};
    Info info{};
    Ref ref{};
    uint64_t hash = 0;
    uint64_t layoutKey = 0;
    uint64_t stamp = 0;
    uint32_t kind = 0;
    uint32_t configVersion = 0;
    bool valid = false;
  };

  // Single writer: a plain load + store is enough and avoids an atomic RMW.
  static void bump(std::atomic<uint64_t>& counter) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
  }

  void sync_generation(uint32_t generation) noexcept {
    if (!mHaveGeneration || mGeneration != generation) {
      clear();
      mGeneration = generation;
      mHaveGeneration = true;
    }
  }

  std::unique_ptr<Entry[]> mEntries;
  uint32_t mGeneration = 0;
  bool mHaveGeneration = false;
  uint64_t mClock = 0;
  uint32_t mOccupied = 0;
  uint32_t mMaxOccupied = 0;
  std::atomic<uint64_t> mHits{0};
  std::atomic<uint64_t> mMisses{0};
  std::atomic<uint64_t> mEvictions{0};
  std::atomic<uint32_t> mOccupiedPub{0};
  std::atomic<uint32_t> mMaxOccupiedPub{0};
};

// The one entry point the draw path uses. Disabled: only `resolve()` runs -- the
// memo is not read, written or counted. Enabled: a hit returns the stored result;
// a miss runs `resolve()` (the unchanged full lookup, which must fill `ref` and
// `info`) and stores what it produced. `hashOf()` is only called when enabled.
template <class Memo, class Config, class Ref, class Info, class HashOf, class Resolve>
inline void memo_resolve(Memo& memo, bool enabled, uint32_t kind, uint64_t layoutKey, uint32_t generation,
                         uint32_t configVersion, const Config& config, HashOf&& hashOf, Ref& ref, Info& info,
                         Resolve&& resolve) {
  if (!enabled) {
    resolve();
    return;
  }
  const uint64_t hash = hashOf();
  if (memo.find(kind, layoutKey, generation, configVersion, hash, config, ref, info)) {
    return;
  }
  resolve();
  memo.insert(kind, layoutKey, generation, configVersion, hash, config, ref, info);
}

} // namespace aurora::gx
