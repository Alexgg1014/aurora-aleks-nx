#pragma once

#include "internal.hpp"

#include <sqlite3.h>

#include <cstdlib>
#include <string>

#include "sqlite_cache_open.hpp"

#ifdef __SWITCH__
#include "sqlite_switch_paths.hpp"
#endif

namespace aurora::sqlite {

// The on-disk caches (the pipeline list and Dawn's compiled shaders) are opened here.
// On the console they go through SQLite's own VFS with no locking ("unix-none"): the
// default one needs fcntl(F_SETLK), which libnx's filesystem does not have, and nothing
// but this process ever opens these files.
//
// aleks2 (SHADERS-c1): H38 used journal_mode=MEMORY, synchronous=OFF here, which leaves
// nothing on the card to roll back a commit cut short -- the "database disk image is
// malformed" after forced power-offs. Both caches now open through open_cache_checked
// (sqlite_cache_open.hpp): a journal on disk (mode and sync level from
// AURORA_CACHE_JOURNAL/_SYNC, defaults measured by test-sqlite-cache.sh), a check after
// an unclean exit, and a damaged file set aside and recreated instead of the cache simply
// being switched off.
inline LogFn log_to(const Module& log) {
  return [&log](LogLevel level, const std::string& text) {
    switch (level) {
    case LogLevel::Debug:
      log.debug("{}", text);
      break;
    case LogLevel::Info:
      log.info("{}", text);
      break;
    case LogLevel::Warn:
      log.warn("{}", text);
      break;
    case LogLevel::Error:
      log.error("{}", text);
      break;
    }
  };
}

inline const CacheOpenOptions& cache_options(const Module& log) {
  static const CacheOpenOptions options = [&log] {
    auto o = cache_options_from_env(log_to(log));
#ifdef __SWITCH__
    o.vfs = register_path_vfs() ? kVfsName : "unix-none";
    // V1-c2: no WAL on the console (refused there, and its failed attempt left the
    // connection unusable); DELETE/NORMAL was never damaged in the host durability test.
    // V1-c3: TRUNCATE by default (the game normally sets the pair the self-test chose):
    // never damaged in the host test, and the -journal is kept instead of being created and
    // deleted at every commit.
    if (o.journal == JournalChoice::Wal) {
      o.journal = JournalChoice::Truncate;
    }
    // V1-c4: POSIX file semantics on top of libnx (reads past the end, zero-filled gaps,
    // stat of an open file; sqlite_cache_open.cpp). Without them every new database on the
    // console was "malformed" (V1-c2/c3). AURORA_CACHE_LIBNX_SHIMS=0 turns them off (lab).
    if (const char* sh = std::getenv("AURORA_CACHE_LIBNX_SHIMS"); sh == nullptr || sh[0] != '0') {
      if (install_libnx_io_shims("unix-none", log_to(log))) {
        log.info("libnx I/O shims installed (clamped reads, zero-filled gaps, busy-file stat)");
      } else {
        log.error("could not install the libnx I/O shims; the caches will probably fail");
      }
    }
#endif
    log.info("cache journal_mode={} synchronous={} check={}", journal_name(o.journal), sync_name(o.sync),
             check_name(o.check));
    return o;
  }();
  return options;
}

// allowCheck false: never quick_check at open (V1-rc3: the 75 MB Dawn cache took 18.5 s after an unclean exit;
// damage is still caught when SQLite reports it and the file is then set aside at the next start).
inline int open_cache(const std::string& path, sqlite3** db, const InitFn& init, CacheOpenReport& report,
                      const Module& log, bool allowCheck = true) {
  CacheOpenOptions options = cache_options(log);
  if (!allowCheck) {
    options.check = CheckPolicy::Never;
  }
  const auto ret = open_cache_checked(path, db, options, init, report, log_to(log));
  if (ret == SQLITE_OK) {
    log.info("{}: journal_mode={}{}{}", path, report.journalMode,
             report.checked ? fmt::format(", quick_check {:.0f} ms", report.checkMs) : std::string{},
             report.dirtyAtOpen ? " (last session did not close it)" : "");
  }
  return ret;
}

inline int exec(sqlite3* db, const char* sql) { return sqlite3_exec(db, sql, nullptr, nullptr, nullptr); }

template <typename T>
int exec(sqlite3* db, const char* sql, T callback, char** errmsg = nullptr) {
  return sqlite3_exec(
      db, sql,
      [](void* cb, int argc, char** argv, char** columns) -> int {
        auto& fp = *static_cast<T*>(cb);
        if constexpr (std::is_same_v<decltype(fp(0, 0, 0)), void>) {
          fp(argc, argv, columns);
          return 0;
        } else {
          return fp(argc, argv, columns);
        }
      },
      &callback, errmsg);
}

class Transaction {
public:
  Transaction(sqlite3* db, Module& log, bool immediate = false) : m_db(db), m_log(log) {
    const auto type = immediate ? "BEGIN IMMEDIATE" : "BEGIN";
    const auto ret = sqlite3_exec(m_db, type, nullptr, nullptr, nullptr);
    if (ret != SQLITE_OK) {
      const int code = sqlite3_extended_errcode(m_db);
      m_log.error("Could not begin a transaction ({}): {}", code, sqlite3_errmsg(m_db));
      return;
    }
    m_active = true;
  }

  Transaction(const Transaction&) = delete;
  Transaction& operator=(const Transaction&) = delete;
  Transaction(Transaction&&) = delete;
  Transaction& operator=(Transaction&&) = delete;

  ~Transaction() {
    if (!m_active) {
      return;
    }

    const auto ret = sqlite3_exec(m_db, "ROLLBACK", nullptr, nullptr, nullptr);
    if (ret != SQLITE_OK) {
      m_log.error("Failed to roll back transaction (uh oh?): {}", sqlite3_errmsg(m_db));
    }
  }

  void commit() {
    if (!m_active) {
      return;
    }

    const auto ret = sqlite3_exec(m_db, "COMMIT", nullptr, nullptr, nullptr);
    if (ret != SQLITE_OK) {
      m_log.error("Failed to commit transaction: {}", sqlite3_errmsg(m_db));
      return;
    }

    m_active = false;
  }

  explicit operator bool() const { return m_active; }

private:
  sqlite3* m_db = nullptr;
  Module& m_log;
  bool m_active = false;
};

} // namespace aurora::sqlite
