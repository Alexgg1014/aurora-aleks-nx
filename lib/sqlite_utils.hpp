#pragma once

#include "internal.hpp"

#include <sqlite3.h>

#ifdef __SWITCH__
#include "sqlite_switch_paths.hpp"
#endif

namespace aurora::sqlite {

// The on-disk caches (the pipeline list and Dawn's compiled shaders) are opened here.
// On the console they go through SQLite's own VFS with no locking ("unix-none"): the
// default one needs fcntl(F_SETLK) for its journal, which libnx's filesystem does
// not have, and nothing but this process ever opens these files. With
// journal_mode=MEMORY no journal file is made either, so nothing on the card needs
// a lock at all.
#ifdef __SWITCH__
inline constexpr const char* kCachePragmas = "PRAGMA journal_mode=MEMORY; PRAGMA synchronous=OFF;";

inline int open_cache(const char* path, sqlite3** db) {
  register_path_vfs();
  return sqlite3_open_v2(path, db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_PRIVATECACHE, kVfsName);
}
#else
inline constexpr const char* kCachePragmas = "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;";

inline int open_cache(const char* path, sqlite3** db) { return sqlite3_open(path, db); }
#endif

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
      m_log.error("Failed to start transaction: {} (xerr={})", sqlite3_errmsg(m_db),
                  sqlite3_extended_errcode(m_db));
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
