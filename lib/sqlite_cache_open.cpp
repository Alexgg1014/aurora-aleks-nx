// aleks2 (SHADERS-c1): see sqlite_cache_open.hpp.
#include "sqlite_cache_open.hpp"

#include <algorithm>
#include <atomic>
#include <fcntl.h>
#include <unistd.h>
#include <utility>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace aurora::sqlite {
namespace {

void say(const LogFn& log, LogLevel level, const std::string& text) {
  if (log) {
    log(level, text);
  }
}

bool file_exists(const std::string& path) {
  if (std::FILE* f = std::fopen(path.c_str(), "rb")) {
    std::fclose(f);
    return true;
  }
  return false;
}

std::string marker_path(const std::string& path) { return path + ".open"; }

void write_marker(const std::string& path) {
  if (std::FILE* f = std::fopen(marker_path(path).c_str(), "wb")) {
    std::fputc('1', f);
    std::fclose(f);
  }
}

// "<dir>/pipeline_cache.db" -> {"<dir>/", "pipeline_cache"}; the extension is ".db" for
// both caches, anything else keeps its full name as the stem.
std::pair<std::string, std::string> split(const std::string& path) {
  const auto slash = path.find_last_of("/\\");
  std::string dir = slash == std::string::npos ? std::string{} : path.substr(0, slash + 1);
  std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
  if (name.size() > 3 && name.compare(name.size() - 3, 3, ".db") == 0) {
    name.resize(name.size() - 3);
  }
  return {dir, name};
}

int apply_pragmas(sqlite3* db, const CacheOpenOptions& options, std::string& journalMode, const LogFn& log) {
  // WAL needs shared memory for its index unless the connection holds the file
  // exclusively; the console has no locks at all (unix-none), so WAL is only safe
  // here in EXCLUSIVE mode, where the index lives on the heap. These caches have one
  // connection each, which EXCLUSIVE suits anyway.
  const char* pre = options.journal == JournalChoice::Wal ? "PRAGMA locking_mode=EXCLUSIVE;" : "";
  int ret = sqlite3_exec(db, pre, nullptr, nullptr, nullptr);
  if (ret != SQLITE_OK) {
    return ret;
  }
  const std::string journal = std::string{"PRAGMA journal_mode="} + journal_name(options.journal) + ";";
  journalMode.clear();
  char* err = nullptr;
  ret = sqlite3_exec(
      db, journal.c_str(),
      [](void* out, int argc, char** argv, char**) -> int {
        if (argc > 0 && argv[0] != nullptr) {
          *static_cast<std::string*>(out) = argv[0];
        }
        return 0;
      },
      &journalMode, &err);
  if (err != nullptr) {
    sqlite3_free(err);
  }
  if (ret != SQLITE_OK) {
    return ret;
  }
  if (journalMode != journal_name(options.journal)) {
    // SQLite answers with the mode it actually uses: WAL can be refused by a VFS. Never
    // stay on a mode without a journal on disk because of that: fall back to DELETE,
    // the other mode the durability test never saw damaged.
    say(log, LogLevel::Warn,
        "asked for journal_mode=" + std::string{journal_name(options.journal)} + ", SQLite uses '" + journalMode +
            "'; falling back to delete");
    if (options.journal == JournalChoice::Wal) {
      sqlite3_exec(db, "PRAGMA locking_mode=NORMAL;", nullptr, nullptr, nullptr);
    }
    ret = sqlite3_exec(
        db, "PRAGMA journal_mode=delete;",
        [](void* out, int argc, char** argv, char**) -> int {
          if (argc > 0 && argv[0] != nullptr) {
            *static_cast<std::string*>(out) = argv[0];
          }
          return 0;
        },
        &journalMode, nullptr);
    if (ret != SQLITE_OK) {
      return ret;
    }
  }
  const std::string sync = std::string{"PRAGMA synchronous="} + sync_name(options.sync) + ";";
  return sqlite3_exec(db, sync.c_str(), nullptr, nullptr, nullptr);
}

// PRAGMA quick_check: "ok", or the first problem it found (in `problem`).
int quick_check(sqlite3* db, std::string& problem) {
  problem.clear();
  sqlite3_stmt* stmt = nullptr;
  int ret = sqlite3_prepare_v2(db, "PRAGMA quick_check(1)", -1, &stmt, nullptr);
  if (ret != SQLITE_OK) {
    problem = sqlite3_errmsg(db);
    return ret;
  }
  ret = sqlite3_step(stmt);
  if (ret == SQLITE_ROW) {
    const auto* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    problem = text != nullptr ? text : "";
    ret = SQLITE_OK;
  } else {
    problem = sqlite3_errmsg(db);
  }
  const int ext = ret == SQLITE_OK ? SQLITE_OK : sqlite3_extended_errcode(db);
  sqlite3_finalize(stmt);
  return ret == SQLITE_OK ? SQLITE_OK : (ext != SQLITE_OK ? ext : ret);
}

struct Attempt {
  int code = SQLITE_OK;
  std::string why;
  bool damaged = false;
};

Attempt try_open(const std::string& path, sqlite3** db, const CacheOpenOptions& options, const InitFn& init,
                 bool check, CacheOpenReport& report, const LogFn& log) {
  Attempt a;
  *db = nullptr;
  a.code = sqlite3_open_v2(path.c_str(), db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_PRIVATECACHE,
                           options.vfs);
  const auto fail = [&](const char* step) {
    a.why = std::string{step} + ": " + (*db != nullptr ? sqlite3_errmsg(*db) : sqlite3_errstr(a.code));
    a.damaged = is_corruption(a.code);
    if (*db != nullptr) {
      sqlite3_close(*db);
      *db = nullptr;
    }
    return a;
  };
  if (a.code != SQLITE_OK) {
    return fail("open");
  }
  sqlite3_extended_result_codes(*db, 1);
  a.code = apply_pragmas(*db, options, report.journalMode, log);
  if (a.code != SQLITE_OK) {
    return fail("pragmas");
  }
  if (check) {
    const auto start = std::chrono::steady_clock::now();
    std::string problem;
    a.code = quick_check(*db, problem);
    report.checked = true;
    report.checkMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (a.code != SQLITE_OK) {
      return fail("quick_check");
    }
    if (problem != "ok") {
      a.code = SQLITE_CORRUPT;
      a.why = "quick_check: " + problem;
      a.damaged = true;
      sqlite3_close(*db);
      *db = nullptr;
      return a;
    }
  }
  if (init) {
    a.code = init(*db);
    if (a.code != SQLITE_OK) {
      const int ext = sqlite3_extended_errcode(*db);
      if (is_corruption(ext)) {
        a.code = ext;
      }
      return fail("init");
    }
  }
  return a;
}

}  // namespace

const char* journal_name(JournalChoice journal) {
  switch (journal) {
  case JournalChoice::Wal:
    return "wal";
  case JournalChoice::Delete:
    return "delete";
  case JournalChoice::Truncate:
    return "truncate";
  case JournalChoice::Persist:
    return "persist";
  case JournalChoice::Memory:
    return "memory";
  }
  return "wal";
}

const char* sync_name(SyncChoice sync) {
  switch (sync) {
  case SyncChoice::Off:
    return "off";
  case SyncChoice::Normal:
    return "normal";
  case SyncChoice::Full:
    return "full";
  }
  return "normal";
}

const char* check_name(CheckPolicy check) {
  switch (check) {
  case CheckPolicy::Dirty:
    return "dirty";
  case CheckPolicy::Always:
    return "always";
  case CheckPolicy::Never:
    return "never";
  }
  return "dirty";
}

namespace {
template <typename T, size_t N>
void read_choice(const char* name, T& field, const T (&choices)[N], const char* (*name_of)(T), const LogFn& log) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return;
  }
  const std::string text{value};
  for (const T choice : choices) {
    if (text == name_of(choice)) {
      field = choice;
      return;
    }
  }
  say(log, LogLevel::Warn, std::string{name} + "='" + text + "' is not a known value; using the default");
}
}  // namespace

CacheOpenOptions cache_options_from_env(const LogFn& log) {
  CacheOpenOptions options;
  constexpr JournalChoice journals[] = {JournalChoice::Wal, JournalChoice::Delete, JournalChoice::Truncate,
                                        JournalChoice::Persist, JournalChoice::Memory};
  constexpr SyncChoice syncs[] = {SyncChoice::Off, SyncChoice::Normal, SyncChoice::Full};
  constexpr CheckPolicy checks[] = {CheckPolicy::Dirty, CheckPolicy::Always, CheckPolicy::Never};
  read_choice("AURORA_CACHE_JOURNAL", options.journal, journals, journal_name, log);
  read_choice("AURORA_CACHE_SYNC", options.sync, syncs, sync_name, log);
  read_choice("AURORA_CACHE_CHECK", options.check, checks, check_name, log);
  return options;
}

bool is_corruption(int resultCode) {
  const int primary = resultCode & 0xff;
  // IOERR_SHORT_READ: a file cut shorter than its own header says (a truncated copy).
  return primary == SQLITE_CORRUPT || primary == SQLITE_NOTADB || resultCode == SQLITE_IOERR_SHORT_READ;
}

std::string utc_stamp() {
  const std::time_t now = std::time(nullptr);
  // 2020-01-01: before that the console clock is simply not set.
  if (now < 1577836800) {
    static int boot = 0;
    return "boot" + std::to_string(++boot);
  }
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  char text[32];
  std::strftime(text, sizeof(text), "%Y%m%d-%H%M%S", &tm);
  return text;
}

std::string quarantine_file(const std::string& path, std::string_view stamp, const LogFn& log) {
  // aleks2 (V1-rc5): the empty file that replaces it is not a copy of any bundled cache.
  std::remove((path + ".seed").c_str());
  const auto [dir, stem] = split(path);
  std::string target;
  for (int n = 0; n < 100; ++n) {
    target = dir + stem + ".corrupt-" + std::string{stamp} + (n == 0 ? "" : "-" + std::to_string(n)) + ".db";
    if (!file_exists(target)) {
      break;
    }
  }
  if (std::rename(path.c_str(), target.c_str()) != 0) {
    say(log, LogLevel::Error, "could not move the damaged cache '" + path + "' aside to '" + target + "'");
    return {};
  }
  // The journal of a damaged file belongs to it: rolling it back into a new, empty
  // database would be wrong. It goes with the file, same stamp.
  for (const char* suffix : {"-journal", "-wal", "-shm"}) {
    const std::string sibling = path + suffix;
    if (file_exists(sibling)) {
      const std::string to = target + suffix;
      if (std::rename(sibling.c_str(), to.c_str()) != 0) {
        say(log, LogLevel::Warn, "could not move '" + sibling + "' aside; removing it so the new cache starts clean");
        std::remove(sibling.c_str());
      }
    }
  }
  return target;
}

int prune_quarantine(const std::string& path, int keep, const LogFn& log) {
  const auto [dir, stem] = split(path);
  const std::string prefix = stem + ".corrupt-";
  std::vector<std::string> copies;
  std::error_code ec;
  for (std::filesystem::directory_iterator it{dir.empty() ? std::string{"."} : dir, ec}, end; !ec && it != end;
       it.increment(ec)) {
    const std::string name = it->path().filename().string();
    if (name.rfind(prefix, 0) == 0 && name.size() > 3 && name.compare(name.size() - 3, 3, ".db") == 0) {
      copies.push_back(name);
    }
  }
  if (static_cast<int>(copies.size()) <= keep) {
    return 0;
  }
  // The stamp sorts by time ("boot<N>" ones, from an unset clock, sort after digits and
  // are therefore kept first; they are rare and still bounded below).
  std::sort(copies.begin(), copies.end());
  int removed = 0;
  for (size_t i = 0; i + keep < copies.size(); ++i) {
    const std::string full = dir + copies[i];
    if (std::remove(full.c_str()) == 0) {
      ++removed;
      say(log, LogLevel::Info, "removed an older quarantined cache: " + full);
    }
    for (const char* suffix : {"-journal", "-wal", "-shm"}) {
      std::remove((full + suffix).c_str());
    }
  }
  return removed;
}

int open_cache_checked(const std::string& path, sqlite3** db, const CacheOpenOptions& options, const InitFn& init,
                       CacheOpenReport& report, const LogFn& log) {
  report = {};
  // V1-c2: a -wal/-shm left by an attempt at WAL (V1-c1 on the console: refused, never
  // used) would make SQLite try WAL again at open. Without WAL they hold nothing.
  if (options.journal != JournalChoice::Wal) {
    for (const char* suffix : {"-wal", "-shm"}) {
      const std::string orphan = path + suffix;
      if (file_exists(orphan) && std::remove(orphan.c_str()) == 0) {
        say(log, LogLevel::Warn, "removed '" + orphan + "' (no WAL in use)");
      }
    }
  }
  report.dirtyAtOpen = file_exists(marker_path(path));
  const bool check = options.check == CheckPolicy::Always || (options.check == CheckPolicy::Dirty && report.dirtyAtOpen);

  Attempt a = try_open(path, db, options, init, check, report, log);
  if (a.code == SQLITE_OK) {
    write_marker(path);
    return SQLITE_OK;
  }
  // V1-c2: anything but "cannot open at all" (a missing directory) is worth one clean
  // start: an unusable file is set aside like a damaged one and recreated. The console's
  // first failure (IOERR_FSTAT on BEGIN) was not damage, and leaving the cache off for
  // good was the wrong answer.
  if (!a.damaged && (a.code & 0xff) == SQLITE_CANTOPEN) {
    say(log, LogLevel::Error, "cache '" + path + "' cannot be opened (not repaired): " + a.why);
    return a.code;
  }

  report.reason = a.why;
  report.quarantinedAs = quarantine_file(path, utc_stamp(), log);
  if (report.quarantinedAs.empty()) {
    // Never delete a file we could not move: the session runs without this cache.
    return a.code;
  }
  report.repaired = true;
  say(log, LogLevel::Warn,
      "cache '" + path + "' was " + (a.damaged ? "damaged" : "unusable") + " (" + a.why + "); set aside as '" +
          report.quarantinedAs + "' and started a new one");
  report.removedOldQuarantine = prune_quarantine(path, options.keepQuarantined, log);

  CacheOpenReport retry;
  a = try_open(path, db, options, init, false, retry, log);
  report.journalMode = retry.journalMode;
  if (a.code != SQLITE_OK) {
    say(log, LogLevel::Error, "the new cache '" + path + "' could not be created either: " + a.why);
    return a.code;
  }
  write_marker(path);
  return SQLITE_OK;
}

namespace {
sqlite3_syscall_ptr g_realStat = nullptr;
LogFn g_statLog;
bool g_statNoted = false;

// libnx's stat() opens the file to read its size, and Horizon refuses to open a file this
// process already has open for writing: stat() on an open database fails (not ENOENT).
// SQLite asks stat() only for the mode/owner to give a new -journal (getFileMode), and
// turns the failure into SQLITE_IOERR_FSTAT on the first write transaction. A file that
// exists but cannot be stat'ed right now is reported as a plain 0644 file instead.
int busy_tolerant_stat(const char* path, struct stat* out) {
  using StatFn = int (*)(const char*, struct stat*);
  const int rc = reinterpret_cast<StatFn>(g_realStat)(path, out);
  if (rc == 0 || errno == ENOENT || errno == ENOTDIR || errno == ENAMETOOLONG) {
    return rc;
  }
  const int err = errno;
  std::memset(out, 0, sizeof(*out));
  out->st_mode = S_IFREG | 0644;
  out->st_nlink = 1;
  if (!g_statNoted && g_statLog) {
    g_statNoted = true;
    g_statLog(LogLevel::Info, std::string{"stat() refused on an open file (errno "} + std::to_string(err) + ", '" +
                                  path + "'); treating it as a plain file");
  }
  return 0;
}
}  // namespace

bool install_busy_stat_fallback(const char* baseVfs, const LogFn& log) {
  sqlite3_vfs* vfs = sqlite3_vfs_find(baseVfs);
  if (vfs == nullptr || vfs->iVersion < 3 || vfs->xSetSystemCall == nullptr) {
    return false;
  }
  if (g_realStat != nullptr) {
    return true;
  }
  g_realStat = vfs->xGetSystemCall(vfs, "stat");
  if (g_realStat == nullptr) {
    return false;
  }
  g_statLog = log;
  return vfs->xSetSystemCall(vfs, "stat", reinterpret_cast<sqlite3_syscall_ptr>(busy_tolerant_stat)) == SQLITE_OK;
}

void uninstall_busy_stat_fallback(const char* baseVfs) {
  sqlite3_vfs* vfs = sqlite3_vfs_find(baseVfs);
  if (vfs == nullptr || vfs->iVersion < 3 || vfs->xSetSystemCall == nullptr || g_realStat == nullptr) {
    return;
  }
  vfs->xSetSystemCall(vfs, "stat", nullptr);  // nullptr: SQLite's default
  g_realStat = nullptr;
}

// ---- V1-c4: libnx file semantics under SQLite ---------------------------------------
//
// What the console's self-test (V1-c3) measured, and what SQLite's unix VFS assumes:
//   read() starting past the end of the file  libnx: -1/EIO      POSIX: 0 (short read)
//   bytes skipped by a write past the end     libnx: not zeros   POSIX: zeros
//   stat()/access() of a file we have open    libnx: EIO         POSIX: works
// SQLite turns a read failing with EIO into SQLITE_IOERR_CORRUPTFS and reports it as
// SQLITE_CORRUPT: on the console every new database was "malformed" (V1-c2, V1-c3; and
// H38's empty dawn_cache.db), in every journal mode. The emulator in platforms/switch/
// tools/sqlite_libnx_emulation_test.cpp reproduces exactly that. These wrappers, in
// SQLite's own syscall table, give it POSIX semantics on top of libnx: a read is clamped to
// the file size (the rest is a short read, which SQLite zero-fills), a write past the end
// first fills the gap with zeros, and stat() keeps the busy-file fallback above.
namespace {
using ReadFn = ssize_t (*)(int, void*, size_t);
using WriteFn = ssize_t (*)(int, const void*, size_t);
using PreadFn = ssize_t (*)(int, void*, size_t, off_t);
using PwriteFn = ssize_t (*)(int, const void*, size_t, off_t);
using FstatFn = int (*)(int, struct stat*);
sqlite3_syscall_ptr g_realRead = nullptr;
sqlite3_syscall_ptr g_realWrite = nullptr;
sqlite3_syscall_ptr g_realPread = nullptr;
sqlite3_syscall_ptr g_realPwrite = nullptr;
sqlite3_syscall_ptr g_realFstat = nullptr;
std::atomic<uint32_t> g_clampedReads{0};
std::atomic<uint32_t> g_filledGaps{0};

off_t file_size(int fd) {
  struct stat st{};
  return reinterpret_cast<FstatFn>(g_realFstat)(fd, &st) == 0 ? st.st_size : -1;
}

// Writes zeros from `from` to `to` with `write_at` (one of the two real writers).
template <typename WriteAt>
bool fill_zeros(off_t from, off_t to, WriteAt write_at) {
  static const char zeros[4096] = {};
  while (from < to) {
    const size_t n = static_cast<size_t>(std::min<off_t>(to - from, sizeof(zeros)));
    const ssize_t w = write_at(zeros, n, from);
    if (w <= 0) {
      return false;
    }
    from += w;
  }
  g_filledGaps.fetch_add(1, std::memory_order_relaxed);
  return true;
}

ssize_t shim_pread(int fd, void* buf, size_t n, off_t off) {
  const off_t size = file_size(fd);
  if (size >= 0) {
    if (off >= size) {
      g_clampedReads.fetch_add(1, std::memory_order_relaxed);
      return 0;
    }
    if (static_cast<off_t>(off + n) > size) {
      n = static_cast<size_t>(size - off);
    }
  }
  return reinterpret_cast<PreadFn>(g_realPread)(fd, buf, n, off);
}

ssize_t shim_pwrite(int fd, const void* buf, size_t n, off_t off) {
  const off_t size = file_size(fd);
  if (size >= 0 && off > size) {
    fill_zeros(size, off, [fd](const void* b, size_t len, off_t at) {
      return reinterpret_cast<PwriteFn>(g_realPwrite)(fd, b, len, at);
    });
  }
  return reinterpret_cast<PwriteFn>(g_realPwrite)(fd, buf, n, off);
}

// The console's build has no pread/pwrite: SQLite does lseek() and then read()/write().
ssize_t shim_read(int fd, void* buf, size_t n) {
  const off_t off = ::lseek(fd, 0, SEEK_CUR);
  const off_t size = file_size(fd);
  if (off >= 0 && size >= 0) {
    if (off >= size) {
      g_clampedReads.fetch_add(1, std::memory_order_relaxed);
      return 0;
    }
    if (static_cast<off_t>(off + n) > size) {
      n = static_cast<size_t>(size - off);
    }
  }
  return reinterpret_cast<ReadFn>(g_realRead)(fd, buf, n);
}

ssize_t shim_write(int fd, const void* buf, size_t n) {
  const off_t off = ::lseek(fd, 0, SEEK_CUR);
  const off_t size = file_size(fd);
  if (off >= 0 && size >= 0 && off > size) {
    fill_zeros(size, off, [fd](const void* b, size_t len, off_t at) {
      if (::lseek(fd, at, SEEK_SET) != at) {
        return static_cast<ssize_t>(-1);
      }
      return reinterpret_cast<WriteFn>(g_realWrite)(fd, b, len);
    });
    ::lseek(fd, off, SEEK_SET);
  }
  return reinterpret_cast<WriteFn>(g_realWrite)(fd, buf, n);
}

void swap_in(sqlite3_vfs* vfs, const char* name, sqlite3_syscall_ptr& real, sqlite3_syscall_ptr shim) {
  if (real != nullptr) {
    return;
  }
  real = vfs->xGetSystemCall(vfs, name);
  if (real != nullptr) {
    vfs->xSetSystemCall(vfs, name, shim);
  }
}
}  // namespace

bool install_libnx_io_shims(const char* baseVfs, const LogFn& log) {
  sqlite3_vfs* vfs = sqlite3_vfs_find(baseVfs);
  if (vfs == nullptr || vfs->iVersion < 3 || vfs->xSetSystemCall == nullptr) {
    return false;
  }
  if (g_realFstat == nullptr) {
    g_realFstat = vfs->xGetSystemCall(vfs, "fstat");
  }
  if (g_realFstat == nullptr) {
    return false;
  }
  swap_in(vfs, "read", g_realRead, reinterpret_cast<sqlite3_syscall_ptr>(shim_read));
  swap_in(vfs, "write", g_realWrite, reinterpret_cast<sqlite3_syscall_ptr>(shim_write));
  swap_in(vfs, "pread", g_realPread, reinterpret_cast<sqlite3_syscall_ptr>(shim_pread));
  swap_in(vfs, "pwrite", g_realPwrite, reinterpret_cast<sqlite3_syscall_ptr>(shim_pwrite));
  return install_busy_stat_fallback(baseVfs, log);
}

void uninstall_libnx_io_shims(const char* baseVfs) {
  sqlite3_vfs* vfs = sqlite3_vfs_find(baseVfs);
  if (vfs == nullptr || vfs->iVersion < 3 || vfs->xSetSystemCall == nullptr) {
    return;
  }
  for (auto [name, real] : {std::pair<const char*, sqlite3_syscall_ptr*>{"read", &g_realRead},
                            {"write", &g_realWrite}, {"pread", &g_realPread}, {"pwrite", &g_realPwrite}}) {
    if (*real != nullptr) {
      vfs->xSetSystemCall(vfs, name, nullptr);
      *real = nullptr;
    }
  }
  uninstall_busy_stat_fallback(baseVfs);
}

LibnxShimCounters libnx_shim_counters() {
  return {g_clampedReads.load(std::memory_order_relaxed), g_filledGaps.load(std::memory_order_relaxed)};
}

void close_cache(sqlite3* db, const std::string& path, bool clean) {
  if (db != nullptr) {
    sqlite3_close(db);
  }
  if (clean) {
    std::remove(marker_path(path).c_str());
  }
}

}  // namespace aurora::sqlite
