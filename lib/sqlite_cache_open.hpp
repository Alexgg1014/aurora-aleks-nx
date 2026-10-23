// aleks2 (SHADERS-c1): open an on-disk cache database and repair it when it is damaged.
//
// Both caches (pipeline_cache.db, the recipes; dawn_cache.db, the compiled blobs) can
// be rebuilt from scratch, so a damaged file is never worth a broken session: it is
// set aside under a dated name (never deleted on the spot), a fresh one is created,
// and the log says so. On the console the damage came from the journal: with
// journal_mode=MEMORY nothing on the card can roll back a commit cut short by a crash
// or a forced power-off. The journal mode and synchronous level are now runtime
// options (see CacheOpenOptions) so the console can compare them. The default, WAL with
// synchronous=NORMAL, is what the host durability test (platforms/switch/tools/
// test-sqlite-cache.sh) never saw damaged after a crash or a power cut, at ~0.1 syncs per
// commit (DELETE/NORMAL: also never damaged, ~2.2 syncs; MEMORY/OFF: damaged after half
// the crashes and every power cut).
//
// Nothing here knows about Aurora: SQLite, the standard library and a log callback,
// so the host test builds it as it is.
#pragma once

#include <sqlite3.h>

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace aurora::sqlite {

enum class JournalChoice : uint8_t {
  Wal,       ///< WAL with locking_mode=EXCLUSIVE (no -shm file; the console has no locks)
  Delete,    ///< rollback journal, deleted at commit
  Truncate,  ///< rollback journal, truncated at commit
  Persist,   ///< rollback journal, header zeroed at commit
  Memory,    ///< no journal on disk (what H38 shipped; cannot survive a crash mid-commit)
};

enum class SyncChoice : uint8_t { Off, Normal, Full };

enum class CheckPolicy : uint8_t {
  Dirty,   ///< quick_check only when the last session did not close the file
  Always,  ///< quick_check on every open
  Never,   ///< trust the file; still repaired when SQLite itself reports damage
};

struct CacheOpenOptions {
  JournalChoice journal = JournalChoice::Wal;
  SyncChoice sync = SyncChoice::Normal;
  CheckPolicy check = CheckPolicy::Dirty;
  /// Quarantined copies kept per cache; older ones are removed (and logged).
  int keepQuarantined = 2;
  /// The VFS to open with (nullptr: the default one).
  const char* vfs = nullptr;
};

/// Reads AURORA_CACHE_JOURNAL (wal|delete|truncate|persist|memory),
/// AURORA_CACHE_SYNC (off|normal|full) and AURORA_CACHE_CHECK (dirty|always|never).
/// Unknown values keep the default and are reported through `log`.
enum class LogLevel : uint8_t { Debug, Info, Warn, Error };
using LogFn = std::function<void(LogLevel, const std::string&)>;
CacheOpenOptions cache_options_from_env(const LogFn& log);

const char* journal_name(JournalChoice journal);
const char* sync_name(SyncChoice sync);
const char* check_name(CheckPolicy check);

struct CacheOpenReport {
  bool dirtyAtOpen = false;      ///< the previous session left the file open
  bool checked = false;          ///< quick_check ran
  double checkMs = 0.0;          ///< and took this long
  bool repaired = false;         ///< the file was set aside and recreated
  std::string reason;            ///< why (the SQLite message or the quick_check line)
  std::string quarantinedAs;     ///< where the damaged file went ("" if it could not be moved)
  std::string journalMode;       ///< what SQLite reports after the pragma
  int removedOldQuarantine = 0;  ///< older quarantined copies removed to respect keepQuarantined
};

/// True for the result codes that mean "this file is damaged", as opposed to "could
/// not be opened" or "busy".
bool is_corruption(int resultCode);

/// Prepares the freshly opened database (tables, statements...). Return SQLITE_OK, or
/// the failing result code: a corruption code makes the caller repair and retry once.
using InitFn = std::function<int(sqlite3*)>;

/// Opens (creating if needed) the cache at `path`, applies the pragmas, checks it
/// according to `options.check`, runs `init`, and if any step reports damage sets the
/// file aside as `<stem>.corrupt-<UTC stamp>.db`, recreates it and runs `init` again.
/// On success *db is open and a marker `<path>.open` says the session has it open.
/// Returns SQLITE_OK or the error that made the cache unusable (then *db is null).
int open_cache_checked(const std::string& path, sqlite3** db, const CacheOpenOptions& options,
                       const InitFn& init, CacheOpenReport& report, const LogFn& log);

/// V1-c2: makes SQLite's unix VFS (`baseVfs`, e.g. "unix-none") tolerate stat() failing
/// on a file that exists but is open for writing (libnx), which otherwise breaks every
/// journal mode that creates a -journal or -wal file. Process-wide; safe to call again.
bool install_busy_stat_fallback(const char* baseVfs, const LogFn& log);
/// V1-c3: puts SQLite's own stat() back (for the self-test's two phases).
void uninstall_busy_stat_fallback(const char* baseVfs);

/// V1-c4: POSIX file semantics for SQLite on top of libnx, in SQLite's syscall table:
/// reads clamped to the file size (short read instead of EIO), writes past the end
/// zero-fill the gap first, and the busy-file stat() fallback. Process-wide.
bool install_libnx_io_shims(const char* baseVfs, const LogFn& log);
void uninstall_libnx_io_shims(const char* baseVfs);
struct LibnxShimCounters {
  uint32_t clampedReads = 0;  ///< reads that started at or past the end (libnx would have failed)
  uint32_t filledGaps = 0;    ///< writes past the end whose gap was zero-filled
};
LibnxShimCounters libnx_shim_counters();

/// Closes the database. `clean` removes the open marker, so the next session does not
/// re-check it; pass false after the session saw damage, so the next open checks it.
void close_cache(sqlite3* db, const std::string& path, bool clean);

/// Moves `path` (and its -journal/-wal/-shm siblings) aside. Exposed for the test.
/// Returns the new name of the main file, or "" if it could not be moved.
std::string quarantine_file(const std::string& path, std::string_view stamp, const LogFn& log);

/// Removes all but the newest `keep` quarantined copies of `path`. Returns how many went.
int prune_quarantine(const std::string& path, int keep, const LogFn& log);

/// "20261015-231500" (UTC), or "boot<N>" when the clock is not set.
std::string utc_stamp();

}  // namespace aurora::sqlite
