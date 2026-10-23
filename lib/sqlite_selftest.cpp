// aleks2 (V1-c3): on-console self-test of the file layer under SQLite.
//
// V1-c1 (WAL) failed with IOERR_FSTAT and V1-c2 (DELETE) with "malformed" on brand-new
// files, on a card where H38 (MEMORY/OFF) wrote fine. The host tests cannot reproduce
// libnx's fsdev, so the console tells us: in a scratch folder, the raw calls SQLite's unix
// VFS makes (open, fstat, stat/access on an open file, a second open, writes past the end,
// ftruncate, fsync, rename, unlink, opening a directory) and then every journal_mode /
// synchronous pair through a full cycle (create, CREATE TABLE, INSERT, COMMIT, UPDATE,
// close, reopen, quick_check, count). One launch, one report; the first pair that passes
// in a preference order is what the caches use.
#include "sqlite_selftest.hpp"

#include <sqlite3.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace aurora::sqlite {
namespace {

std::string errno_text(int e) { return std::to_string(e) + " (" + std::strerror(e) + ")"; }

struct Probe {
  std::string& out;
  void line(const std::string& text) { out += text + "\n"; }
  void rc(const char* what, int rc, int err) {
    line(std::string{"probe "} + what + ": " + (rc >= 0 ? "ok rc=" + std::to_string(rc) : "FAIL errno=" + errno_text(err)));
  }
};

void probe_files(const std::string& dir, std::string& out) {
  Probe p{out};
  const std::string f = dir + "/probe.bin";
  const std::string g = dir + "/probe-renamed.bin";
  const std::string none = dir + "/probe-does-not-exist.bin";
  std::remove(f.c_str());
  std::remove(g.c_str());

  int fd = ::open(f.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  p.rc("open(O_RDWR|O_CREAT|O_TRUNC)", fd, errno);
  if (fd < 0) {
    return;
  }
  struct stat st{};
  int r = ::fstat(fd, &st);
  p.line("probe fstat(new, empty): " + (r == 0 ? "ok size=" + std::to_string(st.st_size) + " mode=" +
                                                    std::to_string(st.st_mode & 0777)
                                              : "FAIL errno=" + errno_text(errno)));
  r = ::stat(f.c_str(), &st);
  p.line("probe stat(while open for write): " +
         (r == 0 ? "ok size=" + std::to_string(st.st_size) : "FAIL errno=" + errno_text(errno)));
  r = ::access(f.c_str(), F_OK);
  p.rc("access(existing, while open)", r, errno);
  r = ::access(none.c_str(), F_OK);
  p.line(std::string{"probe access(missing): "} + (r == 0 ? "says EXISTS (wrong)" : "ok errno=" + errno_text(errno)));
  int fd2 = ::open(f.c_str(), O_RDWR, 0644);
  p.rc("second open(O_RDWR) while open", fd2, errno);
  if (fd2 >= 0) {
    ::close(fd2);
  }
  int fd3 = ::open(f.c_str(), O_RDONLY, 0644);
  p.rc("second open(O_RDONLY) while open", fd3, errno);
  if (fd3 >= 0) {
    ::close(fd3);
  }

  std::vector<unsigned char> page(4096, 0xA5);
  off_t o = ::lseek(fd, 0, SEEK_SET);
  ssize_t w = ::write(fd, page.data(), page.size());
  p.line("probe lseek(0)+write(4096): lseek=" + std::to_string(static_cast<long long>(o)) +
         " write=" + std::to_string(static_cast<long long>(w)) + (w < 0 ? " errno=" + errno_text(errno) : ""));
  o = ::lseek(fd, 8192, SEEK_SET);
  w = ::write(fd, page.data(), page.size());
  p.line("probe lseek(8192)+write past EOF: lseek=" + std::to_string(static_cast<long long>(o)) +
         " write=" + std::to_string(static_cast<long long>(w)) + (w < 0 ? " errno=" + errno_text(errno) : ""));
  r = ::fstat(fd, &st);
  p.line("probe fstat(after writes, expect 12288): " +
         (r == 0 ? "size=" + std::to_string(st.st_size) : "FAIL errno=" + errno_text(errno)));
  // The hole must read back as zeros.
  std::vector<unsigned char> back(4096, 0x11);
  o = ::lseek(fd, 4096, SEEK_SET);
  ssize_t rd = ::read(fd, back.data(), back.size());
  bool zeros = rd == 4096;
  for (ssize_t i = 0; zeros && i < rd; ++i) {
    zeros = back[i] == 0;
  }
  p.line("probe read hole 4096..8192: read=" + std::to_string(static_cast<long long>(rd)) +
         (zeros ? " zeros ok" : " NOT ZEROS"));
  o = ::lseek(fd, 0, SEEK_SET);
  rd = ::read(fd, back.data(), back.size());
  p.line(std::string{"probe read page 0: read="} + std::to_string(static_cast<long long>(rd)) +
         (rd == 4096 && back[0] == 0xA5 && back[4095] == 0xA5 ? " data ok" : " DATA WRONG"));
  o = ::lseek(fd, 20000, SEEK_SET);
  rd = ::read(fd, back.data(), back.size());
  p.line("probe read past EOF: read=" + std::to_string(static_cast<long long>(rd)) + " (expect 0)");
  r = ::ftruncate(fd, 4096);
  p.rc("ftruncate(4096)", r, errno);
  r = ::fstat(fd, &st);
  p.line("probe fstat(after ftruncate, expect 4096): " +
         (r == 0 ? "size=" + std::to_string(st.st_size) : "FAIL errno=" + errno_text(errno)));
  r = ::ftruncate(fd, 0);
  p.rc("ftruncate(0)", r, errno);
  r = ::fsync(fd);
  p.rc("fsync", r, errno);
  r = ::close(fd);
  p.rc("close", r, errno);
  r = ::stat(f.c_str(), &st);
  p.line("probe stat(after close): " + (r == 0 ? "ok size=" + std::to_string(st.st_size) : "FAIL errno=" + errno_text(errno)));
  int fdx = ::open(f.c_str(), O_RDWR | O_CREAT | O_EXCL, 0644);
  p.line(std::string{"probe open(O_CREAT|O_EXCL) on existing: "} +
         (fdx >= 0 ? "opened (wrong)" : "refused errno=" + errno_text(errno)));
  if (fdx >= 0) {
    ::close(fdx);
  }
  r = ::rename(f.c_str(), g.c_str());
  p.rc("rename", r, errno);
  r = ::unlink(g.c_str());
  p.rc("unlink", r, errno);
  r = ::access(g.c_str(), F_OK);
  p.line(std::string{"probe access(after unlink): "} + (r == 0 ? "says EXISTS (wrong)" : "ok errno=" + errno_text(errno)));
  int dfd = ::open(dir.c_str(), O_RDONLY, 0);
  p.line(std::string{"probe open(directory, O_RDONLY): "} + (dfd >= 0 ? "ok" : "FAIL errno=" + errno_text(errno)));
  if (dfd >= 0) {
    r = ::fsync(dfd);
    p.rc("fsync(directory)", r, errno);
    ::close(dfd);
  }
}

void remove_db(const std::string& path) {
  for (const char* s : {"", "-journal", "-wal", "-shm"}) {
    std::remove((path + s).c_str());
  }
}

struct Combo {
  const char* journal;
  const char* sync;
};

// Order of preference for the caches: never damaged in the host durability test, and no
// journal file created and deleted at every commit (TRUNCATE/PERSIST keep it); then
// DELETE; MEMORY last (works where nothing else does, but cannot survive a crash).
constexpr Combo kPreference[] = {
    {"truncate", "normal"}, {"persist", "normal"}, {"delete", "normal"}, {"truncate", "full"},
    {"persist", "full"},    {"delete", "full"},    {"memory", "normal"}, {"memory", "off"},
};

std::string sqlite_fail(sqlite3* db, const char* step, int rc) {
  const int ext = db != nullptr ? sqlite3_extended_errcode(db) : rc;
  const int sys = db != nullptr ? sqlite3_system_errno(db) : 0;
  return std::string{"FAIL at "} + step + ": rc=" + std::to_string(ext) + " (" + sqlite3_errstr(ext) + ")" +
         " msg='" + (db != nullptr ? sqlite3_errmsg(db) : "") + "' errno=" + errno_text(sys);
}

// The full cycle for one pair. "" on success, the failure otherwise.
std::string run_combo(const std::string& path, const char* vfs, const char* journal, const char* sync) {
  remove_db(path);
  sqlite3* db = nullptr;
  int rc = sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_PRIVATECACHE, vfs);
  if (rc != SQLITE_OK) {
    const auto f = sqlite_fail(db, "open", rc);
    sqlite3_close(db);
    return f;
  }
  sqlite3_extended_result_codes(db, 1);
  const auto pragmas = [&](sqlite3* d) -> std::string {
    std::string mode;
    if (std::strcmp(journal, "wal") == 0) {
      sqlite3_exec(d, "PRAGMA locking_mode=EXCLUSIVE;", nullptr, nullptr, nullptr);
    }
    const std::string q = std::string{"PRAGMA journal_mode="} + journal + ";";
    int r = sqlite3_exec(
        d, q.c_str(),
        [](void* m, int n, char** v, char**) {
          if (n > 0 && v[0] != nullptr) {
            *static_cast<std::string*>(m) = v[0];
          }
          return 0;
        },
        &mode, nullptr);
    if (r != SQLITE_OK) {
      return sqlite_fail(d, "journal_mode", r);
    }
    if (mode != journal) {
      return std::string{"FAIL at journal_mode: SQLite uses '"} + mode + "'";
    }
    const std::string s = std::string{"PRAGMA synchronous="} + sync + ";";
    r = sqlite3_exec(d, s.c_str(), nullptr, nullptr, nullptr);
    return r == SQLITE_OK ? std::string{} : sqlite_fail(d, "synchronous", r);
  };
  const auto step = [&](sqlite3* d, const char* name, const char* sql) -> std::string {
    const int r = sqlite3_exec(d, sql, nullptr, nullptr, nullptr);
    return r == SQLITE_OK ? std::string{} : sqlite_fail(d, name, r);
  };
  std::string f = pragmas(db);
  if (f.empty()) f = step(db, "CREATE TABLE", "BEGIN IMMEDIATE; CREATE TABLE t(k INTEGER PRIMARY KEY, v BLOB); COMMIT;");
  if (f.empty()) {
    f = step(db, "BEGIN (insert)", "BEGIN IMMEDIATE;");
    sqlite3_stmt* st = nullptr;
    if (f.empty() && sqlite3_prepare_v2(db, "INSERT INTO t(v) VALUES (randomblob(4096))", -1, &st, nullptr) == SQLITE_OK) {
      for (int i = 0; i < 32 && f.empty(); ++i) {
        const int r = sqlite3_step(st);
        if (r != SQLITE_DONE) {
          f = sqlite_fail(db, "INSERT", r);
        }
        sqlite3_reset(st);
      }
      sqlite3_finalize(st);
    }
    if (f.empty()) f = step(db, "COMMIT (insert)", "COMMIT;");
    else sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr);
  }
  if (f.empty()) f = step(db, "UPDATE", "BEGIN IMMEDIATE; UPDATE t SET v = randomblob(4096) WHERE k % 3 = 0; COMMIT;");
  if (f.empty()) f = step(db, "DELETE", "BEGIN IMMEDIATE; DELETE FROM t WHERE k % 5 = 0; COMMIT;");
  sqlite3_close(db);
  db = nullptr;
  if (f.empty()) {
    rc = sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_PRIVATECACHE, vfs);
    if (rc != SQLITE_OK) {
      f = sqlite_fail(db, "reopen", rc);
    } else {
      sqlite3_extended_result_codes(db, 1);
      f = pragmas(db);
      if (f.empty()) {
        std::string check;
        int count = -1;
        int r = sqlite3_exec(
            db, "PRAGMA quick_check;",
            [](void* c, int n, char** v, char**) {
              if (n > 0 && v[0] != nullptr) *static_cast<std::string*>(c) = v[0];
              return 0;
            },
            &check, nullptr);
        if (r != SQLITE_OK) {
          f = sqlite_fail(db, "quick_check", r);
        } else if (check != "ok") {
          f = "FAIL at quick_check: " + check;
        } else {
          r = sqlite3_exec(
              db, "SELECT COUNT(*) FROM t;",
              [](void* c, int n, char** v, char**) {
                if (n > 0 && v[0] != nullptr) *static_cast<int*>(c) = std::atoi(v[0]);
                return 0;
              },
              &count, nullptr);
          if (r != SQLITE_OK) {
            f = sqlite_fail(db, "count", r);
          } else if (count != 26) {
            // 32 rows k=1..32, k%5==0 removed (5, 10, ..., 30): 26 left.
            f = "FAIL at count: " + std::to_string(count) + " rows (expected 26)";
          }
        }
      }
    }
    sqlite3_close(db);
  }
  remove_db(path);
  return f;
}

}  // namespace

SelfTestResult run_sqlite_selftest(const std::string& dir, const char* vfs) {
  SelfTestResult result;
  std::string& out = result.report;
  const auto start = std::chrono::steady_clock::now();
  out += "sqlite selftest: sqlite " + std::string{sqlite3_libversion()} + ", dir " + dir + ", vfs " +
         (vfs != nullptr ? vfs : "(default)") + "\n";
  ::mkdir(dir.c_str(), 0755);
  probe_files(dir, out);

  const char* journals[] = {"memory", "truncate", "persist", "delete", "off", "wal"};
  const char* syncs[] = {"off", "normal", "full"};
  std::vector<std::pair<std::string, bool>> passed;
  for (const char* j : journals) {
    for (const char* s : syncs) {
      if (std::strcmp(j, "wal") == 0 && std::strcmp(s, "normal") != 0) {
        continue;
      }
      const auto t0 = std::chrono::steady_clock::now();
      const std::string path = dir + "/selftest-" + j + "-" + s + ".db";
      const std::string fail = run_combo(path, vfs, j, s);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      char msText[32];
      std::snprintf(msText, sizeof msText, "%.0f ms", ms);
      out += std::string{"combo journal="} + j + " sync=" + s + ": " + (fail.empty() ? "OK" : fail) + " (" + msText +
             ")\n";
      passed.emplace_back(std::string{j} + "/" + s, fail.empty());
    }
  }
  for (const auto& pref : kPreference) {
    const std::string name = std::string{pref.journal} + "/" + pref.sync;
    for (const auto& [n, ok] : passed) {
      if (n == name && ok) {
        result.journal = pref.journal;
        result.sync = pref.sync;
        break;
      }
    }
    if (!result.journal.empty()) {
      break;
    }
  }
  const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  char t[32];
  std::snprintf(t, sizeof t, "%.1f s", total);
  out += "best journal=" + (result.journal.empty() ? std::string{"none"} : result.journal) +
         " sync=" + (result.sync.empty() ? std::string{"none"} : result.sync) + " (selftest " + t + ")\n";
  std::remove((dir + "/probe.bin").c_str());
  ::rmdir(dir.c_str());
  return result;
}

}  // namespace aurora::sqlite

#ifndef AURORA_SELFTEST_NO_C_API  // the host test builds the self-test alone
// ---- C API (include/aurora/shader_cache.h) ------------------------------------------
#include "sqlite_cache_open.hpp"
#ifdef __SWITCH__
#include "sqlite_switch_paths.hpp"
#endif
#include <aurora/shader_cache.h>

#include <algorithm>

namespace {
void copy_out(char* out, size_t size, const std::string& text) {
  if (out == nullptr || size == 0) {
    return;
  }
  const size_t n = std::min(size - 1, text.size());
  std::memcpy(out, text.data(), n);
  out[n] = 0;
}
}  // namespace

int aurora_sqlite_selftest(const char* dir, char* report, size_t reportSize, char* journal, size_t journalSize,
                           char* sync, size_t syncSize) {
  using namespace aurora::sqlite;
  const char* vfs = nullptr;
#ifdef __SWITCH__
  vfs = register_path_vfs() ? kVfsName : "unix-none";
#endif
  const std::string base = dir != nullptr ? dir : "sqlite-selftest";
  // Phase 1: libnx as it is. Phase 2: with the libnx I/O shims (V1-c4: clamped reads,
  // zero-filled gaps, busy-file stat), which is what the caches use on the console.
  uninstall_libnx_io_shims("unix-none");
  const auto raw = run_sqlite_selftest(base, vfs);
  install_libnx_io_shims("unix-none", {});
  const auto shimmed = run_sqlite_selftest(base, vfs);
  const auto counters = libnx_shim_counters();
  uninstall_libnx_io_shims("unix-none");

  int usesShims = 1;
  std::string j = shimmed.journal;
  std::string s = shimmed.sync;
  if (j.empty() && !raw.journal.empty()) {
    j = raw.journal;
    s = raw.sync;
    usesShims = 0;
  }
  std::string text = "== phase 1: libnx as it is\n" + raw.report + "== phase 2: with the libnx I/O shims\n" +
                     shimmed.report + "shims: " + std::to_string(counters.clampedReads) + " reads clamped, " +
                     std::to_string(counters.filledGaps) + " gaps zero-filled\n" + "chosen journal=" +
                     (j.empty() ? "none" : j) + " sync=" + (s.empty() ? "none" : s) + " shims=" +
                     (usesShims ? "1" : "0") + "\n";
  copy_out(report, reportSize, text);
  copy_out(journal, journalSize, j);
  copy_out(sync, syncSize, s);
  if (j.empty()) {
    return -1;
  }
  return usesShims;
}
#endif
