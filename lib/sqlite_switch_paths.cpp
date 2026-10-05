// aleks2: teach SQLite what a Switch path looks like.
//
// SQLite decides whether a path is absolute by looking at one character: if it
// does not start with '/', unixFullPathname treats it as relative, prepends
// getcwd() and opens that instead. Every path on this console starts with a
// mount -- sdmc:/switch/dusklight/cache/dawn_cache.db -- so the shader cache
// and the pipeline cache were both being looked for somewhere that does not
// exist, and both answered SQLITE_CANTOPEN.
//
// It is the same mistake std::filesystem makes here, and the game already works
// around that one (src/dusk/data.cpp, is_mount_qualified). This is that fix for
// SQLite.
//
// Rather than a VFS of our own, this is the stock "unix-none" VFS -- SQLite's
// unix implementation with locking disabled, which is what libnx needs since it
// has no fcntl(F_SETLK) -- with a single method replaced. Everything that
// actually touches the filesystem stays SQLite's own code; only the question
// "what is the full name of this path" is answered differently.
#ifdef __SWITCH__
#include "sqlite_switch_paths.hpp"

#include <sqlite3.h>

#include <cstring>

#include "internal.hpp"

namespace aurora::sqlite {
namespace {

constexpr Module Log{"aurora::sqlite"};

sqlite3_vfs* g_base = nullptr;
sqlite3_vfs g_vfs{};

// "<device>:/..." -- a mount name, a colon, then a slash. Anything else is
// handed back to SQLite to resolve the way it always would.
bool is_mount_qualified(const char* path) {
  if (path == nullptr) {
    return false;
  }
  const char* colon = std::strchr(path, ':');
  if (colon == nullptr || colon == path || colon[1] != '/') {
    return false;
  }
  for (const char* c = path; c < colon; c++) {
    const bool ok = (*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
                    (*c >= '0' && *c <= '9') || *c == '-' || *c == '_';
    if (!ok) {
      return false;
    }
  }
  return true;
}

int full_pathname(sqlite3_vfs* vfs, const char* path, int outLen, char* out) {
  if (!is_mount_qualified(path)) {
    return g_base->xFullPathname(vfs, path, outLen, out);
  }
  // Already as absolute as this filesystem gets: copy it through untouched.
  const size_t length = std::strlen(path);
  if (outLen <= 0 || length + 1 > static_cast<size_t>(outLen)) {
    return SQLITE_CANTOPEN;
  }
  std::memcpy(out, path, length + 1);
  return SQLITE_OK;
}

}  // namespace

bool register_path_vfs() {
  if (g_vfs.zName != nullptr) {
    return true;
  }
  g_base = sqlite3_vfs_find("unix-none");
  if (g_base == nullptr) {
    Log.error("SQLite has no unix-none VFS; the caches cannot be opened");
    return false;
  }
  // Copy the whole thing, so a future SQLite that adds methods keeps working,
  // and replace exactly one.
  //
  // pAppData is left alone, and that is the point: it looks like a spare field
  // for whoever wraps a VFS, and it is not -- SQLite's unix implementation
  // keeps its own syscall table there and dereferences it on the first open.
  // Putting the base VFS in it crashed the console with an alignment fault
  // before a single file was touched.
  g_vfs = *g_base;
  g_vfs.zName = kVfsName;
  g_vfs.xFullPathname = full_pathname;

  const int ret = sqlite3_vfs_register(&g_vfs, 0);
  if (ret != SQLITE_OK) {
    Log.error("Could not register the {} VFS: {}", kVfsName, ret);
    g_vfs.zName = nullptr;
    return false;
  }
  return true;
}

}  // namespace aurora::sqlite
#endif
