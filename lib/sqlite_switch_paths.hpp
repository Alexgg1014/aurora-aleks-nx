// aleks2: see sqlite_switch_paths.cpp -- the stock unix-none VFS with mount
// qualified paths ("sdmc:/...") understood as absolute.
#pragma once

namespace aurora::sqlite {

// The name to pass as sqlite3_open_v2's zVfs on Switch.
inline constexpr const char* kVfsName = "switch-paths";

// Registers it. Safe to call more than once; false means SQLite is unusable
// here and the caller should carry on without a cache.
bool register_path_vfs();

}  // namespace aurora::sqlite
