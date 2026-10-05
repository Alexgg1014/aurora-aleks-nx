// aleks2: the public face of upstream's aurora::io.
//
// The module and write_file_atomic itself are upstream's (lib/io.cpp); what was
// missing was a header the game could include, since lib/io.hpp is internal to
// Aurora. This is that header and nothing more.
//
// The game leans on it where a half-written file would be worse than no file:
// the settings and the guide catalogue. On Switch that matters twice over --
// libnx cannot rename onto an existing file, so the naive write-then-rename
// loses the original.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

namespace aurora::io {

// Writes through a temporary and swaps it in, so a reader sees either the old
// contents or the new ones. False if the file is unchanged on disk.
bool write_file_atomic(const std::filesystem::path& path, std::span<const uint8_t> data) noexcept;

} // namespace aurora::io
