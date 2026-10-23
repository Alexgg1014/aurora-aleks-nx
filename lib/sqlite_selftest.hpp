// aleks2 (V1-c3): see sqlite_selftest.cpp.
#pragma once

#include <string>

namespace aurora::sqlite {

struct SelfTestResult {
  std::string report;   ///< one line per probe and per journal/sync pair
  std::string journal;  ///< the preferred pair that passed ("" if none did)
  std::string sync;
};

/// Runs the probes and every pair in `dir` (created and removed), through `vfs`.
SelfTestResult run_sqlite_selftest(const std::string& dir, const char* vfs);

}  // namespace aurora::sqlite
