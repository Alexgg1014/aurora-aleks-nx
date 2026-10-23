// aleks2 (deko3d P1): see dk_runtime.hpp. MIT, this project's.
#include "dk_runtime.hpp"

#include "logging.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace aurora::gfx::dk {
namespace {
constexpr Module Log{"aurora::deko"};
constexpr const char* kOptionsPath = "sdmc:/switch/dusklight/deko3d.ini";
constexpr const char* kMarkerPath = "sdmc:/switch/dusklight/cache/deko3d_boot.txt";
constexpr int kFailedStartsBeforeFallback = 2;

std::atomic<bool> g_active{false};
bool g_fellBack = false;
bool g_markerWritten = false;
} // namespace

bool active() noexcept { return g_active.load(std::memory_order_relaxed); }
void set_active(bool on) noexcept { g_active.store(on, std::memory_order_relaxed); }

void load_options() noexcept {
  FILE* f = std::fopen(kOptionsPath, "r");
  if (f == nullptr) {
    Log.info("lab options: no {} (defaults)", kOptionsPath);
    return;
  }
  char line[256];
  int count = 0;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    std::string s{line};
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
      s.pop_back();
    }
    const size_t start = s.find_first_not_of(" \t");
    if (start == std::string::npos || s[start] == '#' || s[start] == ';' || s[start] == '[') {
      continue;
    }
    const size_t eq = s.find('=', start);
    if (eq == std::string::npos) {
      continue;
    }
    std::string name = s.substr(start, eq - start);
    while (!name.empty() && (name.back() == ' ' || name.back() == '\t')) {
      name.pop_back();
    }
    std::string value = s.substr(eq + 1);
    const size_t vs = value.find_first_not_of(" \t");
    value = vs == std::string::npos ? std::string{} : value.substr(vs);
    if (name.empty()) {
      continue;
    }
    setenv(name.c_str(), value.c_str(), 1);
    Log.info("lab option {}={}", name, value);
    ++count;
  }
  std::fclose(f);
  Log.info("lab options: {} from {}", count, kOptionsPath);
}

bool boot_attempt() noexcept {
  int failed = 0;
  if (FILE* f = std::fopen(kMarkerPath, "r")) {
    if (std::fscanf(f, "%d", &failed) != 1) {
      failed = 1;
    }
    std::fclose(f);
  }
  if (failed >= kFailedStartsBeforeFallback) {
    Log.warn("{} starts in a row with deko3d never presented a frame: this start uses Dawn/NVK", failed);
    std::remove(kMarkerPath);
    g_fellBack = true;
    return false;
  }
  if (FILE* f = std::fopen(kMarkerPath, "w")) {
    std::fprintf(f, "%d\n", failed + 1);
    std::fclose(f);
    g_markerWritten = true;
  }
  return true;
}

void boot_presented() noexcept {
  if (!g_markerWritten) {
    return;
  }
  g_markerWritten = false;
  std::remove(kMarkerPath);
  Log.info("first frame presented with deko3d: boot marker cleared");
}

bool fell_back() noexcept { return g_fellBack; }

} // namespace aurora::gfx::dk

extern "C" void aurora_dk_log(const char* line, size_t length) {
  aurora::log_internal(LOG_INFO, "aurora::deko", line, static_cast<unsigned int>(length));
}
