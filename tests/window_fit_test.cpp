// Host test for lib/window_fit.hpp (proportional framebuffer clamp used by the Switch build).
#include "../lib/window_fit.hpp"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

using aurora::window::fit_within;
using aurora::window::FitSize;

static int fails = 0;
static void chk(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL  %s\n", what);
    ++fails;
  }
}
static bool is(FitSize s, int64_t w, int64_t h) { return s.w == w && s.h == h; }

int main() {
  chk(is(fit_within(1920, 1080, 1280, 720), 1280, 720), "1920x1080 in 1280x720");
  chk(is(fit_within(11264, 6336, 1920, 1080), 1920, 1080), "x12 (11264x6336) in 1920x1080");
  chk(is(fit_within(2880, 2160, 1280, 720), 960, 720), "4:3 2880x2160 in 1280x720");
  chk(is(fit_within(720, 540, 1280, 720), 720, 540), "720x540 unchanged");
  chk(is(fit_within(1280, 720, 1280, 720), 1280, 720), "exact fit unchanged");
  chk(is(fit_within(3840, 2160, 1920, 1080), 1920, 1080), "2160p docked");
  chk(is(fit_within(0, 0, 1280, 720), 1, 1), "zero input");
  chk(is(fit_within(-5, 720, 1280, 720), 1, 1), "negative width");
  chk(is(fit_within(1280, -1, 1280, 720), 1, 1), "negative height");
  chk(is(fit_within(100, 100, 0, 0), 1, 1), "non-positive maximum");
  chk(is(fit_within(100000, 10, 1280, 720), 1280, 1), "extreme panorama keeps at least 1 row");
  chk(is(fit_within(10, 100000, 1280, 720), 1, 720), "extreme tower keeps at least 1 column");

  // Invariants over a wide table of sizes and both maxima.
  const int64_t sizes[] = {1,    2,    3,    7,    64,   360,  480,  528,  540,  640,  720,  800,
                           960,  1056, 1080, 1280, 1440, 1584, 1600, 1920, 2160, 2880, 3840, 6336,
                           7680, 11264, 16384, 100000};
  const FitSize maxima[] = {{1280, 720}, {1920, 1080}};
  for (const auto m : maxima) {
    for (const int64_t w : sizes) {
      for (const int64_t h : sizes) {
        const FitSize r = fit_within(w, h, m.w, m.h);
        chk(r.w >= 1 && r.h >= 1, "minimum 1x1");
        chk(r.w <= m.w && r.h <= m.h, "never exceeds the maximum");
        const FitSize again = fit_within(r.w, r.h, m.w, m.h);
        chk(is(again, r.w, r.h), "idempotent");
        if (w <= m.w && h <= m.h) {
          chk(is(r, w, h), "sizes that fit are unchanged");
        } else if (r.w > 1 && r.h > 1 && w > 1 && h > 1) {
          // Aspect error: compare the rounded axis against the exact proportional value
          // within one pixel. The limiting axis is exact.
          const bool widthLimits = w * m.h >= h * m.w;
          if (widthLimits) {
            const double exactH = static_cast<double>(h) * static_cast<double>(m.w) / static_cast<double>(w);
            chk(std::abs(static_cast<double>(r.h) - exactH) <= 1.0, "aspect within 1 px (height)");
            chk(r.w == m.w, "width is the limiting axis");
          } else {
            const double exactW = static_cast<double>(w) * static_cast<double>(m.h) / static_cast<double>(h);
            chk(std::abs(static_cast<double>(r.w) - exactW) <= 1.0, "aspect within 1 px (width)");
            chk(r.h == m.h, "height is the limiting axis");
          }
        }
      }
    }
  }
  if (fails != 0) {
    std::printf("%d failure(s)\n", fails);
    return 1;
  }
  std::printf("window_fit: all checks passed\n");
  return 0;
}
