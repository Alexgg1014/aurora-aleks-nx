#pragma once

// Proportional fit of a framebuffer size inside a maximum size.
//
// The Switch build derives the largest framebuffer the console can show from the applet
// operation mode (1280x720 handheld, 1920x1080 docked). Anything the settings ask for beyond
// that is scaled down uniformly, so the picture is never stretched. An independent min() on
// width and height would change the aspect ratio.
//
// Standard library only, so it can be tested on a host without SDL.

#include <cstdint>

namespace aurora::window {

struct FitSize {
  int64_t w;
  int64_t h;
};

// Returns (w, h) unchanged when it already fits. Otherwise scales both by the limiting axis,
// rounding to nearest. Properties:
//   * never exceeds maxW x maxH (a non-positive maximum is treated as 1);
//   * never smaller than 1x1;
//   * the aspect ratio is kept to within one pixel on the rounded axis;
//   * idempotent: fit_within(fit_within(x)) == fit_within(x);
//   * non-positive input is returned as 1x1 without dividing.
constexpr FitSize fit_within(int64_t w, int64_t h, int64_t maxW, int64_t maxH) noexcept {
  if (maxW < 1) {
    maxW = 1;
  }
  if (maxH < 1) {
    maxH = 1;
  }
  if (w < 1 || h < 1) {
    return {1, 1};
  }
  if (w <= maxW && h <= maxH) {
    return {w, h};
  }
  // Compare w/maxW with h/maxH by cross product. Inputs are bounded by int32 range in
  // practice, so the products fit in 64 bits.
  int64_t outW;
  int64_t outH;
  if (w * maxH >= h * maxW) {
    // Width limits.
    outW = maxW;
    outH = (h * maxW + w / 2) / w;
  } else {
    outH = maxH;
    outW = (w * maxH + h / 2) / h;
  }
  if (outW < 1) {
    outW = 1;
  }
  if (outH < 1) {
    outH = 1;
  }
  if (outW > maxW) {
    outW = maxW;
  }
  if (outH > maxH) {
    outH = maxH;
  }
  return {outW, outH};
}

} // namespace aurora::window
