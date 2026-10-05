/*
 * aux_layout_switch.cpp — rectangle arithmetic for the Switch display modes.
 *
 * kNormal IS NOT COMPUTED HERE, ON PURPOSE.
 *
 * Normal mode must stay the port's existing presentation path: the EFB->XFB
 * blit straight through set_present_viewport, no intermediate surface, no
 * second allocation, no new scaling code. Routing it through this file "for
 * consistency" would put every future layout change on the critical path of
 * the mode that has to keep working -- and this port already has a documented
 * case of exactly that going wrong (the aux present that blanked every
 * single-screen device; see docs/dualscreen-fork.md). So the presenter
 * branches once, early, and only the Flip path arrives here. compute()
 * returns a ZEROED layout for kNormal rather than a fitted rect, so a caller
 * that ignores the branch gets an obviously wrong result instead of a subtly
 * wrong one.
 */

#include <aurora/aux_layout_switch.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>

namespace aurora::auxwin::layout {
namespace {
std::atomic<int> g_mode{kNormal};
std::atomic<int> g_companionPct{100};
std::atomic<int> g_gap{0};
std::atomic<bool> g_normalOverlay{false};

constexpr int kMinPct = 50;
constexpr int kMaxPct = 100;

/// The game may never be squeezed below this share of the canvas height. A
/// hand-edited config asking for a huge companion gets clamped rather than
/// handed a sliver of a game.
constexpr float kMaxGameHeightShare = 0.75f;
} // namespace

Mode mode() noexcept {
  const int m = g_mode.load(std::memory_order_relaxed);
  return m == kFlip270 ? kFlip270 : kNormal;
}

void set_mode(Mode m) noexcept {
  g_mode.store(m == kFlip270 ? kFlip270 : kNormal, std::memory_order_relaxed);
}

void set_params(int companionPct, int gap) noexcept {
  g_companionPct.store(std::clamp(companionPct, kMinPct, kMaxPct), std::memory_order_relaxed);
  g_gap.store(std::max(gap, 0), std::memory_order_relaxed);
}

/*
 * FLIP GRIP, on a 720x1280 LOGICAL canvas the presenter then turns 270 degrees
 * onto the physical panel:
 *
 *   +---------------------+  0
 *   |   game 720 x 540    |   4:3 at full canvas width
 *   +---------------------+  540
 *   |        gap          |
 *   +---------------------+
 *   |  companion 720x740  |
 *   +---------------------+  1280
 *
 * The game takes the full canvas width and its own aspect decides its height:
 * 4:3 gives 720x540, 16:9 gives 720x405. Note what does NOT apply here -- the
 * ALEKS ports pinned their game to an exact integer scale because a GBA
 * framebuffer nearest-filtered at 2.625x shimmers. Twilight Princess renders
 * 3D at whatever resolution it is asked for, so the right thing is to render
 * native to this rect and scale nothing at all.
 *
 * FIT RULE, one sentence: the sizes win and the gap gives way. The gap is
 * clamped to whatever room the two sizes leave, so the settings row keeps
 * showing the gap that was asked for and the stack can never overflow.
 */
void compute(int outW, int outH, float gameAspect, Layout* out) noexcept {
  if (out == nullptr) {
    return;
  }
  *out = Layout{};
  out->mode = mode();
  out->logicalW = outW;
  out->logicalH = outH;

  if (out->mode != kFlip270) {
    return; // see the file comment: normal mode never comes through here
  }
  if (outW <= 0 || outH <= 0 || !(gameAspect > 0.f)) {
    out->mode = kNormal; // nothing sane to compute; fall back rather than divide by zero
    return;
  }

  out->logicalW = kFlipCanvasW;
  out->logicalH = kFlipCanvasH;
  out->rotationDegrees = 270;

  int gameW = kFlipCanvasW;
  int gameH = static_cast<int>(std::lround(static_cast<float>(kFlipCanvasW) / gameAspect));
  const int maxGameH = static_cast<int>(kFlipCanvasH * kMaxGameHeightShare);
  if (gameH > maxGameH) {
    // Aspect-fit inside the cap instead of cropping, so an unusual aspect
    // narrows the game rather than eating the companion.
    gameH = maxGameH;
    gameW = static_cast<int>(std::lround(static_cast<float>(gameH) * gameAspect));
    gameW = std::min(gameW, kFlipCanvasW);
  }
  if (gameH < 1) {
    gameH = 1;
  }

  const int pct = std::clamp(g_companionPct.load(std::memory_order_relaxed), kMinPct, kMaxPct);
  const int avail = kFlipCanvasH - gameH;
  int gap = std::clamp(g_gap.load(std::memory_order_relaxed), 0, std::max(avail, 0));
  int bandH = avail - gap;
  if (bandH < 0) {
    bandH = 0;
    gap = avail > 0 ? avail : 0;
  }

  const int compW = kFlipCanvasW * pct / 100;
  const int compH = bandH * pct / 100;

  // Centre the whole stack so the spare canvas splits evenly above and below.
  const int stackH = gameH + gap + compH;
  const int stackY = std::max((kFlipCanvasH - stackH) / 2, 0);

  out->game = Rect{(kFlipCanvasW - gameW) / 2, stackY, gameW, gameH};
  out->companion = Rect{(kFlipCanvasW - compW) / 2, stackY + gameH + gap, compW, compH};
}

bool normal_overlay_visible() noexcept { return g_normalOverlay.load(std::memory_order_relaxed); }

void set_normal_overlay_visible(bool visible) noexcept {
  g_normalOverlay.store(visible, std::memory_order_relaxed);
}

Rect normal_overlay_rect(int outW, int outH, float companionAspect) noexcept {
  if (outW <= 0 || outH <= 0 || !(companionAspect > 0.f)) {
    return Rect{};
  }
  int w = static_cast<int>(std::lround(static_cast<float>(outH) * companionAspect));
  int h = outH;
  if (w > outW) {
    w = outW;
    h = static_cast<int>(std::lround(static_cast<float>(outW) / companionAspect));
  }
  return Rect{(outW - w) / 2, (outH - h) / 2, w, h};
}

/*
 * Logical -> physical for a 270-degree turn.
 *
 * The presenter turns the portrait canvas clockwise onto the landscape panel,
 * so a logical point (lx, ly) lands at physical (ly, outH - lx): the canvas's
 * top-left corner goes to the panel's BOTTOM-left. Solving that for a whole
 * rect swaps width and height.
 *
 * The touch path must run the exact inverse of this, and getting it wrong is
 * not a subtle failure -- in the ALEKS port the first version implemented the
 * inverse of a 90-degree turn and every single tap missed its target. Verify
 * against corners on hardware, not by reading: logical (0,0) must present at
 * physical (0, outH).
 */
Rect to_physical(const Rect& l, int outW, int outH, int rotationDegrees) noexcept {
  if (rotationDegrees != 270) {
    return l;
  }
  (void)outW;
  return Rect{l.y, outH - l.x - l.w, l.h, l.w};
}

} // namespace aurora::auxwin::layout
