#pragma once

/*
 * aux_layout_switch.hpp — where the game and the companion go on the Switch's
 * single panel.
 *
 * Renderer-independent by design, and deliberately not a framework: this
 * computes rectangles and nothing else. No WebGPU type appears here, no
 * drawing happens in the .cpp, and the presenter stays the only place that
 * knows how pixels reach the screen. That split is what let the Flip Grip
 * modes be added to the ALEKS ports without the presenter learning anything
 * new about layout, and it is why this file is a straight adaptation of
 * port/port_display_layout.c from that work.
 *
 * Two modes only. There is no DUAL horizontal here — see the project's layout
 * decision.
 */

#include <cstdint>

namespace aurora::auxwin::layout {

enum Mode : int {
  /// The game alone, exactly as the port presents it today. This mode must
  /// never route through compute() — see the note in the .cpp.
  kNormal = 0,
  /// Game above, companion below, on a 720x1280 logical canvas rotated 270
  /// degrees onto the physical panel. Handheld only: you cannot turn a dock.
  kFlip270 = 1,
};

/// The Flip Grip logical canvas. 720 wide because that is the panel's short
/// edge; 1280 tall because that is its long edge. The console is physically
/// turned, so this portrait canvas is what the player actually sees.
inline constexpr int kFlipCanvasW = 720;
inline constexpr int kFlipCanvasH = 1280;

struct Rect {
  int x = 0, y = 0, w = 0, h = 0;
  constexpr bool valid() const { return w > 0 && h > 0; }
};

struct Layout {
  Mode mode = kNormal;
  /// The space `game` and `companion` are expressed in. Equal to the physical
  /// output in kNormal; kFlipCanvasW/H in kFlip270.
  int logicalW = 0, logicalH = 0;
  /// Degrees the logical canvas is turned by on the way to the panel. 0 or 270.
  int rotationDegrees = 0;
  /// Both in LOGICAL coordinates. companion.w == 0 means "draw only the game".
  Rect game;
  Rect companion;
};

/// The mode being presented. Pushed from the dusk layer; read fresh every
/// frame so a change takes effect on the next one with no cached state.
Mode mode() noexcept;
void set_mode(Mode m) noexcept;

/// Companion size as a percentage of the space left over (50..100) and the gap
/// between the two, in logical pixels. Invalid combinations are CLAMPED, never
/// skipped, so every setting the UI offers stays selectable.
void set_params(int companionPct, int gap) noexcept;

/// Fill `out` for a panel of outW x outH. `gameAspect` is the game viewport's
/// width/height (4/3 or 16/9). Pure arithmetic over its arguments; keeps no
/// state beyond the mode/params above, and is safe to call every frame.
void compute(int outW, int outH, float gameAspect, Layout* out) noexcept;

/// The companion's rect in kNormal mode, while it is summoned, in PHYSICAL
/// panel pixels: aspect-fitted at full height and centred.
///
/// Deliberately NOT part of compute(). Normal mode's game path must never
/// route through the layout code, and the summoned companion is an overlay on
/// top of an unchanged game -- not a layout. Same split the ALEKS ports use.
Rect normal_overlay_rect(int outW, int outH, float companionAspect) noexcept;

/// Whether the summoned companion is up in kNormal mode. Meaningless in
/// kFlip270, where the companion is always present.
bool normal_overlay_visible() noexcept;
void set_normal_overlay_visible(bool visible) noexcept;

/// Map a LOGICAL rect onto the physical panel for the current rotation.
/// Width and height swap under 270 degrees. Also used by the touch path, which
/// must run the inverse of exactly this.
Rect to_physical(const Rect& logical, int outW, int outH, int rotationDegrees) noexcept;

} // namespace aurora::auxwin::layout
