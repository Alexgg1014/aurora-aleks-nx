#pragma once

#ifdef AURORA_ENABLE_GX

#include <webgpu/webgpu_cpp.h>

union SDL_Event;

namespace aurora::auxwin {

/// Returns true if the event targets the auxiliary window and was consumed.
/// Called from the main event loop before regular processing.
bool filter_event(const SDL_Event& event);

/// Present the auxiliary output. Render worker thread only, after the main
/// frame's queue submit.
///
/// `frameTarget` is the primary swapchain view for this frame, or null when
/// the frame is not presentable. The desktop/Android implementation owns its
/// own surface and ignores it; the Switch implementation has no second panel
/// to own and composites into this one. Passed as an argument rather than
/// stashed in a global so it cannot go stale, and so nothing on the main
/// frame's path has to be reordered to supply it.
void present(const wgpu::TextureView& frameTarget);

#ifdef __SWITCH__
/// aleks2 (deko3d rc12): the companion for the deko3d present (lib/deko/dk_present.cpp), which does the GPU work
/// present() does with WebGPU. Render worker. False: nothing to draw this frame.
struct DekoCompanion {
  WGPUTextureView newSource = nullptr; ///< a freshly drawn dashboard to copy into the deko3d-owned image, or null
  uint32_t width = 0, height = 0;      ///< the size of the image to draw (the new source's, else the owned one's)
  int x = 0, y = 0, w = 0, h = 0;      ///< where, in swapchain pixels
  bool rotated = false;                ///< Flip Grip: turned 270 degrees like the game picture
  float dim = 0.f;                     ///< out = src * (1 - dim)
};
bool deko_companion(int outW, int outH, uint32_t ownedW, uint32_t ownedH, DekoCompanion* out);
#endif

} // namespace aurora::auxwin

#endif
