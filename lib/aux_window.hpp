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

} // namespace aurora::auxwin

#endif
