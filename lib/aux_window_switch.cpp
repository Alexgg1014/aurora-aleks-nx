/*
 * aux_window_switch.cpp — the aurora::auxwin API on a console with ONE panel.
 *
 * The Android donor's whole design assumes a second presentation surface: an
 * SDL window on desktop, an ANativeWindow from a Presentation on Android.
 * Switch 1 drives either the internal panel or HDMI, never both, so there is
 * no second surface to hand a texture to, and no amount of engineering makes
 * one.
 *
 * What there IS, and what makes this cheap: the companion is already rendered
 * to an offscreen texture by dusk::dualscreen, and aurora already puts the
 * game on screen with a viewport'd fullscreen-triangle blit. So the companion
 * is one more blit of the same kind, into a different rect of the SAME
 * swapchain image. Nothing above this file changes: the API below is exactly
 * the one the donor calls, so the companion sources and the whole
 * capture/present bracket in dusk/dualscreen.cpp work unmodified.
 *
 * Everything here happens AFTER the main frame's queue submit and BEFORE the
 * swapchain is presented, so the game's image is already in the target and the
 * companion composites on top of it with LoadOp::Load. No step the main screen
 * depends on lives in this file. That is the rule the donor learned the hard
 * way (docs/dualscreen-fork.md), and on Switch it matters more still, because
 * Normal mode has to keep working with the companion off.
 */

#include <aurora/aux_window.hpp>

#include <aurora/aux_layout_switch.hpp>

#include "aux_window.hpp"
#include "internal.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <mutex>
#include <vector>

namespace aurora::auxwin {
namespace {
Module Log("aurora::auxwin");

namespace lay = layout;

struct Source {
  wgpu::TextureView view;
  uint32_t width = 0;
  uint32_t height = 0;
};

std::mutex g_mutex;
Source g_source;
wgpu::Sampler g_sampler;
bool g_active = false;

/// Our OWN copy of the dashboard, and the reason it exists.
///
/// set_source() hands us a view from gfx::resolve_pass, whose contract says
/// "valid for the current frame" -- it is a pooled snapshot that gets recycled.
/// On desktop that is fine because the aux surface presents it in the same
/// frame. Here the companion is redrawn only every few frames (see the cadence
/// in dusk/dualscreen.cpp) and the panel keeps showing it in between, so
/// holding the pooled view would eventually present a recycled texture.
///
/// So on the frames where a new source actually arrives we blit it into this
/// texture, which we own outright, and every present reads from here. The extra
/// blit costs one small pass on redraw frames only -- far less than the GX
/// redraw it lets us skip.
wgpu::Texture g_ownedTexture;
wgpu::TextureView g_ownedView;
uint32_t g_ownedW = 0;
uint32_t g_ownedH = 0;
bool g_ownedValid = false;
bool g_sourceIsNew = false;

std::atomic<float> g_dim{0.f};

/// The companion viewport for the frame just presented, in LOGICAL pixels.
/// get_surface_size() reports this so the donor's computeAuxCanvas keeps
/// matching its canvas to the region it will occupy. That function never cared
/// whether the "surface" was a panel or a rectangle, which is the single most
/// fortunate thing about the donor's API for this port.
std::atomic<uint32_t> g_reportedW{0};
std::atomic<uint32_t> g_reportedH{0};

/// Reported before the first frame has established a real rect, so an early
/// caller gets a sane canvas instead of zero.
constexpr uint32_t kFallbackW = 720;
constexpr uint32_t kFallbackH = 740;

/// Render-target cap for the dashboard, in pixels on its longest side.
///
/// computeAuxCanvas() in dusk/dualscreen.cpp takes whatever get_surface_size()
/// reports and, if it is inside its own 320..2048 range, renders the dashboard
/// at exactly that size. Reporting the raw Flip rect (720x740) would render
/// 532,800 px of 2D every redraw -- MORE pixels than the game viewport itself
/// (720x540) on a GM20B that measured ~30 fps at 540p before any of this
/// existed. Capping the long side to 640 gives 623x640, about 75% of the
/// pixels, for a mild softening the blit smooths over.
///
/// The aspect is preserved exactly, so the dashboard's own layout maths (which
/// matches the canvas to this ratio) is unaffected -- only the sampling density
/// changes.
///
/// Settable, because how much of the frame budget the second screen may take is
/// the player's call and not ours: 720 is the panel's own resolution (no
/// downscale at all), 640 the default, and a Switch that is struggling can go
/// lower still. See set_target_long_side_cap.
constexpr uint32_t kTargetLongSideCapDefault = 640;
constexpr uint32_t kTargetLongSideCapMin = 320;
constexpr uint32_t kTargetLongSideCapMax = 2048;
std::atomic<uint32_t> g_targetLongSideCap{kTargetLongSideCapDefault};

/// Shape of the summoned companion in Normal mode. Matches what the Flip
/// layout gives it, so the two modes do not need different art.
constexpr float kNormalOverlayAspect = 720.f / 740.f;

void ensure_sampler_locked() {
  if (g_sampler != nullptr) {
    return;
  }
  const wgpu::SamplerDescriptor samplerDescriptor{
      .label = "Companion Sampler",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .addressModeW = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
  };
  g_sampler = webgpu::g_device.CreateSampler(&samplerDescriptor);
}

/// Aspect-fit srcW x srcH inside dst; never stretch. Result is in dst's space.
lay::Rect fit(const lay::Rect& dst, uint32_t srcW, uint32_t srcH) {
  if (srcW == 0 || srcH == 0 || !dst.valid()) {
    return dst;
  }
  const float scale = std::min(static_cast<float>(dst.w) / static_cast<float>(srcW),
                               static_cast<float>(dst.h) / static_cast<float>(srcH));
  const int w = std::max(static_cast<int>(std::lround(static_cast<float>(srcW) * scale)), 1);
  const int h = std::max(static_cast<int>(std::lround(static_cast<float>(srcH) * scale)), 1);
  return lay::Rect{dst.x + (dst.w - w) / 2, dst.y + (dst.h - h) / 2, w, h};
}

/// Copy a freshly drawn dashboard into the texture we own, (re)creating it when
/// the size changes. Done as a render-pass blit rather than a texture-to-texture
/// copy because set_source only hands us a view, not the texture behind it.
///
/// The target format matches the swapchain so the existing copy pipelines --
/// which are built against surfaceConfiguration.format -- can be reused for
/// both this pass and the present blit, with no extra pipeline.
void adopt_source(const Source& source, const wgpu::Sampler& sampler) {
  const auto format = webgpu::g_graphicsConfig.surfaceConfiguration.format;
  if (!g_ownedValid || g_ownedW != source.width || g_ownedH != source.height || g_ownedTexture == nullptr) {
    const wgpu::TextureDescriptor descriptor{
        .label = "Companion Owned Target",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
        .dimension = wgpu::TextureDimension::e2D,
        .size = {.width = source.width, .height = source.height, .depthOrArrayLayers = 1},
        .format = format,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    g_ownedView = {};
    g_ownedTexture = webgpu::g_device.CreateTexture(&descriptor);
    if (g_ownedTexture == nullptr) {
      g_ownedValid = false;
      return;
    }
    g_ownedView = g_ownedTexture.CreateView();
    g_ownedW = source.width;
    g_ownedH = source.height;
  }
  if (g_ownedView == nullptr || webgpu::g_CopyPipeline == nullptr) {
    g_ownedValid = false;
    return;
  }

  const webgpu::TextureWithSampler binding{.view = source.view, .sampler = sampler};
  const auto bindGroup = webgpu::create_copy_bind_group(binding);
  if (bindGroup == nullptr) {
    return;
  }
  const wgpu::CommandEncoderDescriptor encoderDescriptor{.label = "Companion adopt encoder"};
  const auto encoder = webgpu::g_device.CreateCommandEncoder(&encoderDescriptor);
  {
    const std::array attachments{
        wgpu::RenderPassColorAttachment{
            .view = g_ownedView,
            .loadOp = wgpu::LoadOp::Clear,
            .storeOp = wgpu::StoreOp::Store,
            .clearValue = {0.0, 0.0, 0.0, 1.0},
        },
    };
    const wgpu::RenderPassDescriptor renderPassDescriptor{
        .label = "Companion adopt pass",
        .colorAttachmentCount = attachments.size(),
        .colorAttachments = attachments.data(),
    };
    const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
    pass.SetPipeline(webgpu::g_CopyPipeline);
    pass.SetBindGroup(0, bindGroup, 0, nullptr);
    pass.SetViewport(0.f, 0.f, static_cast<float>(g_ownedW), static_cast<float>(g_ownedH), 0.f, 1.f);
    pass.Draw(3);
    pass.End();
  }
  const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Companion adopt buffer"};
  const auto buffer = encoder.Finish(&cmdBufDescriptor);
  webgpu::g_queue.Submit(1, &buffer);
  g_ownedValid = true;
}

/// Where the companion goes this frame, in PHYSICAL panel pixels, plus whether
/// the blit has to be turned. False means do not draw it at all, which is the
/// ordinary case in Normal mode.
bool companion_target(int outW, int outH, uint32_t srcW, uint32_t srcH, lay::Rect* outRect, bool* outRotated,
                      uint32_t* reportW, uint32_t* reportH) {
  if (lay::mode() == lay::kFlip270) {
    lay::Layout l{};
    // 4:3 is Twilight Princess's authored aspect and the Flip default; the
    // dusk layer overrides it through set_params when the user picks 16:9.
    lay::compute(outW, outH, 4.f / 3.f, &l);
    if (l.mode != lay::kFlip270 || !l.companion.valid()) {
      return false;
    }
    *reportW = static_cast<uint32_t>(l.companion.w);
    *reportH = static_cast<uint32_t>(l.companion.h);
    const lay::Rect fitted = fit(l.companion, srcW, srcH);
    *outRect = lay::to_physical(fitted, outW, outH, l.rotationDegrees);
    *outRotated = true;
    return outRect->valid();
  }

  if (!lay::normal_overlay_visible()) {
    return false; // game only: the default in Normal mode
  }
  const lay::Rect overlay = lay::normal_overlay_rect(outW, outH, kNormalOverlayAspect);
  if (!overlay.valid()) {
    return false;
  }
  *reportW = static_cast<uint32_t>(overlay.w);
  *reportH = static_cast<uint32_t>(overlay.h);
  *outRect = fit(overlay, srcW, srcH);
  *outRotated = false;
  return outRect->valid();
}
} // namespace

bool create(const CreateInfo& info) {
  // There is no window to open. The "aux window" is a rect of the panel the
  // game is already using, so this only arms the compositor.
  (void)info;
  std::lock_guard lock{g_mutex};
  ensure_sampler_locked();
  g_active = true;
  Log.info("Companion compositor armed (single panel; no aux surface)");
  return true;
}

void destroy() {
  std::lock_guard lock{g_mutex};
  g_active = false;
  g_source = {};
  g_sampler = {};
  g_ownedView = {};
  g_ownedTexture = {};
  g_ownedW = 0;
  g_ownedH = 0;
  g_ownedValid = false;
  g_sourceIsNew = false;
  g_reportedW = 0;
  g_reportedH = 0;
}

bool is_open() noexcept {
  std::lock_guard lock{g_mutex};
  return g_active;
}

void set_source(wgpu::TextureView view, uint32_t width, uint32_t height) {
  std::lock_guard lock{g_mutex};
  g_source = Source{std::move(view), width, height};
  // Flag it so present() copies it into our own texture exactly once. On the
  // frames the cadence skips, set_source is never called and this stays false,
  // so present() reuses what it already owns.
  g_sourceIsNew = true;
}

void set_dim(float amount) { g_dim.store(std::clamp(amount, 0.f, 1.f)); }

void set_target_long_side_cap(uint32_t pixels) {
  g_targetLongSideCap.store(std::clamp(pixels, kTargetLongSideCapMin, kTargetLongSideCapMax),
                            std::memory_order_relaxed);
}

bool get_surface_size(uint32_t* width, uint32_t* height) {
  uint32_t w = g_reportedW.load();
  uint32_t h = g_reportedH.load();
  if (w == 0 || h == 0) {
    w = kFallbackW;
    h = kFallbackH;
  }
  // Cap the render target, preserving the aspect exactly (see the constant).
  const uint32_t cap = g_targetLongSideCap.load(std::memory_order_relaxed);
  const uint32_t longSide = std::max(w, h);
  if (longSide > cap) {
    w = std::max(w * cap / longSide, 1u);
    h = std::max(h * cap / longSide, 1u);
  }
  if (width != nullptr) {
    *width = w;
  }
  if (height != nullptr) {
    *height = h;
  }
  return is_open();
}

// --- Not applicable on Switch --------------------------------------------
// Kept as no-ops rather than removed: the donor calls them unconditionally,
// and an empty implementation is far less likely to rot than #ifdefs scattered
// through shared code.

void set_native_window(void* nativeWindow, uint32_t width, uint32_t height) {
  (void)nativeWindow;
  (void)width;
  (void)height;
}

bool consume_close_request() { return false; }

bool filter_event(const SDL_Event& event) {
  (void)event;
  return false; // there is no SDL event loop on Switch
}

// Capture is a development aid on the donor and ships disabled there. Not
// wired here; the console has its own capture button.
void request_capture() {}

bool take_capture(std::vector<uint8_t>& out, uint32_t* width, uint32_t* height) {
  (void)out;
  (void)width;
  (void)height;
  return false;
}

void present(const wgpu::TextureView& frameTarget) {
  if (frameTarget == nullptr) {
    return; // frame not presentable; the main path already reported it
  }

  Source source;
  wgpu::Sampler sampler;
  bool sourceIsNew = false;
  {
    std::lock_guard lock{g_mutex};
    if (!g_active) {
      return;
    }
    source = g_source;
    sampler = g_sampler;
    sourceIsNew = g_sourceIsNew;
    g_sourceIsNew = false;
  }
  if (sampler == nullptr) {
    return;
  }
  // Adopt a freshly drawn dashboard into our own texture. Only on redraw
  // frames; the cadence means most frames fall straight through to the blit
  // below reusing what we already hold.
  if (sourceIsNew && source.view != nullptr && source.width != 0 && source.height != 0) {
    adopt_source(source, sampler);
  }
  if (!g_ownedValid || g_ownedView == nullptr) {
    return; // nothing has ever been drawn yet
  }
  const Source owned{g_ownedView, g_ownedW, g_ownedH};

  const int outW = static_cast<int>(webgpu::g_graphicsConfig.surfaceConfiguration.width);
  const int outH = static_cast<int>(webgpu::g_graphicsConfig.surfaceConfiguration.height);
  if (outW <= 0 || outH <= 0) {
    return;
  }

  lay::Rect dst{};
  bool rotated = false;
  uint32_t reportW = 0;
  uint32_t reportH = 0;
  if (!companion_target(outW, outH, owned.width, owned.height, &dst, &rotated, &reportW, &reportH)) {
    return;
  }
  g_reportedW.store(reportW);
  g_reportedH.store(reportH);

  // out = src * constant, applied at PRESENT time rather than painted into the
  // picture. Deliberate in the donor and carried over unchanged: on frames
  // where the capture is skipped (a stage transition tears the HUD meter down)
  // the companion keeps presenting its last texture and the fade keeps
  // advancing over it.
  const float dim = g_dim.load();
  const double keep = 1.0 - static_cast<double>(dim);
  const wgpu::Color blendConstant{keep, keep, keep, 1.0};
  const bool wantDim = dim > 0.001f;

  wgpu::RenderPipeline pipeline;
  bool usingDim = false;
  if (rotated) {
    if (wantDim && webgpu::g_CopyDimRot270Pipeline != nullptr) {
      pipeline = webgpu::g_CopyDimRot270Pipeline;
      usingDim = true;
    } else {
      pipeline = webgpu::g_CopyRot270Pipeline;
    }
  } else {
    if (wantDim && webgpu::g_CopyDimPipeline != nullptr) {
      pipeline = webgpu::g_CopyDimPipeline;
      usingDim = true;
    } else {
      pipeline = webgpu::g_CopyPipeline;
    }
  }
  if (pipeline == nullptr) {
    return;
  }

  const webgpu::TextureWithSampler binding{.view = owned.view, .sampler = sampler};
  const auto bindGroup = webgpu::create_copy_bind_group(binding);
  if (bindGroup == nullptr) {
    return;
  }

  const wgpu::CommandEncoderDescriptor encoderDescriptor{.label = "Companion encoder"};
  const auto encoder = webgpu::g_device.CreateCommandEncoder(&encoderDescriptor);
  {
    // Load, never Clear: the game's frame is already in this image and the
    // companion goes on top of it.
    const std::array attachments{
        wgpu::RenderPassColorAttachment{
            .view = frameTarget,
            .loadOp = wgpu::LoadOp::Load,
            .storeOp = wgpu::StoreOp::Store,
        },
    };
    const wgpu::RenderPassDescriptor renderPassDescriptor{
        .label = "Companion blit pass",
        .colorAttachmentCount = attachments.size(),
        .colorAttachments = attachments.data(),
    };
    const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
    pass.SetPipeline(pipeline);
    if (usingDim) {
      pass.SetBlendConstant(&blendConstant);
    }
    pass.SetBindGroup(0, bindGroup, 0, nullptr);
    pass.SetViewport(static_cast<float>(dst.x), static_cast<float>(dst.y), static_cast<float>(dst.w),
                     static_cast<float>(dst.h), 0.f, 1.f);
    pass.Draw(3);
    pass.End();
  }
  const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Companion command buffer"};
  const auto buffer = encoder.Finish(&cmdBufDescriptor);
  webgpu::g_queue.Submit(1, &buffer);
}

bool deko_companion(int outW, int outH, uint32_t ownedW, uint32_t ownedH, DekoCompanion* out) {
  // aleks2 (deko3d rc12): present()'s decisions, the GPU work left to dk_present.cpp
  Source source;
  bool sourceIsNew = false;
  {
    std::lock_guard lock{g_mutex};
    if (!g_active) {
      return false;
    }
    source = g_source;
    sourceIsNew = g_sourceIsNew;
    g_sourceIsNew = false;
  }
  *out = {};
  if (sourceIsNew && source.view != nullptr && source.width != 0 && source.height != 0) {
    out->newSource = source.view.Get();
    ownedW = source.width;
    ownedH = source.height;
  }
  if (ownedW == 0 || ownedH == 0 || outW <= 0 || outH <= 0) {
    return false; // nothing has ever been drawn yet
  }
  lay::Rect dst{};
  bool rotated = false;
  uint32_t reportW = 0;
  uint32_t reportH = 0;
  const bool draw = companion_target(outW, outH, ownedW, ownedH, &dst, &rotated, &reportW, &reportH);
  if (draw) {
    g_reportedW.store(reportW);
    g_reportedH.store(reportH);
  }
  out->width = ownedW;
  out->height = ownedH;
  if (draw) { // otherwise w = h = 0: copy the new source, draw nothing
    out->x = dst.x;
    out->y = dst.y;
    out->w = dst.w;
    out->h = dst.h;
  }
  out->rotated = rotated;
  out->dim = g_dim.load();
  // a new source is copied even when the layout hides it, as present() does
  return draw || out->newSource != nullptr;
}

} // namespace aurora::auxwin
