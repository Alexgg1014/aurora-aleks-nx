#include <aurora/aurora.h>
#include <aurora/time.hpp>
#include <algorithm>
#include <array>

#ifdef AURORA_ENABLE_GX
#include "aux_window.hpp"
#ifdef __SWITCH__
#include <aurora/aux_layout_switch.hpp>
#endif
#include "gfx/resources.hpp"
#include "gfx/frame.hpp"
#include "gfx/recording.hpp"
#include "gfx/render_worker.hpp"
#include "gx/command_processor.hpp"
#include "gx/fifo.hpp"
#include "gx/gx.hpp"
#include "gx/texture.hpp"
#include "imgui.hpp"
#include "webgpu/gpu.hpp"
#include "webgpu/gpu_prof.hpp"
#include "webgpu/gpu_timing.hpp"
#include <webgpu/webgpu_cpp.h>
#endif

#ifdef AURORA_ENABLE_RMLUI
#include "rmlui.hpp"
#if defined(AURORA_GFX_DEKO3D)
// aleks2 (deko3d P1): the run-time selectable deko3d renderer (lib/deko)
#include "dk_gfx.hpp"
#include "dk_runtime.hpp"
#endif
#endif

#include "input.hpp"
#include "internal.hpp"
#include "thread.hpp"
#include "window.hpp"

#include <SDL3/SDL_filesystem.h>
#include <magic_enum.hpp>

#include "system_info.hpp"
#include "tracy/Tracy.hpp"

namespace aurora {
AuroraConfig g_config;
uint32_t g_sdlCustomEventsStart;
char g_gameName[4];

namespace {
constexpr Module Log{"aurora"};

#ifdef AURORA_ENABLE_GX
// GPU
using webgpu::g_device;
using webgpu::g_queue;
using webgpu::g_surface;

uint32_t clamp_scissor_coord(double value, uint32_t maximum) noexcept {
  if (!std::isfinite(value)) {
    return 0;
  }
  return static_cast<uint32_t>(std::clamp(value, 0.0, static_cast<double>(maximum)));
}

void set_present_viewport(const wgpu::RenderPassEncoder& pass, const gfx::Viewport& viewport, uint32_t surfaceWidth,
                          uint32_t surfaceHeight) noexcept {
  pass.SetViewport(viewport.left, viewport.top, viewport.width, viewport.height, viewport.znear, viewport.zfar);
  const auto scissorX = clamp_scissor_coord(std::floor(viewport.left), surfaceWidth);
  const auto scissorY = clamp_scissor_coord(std::floor(viewport.top), surfaceHeight);
  const auto scissorRight = clamp_scissor_coord(std::ceil(viewport.left + viewport.width), surfaceWidth);
  const auto scissorBottom = clamp_scissor_coord(std::ceil(viewport.top + viewport.height), surfaceHeight);
  pass.SetScissorRect(scissorX, scissorY, scissorRight - scissorX, scissorBottom - scissorY);
}
#endif

#ifdef AURORA_ENABLE_GX
constexpr std::array PreferredBackendOrder{
#ifdef ENABLE_BACKEND_WEBGPU
    BACKEND_WEBGPU,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D12
    BACKEND_D3D12,
#endif
#ifdef DAWN_ENABLE_BACKEND_METAL
    BACKEND_METAL,
#endif
#ifdef DAWN_ENABLE_BACKEND_VULKAN
    BACKEND_VULKAN,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D11
    BACKEND_D3D11,
#endif
// #ifdef DAWN_ENABLE_BACKEND_DESKTOP_GL
//     BACKEND_OPENGL,
// #endif
#ifdef DAWN_ENABLE_BACKEND_OPENGLES
    BACKEND_OPENGLES,
#endif
#ifdef DAWN_ENABLE_BACKEND_NULL
    BACKEND_NULL,
#endif
};
#else
constexpr std::array<AuroraBackend, 0> PreferredBackendOrder{};
#endif

bool g_initialFrame = false;

AuroraInfo initialize(int argc, char* argv[], const AuroraConfig& config) noexcept {
  g_config = config;
  Log.info("Aurora initializing");
  log_system_information();
  if (g_config.appName == nullptr) {
    g_config.appName = "Aurora";
  } else {
    g_config.appName = strdup(g_config.appName);
  }
  if (g_config.userPath == nullptr) {
    g_config.userPath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.userPath = strdup(g_config.userPath);
  }
  if (g_config.cachePath == nullptr) {
    g_config.cachePath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.cachePath = strdup(g_config.cachePath);
  }
  if (g_config.resourcesPath == nullptr) {
    g_config.resourcesPath = SDL_GetBasePath();
  } else {
    g_config.resourcesPath = strdup(g_config.resourcesPath);
  }
  if (g_config.msaa == 0) {
    g_config.msaa = 1;
  }
  if (g_config.maxTextureAnisotropy == 0) {
    g_config.maxTextureAnisotropy = 16;
  }
  AURORA_ASSERT(window::initialize(), "Error initializing window");

  g_sdlCustomEventsStart = SDL_RegisterEvents(2);
  AURORA_ASSERT(g_sdlCustomEventsStart, "Failed to allocate user events: {}", SDL_GetError());
  AURORA_ASSERT(window::initialize_event_watch(), "Error initializing SDL event watch");

#ifdef AURORA_ENABLE_GX
  /* Attempt to create a window using the calling application's desired backend */
  AuroraBackend selectedBackend = config.desiredBackend;
  bool windowCreated = false;
  if (selectedBackend != BACKEND_AUTO && window::create_window(selectedBackend)) {
    if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
      windowCreated = true;
    } else {
      window::destroy_window();
    }
  }

  if (!windowCreated) {
    for (const auto backendType : PreferredBackendOrder) {
      selectedBackend = backendType;
      if (!window::create_window(selectedBackend)) {
        continue;
      }
      if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
        windowCreated = true;
        break;
      } else {
        window::destroy_window();
      }
    }
  }

  AURORA_ASSERT(windowCreated, "Error creating window: {}", SDL_GetError());

  // Initialize SDL_Renderer for ImGui when we can't use a Dawn backend
  bool sdlRenderer = webgpu::g_backendType == wgpu::BackendType::Null;
#if defined(AURORA_GFX_DEKO3D)
  // aleks2 (deko3d P2): Dawn's Null device on purpose (the object model); lib/deko draws ImGui and owns
  // the window. The Switch's SDL has no render driver: P1 aborted here (after SwitchWaker patch 0013).
  sdlRenderer = sdlRenderer && !gfx::dk::active();
#endif
  if (sdlRenderer) {
    AURORA_ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
  }
#else
  AuroraBackend selectedBackend = BACKEND_NULL;
  AURORA_ASSERT(window::create_window(BACKEND_NULL), "Error creating window: {}", SDL_GetError());
  AURORA_ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
#endif

  window::show_window();
  thread::set_current({
      .name = "Main thread",
      .affinity = thread::Affinity::SharedCache,
  });

#ifdef AURORA_ENABLE_GX
  gfx::initialize();
  gx::fifo::init();
  imgui::create_context();
#endif
  const auto size = window::get_window_size();
  Log.info("Using framebuffer size {}x{} scale {}", size.fb_width, size.fb_height, size.scale);
#ifdef AURORA_ENABLE_GX
  if (g_config.imGuiInitCallback != nullptr) {
    g_config.imGuiInitCallback(&size);
  }
  imgui::initialize();
#endif

#ifdef AURORA_ENABLE_RMLUI
  rmlui::initialize(size);
#endif

  g_initialFrame = true;
  g_config.desiredBackend = selectedBackend;
  return {
      .backend = selectedBackend,
      .userPath = g_config.userPath,
      .cachePath = g_config.cachePath,
      .window = window::get_sdl_window(),
      .windowSize = size,
  };
}

void shutdown() noexcept {
#ifdef AURORA_ENABLE_GX
  gx::fifo::shutdown();
  gfx::render_worker::synchronize();
#ifdef AURORA_ENABLE_RMLUI
  rmlui::shutdown();
#endif
  imgui::shutdown();
  gfx::shutdown();
  webgpu::shutdown();
#endif
  input::shutdown();
  window::shutdown();
}

const AuroraEvent* update() noexcept {
  ZoneScoped;
  if (g_initialFrame) {
    g_initialFrame = false;
    input::initialize();
  }
#ifdef AURORA_ENABLE_GX
  gx::update();
#endif
  return window::poll_events();
}

bool begin_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  {
    if (!window::is_presentable()) {
      webgpu::release_surface();
      return false;
    }
    if (window::is_paused()) {
      return false;
    }
    bool needsSurface = true;
#if defined(AURORA_GFX_DEKO3D)
    needsSurface = !gfx::dk::active();  // (lib/deko presents to the default window)
#endif
    if (needsSurface && !g_surface) {
      webgpu::refresh_surface(true);
      if (!g_surface) {
        return false;
      }
    }
  }

  imgui::new_frame(window::get_window_size());
  if (!gfx::begin_frame()) {
    return false;
  }
  gx::fifo::begin_frame();
#endif
  return true;
}

void end_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  gx::fifo::drain(gx::fifo::DrainSite::EndFrame);
  gx::fifo::end_frame();
  gx::texture::end_frame();
  gfx::finish();
  auto imguiDrawData = imgui::freeze();

  const auto& presentSource = webgpu::present_source();
  const auto viewport = webgpu::calculate_present_viewport(webgpu::g_graphicsConfig.surfaceConfiguration.width,
                                                           webgpu::g_graphicsConfig.surfaceConfiguration.height,
                                                           presentSource.size.width, presentSource.size.height);

  wgpu::BindGroup rmlBindGroup;
  bool rmlOverlay = false;
#if AURORA_ENABLE_RMLUI
  if (rmlui::is_initialized()) {
    auto rmlFrame = rmlui::record_frame(viewport);
    rmlBindGroup = std::move(rmlFrame.bindGroup);
    rmlOverlay = rmlFrame.overlay;
  }
#endif

  // aleks2 (V1): the input -> present probe: this frame's first controller read
  const int64_t latencyInputNs = gfx::latency_take_input();
  gfx::end_frame([rmlBindGroup = std::move(rmlBindGroup), rmlOverlay, viewport, latencyInputNs,
                  imguiDrawData = std::move(imguiDrawData)](
                     wgpu::CommandEncoder& encoder, std::vector<gfx::AfterSubmitCallback> afterSubmitCallbacks) {
#if defined(AURORA_GFX_DEKO3D)
    if (gfx::dk::active()) {
      // aleks2 (deko3d P1): the frame was recorded into deko3d as its passes were sealed (lib/deko);
      // lib/deko presents: the EFB (or RmlUi's composite of it) resampled and fitted, RmlUi's overlay,
      // ImGui, the submit and the present. Single screen, normal layout (no aux window, no Flip Grip).
      (void)encoder;
      (void)viewport;
      gfx::dk::set_present_layers(rmlBindGroup.Get(), rmlOverlay);
      imgui::present_deko3d(imguiDrawData);
      gfx::dk::set_present_layers(nullptr, false);
      gfx::after_present();
      gfx::latency_note_presented(latencyInputNs);  // aleks2 (V1): input -> present probe
      gfx::dk::boot_presented();
      for (auto& callback : afterSubmitCallbacks) {
        if (callback) {
          callback();
        }
      }
      gfx::after_submit();
      return;
    }
#endif
    wgpu::Texture currentTexture;
    wgpu::TextureView currentView;
    auto surfaceStatus = wgpu::SurfaceGetCurrentTextureStatus::Error;
    {
      window::SurfaceLock surfaceLock;
      if (window::is_presentable() && g_surface) {
        ZoneScopedN("Acquire texture");
        wgpu::SurfaceTexture surfaceTexture;
        g_surface.GetCurrentTexture(&surfaceTexture);
        surfaceStatus = surfaceTexture.status;
        if (surfaceStatus == wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal) {
          currentTexture = std::move(surfaceTexture.texture);
          currentView = currentTexture.CreateView();
        }
      }
    }

    const bool canPresent = currentTexture && currentView;
    if (canPresent) {
      wgpu::BindGroup presentBindGroup;
      if (rmlBindGroup && !rmlOverlay) {
        presentBindGroup = rmlBindGroup;
      } else {
        const auto& resampledSource = webgpu::resample_present_source(encoder, viewport);
        presentBindGroup = webgpu::create_copy_bind_group(resampledSource);
      }
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = currentView,
                .loadOp = wgpu::LoadOp::Clear,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "EFB copy render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
            .timestampWrites = webgpu::gpu_prof::pass_writes("Present blit"),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        // Copy EFB -> XFB (swapchain)
        //
        // ORDER MATTERS: pipeline, THEN bind group, then viewport. Setting a
        // bind group before the pipeline leaves it unbound, because the
        // pipeline change invalidates bind groups whose layout does not match.
        // Getting this backwards while adding the Flip branch below is what
        // made the very first Switch build abort in the WSI present with
        // LibnxError_BadGfxDequeueBuffer -- with the companion switched OFF,
        // which is what made it look like a driver problem rather than a
        // two-line state-order bug.
#ifdef __SWITCH__
        // Flip Grip decides ONCE, here, and Normal keeps the original calls
        // below untouched. The mode that has to keep working must not sit
        // downstream of the layout code -- same split as the ALEKS ports'
        // port_display_layout.c.
        namespace lay = auxwin::layout;
        bool flipped = false;
        lay::Rect flipRect{};
        if (lay::mode() == lay::kFlip270 && webgpu::g_CopyRot270Pipeline != nullptr) {
          const int outW = static_cast<int>(webgpu::g_graphicsConfig.surfaceConfiguration.width);
          const int outH = static_cast<int>(webgpu::g_graphicsConfig.surfaceConfiguration.height);
          lay::Layout l{};
          const float aspect = viewport.height > 0.f ? viewport.width / viewport.height : 4.f / 3.f;
          lay::compute(outW, outH, aspect, &l);
          if (l.mode == lay::kFlip270 && l.game.valid()) {
            flipRect = lay::to_physical(l.game, outW, outH, l.rotationDegrees);
            flipped = flipRect.valid();
          }
        }
        pass.SetPipeline(flipped ? webgpu::g_CopyRot270Pipeline : webgpu::g_CopyPipeline);
#else
        pass.SetPipeline(webgpu::g_CopyPipeline);
#endif
        pass.SetBindGroup(0, presentBindGroup, 0, nullptr);
#ifdef __SWITCH__
        if (flipped) {
          pass.SetViewport(static_cast<float>(flipRect.x), static_cast<float>(flipRect.y),
                           static_cast<float>(flipRect.w), static_cast<float>(flipRect.h), 0.f, 1.f);
        } else
#endif
        {
          set_present_viewport(pass, viewport, webgpu::g_graphicsConfig.surfaceConfiguration.width,
                               webgpu::g_graphicsConfig.surfaceConfiguration.height);
        }

        pass.Draw(3);
        if (rmlBindGroup && rmlOverlay) {
          // The overlay has to take the same turn as the game underneath it, or
          // its text reads sideways while everything around it is upright.
#ifdef __SWITCH__
          pass.SetPipeline(flipped ? webgpu::g_CopyPremultipliedAlphaRot270Pipeline
                                   : webgpu::g_CopyPremultipliedAlphaPipeline);
#else
          pass.SetPipeline(webgpu::g_CopyPremultipliedAlphaPipeline);
#endif
          pass.SetBindGroup(0, rmlBindGroup, 0, nullptr);
          pass.Draw(3);
        }
        pass.End();
      }
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = currentView,
                .loadOp = wgpu::LoadOp::Load,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "ImGui render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
            .timestampWrites = webgpu::gpu_prof::pass_writes("ImGui"),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        pass.SetViewport(0.f, 0.f, static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.width),
                         static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.height), 0.f, 1.f);
        imgui::render(pass, imguiDrawData);
        pass.End();
      }
    } else {
      Log.info("Skipping present; window not presentable");
    }
    // The main frame's submit must never depend on the aux window.
    // See docs/dualscreen-fork.md (Verification) for why: an earlier revision
    // moved Finish/Submit inside the aux present, behind an early return taken
    // whenever no aux surface existed, and every single-screen device drew
    // nothing while running perfectly.
    webgpu::gpu_prof::frame_end(encoder);
    webgpu::gpu_timing::frame_end(encoder);
    const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Redraw command buffer"};
    const auto buffer = encoder.Finish(&cmdBufDescriptor);
    {
      ZoneScopedN("Queue Submit");
      g_queue.Submit(1, &buffer);
    }
    webgpu::gpu_prof::after_submit();
    webgpu::gpu_timing::after_submit();
    auxwin::present(currentView);
    if (canPresent && g_surface) {
      ZoneScopedN("Present");
      wgpu::ConvertibleStatus status = wgpu::Status::Error;
      {
        window::SurfaceLock surfaceLock;
        if (window::is_presentable()) {
          status = g_surface.Present();
        }
      }
      if (status) {
        gfx::after_present();
      gfx::latency_note_presented(latencyInputNs);  // aleks2 (V1): input -> present probe
      } else {
        Log.warn("Surface present failed");
        webgpu::release_surface();
      }
    } else if (g_surface) {
      switch (surfaceStatus) {
      case wgpu::SurfaceGetCurrentTextureStatus::Timeout:
        Log.warn("Surface texture acquisition timed out");
        break;
      case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
      case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
        Log.info("Surface texture is {}, reconfiguring swapchain", magic_enum::enum_name(surfaceStatus));
        window::push_custom_event(window::CustomEvent::RefreshSurface);
        break;
      case wgpu::SurfaceGetCurrentTextureStatus::Lost:
        Log.warn("Surface texture is {}, releasing surface", magic_enum::enum_name(surfaceStatus));
        webgpu::release_surface();
        break;
      case wgpu::SurfaceGetCurrentTextureStatus::Error:
        Log.warn("Surface texture is {}, dropping surface", magic_enum::enum_name(surfaceStatus));
        g_surface = {};
        break;
      default:
        if (!window::is_presentable()) {
          webgpu::release_surface();
        } else {
          Log.error("Failed to get surface texture: {}", magic_enum::enum_name(surfaceStatus));
        }
        break;
      }
    }
    for (auto& callback : afterSubmitCallbacks) {
      if (callback) {
        callback();
      }
    }
    gfx::after_submit();

    TracyPlotConfig("aurora: lastVertSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastUniformSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastIndexSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastStorageSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastTextureUploadSize", tracy::PlotFormatType::Memory, false, true, 0);

    const auto& stats = gfx::detail::resources().stats;
    TracyPlot("aurora: queuedPipelines", static_cast<int64_t>(stats.queuedPipelines));
    TracyPlot("aurora: createdPipelines", static_cast<int64_t>(stats.createdPipelines));
    TracyPlot("aurora: drawCallCount", static_cast<int64_t>(stats.drawCallCount));
    TracyPlot("aurora: mergedDrawCallCount", static_cast<int64_t>(stats.mergedDrawCallCount));
    TracyPlot("aurora: lastVertSize", static_cast<int64_t>(stats.lastVertSize));
    TracyPlot("aurora: lastUniformSize", static_cast<int64_t>(stats.lastUniformSize));
    TracyPlot("aurora: lastIndexSize", static_cast<int64_t>(stats.lastIndexSize));
    TracyPlot("aurora: lastStorageSize", static_cast<int64_t>(stats.lastStorageSize));
    TracyPlot("aurora: lastTextureUploadSize", static_cast<int64_t>(stats.lastTextureUploadSize));
  });

#endif
}
} // namespace
} // namespace aurora

// C API bindings
AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config) {
  return aurora::initialize(argc, argv, *config);
}
void aurora_shutdown() { aurora::shutdown(); }
const AuroraEvent* aurora_update() { return aurora::update(); }
bool aurora_begin_frame() { return aurora::begin_frame(); }
void aurora_end_frame() { aurora::end_frame(); }
AuroraBackend aurora_get_backend() { return aurora::g_config.desiredBackend; }
const AuroraBackend* aurora_get_available_backends(size_t* count) {
#if defined(AURORA_GFX_DEKO3D)
  // aleks2 (deko3d P1): deko3d is offered (Settings, "Experimental") but is never in the fallback order
  static const auto withDeko3d = [] {
    std::array<AuroraBackend, aurora::PreferredBackendOrder.size() + 1> list{};
    std::copy(aurora::PreferredBackendOrder.begin(), aurora::PreferredBackendOrder.end(), list.begin());
    list.back() = BACKEND_DEKO3D;
    return list;
  }();
  if (count != nullptr) {
    *count = withDeko3d.size();
  }
  return withDeko3d.data();
#else
  if (count != nullptr) {
    *count = aurora::PreferredBackendOrder.size();
  }
  return aurora::PreferredBackendOrder.data();
#endif
}
// aleks2 (deko3d P1): true when deko3d was asked for but this start fell back to Dawn/NVK (two starts in
// a row that never presented a frame); the game resets its setting and tells the player
bool aurora_deko3d_fell_back() {
#if defined(AURORA_GFX_DEKO3D)
  return aurora::gfx::dk::fell_back();
#else
  return false;
#endif
}
void aurora_set_log_level(AuroraLogLevel level) { aurora::g_config.logLevel = level; }
void aurora_set_pause_on_focus_lost(bool value) { aurora::g_config.pauseOnFocusLost = value; }
void aurora_set_background_input(bool value) {
  aurora::g_config.allowJoystickBackgroundEvents = value;
  aurora::window::set_background_input(value);
}
void aurora_set_resampler(AuroraSampler sampler) {
#ifdef AURORA_ENABLE_GX
  aurora::webgpu::set_resampler(sampler);
#else
  (void)sampler;
#endif
}
void aurora_set_timescale(float scale) { aurora::time::set_scale(scale); }
float aurora_get_timescale() { return aurora::time::scale(); }
