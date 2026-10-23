#pragma once

#include <aurora/aurora.h>
#include <aurora/math.hpp>

#include "wgpu.hpp"

#include <array>
#include <cstdint>

struct SDL_Window;

namespace aurora::webgpu {
inline constexpr wgpu::TextureFormat NormalBufferFormat = wgpu::TextureFormat::RGB10A2Unorm;

struct GraphicsConfig {
  wgpu::SurfaceConfiguration surfaceConfiguration;
  wgpu::TextureFormat depthFormat;
  uint32_t msaaSamples;
  uint16_t textureAnisotropy;
  bool normalBuffer = false;
};
struct TextureWithSampler {
  wgpu::Texture texture;
  wgpu::TextureView view;
  wgpu::Extent3D size;
  wgpu::TextureFormat format;
  wgpu::Sampler sampler;
};
struct Viewport {
  float left;
  float top;
  float width;
  float height;
  float znear;
  float zfar;

  bool operator==(const Viewport& rhs) const {
    return left == rhs.left && top == rhs.top && width == rhs.width && height == rhs.height && znear == rhs.znear &&
           zfar == rhs.zfar;
  }
  bool operator!=(const Viewport& rhs) const { return !(*this == rhs); }
};

extern wgpu::Device g_device;
extern wgpu::Queue g_queue;
extern wgpu::Surface g_surface;
extern wgpu::BackendType g_backendType;
extern GraphicsConfig g_graphicsConfig;
extern TextureWithSampler g_frameBuffer;
extern TextureWithSampler g_frameBufferResolved;
extern TextureWithSampler g_depthBuffer;
extern TextureWithSampler g_normalBuffer;
extern wgpu::RenderPipeline g_CopyPipeline;
extern wgpu::RenderPipeline g_CopyPremultipliedAlphaPipeline;
extern wgpu::RenderPipeline g_CopyPremultipliedAlphaRot270Pipeline;
/// g_CopyPipeline modulated by the render pass's blend constant (out = src *
/// constant). Used by the aux window to apply its dim at present time.
extern wgpu::RenderPipeline g_CopyDimPipeline;
extern wgpu::RenderPipeline g_CopyRot270Pipeline;
extern wgpu::RenderPipeline g_CopyDimRot270Pipeline;
extern wgpu::BindGroup g_CopyBindGroup;
extern wgpu::Instance g_instance;
extern wgpu::AdapterInfo g_adapterInfo;
extern bool g_hasCoreFeatures;
extern bool g_bcTexturesSupported;
extern bool g_astcTexturesSupported;
extern bool g_textureComponentSwizzleSupported;
extern bool g_dualSourceBlendingSupported;

bool initialize(AuroraBackend backend, bool allowCpu);
void shutdown();
void release_surface() noexcept;
bool refresh_surface(bool recreate = true);
void resize_swapchain(uint32_t width, uint32_t height, uint32_t nativeWidth, uint32_t nativeHeight, bool force = false);
TextureWithSampler create_render_texture(uint32_t width, uint32_t height, bool multisampled);
bool enable_normal_buffer();
const TextureWithSampler& present_source() noexcept;
wgpu::BindGroup create_copy_bind_group(const TextureWithSampler& source);
void set_resampler(AuroraSampler sampler) noexcept;
AuroraSampler get_resampler() noexcept;
Viewport calculate_present_viewport(uint32_t surface_width, uint32_t surface_height, uint32_t content_width,
                                    uint32_t content_height) noexcept;
const TextureWithSampler& resample_present_source(const wgpu::CommandEncoder& encoder, const Viewport& viewport);
void draw_clear(const wgpu::RenderPassEncoder& pass, bool clearColor, bool clearAlpha, bool clearDepth,
                const Vec4<float>& clearColorValue, float clearDepthValue);

size_t load_from_cache(void const* key, size_t keySize, void* value, size_t valueSize, void* userdata);
void store_to_cache(void const* key, size_t keySize, void const* value, size_t valueSize, void* userdata);
void cache_prune();
void cache_shutdown();
bool cache_copy_running();  // V1-rc3: the bundled shader cache is still being copied (gpu_cache.cpp)

// aleks2 (SHADERS-c1): the Dawn cache's health and counters for this launch.
struct CacheCounters {
  uint32_t hits = 0;
  uint32_t misses = 0;
  uint32_t stores = 0;
  double storeMsTotal = 0.0;
  double storeMsMax = 0.0;
  bool opened = false;
  bool dirty = false;
  bool repaired = false;
  double checkMs = 0.0;
  std::string journal;
};
CacheCounters cache_counters();
// "<device>|<description>": what a Dawn seed is keyed on.
std::string driver_identity();

} // namespace aurora::webgpu
