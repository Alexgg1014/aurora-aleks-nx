// Adapted from SwitchWaker (MIT), 084291f: switch/deko/aurora/dk_present.cpp. Changes for Dusklight
// (aleks2 deko3d P1): RmlUi's layer (set_present_layers): its composite in place of the game's picture,
// or over it with premultiplied alpha, as aurora.cpp's present pass does for Dawn. (aleks2 deko3d rc12): Flip Grip
// (the picture turned 270 degrees into the layout's game rect, aurora.cpp) and the companion screen (the dashboard
// copied into an image of ours when it is redrawn and blitted into its rect after ImGui, dimmed with the blend
// constant: aux_window_switch.cpp's present(), whose decisions come from auxwin::deko_companion).
//
// The present and the readback of the deko3d NRO's Aurora side (dk_gfx.hpp; docs/DEKO3D_MIGRATION_PLAN.md
// phase 3).
//
// Present (switch/deko frame_present's picture, Aurora's aurora.cpp end_frame for WebGPU): the EFB
// (webgpu::present_source) is resampled to the size it is shown at, area or bilinear as the options
// menu says (webgpu::get_resampler; the present_resample fixed shader), then drawn into the swapchain
// image in the viewport webgpu::calculate_present_viewport fits it to (the aspect fit of patch 0006),
// with the xfb_copy shader; ImGui comes after it.
//
// Readback (read_texture): a copy-engine copy into CPU-visible memory on a command buffer of its own,
// after what the frame has recorded so far, and a wait for its fence (~1 ms for the picto box's
// shutter area; COS_SHOT's frames).
#include "dk_encode.hpp"
#include "dk_gfx.hpp"

#include "dk_aurora.h"

#include "aux_window.hpp"
#include "gfx/frame.hpp"
#include "webgpu/gpu.hpp"

#include <aurora/aux_layout_switch.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace aurora::gfx::dk {
using swdk::dklog;
using swdk::R;

namespace {

// the resampled picture: an image of our own, at the size the picture is shown
struct Resampled {
    bool valid = false;
    uint32_t width = 0, height = 0;
    DkImage image{};
    swdk::ImageAlloc mem;
    uint32_t slot = 0;
} g_resampled;

struct ResampleUniforms {  // present_resample.wgsl's Uniforms (std140)
    uint32_t samplerMode;
    float frameWidth;
    float frameHeight;
    uint32_t pad;
};

Viewport g_viewport{};

// the companion screen: our own copy of the dashboard (auxwin's owned texture) and this frame's blit
struct Companion {
    bool valid = false;
    uint32_t width = 0, height = 0;
    DkImage image{};
    swdk::ImageAlloc mem;
    uint32_t slot = 0;
    auxwin::DekoCompanion draw{};  // w == 0: nothing to blit this frame
} g_companion;

bool ensure_companion(uint32_t w, uint32_t h) {
    if (g_companion.valid && g_companion.width == w && g_companion.height == h) return true;
    if (g_companion.valid) swdk::image_free_later(g_companion.mem);
    g_companion.valid = false;
    DkImageLayoutMaker m;
    dkImageLayoutMakerDefaults(&m, R.device);
    m.format = DkImageFormat_RGBA8_Unorm;
    m.flags = DkImageFlags_UsageRender | DkImageFlags_Usage2DEngine;
    m.dimensions[0] = w;
    m.dimensions[1] = h;
    swdk::image_tile_size_fix(m, h);
    DkImageLayout layout;
    dkImageLayoutInitialize(&layout, &m);
    g_companion.mem = swdk::image_alloc(uint32_t(dkImageLayoutGetSize(&layout)), dkImageLayoutGetAlignment(&layout));
    dkImageInitialize(&g_companion.image, &layout, g_companion.mem.block, g_companion.mem.offset);
    if (g_companion.slot == 0) g_companion.slot = swdk::image_slot_alloc();
    if (g_companion.slot == 0) return false;
    swdk::write_image_descriptor(g_companion.slot, g_companion.image);
    g_companion.valid = true;
    g_companion.width = w;
    g_companion.height = h;
    dklog("present: companion screen image %ux%u", w, h);
    return true;
}

// stage 0 (before the swapchain is the target): copy a freshly drawn dashboard, decide this frame's blit
void companion_prepare() {
    g_companion.draw = {};
    auxwin::DekoCompanion c;
    if (!auxwin::deko_companion(int(R.width), int(R.height), g_companion.valid ? g_companion.width : 0,
                                g_companion.valid ? g_companion.height : 0, &c)) {
        return;
    }
    if (c.newSource) {
        View* v = find_view(c.newSource);
        if (v && v->tex && v->tex->valid && ensure_companion(c.width, c.height)) {
            DkImageView target;
            dkImageViewDefaults(&target, &g_companion.image);
            const FixedBind binds[] = {
                {swdk::SlotBinding::Kind::Sampler, 1, dkMakeTextureHandle(view_slot(v), swdk::kSamplerLinearClamp)},
            };
            fixed_draw("xfb_copy", "vs_main", "fs_opaque", &target, c.width, c.height, true, binds, 1);
        }
    }
    if (c.w > 0 && c.h > 0 && g_companion.valid) g_companion.draw = c;
}

// viewport, scissor and swizzle of a present blit (set_present_viewport in aurora.cpp)
void present_rect(float x, float y, float w, float h) {
    const DkViewport vp = {x, y, w, h, 0.f, 1.f};
    dkCmdBufSetViewports(R.cmd, 0, &vp, 1);
    const auto clampc = [](float v, uint32_t max) {
        return uint32_t(std::clamp(v, 0.f, float(max)));
    };
    const uint32_t x0 = clampc(std::floor(x), R.width), y0 = clampc(std::floor(y), R.height);
    const uint32_t x1 = clampc(std::ceil(x + w), R.width), y1 = clampc(std::ceil(y + h), R.height);
    const DkScissor sc = {x0, y0, x1 - x0, y1 - y0};
    dkCmdBufSetScissors(R.cmd, 0, &sc, 1);
    const DkViewportSwizzle sw = {DkSwizzle_PositiveX, swdk::C.flipPresent ? DkSwizzle_NegativeY : DkSwizzle_PositiveY,
                                  DkSwizzle_PositiveZ, DkSwizzle_PositiveW};
    dkCmdBufSetViewportSwizzles(R.cmd, 0, &sw, 1);
}

// stage 2 (after ImGui): the companion into its rect; turned with the picture in Flip Grip; dimmed with the constant
void companion_blit() {
    const auxwin::DekoCompanion& c = g_companion.draw;
    if (c.w <= 0 || c.h <= 0 || !g_companion.valid) return;
    present_rect(float(c.x), float(c.y), float(c.w), float(c.h));
    const FixedBind binds[] = {
        {swdk::SlotBinding::Kind::Sampler, 1, dkMakeTextureHandle(g_companion.slot, swdk::kSamplerLinearClamp)},
    };
    const char* vs = c.rotated ? "vs_rot270" : "vs_main";
    if (c.dim > 0.001f) {
        const float keep = 1.f - std::clamp(c.dim, 0.f, 1.f);
        DkBlendState modulate;
        dkBlendStateDefaults(&modulate);
        dkBlendStateSetOps(&modulate, DkBlendOp_Add, DkBlendOp_Add);
        dkBlendStateSetFactors(&modulate, DkBlendFactor_ConstColor, DkBlendFactor_Zero, DkBlendFactor_One,
                               DkBlendFactor_Zero);
        dkCmdBufSetBlendConst(R.cmd, keep, keep, keep, 1.f);
        fixed_draw("xfb_copy", vs, "fs_opaque", nullptr, 0, 0, false, binds, 1, &modulate);
    } else {
        fixed_draw("xfb_copy", vs, "fs_opaque", nullptr, 0, 0, false, binds, 1);
    }
}
WGPUBindGroup g_rmlGroup = nullptr;  // this present's RmlUi layer (render worker)
bool g_rmlOverlay = false;
View* g_rmlView = nullptr;           // its texture, for the frame being presented
Samp* g_rmlSampler = nullptr;

// RmlUi's composite: the copy bind group (binding 0 sampler, binding 1 the view, webgpu::create_copy_bind_group)
bool rml_layer() {
    g_rmlView = nullptr;
    g_rmlSampler = nullptr;
    BindGroup* g = g_rmlGroup ? find_bind_group(g_rmlGroup) : nullptr;
    if (!g) return false;
    for (const auto& e : g->entries) {
        if (e.view) g_rmlView = e.view;
        if (e.sampler) g_rmlSampler = e.sampler;
    }
    return g_rmlView && g_rmlView->tex && g_rmlView->tex->valid;
}

uint32_t extent(float v) { return std::max(1u, static_cast<uint32_t>(std::lround(std::max(v, 1.f)))); }

bool ensure_resampled(uint32_t w, uint32_t h) {
    if (g_resampled.valid && g_resampled.width == w && g_resampled.height == h) return true;
    if (g_resampled.valid) swdk::image_free_later(g_resampled.mem);
    DkImageLayoutMaker m;
    dkImageLayoutMakerDefaults(&m, R.device);
    m.format = DkImageFormat_RGBA8_Unorm;
    m.flags = DkImageFlags_UsageRender | DkImageFlags_Usage2DEngine;
    m.dimensions[0] = w;
    m.dimensions[1] = h;
    swdk::image_tile_size_fix(m, h);
    DkImageLayout layout;
    dkImageLayoutInitialize(&layout, &m);
    g_resampled.mem = swdk::image_alloc(uint32_t(dkImageLayoutGetSize(&layout)), dkImageLayoutGetAlignment(&layout));
    dkImageInitialize(&g_resampled.image, &layout, g_resampled.mem.block, g_resampled.mem.offset);
    if (g_resampled.slot == 0) g_resampled.slot = swdk::image_slot_alloc();
    if (g_resampled.slot == 0) return false;
    swdk::write_image_descriptor(g_resampled.slot, g_resampled.image);
    g_resampled.valid = true;
    g_resampled.width = w;
    g_resampled.height = h;
    dklog("present: the picture resampled to %ux%u", w, h);
    return true;
}

void picture(void*, int stage) {
    const auto& source = webgpu::present_source();
    Tex* src = find_texture(source.texture.Get());
    if (stage == 0) {
        companion_prepare();  // first: the steps below may return early
        g_viewport = webgpu::calculate_present_viewport(R.width, R.height, source.size.width, source.size.height);
        // RmlUi composited the game itself: its image is the picture, no resample
        if (rml_layer() && !g_rmlOverlay) return;
        if (!src || !src->valid || g_viewport.width <= 0.f || g_viewport.height <= 0.f) return;
        const uint32_t w = extent(g_viewport.width), h = extent(g_viewport.height);
        if (!ensure_resampled(w, h)) return;
        const swdk::StreamAlloc u = swdk::stream_alloc(256, DK_UNIFORM_BUF_ALIGNMENT);
        if (!u) return;
        const ResampleUniforms uniforms = {webgpu::get_resampler() == SAMPLER_AREA ? 1u : 0u, float(w), float(h), 0};
        memcpy(u.cpu, &uniforms, sizeof uniforms);
        DkImageView srcView;
        dkImageViewDefaults(&srcView, &src->image);
        // a descriptor of its own for the source, written again when the EFB image changes (a resize)
        static uint32_t srcSlot = 0;
        static DkGpuAddr srcAddr = 0;
        const DkGpuAddr addr = dkImageGetGpuAddr(&src->image);
        if (srcAddr != addr || srcSlot == 0) {
            if (srcSlot == 0) srcSlot = swdk::image_slot_alloc();
            if (srcSlot == 0) return;
            swdk::write_image_view_descriptor(srcSlot, srcView);
            srcAddr = addr;
        }
        const DkResHandle h0 = dkMakeTextureHandle(srcSlot, swdk::kSamplerLinearClamp);
        // present_resample: uniforms 0, the texture with its sampler and again for textureLoad
        const FixedBind binds[] = {
            {swdk::SlotBinding::Kind::Uniform, 0, 0, u.gpu, 256},
            {swdk::SlotBinding::Kind::Sampler, 2, h0},
        };
        DkImageView target;
        dkImageViewDefaults(&target, &g_resampled.image);
        fixed_draw("present_resample", "vs_main", "fs_main", &target, w, h, true, binds, 2);
        // aleks2 (V1, COS_DK_SKIP_PRESENT_CLEAR): stage 1 will blit the picture opaque over the whole image (no Flip
        // Grip, no letterbox); the colour clear before it is then redundant. Idea from ChanseyIsTheBest/renut_nx (c319b29, docs/deko3d-renderer.md and render-thread-proposals), used with his permission (2026-10-19).
        swdk::g_pictureCoversTarget = g_resampled.valid && auxwin::layout::mode() != auxwin::layout::kFlip270 &&
                                      g_viewport.left <= 0.f && g_viewport.top <= 0.f &&
                                      g_viewport.left + g_viewport.width >= float(R.width) &&
                                      g_viewport.top + g_viewport.height >= float(R.height);
        return;
    }
    if (stage == 1) {
        const bool rmlPicture = g_rmlView && !g_rmlOverlay;
        if ((!rmlPicture && !g_resampled.valid) || g_viewport.width <= 0.f) return;
        // Flip Grip (aurora.cpp): the layout's game rect, the picture turned 270 degrees; otherwise
        // set_present_viewport's viewport, and the scissor around it
        namespace lay = auxwin::layout;
        bool flipped = false;
        lay::Rect flipRect{};
        if (lay::mode() == lay::kFlip270) {
            lay::Layout l{};
            const float aspect = g_viewport.height > 0.f ? g_viewport.width / g_viewport.height : 4.f / 3.f;
            lay::compute(int(R.width), int(R.height), aspect, &l);
            if (l.mode == lay::kFlip270 && l.game.valid()) {
                flipRect = lay::to_physical(l.game, int(R.width), int(R.height), l.rotationDegrees);
                flipped = flipRect.valid();
            }
        }
        if (flipped) {
            present_rect(float(flipRect.x), float(flipRect.y), float(flipRect.w), float(flipRect.h));
        } else {
            present_rect(g_viewport.left, g_viewport.top, g_viewport.width, g_viewport.height);
        }
        const char* vs = flipped ? "vs_rot270" : "vs_main";
        const DkResHandle base = rmlPicture ? texture_handle(g_rmlView, g_rmlSampler)
                                            : dkMakeTextureHandle(g_resampled.slot, swdk::kSamplerLinearClamp);
        const FixedBind binds[] = {
            {swdk::SlotBinding::Kind::Sampler, 1, base},
        };
        fixed_draw("xfb_copy", vs, "fs_opaque", nullptr, 0, 0, false, binds, 1);
        if (g_rmlView && g_rmlOverlay) {
            // RmlUi over the game (aurora.cpp: g_CopyPremultipliedAlphaPipeline)
            DkBlendState premultiplied;
            dkBlendStateDefaults(&premultiplied);
            dkBlendStateSetOps(&premultiplied, DkBlendOp_Add, DkBlendOp_Add);
            dkBlendStateSetFactors(&premultiplied, DkBlendFactor_One, DkBlendFactor_InvSrcAlpha, DkBlendFactor_One,
                                   DkBlendFactor_InvSrcAlpha);
            const FixedBind overlay[] = {
                {swdk::SlotBinding::Kind::Sampler, 1, texture_handle(g_rmlView, g_rmlSampler)},
            };
            // the overlay takes the same turn as the game underneath it (aurora.cpp)
            fixed_draw("xfb_copy", vs, "fs_premultiplied_alpha", nullptr, 0, 0, false, overlay, 1, &premultiplied);
        }
        gpu_mark(GpuWork::Present);
        return;
    }
    gpu_mark(GpuWork::Imgui);  // stage 2: after ImGui
    companion_blit();          // aux_window_switch.cpp's present(): after the main frame, on top of it
    gpu_mark(GpuWork::Present);
}

// ---- readback
DkCmdBuf g_rbCmd = nullptr;
DkMemBlock g_rbCmdMem = nullptr;
constexpr uint32_t kRbCmdSize = 64u << 10;

}  // namespace

void present_init() { swdk::set_picture(picture, nullptr); }

void set_present_layers(WGPUBindGroup rml, bool overlay) {
    g_rmlGroup = rml;
    g_rmlOverlay = overlay;
}

bool read_texture(WGPUTexture texture, std::vector<uint8_t>& pixels, uint32_t& width, uint32_t& height,
                  wgpu::TextureFormat& format, uint32_t mip) {
    Tex* t = find_texture(texture);
    if (!t || !t->valid || t->compressed || t->blockBytes != 4) {
        dklog("readback: texture %p %s", static_cast<void*>(texture),
              !t ? "has no deko3d image" : "is not 4 bytes per texel");
        return false;
    }
    if (mip >= t->mips) return false;
    width = std::max(t->width >> mip, 1u);
    height = std::max(t->height >> mip, 1u);
    format = static_cast<wgpu::TextureFormat>(t->wformat);
    const uint32_t rowBytes = width * 4;
    swdk::Block dst = swdk::block_create(rowBytes * height, "a readback");
    if (!g_rbCmd) {
        DkCmdBufMaker cm;
        dkCmdBufMakerDefaults(&cm, R.device);
        g_rbCmd = dkCmdBufCreate(&cm);
        DkMemBlockMaker mm;
        dkMemBlockMakerDefaults(&mm, R.device, kRbCmdSize);
        g_rbCmdMem = dkMemBlockCreate(&mm);
    }
    // what the frame recorded so far goes first (the copy reads its results)
    if (R.frameOpen) swdk::submit("readback");
    dkCmdBufClear(g_rbCmd);
    dkCmdBufAddMemory(g_rbCmd, g_rbCmdMem, 0, kRbCmdSize);
    dkCmdBufBarrier(g_rbCmd, DkBarrier_Full, DkInvalidateFlags_Image | DkInvalidateFlags_L2Cache);
    const DkImageView v = level_view(t, mip);
    const DkImageRect rect = {0, 0, 0, width, height, 1};
    const DkCopyBuf out = {dst.gpu, rowBytes, rowBytes * height};
    dkCmdBufCopyImageToBuffer(g_rbCmd, &v, &rect, &out, 0);
    dkCmdBufBarrier(g_rbCmd, DkBarrier_Full, DkInvalidateFlags_L2Cache);
    DkFence fence;
    dkCmdBufSignalFence(g_rbCmd, &fence, true);
    const DkCmdList list = dkCmdBufFinishList(g_rbCmd);
    swdk::check_queue("a readback");
    {
        std::lock_guard<std::mutex> lock(swdk::fence_mutex());
        dkQueueSubmitCommands(R.queue, list);
        dkQueueFlush(R.queue);
    }
    const DkResult r = dkFenceWait(&fence, 5'000'000'000ll);
    if (r != DkResult_Success) {
        dklog("readback: the copy of %ux%u did not complete (result %d)", width, height, int(r));
        swdk::block_free_deferred(dst.block);
        return false;
    }
    pixels.assign(dst.cpu, dst.cpu + size_t(rowBytes) * height);
    dkMemBlockDestroy(dst.block);
    return true;
}

}  // namespace aurora::gfx::dk

using namespace aurora::gfx::dk;

// The perf lines' figures (cos_switch_stats.cpp, COS_SWITCH_DEKO3D): running totals in the order of
// dk_aurora.h's AuroraSwitchDkStat
extern "C" void aurora_switch_dk_gfx_stats(uint64_t* out, size_t count) {
    const EncodeStats& s = encode_stats();
    const swdk::ShaderTotals sh = swdk::shader_totals();
    const swdk::SubmitStats sub = swdk::submit_stats();
    const ObjectStats o = object_stats();
    const uint64_t v[AURORA_SWITCH_DK_STAT_COUNT] = {
        s.passes,
        s.draws,
        s.clears,
        s.skippedPipeline,
        s.skippedShader + s.skippedOther,
        s.pipelineBinds,
        s.shaderBinds,
        s.textureBinds,
        s.uniformBinds,
        s.uploads,
        s.uploadBytes,
        s.copies,
        s.conversions,
        s.barriers,
        sub.submits,
        sub.frames,
        s.encodeNs,
        s.gpuFrames,
        s.gpuTotalNs,
        s.gpuNs[size_t(GpuWork::Efb)],
        s.gpuNs[size_t(GpuWork::Conversion)],
        s.gpuNs[size_t(GpuWork::Copy)],
        s.gpuNs[size_t(GpuWork::Present)],
        s.gpuNs[size_t(GpuWork::Imgui)],
        s.gpuNs[size_t(GpuWork::Other)] + s.gpuNs[size_t(GpuWork::FrameStart)],
        s.gpuDropped,
        sh.loaded,
        sh.pending,
        sh.failed,
        sh.compiled,
        sh.skippedDraws,
        o.textures,
        o.imageBytes,
    };
    for (size_t i = 0; i < count && i < AURORA_SWITCH_DK_STAT_COUNT; i++) out[i] = v[i];
}

// pc_capture.cpp and pc_shot.cpp (COS_SWITCH_DEKO3D): Aurora's texture level 0 as 4-byte texels
extern "C" bool aurora_switch_dk_read_texture(void* texture, uint8_t** pixels, uint32_t* width, uint32_t* height,
                                              uint32_t* wgpuFormat) {
    swdk::host_alloc_thread();
    static std::vector<uint8_t> buffer;
    wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
    if (!read_texture(static_cast<WGPUTexture>(texture), buffer, *width, *height, format)) return false;
    *pixels = buffer.data();
    *wgpuFormat = static_cast<uint32_t>(format);
    return true;
}
