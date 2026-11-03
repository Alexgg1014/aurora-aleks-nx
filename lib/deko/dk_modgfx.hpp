#pragma once
// v1.1 (aleks2): mods' gfx service on the deko3d renderer (dk_modgfx.cpp). MIT, this project's.
//
// The encoder (dk_encode.cpp) calls these where Aurora's Dawn path runs a mod's runtime draw type or encoder
// task; the game's loader binds a mod's wgpu* imports to aurora_dk_wgpu_proc's functions, which do the deko3d
// work while the renderer is deko3d and forward to Dawn otherwise.
#include <cstddef>
#include <cstdint>

#include <webgpu/webgpu.h>

namespace aurora::gfx::detail {
struct CustomDrawCommand;
struct EncoderTask;
struct RenderPass;
struct FramePacket;
}  // namespace aurora::gfx::detail

namespace aurora::gfx::dk {

// Render worker, inside encode_pass with the pass's targets bound (tw x th): one CustomDraw command. The
// encoder's state cache is invalidated afterwards; the caller restores its viewport and scissor.
void modgfx_custom_draw(const detail::CustomDrawCommand& draw, const detail::RenderPass& pass, uint32_t tw,
                        uint32_t th);
// Render worker, between passes: one EncoderTask op.
void modgfx_encoder_task(detail::FramePacket& frame, const detail::EncoderTask& task);
// Render worker, at the start of every op: queued writes (wgpuQueueWriteBuffer / wgpuQueueWriteTexture) are
// copied.
void modgfx_flush_writes();
// Render worker, at every frame's start: which deko3d frame records which game frame (MapAsync), and the
// buffers released long enough ago are freed.
void modgfx_frame_begin(const detail::FramePacket& frame);
// Aurora's per-frame event pump (gfx/frame.cpp process_events): MapAsync callbacks whose data is ready.
void modgfx_process_events();

}  // namespace aurora::gfx::dk

extern "C" {
// The function the loader binds for a wgpu* name, or null (the name keeps Dawn's own function).
void* aurora_dk_wgpu_proc(const char* name);
// The names aurora_dk_wgpu_proc knows, one per index; null past the last.
const char* aurora_dk_wgpu_proc_name(size_t index);
// Times a mod's work could not be done with deko3d (an unsupported call, a shader that does not translate or
// compile): the game compares it around each callback and moves that mod to Vulkan (deko3d_policy).
uint32_t aurora_dk_modgfx_failures(void);
// Whether this process renders with deko3d.
bool aurora_dk_active(void);
}
