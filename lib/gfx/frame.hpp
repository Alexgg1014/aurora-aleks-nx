#pragma once

#include "resources.hpp"
#include "frame_packet.hpp"

#include <optional>

namespace aurora::gfx::detail {

// Frames in flight. The console's present queue is deeper than a desktop's, so it
// keeps one more recording slot; the number of staging buffers is five on every
// platform.
#if defined(__SWITCH__)
inline constexpr size_t FrameSlotCount = 3;
#else
inline constexpr size_t FrameSlotCount = 2;
#endif
inline constexpr size_t StagingBufferCount = 5;
inline constexpr uint64_t StagingBufferSize = UniformBufferSize + VertexBufferSize + IndexBufferSize +
                                              StorageBufferSize + (UseTextureBuffer ? TextureUploadSize : 0);

const wgpu::Buffer& staging_buffer(size_t slot);

struct RegisteredDrawType {
  DrawCallback draw = nullptr;
  void* userdata = nullptr;
};

struct RegisteredEncoderTaskType {
  EncoderTaskCallback callback = nullptr;
  void* userdata = nullptr;
  EncoderTaskCompletionCallback afterSubmit = nullptr;
};

std::optional<RegisteredDrawType> find_runtime_draw_type(DrawTypeId id);
std::optional<RegisteredEncoderTaskType> find_runtime_encoder_task_type(EncoderTaskId id);

} // namespace aurora::gfx::detail

namespace aurora::gfx {
void initialize();
void shutdown();
bool begin_frame();
void end_frame(EndFrameCallback callback);
uint32_t current_frame() noexcept;
void after_submit() noexcept;
void gpu_synchronize();
void after_present() noexcept;
// aleks2 (V1): the input -> present probe (frame.cpp): the first PADRead of the frame being ended, then its present
int64_t latency_take_input() noexcept;
void latency_note_presented(int64_t inputNs) noexcept;
float calculate_fps() noexcept;
} // namespace aurora::gfx
