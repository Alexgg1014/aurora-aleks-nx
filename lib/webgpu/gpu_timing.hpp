#pragma once

// aleks2 (V1-c1): whole-frame GPU time without Tracy, for the benchmark and for the
// dynamic-resolution controller's "is the GPU the limit?" question.
//
// Each frame's encoder gets an empty 1x1 render pass at its start and another at its end;
// the first writes a timestamp at the beginning of its pass, the second at the end of its
// pass. Pass timestampWrites are plain WebGPU (only the TimestampQuery feature), unlike
// encoder.WriteTimestamp, which Dawn refuses without allow_unsafe_apis -- and a refused
// command is an uncaptured error, which Aurora treats as fatal.
//
// Results come back through a small ring of mapped buffers, two or three frames late.
// Nothing is written unless a consumer turned timing on (aurora_gpu_timing_set_enabled).

#include <webgpu/webgpu_cpp.h>

namespace aurora::webgpu::gpu_timing {

// After the device exists. Idempotent; a device without TimestampQuery leaves it inert.
void initialize();
void shutdown();

// Render worker, around the frame's encoder (same places as gpu_prof).
void frame_begin(const wgpu::CommandEncoder& encoder);
void frame_end(const wgpu::CommandEncoder& encoder);
void after_submit();

// aleks2 (V1-rc16): whether a consumer asked for GPU time (aurora_gpu_timing_set_enabled). The deko3d encoder
// writes its per-pass timestamps only while this is true.
bool requested();

} // namespace aurora::webgpu::gpu_timing
