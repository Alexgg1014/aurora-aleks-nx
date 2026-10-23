#pragma once

#include "types.hpp"

#include <vector>

namespace aurora::gfx::depth_peek {

void initialize();
void shutdown();

void request_snapshot() noexcept;
bool read_latest(uint16_t x, uint16_t y, uint32_t& z) noexcept;

void encode_frame_snapshot(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& depthView,
                           wgpu::Extent3D sourceSize, uint32_t msaaSamples) noexcept;
void after_submit() noexcept;

#if defined(AURORA_GFX_DEKO3D)
// aleks2 (deko3d P1, after SwitchWaker patch 0016): with the deko3d renderer there is no compute pass
// on Dawn's Null device. Its encoder (lib/deko/dk_encode.cpp) takes the pending request when it
// records the frame's last EFB pass and copies the depth itself; read_latest reads its copies
// (dk::depth_peek_read). Logical EFB pixel (x, y) is depth image pixel
// floor(offset + (x + 0.5, y + 0.5) * scale), as the compute shader samples it.
struct SnapshotMapping {
  uint32_t logicalWidth = 0;
  uint32_t logicalHeight = 0;
  float offsetX = 0.f;
  float offsetY = 0.f;
  float scaleX = 1.f;
  float scaleY = 1.f;
};
// render worker: whether GXPeekZ asked for a snapshot (cleared; the same rate limit as the compute
// pass), and the mapping for a depth image whose EFB part is sourceWidth x sourceHeight
bool take_snapshot_request(uint32_t sourceWidth, uint32_t sourceHeight, SnapshotMapping& mapping) noexcept;
#endif

namespace testing {
void reset() noexcept;
bool snapshot_requested() noexcept;
void set_latest(uint32_t width, uint32_t height, const std::vector<uint32_t>& data);
} // namespace testing

} // namespace aurora::gfx::depth_peek
