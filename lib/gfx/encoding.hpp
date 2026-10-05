#pragma once

#include "frame_packet.hpp"

#include <optional>

namespace aurora::gfx {

bool bind_pipeline(PipelineRef ref, const wgpu::RenderPassEncoder& pass);
void bind_pipeline(const wgpu::RenderPipeline& pipeline, const wgpu::RenderPassEncoder& pass);

// Most consecutive draws in a pass bind the same texture group and the same blend
// constant as the one before. The filter remembers what was last set on the pass
// encoder and issues a bind only when it differs. It knows nothing about writes
// made behind its back, so whoever starts a pass, or sets either of these by hand,
// calls forget().
class PassBindFilter {
public:
  void forget() noexcept {
    mTextureGroup.reset();
    mBlendAlpha.reset();
  }
  void texture_group(const wgpu::RenderPassEncoder& pass, BindGroupRef group);
  void blend_alpha(const wgpu::RenderPassEncoder& pass, uint32_t dstAlpha);

private:
  std::optional<BindGroupRef> mTextureGroup;
  std::optional<uint32_t> mBlendAlpha;
};

// The filter for the pass being encoded. Passes are encoded on one thread at a time.
PassBindFilter& pass_binds() noexcept;

namespace detail {
void encode_op(wgpu::CommandEncoder& encoder, FramePacket& frame, const FrameOp& op);
}

} // namespace aurora::gfx
