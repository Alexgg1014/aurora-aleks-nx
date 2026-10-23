#pragma once

#include "types.hpp"

namespace aurora::gfx {

BindGroupRef bind_group_ref(const WGPUBindGroupDescriptor& descriptor);
wgpu::BindGroup find_bind_group(BindGroupRef id);
wgpu::Sampler sampler_ref(const wgpu::SamplerDescriptor& descriptor);
#if defined(AURORA_GFX_DEKO3D)
// aleks2 (deko3d P1): the deko3d encoder's lookup (no reference counting per draw)
WGPUBindGroup find_bind_group_raw(BindGroupRef id);
#endif

namespace detail {
void clear_bind_group_cache();
void expire_cached_bind_groups();
void shutdown_resource_cache();
} // namespace detail

} // namespace aurora::gfx
