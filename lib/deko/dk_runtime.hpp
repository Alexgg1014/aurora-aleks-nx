#pragma once
// aleks2 (deko3d P1): the run-time switch between Dawn/NVK and the deko3d renderer, and the glue of
// lib/deko to this Aurora (log, lab options, boot marker). MIT, this project's.
//
// active() is set once, by webgpu::initialize, before any device exists; every hook of lib/deko in
// Aurora is behind it (and behind AURORA_GFX_DEKO3D at compile time), so with Dawn nothing of
// deko3d runs and no deko3d memory is reserved.
#include <cstddef>
#include <cstdint>

namespace aurora::gfx::dk {

bool active() noexcept;
void set_active(bool on) noexcept;

// The lab options: sdmc:/switch/dusklight/deko3d.ini, one NAME=value per line ('#' comments), set as
// environment variables before the renderer reads them (the COS_DK_* switches of lib/deko: test
// pattern, submit interval, conventions, window size...). Missing file: the defaults.
void load_options() noexcept;

// The fallback (deko3d-p1-plan.md): a marker written before the device is created and removed at the
// first present. boot_attempt() returns false when two starts in a row never presented: the caller
// then uses Dawn, and fell_back() stays true for the game to reset its setting and say so.
bool boot_attempt() noexcept;
void boot_presented() noexcept;
bool fell_back() noexcept;

}  // namespace aurora::gfx::dk

// lib/deko's log lines (swdk::dklog) into Aurora's log, module "aurora::deko"
extern "C" void aurora_dk_log(const char* line, size_t length);
