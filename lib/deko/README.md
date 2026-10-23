# lib/deko — the experimental deko3d renderer (Dusklight Switch, deko3d P1)

A second way to put Aurora's frames on the Switch's GPU, chosen at run time (`BACKEND_DEKO3D`,
Settings > Graphics Backend > "deko3d (Experimental)"). Dawn/NVK stays the default and the fallback.
Aurora records every frame as for WebGPU, with Dawn's Null device as its object model; this directory
turns the recorded frames into deko3d commands. With the Dawn backend none of it runs (`dk::active()`).

## Credit

The design and most of the code come from **SwitchWaker**, by **centollOS** (Pulpparty, depende3000 and the SwitchWaker contributors; <https://github.com/centollOS/SwitchWaker>,
commit `084291f`, MIT), which built the same bridge for The Wind Waker on the same Aurora base
(`encounter/aurora` `3227d76`). Its platform layer comes in turn from **SwitchWakerHD**
(<https://github.com/centollOS/SwitchWakerHD>, `df8fbde`, MPL-2.0). Thank you to both projects.

## Licences

| Path | Origin | Licence |
|---|---|---|
| `mpl/` (`dk.h`, `device.cpp`, `memory.cpp`, `descriptors.cpp`, `overlay.cpp`, `pattern.cpp`, `shaders.cpp`, `dksh_file.*`, `shaders/*`) | SwitchWakerHD via SwitchWaker `switch/deko/` | **MPL-2.0** (header in each file). Our changes are marked at the top of each file; their diffs against the import commit are the change notes. |
| `dk_encode.*`, `dk_objects.*`, `dk_pipeline.*`, `dk_present.cpp`, `dk_gfx.hpp`, `shader_cache.cpp`, `shader_translate.*`, `dk_aurora.h` | SwitchWaker `switch/deko/` and `switch/deko/aurora/` | MIT (SwitchWaker: Copyright (c) 2026 Pulpparty and depende3000, and the SwitchWaker contributors) |
| `dk_rmlui.cpp`, `dk_runtime.*` | this project | MIT |
| deko3d 0.5.0, uam 1.1.0 (built from the SwitchWaker checkout, `switch/deko3d`, `switch/uam`, with their `PATCHES.md`) | devkitPro (fincs), patches by SwitchWakerHD/SwitchWaker | zlib (uam's `mesa-imported/`: MIT) |
| Tint options `depth_zero_to_one`, `disable_position_y_negation` (`switch/dawn/patches/dawn-switch-tint-*.patch`, applied by `cmake/aurora_deko3d.cmake`) | SwitchWaker | MIT (Dawn: BSD-3) |

No GPL code is used (Dolphin, Nezumiiruka and SwitchWaker's `native/dsp_hle` are not).

## What differs from SwitchWaker

- Run-time selection instead of a separate NRO: every hook is behind `dk::active()`.
- RmlUi (menus, Settings, mod dialogs): `dk_rmlui.cpp`, the 12 pipeline kinds, stencil clips, the
  composite at the present. SwitchWaker builds Aurora without RmlUi.
- Paths: the offline DKSH cache is `romfs:/initial_dksh_cache.bin`; what the console compiles goes to
  `sdmc:/switch/dusklight/cache/dksh_local.bin`; lab switches (`COS_DK_*`) from
  `sdmc:/switch/dusklight/deko3d.ini`.
- Dawn's shader cache (`dawn_cache.db`) is not touched in deko3d mode.
- Not ported yet: the companion screen and Flip Grip (deko3d mode is single screen, normal layout),
  dynamic resolution, mods' gfx-service draws and encoder tasks (skipped, logged).
