#pragma once
// aleks2 (FIFO-opt, DIRECT_INDICES): the indices prepare_idx_buffer (command_processor.cpp)
// appends, written to memory the caller already reserved. Same values, same order and the
// same count, quirks included (a quad list whose count is not a multiple of 4 still ends
// in a whole quad; a fan or strip of fewer than 3 vertices gives its vertices as they are).
// Free of Aurora types other than the GX enum so the host test can compare it with a copy
// of prepare_idx_buffer (tests/fifo_fast_index_test.cpp).
#include <dolphin/gx/GXEnum.h>

#include <cstdint>
#include <cstring>

namespace aurora::gx::index_gen {

// Indices prepare_idx_buffer appends for one primitive; 0 for a type it does not support.
inline uint32_t count(GXPrimitive prim, uint16_t vtxCount) noexcept {
  switch (prim) {
  case GX_QUADS:
    return ((static_cast<uint32_t>(vtxCount) + 3) / 4) * 6;
  case GX_TRIANGLES:
    return vtxCount;
  case GX_TRIANGLEFAN:
  case GX_TRIANGLESTRIP:
    return vtxCount <= 3 ? vtxCount : 3 + (static_cast<uint32_t>(vtxCount) - 3) * 3;
  case GX_LINES:
  case GX_LINESTRIP:
  case GX_POINTS:
    return 6;
  default:
    return 0;
  }
}

// Writes count(prim, vtxCount) indices to `out`, which must be 2-byte aligned (index
// ranges always are: they start on a 4-byte boundary or right after other u16 indices).
inline void write(uint8_t* bytes, GXPrimitive prim, uint16_t vtxStart, uint16_t vtxCount) noexcept {
  auto* out = reinterpret_cast<uint16_t*>(bytes);
  const auto put = [&out](uint16_t v) { *out++ = v; };
  switch (prim) {
  case GX_QUADS:
    for (uint16_t v = 0; v < vtxCount; v += 4) {
      const uint16_t i0 = vtxStart + v;
      put(i0);
      put(i0 + 1);
      put(i0 + 2);
      put(i0 + 2);
      put(i0 + 3);
      put(i0);
    }
    break;
  case GX_TRIANGLES:
    for (uint16_t v = 0; v < vtxCount; ++v) {
      put(vtxStart + v);
    }
    break;
  case GX_TRIANGLEFAN:
    for (uint16_t v = 0; v < vtxCount; ++v) {
      const uint16_t idx = vtxStart + v;
      if (v < 3) {
        put(idx);
        continue;
      }
      put(vtxStart);
      put(idx - 1);
      put(idx);
    }
    break;
  case GX_TRIANGLESTRIP:
    for (uint16_t v = 0; v < vtxCount; ++v) {
      const uint16_t idx = vtxStart + v;
      if (v < 3) {
        put(idx);
        continue;
      }
      if ((v & 1) == 0) {
        put(idx - 2);
        put(idx - 1);
      } else {
        put(idx - 1);
        put(idx - 2);
      }
      put(idx);
    }
    break;
  case GX_LINES:
  case GX_LINESTRIP:
  case GX_POINTS:
    put(0);
    put(1);
    put(3);
    put(3);
    put(2);
    put(0);
    break;
  default:
    break;
  }
}

} // namespace aurora::gx::index_gen
