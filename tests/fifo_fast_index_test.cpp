// aleks2 (FIFO-opt): index_gen (DIRECT_INDICES) against prepare_idx_buffer, byte for byte.
// The reference below is command_processor.cpp's prepare_idx_buffer as of 54fd97a, with
// ByteBuffer replaced by a vector (same append order). Standalone:
//   g++ -std=c++20 -O2 -Iinclude tests/fifo_fast_index_test.cpp -o idx_test && ./idx_test
#include "../lib/gx/index_gen.hpp"

#include <array>
#include <cstdio>
#include <vector>

namespace {
struct Buf {
  std::vector<uint8_t> bytes;
  template <typename T>
  void append(const T& v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    bytes.insert(bytes.end(), p, p + sizeof(T));
  }
  void reserve_extra(size_t) {}
};

u16 prepare_idx_buffer(Buf& buf, GXPrimitive prim, u16 vtxStart, u16 vtxCount) {
  u16 numIndices = 0;
  if (prim == GX_QUADS) {
    for (u16 v = 0; v < vtxCount; v += 4) {
      u16 idx0 = vtxStart + v;
      u16 idx1 = vtxStart + v + 1;
      u16 idx2 = vtxStart + v + 2;
      u16 idx3 = vtxStart + v + 3;
      buf.append(idx0);
      buf.append(idx1);
      buf.append(idx2);
      numIndices += 3;
      buf.append(idx2);
      buf.append(idx3);
      buf.append(idx0);
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLES) {
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      buf.append(idx);
      ++numIndices;
    }
  } else if (prim == GX_TRIANGLEFAN) {
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      buf.append(std::array{vtxStart, static_cast<u16>(idx - 1), idx});
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLESTRIP) {
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      if ((v & 1) == 0) {
        buf.append(std::array{static_cast<u16>(idx - 2), static_cast<u16>(idx - 1), idx});
      } else {
        buf.append(std::array{static_cast<u16>(idx - 1), static_cast<u16>(idx - 2), idx});
      }
      numIndices += 3;
    }
  } else {
    buf.append<u16>(0);
    buf.append<u16>(1);
    buf.append<u16>(3);
    buf.append<u16>(3);
    buf.append<u16>(2);
    buf.append<u16>(0);
    numIndices = 6;
  }
  return numIndices;
}
} // namespace

int main() {
  const GXPrimitive prims[] = {GX_QUADS, GX_TRIANGLES, GX_TRIANGLEFAN, GX_TRIANGLESTRIP,
                               GX_LINES, GX_LINESTRIP, GX_POINTS};
  const u16 starts[] = {0, 1, 7, 4093, 65000, 65535};
  size_t cases = 0;
  int failures = 0;
  for (const auto prim : prims) {
    for (const u16 start : starts) {
      for (u32 count = 0; count <= 600; ++count) {
        Buf ref;
        const u16 refCount = prepare_idx_buffer(ref, prim, start, static_cast<u16>(count));
        const u32 n = aurora::gx::index_gen::count(prim, static_cast<u16>(count));
        std::vector<uint8_t> out(n * sizeof(u16) + 2, 0xCD);
        aurora::gx::index_gen::write(out.data(), prim, start, static_cast<u16>(count));
        const bool ok = static_cast<u16>(n) == refCount && ref.bytes.size() == n * sizeof(u16) &&
                        std::equal(ref.bytes.begin(), ref.bytes.end(), out.begin()) &&
                        out[n * 2] == 0xCD && out[n * 2 + 1] == 0xCD;
        ++cases;
        if (!ok && ++failures <= 10) {
          std::printf("FAIL prim=%d start=%u count=%u: ref %u indices / %zu bytes, new %u\n",
                      static_cast<int>(prim), static_cast<unsigned>(start), static_cast<unsigned>(count),
                      static_cast<unsigned>(refCount), ref.bytes.size(), static_cast<unsigned>(n));
        }
      }
    }
  }
  std::printf("%zu cases, %d failures\n", cases, failures);
  return failures == 0 ? 0 : 1;
}
