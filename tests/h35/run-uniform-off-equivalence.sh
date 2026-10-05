#!/usr/bin/env bash
# aleks2 (H35): shows that uniform reuse mode Off is byte for byte the H34 path.
# Builds tests/h35/uniform_off_equivalence.cpp against shader_info.cpp of BASE (default:
# 0020cbf, the Aurora commit H34 shipped with) and of the working tree, runs both on
# the same 200 000 randomized draws and compares the digests.
#
# Needs: g++ (C++20), libxxhash, abseil, and include paths for fmt, Tracy and Dawn's
# generated webgpu headers:
#   FMT_INC=... TRACY_INC=... DAWN_INC="dawn/include dawn/gen/include" \
#     tests/h35/run-uniform-off-equivalence.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
base="${BASE:-0020cbf6365c171e1f25988567e1d3a51168b7bc}"
out="${TMPDIR:-/tmp}/aurora-h35-off-equivalence"
rm -rf "$out" && mkdir -p "$out/base/gx"
cp -r "$root/lib/gx/." "$out/base/gx/"
git -C "$root" show "$base:lib/gx/shader_info.cpp" > "$out/base/gx/shader_info.cpp"
git -C "$root" show "$base:lib/gx/shader_info.hpp" > "$out/base/gx/shader_info.hpp"
for d in gfx webgpu; do ln -s "$root/lib/$d" "$out/base/$d"; done
for f in internal.hpp logging.hpp; do ln -s "$root/lib/$f" "$out/base/$f"; done
inc=(-I"$root/include" -I"${FMT_INC:?}" -I"${TRACY_INC:?}")
for d in ${DAWN_INC:?}; do inc+=(-I"$d"); done
flags=(-std=c++20 -O1 -DAURORA -DTARGET_PC -DFMT_HEADER_ONLY "${inc[@]}")
libs=(-lxxhash $(pkg-config --libs absl_raw_hash_set 2>/dev/null || true))
g++ "${flags[@]}" -I"$out/base" "$here/uniform_off_equivalence.cpp" "$out/base/gx/shader_info.cpp" -o "$out/base_bin" "${libs[@]}"
g++ "${flags[@]}" -I"$root/lib" -DH35 "$here/uniform_off_equivalence.cpp" "$root/lib/gx/shader_info.cpp" -o "$out/h35_bin" "${libs[@]}"
a="$("$out/base_bin")"
b="$("$out/h35_bin")"
echo "base ($base): $a"
echo "this tree, Off: $b"
if [[ "$a" != "$b" ]]; then
  echo "MISMATCH: Off is not the H34 path" >&2
  exit 1
fi
echo "Off is byte for byte the H34 path"
