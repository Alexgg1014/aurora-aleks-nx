#!/usr/bin/env bash
# aleks2 (FIFO-opt): UNIFORM_IN_PLACE draws the same uniform buffer as the original path.
# Needs g++ (C++20) and include paths for fmt, Tracy, xxhash and Dawn's webgpu headers
# (the build's _deps), e.g. with DEPS=<build>/_deps:
#   FMT_INC=$DEPS/fmt-src/include TRACY_INC=$DEPS/tracy-src/public XXH_INC=$DEPS/xxhash-src \
#   DAWN_INC="$DEPS/dawn-src/include $DEPS/dawn-build/gen/include $DEPS/dawn-src/third_party/abseil-cpp" \
#   ABSL_SRC="<abseil .cc files raw_hash_set needs>" \
#     tests/fifo_opt/run-uniform-in-place-equivalence.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${TMPDIR:-/tmp}/aurora-fifo-opt-uniform"
mkdir -p "$out"
inc=(-I"$root/include" -I"$root/lib" -I"${FMT_INC:?}" -I"${TRACY_INC:?}" -I"${XXH_INC:?}")
for d in ${DAWN_INC:?}; do inc+=(-I"$d"); done
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -DAURORA -DTARGET_PC -DFMT_HEADER_ONLY -DXXH_INLINE_ALL "${inc[@]}" \
  "$here/uniform_in_place_equivalence.cpp" "$root/lib/gx/shader_info.cpp" ${ABSL_SRC:-} -o "$out/bin"
a="$("$out/bin" 0)"
b="$("$out/bin" 4)"
echo "original path:    $a"
echo "UNIFORM_IN_PLACE: $b"
if [[ "${a%% *}" != "${b%% *}" ]]; then
  echo "MISMATCH: the in-place path changed the uniform buffer or a range" >&2
  exit 1
fi
echo "UNIFORM_IN_PLACE is byte for byte the original path"
