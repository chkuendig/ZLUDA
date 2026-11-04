#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Build every W1 fixture with mingw-w64 (x64 + i686) and the Linux counter.
#   build.sh OUTDIR
# Needs: x86_64-w64-mingw32-gcc, i686-w64-mingw32-gcc, cc, python3.
# Output: OUTDIR/{x64,pe32,linux}/..., TOOLCHAIN.txt, SOURCE-SHA256SUMS, SHA256SUMS.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
out=${1:?output directory}
[[ ! -e $out ]] || { echo "output must not exist: $out" >&2; exit 1; }
mkdir -p "$out"/{x64,pe32,linux}
out=$(cd -- "$out" && pwd)
{
  x86_64-w64-mingw32-gcc --version | head -1
  i686-w64-mingw32-gcc --version | head -1
  cc --version | head -1
} > "$out/TOOLCHAIN.txt"
(cd "$here" && find . -type f \( -name '*.c' -o -name '*.h' -o -name '*.S' -o -name '*.sh' -o -name '*.py' \) \
  -not -path './out*' | LC_ALL=C sort | xargs sha256sum) > "$out/SOURCE-SHA256SUMS"

cflags=(-std=gnu11 -O1 -g0 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable
        -Wno-missing-field-initializers -Wno-cast-function-type -fno-omit-frame-pointer -I"$here/include" -I"$here/src")
ldflags=(-static -static-libgcc -Wl,--no-insert-timestamp)

build() { # arch cc
  local arch=$1 cc=$2 dir=$out/$1
  "$cc" "${cflags[@]}" -o "$dir/identity.exe" "$here/src/identity.c" "${ldflags[@]}" -lpsapi
  "$cc" "${cflags[@]}" -o "$dir/realgpu.exe" "$here/src/realgpu.c" "${ldflags[@]}"
  "$cc" "${cflags[@]}" -o "$dir/lifecycle.exe" "$here/src/lifecycle.c" "${ldflags[@]}"
  "$cc" "${cflags[@]}" -shared -o "$dir/lc_helper.dll" "$here/src/lc_helper.c" "${ldflags[@]}"
  "$cc" "${cflags[@]}" -o "$dir/churn.exe" "$here/src/churn.c" "${ldflags[@]}" -lpsapi
  "$cc" "${cflags[@]}" -o "$dir/modload.exe" "$here/src/modload.c" "${ldflags[@]}"
  "$cc" "${cflags[@]}" -o "$dir/interop.exe" "$here/src/interop.c" "${ldflags[@]}" -ldxguid -luuid
}
build x64 x86_64-w64-mingw32-gcc
build pe32 i686-w64-mingw32-gcc
i686-w64-mingw32-gcc "${cflags[@]}" -o "$out/pe32/pe32_stack.exe" "$here/src/pe32_stack.c" "$here/src/pe32_probe.S" "${ldflags[@]}"
for kind in correct old; do
  i686-w64-mingw32-gcc "${cflags[@]}" -DW1_STUB_${kind^^} -shared -o "$out/pe32/w1stub_$kind.dll" \
    "$here/src/pe32_stubdll.c" "${ldflags[@]}" -Wl,--kill-at
done

cc -O2 -g0 -Wall -shared -fPIC -o "$out/linux/libw1-cuda-count.so" \
  "$here/tools/cuda-count/count.c" "$here/tools/cuda-count/tramp.S" -ldl -lpthread
cp "$here/tools/compare.py" "$here/tools/cuda-count/per_frame.py" "$here/run-cpu.py" "$out/linux/"

(cd "$out" && find x64 pe32 linux -type f | LC_ALL=C sort | xargs sha256sum) > "$out/SHA256SUMS"
cat "$out/SHA256SUMS"
