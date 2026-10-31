#!/usr/bin/env bash
# CPU-only fixture sources; execute in the approved external SDK, never install.
set -euo pipefail
: "${FIXTURE_BUILD_ROOT:?}"
: "${WINE_SDK_ROOT:?}"
[[ $FIXTURE_BUILD_ROOT == /* && ! -e $FIXTURE_BUILD_ROOT ]]
here=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$FIXTURE_BUILD_ROOT"
cc --version > "$FIXTURE_BUILD_ROOT/TOOLCHAIN.txt"
x86_64-w64-mingw32-gcc --version >> "$FIXTURE_BUILD_ROOT/TOOLCHAIN.txt"
i686-w64-mingw32-gcc --version >> "$FIXTURE_BUILD_ROOT/TOOLCHAIN.txt"
(cd "$here" && sha256sum backend.c probe.c backend-wow64.c probe32.c lifecycle.c lifecycle-helper.c lifecycle.h transition.c run-lifecycle.py ../wow64/backend.c \
    ../../api/wow64-backend-abi.tsv ../../api/wow64-required.txt build-fixtures.sh run-fixtures.py) \
    > "$FIXTURE_BUILD_ROOT/SOURCE-SHA256SUMS"
python3 - "$here/../../api/wow64-backend-abi.tsv" "$FIXTURE_BUILD_ROOT/wow64-native-contract.h" <<'PY'
from pathlib import Path
import sys
rows = [line.split('\t') for line in Path(sys.argv[1]).read_text().splitlines() if line and not line.startswith('#')]
Path(sys.argv[2]).write_text('/* Independent pinned C native prototypes. */\n' + ''.join(
    f'int {name}({args});\n' for name,args in rows))
PY
cc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared -pthread "$here/backend.c" \
    -Wl,-soname,libcuda.so.1 -o "$FIXTURE_BUILD_ROOT/libcuda.so.1"
x86_64-w64-mingw32-gcc -std=c11 -O2 -Wall -Wextra -Werror -static-libgcc "$here/probe.c" \
    -o "$FIXTURE_BUILD_ROOT/probe.exe"
cc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared -pthread "$here/backend-wow64.c" \
    -I "$FIXTURE_BUILD_ROOT" -Wl,-soname,libcuda.so.1 -o "$FIXTURE_BUILD_ROOT/libcuda-wow64.so.1"
i686-w64-mingw32-gcc -std=c11 -O2 -Wall -Wextra -Werror -static-libgcc "$here/probe32.c" \
    -o "$FIXTURE_BUILD_ROOT/probe32.exe"
x86_64-w64-mingw32-gcc -std=c11 -O2 -Wall -Wextra -Werror -static-libgcc "$here/lifecycle.c" \
    -o "$FIXTURE_BUILD_ROOT/lifecycle.exe"
x86_64-w64-mingw32-gcc -std=c11 -O2 -Wall -Wextra -Werror -static-libgcc -shared \
    "$here/lifecycle-helper.c" -o "$FIXTURE_BUILD_ROOT/lifecycle-helper.dll"
x86_64-w64-mingw32-gcc -std=c11 -O2 -Wall -Wextra -Werror -static-libgcc -D__WINESRC__ \
    -MD -MF "$FIXTURE_BUILD_ROOT/transition.d" \
    -I "$WINE_SDK_ROOT/include-generated" -I "$WINE_SDK_ROOT/include" \
    "$here/transition.c" -o "$FIXTURE_BUILD_ROOT/transition.exe"
(cd "$FIXTURE_BUILD_ROOT" && sha256sum libcuda.so.1 libcuda-wow64.so.1 probe.exe probe32.exe \
    lifecycle.exe lifecycle-helper.dll transition.exe transition.d wow64-native-contract.h TOOLCHAIN.txt SOURCE-SHA256SUMS > SHA256SUMS)
