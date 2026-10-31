#!/usr/bin/env bash
# Test-only native preload. Build inside the same SDK as zoc; never install.
set -euo pipefail
: "${STACK_PROBE_BUILD_ROOT:?}"
[[ $STACK_PROBE_BUILD_ROOT == /* && ! -e $STACK_PROBE_BUILD_ROOT ]]
here=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$STACK_PROBE_BUILD_ROOT"
cc --version > "$STACK_PROBE_BUILD_ROOT/TOOLCHAIN.txt"
cc -std=c11 -O2 -Wall -Wextra -Werror -fstack-clash-protection -fPIC -shared "$here/stack-probe.c" \
    -ldl -o "$STACK_PROBE_BUILD_ROOT/libzluda_stack_probe.so"
(cd "$here" && sha256sum stack-probe.c build-probe.sh measure.py README.md) > "$STACK_PROBE_BUILD_ROOT/SOURCE-SHA256SUMS"
(cd "$STACK_PROBE_BUILD_ROOT" && sha256sum libzluda_stack_probe.so TOOLCHAIN.txt SOURCE-SHA256SUMS > SHA256SUMS)
