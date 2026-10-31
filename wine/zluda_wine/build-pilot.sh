#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Run only in an approved SDK/Deck build context; builds artifacts, never installs.
set -euo pipefail
[[ ${ZLUDA_WINE_DECLARATIONS_ONLY:-0} != 1 ]] || { echo "Check-only mode cannot build shipping artifacts" >&2; exit 2; }
: "${WINE_SDK_ROOT:?}" "${PILOT_BUILD_ROOT:?}"
[[ $WINE_SDK_ROOT == /* && $PILOT_BUILD_ROOT == /* ]] || { echo 'Use absolute SDK/build paths' >&2; exit 2; }
JOBS=${JOBS:-3}
[[ $JOBS =~ ^[1-3]$ ]] || { echo 'JOBS must be 1..3' >&2; exit 2; }
[[ $(rustc --version) == 'rustc 1.99.0 '* ]]
here=$(cd -- "$(dirname -- "$0")" && pwd)
[[ ! -e $PILOT_BUILD_ROOT ]] || { echo 'PILOT_BUILD_ROOT must be fresh' >&2; exit 1; }
mkdir -p "$PILOT_BUILD_ROOT/artifacts"
export CARGO_BUILD_JOBS=$JOBS CARGO_TARGET_DIR=$PILOT_BUILD_ROOT/target
cd "$here"
cargo fetch --locked --target x86_64-pc-windows-gnu
cargo fetch --locked --target x86_64-unknown-linux-gnu
cargo build --locked --offline --release --target x86_64-unknown-linux-gnu
link_log=$PILOT_BUILD_ROOT/x86_64-native-libs.log
if ! cargo rustc --locked --offline --lib --release --target x86_64-pc-windows-gnu \
    -- --print native-static-libs > "$link_log" 2>&1; then
    cat "$link_log"
    exit 1
fi
cat "$link_log"
install -m755 "$CARGO_TARGET_DIR/x86_64-unknown-linux-gnu/release/libnvcuda.so" \
    "$PILOT_BUILD_ROOT/artifacts/nvcuda.so"
python3 "$here/tools/link_pe.py" --triplet x86_64 \
    --archive "$CARGO_TARGET_DIR/x86_64-pc-windows-gnu/release/libnvcuda.a" \
    --native-libs-log "$link_log" --output "$PILOT_BUILD_ROOT/artifacts/nvcuda.dll"
chmod 644 "$PILOT_BUILD_ROOT/artifacts/nvcuda.dll"
"$WINE_SDK_ROOT/bin/winebuild" --builtin "$PILOT_BUILD_ROOT/artifacts/nvcuda.dll"
python3 "$here/tools/pe_artifact.py" --triplet x86_64 "$PILOT_BUILD_ROOT/artifacts/nvcuda.dll"
(cd "$PILOT_BUILD_ROOT/artifacts" && sha256sum nvcuda.dll nvcuda.so > SHA256SUMS)
echo 'Unqualified x64 pilot artifacts only; no Proton tool or prefix was changed.'
