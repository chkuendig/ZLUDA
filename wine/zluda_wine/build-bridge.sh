#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Artifacts only. Caller owns SDK/container authorization and serialization.
set -euo pipefail
[[ ${ZLUDA_WINE_DECLARATIONS_ONLY:-0} != 1 ]] || { echo "Check-only mode cannot build shipping artifacts" >&2; exit 2; }
: "${WINE_SDK_ROOT:?}" "${ZLUDA_WINE_BUILD_ROOT:?}"
[[ $WINE_SDK_ROOT == /* && $ZLUDA_WINE_BUILD_ROOT == /* ]] || { echo 'Use absolute SDK/build paths' >&2; exit 2; }
JOBS=${JOBS:-3}
[[ $JOBS =~ ^[1-3]$ ]] || { echo 'JOBS must be 1..3' >&2; exit 2; }
[[ $(rustc --version) == 'rustc 1.99.0 '* ]]
here=$(cd -- "$(dirname -- "$0")" && pwd)
[[ ! -e $ZLUDA_WINE_BUILD_ROOT ]] || { echo 'ZLUDA_WINE_BUILD_ROOT must be fresh' >&2; exit 1; }
[[ -x $WINE_SDK_ROOT/bin/winebuild && -d $WINE_SDK_ROOT/include && -d $WINE_SDK_ROOT/include-generated ]]
[[ -f $WINE_SDK_ROOT/lib/ntdll.so ]] || { echo 'Matching prepared SDK lib/ntdll.so is required' >&2; exit 1; }
mkdir -p "$ZLUDA_WINE_BUILD_ROOT/artifacts"
: "${ZLUDA_WINE_ABI_RECEIPT:?}" "${ZLUDA_WINE_ABI_RECEIPT_SHA256:?}"
python3 "$here/tools/admit_wine.py" "$ZLUDA_WINE_ABI_RECEIPT" "$WINE_SDK_ROOT"
cp -- "$ZLUDA_WINE_ABI_RECEIPT" "$ZLUDA_WINE_BUILD_ROOT/WINE-CALLBACK-ATTESTATION.json"
export CARGO_BUILD_JOBS=$JOBS CARGO_TARGET_DIR=$ZLUDA_WINE_BUILD_ROOT/target
export CARGO_TERM_COLOR=never
: "${SOURCE_DATE_EPOCH:?}" "${ZLUDA_WINE_SOURCE_ID:?}"
export SOURCE_DATE_EPOCH
cd "$here"
for target in x86_64-unknown-linux-gnu x86_64-pc-windows-gnu i686-pc-windows-gnu; do
    cargo fetch --locked --target "$target"
done
cargo build --locked --offline --release --target x86_64-unknown-linux-gnu
for spec in 'x86_64 x86_64-windows' 'i686 i386-windows'; do
    read -r triplet arch <<< "$spec"
    link_log=$ZLUDA_WINE_BUILD_ROOT/$triplet-native-libs.log
    if ! ZLUDA_WINE_PE_CC=$triplet-w64-mingw32-gcc ZLUDA_WINE_PE_AR=$triplet-w64-mingw32-ar \
        cargo rustc --locked --offline --lib --release --target "$triplet-pc-windows-gnu" \
            -- --print native-static-libs > "$link_log" 2>&1; then
        cat "$link_log"
        exit 1
    fi
    cat "$link_log"
    mkdir -p "$ZLUDA_WINE_BUILD_ROOT/artifacts/$arch"
    python3 "$here/tools/link_pe.py" --triplet "$triplet" \
        --archive "$CARGO_TARGET_DIR/$triplet-pc-windows-gnu/release/libnvcuda.a" \
        --native-libs-log "$link_log" --output "$ZLUDA_WINE_BUILD_ROOT/artifacts/$arch/nvcuda.dll"
    chmod 644 "$ZLUDA_WINE_BUILD_ROOT/artifacts/$arch/nvcuda.dll"
    "$WINE_SDK_ROOT/bin/winebuild" --builtin "$ZLUDA_WINE_BUILD_ROOT/artifacts/$arch/nvcuda.dll"
    python3 "$here/tools/pe_artifact.py" --triplet "$triplet" "$ZLUDA_WINE_BUILD_ROOT/artifacts/$arch/nvcuda.dll"
done
mkdir -p "$ZLUDA_WINE_BUILD_ROOT/artifacts/x86_64-unix"
install -m755 "$CARGO_TARGET_DIR/x86_64-unknown-linux-gnu/release/libnvcuda.so" \
    "$ZLUDA_WINE_BUILD_ROOT/artifacts/x86_64-unix/nvcuda.so"
python3 - "$here" "$ZLUDA_WINE_BUILD_ROOT/artifacts" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
source, out = map(Path, sys.argv[1:])
names = ['x86_64-windows/nvcuda.dll', 'i386-windows/nvcuda.dll', 'x86_64-unix/nvcuda.so']
(out / 'SHA256SUMS').write_text(''.join(hashlib.sha256((out / n).read_bytes()).hexdigest() + '  ' + n + '\n' for n in names))
(out / 'BUILD-CONTRACT.json').write_text(json.dumps({
    'contract': 'zluda-wine-packaging-1',
    'source_commit': __import__('os').environ['ZLUDA_WINE_SOURCE_ID'],
    'rust': subprocess.check_output(['rustc', '--version'], text=True).strip(),
    'modules': names, 'qualification': 'PENDING: inspect equivalence receipts and target-runtime W1/W2'
}, indent=2) + '\n')
PY
echo "Bridge artifacts: $ZLUDA_WINE_BUILD_ROOT/artifacts (qualification required)"
