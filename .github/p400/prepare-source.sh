#!/bin/bash
set -euo pipefail
control="$GITHUB_WORKSPACE/control/.github/p400"
source_root="$GITHUB_WORKSPACE/source"
mkdir -p "$GITHUB_WORKSPACE/dist/$TARGET/diagnostics"
out="$GITHUB_WORKSPACE/dist/$TARGET/diagnostics"
if [ "$TARGET" = trace ]; then origin=https://github.com/vosen/ZLUDA.git; else origin=https://github.com/chkuendig/ZLUDA.git; fi
git -C "$source_root" remote set-url origin "$origin"
test "$(git -C "$source_root" rev-parse HEAD)" = "$SOURCE_COMMIT"
if [ "$TARGET" = zoc ]; then git -C "$source_root" fetch --depth=2 origin "$SOURCE_COMMIT"; fi
test "$(git -C "$source_root" ls-tree HEAD ext/llvm-project | awk '{print $3}')" = "$LLVM_COMMIT"
if [ "$TARGET" != trace ]; then
  git -C "$source_root" submodule update --init --depth=1 --jobs=3 -- ext/llvm-project ext/HiGHS
  GIT_LFS_SKIP_SMUDGE=0 git -C "$source_root" lfs pull --include='llvm_zluda/src/device-libs/ockl.bc,ptx/lib/*.bc' --exclude=''
fi
python3 "$control/source.py" "$source_root" "$TARGET" "$out/SOURCE.json"
