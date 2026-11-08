#!/bin/bash
set -euo pipefail
export PATH="/tools/cargo/bin:$PATH" CARGO_TARGET_DIR=/target CARGO_BUILD_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_INCREMENTAL=0 LD_LIBRARY_PATH=/opt/rocm/lib
[ -z "$(find /target -mindepth 1 -maxdepth 1 -print -quit)" ]
git config --global --add safe.directory /src
git config --global --add safe.directory /src/ext/llvm-project
git config --global --add safe.directory /src/ext/HiGHS
rustc +1.99.0 -vV > /out/diagnostics/RUSTC.txt
cargo +1.99.0 -vV > /out/diagnostics/CARGO.txt
grep -Fx 'release: 1.99.0' /out/diagnostics/RUSTC.txt
dpkg-query -W > /out/diagnostics/PACKAGES.txt
uname -a > /out/diagnostics/UNAME.txt
cp /etc/os-release /out/diagnostics/OS-RELEASE.txt
case "$TARGET" in
  jc2) packages=(-p ptx_parser -p ptx -p zluda --features ptx/ci_build) ;;
  reference) packages=(-p zluda) ;;
  trace) packages=(-p zluda_trace) ;;
  zoc) packages=(-p compiler --bin zoc) ;;
  *) exit 2 ;;
esac
set +e
if [ "$TARGET" = zoc ]; then
  cargo +1.99.0 build --locked --offline --release "${packages[@]}" -j 4 --message-format=json > /out/diagnostics/build.json 2> /out/diagnostics/build.log
else
  cargo +1.99.0 test --locked --offline "${packages[@]}" --lib -j 4 --no-run --message-format=json > /out/diagnostics/test-build.json 2> /out/diagnostics/test-build.log
fi
rc=$?
set -e
if [ "$rc" != 0 ]; then printf 'ENVIRONMENT-MISMATCH: pinned source/toolchain/ROCm build failed (rc=%s)\n' "$rc" | tee /out/diagnostics/STATUS.txt; exit "$rc"; fi
if [ "$TARGET" = trace ]; then
  cargo +1.99.0 tree --locked --offline -p zluda_trace --edges normal,build --target x86_64-unknown-linux-gnu > /out/diagnostics/CARGO-TREE.txt
  cargo +1.99.0 build --locked --offline -p zluda_trace -j 4 --message-format=json > /out/diagnostics/build.json 2> /out/diagnostics/build.log
elif [ "$TARGET" != zoc ]; then
  cp /out/diagnostics/test-build.json /out/diagnostics/build.json
  cp /out/diagnostics/test-build.log /out/diagnostics/build.log
fi
if [ "${COLLECT_ORACLES:-false}" = true ]; then
  test "$TARGET" = jc2
  python3 /plan/collect-oracles.py /out/diagnostics/test-build.json /src /out/diagnostics/oracle-collection /plan/oracle-cases.json
else
  python3 /plan/package-capsule.py
fi
