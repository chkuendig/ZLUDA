#!/bin/bash
set -euo pipefail
work="$RUNNER_TEMP/legacy-review"
mkdir "$work" "$work/context" "$work/tools" "$work/target" legacy-output
rust=rust@sha256:114c7a4425406451c2866b6aafe69fe29b1b298832db1277d411ac73c82d04d6
ubuntu=ubuntu@sha256:8a2321a92b438238f93b4346fff61bc4259c99fdd4d9b52ee86f153fee0c344f
docker pull "$rust"
container=$(docker create "$rust")
docker cp "$container:/usr/local/cargo/." "$work/tools/cargo"
docker cp "$container:/usr/local/rustup/." "$work/tools/rustup"
docker rm "$container"
cp .github/workflows/rocm_setup_build.sh "$work/context/"
cat > "$work/context/Dockerfile" <<'DOCKER'
ARG BASE
FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake ninja-build git git-lfs python3 pkg-config libssl-dev libnuma-dev curl wget ca-certificates gnupg binutils zstd
COPY rocm_setup_build.sh /tmp/rocm_setup_build.sh
RUN bash /tmp/rocm_setup_build.sh 6.3.4
DOCKER
docker build --build-arg "BASE=$ubuntu" -t legacy-review "$work/context" 2>&1 | tee legacy-output/provision.log
docker run --rm --cpus=4 --memory=10g --memory-swap=10g \
  -v "$GITHUB_WORKSPACE:/src" -v "$work/tools:/tools" -v "$work/target:/target" \
  -w /src -e RUSTUP_HOME=/tools/rustup -e CARGO_HOME=/tools/cargo \
  legacy-review bash -euo pipefail -c '
    export PATH="/tools/cargo/bin:$PATH" CARGO_TARGET_DIR=/target CARGO_BUILD_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_INCREMENTAL=0 LD_LIBRARY_PATH=/opt/rocm/lib
    git config --global --add safe.directory /src
    git config --global --add safe.directory /src/ext/llvm-project
    git config --global --add safe.directory /src/ext/HiGHS
    rustc +1.99.0 -vV > legacy-output/RUSTC.txt
    git rev-parse HEAD > legacy-output/SOURCE.txt
    git submodule status --recursive > legacy-output/SUBMODULES.txt
    cargo +1.99.0 test --locked -p ptx --lib -j4 --no-run
    status=0
    cargo +1.99.0 test --locked -p ptx --lib -j4 pass::map_f64_to_f32::tests:: -- --test-threads=1 || status=1
    cargo +1.99.0 test --locked -p ptx --lib -j4 test::legacy_32bit_translation -- --exact --test-threads=1 || status=1
    for kernel in legacy_sm10 legacy_slct; do
      touch "ptx/src/test/ll/$kernel.ll"
      set +e
      TEST_PTX_LLVM_FAIL_DIR=/src/legacy-output cargo +1.99.0 test --locked -p ptx --lib -j4 "test::spirv_run::${kernel}_llvm" -- --exact --test-threads=1 > "legacy-output/$kernel-generate.log" 2>&1
      rc=$?
      set -e
      printf "%s\n" "$rc" > "legacy-output/$kernel-generate.rc"
      if [ "$rc" -ne 101 ] || [ ! -s "legacy-output/$kernel.ll" ]; then
        status=1
        continue
      fi
      cp "legacy-output/$kernel.ll" "ptx/src/test/ll/$kernel.ll"
      cargo +1.99.0 test --locked -p ptx --lib -j4 "test::spirv_run::${kernel}_llvm" -- --exact --test-threads=1 || status=1
    done
    sha256sum legacy-output/*.ll > legacy-output/SHA256SUMS
    exit "$status"
  ' 2>&1 | tee legacy-output/build.log
