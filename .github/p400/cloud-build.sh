#!/bin/bash
set -euo pipefail
control="$GITHUB_WORKSPACE/control/.github/p400"
out="$GITHUB_WORKSPACE/dist/$TARGET/diagnostics"
work="$RUNNER_TEMP/p400-$TARGET"
mkdir "$work"
mkdir "$work/tools" "$work/context"
ubuntu=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["ubuntu"])' "$control/images.json")
rust=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["rust"])' "$control/images.json")
docker pull --platform linux/amd64 "$ubuntu"
docker pull --platform linux/amd64 "$rust"
container=$(docker create --platform linux/amd64 "$rust")
docker cp "$container:/usr/local/cargo/." "$work/tools/cargo"
docker cp "$container:/usr/local/rustup/." "$work/tools/rustup"
docker rm "$container"
cp "$GITHUB_WORKSPACE/source/.github/workflows/rocm_setup_build.sh" "$work/context/rocm_setup_build.sh"
cmp "$work/context/rocm_setup_build.sh" "$GITHUB_WORKSPACE/source/.github/workflows/rocm_setup_build.sh"
cat > "$work/context/Dockerfile" <<'DOCKER'
ARG BASE
FROM ${BASE}
ARG TARGET
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake ninja-build git git-lfs python3 pkg-config libssl-dev libnuma-dev curl wget ca-certificates gnupg binutils zstd
COPY rocm_setup_build.sh /tmp/rocm_setup_build.sh
RUN bash /tmp/rocm_setup_build.sh 6.3.4
DOCKER
image="p400-$TARGET:$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
docker build --platform linux/amd64 --build-arg "BASE=$ubuntu" --build-arg "TARGET=$TARGET" -t "$image" "$work/context" 2>&1 | tee "$out/provision.log"
docker image inspect "$ubuntu" "$rust" "$image" > "$out/IMAGE.json"
cp "$GITHUB_WORKSPACE/cloud-disk.json" "$out/CLOUD-DISK.json"
cp "$GITHUB_WORKSPACE/cloud-disk-before.txt" "$out/CLOUD-DISK-BEFORE.txt"
args=(--rm --platform linux/amd64 --cpus=4 --memory=10g --memory-swap=10g
  -v "$GITHUB_WORKSPACE/source:/src:ro" -v "$control:/plan:ro"
  -v "$work/tools:/tools" -v "$GITHUB_WORKSPACE/dist/$TARGET:/out" -w /src
  -e TARGET -e SOURCE_COMMIT -e LLVM_COMMIT -e COLLECT_ORACLES
  -e GITHUB_REPOSITORY -e GITHUB_SHA -e GITHUB_RUN_ID -e GITHUB_RUN_ATTEMPT -e GITHUB_JOB -e GITHUB_REF
  -e RUSTUP_HOME=/tools/rustup -e CARGO_HOME=/tools/cargo -e GIT_OPTIONAL_LOCKS=0)
docker run "${args[@]}" --network bridge "$image" bash -euo pipefail -c '
  export PATH="/tools/cargo/bin:$PATH"
  git config --global --add safe.directory /src
  git config --global --add safe.directory /src/ext/llvm-project
  git config --global --add safe.directory /src/ext/HiGHS
  rustc +1.99.0 -vV > /out/diagnostics/RUSTC.txt
  cargo +1.99.0 -vV > /out/diagnostics/CARGO.txt
  grep -Fx "release: 1.99.0" /out/diagnostics/RUSTC.txt
  export CARGO_TARGET_DIR=/tmp/fetch-target
  if [ "${COLLECT_ORACLES:-false}" = true ]; then
    rustup component add rustfmt --toolchain 1.99.0
  fi
  cargo +1.99.0 fetch --locked
' 2>&1 | tee "$out/fetch.log"
mkdir "$work/target"
docker run "${args[@]}" --network none -v "$work/target:/target" "$image" bash /plan/build-in-container.sh
python3 "$control/source.py" "$GITHUB_WORKSPACE/source" "$TARGET" "$out/SOURCE.json"
