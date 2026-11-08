#!/bin/bash
set -euo pipefail
mode=${1:?prepare or build required}
case "$mode" in prepare|build) ;; *) exit 2 ;; esac
control="$GITHUB_WORKSPACE/control/.github/p400"
out="$GITHUB_WORKSPACE/dist/$TARGET/diagnostics"
cache_out="$GITHUB_WORKSPACE/dist/$TARGET/compiler-cache"
work="$RUNNER_TEMP/p400-$TARGET"
image="p400-$TARGET:$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
cache_enabled=$(python3 -c 'import json,sys;print(str(json.load(open(sys.argv[1]))[sys.argv[2]]["llvm_built"]).lower())' "$control/pins.json" "$TARGET")
# Foreign refs can run the original cold build, but never restore/save this cache.
if [ "${GITHUB_REF:-}" != refs/heads/ci/p400-capsules-20261107 ]; then cache_enabled=false; fi
args=(--rm --platform linux/amd64 --cpus=4 --memory=10g --memory-swap=10g
  -v "$GITHUB_WORKSPACE/source:/src:ro" -v "$control:/plan:ro"
  -v "$work/tools:/tools" -v "$GITHUB_WORKSPACE/dist/$TARGET:/out" -w /src
  -e TARGET -e SOURCE_COMMIT -e LLVM_COMMIT -e COLLECT_ORACLES
  -e GITHUB_REPOSITORY -e GITHUB_SHA -e GITHUB_RUN_ID -e GITHUB_RUN_ATTEMPT -e GITHUB_JOB -e GITHUB_REF
  -e RUSTUP_HOME=/tools/rustup -e CARGO_HOME=/tools/cargo -e GIT_OPTIONAL_LOCKS=0)
probe() {
  docker run "${args[@]}" --network none "$image" bash -euo pipefail -c '
    export PATH="/tools/cargo/bin:$PATH" LD_LIBRARY_PATH=/opt/rocm/lib
    python3 /plan/compiler_cache.py probe
  '
}
if [ "$mode" = prepare ]; then
  mkdir "$work"
  mkdir "$work/tools" "$work/context" "$work/compiler-cache"
  mkdir "$cache_out"
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
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake ninja-build git git-lfs python3 pkg-config libssl-dev libnuma-dev curl wget ca-certificates gnupg binutils zstd ccache
COPY rocm_setup_build.sh /tmp/rocm_setup_build.sh
RUN bash /tmp/rocm_setup_build.sh 6.3.4
DOCKER
  docker build --platform linux/amd64 --build-arg "BASE=$ubuntu" --build-arg "TARGET=$TARGET" -t "$image" "$work/context" 2>&1 | tee "$out/provision.log"
  docker image inspect "$ubuntu" "$rust" "$image" > "$out/IMAGE.json"
  cp "$GITHUB_WORKSPACE/cloud-disk.json" "$out/CLOUD-DISK.json"
  cp "$GITHUB_WORKSPACE/cloud-disk-before.txt" "$out/CLOUD-DISK-BEFORE.txt"
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
  probe > "$cache_out/TOOLCHAIN-BEFORE.json"
  printf 'enabled=%s\n' "$cache_enabled" >> "$GITHUB_OUTPUT"
  if [ "$cache_enabled" = true ]; then
    python3 "$control/compiler_cache.py" identity "$control" "$GITHUB_WORKSPACE/source" \
      "$cache_out/TOOLCHAIN-BEFORE.json" "$TARGET" "$GITHUB_REF" > "$cache_out/IDENTITY.json"
    python3 - "$cache_out/IDENTITY.json" >> "$GITHUB_OUTPUT" <<'PY'
import json,sys
print('namespace='+json.load(open(sys.argv[1]))['namespace'])
PY
  fi
  exit
fi
if [ "$cache_enabled" = true ]; then
  namespace=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["namespace"])' "$cache_out/IDENTITY.json")
  if [ -n "${CACHE_MATCHED_KEY:-}" ] && [[ "$CACHE_MATCHED_KEY" != "$namespace"* ]]; then
    echo 'Compiler cache restore key is outside the exact branch-bound namespace' >&2
    exit 1
  fi
fi
# No target or Rust output is restored. The new target must be genuinely empty.
test -d "$work/tools"
mkdir "$work/target"
docker run "${args[@]}" --network none -v "$work/target:/target" \
  -v "$work/compiler-cache:/compiler-cache" -e "COMPILER_CACHE_ENABLED=$cache_enabled" \
  "$image" bash /plan/build-in-container.sh
python3 "$control/source.py" "$GITHUB_WORKSPACE/source" "$TARGET" "$out/SOURCE.json"
probe > "$cache_out/TOOLCHAIN-AFTER.json"
cmp "$cache_out/TOOLCHAIN-BEFORE.json" "$cache_out/TOOLCHAIN-AFTER.json"
if [ "$cache_enabled" = true ]; then
  python3 "$control/compiler_cache.py" stats "$cache_out/IDENTITY.json" "${CACHE_MATCHED_KEY:-}" \
    "$cache_out/stats-before.txt" "$cache_out/stats-after.txt" > "$cache_out/STATS.json"
  python3 - "$cache_out/STATS.json" "$work/compiler-cache" "$cache_out/CACHE-SIZE.json" >> "$GITHUB_OUTPUT" <<'PYBLOCK'
import json,subprocess,sys
row=json.load(open(sys.argv[1]));directory=sys.argv[2]
apparent=int(subprocess.check_output(['du','-sb',directory],text=True).split()[0])
allocated=int(subprocess.check_output(['du','-sB1',directory],text=True).split()[0])
json.dump({'configured_max_size':'2G','apparent_bytes':apparent,'allocated_bytes':allocated,'save_needed':row['save_needed']},open(sys.argv[3],'w'),indent=2)
print('save_needed='+str(row['save_needed']).lower())
PYBLOCK
  if [ "${COLLECT_ORACLES:-false}" != true ]; then
    python3 "$control/compiler_cache.py" qualification "$GITHUB_WORKSPACE/dist/$TARGET/capsule" \
      "$cache_out/IDENTITY.json" > "$cache_out/QUALIFICATION.json"
  fi
fi
# Workflow saves only on success after these source and toolchain postchecks.
printf '{"source_pre_post_equal":true,"toolchain_pre_post_equal":true}\n' > "$cache_out/SAVE-ELIGIBLE.json"
