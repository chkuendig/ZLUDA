#!/usr/bin/env bash
# Builds and checks libw1-cuda-count.so against a fake backend (CPU only).
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
out=${1:?output dir}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cc -O2 -shared -fPIC -o "$work/libcuda.so.1" "$here/selftest_lib.c"
cc -O2 -o "$work/driver" "$here/selftest_main.c" -ldl
mkfifo "$work/ctl"
W1C_CTL="$work/ctl" W1C_OUT="$work/snap" LD_PRELOAD="$out/libw1-cuda-count.so" "$work/driver" "$work/libcuda.so.1"
sleep 1
cat "$work"/snap.* > "$out/cuda-count-selftest.jsonl"
python3 - "$out/cuda-count-selftest.jsonl" <<'PY'
import json, sys
rows = {r['label']: r for r in map(json.loads, open(sys.argv[1]))}
d = {k: rows['end']['counts'].get(k, 0) - rows['begin']['counts'].get(k, 0) for k in rows['end']['counts']}
assert d == {'cuInit': 1000, 'cuDriverGetVersion': 250, 'cuGetExportTable': 3}, d
assert 'notCuda' not in rows['end']['counts']
tables = rows['end']['export_tables']
assert tables == {'c693336e000000000000000000000000': 3}, tables
assert 'exit' in rows
print('cuda-count self-test PASS', d)
PY
