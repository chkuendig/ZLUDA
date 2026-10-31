#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Verify the reviewed source closure before removing reviewer source trees."""
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[3]
crate = root / 'wine/zluda_wine'
manifest = json.loads((crate / 'LICENSES/PRODUCTION-INPUTS.json').read_text())
for name, digest in manifest['source_inputs'].items():
    path = root / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit('Production input changed; review its provenance: ' + name)
    source = path.read_text()
    for forbidden in ['wine/nvcuda', 'tests/lgpl', 'C-ORACLE.json', 'api/nvcuda.spec',
                      'api/c-dark.tsv', 'api/wine-callback-abi.json', 'api/d3d11-abi.json',
                      'C-cuda.h', 'C-encryption.h', 'dark-oracle.h']:
        if forbidden in source:
            raise SystemExit('Reference input entered production closure: ' + name)
    if 'SPDX-License-Identifier: LGPL' in source:
        raise SystemExit('LGPL source entered production closure: ' + name)
includes = re.findall(r'^\s*#include\s*[<"]([^>"]+)', (crate / 'wine_adapter.c').read_text(), re.M)
if includes != ['stdint.h', 'stddef.h']:
    raise SystemExit('Re-review the C adapter header closure')
if 'WINE_SOURCE_ROOT' in (crate / 'tools/admit_wine.py').read_text():
    raise SystemExit('Production admission regained a Wine source dependency')
print(json.dumps({'production_input_closure': 'passed',
                  'files': len(manifest['source_inputs']), 'adapter_headers': includes}))
