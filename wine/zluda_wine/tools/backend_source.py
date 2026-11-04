# SPDX-License-Identifier: MIT OR Apache-2.0
"""Resolve and byte-admit the explicitly pinned backend source closure."""
import hashlib
import json
import os
from pathlib import Path


def backend_source():
    value = os.environ.get('ZLUDA_WINE_BACKEND_SOURCE')
    if not value or not Path(value).is_absolute():
        raise ValueError('ZLUDA_WINE_BACKEND_SOURCE must name an absolute pinned backend source root')
    root = Path(value).resolve(strict=True)
    contract = json.loads((Path(__file__).resolve().parents[3] / 'BACKEND-SOURCE.json').read_text())
    for name, expected in contract['source_inputs'].items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned backend source differs: ' + name)
    return root
