#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Preflight a reviewed, crate-relative production overlay before importing it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

PREFIX = 'wine/zluda_wine/'


def plan(candidate, crate, previous):
    allowed = {name.removeprefix(PREFIX) for name in json.loads(
        (crate / 'LICENSES/PRODUCTION-INPUTS.json').read_text())['source_inputs']
        if name.startswith(PREFIX)}
    changes = []
    for path in sorted(candidate.rglob('*')):
        if path.is_symlink():
            raise ValueError('candidate symlink refused')
        if not path.is_file():
            continue
        name = path.relative_to(candidate).as_posix()
        if path.suffix == '.md' or name not in allowed or any(
                part in {'.git', 'tests', 'target', 'LICENSES', '__pycache__'}
                for part in Path(name).parts):
            raise ValueError('candidate path is outside the reviewed production allowlist: ' + name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # Unchanged scaffold bytes must not undo subsequent license/build fixes.
        if previous.get(PREFIX + name) != digest:
            changes.append((name, path, digest))
    if not changes:
        raise ValueError('no changed production input')
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-crate', required=True, type=Path,
                        help='candidate/wine/zluda_wine, never the document-bearing handoff root')
    parser.add_argument('--previous-receipt', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    crate = Path(__file__).resolve().parents[1]
    previous = json.loads(args.previous_receipt.read_text())['candidate_hashes']
    changes = plan(args.candidate_crate, crate, previous)
    if args.apply:
        for name, source, digest in changes:
            target = crate / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    print(json.dumps({'action': 'imported' if args.apply else 'preflight',
                      'changed_production_inputs': {name: digest for name, _, digest in changes}}, indent=2))


if __name__ == '__main__':
    main()
