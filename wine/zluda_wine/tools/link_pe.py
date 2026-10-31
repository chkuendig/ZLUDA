#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Link a PE with the pinned export contract, excluding Rust/DllMain helpers."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
from pe_artifact import validate

HERE = Path(__file__).resolve().parents[1]


def definition(triplet):
    rows = [line.split('\t') for line in (HERE / 'api/exports.tsv').read_text().splitlines()
            if line and not line.startswith('#')]
    lines = ['LIBRARY nvcuda', 'EXPORTS']
    for ordinal, row in enumerate(rows, 1):
        if triplet == 'x86_64':
            # Wine 46b29104 tools/winebuild/parser.c:880-945 assigns @ entries
            # in spec source order, starting at one. Preserve those ordinals.
            lines.append(f' {row[0]} @{row[2]}')
        elif row[1] == 'stdcall':
            # Same decorated symbols/--kill-at route as the original C PE32.
            lines.append(f' {row[0]}@{row[3]} @{row[4]}')
    expected = 788 if triplet == 'x86_64' else 690
    assert len(lines) - 2 == expected
    return '\n'.join(lines) + '\n'


def native_libraries(path):
    log = re.sub(r'\x1b\[[0-9;]*m', '', path.read_text())
    values = set(re.findall(r'native-static-libs:\s*([^\n]+)', log))
    if len(values) != 1:
        raise ValueError('expected one unambiguous rustc native-static-libs result')
    libraries = shlex.split(values.pop())
    if not libraries or any(not re.fullmatch(r'-l[A-Za-z0-9_.:+-]+|-Wl,[A-Za-z0-9_,.=+-]+', value)
                            for value in libraries):
        raise ValueError('unexpected native static-link argument; inspect toolchain output')
    return libraries


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--triplet', choices=['x86_64', 'i686'], required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--native-libs-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.archive.is_file():
        raise ValueError('Rust static library is missing')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exports = args.archive.with_suffix('.exports.def')
    exports.write_text(definition(args.triplet))
    command = [args.triplet + '-w64-mingw32-gcc', '-shared', '-static-libgcc', '-Wl,--exclude-all-symbols',
               '-Wl,--gc-sections', str(exports), '-Wl,--whole-archive', str(args.archive),
               '-Wl,--no-whole-archive']
    if args.triplet == 'i686':
        command.append('-Wl,--kill-at')
    command += (['-luser32'] if args.triplet == 'x86_64' else [])
    command += native_libraries(args.native_libs_log) + ['-o', str(args.output)]
    if args.dry_run:
        print(json.dumps(command))
    else:
        subprocess.run(command, check=True)
        validate(args.output, args.triplet, builtin=False)


if __name__ == '__main__':
    main()
