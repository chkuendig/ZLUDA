#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check the clean overlay and emit bounded, complete Cargo diagnostics."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

TARGETS = ('x86_64-unknown-linux-gnu', 'x86_64-pc-windows-gnu', 'i686-pc-windows-gnu')
OUTPUT_LIMIT = 60_000


def files(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlinks are not accepted')
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if any(part in {'.git', 'target', '__pycache__'} for part in path.relative_to(root).parts):
                raise ValueError('unexpected build or version-control payload')
            result[name] = path
    return result


def candidate_files(root):
    result = files(root)
    for name in result:
        if not name.startswith('wine/zluda_wine/') or '/tests/' in name:
            raise ValueError('candidate must contain production bridge files only')
        if not (name.endswith(('.rs', '.toml', '.lock', '.tsv', '.json', '.txt', '.py', '.c', '.h', '.sh', '.md', '.spec'))):
            raise ValueError('unexpected candidate file type')
    if not result:
        raise ValueError('candidate is empty')
    return result


def run(command, cwd=None):
    # Spool output within the disposable container, keeping memory bounded.
    # The only source mounts are clean input, production candidate and cache.
    with tempfile.TemporaryFile() as output:
        timed_out = False
        try:
            process = subprocess.run(command, cwd=cwd, stdout=output,
                                     stderr=subprocess.STDOUT, timeout=1200)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
        output.seek(0, os.SEEK_END)
        total = output.tell()
        output.seek(0)
        if total <= OUTPUT_LIMIT:
            shown = output.read()
        else:
            head = output.read(OUTPUT_LIMIT // 2)
            output.seek(-(OUTPUT_LIMIT // 2), os.SEEK_END)
            tail = output.read()
            omitted = total - len(head) - len(tail)
            marker = f'\n... [{omitted} output bytes omitted; head and tail retained] ...\n'.encode()
            shown = head + marker + tail
    return {'returncode': returncode, 'timed_out': timed_out,
            'output_bytes': total, 'truncated': total > OUTPUT_LIMIT,
            'output': shown.decode('utf-8', errors='replace')}


def report(phase, command, cwd=None, target=None):
    result = run(command, cwd)
    receipt = {'phase': phase, 'command': command, 'target': target,
               'status': 'passed' if result['returncode'] == 0 else 'failed',
               'exit_code': result['returncode'], 'timed_out': result['timed_out'],
               'output_bytes': result['output_bytes'], 'truncated': result['truncated'],
               'output_limit_bytes': OUTPUT_LIMIT}
    print(json.dumps(receipt), flush=True)
    label = f'{phase} {target}' if target else phase
    print(f'--- {label}: combined stdout/stderr ---', flush=True)
    print(result['output'], end='' if result['output'].endswith('\n') else '\n', flush=True)
    print(f'--- end {label} ---', flush=True)
    return result['returncode']


def main():
    source = Path('/work/source')
    source.mkdir(parents=True, exist_ok=False)
    candidate = candidate_files(Path('/candidate'))
    for entries in (files(Path('/input')), candidate):
        for name, path in entries.items():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    if any('/tests/' in p for p in files(source)):
        raise ValueError('quick-check source contains tests')
    os.environ.update(CARGO_HOME='/cache/cargo', CARGO_TARGET_DIR='/work/target',
                      CARGO_BUILD_JOBS='3', CARGO_TERM_COLOR='never',
                      ZLUDA_WINE_DECLARATIONS_ONLY='1')
    # Toolchain preparation is part of the separately cleared single container.
    setup = [('rustup', 'component', 'add', 'clippy'),
             ('rustup', 'target', 'add', *TARGETS)]
    for command in setup:
        if report('toolchain', command):
            return 2
    manifest = source / 'wine/zluda_wine/Cargo.toml'
    if report('dependency-cache', ['cargo', 'fetch', '--locked', '--manifest-path', str(manifest), '-vv']):
        return 2
    failed = False
    for target in TARGETS:
        for action in ('check', 'clippy'):
            command = ['cargo', action, '--locked', '--offline', '--lib', '--target', target,
                       '--manifest-path', str(manifest), '--message-format=human', '-vv']
            failed |= report(action, command, manifest.parent, target) != 0
    return int(failed)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({'phase': 'quick-check', 'status': 'failed', 'details': str(error)}), flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.exit(2)
