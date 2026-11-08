"""Compiler-cache identity and reproducibility receipts; never an admission gate substitute."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def identity(plan, source, toolchain, target, branch):
    plan, source = Path(plan), Path(source)
    pins, images = load(plan / 'pins.json'), load(plan / 'images.json')
    pin = pins[target]
    if not pin['llvm_built']:
        raise ValueError('compiler-object cache requires an LLVM-building target')
    for name in ('compiler_binaries', 'compiler_versions', 'packages', 'rustc', 'cmake', 'ninja', 'ccache'):
        if not toolchain.get(name):
            raise ValueError('missing exact toolchain identity: ' + name)
    # Native source HEAD is deliberately absent: ccache hashes each source/header.
    # The LLVM source, configuration/build script and exact compiler inputs are pinned.
    if branch != 'refs/heads/ci/p400-capsules-20261107':
        raise ValueError('cache branch is not the admitted producer branch')
    value = {
        'schema': 1,
        'branch': branch,
        'llvm': pin['llvm_gitlink'],
        'llvm_build_rs_blob': pin['llvm_build_rs_blob'],
        'llvm_manifest_blob': pin['llvm_manifest_blob'],
        'cargo_lock_blob': pin['cargo_lock_blob'],
        'images': images,
        'toolchain': toolchain,
        'profile': 'release' if target == 'zoc' else 'test-debug',
        'configuration_sha256': {name: sha(plan / name) for name in (
            'cloud-build.sh', 'build-in-container.sh', 'compiler_cache.py')},
        'rocm_setup_sha256': sha(source / '.github/workflows/rocm_setup_build.sh'),
        'source_build_rs_sha256': sha(source / 'llvm_zluda/build.rs'),
        'source_llvm_manifest_sha256': sha(source / 'llvm_zluda/Cargo.toml'),
        'policy': {'cache': 'C/C++ compiler objects only', 'target_initially_empty': True,
                   'cargo_incremental': False, 'compiler_check': 'content',
                   'source_mount': '/src', 'build_mount': '/target', 'cache_dir': '/compiler-cache'},
    }
    digest = hashlib.sha256(encoded(value)).hexdigest()
    return {'namespace': 'p400-llvm-objects-v1-' + branch + '-' + digest + '-', 'identity': value}


def probe_toolchain():
    """Executed only inside the prepared cloud image, before/after the build."""
    def command(*argv):
        return subprocess.check_output(argv, text=True, timeout=60).strip()
    binaries, versions = {}, {}
    for name in ('cc', 'c++'):
        path = Path(shutil.which(name)).resolve(strict=True)
        binaries[str(path)] = sha(path)
        versions[name] = command(name, '--version')
    for tool, executable in (('cc', 'cc1'), ('c++', 'cc1plus')):
        path = Path(command(tool, '-print-prog-name=' + executable)).resolve(strict=True)
        binaries[str(path)] = sha(path)
    return {'compiler_binaries': binaries, 'compiler_versions': versions,
            'packages': command('dpkg-query', '-W'),
            'rustc': command('rustc', '+1.99.0', '-vV'),
            'cmake': command('cmake', '--version'), 'ninja': command('ninja', '--version'),
            'ccache': {'version': command('ccache', '--version'),
                       'binary_sha256': sha(Path(shutil.which('ccache')).resolve(strict=True))},
            'environment': {name: os.environ.get(name) for name in (
                'CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'RUSTFLAGS')},
            'configuration': {'compiler_check': 'content', 'sloppiness': '',
                              'hard_link': False, 'file_clone': False,
                              'max_size': '2G', 'cargo_build_jobs': 4, 'cmake_parallel': 4}}


def stats(text):
    result = {}
    for line in text.splitlines():
        key, value = line.split()
        if key in result:
            raise ValueError('duplicate cache statistic')
        result[key] = int(value)
    return result


def cache_receipt(cache_identity, matched_key, before, after):
    prefix = cache_identity['namespace']
    if matched_key and not matched_key.startswith(prefix):
        raise ValueError('restored cache key is outside the complete branch-bound namespace')
    old, new = stats(before), stats(after)
    counters = ('direct_cache_hit', 'preprocessed_cache_hit', 'cache_miss')
    if any(name not in old or name not in new for name in counters):
        raise ValueError('required compiler cache counters missing')
    delta = {key: new[key] - old[key] for key in counters}
    if any(value < 0 for value in delta.values()):
        raise ValueError('compiler cache counters went backwards')
    return {'namespace': prefix, 'cache_matched_key': matched_key or None,
            'restored': bool(matched_key), 'save_needed': not matched_key or delta['cache_miss'] > 0,
            'counters_before': old, 'counters_after': new,
            'delta': delta, 'total_hits': delta['direct_cache_hit'] + delta['preprocessed_cache_hit'],
            'scope': 'C/C++ objects only; Cargo target was independently empty'}


def qualification(capsule, cache_identity):
    capsule = Path(capsule)
    manifest = load(capsule / 'MANIFEST.json')
    freshness = manifest['freshness']
    if (freshness.get('target_initially_empty') is not True or freshness.get('fresh_true') != 0
            or freshness.get('LLVM_BUILT') is not True or not freshness.get('llvm_build_script_events')
            or not freshness.get('llvm_compiled_build_script')):
        raise ValueError('missing actual fresh LLVM build evidence')
    if manifest['source'].get('source_clean') is not True:
        raise ValueError('source is not clean')
    expected = manifest['executables']
    if not expected or any(row.get('fresh') is not False for row in expected.values()):
        raise ValueError('expected fresh executable artifacts')
    for name, row in expected.items():
        if sha(capsule / name) != row['sha256']:
            raise ValueError('executable differs from qualified manifest')
    runtime = manifest['runtime']
    if not runtime:
        raise ValueError('missing runtime closure')
    for name, row in runtime.items():
        if sha(capsule / 'runtime/lib' / name) != row['sha256']:
            raise ValueError('runtime differs from qualified manifest')
    receipt_hashes = {}
    # Compare raw producer receipts, not a weaker reformulation of CPU outcomes.
    for name, result in manifest['cpu'].items():
        if not result.get('results') or any(row.get('returncode') != 0 or row.get('one_test_passed') is not True
                                           for row in result['results']):
            raise ValueError('CPU qualification missing/failed')
        path = capsule / 'receipts' / ('cpu-' + name) / 'RESULT.json'
        if load(path) != result:
            raise ValueError('CPU receipt does not match manifest')
        receipt_hashes['cpu-' + name] = sha(path)
    if manifest['target'] == 'jc2':
        suite = manifest['ptx_cpu_suite']
        if (suite.get('passed') is not True or suite.get('returncode') != 0
                or suite['counts']['failed'] != 0 or suite['counts']['passed'] <= 0):
            raise ValueError('full PTX CPU suite failed/missing')
        path = capsule / 'receipts/cpu-ptx-full/RESULT.json'
        if load(path) != suite:
            raise ValueError('full PTX receipt does not match manifest')
        receipt_hashes['cpu-ptx-full'] = sha(path)
    if manifest['target'] == 'zoc':
        receipt_hashes['zoc-smoke'] = sha(capsule / 'receipts/zoc-smoke/RESULT.json')
    for name in ('SOURCE.json', 'receipts/FRESHNESS.json'):
        receipt_hashes[name] = sha(capsule / name)
    payload = {}
    for name, row in manifest['files'].items():
        if name.split('/', 1)[0] in ('bin', 'lib', 'runtime') and row['kind'] == 'file':
            if sha(capsule / name) != row['sha256']:
                raise ValueError('payload executable/DSO differs from manifest')
            payload[name] = row['sha256']
    return {'schema': 1, 'payload_files_sha256': payload, 'target': manifest['target'], 'source': manifest['source'],
            'cache_namespace': cache_identity['namespace'],
            'producer_receipts_sha256': receipt_hashes,
            'executables': {name: row['sha256'] for name, row in sorted(expected.items())},
            'runtime': runtime, 'loader': manifest['loader']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('probe')
    stamp = commands.add_parser('timestamp')
    stamp.add_argument('phase')
    report = commands.add_parser('stats')
    report.add_argument('cache_identity', type=Path)
    report.add_argument('matched_key')
    report.add_argument('before', type=Path)
    report.add_argument('after', type=Path)
    key = commands.add_parser('identity')
    key.add_argument('plan', type=Path)
    key.add_argument('source', type=Path)
    key.add_argument('toolchain', type=Path)
    key.add_argument('target')
    key.add_argument('branch')
    check = commands.add_parser('qualification')
    check.add_argument('capsule', type=Path)
    check.add_argument('cache_identity', type=Path)
    args = parser.parse_args()
    if args.command == 'timestamp':
        result = {'phase': args.phase, 'monotonic': time.monotonic(), 'epoch': time.time()}
    elif args.command == 'probe':
        result = probe_toolchain()
    elif args.command == 'stats':
        result = cache_receipt(load(args.cache_identity), args.matched_key, args.before.read_text(), args.after.read_text())
    elif args.command == 'identity':
        result = identity(args.plan, args.source, load(args.toolchain), args.target, args.branch)
    else:
        result = qualification(args.capsule, load(args.cache_identity))
    print(encoded(result).decode(), end='')


if __name__ == '__main__':
    main()
