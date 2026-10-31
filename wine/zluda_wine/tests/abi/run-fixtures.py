#!/usr/bin/env python3
"""Run one CPU-only PE64 or PE32/WoW64 profile in an isolated A2 Wine tree."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess

GE_STUBS = {
    'x86_64-windows/nvcuda.dll': '0939c5e19c3cce99668cd0f07db99aa2ede810690c37af9c70df273b924fdfd3',
    'i386-windows/nvcuda.dll': '2685d96065ea86c301a3e56f06b61e1e96a22645b7f69e9866721ad2f9f8833f',
}
ON_FILES = set(GE_STUBS) | {'x86_64-unix/nvcuda.so'}
MODES = ('off-unset', 'on', 'off-zero', 'on-missing-backend', 'on-relative-backend', 'off-again')
REQUIRED_EVENTS = {'backend-loaded': 1, 'cuInit': 2, 'native-dark-query': 2,
                   'dark-five-arguments': 1, 'dark-void-argument': 1,
                   'native-pthread-callback': 1, 'unsupported-host-status': 1, 'native-pthread-occupancy': 1,
                   'inline-native-log-callback': 1}
WOW64_EVENTS = {'backend-loaded': 1, 'cuInit': 2, 'sparse-512MiB-pool': 1,
                'bounded-interior-htod': 1, 'bounded-interior-dtoh': 1,
                'pool-released': 1, 'context-released': 1}


def architecture(args):
    wow = getattr(args, 'winearch', 'win64') == 'wow64'
    if getattr(args, 'winearch', 'win64') not in ('win64', 'wow64'):
        raise ValueError('unknown fixture Wine architecture')
    return {'winearch': 'wow64' if wow else 'win64',
            'pe_machine': '0x014c' if wow else '0x8664', 'pe_bits': 32 if wow else 64,
            'native_bits': 64, 'expected_exports': 690 if wow else 788,
            'unix_call_table': '__wine_unix_call_wow64_funcs' if wow else '__wine_unix_call_funcs'}


def admit_binaries(args, arch):
    pe = args.probe.read_bytes()
    if len(pe) < 64 or pe[:2] != b'MZ':
        raise ValueError('fixture probe must be a PE executable')
    offset = struct.unpack_from('<I', pe, 60)[0]
    if offset > len(pe)-26 or pe[offset:offset+4] != b'PE\0\0':
        raise ValueError('fixture probe has an invalid PE header')
    machine = struct.unpack_from('<H', pe, offset+4)[0]
    magic = struct.unpack_from('<H', pe, offset+24)[0]
    if machine != int(arch['pe_machine'],16) or magic != (0x10b if arch['pe_bits']==32 else 0x20b):
        raise ValueError('fixture probe machine does not match --winearch')
    elf = args.backend.read_bytes()[:64]
    if (len(elf)!=64 or elf[:6]!=b'\x7fELF\x02\x01' or
            struct.unpack_from('<HH',elf,16)!=(3,62)):
        raise ValueError('fixture backend must be an ELF64 x86_64 shared library')


def wow64_receipt(output, mode):
    rows = [line.removeprefix('ARCHITECTURE: ') for line in output.splitlines()
            if line.startswith('ARCHITECTURE: ')]
    expected = {'process_machine': '0x014c', 'native_machine': '0x8664',
                'pointer_bits': 32, 'module_machine': '0x014c', 'module_bits': 32,
                'cu_deviceptr_bits': 32, 'exports': 0 if mode.startswith('off') else 690}
    value = json.loads(rows[0]) if len(rows)==1 else None
    if (not isinstance(value,dict) or value!=expected or
            any(type(value[k]) is not type(v) for k,v in expected.items())):
        raise RuntimeError('missing/mismatched real PE32/WoW64 architecture receipt')
    return expected


def wow64_witness(path, args):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('missing/nonregular ELF64 WoW64 witness')
    value = json.loads(path.read_text())
    expected = {'native_bits': 64, 'unixlib_path': str(args.on_root/'x86_64-unix/nvcuda.so'),
                'unixlib_mapped': True, 'i386_unix_cuda_mapped': False, 'unexpected_cuda_mapping': False,
                'pool_reserved_bytes': 512*1024*1024, 'htod_offset': 260, 'dtoh_offset': 260,
                'pool_released': True, 'context_released': True}
    if (not isinstance(value,dict) or set(value) != set(expected)|{'native_pid','pool_native_base'} or
            any(value[k]!=v or type(value[k]) is not type(v) for k,v in expected.items())):
        raise RuntimeError('invalid ELF64 WoW64 table/pool translation witness')
    if (type(value['native_pid']) is not int or value['native_pid']<=0 or
            type(value['pool_native_base']) is not int or value['pool_native_base']<=0xffffffff):
        raise RuntimeError('WoW64 backend did not witness a full-width native process/address')
    return value


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root, expected):
    files = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('fixture module roots must contain regular files: ' + str(path))
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha(path)
    if set(files) != set(expected):
        raise ValueError('fixture module root has unexpected/missing files: ' + str(root))
    return files


def require_no_canonical_module(dll_dir):
    for arch in ('x86_64-windows', 'i386-windows', 'x86_64-unix', 'i386-unix'):
        for name in ('nvcuda.dll', 'nvcuda.dll.so', 'nvcuda.so'):
            path = dll_dir / arch / name
            if path.exists() or path.is_symlink():
                raise ValueError('canonical dll_dir still contains nvcuda: ' + str(path))


def environment(args, mode, audit):
    env = dict(os.environ)
    for name in ('PROTON_ENABLE_ZLUDA', 'ZLUDA_WINE_BACKEND_PATH', 'ZLUDA_WINE_BRIDGE',
                 'ZLUDA_FIXTURE_FAIL', 'ZLUDA_WINE_FIXTURE_WITNESS', 'ZLUDA_WINE_FIXTURE_UNIXLIB'):
        env.pop(name, None)
    enabled = mode.startswith('on')
    env.update(WINEPREFIX=str(args.prefix), WINEARCH=architecture(args)['winearch'], WINEDEBUG='-all',
               WINEDLLPATH=str(args.on_root if enabled else args.off_root),
               WINEDLLOVERRIDES='nvcuda=b;winemenubuilder.exe=d',
               ZLUDA_WINE_FIXTURE_AUDIT=str(audit))
    if architecture(args)['winearch']=='wow64':
        env.update(ZLUDA_WINE_FIXTURE_WITNESS=str(audit.with_suffix('.json')),
                   ZLUDA_WINE_FIXTURE_UNIXLIB=str(args.on_root/'x86_64-unix/nvcuda.so'))
    if mode == 'off-zero':
        env['PROTON_ENABLE_ZLUDA'] = '0'
    elif enabled:
        env['PROTON_ENABLE_ZLUDA'] = '1'
        if mode == 'on':
            env['ZLUDA_WINE_BACKEND_PATH'] = str(args.backend)
        elif mode == 'on-relative-backend':
            env['ZLUDA_WINE_BACKEND_PATH'] = args.backend.name
    return env


def execute(command, env, timeout):
    try:
        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout)
        return result.returncode, result.stdout
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b''
        if isinstance(output, bytes):
            output = output.decode(errors='replace')
        return 124, output + '\nFIXTURE TIMEOUT\n'


def verify_server_lock_absent(prefix, server_base=None):
    """Check Wine's prefix dev/inode lock directly; stale files are harmless.

    Wine 46b29104 server/request.c:625-785 uses a POSIX write lock on byte 0
    of this file. -k returns 1 for both no owner and a silent F_GETLK failure;
    -w ignores F_SETLKW failure, so exit codes alone do not prove shutdown.
    Never create, truncate, unlink or take a blocking lock during this audit.
    """
    try:
        prefix_info = prefix.stat()
    except FileNotFoundError:
        return {'absent': True, 'reason': 'prefix-absent'}
    if not stat.S_ISDIR(prefix_info.st_mode) or prefix_info.st_uid != os.getuid():
        raise RuntimeError('cleanup prefix is not a directory owned by this uid')
    base = server_base if server_base is not None else Path('/tmp') / ('.wine-' + str(os.getuid()))
    server = base / f'server-{prefix_info.st_dev:x}-{prefix_info.st_ino:x}'
    result = {'path': str(server/'lock'), 'prefix_device': prefix_info.st_dev,
              'prefix_inode': prefix_info.st_ino, 'absent': False}
    directories = []
    try:
        for directory in (base, Path(server.name)):
            try:
                descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                     dir_fd=directories[-1] if directories else None)
            except FileNotFoundError:
                return dict(result, absent=True, reason='server-directory-absent')
            directories.append(descriptor)
            if os.fstat(descriptor).st_uid != os.getuid():
                raise RuntimeError('cleanup server directory is not owned by this uid: ' + str(directory))
        try:
            descriptor = os.open('lock', os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directories[-1])
        except FileNotFoundError:
            return dict(result, absent=True, reason='lock-file-absent')
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise RuntimeError('cleanup server lock is not a regular file owned by this uid')
            try:
                fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, 0, os.SEEK_SET)
            except BlockingIOError:
                return dict(result, reason='server-lock-still-held')
            fcntl.lockf(descriptor, fcntl.LOCK_UN, 1, 0, os.SEEK_SET)
            return dict(result, absent=True, reason='server-lock-unheld')
        finally:
            os.close(descriptor)
    finally:
        for descriptor in reversed(directories):
            os.close(descriptor)


def cleanup_server(args, env):
    result = {'steps': [], 'verified_absent': False}
    for option, log in (('-k','wineserver-cleanup.log'), ('-w','wineserver-wait.log'),
                        ('-k0','wineserver-verify.log')):
        command = [str(args.wineserver), option]
        code, output = execute(command, env, 10)
        (args.results/log).write_text(output)
        # -k0 is a signal-0 probe: zero means an owner is still alive.
        accepted = (code in (0, 1) and (code == 0 or not output.strip())) if option == '-k' else (
            code == 0 if option == '-w' else code == 1 and not output.strip())
        result['steps'].append({'command': command, 'returncode': code, 'log': log, 'accepted': accepted})
    try:
        result['lock'] = verify_server_lock_absent(args.prefix)
    except (OSError, RuntimeError) as error:
        result['lock'] = {'absent': False, 'error': str(error)}
    result['verified_absent'] = all(step['accepted'] for step in result['steps']) and result['lock']['absent']
    return result


def run(args):
    for name, path in vars(args).items():
        if isinstance(path, Path) and not path.is_absolute():
            raise ValueError(name + ' must be absolute')
    if args.prefix.exists() or args.results.exists():
        raise ValueError('fixture prefix and results must be fresh')
    for name in ('wine','wineserver','backend','probe'):
        if not getattr(args,name).is_file():
            raise ValueError('missing fixture input: '+name)
    arch = architecture(args)
    admit_binaries(args, arch)
    if not args.dll_dir.is_dir():
        raise ValueError('missing prepared Wine dll_dir')
    require_no_canonical_module(args.dll_dir)
    before = {'on': snapshot(args.on_root, ON_FILES), 'off': snapshot(args.off_root, GE_STUBS),
              'backend': sha(args.backend), 'probe': sha(args.probe)}
    if before['off'] != GE_STUBS:
        raise ValueError('off root differs from the exact pinned GE stubs')
    for output in (args.prefix,args.results):
        if any(output==root or root in output.parents for root in (args.on_root,args.off_root,args.dll_dir)):
            raise ValueError('fixture output must be outside immutable module roots')
    args.results.mkdir(parents=True)
    manifest = {'qualification': 'pending-real-Wine-execution', 'architecture': arch, 'inputs': before,
                'paths': {key: str(value) for key, value in vars(args).items()}, 'cases': []}
    sources = Path(__file__).resolve().parent
    manifest['fixture_sources'] = {name: sha(sources/name) for name in (
        'backend.c','probe.c','backend-wow64.c','probe32.c','../wow64/backend.c',
        '../../api/wow64-backend-abi.tsv','../../api/wow64-required.txt','build-fixtures.sh','run-fixtures.py')}
    result_path = args.results/'RESULT.json'

    def save():
        result_path.write_text(json.dumps(manifest, indent=2) + '\n')

    boot_env = environment(args, 'off-unset', args.results/'bootstrap-backend.log')
    try:
        command = [str(args.wine), 'wineboot.exe', '-u']
        code, output = execute(command, boot_env, 120)
        (args.results/'bootstrap.log').write_text(output)
        manifest['bootstrap'] = {'command': command, 'returncode': code}
        save()
        if code or (args.results/'bootstrap-backend.log').exists() or (args.results/'bootstrap-backend.json').exists():
            raise RuntimeError('off-root Wine bootstrap failed or loaded the backend')
        # A2 launcher bootstrap uses regular GE stubs in this disposable prefix.
        # Wine's marked PE loader rebinds these same bytes through the selected
        # module root for each new process; never swap on-root bytes here.
        for module, directory in (('x86_64-windows/nvcuda.dll','system32'),
                                  ('i386-windows/nvcuda.dll','syswow64')):
            target = args.prefix/'drive_c/windows'/directory/'nvcuda.dll'
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            shutil.copyfile(args.off_root/module,target)
        for mode in MODES:
            audit = args.results/(mode+'-backend.log')
            env = environment(args,mode,audit)
            command = [str(args.wine),str(args.probe)]
            if mode.startswith('off'):
                command.append('--expect-stub')
            elif mode != 'on':
                command.append('--expect-backend-refusal')
            code, output = execute(command,env,90)
            (args.results/(mode+'.log')).write_text(output)
            events = audit.read_text().splitlines() if audit.exists() else []
            manifest['cases'].append({'mode': mode, 'command': command, 'returncode': code,
                                      'module_root': env['WINEDLLPATH'],
                                      'backend_path': env.get('ZLUDA_WINE_BACKEND_PATH'), 'events': events})
            save()
            if code or 'PASS:' not in output:
                raise RuntimeError('PE ABI fixture failed: '+mode)
            if arch['winearch']=='wow64':
                manifest['cases'][-1]['architecture'] = wow64_receipt(output, mode)
            if mode == 'on':
                expected = WOW64_EVENTS if arch['winearch']=='wow64' else REQUIRED_EVENTS
                if Counter(events) != Counter(expected):
                    raise RuntimeError('unexpected/missing fake-backend event: '+repr(events))
                if arch['winearch']=='wow64':
                    manifest['cases'][-1]['wow64_witness'] = wow64_witness(audit.with_suffix('.json'),args)
            elif events:
                raise RuntimeError('off/refused case loaded or called the backend: '+mode)
            elif audit.with_suffix('.json').exists():
                raise RuntimeError('off/refused case produced a backend witness: '+mode)
            save()
        require_no_canonical_module(args.dll_dir)
        after = {'on': snapshot(args.on_root, ON_FILES), 'off': snapshot(args.off_root, GE_STUBS),
                 'backend': sha(args.backend), 'probe': sha(args.probe)}
        if before != after:
            raise RuntimeError('fixture changed immutable input files')
        for module, directory in (('x86_64-windows/nvcuda.dll','system32'),
                                  ('i386-windows/nvcuda.dll','syswow64')):
            target = args.prefix/'drive_c/windows'/directory/'nvcuda.dll'
            if target.is_symlink() or sha(target) != GE_STUBS[module]:
                raise RuntimeError('fixture changed a prefix stub or made it a symlink')
        manifest['qualification'] = ('CPU-Wine-PE32-WoW64-pass; GPU-W1-W2-pending'
            if arch['winearch']=='wow64' else 'CPU-Wine-PE64-pass; GPU-W1-W2-pending')
    finally:
        manifest['cleanup'] = cleanup_server(args, boot_env)
        save()
        if not manifest['cleanup']['verified_absent']:
            raise RuntimeError('isolated wineserver cleanup failed or server survives')
    print(manifest['qualification'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--winearch', choices=('win64','wow64'), default='win64')
    for argument in ('wine','wineserver','dll-dir','on-root','off-root','backend','probe','prefix','results'):
        parser.add_argument('--'+argument, type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
