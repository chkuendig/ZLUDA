#!/usr/bin/env python3
"""Lifecycle/provenance integration for the native reader and production producer.

The shim is a cdylib including impl/recording.rs, exporting recording_init and
recording_submit, with a .fini_array callback to recording::shutdown. No HIP/GPU
work is needed for these reader/lifecycle cases. Actual CUDA-path counting and
the production library's dlclose remain separate integration checks.
"""
import argparse
import ctypes as C
import _ctypes
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

BUILD = '2b591706924dd0d88aa82a12827caf26fc79a417'
CASES = ('live', 'unload-reload', 'daemon-stall', 'replace-record',
         'modify-library', 'duplicate', 'wrong-hash', 'wrong-build',
         'wrong-app', 'stale-start', 'stale-only', 'partial-publication', 'process-death')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def wait_until(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise RuntimeError('condition timed out')


def case(args):
    with tempfile.TemporaryDirectory(prefix='zluda-reader-test-', dir='/dev/shm') as temp:
        root = Path(temp)
        library = root/'producer.so'
        shutil.copyfile(args.shim, library)
        os.environ.update(XDG_RUNTIME_DIR=temp, SteamAppId='123', ZLUDA_RECORDING='1')
        lib = C.CDLL(str(library))
        lib.recording_init()
        directory = root/'zluda-recording'
        record, = directory.glob('*.rec')
        original = record.name
        digest = hashlib.sha256(library.read_bytes()).hexdigest()
        server = None
        socket = root/'reader.sock'
        auxiliary = None
        crashed = None

        def start(pid=None, build=BUILD, app='123', sha=digest):
            nonlocal server
            executable = Path(sys.executable).name
            # /proc cmdline's argv[0] may be python3 rather than the resolved binary.
            if pid is None:
                executable = Path(Path('/proc/self/cmdline').read_bytes().split(b'\0')[0].decode()).name
            server = subprocess.Popen([str(args.reader), 'serve', str(pid or os.getpid()), app,
                build, executable, str(library), sha, str(socket), temp],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

        def get():
            return subprocess.check_output([str(args.reader), 'get', str(socket)],
                                           text=True, timeout=1).strip()

        def ready():
            wait_until(lambda: socket.exists() or server.poll() is not None)
            if server.poll() is not None:
                raise RuntimeError(server.stderr.read())
            wait_until(lambda: get() != 'unavailable')

        def rejected(expected=0):
            wait_until(lambda: server.poll() is not None)
            require(server.returncode == expected, f'unexpected reader exit {server.returncode}: '+server.stderr.read())
            require(get() == 'unavailable', 'invalid source was displayed as idle')

        try:
            if args.child == 'wrong-hash':
                start(sha='0'*64); rejected(2)
            elif args.child == 'wrong-build':
                start(build='f'*40); rejected(2)
            elif args.child == 'wrong-app':
                start(app='124'); rejected(2)
            elif args.child == 'stale-only':
                stale = bytearray(record.read_bytes())
                ticks = int.from_bytes(stale[24:32], 'little')
                stale[24:32] = (ticks-1).to_bytes(8, 'little')
                stale_file = directory/'stale.rec'; stale_file.write_bytes(stale); stale_file.chmod(0o600)
                record.unlink()
                start(); rejected(2)
            elif args.child == 'process-death':
                # A separate producer process dies without running its destructor.
                crashed = subprocess.Popen([sys.executable, __file__, '--reader', str(args.reader),
                    '--shim', str(library), '--crash-producer'], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, text=True, env=os.environ.copy())
                require(crashed.stdout.readline().strip() == 'READY', 'producer failed')
                start(pid=crashed.pid); ready()
                crashed.kill(); crashed.wait(timeout=3)
                rejected()
                require(len(list(directory.glob('*.rec'))) == 2, 'crash leftover not retained')
            else:
                if args.child == 'stale-start':
                    stale = bytearray(record.read_bytes())
                    ticks = int.from_bytes(stale[24:32], 'little')
                    stale[24:32] = (ticks-1).to_bytes(8, 'little')
                    stale_file = directory/'stale.rec'; stale_file.write_bytes(stale); stale_file.chmod(0o600)
                elif args.child == 'partial-publication':
                    (directory/'.pending-test').write_bytes(b'incomplete')
                start(); ready()
                require(get() == '0.0', 'valid idle instance did not report zero')
                if args.child == 'live':
                    for _ in range(1000): lib.recording_submit()
                    def positive():
                        value = get()
                        return value != 'unavailable' and float(value) > 0
                    wait_until(positive)
                elif args.child == 'unload-reload':
                    _ctypes.dlclose(lib._handle); lib = None
                    rejected(); require(not record.exists(), 'dlclose did not remove instance')
                    lib = C.CDLL(str(library)); lib.recording_init()
                    replacement, = directory.glob('*.rec')
                    require(replacement.name != original, 'reload reused instance')
                    start(); ready(); require(get() == '0.0', 'reload did not reset count')
                elif args.child == 'daemon-stall':
                    server.send_signal(signal.SIGSTOP)
                    try:
                        begin = time.monotonic(); require(get() == 'unavailable', 'stalled daemon returned old value')
                        require(time.monotonic()-begin < .5, 'stalled daemon blocked adapter')
                        time.sleep(2.1)
                    finally: server.send_signal(signal.SIGCONT)
                    require(get() == 'unavailable', 'long sample gap accepted')
                    wait_until(lambda: get() == '0.0')
                elif args.child == 'replace-record':
                    replacement = directory/'replacement'; replacement.write_bytes(record.read_bytes()); replacement.chmod(0o600)
                    replacement.replace(record); rejected()
                elif args.child == 'modify-library':
                    # Change only file metadata: loaded executable bytes remain intact.
                    stat = library.stat(); os.utime(library, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000))
                    rejected()
                elif args.child == 'duplicate':
                    duplicate = root/'duplicate.so'; shutil.copyfile(library, duplicate)
                    auxiliary = C.CDLL(str(duplicate)); auxiliary.recording_init(); rejected()
                elif args.child not in ('stale-start', 'partial-publication'):
                    raise RuntimeError('unknown case')
            print('PASS reader '+args.child, flush=True)
        finally:
            if server and server.poll() is None:
                server.terminate(); server.wait(timeout=3)
            if server:
                expected = 2 if args.child in ('wrong-hash','wrong-build','wrong-app','stale-only') else 0
                require(server.returncode == expected, f'reader exit {server.returncode}: '+server.stderr.read())
            if crashed and crashed.poll() is None: crashed.kill(); crashed.wait(timeout=3)
            if auxiliary: _ctypes.dlclose(auxiliary._handle)
            if lib: _ctypes.dlclose(lib._handle)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reader', type=Path, required=True)
    p.add_argument('--shim', type=Path, required=True)
    p.add_argument('--child', choices=CASES)
    p.add_argument('--crash-producer', action='store_true')
    args = p.parse_args()
    if args.crash_producer:
        lib = C.CDLL(str(args.shim)); lib.recording_init()
        print('READY', flush=True); sys.stdin.read(); return
    if args.child:
        case(args); return
    env = {k:v for k,v in os.environ.items() if not k.startswith('ZLUDA_') and k not in ('LD_PRELOAD', 'PYTHONOPTIMIZE')}
    for name in CASES:
        subprocess.run([sys.executable, __file__, '--reader', str(args.reader.resolve()),
            '--shim', str(args.shim.resolve()), '--child', name], env=env, check=True, timeout=15)
    print('PASS reader suite: 13 isolated cases', flush=True)


if __name__ == '__main__': main()
