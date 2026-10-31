#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Admit only reviewer-attested prepared Wine bytes; no source fragments."""
import hashlib
import json
import os
from pathlib import Path
import sys

def main():
    receipt_path, sdk = map(Path, sys.argv[1:])
    expected = os.environ.get('ZLUDA_WINE_ABI_RECEIPT_SHA256', '')
    raw = receipt_path.read_bytes()
    if len(expected) != 64 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('reviewer receipt digest must be explicitly pinned')
    r = json.loads(raw)
    if r['contract'] != 'zluda-wine-prepared-abi-2' or r['wine_pin'] != '46b29104e3741fe23bf5e2547196a253aab88c89':
        raise ValueError('wrong prepared-Wine contract')
    for flag in ['callback_id_zero', 'same_thread_destructor', 'process_exit', 'thread_exit',
                 'loader_lock_refusal', 'locally_scoped_strong_provider', 'eager_binding']:
        if r['verified'].get(flag) is not True:
            raise ValueError('reviewer did not admit ' + flag)
    required = {'include/ntuser.h', 'include/winternl.h', 'include/winnt.h',
                'dlls/user32/user_main.c', 'dlls/ntdll/signal_x86_64.c',
                'dlls/ntdll/unix/signal_x86_64.c', 'dlls/ntdll/loader.c',
                'dlls/ntdll/unix/thread.c', 'dlls/ntdll/unix/virtual.c',
                'dlls/ntdll/unix/unix_private.h'}
    if not required <= r['prepared_files'].keys() or 'lib/ntdll.so' not in r['sdk_files']:
        raise ValueError('incomplete prepared-source or strong-provider extent')
    # Source identities are reviewer attestations bound by the explicitly
    # pinned receipt digest. Production never opens those Wine source files.
    for records in [r['prepared_files'], r['sdk_files']]:
        for name, digest in records.items():
            path = Path(name)
            if not path.parts or path.is_absolute() or '..' in path.parts:
                raise ValueError('invalid receipt member')
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('invalid receipt digest')
    if not sdk.is_absolute():
        raise ValueError('prepared SDK root must be absolute')
    for name, digest in r['sdk_files'].items():
        path = Path(name)
        if path.parts[0] in {'include', 'include-generated'} or path.suffix in {'.c', '.h', '.idl', '.rs'}:
            raise ValueError('SDK source belongs in reviewer attestation metadata, not production inputs')
        print('cargo:rerun-if-changed=' + str(sdk / path))
        if hashlib.sha256((sdk / path).read_bytes()).hexdigest() != digest:
            raise ValueError('prepared SDK bytes differ: ' + name)

if __name__ == '__main__':
    main()
