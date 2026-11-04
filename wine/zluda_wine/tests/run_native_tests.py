#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Run non-oracle Rust unit tests and the independent ELF provider-scope fixture."""
import os
from pathlib import Path
import subprocess
import tempfile

crate=Path(__file__).resolve().parents[1]
subprocess.run(['cargo','+1.99.0','test','--locked','-j','3','--lib',
                '--manifest-path',str(crate/'Cargo.toml'),'--','--test-threads=1'],check=True)
with tempfile.TemporaryDirectory(prefix='zluda-native-tests-') as tmp:
    out=Path(tmp)
    provider=crate/'tests/abi/callback-loader-provider.c'
    consumer=crate/'tests/abi/callback-loader-consumer.c'
    common=['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC']
    for name,extra in [('callback-provider.so',[]),('callback-empty-provider.so',['-DOMIT_CALLBACK'])]:
        subprocess.run(common+['-Wl,-soname,zluda-callback-fixture-ntdll.so']+extra+[str(provider),'-o',str(out/name)],check=True)
    subprocess.run(common+['-Wl,-z,defs',str(consumer),str(out/'callback-provider.so'),'-o',str(out/'callback-consumer.so')],check=True)
    harness=out/'scope.rs'
    harness.write_text('mod unix_context_storage { #[path='+__import__('json').dumps(str(crate/'src/unix_context_storage/loader_tests.rs'))+'] mod loader_tests; }\n')
    binary=out/'scope-tests'
    subprocess.run(['rustup','run','1.99.0','rustc','--edition=2021','--test',str(harness),'-l','dl','-o',str(binary)],env=dict(os.environ,OUT_DIR=str(out)),check=True)
    subprocess.run([str(binary),'--test-threads=1'],check=True)
