#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate target declarations, then run the real PE32 frontend on Windows."""
import os
from pathlib import Path
import subprocess

crate=Path(__file__).resolve().parents[1]
target='i686-pc-windows-msvc'
subprocess.run(['cargo','+1.99.0','check','--locked','-j','3','--lib','--target',target,
                '--manifest-path',str(crate/'Cargo.toml')],check=True)
target_dir=Path(os.environ.get('CARGO_TARGET_DIR',str(crate/'target')))
outputs=list((target_dir/target/'debug/build').glob('*/out/cuda_types.rs'))
assert len(outputs)==1,outputs
environment=dict(os.environ,OUT_DIR=str(outputs[0].parent))
binary=target_dir/'pe32-luid-probe.exe'
subprocess.run(['rustup','run','1.99.0','rustc','--edition=2021','--target',target,
                '-C','opt-level=1',str(crate/'tests/pe32_luid_probe.rs'),'-o',str(binary)],env=environment,check=True)
subprocess.run([str(binary)],check=True)
