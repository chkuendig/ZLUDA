#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Audit generated PE metadata offline. Never load or execute the module."""
import argparse
import json
import re
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parents[1]
SYSTEM_DLLS = {'kernel32.dll', 'ntdll.dll', 'msvcrt.dll', 'userenv.dll',
              'ws2_32.dll', 'dbghelp.dll', 'advapi32.dll', 'bcryptprimitives.dll', 'ucrtbase.dll', 'user32.dll'}


def inspect(data):
    def fields(fmt, offset):
        if offset < 0 or offset + struct.calcsize(fmt) > len(data):
            raise ValueError('PE metadata outside file')
        return struct.unpack_from(fmt, data, offset)
    if data[:2] != b'MZ':
        raise ValueError('missing DOS signature')
    nt, = fields('<I', 0x3c)
    if data[nt:nt+4] != b'PE\0\0':
        raise ValueError('missing PE signature')
    machine, section_count, _, _, _, optional_size, flags = fields('<HHIIIHH', nt+4)
    optional = nt+24
    magic, = fields('<H', optional)
    directory = optional + {0x10b: 96, 0x20b: 112}.get(magic, 0)
    if directory == optional or not flags & 0x2000:
        raise ValueError('expected PE DLL optional header')
    # Respect the declared optional-header directory extent. A short header
    # must not hide imports by borrowing bytes from the section table.
    count, = fields('<I', directory-4)
    if not 2 <= count <= 16 or directory+count*8 > optional+optional_size:
        raise ValueError('invalid PE directory extent')
    if count > 13 and any(fields('<II', directory+13*8)):
        # These MinGW builds have no delay-load machinery. Reject it outright
        # instead of leaving an unaudited dependency outside the normal list.
        raise ValueError('unexpected PE delayed import directory')
    sections = [fields('<8sIIII', optional+optional_size+i*40) for i in range(section_count)]

    def offset(rva, size=1):
        for _, _, virtual, length, start in sections:
            if virtual <= rva and rva+size <= virtual+length:
                result = start+rva-virtual
                if result+size <= len(data):
                    return result
        raise ValueError('PE RVA outside file-backed sections')

    def string(rva):
        start = offset(rva)
        end = data.find(b'\0', start, min(len(data), start+1024))
        if end < 0:
            raise ValueError('unterminated PE metadata string')
        return data[start:end].decode('ascii')

    export, export_size = fields('<II', directory)
    if not export or export_size < 40:
        raise ValueError('missing PE export table')
    base, functions, names, addresses, name_rvas, ordinals = fields('<IIIIII', offset(export,40)+16)
    if not 0 < names <= functions <= 65536:
        raise ValueError('invalid PE export counts')
    exports = []
    for i in range(names):
        name, = fields('<I', offset(name_rvas+4*i,4))
        ordinal, = fields('<H', offset(ordinals+2*i,2))
        if ordinal >= functions:
            raise ValueError('export ordinal outside address table')
        function, = fields('<I', offset(addresses+4*ordinal,4))
        if not function or export <= function < export+export_size:
            raise ValueError('missing or forwarded CUDA export')
        offset(function)
        exports.append((base+ordinal, string(name)))
    if len({name for _, name in exports}) != names:
        raise ValueError('duplicate PE export name')
    imports = []
    import_rva, import_size = fields('<II', directory+8)
    if import_rva:
        if import_size < 20:
            raise ValueError('invalid PE import directory')
        for i in range(import_size//20):
            entry = fields('<IIIII', offset(import_rva+20*i,20))
            if not any(entry):
                break
            imports.append(string(entry[3]))
        else:
            raise ValueError('unterminated PE import descriptors')
    return {'machine': machine, 'magic': magic, 'builtin': data[0x40:0x50] == b'Wine builtin DLL',
            'function_count': functions, 'exports': sorted(exports), 'imports': imports}


def validate(path, triplet, builtin=True, oracle=None):
    record = inspect(path.read_bytes())
    machine, magic = (0x8664, 0x20b) if triplet == 'x86_64' else (0x14c, 0x10b)
    if (record['machine'], record['magic']) != (machine, magic) or (builtin and not record['builtin']):
        raise ValueError('PE machine/format/builtin contract mismatch')
    rows = [line.split('\t') for line in (oracle or HERE/'api/exports.tsv').read_text().splitlines()
            if line and not line.startswith('#')]
    if triplet == 'x86_64':
        expected = [(int(row[2]), row[0]) for row in rows]
    else:
        expected = [(int(row[4]), row[0]) for row in rows if row[1]=='stdcall']
    if record['exports'] != sorted(expected) or record['function_count'] != len(expected):
        raise ValueError('PE CUDA export names/ordinals differ from pinned C contract')
    unexpected = [name for name in record['imports'] if name.lower() not in SYSTEM_DLLS
                  and not re.fullmatch(r'(?:api|ext)-ms-win-[a-z0-9-]+\.dll', name.lower())]
    if unexpected:
        raise ValueError('unexpected PE runtime dependency: '+', '.join(unexpected))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--triplet', required=True, choices=['x86_64','i686'])
    parser.add_argument('module', type=Path)
    args = parser.parse_args()
    result = validate(args.module,args.triplet)
    print(json.dumps({key: value for key,value in result.items() if key!='exports'},sort_keys=True))
