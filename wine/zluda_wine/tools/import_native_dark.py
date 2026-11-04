#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Describe and verify only the canonical permissive Rust dark signatures."""
import argparse
import hashlib
from pathlib import Path
import re

HERE = Path(__file__).resolve().parents[1]


def close(text, start, opening, closing):
    depth, position = 1, start
    while depth:
        depth += (text[position] == opening) - (text[position] == closing)
        position += 1
    return position


def capture():
    path = HERE.parents[1] / 'dark_api/src/lib.rs'
    original = path.read_text()
    # The selected CUDA declarations have no comment-like string contents.
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', original, flags=re.S)
    records = []
    for match in re.finditer(r'"\{([0-9A-Fa-f-]+)\}"\s*=>\s*(\w+)\[(\d+)\]\s*\{', text):
        uuid, table, words = match[1].replace('-', '').lower(), match[2], int(match[3])
        if table == 'ZLUDA_TRACE_INTERNAL':
            continue
        body = text[match.end():close(text, match.end(), '{', '}') - 1]
        slots = {}
        for entry in re.finditer(r'\[(\d+)\]\s*=\s*(\w+)', body):
            index, name = int(entry[1]), entry[2]
            assert index not in slots and index < words
            if name in ('SIZE_OF', 'NULL'):
                signature = '-'
            else:
                position = entry.end()
                while body[position].isspace():
                    position += 1
                assert body[position] == '('
                end = close(body, position + 1, '(', ')')
                ret = re.match(r'\s*->\s*(\(\)|[\w:]+)', body[end:])
                assert ret, name
                signature = f'fn {name}(' + ' '.join(body[position + 1:end - 1].split()) + ') -> ' + ret[1]
            evidence = re.search(r'\[' + str(index) + r'\]\s*=\s*' + re.escape(name) + r'\b',
                                 original[original.index('"{' + match[1]):])
            line = original[:original.index('"{' + match[1]) + evidence.start()].count('\n') + 1
            slots[index] = (name, signature, line)
        for index in range(words):
            name, signature, line = slots.get(index, ('NULL', '-', 0))
            records.append([uuid, table, str(words), str(index), name, signature, str(line)])
    assert len({row[1] for row in records}) == 11
    table = '# SPDX-License-Identifier: MIT OR Apache-2.0\n# Canonical dark_api/src/lib.rs SHA256 ' + hashlib.sha256(path.read_bytes()).hexdigest() + '\n'
    table += '# uuid-raw-hex\ttable\tnative-words\tslot\tentry\tRust-signature\tsource-line\n'
    table += ''.join('\t'.join(row) + '\n' for row in records)
    return {'native-dark.tsv': table.encode()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for name, data in capture().items():
        target = HERE / 'api' / name
        if args.check:
            if target.read_bytes() != data:
                raise SystemExit('Canonical dark contract changed: ' + name)
        else:
            target.write_bytes(data)
    print('Canonical dark signatures captured: 11 tables, reserved slots explicit')
