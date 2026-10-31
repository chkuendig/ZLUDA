#!/usr/bin/env python3
"""Validate design descriptors against canonical source; no FFI or code execution."""
import hashlib
import json
from pathlib import Path
import re
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
source_path = ROOT / 'dark_api/src/lib.rs'
source = source_path.read_text()
tables = json.loads((HERE / 'dark-tables.json').read_text())['tables']
seen = set()
records = []
for table in tables:
    raw = uuid.UUID(table['uuid']).bytes
    assert raw not in seen
    seen.add(raw)
    pattern = r'"\{' + re.escape(table['uuid']) + r'\}"\s*=>\s*' + table['name'] + r'\[(\d+)\]\s*\{'
    match = re.search(pattern, source)
    assert match and int(match[1]) == table['native_words'], table['name']
    start = match.end()
    depth, end = 1, start
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    body = source[start:end - 1]
    entries = re.findall(r'\[(\d+)\]\s*=\s*(\w+)', body)
    slots = {int(slot): name for slot, name in entries}
    assert len(slots) == len(entries) and max(slots) < table['native_words']
    size_slots = [slot for slot, name in slots.items() if name == 'SIZE_OF']
    assert size_slots == ([] if table['size_slot'] is None else [table['size_slot']]), table['name']
    returns = {}
    for slot, name in slots.items():
        if name in ['SIZE_OF', 'NULL']:
            continue
        signature = re.search(r'\b' + name + r'\s*\(', body)
        nesting, position = 1, signature.end()
        while nesting:
            nesting += (body[position] == '(') - (body[position] == ')')
            position += 1
        returns[slot] = re.match(r'\s*->\s*(\(\)|[\w:]+)', body[position:])[1]
    if table['name'] == 'CUDART_INTERFACE':
        assert returns[7] == '()'
    if table['name'] == 'CONTEXT_CHECKS':
        assert returns[3] == 'u32' and returns[2].endswith('CUresult')
    records.append(dict(table, raw_uuid_hex=raw.hex(), slots=slots, returns=returns,
                        reserved_null_slots=sorted(set(range(table['native_words'])) - slots.keys())))
print(json.dumps({'scope': 'Native source layout design check; no PE implementation qualified',
                  'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  'tables': records}, indent=2))
