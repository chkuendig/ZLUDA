#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare two fixture JSONL outputs (control vs candidate).

  compare.py CONTROL.jsonl CANDIDATE.jsonl

Drops volatile fields (names starting with "v_", plus "arch" only if equal)
and compares the remaining check/event records in order. Exit 0 when equal,
1 with a list of differences otherwise. Equal outputs mean the candidate
returns the same statuses, values and pass/fail results as the control; a
check that fails on both is still a failure for the gate.
"""
import json
import sys


def records(path):
    out = []
    for line in open(path, errors='replace'):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append({k: v for k, v in row.items() if not k.startswith('v_')})
    return out


def key(row):
    return row.get('check') or row.get('event') or '?'


control, candidate = records(sys.argv[1]), records(sys.argv[2])
diffs = []
for i in range(max(len(control), len(candidate))):
    a = control[i] if i < len(control) else None
    b = candidate[i] if i < len(candidate) else None
    if a != b:
        diffs.append({'index': i, 'control': a, 'candidate': b})
for d in diffs:
    print(json.dumps(d, sort_keys=True))
print(json.dumps({'control_records': len(control), 'candidate_records': len(candidate),
                  'differences': len(diffs)}))
sys.exit(1 if diffs else 0)
