#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Per-pass and per-frame backend call counts from libw1-cuda-count snapshots.

  per_frame.py SNAPSHOTS.jsonl BEGIN:END:FRAMES [BEGIN:END:FRAMES ...]

FRAMES comes from the title's own frame counter for the same pass (the
harness's renderer/frame-count source), never from this tool. Startup is
reported as the counts at the first BEGIN marker.
"""
import json
import sys

rows = {}
for line in open(sys.argv[1]):
    row = json.loads(line)
    rows.setdefault(row['label'], row)
passes = []
for spec in sys.argv[2:]:
    begin, end, frames = spec.split(':')
    b, e, frames = rows[begin], rows[end], int(frames)
    delta = {k: v - b['counts'].get(k, 0) for k, v in e['counts'].items() if v - b['counts'].get(k, 0)}
    passes.append({'pass': f'{begin}..{end}', 'frames': frames,
                   'seconds': (e['t_ns'] - b['t_ns']) / 1e9,
                   'calls': delta, 'per_frame': {k: v / frames for k, v in sorted(delta.items())},
                   'calls_per_frame_total': sum(delta.values()) / frames})
first = rows[sys.argv[2].split(':')[0]] if len(sys.argv) > 2 else None
print(json.dumps({'startup': first['counts'] if first else None,
                  'startup_export_tables': first['export_tables'] if first else None,
                  'passes': passes}, indent=2, sort_keys=True))
