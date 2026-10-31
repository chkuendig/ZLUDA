#!/usr/bin/env python3
"""One local retained PTX module, serial zoc compiles on a guarded1MiB stack.

Run only under a separately cleared exact container command. No PTX bytes are
copied, printed, committed or uploaded. The compiler performs no GPU launch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(value):
    expected={'reserve_bytes','low_guard_bytes','upper_guard_bytes','usable_bytes',
              'touched_high_water_bytes','rounded_observed_bound_bytes',
              'initial_tid','compile_tid','same_thread','main_status','seed'}
    if (not isinstance(value,dict) or set(value)!=expected or value['same_thread'] is not True or
        any(type(value[k]) is not int for k in expected-{'same_thread'}) or
        value['main_status']!=0 or value['initial_tid']<=0 or value['initial_tid']!=value['compile_tid'] or
        value['reserve_bytes']!=1048576 or value['low_guard_bytes']!=4096 or value['upper_guard_bytes']!=4096 or
        value['usable_bytes']!=1048576-value['low_guard_bytes'] or
        not 0<value['touched_high_water_bytes']<=value['rounded_observed_bound_bytes']<=value['usable_bytes'] or
        value['rounded_observed_bound_bytes']%4096 or
        not 128<=value['rounded_observed_bound_bytes']-value['touched_high_water_bytes']<4224):
        raise RuntimeError('invalid same-thread guarded-stack receipt')
    return value


def run(args):
    for key,path in vars(args).items():
        if isinstance(path,Path) and (not path.is_absolute() or path.is_symlink()):
            raise ValueError('use absolute nonsymlink inputs/outputs: '+key)
    if args.results.exists():raise ValueError('results directory must be fresh')
    manifest=json.loads(args.corpus_manifest.read_text())
    candidates=[m for m in manifest['modules'] if m['sha256']==args.module_sha256]
    if len(candidates)!=1:raise ValueError('module must have exactly one pinned corpus entry')
    module=candidates[0];ptx=Path(module['path'])
    if not ptx.is_absolute() or ptx.is_symlink() or sha(ptx)!=module['sha256'] or ptx.stat().st_size!=module['bytes']:
        raise ValueError('retained PTX module hash/size/path mismatch')
    args.results.mkdir(parents=True)
    result={'qualification':'pending-real-zoc-measurement','arch':args.arch,'module':module,
            'inputs':{name:sha(getattr(args,name)) for name in ('zoc','loader','preload','corpus_manifest')},'cases':[]}
    def save():(args.results/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    for mode,seed in [('calibration',91),('guard-overflow',91),('compile',91),('compile',37),('compile',203)]:
        name=mode+'-'+str(seed);report=args.results/(name+'.json');output=args.results/(name+'.log')
        products=args.results/(name+'-products');products.mkdir()
        env={key:value for key,value in os.environ.items() if key not in ('LD_PRELOAD','LD_LIBRARY_PATH','ZLUDA_STACK_PROBE_SELFTEST')}
        env.update(ZLUDA_STACK_PROBE_REPORT=str(report),ZLUDA_STACK_PROBE_SEED=str(seed))
        if mode!='compile':env['ZLUDA_STACK_PROBE_SELFTEST']='overflow' if mode=='guard-overflow' else 'calibration'
        command=[str(args.loader),'--library-path',str(args.library_dir),'--preload',str(args.preload),
                 str(args.zoc),'--arch',args.arch,'--output-dir',str(products),str(ptx)]
        try:
            with output.open('w') as stream:
                code=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900).returncode
        except subprocess.TimeoutExpired:code=124
        case={'mode':mode,'seed':seed,'command':command,'returncode':code,'log':output.name}
        result['cases'].append(case);save()
        if mode=='guard-overflow':
            if code!=125 or 'STACK_PROBE_GUARD_HIT' not in output.read_text() or report.exists():
                raise RuntimeError('stack guard selftest did not fault at its guard')
            continue
        if code or not report.is_file():raise RuntimeError('real compiler/calibration failed; headroom unqualified')
        value=validate(json.loads(report.read_text()));case['measurement']=value
        if value['seed']!=seed:raise RuntimeError('stack paint seed did not match the requested run')
        if mode=='calibration' and not 65536<=value['touched_high_water_bytes']<=131072:
            raise RuntimeError('stack paint failed known-depth calibration')
        if mode=='compile':
            elves=list(products.glob('*.elf'))
            if len(elves)!=1 or not elves[0].is_file():raise RuntimeError('zoc did not complete its ELF codegen/link pipeline')
            case['products']={p.name:sha(p) for p in products.iterdir() if p.is_file()}
        save()
    observed=max(c['measurement']['rounded_observed_bound_bytes'] for c in result['cases'] if c['mode']=='compile')
    #12KiB remains reserved for Wine's min_kernel_stack; the real CUDA wrapper
    # and loader frames are not established by this compiler-only measurement.
    available=1048576-4096-12288
    result.update(qualification='real-zoc-pipeline-stack-measured; Wine-cuModuleLoad-W1-pending',
        max_rounded_observed_bound_bytes=observed,conservative_comparison_budget_bytes=available,
        pipeline_headroom_ratio=available/observed,proposed_stack_override=None)
    save();print(result['qualification'])


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('zoc','loader','library-dir','preload','corpus-manifest','results'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--module-sha256',required=True)
    parser.add_argument('--arch',choices=('gfx1030','gfx1033'),default='gfx1030')
    run(parser.parse_args())


if __name__=='__main__':main()
