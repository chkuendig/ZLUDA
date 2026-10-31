#!/usr/bin/env python3
"""Watchdog actual Wine DLL/ExitProcess destructors and DLL_THREAD_DETACH."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import shutil
import statistics

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('wine_fixture_runner', HERE/'run-fixtures.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
MODES = ('free-library','exit-process','thread-exit')
QUALIFICATION = 'real-Wine-synchronous-context-destructors-and-TLS-thread-detach'
RESET_EVENTS = {'backend-loaded':1,'cuInit':1,'native-dark-query':1,'context-set':1,
                'context-get':1,'context-reset-begin':1,'context-destructor-native':1,'context-reset-end':1}
EXPECTED_EVENTS = {'free-library':dict(RESET_EVENTS),'exit-process':dict(RESET_EVENTS),
                   'thread-exit':{'backend-loaded':1,'cuInit':1,'native-dark-query':1}}
EXPECTED_PE_EVENTS = {
    'free-library':['detach-begin','loader-worker-refused','same-thread-destructor','detach-returned','free-library-returned'],
    'exit-process':['calling-exit-process','detach-begin','loader-worker-refused','same-thread-destructor','detach-returned'],
    'thread-exit':['tls-first-self-remove','tls-second-self-remove','tls-thread-exit-returned'],
}


def validate_case(mode, code, output, events, pe_events):
    if mode not in MODES or code or 'PASS:' not in output:
        diagnostics=[line for line in output.splitlines() if line.startswith('LIFECYCLE ')]
        raise RuntimeError('lifecycle process failed or timed out: '+mode+
                           '; rc='+str(code)+'; '+'; '.join(diagnostics[-8:]))
    if Counter(events)!=Counter(EXPECTED_EVENTS[mode]):
        raise RuntimeError('lifecycle native event mismatch: '+mode)
    if pe_events!=EXPECTED_PE_EVENTS[mode]:
        raise RuntimeError('lifecycle PE callback order/thread/refcount mismatch: '+mode)


def timing_receipt(code,output):
    rows=[line.removeprefix('TRANSITIONS: ') for line in output.splitlines() if line.startswith('TRANSITIONS: ')]
    if code or 'PASS:' not in output or len(rows)!=1:
        raise RuntimeError('Unix transition timing process failed or timed out')
    value=json.loads(rows[0])
    if (not isinstance(value,dict) or set(value)!={'iterations','frequency_hz','samples_ticks','baseline_ticks'} or
        type(value['iterations']) is not int or value['iterations']!=200000 or
        type(value['frequency_hz']) is not int or value['frequency_hz']<=0 or
        any(not isinstance(value[k],list) or len(value[k])!=11 or
            any(type(v) is not int or v<=0 for v in value[k]) for k in ('samples_ticks','baseline_ticks'))):
        raise RuntimeError('invalid Unix transition timing receipt')
    scale=1e9/value['frequency_hz']/value['iterations']
    return dict(value,median_ns_per_call=statistics.median(value['samples_ticks'])*scale,
        median_windows_baseline_ns=statistics.median(value['baseline_ticks'])*scale,
        paired_delta_ns=[(a-b)*scale for a,b in zip(value['samples_ticks'],value['baseline_ticks'])],
        scope='hosted-CPU-pinned-Wine-noop; full-CUDA-wrapper-and-Deck-W1-W2-pending')


def run(args):
    for name,path in vars(args).items():
        if not isinstance(path,Path) or not path.is_absolute():
            raise ValueError('lifecycle paths must be absolute: '+name)
    if args.prefix.exists() or args.results.exists():
        raise ValueError('lifecycle prefix/results must be fresh')
    for name in ('wine','wineserver','backend','probe','helper','transition'):
        if not getattr(args,name).is_file():
            raise ValueError('missing lifecycle fixture: '+name)
    common.admit_binaries(args,common.architecture(args))
    common.require_no_canonical_module(args.dll_dir)
    def inputs():
        return {'on':common.snapshot(args.on_root,common.ON_FILES),
                'off':common.snapshot(args.off_root,common.GE_STUBS),
                **{name:common.sha(getattr(args,name)) for name in ('backend','probe','helper','transition')}}
    before=inputs()
    if before['off']!=common.GE_STUBS:
        raise ValueError('lifecycle off root differs from pinned GE stubs')
    for output in (args.prefix,args.results):
        if any(output==root or root in output.parents for root in (args.dll_dir,args.on_root,args.off_root)):
            raise ValueError('lifecycle output overlaps immutable tool inputs')
    args.results.mkdir(parents=True)
    manifest={'qualification':'pending-real-Wine-lifecycle','inputs':before,
              'paths':{name:str(path) for name,path in vars(args).items()},'cases':[]}
    def save():
        (args.results/'RESULT.json').write_text(json.dumps(manifest,indent=2)+'\n')
    boot=common.environment(args,'off-unset',args.results/'bootstrap-backend.log')
    try:
        code,output=common.execute([str(args.wine),'wineboot.exe','-u'],boot,120)
        (args.results/'bootstrap.log').write_text(output)
        manifest['bootstrap']={'returncode':code};save()
        if code or (args.results/'bootstrap-backend.log').exists():
            raise RuntimeError('lifecycle off-root bootstrap failed or loaded a backend')
        # Deliberately the same disposable-prefix admission as the direct CPU
        # gate. The independent Proton launcher gate owns no-repair coverage.
        for module,directory in (('x86_64-windows/nvcuda.dll','system32'),('i386-windows/nvcuda.dll','syswow64')):
            target=args.prefix/'drive_c/windows'/directory/'nvcuda.dll'
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() or target.is_symlink():target.unlink()
            shutil.copyfile(args.off_root/module,target)
        for mode in MODES:
            audit=args.results/(mode+'-backend.log')
            pe_log=args.results/(mode+'-pe.log')
            env=common.environment(args,'on',audit)
            windows_path=lambda path:'Z:'+str(path).replace('/','\\')
            command=[str(args.wine),str(args.probe),mode,windows_path(args.helper),windows_path(pe_log)]
            code,output=common.execute(command,env,30)
            (args.results/(mode+'.log')).write_text(output)
            events=audit.read_text().splitlines() if audit.exists() else []
            pe_events=pe_log.read_text().splitlines() if pe_log.exists() else []
            manifest['cases'].append({'mode':mode,'returncode':code,'command':command,
                                      'log':mode+'.log','events':events,'pe_events':pe_events,
                                      'module_root':env['WINEDLLPATH'],'backend_path':env['ZLUDA_WINE_BACKEND_PATH']})
            save();validate_case(mode,code,output,events,pe_events)
        command=[str(args.wine),str(args.transition)]
        audit=args.results/'unix-call-cost-backend.log'
        code,output=common.execute(command,common.environment(args,'on',audit),30)
        (args.results/'unix-call-cost.log').write_text(output)
        manifest['unix_call_cost']=timing_receipt(code,output)
        if audit.exists():
            raise RuntimeError('pure Unix transition timing unexpectedly loaded the CUDA backend')
        save()
        if inputs()!=before:
            raise RuntimeError('lifecycle changed immutable modules/backend/probes')
        for module,directory in (('x86_64-windows/nvcuda.dll','system32'),('i386-windows/nvcuda.dll','syswow64')):
            target=args.prefix/'drive_c/windows'/directory/'nvcuda.dll'
            if target.is_symlink() or common.sha(target)!=common.GE_STUBS[module]:
                raise RuntimeError('lifecycle changed the prefix GE stub')
        manifest['qualification']=QUALIFICATION
    finally:
        manifest['cleanup']=common.cleanup_server(args,boot);save()
        if not manifest['cleanup']['verified_absent']:
            raise RuntimeError('lifecycle wineserver survives or cleanup failed')
    print(QUALIFICATION)


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('wine','wineserver','dll-dir','on-root','off-root','backend','probe','helper','transition','prefix','results'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
