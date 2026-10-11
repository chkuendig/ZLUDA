#!/usr/bin/env python3
"""Native survey reader/shim lifecycle checks; no GPU work.
The shim must compile exact launch_survey.rs and export survey_init/observe,
with its shutdown callback in .fini_array. Production CUDA counting is separate.
"""
import argparse,ctypes as C,_ctypes,hashlib,json,os,select,shutil,subprocess,sys,tempfile,time,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
CASES=('live','unload','replace','duplicate','modify-library','wrong-hash','wrong-build','pending','malformed-row','stale-only','concurrent-publication','process-death')
def require(value,message):
 if not value:raise RuntimeError(message)
def case(a):
 with tempfile.TemporaryDirectory(prefix='survey-reader-',dir='/dev/shm') as temp:
  root=Path(temp);library=root/'producer.so';shutil.copyfile(a.shim,library)
  os.environ.update(XDG_RUNTIME_DIR=temp,SteamAppId='123',ZLUDA_LAUNCH_SURVEY='1')
  lib=C.CDLL(str(library));lib.survey_init();server=None;aux=None
  directory=root/'zluda-launch-survey';record,=directory.glob('*.survey')
  try:
   if a.child=='malformed-row':
    # Deliberately malformed row is prepared before the reader starts.
    with record.open('r+b') as f:f.seek(4096);f.write(b'x'*384);f.seek(4096+448);f.write((1).to_bytes(8,'little'))
   if a.child=='pending':(directory/'.pending-test').write_bytes(b'incomplete')
   if a.child=='stale-only':
    data=bytearray(record.read_bytes());ticks=int.from_bytes(data[24:32],'little');data[24:32]=(ticks-1).to_bytes(8,'little');record.write_bytes(data)
   digest=hashlib.sha256(library.read_bytes()).hexdigest()
   game=Path(Path('/proc/self/cmdline').read_bytes().split(b'\0')[0].decode()).name
   producer=None;pid=os.getpid()
   if a.child=='process-death':
    producer=subprocess.Popen([sys.executable,__file__,'--reader',str(a.reader),'--shim',str(library),'--build',a.build,'--producer'],stdout=subprocess.PIPE,stdin=subprocess.PIPE,env=os.environ.copy())
    require(select.select([producer.stdout],[],[],3)[0],'producer readiness timeout')
    require(producer.stdout.readline()==b'READY\n','child producer failed');pid=producer.pid
   server=subprocess.Popen([str(a.reader),'survey',str(pid),'123','f'*40 if a.child=='wrong-build' else a.build,game,str(library),'0'*64 if a.child=='wrong-hash' else digest,temp],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   def snapshot(timeout=4):
    ready,_,_=select.select([server.stdout],[],[],max(0,timeout))
    require(ready,'snapshot timeout');line=server.stdout.readline()
    if not line:
     server.wait(timeout=1);raise RuntimeError('reader stopped: '+server.stderr.read().decode())
    return json.loads(line)
   def counted(expected,predicate=lambda snapshot:True):
    deadline=time.monotonic()+4;previous=0
    while time.monotonic()<deadline:
     current=snapshot(deadline-time.monotonic());value=current['total_success']
     require(value>=previous,'counter regressed');previous=value
     if value==expected and predicate(current):return current
    raise RuntimeError('count deadline')
   def ended(expected):
    stdout,stderr=server.communicate(timeout=4);require(server.returncode==expected,f'exit {server.returncode}, expected {expected}: '+stderr.decode())
    # Every published record must be complete JSON, even after invalidation.
    output=stdout.splitlines()
    if a.child in ('wrong-hash','wrong-build','stale-only','malformed-row'):require(not output,'rejected source published snapshots')
    for line in output:json.loads(line)
   if a.child in ('wrong-hash','wrong-build','stale-only','malformed-row'):ended(2)
   else:
    first=snapshot();require(first['total_success']==0 and first['rows']==[],'initial snapshot differs')
    if a.child=='live':
     for _ in range(17):lib.survey_observe()
     current=counted(17);require(current['total_success']==17 and len(current['rows'])==1,'counts differ')
     require(current['rows'][0]['name']=='TurbulenceFS_test' and current['rows'][0]['success']==17,'published name/count differs')
    elif a.child=='concurrent-publication':
     barrier=threading.Barrier(4);lib.survey_observe_id.argtypes=[C.c_uint]
     def publish(worker):
      barrier.wait(timeout=3)
      for i in range(worker*32,(worker+1)*32):lib.survey_observe_id(i)
     with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(publish,range(4)))
     current=counted(128,lambda r:len(r['rows'])==128 and sum(row['success'] for row in r['rows'])==128);require(current['total_success']==128 and len(current['rows'])==128,'concurrent publication differs')
     require(len({r['name'] for r in current['rows']})==128 and all(r['success']==1 for r in current['rows']),'published metadata differs')
    elif a.child=='process-death':
     producer.kill();producer.wait(timeout=3);ended(0)
    elif a.child=='unload':
     _ctypes.dlclose(lib._handle);lib=None;require(not record.exists(),'destructor did not remove file');ended(0)
    elif a.child=='replace':
     replacement=directory/'replacement';replacement.write_bytes(record.read_bytes());replacement.chmod(0o600);replacement.replace(record);ended(2)
    elif a.child=='duplicate':
     duplicate=root/'duplicate.so';shutil.copyfile(library,duplicate);aux=C.CDLL(str(duplicate));aux.survey_init();ended(2)
    elif a.child=='modify-library':
     stat=library.stat();os.utime(library,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000));ended(2)
    elif a.child!='pending':raise RuntimeError('unknown case')
   print('PASS survey reader '+a.child,flush=True)
  finally:
   errors=[]
   if server and server.poll() is None:
    try:
     server.terminate();stdout,stderr=server.communicate(timeout=4)
     require(server.returncode==0,'reader termination failed: '+stderr.decode())
     for line in stdout.splitlines():json.loads(line)
    except Exception as exc:
     errors.append(str(exc))
     if server.poll() is None:
      server.kill();server.communicate(timeout=2)
   if server is not None:
    expected=2 if a.child in ('wrong-hash','wrong-build','stale-only','malformed-row','replace','duplicate','modify-library') else 0
    if server.returncode!=expected:errors.append(f'final reader exit {server.returncode}, expected {expected}')
   try:
    if 'producer' in locals() and producer and producer.poll() is None:producer.kill();producer.communicate(timeout=3)
   except Exception as exc:errors.append(str(exc))
   for owned in (aux,lib):
    try:
     if owned:_ctypes.dlclose(owned._handle)
    except Exception as exc:errors.append(str(exc))
   require(not errors,'cleanup: '+repr(errors))

def main():
 p=argparse.ArgumentParser();p.add_argument('--reader',type=Path,required=True);p.add_argument('--shim',type=Path,required=True);p.add_argument('--build',required=True);p.add_argument('--child',choices=CASES);p.add_argument('--producer',action='store_true');a=p.parse_args()
 if a.producer:
  lib=C.CDLL(str(a.shim));lib.survey_init();print('READY',flush=True);sys.stdin.read();return
 if a.child:case(a);return
 env={k:v for k,v in os.environ.items() if not k.startswith('ZLUDA_') and k not in ('LD_PRELOAD','PYTHONOPTIMIZE')}
 for name in CASES:subprocess.run([sys.executable,__file__,'--reader',str(a.reader.resolve()),'--shim',str(a.shim.resolve()),'--build',a.build,'--child',name],env=env,check=True,timeout=15)
 print('PASS survey reader suite: 12 isolated cases',flush=True)
if __name__=='__main__':main()
