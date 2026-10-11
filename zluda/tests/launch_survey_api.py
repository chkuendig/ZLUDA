#!/usr/bin/env python3
"""Native survey hook/count gate. Host timeout is enforced by the caller."""
import argparse,ctypes as C,hashlib,json,os,selectors,subprocess,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import null_stream_api as fixture

def require(value,message):
 if not value:raise RuntimeError(message)

def main():
 p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--reader',type=Path,required=True);p.add_argument('--build',required=True);args=p.parse_args()
 require(os.environ.get('ZLUDA_LAUNCH_SURVEY')=='1','survey opt-in absent')
 require(not any(k.startswith('ZLUDA_') and k!='ZLUDA_LAUNCH_SURVEY' for k in os.environ),'unclean ZLUDA environment')
 require('PYTHONOPTIMIZE' not in os.environ and 'LD_PRELOAD' not in os.environ,'unclean process')
 library=args.library.resolve();digest=hashlib.sha256(library.read_bytes()).hexdigest();executable=Path(Path('/proc/self/cmdline').read_bytes().split(b'\0')[0].decode()).name
 temp=os.environ['XDG_RUNTIME_DIR'];os.environ['SteamAppId']='0'
 server=None;selector=selectors.DefaultSelector();last=None
 def total():
  nonlocal server,last
  if server is None:
   server=subprocess.Popen([str(args.reader),'survey',str(os.getpid()),'0',args.build,executable,str(library),digest,temp],stdout=subprocess.PIPE,text=True,bufsize=1)
   selector.register(server.stdout,selectors.EVENT_READ)
  # Wait for a newly collected cumulative snapshot rather than reusing a cached one.
  deadline=time.monotonic()+5
  while time.monotonic()<deadline:
   if server.poll() is not None:raise RuntimeError('survey reader stopped')
   if selector.select(.2):
    line=server.stdout.readline()
    if line:last=json.loads(line);return last
  raise RuntimeError('survey snapshot deadline')
 try:
  d = fixture.Driver(library, primary_context=True)
  require(total()['total_success'] == 3, 'warmup launch count differs')
  flag = d.alloc()
  for _ in range(7): d.launch('produce', None, flag)
  for _ in range(11): d.launch('produce', None, flag, ptsz=True)
  streams = [d.stream(1), d.stream(1)]
  barrier = threading.Barrier(2)

  def worker(stream):
      d.call('cuCtxSetCurrent', [C.c_void_p], d.context)
      try:
          barrier.wait(timeout=10)
          for i in range(40): d.launch('produce', stream, flag, ptsz=bool(i%2))
          d.sync(stream)
      finally: d.call('cuCtxSetCurrent', [C.c_void_p], None)

  with ThreadPoolExecutor(max_workers=2) as pool:
      list(pool.map(worker, streams))
  d.sync(None)
  require(total()['total_success'] == 101, 'normal/ptsz/concurrent launches were not counted exactly once')
  require(d.read(flag) == 1, 'produce kernel output wrong')
  # A zero grid is rejected by HIP without submitting a kernel.
  fn = d.lib.cuLaunchKernel
  fn.argtypes = [C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p]
  fn.restype = C.c_int
  value = C.c_uint64(flag.value)
  params = (C.c_void_p*1)(C.cast(C.byref(value), C.c_void_p).value)
  failed = fn(d.functions['produce'],0,1,1,1,1,1,0,None,params,None)
  require(failed != 0, 'expected a rejected zero-grid launch')
  require(total()['total_success'] == 101, 'failed HIP launch changed counter')
  final=total()
  require(final['total_failure']==1,'failed HIP return was not counted')
  require(final['dropped']==0 and final['unknown']==0,'survey has missing coverage')
  rows=final['rows']
  require(sum(r['success'] for r in rows if r['name']=='produce')==99,'produce success count differs')
  require(sum(r['failure'] for r in rows if r['name']=='produce')==1,'produce failure count differs')
  for name in ('poll','step'):require(sum(r['success'] for r in rows if r['name']==name)==1,'warmup name missing')
  require(all(r['tls_context']==d.context.value for r in rows),'TLS context identity differs')
  d.close()
  print('PASS survey CUDA: 101 successes, one HIP rejection, normal/ptsz/concurrent, exact named counts',flush=True)
 finally:
  if server is not None:
   server.terminate();server.wait(timeout=5);require(server.returncode==0,'reader exit rejected')
  selector.close()
if __name__=='__main__':main()
