#!/usr/bin/env python3
"""Native full-field descriptor checks. No kernels; run under host timeout."""
import argparse
import ctypes as C
import hashlib,json,sys
from pathlib import Path
V=C.c_void_p;U=C.c_uint;S=C.c_size_t
class A2(C.Structure):
 _fields_=[('width',S),('height',S),('format',U),('channels',U)]
class A3(C.Structure):
 _fields_=[('width',S),('height',S),('depth',S),('format',U),('channels',U),('flags',U)]
parser=argparse.ArgumentParser();parser.add_argument('library',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--expect-mipmap-unsupported',action='store_true');args=parser.parse_args()
libpath=args.library.resolve();out=args.output;out.mkdir()
lib=C.CDLL(str(libpath));checks=[];negative=[];mip_unsupported=[];ctx=V();failure=None;cleanup=None
result={'success':False,'library_sha256':hashlib.sha256(libpath.read_bytes()).hexdigest()}
def call(name,types,*args):
 fn=getattr(lib,name);fn.argtypes=types;fn.restype=C.c_int;rc=fn(*args)
 if rc:raise RuntimeError((name,rc))
def check(desc,got,label):
 expected=[getattr(desc,k) for k,_ in desc._fields_];actual=[getattr(got,k) for k,_ in got._fields_]
 checks.append({'case':label,'expected':expected,'actual':actual})
 if expected!=actual:raise RuntimeError((label,expected,actual))
def rejected(name,kind,arr,null_output):
 fn=getattr(lib,name);fn.argtypes=[C.POINTER(kind),V];fn.restype=C.c_int
 got=kind();C.memset(C.byref(got),0xa5,C.sizeof(got));before=bytes(got)
 rc=fn(None if null_output else C.byref(got),arr)
 negative.append({'api':name,'null_output':null_output,'returncode':rc,'output_unchanged':bytes(got)==before})
 if rc==0 or bytes(got)!=before:raise RuntimeError(('invalid query accepted',name,rc))
try:
 call('cuInit',[U],0);call('cuCtxCreate_v2',[C.POINTER(V),U,C.c_int],C.byref(ctx),0,0)
 for fmt in (16,9,8,32,3):
  for ch in (1,4):
   for flags in (0,2):
    arr=V();desc=A3(16,8,4,fmt,ch,flags);got=A3()
    call('cuArray3DCreate_v2',[C.POINTER(V),C.POINTER(A3)],C.byref(arr),C.byref(desc))
    call('cuArray3DGetDescriptor_v2',[C.POINTER(A3),V],C.byref(got),arr);check(desc,got,f'3D/{fmt}/{ch}/{flags}')
    rejected('cuArray3DGetDescriptor_v2',A3,arr,True);call('cuArrayDestroy',[V],arr)
   arr=V();desc=A2(16,8,fmt,ch);got=A2()
   call('cuArrayCreate_v2',[C.POINTER(V),C.POINTER(A2)],C.byref(arr),C.byref(desc))
   call('cuArrayGetDescriptor_v2',[C.POINTER(A2),V],C.byref(got),arr);check(desc,got,f'2D/{fmt}/{ch}')
   rejected('cuArrayGetDescriptor_v2',A2,arr,True);call('cuArrayDestroy',[V],arr)
   for flags in (0,2):
    mip=V();desc3=A3(16,8,0,fmt,ch,flags)
    fn=lib.cuMipmappedArrayCreate;fn.argtypes=[C.POINTER(V),C.POINTER(A3),U];fn.restype=C.c_int
    rc=fn(C.byref(mip),C.byref(desc3),3)
    if args.expect_mipmap_unsupported:
     mip_unsupported.append({'format':fmt,'channels':ch,'flags':flags,'returncode':rc,'handle':mip.value})
     if rc!=801 or mip.value:raise RuntimeError(('expected target mipmap limitation',rc,mip.value))
     continue
    if rc:raise RuntimeError(('cuMipmappedArrayCreate',rc))
    for repeat in range(2):
     for level in range(3):
      arr=V();call('cuMipmappedArrayGetLevel',[C.POINTER(V),V,U],C.byref(arr),mip,level)
      got=A3();call('cuArray3DGetDescriptor_v2',[C.POINTER(A3),V],C.byref(got),arr)
      check(A3(16>>level,8>>level,0,fmt,ch,flags),got,f'mip/{fmt}/{ch}/{flags}/{repeat}/{level}')
    call('cuMipmappedArrayDestroy',[V],mip)
 for name,kind in [('cuArrayGetDescriptor_v2',A2),('cuArray3DGetDescriptor_v2',A3)]:rejected(name,kind,None,False)
 if len(checks)!=(30 if args.expect_mipmap_unsupported else 150) or len(negative)!=32 or len(mip_unsupported)!=(20 if args.expect_mipmap_unsupported else 0):raise RuntimeError(('wrong check count',len(checks),len(negative)))
except BaseException as e:failure=repr(e)
finally:
 try:
  if ctx.value:call('cuCtxDestroy_v2',[V],ctx)
  cleanup='ok'
 except BaseException as e:cleanup=repr(e);failure=failure or cleanup
 (out/'maps.txt').write_text(Path('/proc/self/maps').read_text())
 result.update(success=failure is None,checks=checks,negative=negative,mip_unsupported=mip_unsupported,cleanup=cleanup,error=failure)
 (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if failure:raise SystemExit(failure)
print(f'PASS array descriptor: {len(checks)} complete field comparisons; 32 negative queries; {len(mip_unsupported)} unsupported mipmap controls; cleanup ok',flush=True)
