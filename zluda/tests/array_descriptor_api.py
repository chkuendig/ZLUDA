#!/usr/bin/env python3
"""Native descriptor round trips; library and output path are explicit.
Run in a clean environment with a host timeout. Does not launch kernels.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
V=C.c_void_p;U=C.c_uint;S=C.c_size_t
class A2(C.Structure):
 _fields_=[('width',S),('height',S),('format',U),('channels',U)]
class A3(C.Structure):
 _fields_=[('width',S),('height',S),('depth',S),('format',U),('channels',U),('flags',U)]
libpath=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]);out.mkdir()
lib=C.CDLL(str(libpath));checks=[]
def call(name,types,*args):
 fn=getattr(lib,name);fn.argtypes=types;fn.restype=C.c_int;rc=fn(*args)
 if rc:raise RuntimeError((name,rc))
def check(desc,got,label):
 expected=[getattr(desc,k) for k,_ in desc._fields_];actual=[getattr(got,k) for k,_ in got._fields_]
 checks.append({'case':label,'expected':expected,'actual':actual})
 if expected!=actual:raise RuntimeError((label,expected,actual))
call('cuInit',[U],0);ctx=V();call('cuCtxCreate_v2',[C.POINTER(V),U,C.c_int],C.byref(ctx),0,0)
success=False
try:
 for fmt in (16,9,8,32,3):
  for ch in (1,4):
   for flags in (0,2):
    arr=V();desc=A3(16,8,4,fmt,ch,flags);got=A3()
    call('cuArray3DCreate_v2',[C.POINTER(V),C.POINTER(A3)],C.byref(arr),C.byref(desc))
    call('cuArray3DGetDescriptor_v2',[C.POINTER(A3),V],C.byref(got),arr);check(desc,got,f'3D/{fmt}/{ch}/{flags}')
    desc2=A2(16,8,fmt,ch);got2=A2();call('cuArrayGetDescriptor_v2',[C.POINTER(A2),V],C.byref(got2),arr);check(desc2,got2,f'3D-as-2D/{fmt}/{ch}/{flags}')
    call('cuArrayDestroy',[V],arr)
   arr=V();desc=A2(16,8,fmt,ch);got=A2()
   call('cuArrayCreate_v2',[C.POINTER(V),C.POINTER(A2)],C.byref(arr),C.byref(desc))
   call('cuArrayGetDescriptor_v2',[C.POINTER(A2),V],C.byref(got),arr);check(desc,got,f'2D/{fmt}/{ch}')
   call('cuArrayDestroy',[V],arr)
   mip=V();desc3=A3(16,8,0,fmt,ch,0)
   call('cuMipmappedArrayCreate',[C.POINTER(V),C.POINTER(A3),U],C.byref(mip),C.byref(desc3),3)
   for level in range(3):
    arr=V();call('cuMipmappedArrayGetLevel',[C.POINTER(V),V,U],C.byref(arr),mip,level)
    got=A3();call('cuArray3DGetDescriptor_v2',[C.POINTER(A3),V],C.byref(got),arr)
    check(A3(16>>level,8>>level,0,fmt,ch,0),got,f'mip/{fmt}/{ch}/{level}')
   call('cuMipmappedArrayDestroy',[V],mip)
 if len(checks)!=80:raise RuntimeError(('wrong check count',len(checks)))
 success=True;print('PASS array descriptor: 80 complete field comparisons',flush=True)
finally:
 (out/'maps.txt').write_text(Path('/proc/self/maps').read_text())
 (out/'result.json').write_text(json.dumps({'success':success,'checks':checks,'library_sha256':hashlib.sha256(libpath.read_bytes()).hexdigest()},indent=2)+'\n')
 call('cuCtxDestroy_v2',[V],ctx)
