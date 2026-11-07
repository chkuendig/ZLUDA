import ctypes
import json
import pathlib

out = pathlib.Path('/out/diagnostics')
lib = ctypes.CDLL(str(out/'libzluda_trace.so'), mode=ctypes.RTLD_LOCAL)
for name in ('cuInit','cuDriverGetVersion','cuMemcpy2DAsync_v2','cuArray3DGetDescriptor_v2','cuGetProcAddress','cuGetProcAddress_v2','cuGetExportTable'):
    getattr(lib, name)
hip = {}
for name in ('hipHostRegister','hipHostUnregister'):
    try:
        getattr(lib, name)
        hip[name] = True
    except AttributeError:
        hip[name] = False
lib.cuInit.argtypes = [ctypes.c_uint]
lib.cuInit.restype = ctypes.c_int
lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
lib.cuDriverGetVersion.restype = ctypes.c_int
assert lib.cuInit(0) == 801
version = ctypes.c_int()
assert lib.cuDriverGetVersion(ctypes.byref(version)) == 0 and version.value == 12345
logs = list((out/'smoke-log').rglob('log.txt'))
assert len(logs) == 1, logs
text = logs[0].read_bytes().replace(b'\0',b'').decode()
assert any('cuInit(' in line and ' -> CUDA_ERROR_NOT_SUPPORTED' in line for line in text.splitlines())
assert any('cuDriverGetVersion(' in line and ' -> CUDA_SUCCESS' in line for line in text.splitlines())
(out/'SMOKE.json').write_text(json.dumps({'status':'PASS_SYNTHETIC_COMPLETION_LOGGING_ONLY','cuInit_return':801,'cuDriverGetVersion_return':0,'version':version.value,'hip_symbols_in_tracer_handle_closure':hip,'PE32_admission':'NOT ADMITTED; no direct HIP forwarding and no actual shipped HIP7 provenance proof','scope':'fake backend only; no GPU, product backend, game or Deck executed'},indent=2)+'\n')
