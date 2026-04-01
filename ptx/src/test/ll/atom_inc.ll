@__zluda_alloc_min = external addrspace(1) externally_initialized global i64
@__zluda_alloc_max = external addrspace(1) externally_initialized global i64
@__zluda_scatter_sink = external addrspace(1) externally_initialized global [4096 x i8]

define amdgpu_kernel void @atom_inc(ptr addrspace(4) byref(i64) %"47", ptr addrspace(4) byref(i64) %"48") #0 {
  %"49" = alloca i64, align 8, addrspace(5)
  %"50" = alloca i64, align 8, addrspace(5)
  %"51" = alloca i32, align 4, addrspace(5)
  %"52" = alloca i32, align 4, addrspace(5)
  %"53" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"46"

"46":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"47", align 8
  store i64 %2, ptr addrspace(5) %"49", align 8
  %3 = load i64, ptr addrspace(4) %"48", align 8
  store i64 %3, ptr addrspace(5) %"50", align 8
  %4 = load i64, ptr addrspace(5) %"49", align 8
  %"68" = inttoptr i64 %4 to ptr
  %5 = atomicrmw uinc_wrap ptr %"68", i32 101 syncscope("agent-one-as") monotonic, align 4
  store i32 %5, ptr addrspace(5) %"51", align 4
  %6 = load i64, ptr addrspace(5) %"49", align 8
  %"69" = inttoptr i64 %6 to ptr addrspace(1)
  %7 = ptrtoint ptr addrspace(1) %"69" to i64
  %8 = load i64, ptr addrspace(1) @__zluda_alloc_min, align 4
  %9 = load i64, ptr addrspace(1) @__zluda_alloc_max, align 4
  %10 = icmp uge i64 %7, %8
  %11 = icmp ule i64 %7, %9
  %12 = and i1 %10, %11
  %13 = select i1 %12, ptr addrspace(1) %"69", ptr addrspace(1) @__zluda_scatter_sink
  %14 = atomicrmw uinc_wrap ptr addrspace(1) %13, i32 101 syncscope("agent-one-as") monotonic, align 4
  store i32 %14, ptr addrspace(5) %"52", align 4
  %15 = load i64, ptr addrspace(5) %"49", align 8
  %"70" = inttoptr i64 %15 to ptr
  %16 = load i32, ptr %"70", align 4
  store i32 %16, ptr addrspace(5) %"53", align 4
  %17 = load i64, ptr addrspace(5) %"50", align 8
  %18 = load i32, ptr addrspace(5) %"51", align 4
  %"71" = inttoptr i64 %17 to ptr
  store i32 %18, ptr %"71", align 4
  %19 = load i64, ptr addrspace(5) %"50", align 8
  %"72" = inttoptr i64 %19 to ptr
  %"43" = getelementptr inbounds i8, ptr %"72", i64 4
  %20 = load i32, ptr addrspace(5) %"52", align 4
  store i32 %20, ptr %"43", align 4
  %21 = load i64, ptr addrspace(5) %"50", align 8
  %"73" = inttoptr i64 %21 to ptr
  %"45" = getelementptr inbounds i8, ptr %"73", i64 8
  %22 = load i32, ptr addrspace(5) %"53", align 4
  store i32 %22, ptr %"45", align 4
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
