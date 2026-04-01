@shared_mem1 = external addrspace(3) global [128 x i8], align 4
@__zluda_alloc_min = external addrspace(1) externally_initialized global i64
@__zluda_alloc_max = external addrspace(1) externally_initialized global i64
@__zluda_scatter_sink = external addrspace(1) externally_initialized global [4096 x i8]

define amdgpu_kernel void @shared_variable(ptr addrspace(4) byref(i64) %"41", ptr addrspace(4) byref(i64) %"42") #0 {
  %"43" = alloca i64, align 8, addrspace(5)
  %"44" = alloca i64, align 8, addrspace(5)
  %"45" = alloca i64, align 8, addrspace(5)
  %"46" = alloca i64, align 8, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"40"

"40":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"41", align 8
  store i64 %2, ptr addrspace(5) %"43", align 8
  %3 = load i64, ptr addrspace(4) %"42", align 8
  store i64 %3, ptr addrspace(5) %"44", align 8
  %4 = load i64, ptr addrspace(5) %"43", align 8
  %"55" = inttoptr i64 %4 to ptr addrspace(1)
  %5 = load i64, ptr addrspace(1) %"55", align 8
  store i64 %5, ptr addrspace(5) %"45", align 8
  %6 = load i64, ptr addrspace(5) %"45", align 8
  store i64 %6, ptr addrspace(3) @shared_mem1, align 8
  %7 = load i64, ptr addrspace(3) @shared_mem1, align 8
  store i64 %7, ptr addrspace(5) %"46", align 8
  %8 = load i64, ptr addrspace(5) %"44", align 8
  %9 = load i64, ptr addrspace(5) %"46", align 8
  %"58" = inttoptr i64 %8 to ptr addrspace(1)
  %10 = ptrtoint ptr addrspace(1) %"58" to i64
  %11 = load i64, ptr addrspace(1) @__zluda_alloc_min, align 4
  %12 = load i64, ptr addrspace(1) @__zluda_alloc_max, align 4
  %13 = icmp uge i64 %10, %11
  %14 = icmp ule i64 %10, %12
  %15 = and i1 %13, %14
  %16 = select i1 %15, ptr addrspace(1) %"58", ptr addrspace(1) @__zluda_scatter_sink
  store i64 %9, ptr addrspace(1) %16, align 8
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
