@params = addrspace(1) externally_initialized global [16 x i8] zeroinitializer, align 8
@llvm.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @params to ptr)], section "llvm.metadata"

define amdgpu_kernel void @const_zero_init(ptr addrspace(4) byref(i64) %"45", ptr addrspace(4) byref(i64) %"46") #0 {
  %"47" = alloca i64, align 8, addrspace(5)
  %"48" = alloca i64, align 8, addrspace(5)
  %"49" = alloca i64, align 8, addrspace(5)
  %"50" = alloca i64, align 8, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"44"

"44":                                             ; preds = %1
  fence seq_cst
  %2 = load i64, ptr addrspace(4) %"45", align 8
  store i64 %2, ptr addrspace(5) %"47", align 8
  %3 = load i64, ptr addrspace(4) %"46", align 8
  store i64 %3, ptr addrspace(5) %"48", align 8
  %4 = load i64, ptr addrspace(1) @params, align 8
  store i64 %4, ptr addrspace(5) %"49", align 8
  %5 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @params, i64 8), align 8
  store i64 %5, ptr addrspace(5) %"50", align 8
  %6 = load i64, ptr addrspace(5) %"48", align 8
  %7 = load i64, ptr addrspace(5) %"49", align 8
  %"61" = inttoptr i64 %6 to ptr
  store i64 %7, ptr %"61", align 8
  %8 = load i64, ptr addrspace(5) %"48", align 8
  %"62" = inttoptr i64 %8 to ptr
  %"43" = getelementptr inbounds i8, ptr %"62", i64 8
  %9 = load i64, ptr addrspace(5) %"50", align 8
  store i64 %9, ptr %"43", align 8
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
