@params = addrspace(1) externally_initialized global [16 x i8] zeroinitializer, align 4
@unused_by_kernels = addrspace(1) externally_initialized global i32 0, align 4
@counter = addrspace(1) externally_initialized global i32 0, align 4
@llvm.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @params to ptr), ptr addrspacecast (ptr addrspace(1) @unused_by_kernels to ptr)], section "llvm.metadata"

define amdgpu_kernel void @global_const_uninit(ptr addrspace(4) byref(i64) %"43") #0 {
  %"44" = alloca i64, align 8, addrspace(5)
  %"45" = alloca i32, align 4, addrspace(5)
  %"46" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"42"

"42":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"43", align 8
  store i64 %2, ptr addrspace(5) %"44", align 8
  %3 = load i32, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @params, i64 4), align 4
  store i32 %3, ptr addrspace(5) %"45", align 4
  %4 = load i32, ptr addrspace(1) @counter, align 4
  store i32 %4, ptr addrspace(5) %"46", align 4
  %5 = load i32, ptr addrspace(5) %"45", align 4
  %6 = load i32, ptr addrspace(5) %"46", align 4
  %"50" = add i32 %5, %6
  store i32 %"50", ptr addrspace(5) %"45", align 4
  %7 = load i64, ptr addrspace(5) %"44", align 8
  %8 = load i32, ptr addrspace(5) %"45", align 4
  %"56" = inttoptr i64 %7 to ptr addrspace(1)
  store i32 %8, ptr addrspace(1) %"56", align 4
  ret void
}

attributes #0 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode" "uniform-work-group-size"="true" }
