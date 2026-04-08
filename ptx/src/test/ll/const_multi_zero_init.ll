@gParams = addrspace(1) externally_initialized global [16 x i8] zeroinitializer, align 8
@gBounds = addrspace(1) externally_initialized global [16 x i8] zeroinitializer, align 8
@llvm.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @gParams to ptr), ptr addrspacecast (ptr addrspace(1) @gBounds to ptr)], section "llvm.metadata"

define amdgpu_kernel void @const_multi_zero_init(ptr addrspace(4) byref(i64) %"54", ptr addrspace(4) byref(i64) %"55") #0 {
  %"56" = alloca i64, align 8, addrspace(5)
  %"57" = alloca i64, align 8, addrspace(5)
  %"58" = alloca i64, align 8, addrspace(5)
  %"59" = alloca i64, align 8, addrspace(5)
  %"60" = alloca i64, align 8, addrspace(5)
  %"61" = alloca i64, align 8, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"53"

"53":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"54", align 8
  store i64 %2, ptr addrspace(5) %"56", align 8
  %3 = load i64, ptr addrspace(4) %"55", align 8
  store i64 %3, ptr addrspace(5) %"57", align 8
  %4 = load i64, ptr addrspace(1) @gParams, align 8
  store i64 %4, ptr addrspace(5) %"58", align 8
  %5 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @gParams, i64 8), align 8
  store i64 %5, ptr addrspace(5) %"59", align 8
  %6 = load i64, ptr addrspace(1) @gBounds, align 8
  store i64 %6, ptr addrspace(5) %"60", align 8
  %7 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @gBounds, i64 8), align 8
  store i64 %7, ptr addrspace(5) %"61", align 8
  %8 = load i64, ptr addrspace(5) %"57", align 8
  %9 = load i64, ptr addrspace(5) %"58", align 8
  %"80" = inttoptr i64 %8 to ptr
  store i64 %9, ptr %"80", align 8
  %10 = load i64, ptr addrspace(5) %"57", align 8
  %"81" = inttoptr i64 %10 to ptr
  %"48" = getelementptr inbounds i8, ptr %"81", i64 8
  %11 = load i64, ptr addrspace(5) %"59", align 8
  store i64 %11, ptr %"48", align 8
  %12 = load i64, ptr addrspace(5) %"57", align 8
  %"82" = inttoptr i64 %12 to ptr
  %"50" = getelementptr inbounds i8, ptr %"82", i64 16
  %13 = load i64, ptr addrspace(5) %"60", align 8
  store i64 %13, ptr %"50", align 8
  %14 = load i64, ptr addrspace(5) %"57", align 8
  %"83" = inttoptr i64 %14 to ptr
  %"52" = getelementptr inbounds i8, ptr %"83", i64 24
  %15 = load i64, ptr addrspace(5) %"61", align 8
  store i64 %15, ptr %"52", align 8
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
