declare hidden <2 x i32> @__zluda_ptx_impl_shfl_sync_idx_b32_pred(i32, i32, i32, i32) #0

declare hidden i32 @__zluda_ptx_impl_sreg_tid(i8) #0

define amdgpu_kernel void @shfl_sync_idx_multi_reg(ptr addrspace(4) byref(i64) %"59") #1 {
  %"60" = alloca i64, align 8, addrspace(5)
  %"61" = alloca i64, align 8, addrspace(5)
  %"62" = alloca i32, align 4, addrspace(5)
  %"63" = alloca i32, align 4, addrspace(5)
  %"64" = alloca i32, align 4, addrspace(5)
  %"65" = alloca i32, align 4, addrspace(5)
  %"66" = alloca i32, align 4, addrspace(5)
  %"67" = alloca i1, align 1, addrspace(5)
  %"68" = alloca i1, align 1, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"57"

"57":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"59", align 8
  store i64 %2, ptr addrspace(5) %"60", align 8
  %"44" = call i32 @__zluda_ptx_impl_sreg_tid(i8 0)
  store i32 %"44", ptr addrspace(5) %"62", align 4
  %3 = load i32, ptr addrspace(5) %"62", align 4
  %"71" = add i32 %3, 1
  store i32 %"71", ptr addrspace(5) %"63", align 4
  %4 = load i32, ptr addrspace(5) %"62", align 4
  %"73" = add i32 %4, 1
  store i32 %"73", ptr addrspace(5) %"64", align 4
  %5 = load i32, ptr addrspace(5) %"64", align 4
  %"75" = mul i32 %5, 100
  store i32 %"75", ptr addrspace(5) %"64", align 4
  %6 = load i32, ptr addrspace(5) %"63", align 4
  %"100" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_idx_b32_pred(i32 %6, i32 0, i32 31, i32 -1)
  %"94" = extractelement <2 x i32> %"100", i8 0
  %"101" = extractelement <2 x i32> %"100", i8 1
  %7 = trunc i32 %"101" to i1
  store i32 %"94", ptr addrspace(5) %"65", align 4
  store i1 %7, ptr addrspace(5) %"67", align 1
  %8 = load i32, ptr addrspace(5) %"64", align 4
  %"108" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_idx_b32_pred(i32 %8, i32 0, i32 31, i32 -1)
  %"96" = extractelement <2 x i32> %"108", i8 0
  %"109" = extractelement <2 x i32> %"108", i8 1
  %9 = trunc i32 %"109" to i1
  store i32 %"96", ptr addrspace(5) %"66", align 4
  store i1 %9, ptr addrspace(5) %"68", align 1
  %10 = load i32, ptr addrspace(5) %"62", align 4
  %11 = zext i32 %10 to i64
  store i64 %11, ptr addrspace(5) %"61", align 8
  %12 = load i64, ptr addrspace(5) %"61", align 8
  %"85" = mul i64 %12, 8
  store i64 %"85", ptr addrspace(5) %"61", align 8
  %13 = load i64, ptr addrspace(5) %"60", align 8
  %14 = load i64, ptr addrspace(5) %"61", align 8
  %"87" = add i64 %13, %14
  store i64 %"87", ptr addrspace(5) %"60", align 8
  %15 = load i64, ptr addrspace(5) %"60", align 8
  %16 = load i32, ptr addrspace(5) %"65", align 4
  %"98" = inttoptr i64 %15 to ptr
  store i32 %16, ptr %"98", align 4
  %17 = load i64, ptr addrspace(5) %"60", align 8
  %"99" = inttoptr i64 %17 to ptr
  %"56" = getelementptr inbounds i8, ptr %"99", i64 4
  %18 = load i32, ptr addrspace(5) %"66", align 4
  store i32 %18, ptr %"56", align 4
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="dynamic" "denormal-fp-math-f32"="dynamic" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
