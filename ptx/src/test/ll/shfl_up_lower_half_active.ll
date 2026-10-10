declare hidden i32 @__zluda_ptx_impl_activemask() #0

declare hidden <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32, i32, i32, i32) #0

declare hidden i32 @__zluda_ptx_impl_sreg_tid(i8) #0

declare hidden i32 @__zluda_ptx_impl_sreg_laneid() #0

define amdgpu_kernel void @shfl_up_lower_half_active(ptr addrspace(4) byref(i64) %"60") #1 {
  %"61" = alloca i64, align 8, addrspace(5)
  %"62" = alloca i64, align 8, addrspace(5)
  %"63" = alloca i32, align 4, addrspace(5)
  %"64" = alloca i32, align 4, addrspace(5)
  %"65" = alloca i32, align 4, addrspace(5)
  %"66" = alloca i1, align 1, addrspace(5)
  %"67" = alloca i1, align 1, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"57"

"57":                                             ; preds = %1
  fence seq_cst
  %2 = load i64, ptr addrspace(4) %"60", align 8
  store i64 %2, ptr addrspace(5) %"61", align 8
  %"49" = call i32 @__zluda_ptx_impl_sreg_tid(i8 0)
  store i32 %"49", ptr addrspace(5) %"63", align 4
  %"50" = call i32 @__zluda_ptx_impl_sreg_laneid()
  store i32 %"50", ptr addrspace(5) %"64", align 4
  %3 = load i32, ptr addrspace(5) %"63", align 4
  store i32 %3, ptr addrspace(5) %"65", align 4
  %4 = load i32, ptr addrspace(5) %"64", align 4
  %5 = icmp ult i32 %4, 16
  store i1 %5, ptr addrspace(5) %"66", align 1
  %6 = load i1, ptr addrspace(5) %"66", align 1
  br i1 %6, label %"21", label %"20"

"20":                                             ; preds = %"57"
  %7 = load i32, ptr addrspace(5) %"65", align 4
  %"76" = add i32 %7, 2000
  store i32 %"76", ptr addrspace(5) %"65", align 4
  br label %"21"

"21":                                             ; preds = %"20", %"57"
  %8 = load i1, ptr addrspace(5) %"66", align 1
  br i1 %8, label %"23", label %"12"

"23":                                             ; preds = %"21"
  %9 = load i32, ptr addrspace(5) %"63", align 4
  %"97" = call i32 @__zluda_ptx_impl_activemask()
  %"100" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %9, i32 1, i32 0, i32 %"97")
  %"94" = extractelement <2 x i32> %"100", i8 0
  %"101" = extractelement <2 x i32> %"100", i8 1
  %10 = trunc i32 %"101" to i1
  store i32 %"94", ptr addrspace(5) %"65", align 4
  store i1 %10, ptr addrspace(5) %"67", align 1
  %11 = load i1, ptr addrspace(5) %"67", align 1
  br i1 %11, label %"25", label %"24"

"24":                                             ; preds = %"23"
  %12 = load i32, ptr addrspace(5) %"65", align 4
  %"83" = add i32 %12, 1000
  store i32 %"83", ptr addrspace(5) %"65", align 4
  br label %"25"

"25":                                             ; preds = %"24", %"23"
  br label %"12"

"12":                                             ; preds = %"25", %"21"
  %13 = load i32, ptr addrspace(5) %"63", align 4
  %14 = zext i32 %13 to i64
  store i64 %14, ptr addrspace(5) %"62", align 8
  %15 = load i64, ptr addrspace(5) %"62", align 8
  %"87" = mul i64 %15, 4
  store i64 %"87", ptr addrspace(5) %"62", align 8
  %16 = load i64, ptr addrspace(5) %"61", align 8
  %17 = load i64, ptr addrspace(5) %"62", align 8
  %"89" = add i64 %16, %17
  store i64 %"89", ptr addrspace(5) %"61", align 8
  %18 = load i64, ptr addrspace(5) %"61", align 8
  %19 = load i32, ptr addrspace(5) %"65", align 4
  %"96" = inttoptr i64 %18 to ptr
  store i32 %19, ptr %"96", align 4
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="dynamic" "denormal-fp-math-f32"="dynamic" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
