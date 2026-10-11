declare hidden i32 @__zluda_ptx_impl_activemask() #0

declare hidden i32 @__zluda_ptx_impl_shfl_sync_idx_b32(i32, i32, i32, i32) #0

declare hidden <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32, i32, i32, i32) #0

declare hidden i32 @__zluda_ptx_impl_sreg_tid(i8) #0

declare hidden i32 @__zluda_ptx_impl_sreg_laneid() #0

define amdgpu_kernel void @shfl_scan_idx_lower_half_active_w16(ptr addrspace(4) byref(i64) %"76") #1 {
  %"77" = alloca i64, align 8, addrspace(5)
  %"78" = alloca i64, align 8, addrspace(5)
  %"79" = alloca i32, align 4, addrspace(5)
  %"80" = alloca i32, align 4, addrspace(5)
  %"81" = alloca i32, align 4, addrspace(5)
  %"82" = alloca i32, align 4, addrspace(5)
  %"83" = alloca i32, align 4, addrspace(5)
  %"84" = alloca i1, align 1, addrspace(5)
  %"85" = alloca i1, align 1, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"73"

"73":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"76", align 8
  store i64 %2, ptr addrspace(5) %"77", align 8
  %"57" = call i32 @__zluda_ptx_impl_sreg_tid(i8 0)
  store i32 %"57", ptr addrspace(5) %"79", align 4
  %"58" = call i32 @__zluda_ptx_impl_sreg_laneid()
  store i32 %"58", ptr addrspace(5) %"80", align 4
  %3 = load i32, ptr addrspace(5) %"79", align 4
  %"89" = add i32 %3, 1
  store i32 %"89", ptr addrspace(5) %"81", align 4
  %4 = load i32, ptr addrspace(5) %"79", align 4
  store i32 %4, ptr addrspace(5) %"83", align 4
  %5 = load i32, ptr addrspace(5) %"80", align 4
  %6 = icmp ult i32 %5, 16
  store i1 %6, ptr addrspace(5) %"84", align 1
  %7 = load i1, ptr addrspace(5) %"84", align 1
  br i1 %7, label %"23", label %"22"

"22":                                             ; preds = %"73"
  %8 = load i32, ptr addrspace(5) %"83", align 4
  %"96" = add i32 %8, 2000
  store i32 %"96", ptr addrspace(5) %"83", align 4
  br label %"23"

"23":                                             ; preds = %"22", %"73"
  %9 = load i1, ptr addrspace(5) %"84", align 1
  br i1 %9, label %"25", label %"12"

"25":                                             ; preds = %"23"
  %10 = load i32, ptr addrspace(5) %"81", align 4
  %"152" = call i32 @__zluda_ptx_impl_activemask()
  %"155" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %10, i32 1, i32 4096, i32 %"152")
  %"141" = extractelement <2 x i32> %"155", i8 0
  %"156" = extractelement <2 x i32> %"155", i8 1
  %"100" = trunc i32 %"156" to i1
  store i32 %"141", ptr addrspace(5) %"82", align 4
  store i1 %"100", ptr addrspace(5) %"85", align 1
  %11 = load i1, ptr addrspace(5) %"85", align 1
  br i1 %11, label %"26", label %"27"

"26":                                             ; preds = %"25"
  %12 = load i32, ptr addrspace(5) %"82", align 4
  %13 = load i32, ptr addrspace(5) %"81", align 4
  %"103" = add i32 %12, %13
  store i32 %"103", ptr addrspace(5) %"81", align 4
  br label %"27"

"27":                                             ; preds = %"26", %"25"
  %14 = load i32, ptr addrspace(5) %"81", align 4
  %"163" = call i32 @__zluda_ptx_impl_activemask()
  %"164" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %14, i32 2, i32 4096, i32 %"163")
  %"143" = extractelement <2 x i32> %"164", i8 0
  %"165" = extractelement <2 x i32> %"164", i8 1
  %"107" = trunc i32 %"165" to i1
  store i32 %"143", ptr addrspace(5) %"82", align 4
  store i1 %"107", ptr addrspace(5) %"85", align 1
  %15 = load i1, ptr addrspace(5) %"85", align 1
  br i1 %15, label %"28", label %"29"

"28":                                             ; preds = %"27"
  %16 = load i32, ptr addrspace(5) %"82", align 4
  %17 = load i32, ptr addrspace(5) %"81", align 4
  %"110" = add i32 %16, %17
  store i32 %"110", ptr addrspace(5) %"81", align 4
  br label %"29"

"29":                                             ; preds = %"28", %"27"
  %18 = load i32, ptr addrspace(5) %"81", align 4
  %"166" = call i32 @__zluda_ptx_impl_activemask()
  %"167" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %18, i32 4, i32 4096, i32 %"166")
  %"145" = extractelement <2 x i32> %"167", i8 0
  %"168" = extractelement <2 x i32> %"167", i8 1
  %"114" = trunc i32 %"168" to i1
  store i32 %"145", ptr addrspace(5) %"82", align 4
  store i1 %"114", ptr addrspace(5) %"85", align 1
  %19 = load i1, ptr addrspace(5) %"85", align 1
  br i1 %19, label %"30", label %"31"

"30":                                             ; preds = %"29"
  %20 = load i32, ptr addrspace(5) %"82", align 4
  %21 = load i32, ptr addrspace(5) %"81", align 4
  %"117" = add i32 %20, %21
  store i32 %"117", ptr addrspace(5) %"81", align 4
  br label %"31"

"31":                                             ; preds = %"30", %"29"
  %22 = load i32, ptr addrspace(5) %"81", align 4
  %"169" = call i32 @__zluda_ptx_impl_activemask()
  %"170" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %22, i32 8, i32 4096, i32 %"169")
  %"147" = extractelement <2 x i32> %"170", i8 0
  %"171" = extractelement <2 x i32> %"170", i8 1
  %"121" = trunc i32 %"171" to i1
  store i32 %"147", ptr addrspace(5) %"82", align 4
  store i1 %"121", ptr addrspace(5) %"85", align 1
  %23 = load i1, ptr addrspace(5) %"85", align 1
  br i1 %23, label %"32", label %"33"

"32":                                             ; preds = %"31"
  %24 = load i32, ptr addrspace(5) %"82", align 4
  %25 = load i32, ptr addrspace(5) %"81", align 4
  %"124" = add i32 %24, %25
  store i32 %"124", ptr addrspace(5) %"81", align 4
  br label %"33"

"33":                                             ; preds = %"32", %"31"
  %26 = load i32, ptr addrspace(5) %"81", align 4
  %"172" = call i32 @__zluda_ptx_impl_activemask()
  %"149" = call i32 @__zluda_ptx_impl_shfl_sync_idx_b32(i32 %26, i32 15, i32 4127, i32 %"172")
  store i32 %"149", ptr addrspace(5) %"82", align 4
  %27 = load i32, ptr addrspace(5) %"81", align 4
  %28 = load i32, ptr addrspace(5) %"82", align 4
  %"129" = add i32 %27, %28
  store i32 %"129", ptr addrspace(5) %"83", align 4
  br label %"12"

"12":                                             ; preds = %"33", %"23"
  %29 = load i32, ptr addrspace(5) %"79", align 4
  %"132" = zext i32 %29 to i64
  store i64 %"132", ptr addrspace(5) %"78", align 8
  %30 = load i64, ptr addrspace(5) %"78", align 8
  %"134" = mul i64 %30, 4
  store i64 %"134", ptr addrspace(5) %"78", align 8
  %31 = load i64, ptr addrspace(5) %"77", align 8
  %32 = load i64, ptr addrspace(5) %"78", align 8
  %"136" = add i64 %31, %32
  store i64 %"136", ptr addrspace(5) %"77", align 8
  %33 = load i64, ptr addrspace(5) %"77", align 8
  %34 = load i32, ptr addrspace(5) %"83", align 4
  %"151" = inttoptr i64 %33 to ptr
  store i32 %34, ptr %"151", align 4
  ret void
}

attributes #0 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="dynamic" "denormal-fp-math-f32"="dynamic" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
