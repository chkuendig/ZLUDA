declare hidden i32 @__zluda_ptx_impl_activemask() #0

declare hidden <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32, i32, i32, i32) #0

declare hidden i32 @__zluda_ptx_impl_sreg_tid(i8) #0

define amdgpu_kernel void @shfl_prefix_sum(ptr addrspace(4) byref(i64) %"66") #1 {
  %"67" = alloca i64, align 8, addrspace(5)
  %"68" = alloca i64, align 8, addrspace(5)
  %"69" = alloca i32, align 4, addrspace(5)
  %"70" = alloca i32, align 4, addrspace(5)
  %"71" = alloca i32, align 4, addrspace(5)
  %"72" = alloca i1, align 1, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"64"

"64":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"66", align 8
  store i64 %2, ptr addrspace(5) %"67", align 8
  %"51" = call i32 @__zluda_ptx_impl_sreg_tid(i8 0)
  store i32 %"51", ptr addrspace(5) %"69", align 4
  store i32 1, ptr addrspace(5) %"70", align 4
  %3 = load i32, ptr addrspace(5) %"70", align 4
  %"131" = call i32 @__zluda_ptx_impl_activemask()
  %"134" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %3, i32 1, i32 0, i32 %"131")
  %"120" = extractelement <2 x i32> %"134", i8 0
  %"135" = extractelement <2 x i32> %"134", i8 1
  %"77" = trunc i32 %"135" to i1
  store i32 %"120", ptr addrspace(5) %"71", align 4
  store i1 %"77", ptr addrspace(5) %"72", align 1
  %4 = load i1, ptr addrspace(5) %"72", align 1
  br i1 %4, label %"18", label %"19"

"18":                                             ; preds = %"64"
  %5 = load i32, ptr addrspace(5) %"71", align 4
  %6 = load i32, ptr addrspace(5) %"70", align 4
  %"80" = add i32 %5, %6
  store i32 %"80", ptr addrspace(5) %"70", align 4
  br label %"19"

"19":                                             ; preds = %"18", %"64"
  %7 = load i32, ptr addrspace(5) %"70", align 4
  %"142" = call i32 @__zluda_ptx_impl_activemask()
  %"143" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %7, i32 2, i32 0, i32 %"142")
  %"122" = extractelement <2 x i32> %"143", i8 0
  %"144" = extractelement <2 x i32> %"143", i8 1
  %"84" = trunc i32 %"144" to i1
  store i32 %"122", ptr addrspace(5) %"71", align 4
  store i1 %"84", ptr addrspace(5) %"72", align 1
  %8 = load i1, ptr addrspace(5) %"72", align 1
  br i1 %8, label %"20", label %"21"

"20":                                             ; preds = %"19"
  %9 = load i32, ptr addrspace(5) %"71", align 4
  %10 = load i32, ptr addrspace(5) %"70", align 4
  %"87" = add i32 %9, %10
  store i32 %"87", ptr addrspace(5) %"70", align 4
  br label %"21"

"21":                                             ; preds = %"20", %"19"
  %11 = load i32, ptr addrspace(5) %"70", align 4
  %"145" = call i32 @__zluda_ptx_impl_activemask()
  %"146" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %11, i32 4, i32 0, i32 %"145")
  %"124" = extractelement <2 x i32> %"146", i8 0
  %"147" = extractelement <2 x i32> %"146", i8 1
  %"91" = trunc i32 %"147" to i1
  store i32 %"124", ptr addrspace(5) %"71", align 4
  store i1 %"91", ptr addrspace(5) %"72", align 1
  %12 = load i1, ptr addrspace(5) %"72", align 1
  br i1 %12, label %"22", label %"23"

"22":                                             ; preds = %"21"
  %13 = load i32, ptr addrspace(5) %"71", align 4
  %14 = load i32, ptr addrspace(5) %"70", align 4
  %"94" = add i32 %13, %14
  store i32 %"94", ptr addrspace(5) %"70", align 4
  br label %"23"

"23":                                             ; preds = %"22", %"21"
  %15 = load i32, ptr addrspace(5) %"70", align 4
  %"148" = call i32 @__zluda_ptx_impl_activemask()
  %"149" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %15, i32 8, i32 0, i32 %"148")
  %"126" = extractelement <2 x i32> %"149", i8 0
  %"150" = extractelement <2 x i32> %"149", i8 1
  %"98" = trunc i32 %"150" to i1
  store i32 %"126", ptr addrspace(5) %"71", align 4
  store i1 %"98", ptr addrspace(5) %"72", align 1
  %16 = load i1, ptr addrspace(5) %"72", align 1
  br i1 %16, label %"24", label %"25"

"24":                                             ; preds = %"23"
  %17 = load i32, ptr addrspace(5) %"71", align 4
  %18 = load i32, ptr addrspace(5) %"70", align 4
  %"101" = add i32 %17, %18
  store i32 %"101", ptr addrspace(5) %"70", align 4
  br label %"25"

"25":                                             ; preds = %"24", %"23"
  %19 = load i32, ptr addrspace(5) %"70", align 4
  %"151" = call i32 @__zluda_ptx_impl_activemask()
  %"152" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %19, i32 16, i32 0, i32 %"151")
  %"128" = extractelement <2 x i32> %"152", i8 0
  %"153" = extractelement <2 x i32> %"152", i8 1
  %"105" = trunc i32 %"153" to i1
  store i32 %"128", ptr addrspace(5) %"71", align 4
  store i1 %"105", ptr addrspace(5) %"72", align 1
  %20 = load i1, ptr addrspace(5) %"72", align 1
  br i1 %20, label %"26", label %"27"

"26":                                             ; preds = %"25"
  %21 = load i32, ptr addrspace(5) %"71", align 4
  %22 = load i32, ptr addrspace(5) %"70", align 4
  %"108" = add i32 %21, %22
  store i32 %"108", ptr addrspace(5) %"70", align 4
  br label %"27"

"27":                                             ; preds = %"26", %"25"
  %23 = load i32, ptr addrspace(5) %"69", align 4
  %"111" = zext i32 %23 to i64
  store i64 %"111", ptr addrspace(5) %"68", align 8
  %24 = load i64, ptr addrspace(5) %"68", align 8
  %"113" = mul i64 %24, 4
  store i64 %"113", ptr addrspace(5) %"68", align 8
  %25 = load i64, ptr addrspace(5) %"67", align 8
  %26 = load i64, ptr addrspace(5) %"68", align 8
  %"115" = add i64 %25, %26
  store i64 %"115", ptr addrspace(5) %"67", align 8
  %27 = load i64, ptr addrspace(5) %"67", align 8
  %28 = load i32, ptr addrspace(5) %"70", align 4
  %"130" = inttoptr i64 %27 to ptr
  store i32 %28, ptr %"130", align 4
  ret void
}

attributes #0 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="dynamic" "denormal-fp-math-f32"="dynamic" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
