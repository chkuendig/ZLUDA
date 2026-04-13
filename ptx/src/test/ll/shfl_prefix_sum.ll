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
  fence seq_cst
  %2 = load i64, ptr addrspace(4) %"66", align 8
  store i64 %2, ptr addrspace(5) %"67", align 8
  %"51" = call i32 @__zluda_ptx_impl_sreg_tid(i8 0)
  store i32 %"51", ptr addrspace(5) %"69", align 4
  store i32 1, ptr addrspace(5) %"70", align 4
  %3 = load i32, ptr addrspace(5) %"70", align 4
  %"132" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %3, i32 1, i32 0, i32 -1)
  %"120" = extractelement <2 x i32> %"132", i8 0
  %"133" = extractelement <2 x i32> %"132", i8 1
  %4 = trunc i32 %"133" to i1
  store i32 %"120", ptr addrspace(5) %"71", align 4
  store i1 %4, ptr addrspace(5) %"72", align 1
  %5 = load i1, ptr addrspace(5) %"72", align 1
  br i1 %5, label %"18", label %"19"

"18":                                             ; preds = %"64"
  %6 = load i32, ptr addrspace(5) %"71", align 4
  %7 = load i32, ptr addrspace(5) %"70", align 4
  %"80" = add i32 %6, %7
  store i32 %"80", ptr addrspace(5) %"70", align 4
  br label %"19"

"19":                                             ; preds = %"18", %"64"
  %8 = load i32, ptr addrspace(5) %"70", align 4
  %"141" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %8, i32 2, i32 0, i32 -1)
  %"122" = extractelement <2 x i32> %"141", i8 0
  %"142" = extractelement <2 x i32> %"141", i8 1
  %9 = trunc i32 %"142" to i1
  store i32 %"122", ptr addrspace(5) %"71", align 4
  store i1 %9, ptr addrspace(5) %"72", align 1
  %10 = load i1, ptr addrspace(5) %"72", align 1
  br i1 %10, label %"20", label %"21"

"20":                                             ; preds = %"19"
  %11 = load i32, ptr addrspace(5) %"71", align 4
  %12 = load i32, ptr addrspace(5) %"70", align 4
  %"87" = add i32 %11, %12
  store i32 %"87", ptr addrspace(5) %"70", align 4
  br label %"21"

"21":                                             ; preds = %"20", %"19"
  %13 = load i32, ptr addrspace(5) %"70", align 4
  %"144" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %13, i32 4, i32 0, i32 -1)
  %"124" = extractelement <2 x i32> %"144", i8 0
  %"145" = extractelement <2 x i32> %"144", i8 1
  %14 = trunc i32 %"145" to i1
  store i32 %"124", ptr addrspace(5) %"71", align 4
  store i1 %14, ptr addrspace(5) %"72", align 1
  %15 = load i1, ptr addrspace(5) %"72", align 1
  br i1 %15, label %"22", label %"23"

"22":                                             ; preds = %"21"
  %16 = load i32, ptr addrspace(5) %"71", align 4
  %17 = load i32, ptr addrspace(5) %"70", align 4
  %"94" = add i32 %16, %17
  store i32 %"94", ptr addrspace(5) %"70", align 4
  br label %"23"

"23":                                             ; preds = %"22", %"21"
  %18 = load i32, ptr addrspace(5) %"70", align 4
  %"147" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %18, i32 8, i32 0, i32 -1)
  %"126" = extractelement <2 x i32> %"147", i8 0
  %"148" = extractelement <2 x i32> %"147", i8 1
  %19 = trunc i32 %"148" to i1
  store i32 %"126", ptr addrspace(5) %"71", align 4
  store i1 %19, ptr addrspace(5) %"72", align 1
  %20 = load i1, ptr addrspace(5) %"72", align 1
  br i1 %20, label %"24", label %"25"

"24":                                             ; preds = %"23"
  %21 = load i32, ptr addrspace(5) %"71", align 4
  %22 = load i32, ptr addrspace(5) %"70", align 4
  %"101" = add i32 %21, %22
  store i32 %"101", ptr addrspace(5) %"70", align 4
  br label %"25"

"25":                                             ; preds = %"24", %"23"
  %23 = load i32, ptr addrspace(5) %"70", align 4
  %"150" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_up_b32_pred(i32 %23, i32 16, i32 0, i32 -1)
  %"128" = extractelement <2 x i32> %"150", i8 0
  %"151" = extractelement <2 x i32> %"150", i8 1
  %24 = trunc i32 %"151" to i1
  store i32 %"128", ptr addrspace(5) %"71", align 4
  store i1 %24, ptr addrspace(5) %"72", align 1
  %25 = load i1, ptr addrspace(5) %"72", align 1
  br i1 %25, label %"26", label %"27"

"26":                                             ; preds = %"25"
  %26 = load i32, ptr addrspace(5) %"71", align 4
  %27 = load i32, ptr addrspace(5) %"70", align 4
  %"108" = add i32 %26, %27
  store i32 %"108", ptr addrspace(5) %"70", align 4
  br label %"27"

"27":                                             ; preds = %"26", %"25"
  %28 = load i32, ptr addrspace(5) %"69", align 4
  %29 = zext i32 %28 to i64
  store i64 %29, ptr addrspace(5) %"68", align 8
  %30 = load i64, ptr addrspace(5) %"68", align 8
  %"113" = mul i64 %30, 4
  store i64 %"113", ptr addrspace(5) %"68", align 8
  %31 = load i64, ptr addrspace(5) %"67", align 8
  %32 = load i64, ptr addrspace(5) %"68", align 8
  %"115" = add i64 %31, %32
  store i64 %"115", ptr addrspace(5) %"67", align 8
  %33 = load i64, ptr addrspace(5) %"67", align 8
  %34 = load i32, ptr addrspace(5) %"70", align 4
  %"130" = inttoptr i64 %33 to ptr
  store i32 %34, ptr %"130", align 4
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="dynamic" "denormal-fp-math-f32"="dynamic" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
