declare hidden i32 @__zluda_ptx_impl_suldobj_b_3d_v1_b32(i64, <4 x i32>) #0

declare hidden void @__zluda_ptx_impl_sustobj_b_2d_v1_b32(i64, <2 x i32>, i32) #0

declare hidden void @__zluda_ptx_impl_sustobj_b_3d_v1_b32(i64, <4 x i32>, i32) #0

declare hidden void @__zluda_ptx_impl_sustobj_b_3d_v1_b64(i64, <4 x i32>, i64) #0

declare hidden void @__zluda_ptx_impl_sustobj_b_3d_v4_b32(i64, <4 x i32>, <4 x i32>) #0

define amdgpu_kernel void @surface_3d(ptr addrspace(4) byref(i64) %"57", ptr addrspace(4) byref(i64) %"58") #1 {
  %"59" = alloca i64, align 8, addrspace(5)
  %"60" = alloca i64, align 8, addrspace(5)
  %"61" = alloca i32, align 4, addrspace(5)
  %"62" = alloca i32, align 4, addrspace(5)
  %"63" = alloca i32, align 4, addrspace(5)
  %"64" = alloca i32, align 4, addrspace(5)
  %"65" = alloca i64, align 8, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"56"

"56":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"57", align 8
  store i64 %2, ptr addrspace(5) %"59", align 8
  %3 = load i64, ptr addrspace(4) %"58", align 8
  store i64 %3, ptr addrspace(5) %"60", align 8
  store i32 4, ptr addrspace(5) %"61", align 4
  store i32 1, ptr addrspace(5) %"62", align 4
  store i32 2, ptr addrspace(5) %"63", align 4
  %4 = load i32, ptr addrspace(5) %"61", align 4
  %5 = load i32, ptr addrspace(5) %"62", align 4
  %6 = load i32, ptr addrspace(5) %"63", align 4
  %7 = load i32, ptr addrspace(5) %"63", align 4
  %8 = insertelement <4 x i32> undef, i32 %4, i8 0
  %9 = insertelement <4 x i32> %8, i32 %5, i8 1
  %10 = insertelement <4 x i32> %9, i32 %6, i8 2
  %"46" = insertelement <4 x i32> %10, i32 %7, i8 3
  %11 = load i64, ptr addrspace(5) %"59", align 8
  %"111" = call i32 @__zluda_ptx_impl_suldobj_b_3d_v1_b32(i64 %11, <4 x i32> %"46")
  %"45" = bitcast i32 %"111" to <1 x i32>
  %"76" = extractelement <1 x i32> %"45", i8 0
  store i32 %"76", ptr addrspace(5) %"64", align 4
  %12 = load i32, ptr addrspace(5) %"61", align 4
  %13 = load i32, ptr addrspace(5) %"62", align 4
  %14 = load i32, ptr addrspace(5) %"63", align 4
  %15 = load i32, ptr addrspace(5) %"63", align 4
  %16 = insertelement <4 x i32> undef, i32 %12, i8 0
  %17 = insertelement <4 x i32> %16, i32 %13, i8 1
  %18 = insertelement <4 x i32> %17, i32 %14, i8 2
  %"47" = insertelement <4 x i32> %18, i32 %15, i8 3
  %19 = load i32, ptr addrspace(5) %"64", align 4
  %"48" = insertelement <1 x i32> undef, i32 %19, i8 0
  %20 = load i64, ptr addrspace(5) %"59", align 8
  %"116" = bitcast <1 x i32> %"48" to i32
  call void @__zluda_ptx_impl_sustobj_b_3d_v1_b32(i64 %20, <4 x i32> %"47", i32 %"116")
  %21 = load i32, ptr addrspace(5) %"61", align 4
  %22 = load i32, ptr addrspace(5) %"62", align 4
  %23 = load i32, ptr addrspace(5) %"63", align 4
  %24 = load i32, ptr addrspace(5) %"63", align 4
  %25 = insertelement <4 x i32> undef, i32 %21, i8 0
  %26 = insertelement <4 x i32> %25, i32 %22, i8 1
  %27 = insertelement <4 x i32> %26, i32 %23, i8 2
  %"49" = insertelement <4 x i32> %27, i32 %24, i8 3
  %28 = load i32, ptr addrspace(5) %"64", align 4
  %29 = load i32, ptr addrspace(5) %"64", align 4
  %30 = load i32, ptr addrspace(5) %"64", align 4
  %31 = load i32, ptr addrspace(5) %"64", align 4
  %32 = insertelement <4 x i32> undef, i32 %28, i8 0
  %33 = insertelement <4 x i32> %32, i32 %29, i8 1
  %34 = insertelement <4 x i32> %33, i32 %30, i8 2
  %"50" = insertelement <4 x i32> %34, i32 %31, i8 3
  %35 = load i64, ptr addrspace(5) %"59", align 8
  call void @__zluda_ptx_impl_sustobj_b_3d_v4_b32(i64 %35, <4 x i32> %"49", <4 x i32> %"50")
  %36 = load i32, ptr addrspace(5) %"64", align 4
  %37 = load i32, ptr addrspace(5) %"64", align 4
  %38 = insertelement <2 x i32> undef, i32 %36, i8 0
  %"51" = insertelement <2 x i32> %38, i32 %37, i8 1
  %"121" = bitcast <2 x i32> %"51" to i64
  store i64 %"121", ptr addrspace(5) %"65", align 8
  %39 = load i32, ptr addrspace(5) %"61", align 4
  %40 = load i32, ptr addrspace(5) %"62", align 4
  %41 = load i32, ptr addrspace(5) %"63", align 4
  %42 = load i32, ptr addrspace(5) %"63", align 4
  %43 = insertelement <4 x i32> undef, i32 %39, i8 0
  %44 = insertelement <4 x i32> %43, i32 %40, i8 1
  %45 = insertelement <4 x i32> %44, i32 %41, i8 2
  %"52" = insertelement <4 x i32> %45, i32 %42, i8 3
  %46 = load i64, ptr addrspace(5) %"65", align 8
  %"53" = insertelement <1 x i64> undef, i64 %46, i8 0
  %47 = load i64, ptr addrspace(5) %"59", align 8
  %"126" = bitcast <1 x i64> %"53" to i64
  call void @__zluda_ptx_impl_sustobj_b_3d_v1_b64(i64 %47, <4 x i32> %"52", i64 %"126")
  %48 = load i32, ptr addrspace(5) %"61", align 4
  %49 = load i32, ptr addrspace(5) %"62", align 4
  %50 = insertelement <2 x i32> undef, i32 %48, i8 0
  %"54" = insertelement <2 x i32> %50, i32 %49, i8 1
  %51 = load i32, ptr addrspace(5) %"64", align 4
  %"55" = insertelement <1 x i32> undef, i32 %51, i8 0
  %52 = load i64, ptr addrspace(5) %"59", align 8
  %"129" = bitcast <1 x i32> %"55" to i32
  call void @__zluda_ptx_impl_sustobj_b_2d_v1_b32(i64 %52, <2 x i32> %"54", i32 %"129")
  %53 = load i64, ptr addrspace(5) %"60", align 8
  %54 = load i32, ptr addrspace(5) %"64", align 4
  %"130" = inttoptr i64 %53 to ptr addrspace(1)
  store i32 %54, ptr addrspace(1) %"130", align 4
  ret void
}

attributes #0 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="dynamic" "denormal-fp-math-f32"="dynamic" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode" "uniform-work-group-size"="true" }
attributes #1 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode" "uniform-work-group-size"="true" }
