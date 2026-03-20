declare hidden void @__zluda_ptx_impl_sust_b_3d_b32_trap(i64, <4 x i32>, i32) #0

define amdgpu_kernel void @sust_stub(ptr addrspace(4) byref(i64) %"51") #1 {
  %"52" = alloca i64, align 8, addrspace(5)
  %"53" = alloca i64, align 8, addrspace(5)
  %"54" = alloca i32, align 4, addrspace(5)
  %"55" = alloca i32, align 4, addrspace(5)
  %"56" = alloca i32, align 4, addrspace(5)
  %"57" = alloca i32, align 4, addrspace(5)
  %"58" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"50"

"50":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"51", align 8
  store i64 %2, ptr addrspace(5) %"52", align 8
  store i64 0, ptr addrspace(5) %"53", align 8
  store i32 0, ptr addrspace(5) %"54", align 4
  store i32 0, ptr addrspace(5) %"55", align 4
  store i32 0, ptr addrspace(5) %"56", align 4
  store i32 0, ptr addrspace(5) %"57", align 4
  store i32 42, ptr addrspace(5) %"58", align 4
  %3 = load i32, ptr addrspace(5) %"54", align 4
  %4 = load i32, ptr addrspace(5) %"55", align 4
  %5 = load i32, ptr addrspace(5) %"56", align 4
  %6 = load i32, ptr addrspace(5) %"57", align 4
  %7 = insertelement <4 x i32> undef, i32 %3, i8 0
  %8 = insertelement <4 x i32> %7, i32 %4, i8 1
  %9 = insertelement <4 x i32> %8, i32 %5, i8 2
  %"47" = insertelement <4 x i32> %9, i32 %6, i8 3
  %10 = load i32, ptr addrspace(5) %"58", align 4
  %"48" = insertelement <1 x i32> undef, i32 %10, i8 0
  %11 = load i64, ptr addrspace(5) %"53", align 8
  %"75" = bitcast <1 x i32> %"48" to i32
  call void @__zluda_ptx_impl_sust_b_3d_b32_trap(i64 %11, <4 x i32> %"47", i32 %"75")
  store i32 1, ptr addrspace(5) %"58", align 4
  %12 = load i64, ptr addrspace(5) %"52", align 8
  %13 = load i32, ptr addrspace(5) %"58", align 4
  %"76" = inttoptr i64 %12 to ptr addrspace(1)
  store i32 %13, ptr addrspace(1) %"76", align 4
  ret void
}

attributes #0 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="dynamic" "denormal-fp-math-f32"="dynamic" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
