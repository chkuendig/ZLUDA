declare hidden void @__zluda_ptx_impl_sust_b_2d_b32_trap(i64, <2 x i32>, i32) #0

define amdgpu_kernel void @sust_2d_stub(ptr addrspace(4) byref(i64) %"47") #1 {
  %"48" = alloca i64, align 8, addrspace(5)
  %"49" = alloca i64, align 8, addrspace(5)
  %"50" = alloca i32, align 4, addrspace(5)
  %"51" = alloca i32, align 4, addrspace(5)
  %"52" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"46"

"46":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"47", align 8
  store i64 %2, ptr addrspace(5) %"48", align 8
  store i64 0, ptr addrspace(5) %"49", align 8
  store i32 0, ptr addrspace(5) %"50", align 4
  store i32 0, ptr addrspace(5) %"51", align 4
  store i32 42, ptr addrspace(5) %"52", align 4
  %3 = load i32, ptr addrspace(5) %"50", align 4
  %4 = load i32, ptr addrspace(5) %"51", align 4
  %5 = insertelement <2 x i32> undef, i32 %3, i8 0
  %"43" = insertelement <2 x i32> %5, i32 %4, i8 1
  %6 = load i32, ptr addrspace(5) %"52", align 4
  %"44" = insertelement <1 x i32> undef, i32 %6, i8 0
  %7 = load i64, ptr addrspace(5) %"49", align 8
  %"65" = bitcast <1 x i32> %"44" to i32
  call void @__zluda_ptx_impl_sust_b_2d_b32_trap(i64 %7, <2 x i32> %"43", i32 %"65")
  store i32 1, ptr addrspace(5) %"52", align 4
  %8 = load i64, ptr addrspace(5) %"48", align 8
  %9 = load i32, ptr addrspace(5) %"52", align 4
  %"66" = inttoptr i64 %8 to ptr addrspace(1)
  store i32 %9, ptr addrspace(1) %"66", align 4
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="dynamic" "denormal-fp-math-f32"="dynamic" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
