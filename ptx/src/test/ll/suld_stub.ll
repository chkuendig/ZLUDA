declare hidden i32 @__zluda_ptx_impl_suld_b_3d_b32_zero(i64, <4 x i32>) #0

define amdgpu_kernel void @suld_stub(ptr addrspace(4) byref(i64) %"49") #1 {
  %"50" = alloca i64, align 8, addrspace(5)
  %"51" = alloca i64, align 8, addrspace(5)
  %"52" = alloca i32, align 4, addrspace(5)
  %"53" = alloca i32, align 4, addrspace(5)
  %"54" = alloca i32, align 4, addrspace(5)
  %"55" = alloca i32, align 4, addrspace(5)
  %"56" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"48"

"48":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"49", align 8
  store i64 %2, ptr addrspace(5) %"50", align 8
  store i64 0, ptr addrspace(5) %"51", align 8
  store i32 0, ptr addrspace(5) %"52", align 4
  store i32 0, ptr addrspace(5) %"53", align 4
  store i32 0, ptr addrspace(5) %"54", align 4
  store i32 0, ptr addrspace(5) %"55", align 4
  %3 = load i32, ptr addrspace(5) %"52", align 4
  %4 = load i32, ptr addrspace(5) %"53", align 4
  %5 = load i32, ptr addrspace(5) %"54", align 4
  %6 = load i32, ptr addrspace(5) %"55", align 4
  %7 = insertelement <4 x i32> zeroinitializer, i32 %3, i8 0
  %8 = insertelement <4 x i32> %7, i32 %4, i8 1
  %9 = insertelement <4 x i32> %8, i32 %5, i8 2
  %"47" = insertelement <4 x i32> %9, i32 %6, i8 3
  %10 = load i64, ptr addrspace(5) %"51", align 8
  %"71" = call i32 @__zluda_ptx_impl_suld_b_3d_b32_zero(i64 %10, <4 x i32> %"47")
  %"46" = bitcast i32 %"71" to <1 x i32>
  %"68" = extractelement <1 x i32> %"46", i8 0
  store i32 %"68", ptr addrspace(5) %"56", align 4
  %11 = load i64, ptr addrspace(5) %"50", align 8
  %12 = load i32, ptr addrspace(5) %"56", align 4
  %"72" = inttoptr i64 %11 to ptr addrspace(1)
  store i32 %12, ptr addrspace(1) %"72", align 4
  ret void
}

attributes #0 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="dynamic" "denormal-fp-math-f32"="dynamic" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
