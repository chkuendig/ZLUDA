declare hidden <2 x i32> @__zluda_ptx_impl_shfl_sync_idx_b32_pred(i32, i32, i32, i32) #0

declare hidden i32 @__zluda_ptx_impl_sreg_tid(i8) #0

define amdgpu_kernel void @shfl_broadcast_31(ptr addrspace(4) byref(i64) %"49") #1 {
  %"50" = alloca i64, align 8, addrspace(5)
  %"51" = alloca i64, align 8, addrspace(5)
  %"52" = alloca i32, align 4, addrspace(5)
  %"53" = alloca i32, align 4, addrspace(5)
  %"54" = alloca i32, align 4, addrspace(5)
  %"55" = alloca i1, align 1, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"47"

"47":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"49", align 8
  store i64 %2, ptr addrspace(5) %"50", align 8
  %"41" = call i32 @__zluda_ptx_impl_sreg_tid(i8 0)
  store i32 %"41", ptr addrspace(5) %"52", align 4
  %3 = load i32, ptr addrspace(5) %"52", align 4
  %"58" = add i32 %3, 1
  store i32 %"58", ptr addrspace(5) %"53", align 4
  %4 = load i32, ptr addrspace(5) %"53", align 4
  %"75" = call <2 x i32> @__zluda_ptx_impl_shfl_sync_idx_b32_pred(i32 %4, i32 31, i32 31, i32 -1)
  %"72" = extractelement <2 x i32> %"75", i8 0
  %"76" = extractelement <2 x i32> %"75", i8 1
  %"61" = trunc i32 %"76" to i1
  store i32 %"72", ptr addrspace(5) %"54", align 4
  store i1 %"61", ptr addrspace(5) %"55", align 1
  %5 = load i32, ptr addrspace(5) %"52", align 4
  %"63" = zext i32 %5 to i64
  store i64 %"63", ptr addrspace(5) %"51", align 8
  %6 = load i64, ptr addrspace(5) %"51", align 8
  %"65" = mul i64 %6, 4
  store i64 %"65", ptr addrspace(5) %"51", align 8
  %7 = load i64, ptr addrspace(5) %"50", align 8
  %8 = load i64, ptr addrspace(5) %"51", align 8
  %"67" = add i64 %7, %8
  store i64 %"67", ptr addrspace(5) %"50", align 8
  %9 = load i64, ptr addrspace(5) %"50", align 8
  %10 = load i32, ptr addrspace(5) %"54", align 4
  %"74" = inttoptr i64 %9 to ptr
  store i32 %10, ptr %"74", align 4
  ret void
}

attributes #0 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="dynamic" "denormal-fp-math-f32"="dynamic" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
