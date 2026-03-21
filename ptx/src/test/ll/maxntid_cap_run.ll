declare hidden i32 @__zluda_ptx_impl_sreg_tid(i8) #0

define amdgpu_kernel void @maxntid_cap_run(ptr addrspace(4) byref(i64) %"50", ptr addrspace(4) byref(i64) %"51") #1 {
  %"52" = alloca i64, align 8, addrspace(5)
  %"53" = alloca i64, align 8, addrspace(5)
  %"54" = alloca i64, align 8, addrspace(5)
  %"55" = alloca i64, align 8, addrspace(5)
  %"56" = alloca i32, align 4, addrspace(5)
  %"57" = alloca i1, align 1, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"48"

"48":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"50", align 8
  store i64 %2, ptr addrspace(5) %"52", align 8
  %3 = load i64, ptr addrspace(4) %"51", align 8
  store i64 %3, ptr addrspace(5) %"53", align 8
  %"45" = call i32 @__zluda_ptx_impl_sreg_tid(i8 0)
  store i32 %"45", ptr addrspace(5) %"56", align 4
  %4 = load i32, ptr addrspace(5) %"56", align 4
  %5 = icmp eq i32 %4, 0
  store i1 %5, ptr addrspace(5) %"57", align 1
  %6 = load i1, ptr addrspace(5) %"57", align 1
  br i1 %6, label %"21", label %"13"

"21":                                             ; preds = %"48"
  %7 = load i64, ptr addrspace(5) %"52", align 8
  %"70" = inttoptr i64 %7 to ptr
  %8 = load i64, ptr %"70", align 8
  store i64 %8, ptr addrspace(5) %"54", align 8
  %9 = load i64, ptr addrspace(5) %"54", align 8
  %"66" = add i64 %9, 1
  store i64 %"66", ptr addrspace(5) %"55", align 8
  %10 = load i64, ptr addrspace(5) %"53", align 8
  %11 = load i64, ptr addrspace(5) %"55", align 8
  %"71" = inttoptr i64 %10 to ptr
  store i64 %11, ptr %"71", align 8
  br label %"13"

"13":                                             ; preds = %"21", %"48"
  ret void
}

attributes #0 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="dynamic" "denormal-fp-math-f32"="dynamic" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,256" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
