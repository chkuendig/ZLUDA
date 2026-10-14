define amdgpu_kernel void @atom_add_returns_old(ptr addrspace(4) byref(i64) %"46", ptr addrspace(4) byref(i64) %"47") #0 {
  %"48" = alloca i64, align 8, addrspace(5)
  %"49" = alloca i64, align 8, addrspace(5)
  %"50" = alloca i64, align 8, addrspace(5)
  %"51" = alloca i64, align 8, addrspace(5)
  %"52" = alloca i32, align 4, addrspace(5)
  %"53" = alloca i32, align 4, addrspace(5)
  %"54" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"45"

"45":                                             ; preds = %1
  fence seq_cst
  %2 = load i64, ptr addrspace(4) %"46", align 8
  store i64 %2, ptr addrspace(5) %"49", align 8
  %3 = load i64, ptr addrspace(4) %"47", align 8
  store i64 %3, ptr addrspace(5) %"50", align 8
  %4 = load i64, ptr addrspace(5) %"49", align 8
  %5 = inttoptr i64 %4 to ptr
  %"69" = addrspacecast ptr %5 to ptr addrspace(1)
  store ptr addrspace(1) %"69", ptr addrspace(5) %"49", align 8
  %6 = load i64, ptr addrspace(5) %"50", align 8
  %7 = inttoptr i64 %6 to ptr
  %"71" = addrspacecast ptr %7 to ptr addrspace(1)
  store ptr addrspace(1) %"71", ptr addrspace(5) %"50", align 8
  %8 = load i64, ptr addrspace(5) %"49", align 8
  %"73" = inttoptr i64 %8 to ptr addrspace(1)
  %9 = atomicrmw add ptr addrspace(1) %"73", i32 -1 syncscope("agent-one-as") acq_rel, align 4
  store i32 %9, ptr addrspace(5) %"53", align 4
  %10 = load i64, ptr addrspace(5) %"50", align 8
  %11 = load i32, ptr addrspace(5) %"53", align 4
  %"74" = inttoptr i64 %10 to ptr addrspace(1)
  store i32 %11, ptr addrspace(1) %"74", align 4
  %12 = load i64, ptr addrspace(5) %"49", align 8
  %"75" = inttoptr i64 %12 to ptr addrspace(1)
  %13 = load i32, ptr addrspace(1) %"75", align 4
  store i32 %13, ptr addrspace(5) %"54", align 4
  %14 = load i64, ptr addrspace(5) %"50", align 8
  %"76" = inttoptr i64 %14 to ptr addrspace(1)
  %"44" = getelementptr inbounds i8, ptr addrspace(1) %"76", i64 4
  %15 = load i32, ptr addrspace(5) %"54", align 4
  store i32 %15, ptr addrspace(1) %"44", align 4
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
