define amdgpu_kernel void @shr_u64_32(ptr addrspace(4) byref(i64) %"42", ptr addrspace(4) byref(i64) %"43") #0 {
  %"44" = alloca i64, align 8, addrspace(5)
  %"45" = alloca i64, align 8, addrspace(5)
  %"46" = alloca i64, align 8, addrspace(5)
  %"47" = alloca i64, align 8, addrspace(5)
  %"48" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"41"

"41":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"42", align 8
  store i64 %2, ptr addrspace(5) %"44", align 8
  %3 = load i64, ptr addrspace(4) %"43", align 8
  store i64 %3, ptr addrspace(5) %"45", align 8
  %4 = load i64, ptr addrspace(5) %"44", align 8
  %"59" = inttoptr i64 %4 to ptr
  %5 = load i64, ptr %"59", align 8
  store i64 %5, ptr addrspace(5) %"46", align 8
  %6 = load i64, ptr addrspace(5) %"46", align 8
  %7 = lshr i64 %6, 32
  %"53" = select i1 false, i64 0, i64 %7
  store i64 %"53", ptr addrspace(5) %"47", align 8
  %8 = load i64, ptr addrspace(5) %"47", align 8
  %"55" = trunc i64 %8 to i32
  store i32 %"55", ptr addrspace(5) %"48", align 4
  %9 = load i64, ptr addrspace(5) %"45", align 8
  %10 = load i32, ptr addrspace(5) %"48", align 4
  %"60" = inttoptr i64 %9 to ptr
  store i32 %10, ptr %"60", align 4
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
