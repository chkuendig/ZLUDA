define amdgpu_kernel void @ld_st_large_offset(ptr addrspace(4) byref(i64) %"43", ptr addrspace(4) byref(i64) %"44") #0 {
  %"45" = alloca i64, align 8, addrspace(5)
  %"46" = alloca i64, align 8, addrspace(5)
  %"47" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"42"

"42":                                             ; preds = %1
  fence seq_cst
  %2 = load i64, ptr addrspace(4) %"43", align 8
  store i64 %2, ptr addrspace(5) %"45", align 8
  %3 = load i64, ptr addrspace(4) %"44", align 8
  store i64 %3, ptr addrspace(5) %"46", align 8
  %4 = load i64, ptr addrspace(5) %"45", align 8
  %"54" = inttoptr i64 %4 to ptr addrspace(1)
  %"39" = getelementptr inbounds i8, ptr addrspace(1) %"54", i64 17179869180
  %5 = load i32, ptr addrspace(1) %"39", align 4
  store i32 %5, ptr addrspace(5) %"47", align 4
  %6 = load i64, ptr addrspace(5) %"46", align 8
  %"55" = inttoptr i64 %6 to ptr addrspace(1)
  %"41" = getelementptr inbounds i8, ptr addrspace(1) %"55", i64 17179869180
  %7 = load i32, ptr addrspace(5) %"47", align 4
  store i32 %7, ptr addrspace(1) %"41", align 4
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
