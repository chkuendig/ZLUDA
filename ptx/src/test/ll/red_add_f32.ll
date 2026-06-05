define amdgpu_kernel void @red_add_f32(ptr addrspace(4) byref(i64) %"39", ptr addrspace(4) byref(i64) %"40") #0 {
  %"41" = alloca i64, align 8, addrspace(5)
  %"42" = alloca i64, align 8, addrspace(5)
  %"43" = alloca float, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"38"

"38":                                             ; preds = %1
  fence seq_cst
  %2 = load i64, ptr addrspace(4) %"39", align 8
  store i64 %2, ptr addrspace(5) %"41", align 8
  %3 = load i64, ptr addrspace(4) %"40", align 8
  store i64 %3, ptr addrspace(5) %"42", align 8
  %4 = load i64, ptr addrspace(5) %"41", align 8
  %"50" = inttoptr i64 %4 to ptr
  %5 = load float, ptr %"50", align 4
  store float %5, ptr addrspace(5) %"43", align 4
  %6 = load i64, ptr addrspace(5) %"42", align 8
  %7 = load float, ptr addrspace(5) %"43", align 4
  %"51" = inttoptr i64 %6 to ptr addrspace(1)
  %8 = atomicrmw fadd ptr addrspace(1) %"51", float %7 syncscope("one-as") acq_rel, align 4
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
