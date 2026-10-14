define amdgpu_kernel void @atom_add_zero(ptr addrspace(4) byref(i64) %"40", ptr addrspace(4) byref(i64) %"41") #0 {
  %"42" = alloca i64, align 8, addrspace(5)
  %"43" = alloca i64, align 8, addrspace(5)
  %"44" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"39"

"39":                                             ; preds = %1
  fence seq_cst
  %2 = load i64, ptr addrspace(4) %"40", align 8
  store i64 %2, ptr addrspace(5) %"42", align 8
  %3 = load i64, ptr addrspace(4) %"41", align 8
  store i64 %3, ptr addrspace(5) %"43", align 8
  %4 = load i64, ptr addrspace(5) %"42", align 8
  %"51" = inttoptr i64 %4 to ptr addrspace(1)
  %5 = atomicrmw add ptr addrspace(1) %"51", i32 0 syncscope("agent-one-as") acq_rel, align 4
  store i32 %5, ptr addrspace(5) %"44", align 4
  %6 = load i64, ptr addrspace(5) %"43", align 8
  %7 = load i32, ptr addrspace(5) %"44", align 4
  %"52" = inttoptr i64 %6 to ptr
  store i32 %7, ptr %"52", align 4
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
