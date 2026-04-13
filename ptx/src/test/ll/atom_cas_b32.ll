define amdgpu_kernel void @atom_cas_b32(ptr addrspace(4) byref(i64) %"55", ptr addrspace(4) byref(i64) %"56") #0 {
  %"57" = alloca i64, align 8, addrspace(5)
  %"58" = alloca i64, align 8, addrspace(5)
  %"59" = alloca i32, align 4, addrspace(5)
  %"60" = alloca i32, align 4, addrspace(5)
  %"61" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"54"

"54":                                             ; preds = %1
  fence seq_cst
  %2 = load i64, ptr addrspace(4) %"55", align 8
  store i64 %2, ptr addrspace(5) %"57", align 8
  %3 = load i64, ptr addrspace(4) %"56", align 8
  store i64 %3, ptr addrspace(5) %"58", align 8
  %4 = load i64, ptr addrspace(5) %"57", align 8
  %"81" = inttoptr i64 %4 to ptr addrspace(1)
  %5 = cmpxchg ptr addrspace(1) %"81", i32 -2, i32 0 syncscope("agent-one-as") acq_rel acquire, align 4
  %"80" = extractvalue { i32, i1 } %5, 0
  store i32 %"80", ptr addrspace(5) %"59", align 4
  %6 = load i64, ptr addrspace(5) %"58", align 8
  %7 = load i32, ptr addrspace(5) %"59", align 4
  %"82" = inttoptr i64 %6 to ptr
  store i32 %7, ptr %"82", align 4
  %8 = load i64, ptr addrspace(5) %"57", align 8
  %"83" = inttoptr i64 %8 to ptr addrspace(1)
  %9 = load i32, ptr addrspace(1) %"83", align 4
  store i32 %9, ptr addrspace(5) %"61", align 4
  %10 = load i64, ptr addrspace(5) %"58", align 8
  %"84" = inttoptr i64 %10 to ptr
  %"43" = getelementptr inbounds i8, ptr %"84", i64 4
  %11 = load i32, ptr addrspace(5) %"61", align 4
  store i32 %11, ptr %"43", align 4
  %12 = load i64, ptr addrspace(5) %"57", align 8
  %"85" = inttoptr i64 %12 to ptr addrspace(1)
  %"45" = getelementptr inbounds i8, ptr addrspace(1) %"85", i64 4
  %13 = cmpxchg ptr addrspace(1) %"45", i32 -2, i32 0 syncscope("agent-one-as") acq_rel acquire, align 4
  %"86" = extractvalue { i32, i1 } %13, 0
  store i32 %"86", ptr addrspace(5) %"60", align 4
  %14 = load i64, ptr addrspace(5) %"58", align 8
  %"87" = inttoptr i64 %14 to ptr
  %"49" = getelementptr inbounds i8, ptr %"87", i64 8
  %15 = load i32, ptr addrspace(5) %"60", align 4
  store i32 %15, ptr %"49", align 4
  %16 = load i64, ptr addrspace(5) %"57", align 8
  %"88" = inttoptr i64 %16 to ptr addrspace(1)
  %"51" = getelementptr inbounds i8, ptr addrspace(1) %"88", i64 4
  %17 = load i32, ptr addrspace(1) %"51", align 4
  store i32 %17, ptr addrspace(5) %"61", align 4
  %18 = load i64, ptr addrspace(5) %"58", align 8
  %"89" = inttoptr i64 %18 to ptr
  %"53" = getelementptr inbounds i8, ptr %"89", i64 12
  %19 = load i32, ptr addrspace(5) %"61", align 4
  store i32 %19, ptr %"53", align 4
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
