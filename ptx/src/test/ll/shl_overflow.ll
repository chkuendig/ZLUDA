define amdgpu_kernel void @shl_overflow(ptr addrspace(4) byref(i64) %"60", ptr addrspace(4) byref(i64) %"61") #0 {
  %"62" = alloca i64, align 8, addrspace(5)
  %"63" = alloca i64, align 8, addrspace(5)
  %"64" = alloca i32, align 4, addrspace(5)
  %"65" = alloca i32, align 4, addrspace(5)
  %"66" = alloca i32, align 4, addrspace(5)
  %"67" = alloca i32, align 4, addrspace(5)
  %"68" = alloca i32, align 4, addrspace(5)
  %"69" = alloca i32, align 4, addrspace(5)
  %"70" = alloca i32, align 4, addrspace(5)
  %"71" = alloca i32, align 4, addrspace(5)
  %"72" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"59"

"59":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"60", align 8
  store i64 %2, ptr addrspace(5) %"62", align 8
  %3 = load i64, ptr addrspace(4) %"61", align 8
  store i64 %3, ptr addrspace(5) %"63", align 8
  %4 = load i64, ptr addrspace(5) %"62", align 8
  %"104" = inttoptr i64 %4 to ptr
  %5 = load i32, ptr %"104", align 4
  store i32 %5, ptr addrspace(5) %"64", align 4
  %6 = load i64, ptr addrspace(5) %"62", align 8
  %"105" = inttoptr i64 %6 to ptr
  %"47" = getelementptr inbounds i8, ptr %"105", i64 4
  %7 = load i32, ptr %"47", align 4
  store i32 %7, ptr addrspace(5) %"65", align 4
  %8 = load i64, ptr addrspace(5) %"62", align 8
  %"106" = inttoptr i64 %8 to ptr
  %"49" = getelementptr inbounds i8, ptr %"106", i64 8
  %9 = load i32, ptr %"49", align 4
  store i32 %9, ptr addrspace(5) %"66", align 4
  %10 = load i64, ptr addrspace(5) %"62", align 8
  %"107" = inttoptr i64 %10 to ptr
  %"51" = getelementptr inbounds i8, ptr %"107", i64 12
  %11 = load i32, ptr %"51", align 4
  store i32 %11, ptr addrspace(5) %"67", align 4
  store i32 1, ptr addrspace(5) %"72", align 4
  %12 = load i32, ptr addrspace(5) %"72", align 4
  %13 = load i32, ptr addrspace(5) %"64", align 4
  %14 = icmp uge i32 %13, 32
  %15 = shl i32 %12, %13
  %"108" = select i1 %14, i32 0, i32 %15
  store i32 %"108", ptr addrspace(5) %"68", align 4
  %16 = load i32, ptr addrspace(5) %"72", align 4
  %17 = load i32, ptr addrspace(5) %"65", align 4
  %18 = icmp uge i32 %17, 32
  %19 = shl i32 %16, %17
  %"110" = select i1 %18, i32 0, i32 %19
  store i32 %"110", ptr addrspace(5) %"69", align 4
  %20 = load i32, ptr addrspace(5) %"72", align 4
  %21 = load i32, ptr addrspace(5) %"66", align 4
  %22 = icmp uge i32 %21, 32
  %23 = shl i32 %20, %21
  %"112" = select i1 %22, i32 0, i32 %23
  store i32 %"112", ptr addrspace(5) %"70", align 4
  %24 = load i32, ptr addrspace(5) %"72", align 4
  %25 = load i32, ptr addrspace(5) %"67", align 4
  %26 = icmp uge i32 %25, 32
  %27 = shl i32 %24, %25
  %"114" = select i1 %26, i32 0, i32 %27
  store i32 %"114", ptr addrspace(5) %"71", align 4
  %28 = load i64, ptr addrspace(5) %"63", align 8
  %29 = load i32, ptr addrspace(5) %"68", align 4
  %"116" = inttoptr i64 %28 to ptr
  store i32 %29, ptr %"116", align 4
  %30 = load i64, ptr addrspace(5) %"63", align 8
  %"117" = inttoptr i64 %30 to ptr
  %"54" = getelementptr inbounds i8, ptr %"117", i64 4
  %31 = load i32, ptr addrspace(5) %"69", align 4
  store i32 %31, ptr %"54", align 4
  %32 = load i64, ptr addrspace(5) %"63", align 8
  %"118" = inttoptr i64 %32 to ptr
  %"56" = getelementptr inbounds i8, ptr %"118", i64 8
  %33 = load i32, ptr addrspace(5) %"70", align 4
  store i32 %33, ptr %"56", align 4
  %34 = load i64, ptr addrspace(5) %"63", align 8
  %"119" = inttoptr i64 %34 to ptr
  %"58" = getelementptr inbounds i8, ptr %"119", i64 12
  %35 = load i32, ptr addrspace(5) %"71", align 4
  store i32 %35, ptr %"58", align 4
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
