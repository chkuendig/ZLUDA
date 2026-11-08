; Function Attrs: denormal_fpenv(preservesign)
define amdgpu_kernel void @legacy_typed_globals(ptr addrspace(4) byref(i32) %"54", ptr addrspace(4) byref(i32) %"55", ptr addrspace(4) byref(i64) %"91", ptr addrspace(4) byref(i32) %"93", ptr addrspace(4) byref(i32) %"95", ptr addrspace(4) byref(i32) %"97") #0 {
  %"56" = alloca i32, align 4, addrspace(5)
  %"57" = alloca i32, align 4, addrspace(5)
  %"58" = alloca i32, align 4, addrspace(5)
  %"59" = alloca i32, align 4, addrspace(5)
  %"60" = alloca i32, align 4, addrspace(5)
  %"61" = alloca i32, align 4, addrspace(5)
  %"62" = alloca float, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"53"

"53":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"91", align 8
  %3 = load i32, ptr addrspace(4) %"93", align 4
  %4 = load i32, ptr addrspace(4) %"95", align 4
  %5 = load i32, ptr addrspace(4) %"97", align 4
  store i32 %3, ptr addrspace(5) %"56", align 4
  %6 = load i32, ptr addrspace(5) %"56", align 4
  %7 = zext i32 %6 to i64
  %8 = inttoptr i64 %2 to ptr addrspace(1)
  %"100" = getelementptr inbounds i8, ptr addrspace(1) %8, i64 %7
  %9 = load float, ptr addrspace(1) %"100", align 4
  store float %9, ptr addrspace(5) %"62", align 4
  %10 = load float, ptr addrspace(5) %"62", align 4
  %"102" = bitcast float %10 to i32
  store i32 %"102", ptr addrspace(5) %"58", align 4
  store i32 %4, ptr addrspace(5) %"56", align 4
  %11 = load i32, ptr addrspace(5) %"56", align 4
  %12 = zext i32 %11 to i64
  %13 = inttoptr i64 %2 to ptr addrspace(1)
  %"104" = getelementptr inbounds i8, ptr addrspace(1) %13, i64 %12
  %14 = load float, ptr addrspace(1) %"104", align 4
  store float %14, ptr addrspace(5) %"62", align 4
  %15 = load float, ptr addrspace(5) %"62", align 4
  %"106" = bitcast float %15 to i32
  store i32 %"106", ptr addrspace(5) %"59", align 4
  store i32 %5, ptr addrspace(5) %"56", align 4
  %16 = load i32, ptr addrspace(5) %"56", align 4
  %17 = zext i32 %16 to i64
  %18 = inttoptr i64 %2 to ptr addrspace(1)
  %"108" = getelementptr inbounds i8, ptr addrspace(1) %18, i64 %17
  %19 = load float, ptr addrspace(1) %"108", align 4
  store float %19, ptr addrspace(5) %"62", align 4
  %20 = load float, ptr addrspace(5) %"62", align 4
  %"110" = bitcast float %20 to i32
  store i32 %"110", ptr addrspace(5) %"60", align 4
  %21 = load i32, ptr addrspace(5) %"56", align 4
  %22 = zext i32 %21 to i64
  %23 = inttoptr i64 %2 to ptr addrspace(1)
  %"111" = getelementptr inbounds i8, ptr addrspace(1) %23, i64 %22
  %"46" = getelementptr inbounds i8, ptr addrspace(1) %"111", i64 4
  %24 = load float, ptr addrspace(1) %"46", align 4
  store float %24, ptr addrspace(5) %"62", align 4
  %25 = load float, ptr addrspace(5) %"62", align 4
  %"113" = bitcast float %25 to i32
  store i32 %"113", ptr addrspace(5) %"61", align 4
  %26 = load i32, ptr addrspace(4) %"55", align 4
  store i32 %26, ptr addrspace(5) %"57", align 4
  %27 = load i32, ptr addrspace(5) %"57", align 4
  %28 = load i32, ptr addrspace(5) %"58", align 4
  %29 = zext i32 %27 to i64
  %30 = inttoptr i64 %2 to ptr addrspace(1)
  %"114" = getelementptr inbounds i8, ptr addrspace(1) %30, i64 %29
  store i32 %28, ptr addrspace(1) %"114", align 4
  %31 = load i32, ptr addrspace(5) %"57", align 4
  %32 = zext i32 %31 to i64
  %33 = inttoptr i64 %2 to ptr addrspace(1)
  %"115" = getelementptr inbounds i8, ptr addrspace(1) %33, i64 %32
  %"48" = getelementptr inbounds i8, ptr addrspace(1) %"115", i64 4
  %34 = load i32, ptr addrspace(5) %"59", align 4
  store i32 %34, ptr addrspace(1) %"48", align 4
  %35 = load i32, ptr addrspace(5) %"57", align 4
  %36 = zext i32 %35 to i64
  %37 = inttoptr i64 %2 to ptr addrspace(1)
  %"116" = getelementptr inbounds i8, ptr addrspace(1) %37, i64 %36
  %"50" = getelementptr inbounds i8, ptr addrspace(1) %"116", i64 8
  %38 = load i32, ptr addrspace(5) %"60", align 4
  store i32 %38, ptr addrspace(1) %"50", align 4
  %39 = load i32, ptr addrspace(5) %"57", align 4
  %40 = zext i32 %39 to i64
  %41 = inttoptr i64 %2 to ptr addrspace(1)
  %"117" = getelementptr inbounds i8, ptr addrspace(1) %41, i64 %40
  %"52" = getelementptr inbounds i8, ptr addrspace(1) %"117", i64 12
  %42 = load i32, ptr addrspace(5) %"61", align 4
  store i32 %42, ptr addrspace(1) %"52", align 4
  ret void
}

attributes #0 = { denormal_fpenv(preservesign) "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
