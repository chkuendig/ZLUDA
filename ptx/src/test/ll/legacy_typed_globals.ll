@initialized = addrspace(1) externally_initialized global float 1.500000e+00, align 16
@zero = addrspace(1) externally_initialized global float 0.000000e+00
@values = addrspace(1) externally_initialized global [2 x float] [float 2.500000e+00, float -3.500000e+00]

; Function Attrs: denormal_fpenv(preservesign)
define amdgpu_kernel void @legacy_typed_globals(ptr addrspace(4) byref(i64) %"54", ptr addrspace(4) byref(i64) %"55") #0 {
  %"56" = alloca i64, align 8, addrspace(5)
  %"57" = alloca i64, align 8, addrspace(5)
  %"58" = alloca i32, align 4, addrspace(5)
  %"59" = alloca i32, align 4, addrspace(5)
  %"60" = alloca i32, align 4, addrspace(5)
  %"61" = alloca i32, align 4, addrspace(5)
  %"62" = alloca float, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"53"

"53":                                             ; preds = %1
  store i64 ptrtoint (ptr addrspace(1) @initialized to i64), ptr addrspace(5) %"56", align 8
  %2 = load i64, ptr addrspace(5) %"56", align 8
  %"92" = inttoptr i64 %2 to ptr addrspace(1)
  %3 = load float, ptr addrspace(1) %"92", align 4
  store float %3, ptr addrspace(5) %"62", align 4
  %4 = load float, ptr addrspace(5) %"62", align 4
  %"94" = bitcast float %4 to i32
  store i32 %"94", ptr addrspace(5) %"58", align 4
  store i64 ptrtoint (ptr addrspace(1) @zero to i64), ptr addrspace(5) %"56", align 8
  %5 = load i64, ptr addrspace(5) %"56", align 8
  %"96" = inttoptr i64 %5 to ptr addrspace(1)
  %6 = load float, ptr addrspace(1) %"96", align 4
  store float %6, ptr addrspace(5) %"62", align 4
  %7 = load float, ptr addrspace(5) %"62", align 4
  %"98" = bitcast float %7 to i32
  store i32 %"98", ptr addrspace(5) %"59", align 4
  store i64 ptrtoint (ptr addrspace(1) @values to i64), ptr addrspace(5) %"56", align 8
  %8 = load i64, ptr addrspace(5) %"56", align 8
  %"100" = inttoptr i64 %8 to ptr addrspace(1)
  %9 = load float, ptr addrspace(1) %"100", align 4
  store float %9, ptr addrspace(5) %"62", align 4
  %10 = load float, ptr addrspace(5) %"62", align 4
  %"102" = bitcast float %10 to i32
  store i32 %"102", ptr addrspace(5) %"60", align 4
  %11 = load i64, ptr addrspace(5) %"56", align 8
  %"103" = inttoptr i64 %11 to ptr addrspace(1)
  %"46" = getelementptr inbounds i8, ptr addrspace(1) %"103", i64 4
  %12 = load float, ptr addrspace(1) %"46", align 4
  store float %12, ptr addrspace(5) %"62", align 4
  %13 = load float, ptr addrspace(5) %"62", align 4
  %"105" = bitcast float %13 to i32
  store i32 %"105", ptr addrspace(5) %"61", align 4
  %14 = load i64, ptr addrspace(4) %"55", align 8
  store i64 %14, ptr addrspace(5) %"57", align 8
  %15 = load i64, ptr addrspace(5) %"57", align 8
  %16 = load i32, ptr addrspace(5) %"58", align 4
  %"106" = inttoptr i64 %15 to ptr addrspace(1)
  store i32 %16, ptr addrspace(1) %"106", align 4
  %17 = load i64, ptr addrspace(5) %"57", align 8
  %"107" = inttoptr i64 %17 to ptr addrspace(1)
  %"48" = getelementptr inbounds i8, ptr addrspace(1) %"107", i64 4
  %18 = load i32, ptr addrspace(5) %"59", align 4
  store i32 %18, ptr addrspace(1) %"48", align 4
  %19 = load i64, ptr addrspace(5) %"57", align 8
  %"108" = inttoptr i64 %19 to ptr addrspace(1)
  %"50" = getelementptr inbounds i8, ptr addrspace(1) %"108", i64 8
  %20 = load i32, ptr addrspace(5) %"60", align 4
  store i32 %20, ptr addrspace(1) %"50", align 4
  %21 = load i64, ptr addrspace(5) %"57", align 8
  %"109" = inttoptr i64 %21 to ptr addrspace(1)
  %"52" = getelementptr inbounds i8, ptr addrspace(1) %"109", i64 12
  %22 = load i32, ptr addrspace(5) %"61", align 4
  store i32 %22, ptr addrspace(1) %"52", align 4
  ret void
}

attributes #0 = { denormal_fpenv(preservesign) "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
