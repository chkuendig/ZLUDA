@__zluda_alloc_min = external addrspace(1) externally_initialized global i64
@__zluda_alloc_max = external addrspace(1) externally_initialized global i64
@__zluda_scatter_sink = external addrspace(1) externally_initialized global [4096 x i8]

define amdgpu_kernel void @uint_to_fp_bf16(ptr addrspace(4) byref(i64) %"62", ptr addrspace(4) byref(i64) %"63") #0 {
  %"64" = alloca i64, align 8, addrspace(5)
  %"65" = alloca i64, align 8, addrspace(5)
  %"66" = alloca i16, align 2, addrspace(5)
  %"67" = alloca i1, align 1, addrspace(5)
  %"68" = alloca i16, align 2, addrspace(5)
  %"69" = alloca i32, align 4, addrspace(5)
  %"70" = alloca float, align 4, addrspace(5)
  %"71" = alloca i32, align 4, addrspace(5)
  %"72" = alloca i32, align 4, addrspace(5)
  %"73" = alloca i1, align 1, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"61"

"61":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"62", align 8
  store i64 %2, ptr addrspace(5) %"64", align 8
  %3 = load i64, ptr addrspace(4) %"63", align 8
  store i64 %3, ptr addrspace(5) %"65", align 8
  %4 = load i64, ptr addrspace(5) %"64", align 8
  %"100" = inttoptr i64 %4 to ptr
  %5 = load i32, ptr %"100", align 4
  store i32 %5, ptr addrspace(5) %"71", align 4
  %6 = load i32, ptr addrspace(5) %"71", align 4
  %7 = icmp eq i32 %6, 1
  store i1 %7, ptr addrspace(5) %"67", align 1
  %8 = load i64, ptr addrspace(5) %"64", align 8
  %"102" = inttoptr i64 %8 to ptr
  %"54" = getelementptr inbounds i8, ptr %"102", i64 4
  %9 = load i32, ptr %"54", align 4
  store i32 %9, ptr addrspace(5) %"72", align 4
  %10 = load i32, ptr addrspace(5) %"72", align 4
  %11 = icmp eq i32 %10, 2
  store i1 %11, ptr addrspace(5) %"73", align 1
  %12 = load i1, ptr addrspace(5) %"73", align 1
  br i1 %12, label %"13", label %"27"

"27":                                             ; preds = %"61"
  %13 = load i32, ptr addrspace(5) %"72", align 4
  %14 = icmp eq i32 %13, 3
  store i1 %14, ptr addrspace(5) %"73", align 1
  %15 = load i1, ptr addrspace(5) %"73", align 1
  br i1 %15, label %"14", label %"29"

"29":                                             ; preds = %"27"
  br label %"15"

"13":                                             ; preds = %"61"
  %16 = load i1, ptr addrspace(5) %"67", align 1
  %"88" = select i1 %16, i32 1, i32 0
  store i32 %"88", ptr addrspace(5) %"69", align 4
  %17 = load i64, ptr addrspace(5) %"65", align 8
  %18 = load i32, ptr addrspace(5) %"69", align 4
  %"103" = inttoptr i64 %17 to ptr addrspace(1)
  %19 = ptrtoint ptr addrspace(1) %"103" to i64
  %20 = load i64, ptr addrspace(1) @__zluda_alloc_min, align 4
  %21 = load i64, ptr addrspace(1) @__zluda_alloc_max, align 4
  %22 = icmp uge i64 %19, %20
  %23 = icmp ule i64 %19, %21
  %24 = and i1 %22, %23
  %25 = select i1 %24, ptr addrspace(1) %"103", ptr addrspace(1) @__zluda_scatter_sink
  store i32 %18, ptr addrspace(1) %25, align 4
  br label %"15"

"14":                                             ; preds = %"27"
  %26 = load i1, ptr addrspace(5) %"67", align 1
  %"92" = select i1 %26, i32 1, i32 0
  store i32 %"92", ptr addrspace(5) %"69", align 4
  %27 = load i32, ptr addrspace(5) %"69", align 4
  %"94" = uitofp i32 %27 to float
  store float %"94", ptr addrspace(5) %"70", align 4
  %28 = load float, ptr addrspace(5) %"70", align 4
  %29 = fptrunc float %28 to bfloat
  %"96" = bitcast bfloat %29 to i16
  store i16 %"96", ptr addrspace(5) %"66", align 2
  %30 = load i64, ptr addrspace(5) %"65", align 8
  %31 = load i16, ptr addrspace(5) %"66", align 2
  %"105" = inttoptr i64 %30 to ptr addrspace(1)
  %32 = ptrtoint ptr addrspace(1) %"105" to i64
  %33 = load i64, ptr addrspace(1) @__zluda_alloc_min, align 4
  %34 = load i64, ptr addrspace(1) @__zluda_alloc_max, align 4
  %35 = icmp uge i64 %32, %33
  %36 = icmp ule i64 %32, %34
  %37 = and i1 %35, %36
  %38 = select i1 %37, ptr addrspace(1) %"105", ptr addrspace(1) @__zluda_scatter_sink
  store i16 %31, ptr addrspace(1) %38, align 2
  br label %"15"

"15":                                             ; preds = %"14", %"13", %"29"
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="ieee" "denormal-fp-math-f32"="ieee" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
