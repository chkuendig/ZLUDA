@__zluda_alloc_min = external addrspace(1) externally_initialized global i64
@__zluda_alloc_max = external addrspace(1) externally_initialized global i64
@__zluda_scatter_sink = external addrspace(1) externally_initialized global [4096 x i8]

define amdgpu_kernel void @cvt_relu_f16x2_f32(ptr addrspace(4) byref(i64) %"52", ptr addrspace(4) byref(i64) %"53") #0 {
  %"54" = alloca i64, align 8, addrspace(5)
  %"55" = alloca i64, align 8, addrspace(5)
  %"56" = alloca i32, align 4, addrspace(5)
  %"57" = alloca i32, align 4, addrspace(5)
  %"58" = alloca i32, align 4, addrspace(5)
  %"59" = alloca i32, align 4, addrspace(5)
  %"60" = alloca <2 x half>, align 4, addrspace(5)
  %"61" = alloca <2 x half>, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"51"

"51":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"52", align 8
  store i64 %2, ptr addrspace(5) %"54", align 8
  %3 = load i64, ptr addrspace(4) %"53", align 8
  store i64 %3, ptr addrspace(5) %"55", align 8
  %4 = load i64, ptr addrspace(5) %"54", align 8
  %"83" = inttoptr i64 %4 to ptr
  %5 = load float, ptr %"83", align 4
  %"64" = bitcast float %5 to i32
  store i32 %"64", ptr addrspace(5) %"56", align 4
  %6 = load i64, ptr addrspace(5) %"54", align 8
  %"84" = inttoptr i64 %6 to ptr
  %"44" = getelementptr inbounds i8, ptr %"84", i64 4
  %7 = load float, ptr %"44", align 4
  %"67" = bitcast float %7 to i32
  store i32 %"67", ptr addrspace(5) %"57", align 4
  %8 = load i64, ptr addrspace(5) %"54", align 8
  %"86" = inttoptr i64 %8 to ptr
  %"46" = getelementptr inbounds i8, ptr %"86", i64 8
  %9 = load float, ptr %"46", align 4
  %"69" = bitcast float %9 to i32
  store i32 %"69", ptr addrspace(5) %"58", align 4
  %10 = load i64, ptr addrspace(5) %"54", align 8
  %"88" = inttoptr i64 %10 to ptr
  %"48" = getelementptr inbounds i8, ptr %"88", i64 12
  %11 = load float, ptr %"48", align 4
  %"71" = bitcast float %11 to i32
  store i32 %"71", ptr addrspace(5) %"59", align 4
  %12 = load i32, ptr addrspace(5) %"56", align 4
  %13 = load i32, ptr addrspace(5) %"57", align 4
  %"90" = bitcast i32 %12 to float
  %"91" = bitcast i32 %13 to float
  %14 = fptrunc float %"90" to half
  %15 = fptrunc float %"91" to half
  %16 = call half @llvm.maximum.f16(half %14, half 0xH0000)
  %17 = call half @llvm.maximum.f16(half %15, half 0xH0000)
  %18 = insertelement <2 x half> poison, half %16, i32 1
  %"72" = insertelement <2 x half> %18, half %17, i32 0
  store <2 x half> %"72", ptr addrspace(5) %"60", align 4
  %19 = load i32, ptr addrspace(5) %"58", align 4
  %20 = load i32, ptr addrspace(5) %"59", align 4
  %"92" = bitcast i32 %19 to float
  %"93" = bitcast i32 %20 to float
  %21 = fptrunc float %"92" to half
  %22 = fptrunc float %"93" to half
  %23 = call half @llvm.maximum.f16(half %21, half 0xH0000)
  %24 = call half @llvm.maximum.f16(half %22, half 0xH0000)
  %25 = insertelement <2 x half> poison, half %23, i32 1
  %"75" = insertelement <2 x half> %25, half %24, i32 0
  store <2 x half> %"75", ptr addrspace(5) %"61", align 4
  %26 = load i64, ptr addrspace(5) %"55", align 8
  %27 = load <2 x half>, ptr addrspace(5) %"60", align 4
  %"94" = inttoptr i64 %26 to ptr addrspace(1)
  %"95" = bitcast <2 x half> %27 to i32
  %28 = ptrtoint ptr addrspace(1) %"94" to i64
  %29 = load i64, ptr addrspace(1) @__zluda_alloc_min, align 4
  %30 = load i64, ptr addrspace(1) @__zluda_alloc_max, align 4
  %31 = icmp uge i64 %28, %29
  %32 = icmp ule i64 %28, %30
  %33 = and i1 %31, %32
  %34 = select i1 %33, ptr addrspace(1) %"94", ptr addrspace(1) @__zluda_scatter_sink
  store i32 %"95", ptr addrspace(1) %34, align 4
  %35 = load i64, ptr addrspace(5) %"55", align 8
  %"96" = inttoptr i64 %35 to ptr addrspace(1)
  %"50" = getelementptr inbounds i8, ptr addrspace(1) %"96", i64 4
  %36 = load <2 x half>, ptr addrspace(5) %"61", align 4
  %"97" = bitcast <2 x half> %36 to i32
  %37 = ptrtoint ptr addrspace(1) %"50" to i64
  %38 = load i64, ptr addrspace(1) @__zluda_alloc_min, align 4
  %39 = load i64, ptr addrspace(1) @__zluda_alloc_max, align 4
  %40 = icmp uge i64 %37, %38
  %41 = icmp ule i64 %37, %39
  %42 = and i1 %40, %41
  %43 = select i1 %42, ptr addrspace(1) %"50", ptr addrspace(1) @__zluda_scatter_sink
  store i32 %"97", ptr addrspace(1) %43, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare half @llvm.maximum.f16(half, half) #1

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="ieee" "denormal-fp-math-f32"="ieee" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
