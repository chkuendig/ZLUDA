@shared_buf = external addrspace(3) global [64 x i8], align 4

define amdgpu_kernel void @shared_ptr_offset(ptr addrspace(4) byref(i64) %"43", ptr addrspace(4) byref(i64) %"44") #0 {
  %"45" = alloca i64, align 8, addrspace(5)
  %"46" = alloca i64, align 8, addrspace(5)
  %"47" = alloca i64, align 8, addrspace(5)
  %"48" = alloca i64, align 8, addrspace(5)
  %"49" = alloca i64, align 8, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"42"

"42":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"43", align 8
  store i64 %2, ptr addrspace(5) %"45", align 8
  %3 = load i64, ptr addrspace(4) %"44", align 8
  store i64 %3, ptr addrspace(5) %"46", align 8
  %"63" = zext i32 ptrtoint (ptr addrspace(3) @shared_buf to i32) to i64
  store i64 %"63", ptr addrspace(5) %"47", align 8
  %4 = load i64, ptr addrspace(5) %"47", align 8
  %"64" = add i64 %4, 8
  store i64 %"64", ptr addrspace(5) %"48", align 8
  %5 = load i64, ptr addrspace(5) %"45", align 8
  %"66" = inttoptr i64 %5 to ptr addrspace(1)
  %6 = load i64, ptr addrspace(1) %"66", align 8
  store i64 %6, ptr addrspace(5) %"49", align 8
  %7 = load i64, ptr addrspace(5) %"48", align 8
  %8 = load i64, ptr addrspace(5) %"49", align 8
  %9 = trunc i64 %7 to i32
  %"67" = inttoptr i32 %9 to ptr addrspace(3)
  store i64 %8, ptr addrspace(3) %"67", align 8
  %10 = load i64, ptr addrspace(5) %"48", align 8
  %11 = trunc i64 %10 to i32
  %"68" = inttoptr i32 %11 to ptr addrspace(3)
  %12 = load i64, ptr addrspace(3) %"68", align 8
  store i64 %12, ptr addrspace(5) %"49", align 8
  %13 = load i64, ptr addrspace(5) %"46", align 8
  %14 = load i64, ptr addrspace(5) %"49", align 8
  %"69" = inttoptr i64 %13 to ptr addrspace(1)
  store i64 %14, ptr addrspace(1) %"69", align 8
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
