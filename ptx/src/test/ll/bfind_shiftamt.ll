define amdgpu_kernel void @bfind_shiftamt(ptr addrspace(4) byref(i64) %"49", ptr addrspace(4) byref(i64) %"50") #0 {
  %"51" = alloca i64, align 8, addrspace(5)
  %"52" = alloca i64, align 8, addrspace(5)
  %"53" = alloca i32, align 4, addrspace(5)
  %"54" = alloca i32, align 4, addrspace(5)
  %"55" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"48"

"48":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"49", align 8
  store i64 %2, ptr addrspace(5) %"51", align 8
  %3 = load i64, ptr addrspace(4) %"50", align 8
  store i64 %3, ptr addrspace(5) %"52", align 8
  %4 = load i64, ptr addrspace(5) %"51", align 8
  %"76" = inttoptr i64 %4 to ptr
  %5 = load i32, ptr %"76", align 4
  store i32 %5, ptr addrspace(5) %"53", align 4
  %6 = load i64, ptr addrspace(5) %"51", align 8
  %"77" = inttoptr i64 %6 to ptr
  %"41" = getelementptr inbounds i8, ptr %"77", i64 4
  %7 = load i32, ptr %"41", align 4
  store i32 %7, ptr addrspace(5) %"54", align 4
  %8 = load i64, ptr addrspace(5) %"51", align 8
  %"78" = inttoptr i64 %8 to ptr
  %"43" = getelementptr inbounds i8, ptr %"78", i64 8
  %9 = load i32, ptr %"43", align 4
  store i32 %9, ptr addrspace(5) %"55", align 4
  %10 = load i32, ptr addrspace(5) %"53", align 4
  %11 = call i32 @llvm.ctlz.i32(i32 %10, i1 false)
  %12 = icmp eq i32 %10, 0
  %13 = select i1 %12, i32 -1, i32 %11
  store i32 %13, ptr addrspace(5) %"53", align 4
  %14 = load i32, ptr addrspace(5) %"54", align 4
  %15 = call i32 @llvm.ctlz.i32(i32 %14, i1 false)
  %16 = icmp eq i32 %14, 0
  %17 = select i1 %16, i32 -1, i32 %15
  store i32 %17, ptr addrspace(5) %"54", align 4
  %18 = load i32, ptr addrspace(5) %"55", align 4
  %19 = call i32 @llvm.ctlz.i32(i32 %18, i1 false)
  %20 = icmp eq i32 %18, 0
  %21 = select i1 %20, i32 -1, i32 %19
  store i32 %21, ptr addrspace(5) %"55", align 4
  %22 = load i64, ptr addrspace(5) %"52", align 8
  %23 = load i32, ptr addrspace(5) %"53", align 4
  %"85" = inttoptr i64 %22 to ptr
  store i32 %23, ptr %"85", align 4
  %24 = load i64, ptr addrspace(5) %"52", align 8
  %"86" = inttoptr i64 %24 to ptr
  %"45" = getelementptr inbounds i8, ptr %"86", i64 4
  %25 = load i32, ptr addrspace(5) %"54", align 4
  store i32 %25, ptr %"45", align 4
  %26 = load i64, ptr addrspace(5) %"52", align 8
  %"87" = inttoptr i64 %26 to ptr
  %"47" = getelementptr inbounds i8, ptr %"87", i64 8
  %27 = load i32, ptr addrspace(5) %"55", align 4
  store i32 %27, ptr %"47", align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.ctlz.i32(i32, i1 immarg) #1

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
